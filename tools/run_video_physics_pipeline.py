"""Stage 2 from video: learn contact physics on training episodes, test on unseen ones.

    python tools/run_video_physics_pipeline.py \
        --dataset_root output/cgwm_v3_dynamic \
        --stage1_ply output/cgwm_scan/stage1_can_cov/point_cloud/iteration_7000/point_cloud.ply \
        --stage1_dataset output/cgwm_scan/stage1/can \
        --output_dir output/cgwm_v3_dynamic/stage2_pose

Everything the model needs is read from the dataset (camera, frame timing,
support plane) and the Stage 1 Gaussians (geometry, centre of mass, inertia).
Training uses RGB frames and object masks only.

1. tracking (GPU, once per episode, cached): the pose of every frame from its
   silhouette and chromaticity, without any physics;
2. physics (CPU, parallel): multiple shooting on the tracked poses - one start
   state per ``--window`` frames and shared physics, residuals in pixels;
3. evaluation on unseen episodes with the learned physics frozen:
   * short-horizon forecast - fit the state to the last ``--window`` observed
     frames (from the most recent impact on, if there was one inside them),
     predict ``--horizon`` frames ahead;
   * open loop - fit the state to the first window, roll out the whole episode
     (for bouncing/tumbling scenes this is limited by the chaos of the scene
     itself: MuJoCo re-run from a 0.1 mm perturbed state diverges too);
   * rolling forecast - the forecasts stitched into one trajectory in which
     every frame was predicted 5-14 frames after the last observed one.
   The evaluation is repeated with MuJoCo's own parameters as a reference for
   what the Stage 1 geometry and this simulator reach with exact physics.

``state/trajectory.json`` and the MuJoCo properties are read for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.pose_space_fit import (  # noqa: E402
    PoseFitPool,
    contact_onsets,
    observation_start,
    fit_physics_pose_space,
    fit_states_and_rollout,
    make_pose_track,
)
from stage2.video_physics_fit import (  # noqa: E402
    ContactPhysics,
    ObjectRenderer,
    auto_supersample,
    build_object_model,
    estimate_initial_pose,
    evaluate_trajectory,
    load_episode,
    make_dynamics,
    pose_errors,
    ray_error_components,
    rollout,
    states_from_track,
    to_reference_frame,
    track_poses,
)


ROLLING_LEAD = 5  # frames between the last observed frame and the earliest frame shown from a forecast


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--object_name", default="cola_can")
    parser.add_argument("--stage1_dataset", default=None, type=Path,
                        help="Stage1 dataset of the object (default: <dataset_root>/stage1/can)")
    parser.add_argument("--train_episodes", default="0,1", help="indices in the train split used to learn physics")
    parser.add_argument("--eval_train_rest", action="store_true", help="also evaluate the unused train episodes")
    parser.add_argument("--max_eval_episodes", default=0, type=int, help="0 = all")
    parser.add_argument("--supersample", default=0, type=int,
                        help="render at N x resolution and box-filter; 0 = choose from the object's pixel size "
                             "relative to the Stage 1 images")
    parser.add_argument("--window", default=10, type=int, help="frames per multiple-shooting window")
    parser.add_argument("--horizon", default=15, type=int, help="forecast horizon in frames")
    parser.add_argument("--track_iters", default=30, type=int)
    parser.add_argument("--physics_iters", default=120, type=int)
    parser.add_argument("--physics_lr", default=0.05, type=float)
    parser.add_argument("--state_iters", default=60, type=int, help="iterations of each held-out state fit")
    parser.add_argument("--forecast_stride", default=10, type=int, help="frames between forecast start times")
    parser.add_argument("--tracks_dir", default=None, type=Path, help="pose-track cache (default: <output_dir>/tracks)")
    parser.add_argument("--track_only", action="store_true",
                        help="only track the episodes (fills the pose-track cache on the GPU) and stop")
    parser.add_argument("--skip_eval", action="store_true", help="only learn the physics (writes physics.json)")
    parser.add_argument("--skip_reference", action="store_true",
                        help="do not repeat the evaluation with MuJoCo's own parameters")
    parser.add_argument("--physics_from", default=None, type=Path,
                        help="reuse learned_physics of an earlier report and only run the evaluation")
    parser.add_argument("--workers", default=0, type=int, help="physics worker processes; 0 = CPU count - 2")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def episode_roots(dataset_root: Path, split: str, object_name: str) -> list[Path]:
    base = dataset_root / "contactwm" / "stage2" / "fall_and_rebound" / split / object_name
    return sorted(path for path in base.glob("episode_*") if path.is_dir())


def ground_truth_physics(episode_root: Path) -> dict | None:
    manifest = json.loads((episode_root / "episode_manifest.json").read_text(encoding="utf-8"))
    contact, body = manifest.get("mujoco_contact_properties"), manifest.get("mujoco_body_properties")
    if not contact or not body:
        return None
    return {
        "timeconst": float(contact["object_solref"][0]),
        "dampratio": float(contact["object_solref"][1]),
        "friction": float(contact["effective_sliding_friction"]),
        # the model's mass is 1, so generalized damping is learned per unit mass
        "damping": float(body["freejoint_damping"]) / float(body["actual_mass_kg"]),
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)), "max": float(array.max()), "count": int(array.size)}


def prepare_episode(root, obj, args, tracks_dir, log):
    """Load an episode and track the object's pose in every frame (images only, cached)."""
    episode = load_episode(root, device=args.device)
    cache = tracks_dir / f"{root.parent.parent.name}_{root.name}.npz"
    if cache.exists():
        data = np.load(cache)
        positions, quaternions = torch.tensor(data["positions"]).float(), torch.tensor(data["quaternions"]).float()
        losses, frame0 = data["losses"].tolist(), {"cached": str(cache)}
    else:
        supersample = args.supersample if args.supersample > 0 else auto_supersample(obj, episode)
        renderer = ObjectRenderer(obj.asset, episode.camera, camera_spec=episode.camera_spec, supersample=supersample)
        started = time.time()
        position, quaternion, frame0 = estimate_initial_pose(renderer, episode, log=log)
        positions, quaternions, losses = track_poses(renderer, episode, position, quaternion, iters=args.track_iters, log=log)
        frame0.update({"supersample": supersample, "seconds": time.time() - started})
        np.savez(cache, positions=positions.numpy(), quaternions=quaternions.numpy(), losses=np.asarray(losses))
    mask_area = episode.mask.sum(dim=(1, 2, 3)).cpu()
    track = make_pose_track(episode.name, positions, quaternions, mask_area,
                            camera_position=episode.camera_spec.c2w[:3, 3], fovy_deg=episode.camera_spec.fovy_deg,
                            image_height=episode.rgb.shape[-2], frame_dt=episode.frame_dt,
                            physics_dt=episode.physics_dt, substeps=episode.substeps)
    states = states_from_track(positions, quaternions, episode.frame_dt)
    tracking = {"frame0": frame0, "frame_losses": losses}
    if episode.gt_positions is not None:
        translation, rotation = pose_errors(obj, positions, quaternions, episode.gt_positions, episode.gt_quaternions)
        tracking.update({"translation_m": summarize(translation.tolist()), "rotation_deg": summarize(rotation.tolist())})
        log(f"  [{episode.name}] tracking vs GT: {translation.mean() * 100:.2f} cm (max {translation.max() * 100:.2f}) "
            f"/ {rotation.mean():.2f} deg (max {rotation.max():.2f})")
    # only the ground truth and timing are needed from here on: free the images
    episode.rgb, episode.mask = episode.rgb[:1].cpu(), episode.mask[:1].cpu()
    torch.cuda.empty_cache()
    return episode, track, states, tracking


def main():
    args = parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = args.tracks_dir or args.output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "log.txt"

    def log(message: str):
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    started = time.time()
    stage1_dataset = args.stage1_dataset or args.dataset_root / "stage1" / "can"
    obj = build_object_model(args.stage1_ply, stage1_dataset, device=args.device)
    log(f"object: {json.dumps(obj.report)}")

    train_roots = episode_roots(args.dataset_root, "train", args.object_name)
    test_roots = episode_roots(args.dataset_root, "test", args.object_name)
    chosen = [int(value) for value in str(args.train_episodes).split(",") if value.strip()]
    fit_roots = [train_roots[index] for index in chosen]
    eval_roots = list(test_roots)
    if args.eval_train_rest:
        eval_roots = [root for index, root in enumerate(train_roots) if index not in chosen] + eval_roots
    if args.max_eval_episodes > 0:
        eval_roots = eval_roots[: args.max_eval_episodes]
    if args.skip_eval:
        eval_roots = []
    if args.track_only:
        for root in fit_roots + eval_roots:
            prepare_episode(root, obj, args, tracks_dir, log)
        return
    truth = ground_truth_physics(fit_roots[0])
    previous = None if args.physics_from is None else json.loads(args.physics_from.read_text(encoding="utf-8"))
    workers = args.workers if args.workers > 0 else None

    results = {"train": [], "holdout": []}
    physics_history = []
    if previous is None:
        prepared = [prepare_episode(root, obj, args, tracks_dir, log) for root in fit_roots]
        tracks = [item[1] for item in prepared]
        physics = ContactPhysics()
        with PoseFitPool(obj, tracks, workers) as pool:
            physics, index, fitted, physics_history = fit_physics_pose_space(
                pool, tracks, [item[2] for item in prepared], physics, window=args.window,
                iters=args.physics_iters, physics_lr=args.physics_lr, obj=obj, log=log)
        for episode_index, (episode, track, _, tracking) in enumerate(prepared):
            first = next(state for (owner, t0, _), state in zip(index, fitted) if owner == episode_index and t0 == 0)
            with torch.no_grad():
                positions, quaternions = rollout(make_dynamics(obj, physics, dt=episode.physics_dt), first,
                                                 len(episode.frames), episode.substeps)
            results["train"].append({"episode": episode.name, "tracking": tracking,
                                     **evaluate_trajectory(obj, episode, positions, quaternions)})
        del prepared
    else:
        values = previous["learned_physics"]
        physics = ContactPhysics(values["timeconst"], values["dampratio"], values["friction"], values["damping"])
        physics_history = previous.get("physics_history", [])
        results["train"] = previous.get("train_results", [])
        log(f"physics loaded from {args.physics_from}")
    learned = physics.as_dict()
    log(f"learned physics: {json.dumps(learned)}")
    if truth:
        log(f"MuJoCo physics : {json.dumps(truth)}")
    (args.output_dir / "physics.json").write_text(json.dumps({
        "learned_physics": learned, "mujoco_physics_for_reference": truth, "physics_history": physics_history,
        "train_results": results["train"],
    }, indent=2), encoding="utf-8")

    def evaluate(prepared, pool, eval_physics, label):
        """Forecasts and open-loop rollouts of the held-out episodes under ``eval_physics``."""
        requests, owners = [], []
        for episode_index, (episode, track, states, _) in enumerate(prepared):
            count = len(track)
            requests.append((episode_index, 0, min(args.window, count - 1) + 1, states[0], count))  # open loop
            owners.append((episode_index, None, 0))
            onsets = contact_onsets(obj, track)
            for t in range(args.window, count - 1, args.forecast_stride):
                t0 = observation_start(onsets, t, args.window)
                ahead = min(args.horizon, count - 1 - t)  # the last forecasts are cut by the end of the video
                requests.append((episode_index, t0, t - t0 + 1, states[t0], t - t0 + ahead + 1))
                owners.append((episode_index, t, t - t0))
        fits = fit_states_and_rollout(pool, eval_physics, requests, iters=args.state_iters)
        rows, translation_means, rotation_means, finals, across_means, across_finals = [], [], [], [], [], []
        for episode_index, (episode, track, _, tracking) in enumerate(prepared):
            forecast_rows, metrics, rolling_p, rolling_q = [], {}, None, None
            for (owner, t, observed), fit in zip(owners, fits):  # open loop first, then forecasts by increasing t
                if owner != episode_index:
                    continue
                if t is None:
                    metrics = evaluate_trajectory(obj, episode, fit["positions"], fit["quaternions"])
                    rolling_p, rolling_q = fit["positions"].clone(), fit["quaternions"].clone()
                    continue
                predicted_p, predicted_q = fit["positions"][observed:], fit["quaternions"][observed:]
                frames = list(range(t, t + predicted_p.shape[0]))
                translation, rotation = pose_errors(obj, predicted_p, predicted_q,
                                                    episode.gt_positions[frames], episode.gt_quaternions[frames])
                body_p = to_reference_frame(obj, predicted_p, predicted_q)[0].numpy().astype(np.float64)
                across = ray_error_components(body_p - episode.gt_positions[frames], episode.gt_positions[frames],
                                              episode.camera_spec.c2w[:3, 3])[0]
                forecast_rows.append({"observed_until": t, "observed_frames": observed + 1, "translation_m": translation.tolist(),
                                      "translation_across_ray_m": across.tolist(), "rotation_deg": rotation.tolist()})
                # every frame shows the newest forecast that is at least ROLLING_LEAD frames old
                rolling_p[t + ROLLING_LEAD:t + len(frames)] = predicted_p[ROLLING_LEAD:]
                rolling_q[t + ROLLING_LEAD:t + len(frames)] = predicted_q[ROLLING_LEAD:]
                if len(frames) == args.horizon + 1:
                    translation_means.append(float(translation.mean()))
                    rotation_means.append(float(rotation.mean()))
                    finals.append(float(translation[-1]))
                    across_means.append(float(across.mean()))
                    across_finals.append(float(across[-1]))
            rolling = evaluate_trajectory(obj, episode, rolling_p, rolling_q)
            rolling["lead_frames"] = [ROLLING_LEAD, ROLLING_LEAD + args.forecast_stride - 1]
            rows.append({"episode": episode.name, "tracking": tracking, "forecasts": forecast_rows,
                         "rolling_forecast": rolling, **metrics})
            mean_forecast = np.mean([np.mean(row["translation_m"]) for row in forecast_rows]) if forecast_rows else float("nan")
            log(f"[holdout, {label}] {episode.name}: forecast({args.horizon}f) mean {mean_forecast * 100:.2f} cm | "
                f"open loop mean {metrics['translation_mean_m'] * 100:.2f} cm / {metrics['rotation_mean_deg']:.2f} deg")
        forecast = {
            "horizon_frames": args.horizon, "observed_frames_max": args.window + 1,
            "observation": "the last frames before the forecast, starting after the most recent impact",
            "translation_mean_over_horizon_m": summarize(translation_means),
            "translation_at_horizon_m": summarize(finals),
            "translation_across_ray_mean_over_horizon_m": summarize(across_means),
            "translation_across_ray_at_horizon_m": summarize(across_finals),
            "rotation_mean_over_horizon_deg": summarize(rotation_means),
        }
        return rows, forecast

    forecast, reference_forecast, reference_rows = {}, {}, []
    if eval_roots:
        prepared = [prepare_episode(root, obj, args, tracks_dir, log) for root in eval_roots]
        with PoseFitPool(obj, [item[1] for item in prepared], workers) as pool:
            results["holdout"], forecast = evaluate(prepared, pool, physics, "learned physics")
            if truth and not args.skip_reference:
                # The same evaluation with MuJoCo's own parameters: what the model
                # (Stage 1 geometry + this simulator) achieves when the physics is exact.
                reference = ContactPhysics(truth["timeconst"], truth["dampratio"], truth["friction"], truth["damping"])
                reference_rows, reference_forecast = evaluate(prepared, pool, reference, "MuJoCo physics")

    def open_loop_summary(rows):
        return {
            "translation_mean_m": summarize([row["translation_mean_m"] for row in rows]),
            "translation_across_ray_mean_m": summarize(
                [row["translation_across_ray_mean_m"] for row in rows if "translation_across_ray_mean_m" in row]),
            "translation_max_m": summarize([row["translation_max_m"] for row in rows]),
            "rotation_mean_deg": summarize([row["rotation_mean_deg"] for row in rows]),
        } if rows else {}

    def tracking_summary(rows, key):
        return summarize([row["tracking"][key]["mean"] for row in rows if key in row["tracking"]])

    summary = {
        "physics": {key: {"learned": learned[key], "mujoco": None if not truth else truth[key],
                          "ratio": None if not truth else learned[key] / truth[key]} for key in learned},
        "holdout_tracking_translation_mean_m": tracking_summary(results["holdout"], "translation_m"),
        "holdout_tracking_rotation_mean_deg": tracking_summary(results["holdout"], "rotation_deg"),
        "holdout_forecast": forecast,
        "holdout_open_loop": open_loop_summary(results["holdout"]),
        "holdout_rolling_forecast": open_loop_summary([row["rolling_forecast"] for row in results["holdout"]]),
        "reference_with_mujoco_physics": {
            "holdout_forecast": reference_forecast,
            "holdout_open_loop": open_loop_summary(reference_rows),
            "holdout_rolling_forecast": open_loop_summary([row["rolling_forecast"] for row in reference_rows]),
        },
        "train_open_loop": open_loop_summary(results["train"]),
    }
    report = {
        "dataset_root": str(args.dataset_root),
        "stage1_ply": str(args.stage1_ply),
        "stage1_dataset": str(stage1_dataset),
        "object": obj.report,
        "train_episodes": [str(root) for root in fit_roots],
        "eval_episodes": [str(root) for root in eval_roots],
        "settings": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "learned_physics": learned,
        "mujoco_physics_for_reference": truth,
        "physics_history": physics_history,
        "summary": summary,
        "results": results,
        "ground_truth_trajectory_used_for_training": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"summary: {json.dumps(summary, indent=1)}")


if __name__ == "__main__":
    main()
