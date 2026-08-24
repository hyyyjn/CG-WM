from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from stage2.differentiable_collision_detection import GaussianCollisionBody
from stage2.renderable_gaussian_asset import RenderableGaussianAsset
from stage2.scene_manifest import load_scene_manifest
from tools.run_native_multibody_manifest import (
    build_native_runtime, fit_native_image_only, prefit_native_initial_states,
    refined_geometry_overrides, rollout_manifest, _temporal_window,
    _contact_curriculum_selection, _load_action_wrenches,
    _step_observation_frame, loss_gradient_attribution,
)


class NativeMultiBodyManifestTests(unittest.TestCase):
    def test_loss_gradient_attribution_separates_terms_and_groups(self):
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        attribution = loss_gradient_attribution(
            {"image": x * y, "regularizer": x.square()},
            {"x": [x], "y": [y], "empty": []},
        )
        self.assertAlmostEqual(attribution["image"]["x"], 3.0)
        self.assertAlmostEqual(attribution["image"]["y"], 2.0)
        self.assertAlmostEqual(attribution["regularizer"]["x"], 4.0)
        self.assertEqual(attribution["regularizer"]["y"], 0.0)
        self.assertEqual(attribution["image"]["empty"], 0.0)

    def make_manifest(self, root: Path) -> Path:
        (root / "rgb").mkdir()
        (root / "rgb" / "000000.png").write_bytes(b"x")
        (root / "asset.ply").write_bytes(b"ply")
        for index, position in enumerate(([0, 0, 0.1], [0, 0, 0])):
            (root / f"state{index}.json").write_text(json.dumps({"position": position}), encoding="utf-8")
        payload = {
            "scene_id": "two_arbitrary_objects",
            "bodies": [{
                "id": "alpha", "role": "dynamic", "render": {"gaussian_ply": "asset.ply"},
                "collision": {"type": "gaussian_union"},
                "initialization": {"state_json": "state0.json"},
                "physics": {"mass": {"initial": 2.5},
                            "inertia": {"initial_diagonal": [2.0, 3.0, 4.0]}},
            }],
            "environment": [{
                "id": "beta", "role": "static", "render": {"gaussian_ply": "asset.ply"},
                "collision": {"type": "gaussian_union"},
                "initialization": {"state_json": "state1.json"},
            }],
            "observations": {"rgb_dir": "rgb", "fps": 30},
            "contact_pairs": [{"body_a": "alpha", "body_b": "beta"}],
        }
        path = root / "scene.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def collision_body(*_, **__) -> GaussianCollisionBody:
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        return GaussianCollisionBody(
            centers, torch.tensor([0.05]), centers, torch.tensor([0])
        )

    def test_builds_and_rolls_out_generic_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_scene_manifest(self.make_manifest(Path(temporary)))
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                bodies, _, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
                result = rollout_manifest(manifest, steps=2, device=torch.device("cpu"))
            self.assertEqual([body.id for body in bodies], ["alpha", "beta"])
            self.assertEqual(dynamics.candidate_pairs, ((0, 1),))
            self.assertAlmostEqual(float(torch.nn.functional.softplus(dynamics.mass_parameters[0])), 2.5, places=5)
            components = torch.nn.functional.softplus(dynamics.inertia_parameters[0])
            inertia = torch.stack((components[0] + components[1],
                                   components[0] + components[2],
                                   components[1] + components[2]))
            torch.testing.assert_close(inertia, torch.tensor([2.0, 3.0, 4.0]))
            self.assertEqual(len(result["frames"]), 3)
            self.assertFalse(result["ground_truth_trajectory_used"])

    def test_physics_timestep_preserves_recorded_fixed_step_cadence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["simulation"] = {"physics_timestep": 0.002}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                _, states, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
                result = rollout_manifest(manifest, steps=1, device=torch.device("cpu"))
            self.assertEqual(dynamics.frame_substeps, 17)
            self.assertAlmostEqual(dynamics.config.dt, 0.002)
            self.assertAlmostEqual(dynamics.observation_frame_dt, 0.034)
            self.assertAlmostEqual(dynamics.nominal_observation_frame_dt, 1.0 / 30.0)
            self.assertEqual(result["contact_dynamics_profile"]["substeps_per_frame"], 17)

    def test_explicit_steps_per_frame_overrides_fps_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["simulation"] = {"physics_timestep": 0.002, "steps_per_frame": 16}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                _, _, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
            self.assertEqual(dynamics.frame_substeps, 16)
            self.assertAlmostEqual(dynamics.config.dt, 0.002)
            self.assertAlmostEqual(dynamics.observation_frame_dt, 0.032)

    def test_impedance_prior_maps_time_constant_to_mass_scaled_k_and_d(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["contact_pairs"][0]["impedance_prior"] = {
                "time_constant": 0.02, "damping_ratio": 1.0,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                _, _, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
            self.assertAlmostEqual(float(torch.nn.functional.softplus(dynamics.stiffness)[0]), 6250.0, places=2)
            self.assertAlmostEqual(float(torch.nn.functional.softplus(dynamics.damping)[0]), 250.0, places=3)

    def test_frame_action_wrench_is_held_across_all_substeps(self):
        class CountingDynamics:
            frame_substeps = 4

            def __init__(self):
                self.wrenches = []

            def step(self, states, external_wrenches=None):
                self.wrenches.append(external_wrenches)
                return states, {"count": len(self.wrenches)}

        dynamics = CountingDynamics()
        wrench = torch.ones(1, 6)
        states, diagnostics = _step_observation_frame((object(),), dynamics, external_wrenches=wrench)
        self.assertEqual(len(dynamics.wrenches), 4)
        self.assertTrue(all(value is wrench for value in dynamics.wrenches))
        self.assertEqual(diagnostics["count"], 4)

    def test_missing_inertia_is_initialized_from_gaussian_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            del payload["bodies"][0]["physics"]["inertia"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                _, _, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
            components = torch.nn.functional.softplus(dynamics.inertia_parameters[0])
            inertia = torch.stack((components[0] + components[1],
                                   components[0] + components[2],
                                   components[1] + components[2]))
            # One solid r=5 cm Gaussian with mass 2.5 kg: I=2/5*m*r^2.
            torch.testing.assert_close(inertia, torch.full((3,), 0.0025), rtol=1e-5, atol=1e-7)
            self.assertTrue(bool(inertia[0] <= inertia[1] + inertia[2]))

    def test_manifest_canonical_offset_is_shared_by_collision_and_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["bodies"][0]["initialization"]["canonical_offset"] = [0.1, -0.2, 0.3]
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch(
                "tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply",
                self.collision_body,
            ):
                _, _, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
            expected = torch.tensor([[-0.1, 0.2, -0.3]])
            torch.testing.assert_close(dynamics.bodies[0].local_centers, expected)
            torch.testing.assert_close(dynamics.bodies[0].local_query_points, expected)

    def test_manifest_canonical_offset_requires_three_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["bodies"][0]["initialization"]["canonical_offset"] = [0.1, 0.2]
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch(
                "tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply",
                self.collision_body,
            ):
                with self.assertRaisesRegex(ValueError, "canonical_offset must have length 3"):
                    build_native_runtime(manifest, device=torch.device("cpu"))

    def test_initial_state_can_select_rgb_aligned_trajectory_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_manifest(root)
            trajectory = root / "trajectory.json"
            trajectory.write_text(json.dumps({"states": [
                {"frame_index": 0, "position": [0.1, 0.2, 0.3],
                 "linear_velocity": [1, 2, 3]},
                {"frame_index": 1, "position": [0.4, 0.5, 0.6]},
            ]}), encoding="utf-8")
            payload = json.loads(path.read_text())
            payload["bodies"][0]["initialization"] = {
                "state_json": "trajectory.json", "state_frame": 0,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                _, states, _ = build_native_runtime(manifest, device=torch.device("cpu"))
            torch.testing.assert_close(states[0].position, torch.tensor([0.1, 0.2, 0.3]))
            torch.testing.assert_close(states[0].linear_velocity, torch.tensor([1.0, 2.0, 3.0]))

    def test_trajectory_state_requires_explicit_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_manifest(root)
            (root / "trajectory.json").write_text(
                json.dumps({"states": [{"frame_index": 0}]}), encoding="utf-8"
            )
            payload = json.loads(path.read_text())
            payload["bodies"][0]["initialization"] = {"state_json": "trajectory.json"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                with self.assertRaisesRegex(ValueError, "state_frame is required"):
                    build_native_runtime(manifest, device=torch.device("cpu"))

    def test_static_plane_is_kept_analytic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["environment"][0]["collision"] = {"type": "plane"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                _, states, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
                _, diagnostics = dynamics.step(states)
            self.assertEqual(dynamics.candidate_pairs, ())
            self.assertEqual(len(dynamics.plane_contact_pairs), 1)
            self.assertEqual(len(diagnostics["plane_contacts"]), 1)

    def test_paper_mode_rejects_analytic_plane_contact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["environment"][0]["collision"] = {"type": "plane"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                with self.assertRaisesRegex(ValueError, "Gaussian-union geometry for both bodies"):
                    build_native_runtime(
                        manifest, device=torch.device("cpu"), pipeline_mode="paper_compatible"
                    )

    def test_paper_mode_rejects_experimental_collision_proxy_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["bodies"][0]["collision"].update({
                "primitive_selection": "geometry_feature_support",
                "geometry_feature_weight": 0.25,
            })
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with self.assertRaisesRegex(ValueError, "experimental collision proxy options"):
                build_native_runtime(
                    manifest, device=torch.device("cpu"), pipeline_mode="paper_compatible"
                )

    def test_paper_mode_rejects_collision_subsampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["bodies"][0]["collision"]["max_primitives"] = 256
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with self.assertRaisesRegex(ValueError, "max_primitives"):
                build_native_runtime(
                    manifest, device=torch.device("cpu"), pipeline_mode="paper_compatible"
                )

    def test_paper_mode_loads_all_filtered_stage1_gaussians(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_scene_manifest(self.make_manifest(Path(temporary)))
            with patch(
                "tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply",
                side_effect=self.collision_body,
            ) as loader:
                build_native_runtime(
                    manifest, device=torch.device("cpu"), pipeline_mode="paper_compatible"
                )
            self.assertIsNone(loader.call_args.kwargs["max_primitives"])
            self.assertEqual(loader.call_args.kwargs["radius_convention"], "paper_r2s")

    def test_paper_fit_enables_joint_geometry_refinement_by_default(self):
        class FakeRenderLoss:
            def __init__(self, **kwargs):
                self.base_assets = [RenderableGaussianAsset(
                    xyz=torch.zeros(1, 3), features_dc=torch.zeros(1, 1, 3),
                    features_rest=torch.zeros(1, 0, 3), opacity=torch.zeros(1, 1),
                    scaling=torch.zeros(1, 3), rotation=torch.tensor([[1.0, 0, 0, 0]]),
                    features_geo=torch.zeros(1, 0), foreground_logit=torch.zeros(1, 1),
                    object_ids=torch.zeros(1, dtype=torch.int32), sh_degree=0,
                    source_indices=torch.tensor([0]),
                ) for _ in kwargs["stage1_plys"]]
                self.targets = torch.zeros(len(kwargs["frame_indices"]), 3, 2, 2)
                self.masks = None

            def __call__(self, positions, quaternions, **kwargs):
                loss = positions.square().mean() + quaternions.square().mean() * 0.0
                for centers in kwargs.get("geometry_centers") or []:
                    if centers is not None:
                        loss = loss + centers.square().mean() * 0.0
                return loss, {"gaussian_render_loss": float(loss.detach())}

            def render_sequence(self, positions, quaternions, **kwargs):
                return torch.zeros(positions.shape[0], 3, 2, 2)

        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_scene_manifest(self.make_manifest(Path(temporary)))
            with patch(
                "tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply",
                self.collision_body,
            ):
                result = fit_native_image_only(
                    manifest, fit_iters=0, lr=0.01, stride=1, max_frames=1,
                    width=2, height=2, image_loss="l1", device=torch.device("cpu"),
                    render_loss_factory=FakeRenderLoss, pipeline_mode="paper_compatible",
                )
            self.assertTrue(result["geometry_refinement"]["enabled"])
            self.assertTrue(result["geometry_refinement"]["enabled_by_pipeline_mode"])
            self.assertEqual(
                result["geometry_refinement"]["gradient_route"], "collision_only"
            )
            self.assertTrue(result["geometry_refinement"]["renderer_geometry_detached"])
            self.assertEqual(
                result["geometry_refinement"]["refined_collision_geometry"]["alpha"]["source_indices"],
                [0],
            )

    def test_paper_fit_rejects_frozen_mass_inertia(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_scene_manifest(self.make_manifest(Path(temporary)))
            with self.assertRaisesRegex(ValueError, "mass/inertia cannot be frozen"):
                fit_native_image_only(
                    manifest, fit_iters=0, lr=0.01, stride=1, max_frames=1,
                    width=2, height=2, image_loss="l1", device=torch.device("cpu"),
                    render_loss_factory=lambda **_: None, pipeline_mode="paper_compatible",
                    learn_mass_inertia=False,
                )

    def test_experimental_mode_keeps_feature_collision_proxy_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["bodies"][0]["collision"]["primitive_selection"] = "support_surface"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch(
                "tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply",
                self.collision_body,
            ):
                build_native_runtime(
                    manifest, device=torch.device("cpu"), pipeline_mode="experimental"
                )

    def test_plane_contact_backpropagates_to_mass_and_inertia(self):
        def off_center_body(*_, **__):
            centers = torch.tensor([[0.03, 0.0, 0.0]])
            return GaussianCollisionBody(centers, torch.tensor([0.05]), centers)

        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_manifest(Path(temporary))
            payload = json.loads(path.read_text())
            payload["environment"][0]["collision"] = {"type": "plane"}
            payload["bodies"][0]["physics"]["inertia"]["initial_diagonal"] = [1.0, 1.0, 1.0]
            payload["bodies"][0]["initialization"] = {"state_json": "state0.json"}
            (Path(temporary) / "state0.json").write_text(
                json.dumps({"position": [0, 0, 0.02], "linear_velocity": [0, 0, -1]}),
                encoding="utf-8",
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", off_center_body):
                _, states, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
            dynamics.mass_parameters.requires_grad_(True)
            dynamics.inertia_parameters.requires_grad_(True)
            next_states, _ = dynamics.step(states)
            objective = next_states[0].linear_velocity.square().sum()
            objective = objective + next_states[0].angular_velocity.square().sum()
            objective.backward()
            self.assertGreater(float(dynamics.mass_parameters.grad[0].abs()), 0.0)
            self.assertGreater(float(dynamics.inertia_parameters.grad[0].abs().sum()), 0.0)
            self.assertEqual(float(dynamics.mass_parameters.grad[1]), 0.0)

    def test_image_only_fit_backpropagates_without_gt_trajectory(self):
        class FakeRenderLoss:
            def __init__(self, frame_indices, **_):
                self.targets = torch.zeros((len(frame_indices), 3, 1, 1))
                self.masks = None

            def render_sequence(self, positions, quaternions):
                values = positions.square().mean(dim=(1, 2), keepdim=True)
                return values.reshape(-1, 1, 1, 1).expand(-1, 3, 1, 1)

            def __call__(self, positions, quaternions, target_indices=None):
                rendered = self.render_sequence(positions, quaternions)
                loss = rendered.mean() + 0.01 * quaternions[..., 1:].square().mean()
                return loss, {"gaussian_render_loss": float(loss.detach())}

        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_scene_manifest(self.make_manifest(Path(temporary)))
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                result = fit_native_image_only(
                    manifest, fit_iters=2, lr=0.01, stride=1, max_frames=1,
                    width=16, height=16, image_loss="l1", device=torch.device("cpu"),
                    render_loss_factory=FakeRenderLoss,
                    physics_warmup_fraction=0.5,
                )
            self.assertEqual(result["supervision"], "image_only")
            self.assertFalse(result["ground_truth_trajectory_used_for_training"])
            self.assertEqual(len(result["loss_history"]), 2)
            self.assertEqual(result["loss_history"][0]["curriculum_phase"], "state_warmup")
            self.assertEqual(result["loss_history"][0]["frozen_parameter_groups"], ["physics"])
            self.assertEqual(result["loss_history"][1]["curriculum_phase"], "joint_physics")
            self.assertEqual(result["loss_history"][1]["frozen_parameter_groups"], ["initial_state"])
            self.assertTrue(
                result["optimization_stability"]["freeze_initial_state_after_warmup"]
            )
            self.assertEqual(
                result["optimization_stability"]["best_state_scope"],
                "joint_physics_only",
            )
            self.assertEqual(result["learned_contact_pairs"][0]["body_a"], "alpha")
            physics = result["learned_body_physics"]
            self.assertTrue(physics["alpha"]["learned"])
            self.assertFalse(physics["beta"]["learned"])
            self.assertGreater(physics["alpha"]["mass"], 0.0)
            i_x, i_y, i_z = physics["alpha"]["inertia_diagonal"]
            self.assertLessEqual(i_x, i_y + i_z)
            self.assertLessEqual(i_y, i_x + i_z)
            self.assertLessEqual(i_z, i_x + i_y)

    def test_temporal_windows_cover_full_sequence_and_keep_tail(self):
        frames = list(range(0, 30, 3))
        windows = [
            _temporal_window(frames, iteration=i, window_frames=4, window_step=3)[0]
            for i in range(3)
        ]
        self.assertEqual(windows[0], [0, 3, 6, 9])
        self.assertEqual(windows[-1], [18, 21, 24, 27])
        self.assertEqual(
            _temporal_window(frames, iteration=0, window_frames=0, window_step=1)[0],
            frames,
        )

    def test_contact_curriculum_keeps_frames_around_predicted_contact(self):
        frames = list(range(0, 75, 5))
        selected, indices = _contact_curriculum_selection(
            frames, [35], budget=5, iteration=0
        )
        self.assertEqual(len(selected), 5)
        self.assertIn(35, selected)
        self.assertEqual(selected, [frames[index] for index in indices])

    def test_paper_mode_keeps_manifest_initial_state_fixed(self):
        class FakeRenderLoss:
            received = {}

            def __init__(self, frame_indices, **kwargs):
                self.targets = torch.zeros((len(frame_indices), 3, 1, 1))
                self.masks = None
                self.base_assets = [RenderableGaussianAsset(
                    xyz=torch.zeros(1, 3), features_dc=torch.zeros(1, 1, 3),
                    features_rest=torch.zeros(1, 0, 3), opacity=torch.zeros(1, 1),
                    scaling=torch.zeros(1, 3), rotation=torch.tensor([[1.0, 0, 0, 0]]),
                    features_geo=torch.zeros(1, 0), foreground_logit=torch.zeros(1, 1),
                    object_ids=torch.zeros(1, dtype=torch.int32), sh_degree=0,
                    source_indices=torch.tensor([0]),
                ) for _ in kwargs["stage1_plys"]]
                type(self).received = kwargs

            def render_sequence(self, positions, quaternions, **kwargs):
                return positions.square().mean(dim=(1, 2), keepdim=True).reshape(
                    -1, 1, 1, 1
                ).expand(-1, 3, 1, 1)

            def __call__(self, positions, quaternions, target_indices=None, **kwargs):
                loss = self.render_sequence(positions, quaternions, **kwargs).mean()
                return loss, {"gaussian_render_loss": float(loss.detach())}

        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_scene_manifest(self.make_manifest(Path(temporary)))
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                result = fit_native_image_only(
                    manifest, fit_iters=2, lr=0.01, stride=1, max_frames=1,
                    width=16, height=16, image_loss="l1", device=torch.device("cpu"),
                    render_loss_factory=FakeRenderLoss, pipeline_mode="paper_compatible",
                )
            self.assertFalse(result["initial_state_learning"]["enabled"])
            self.assertAlmostEqual(result["initial_states"]["alpha"]["position"][2], 0.1, places=6)
            self.assertEqual(result["supervision"], "paper_full_image")
            self.assertEqual(result["image_loss_config"]["type"], "l1_loftr")
            self.assertEqual(result["image_loss_config"]["loftr_weight"], 1.0)
            self.assertFalse(result["image_loss_config"]["gt_mask_used_for_loss"])
            self.assertIsNone(FakeRenderLoss.received["gt_mask_dir"])
            self.assertEqual(FakeRenderLoss.received["config"].loss, "l1_loftr")

    def test_manifest_wrench_sequence_drives_matching_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_manifest(root)
            action_path = root / "actions.json"
            action_path.write_text(json.dumps({"frames": [{
                "frame": 0, "bodies": {"alpha": {"force": [2, 0, 0], "torque": [0, 0, 0]}}
            }]}), encoding="utf-8")
            payload = json.loads(path.read_text())
            payload["actions"] = {"type": "wrench_sequence", "path": "actions.json"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                bodies, states, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
            actions, report = _load_action_wrenches(
                manifest, bodies, max_frame=1, device=torch.device("cpu")
            )
            baseline, _ = dynamics.step(states)
            actuated, _ = dynamics.step(states, external_wrenches=actions[0])
            self.assertGreater(float(actuated[0].linear_velocity[0]), float(baseline[0].linear_velocity[0]))
            self.assertEqual(report["nonzero_frame_count"], 1)

    def test_generalized_damping_implicitly_reduces_linear_and_angular_speed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_manifest(root)
            (root / "state0.json").write_text(json.dumps({
                "position": [0, 0, 10], "linear_velocity": [1, 0, 0],
                "angular_velocity": [1, 1, 1],
            }), encoding="utf-8")
            payload = json.loads(path.read_text())
            payload["bodies"][0]["physics"]["generalized_damping"] = {"initial": 0.05}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                _, states, damped_dynamics = build_native_runtime(
                    manifest, device=torch.device("cpu")
                )
                damped, diagnostics = damped_dynamics.step(states)
            self.assertLess(float(damped[0].linear_velocity[0]), 1.0)
            self.assertLess(
                float(torch.linalg.norm(damped[0].angular_velocity)), 3.0 ** 0.5
            )
            self.assertEqual(diagnostics["generalized_damping"][0], 0.05)
            self.assertTrue(bool(torch.isfinite(damped[0].angular_velocity).all()))

    def test_contact_parameters_are_aligned_per_manifest_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_manifest(root)
            (root / "state2.json").write_text(json.dumps({"position": [0.2, 0, 0]}), encoding="utf-8")
            payload = json.loads(path.read_text())
            payload["bodies"].append({
                "id": "gamma", "role": "dynamic", "render": {"gaussian_ply": "asset.ply"},
                "collision": {"type": "gaussian_union"},
                "initialization": {"state_json": "state2.json"},
            })
            payload["contact_pairs"][0].update({"stiffness": {"initial": 100}, "friction": {"initial": 0.1}})
            payload["contact_pairs"].append({
                "body_a": "gamma", "body_b": "beta",
                "stiffness": {"initial": 900}, "friction": {"initial": 0.8},
            })
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                _, _, dynamics = build_native_runtime(manifest, device=torch.device("cpu"))
            self.assertEqual(dynamics.candidate_pairs, ((0, 2), (1, 2)))
            self.assertEqual(tuple(dynamics.stiffness.shape), (2,))
            learned = torch.nn.functional.softplus(dynamics.stiffness)
            self.assertAlmostEqual(float(learned[0]), 100.0, places=3)
            self.assertAlmostEqual(float(learned[1]), 900.0, places=3)

    def test_image_prefit_separates_pose_and_velocity(self):
        class FakePrefitLoss:
            def __call__(self, positions, quaternions):
                times = torch.arange(positions.shape[0], dtype=positions.dtype) / 30.0
                target_x = 0.2 + 0.3 * times
                loss = (positions[:, 0, 0] - target_x).square().mean()
                loss = loss + 0.01 * quaternions[:, 0, 1:].square().mean()
                return loss, {"gaussian_render_loss": float(loss.detach())}

        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_scene_manifest(self.make_manifest(Path(temporary)))
            with patch("tools.run_native_multibody_manifest.load_gaussian_collision_body_from_ply", self.collision_body):
                bodies, states, _ = build_native_runtime(manifest, device=torch.device("cpu"))
            fitted, report = prefit_native_initial_states(
                bodies, states, FakePrefitLoss(), frame_indices=[0, 1, 2], fps=30.0,
                pose_iters=80, velocity_iters=100, velocity_frames=3, lr=0.05,
                velocity_l2=0.0,
            )
            self.assertLess(abs(float(fitted[0].position[0]) - 0.2), 0.02)
            self.assertLess(abs(float(fitted[0].linear_velocity[0]) - 0.3), 0.05)
            self.assertFalse(report["ground_truth_trajectory_used"])
            self.assertEqual(report["frame_indices"], [0, 1, 2])

    def test_geometry_refinement_is_shared_by_collision_and_render(self):
        class Body:
            id = "dynamic_0"

        asset = RenderableGaussianAsset(
            xyz=torch.zeros(3, 3), features_dc=torch.zeros(3, 1, 3),
            features_rest=torch.zeros(3, 0, 3), opacity=torch.zeros(3, 1),
            scaling=torch.zeros(3, 3), rotation=torch.tensor([[1.0, 0, 0, 0]]).repeat(3, 1),
            features_geo=torch.zeros(3, 0), foreground_logit=torch.zeros(3, 1),
            object_ids=torch.ones(3, dtype=torch.int32), sh_degree=0,
            source_indices=torch.tensor([0, 1, 2]),
        )
        collision = GaussianCollisionBody(
            torch.zeros(2, 3), torch.ones(2), torch.zeros(2, 3), torch.tensor([0, 2])
        )
        center_delta = torch.nn.Parameter(torch.ones(2, 3) * 0.1)
        radius_delta = torch.nn.Parameter(torch.ones(2) * 0.1)
        refined, render_centers, render_radii, _ = refined_geometry_overrides(
            [Body()], [collision], [asset], [center_delta], [radius_delta],
            max_center_delta=0.01, max_log_radius_delta=0.2,
        )
        objective = (
            refined[0].local_centers.sum() + refined[0].radii.sum()
            + render_centers[0].sum() + render_radii[0].sum()
        )
        objective.backward()
        self.assertIsNotNone(center_delta.grad)
        self.assertIsNotNone(radius_delta.grad)
        self.assertTrue(torch.allclose(refined[0].local_centers, render_centers[0][[0, 2]]))
        self.assertTrue(refined[0].local_centers.requires_grad)
        self.assertFalse(render_centers[0].requires_grad)
        self.assertFalse(render_radii[0].requires_grad)

    def test_geometry_render_gradient_is_only_enabled_for_ablation_route(self):
        class Body:
            id = "dynamic_0"

        asset = RenderableGaussianAsset(
            xyz=torch.zeros(2, 3), features_dc=torch.zeros(2, 1, 3),
            features_rest=torch.zeros(2, 0, 3), opacity=torch.zeros(2, 1),
            scaling=torch.zeros(2, 3), rotation=torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
            features_geo=torch.zeros(2, 0), foreground_logit=torch.zeros(2, 1),
            object_ids=torch.ones(2, dtype=torch.int32), sh_degree=0,
            source_indices=torch.tensor([0, 1]),
        )
        collision = GaussianCollisionBody(
            torch.zeros(1, 3), torch.ones(1), torch.zeros(1, 3), torch.tensor([1])
        )
        center_delta = torch.nn.Parameter(torch.ones(1, 3) * 0.1)
        radius_delta = torch.nn.Parameter(torch.ones(1) * 0.1)
        _, render_centers, render_radii, _ = refined_geometry_overrides(
            [Body()], [collision], [asset], [center_delta], [radius_delta],
            max_center_delta=0.01, max_log_radius_delta=0.2,
            gradient_route="collision_and_render",
        )
        self.assertTrue(render_centers[0].requires_grad)
        self.assertTrue(render_radii[0].requires_grad)
        (render_centers[0].sum() + render_radii[0].sum()).backward()
        self.assertGreater(float(center_delta.grad.abs().sum()), 0.0)
        self.assertGreater(float(radius_delta.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
