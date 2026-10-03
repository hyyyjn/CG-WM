"""Contact physics from tracked poses: multiple shooting without rendering.

The image tracker (``video_physics_fit.track_poses``) turns every frame into a
pose measurement.  Here the simulator is fitted to those measurements directly,
so one optimisation step costs only the (CPU) dynamics and the windows of a
multiple-shooting problem run in parallel worker processes.

Residuals are expressed in pixels, using the camera geometry of each frame, so
the three kinds of error are commensurable without hand-set weights:

* across the viewing ray a metre moves the object ``f / range`` pixels;
* along the ray it only rescales the silhouette, ``R_px / range`` pixels per
  metre (``R_px``: the silhouette's radius in pixels) - a single camera
  measures depth far less precisely than direction;
* a radian of rotation moves the surface about ``R_px`` pixels.
"""

from __future__ import annotations

import dataclasses
import math
import multiprocessing
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from stage2.differentiable_complementarity_free_contact_dynamics import RigidBodyState
from stage2.renderable_gaussian_asset import quat_mul_wxyz, quat_to_matrix_wxyz

HUBER_PIXELS = 1.0  # residuals beyond one pixel are tracker outliers, not noise


@dataclass
class PoseTrack:
    """Per-frame pose measurements of one episode (centre-of-mass frame)."""

    name: str
    positions: torch.Tensor  # [F, 3]
    quaternions: torch.Tensor  # [F, 4]
    position_weight: torch.Tensor  # [F, 3, 3] pixels per metre
    rotation_weight: torch.Tensor  # [F] pixels per radian
    valid: torch.Tensor  # [F] bool: the object was visible
    frame_dt: float
    physics_dt: float
    substeps: int

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def make_pose_track(name: str, positions: torch.Tensor, quaternions: torch.Tensor, mask_area: torch.Tensor, *,
                    camera_position, fovy_deg: float, image_height: int, frame_dt: float, physics_dt: float,
                    substeps: int) -> PoseTrack:
    """Pose measurements with their pixel-unit weights from the camera geometry."""
    positions = positions.detach().cpu().float()
    quaternions = F.normalize(quaternions.detach().cpu().float(), dim=-1)
    mask_area = torch.as_tensor(mask_area, dtype=torch.float32).reshape(-1)
    focal = 0.5 * float(image_height) / math.tan(0.5 * math.radians(float(fovy_deg)))
    offset = positions - torch.as_tensor(camera_position, dtype=torch.float32)
    distance = offset.norm(dim=-1).clamp_min(1e-6)
    ray = offset / distance.unsqueeze(-1)
    radius_px = torch.sqrt(mask_area.clamp_min(0.0) / math.pi)
    along = ray.unsqueeze(-1) * ray.unsqueeze(-2)
    across = torch.eye(3).expand_as(along) - along
    weight = (focal / distance).reshape(-1, 1, 1) * across + (radius_px / distance).reshape(-1, 1, 1) * along
    return PoseTrack(name, positions, quaternions, weight, radius_px, mask_area >= 1.0,
                     float(frame_dt), float(physics_dt), int(substeps))


def pose_residual_loss(positions: torch.Tensor, quaternions: torch.Tensor, track: PoseTrack, t0: int):
    """Sum of Huber pose residuals (pixels^2) of a rollout that starts at frame ``t0``; also the frame count."""
    frames = slice(t0, t0 + positions.shape[0])
    valid = track.valid[frames]
    translation = (track.position_weight[frames] @ (positions - track.positions[frames]).unsqueeze(-1)).squeeze(-1)
    measured = track.quaternions[frames]
    delta = quat_mul_wxyz(F.normalize(quaternions, dim=-1), torch.cat((measured[:, :1], -measured[:, 1:]), dim=-1))
    rotation = 2.0 * delta[:, 1:] * torch.sign(delta[:, :1] + 1e-12) * track.rotation_weight[frames].unsqueeze(-1)
    squared = (translation ** 2).sum(-1) + (rotation ** 2).sum(-1)
    norm = torch.sqrt(squared + 1e-12)
    huber = torch.where(norm <= HUBER_PIXELS, 0.5 * squared, HUBER_PIXELS * (norm - 0.5 * HUBER_PIXELS))
    return (huber * valid).sum(), int(valid.sum())


def contact_onsets(obj, track: PoseTrack, *, proximity: float = 0.05) -> list[int]:
    """Frames at which the tracked body comes down onto the support plane after being clear of it.

    A frame counts as touching when the lowest collision primitive of the
    tracked pose is within ``proximity`` of the body's size from the plane.
    """
    rotation = quat_to_matrix_wxyz(track.quaternions)
    centers = obj.collision.local_centers.float()
    world = torch.einsum("fij,nj->fni", rotation, centers) + track.positions[:, None, :]
    normal = torch.as_tensor(obj.support_normal, dtype=torch.float32)
    clearance = (world @ normal - float(obj.support_height) - obj.collision.radii.float()).min(dim=1).values
    size = float((centers.max(dim=0).values - centers.min(dim=0).values).max())
    touching = (clearance < proximity * size).tolist()
    return [frame for frame in range(1, len(touching)) if touching[frame] and not touching[frame - 1]]


def lift_out_of_support(obj, state: RigidBodyState) -> RigidBodyState:
    """Move a start state out of the support plane along its normal if any primitive is below it.

    A pose measured from images can sit a few millimetres inside the floor (its
    error along the viewing ray); started there, the contact would eject the
    body violently and the rollout would have nothing to do with the video.
    """
    normal = torch.as_tensor(obj.support_normal, dtype=torch.float32)
    rotation = quat_to_matrix_wxyz(F.normalize(state.quaternion_wxyz.detach().float(), dim=-1))
    world = obj.collision.local_centers.float() @ rotation.T + state.position.detach().float()
    clearance = float((world @ normal - float(obj.support_height) - obj.collision.radii.float()).min())
    if clearance >= 0.0:
        return state
    return RigidBodyState(state.position.detach() - clearance * normal.to(state.position.dtype),
                          state.quaternion_wxyz, state.linear_velocity, state.angular_velocity)


def observation_start(onsets: list[int], t: int, window: int, *, minimum: int = 2) -> int:
    """First frame of the observation used to forecast from frame ``t``.

    The last ``window`` frames, but not before the most recent impact: the
    simulated impact never reproduces the real one exactly, and a state fitted
    across it would start the forecast off the observed motion.
    """
    start = max(0, t - int(window))
    for onset in onsets:
        if onset <= t:
            start = max(start, onset)
    return min(start, max(0, t - int(minimum)))


# --------------------------------------------------------------------------- workers

_WORKER: dict = {}


def _worker_init(obj, tracks):
    torch.set_num_threads(1)
    _WORKER["obj"], _WORKER["tracks"] = obj, tracks


def _physics_from_logs(logs):
    from stage2.video_physics_fit import ContactPhysics

    physics = ContactPhysics()
    with torch.no_grad():
        for parameter, value in zip(physics.parameters(), logs):
            parameter.fill_(float(value))
    return physics


def _window_loss(physics, track: PoseTrack, t0: int, count: int, leaves):
    from stage2.video_physics_fit import make_dynamics, rollout

    p, q, v, w = leaves
    dynamics = make_dynamics(_WORKER["obj"], physics, dt=track.physics_dt)
    positions, quaternions = rollout(dynamics, RigidBodyState(p, F.normalize(q, dim=-1), v, w), count, track.substeps)
    return pose_residual_loss(positions, quaternions, track, t0)


def _rotation_rate(q_from: torch.Tensor, q_to: torch.Tensor, seconds: float) -> torch.Tensor:
    delta = quat_mul_wxyz(q_to, torch.cat((q_from[:1], -q_from[1:])))
    delta = delta if float(delta[0]) >= 0.0 else -delta
    norm = delta[1:].norm()
    angle = 2.0 * torch.atan2(norm, delta[0])
    axis = delta[1:] / norm.clamp_min(1e-12)
    return axis * angle / float(seconds)


def start_state_candidates(track: PoseTrack, t0: int) -> list[RigidBodyState]:
    """The measured pose at ``t0`` with central, forward and backward difference velocities.

    A central difference across an impact averages the velocities before and
    after it (a bounce at the start frame would look like a body at rest); one
    of the one-sided differences then lies on the right side of the impact.
    """
    last = len(track) - 1
    p, q, dt = track.positions, track.quaternions, track.frame_dt
    spans = [(max(0, t0 - 1), min(last, t0 + 1)), (t0, min(last, t0 + 1)), (max(0, t0 - 1), t0)]
    candidates = []
    for lo, hi in spans:
        if hi == lo:
            continue
        seconds = float(hi - lo) * dt
        candidates.append(RigidBodyState(p[t0].clone(), q[t0].clone(), (p[hi] - p[lo]) / seconds,
                                         _rotation_rate(q[lo], q[hi], seconds)))
    return candidates


def _initial_state_task(task):
    """The start-state candidate of a window that best explains it under the given physics (no gradients)."""
    logs, episode_index, t0, count = task
    physics = _physics_from_logs(logs)
    track = _WORKER["tracks"][episode_index]
    best = None
    for candidate in start_state_candidates(track, t0):
        state = lift_out_of_support(_WORKER["obj"], candidate)
        leaves = [state.position, state.quaternion_wxyz, state.linear_velocity, state.angular_velocity]
        with torch.no_grad():
            loss, _ = _window_loss(physics, track, t0, count, leaves)
        if best is None or float(loss) < best[0]:
            best = (float(loss), [item.tolist() for item in leaves])
    return best[1]


def _window_gradient_task(task):
    """Loss and gradients of one multiple-shooting window."""
    logs, episode_index, t0, count, state = task
    physics = _physics_from_logs(logs)
    leaves = [torch.nn.Parameter(torch.as_tensor(item, dtype=torch.float32)) for item in state]
    loss, frames = _window_loss(physics, _WORKER["tracks"][episode_index], t0, count, leaves)
    loss.backward()
    zero = lambda parameter: torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
    return (float(loss.detach()), frames, [float(zero(parameter)) for parameter in physics.parameters()],
            [zero(leaf).tolist() for leaf in leaves])


STATE_LEARNING_RATES = (1e-3, 2e-3, 1e-2, 3e-2)  # position [m], quaternion, velocity [m/s], angular velocity [rad/s]
STATE_FIT_PATIENCE = 10  # stop a state fit after this many iterations without a 0.1% improvement
STATIC_PIXELS = 0.5  # a window whose measurements all stay within this of its first frame shows no motion


def window_is_static(track: PoseTrack, t0: int, count: int) -> bool:
    """True when the object does not visibly move during the window (it is at rest)."""
    frames = slice(t0, t0 + count)
    shift = (track.position_weight[frames] @ (track.positions[frames] - track.positions[t0]).unsqueeze(-1)).squeeze(-1)
    first = track.quaternions[t0]
    delta = quat_mul_wxyz(track.quaternions[frames], torch.cat((first[:1], -first[1:])).expand(count, 4))
    turn = 2.0 * delta[:, 1:].norm(dim=-1) * track.rotation_weight[frames]
    moved = torch.sqrt((shift ** 2).sum(-1) + turn ** 2)
    return bool((moved[track.valid[frames]] < STATIC_PIXELS).all())


def _state_fit_task(task):
    """Start state of frames ``t0 .. t0+count-1`` under fixed physics, then a rollout of ``total`` frames."""
    from stage2.video_physics_fit import make_dynamics, rollout

    logs, episode_index, t0, count, _, iters, total = task
    physics = _physics_from_logs(logs)
    track = _WORKER["tracks"][episode_index]
    start = _initial_state_task((logs, episode_index, t0, count))
    leaves = [torch.nn.Parameter(torch.as_tensor(item, dtype=torch.float32)) for item in start]
    optimizer = torch.optim.Adam([{"params": [leaf], "lr": lr} for leaf, lr in zip(leaves, STATE_LEARNING_RATES)])
    best, stalled = None, 0
    for _ in range(int(iters)):
        optimizer.zero_grad()
        loss, _ = _window_loss(physics, track, t0, count, leaves)
        loss.backward()
        value = float(loss.detach())
        stalled = 0 if best is None or value < (1.0 - 1e-3) * best[0] else stalled + 1
        if best is None or value < best[0]:
            best = (value, [leaf.detach().clone() for leaf in leaves])
        if stalled >= STATE_FIT_PATIENCE:
            break
        optimizer.step()
    p, q, v, w = best[1]
    with torch.no_grad():
        positions, quaternions = rollout(make_dynamics(_WORKER["obj"], physics, dt=track.physics_dt),
                                         RigidBodyState(p, F.normalize(q, dim=-1), v, w), total, track.substeps)
    return best[0], [item.tolist() for item in best[1]], positions.tolist(), quaternions.tolist()


class PoseFitPool:
    """Worker processes holding the collision model and the pose tracks."""

    def __init__(self, obj, tracks: list[PoseTrack], workers: int | None = None):
        light = dataclasses.replace(obj, asset=None)  # the renderable asset lives on the GPU
        if workers is None:
            workers = max(1, (os.cpu_count() or 2) - 2)
        self.workers = int(workers)
        if self.workers <= 1:
            _worker_init(light, tracks)
            self.pool = None
        else:
            self.pool = multiprocessing.get_context("spawn").Pool(self.workers, _worker_init, (light, tracks))

    def map(self, function, tasks):
        if self.pool is None:
            return [function(task) for task in tasks]
        return self.pool.map(function, tasks, chunksize=1)

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# --------------------------------------------------------------------------- fitting


def _state_lists(state: RigidBodyState):
    return [item.detach().cpu().tolist() for item in (
        state.position, state.quaternion_wxyz, state.linear_velocity, state.angular_velocity)]


def fit_physics_pose_space(pool: PoseFitPool, tracks: list[PoseTrack], tracked_states: list[list[RigidBodyState]],
                           physics, *, window: int = 10, iters: int = 150, physics_lr: float = 0.05, obj=None, log=print):
    """Shared physics + one start state per ``window`` frames, fitted to the pose measurements."""
    leaves, index, skipped = [], [], 0
    for episode_index, (track, states) in enumerate(zip(tracks, tracked_states)):
        previous_static = False
        for t0 in range(0, max(1, len(track) - 1), int(window)):
            count = min(len(track), t0 + int(window) + 1) - t0
            static = window_is_static(track, t0, count)
            if static and previous_static:  # a rest phase is one measurement, not one per window
                skipped += 1
                continue
            previous_static = static
            index.append((episode_index, t0, count))
            start = states[t0] if obj is None else lift_out_of_support(obj, states[t0])
            leaves.append([torch.nn.Parameter(item.detach().clone().float()) for item in (
                start.position, start.quaternion_wxyz, start.linear_velocity, start.angular_velocity)])
    log(f"  multiple shooting: {len(index)} windows of {window} frames ({skipped} repeated rest windows left out)")
    if obj is not None:  # start states: the velocity estimate that explains each window best
        logs = [float(parameter.detach()) for parameter in physics.parameters()]
        starts = pool.map(_initial_state_task, [(logs, episode_index, t0, count) for episode_index, t0, count in index])
        leaves = [[torch.nn.Parameter(torch.as_tensor(item, dtype=torch.float32)) for item in start] for start in starts]
    parameters = list(physics.parameters())
    groups = [{"params": parameters, "lr": physics_lr}]
    for window_leaves in leaves:
        groups += [{"params": [leaf], "lr": lr} for leaf, lr in zip(window_leaves, STATE_LEARNING_RATES)]
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(iters), eta_min=0.0)
    history = []
    for iteration in range(int(iters)):
        logs = [float(parameter.detach()) for parameter in parameters]
        tasks = [(logs, episode_index, t0, count, [leaf.detach().tolist() for leaf in window_leaves])
                 for (episode_index, t0, count), window_leaves in zip(index, leaves)]
        results = pool.map(_window_gradient_task, tasks)
        frames = max(1, sum(result[1] for result in results))
        optimizer.zero_grad()
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        total = 0.0
        for (loss, _, physics_gradient, state_gradient), window_leaves in zip(results, leaves):
            total += loss / frames
            for parameter, value in zip(parameters, physics_gradient):
                parameter.grad += value / frames
            for leaf, value in zip(window_leaves, state_gradient):
                leaf.grad = torch.as_tensor(value, dtype=leaf.dtype) / frames
        record = {"iteration": iteration, "loss": total, **physics.as_dict()}
        history.append(record)
        optimizer.step()
        scheduler.step()
        if iteration % 10 == 0 or iteration == int(iters) - 1:
            log(f"  physics it {iteration:3d}: loss {total:.4f} px^2 tau {record['timeconst']:.4f} "
                f"zeta {record['dampratio']:.3f} mu {record['friction']:.3f} damping {record['damping']:.4f}")
    fitted = [RigidBodyState(p.detach(), F.normalize(q.detach(), dim=-1), v.detach(), w.detach())
              for p, q, v, w in leaves]
    return physics, index, fitted, history


def fit_states_and_rollout(pool: PoseFitPool, physics, requests: list[tuple], *, iters: int = 60):
    """For each ``(episode_index, t0, observed_frames, start_state, total_frames)``: fit, then roll out."""
    logs = [float(parameter.detach()) for parameter in physics.parameters()]
    tasks = [(logs, episode_index, t0, count, _state_lists(state), iters, total)
             for episode_index, t0, count, state, total in requests]
    results = []
    for loss, state, positions, quaternions in pool.map(_state_fit_task, tasks):
        p, q, v, w = [torch.tensor(item) for item in state]
        results.append({"loss": loss, "state": RigidBodyState(p, F.normalize(q, dim=-1), v, w),
                        "positions": torch.tensor(positions), "quaternions": torch.tensor(quaternions)})
    return results
