"""Object-name-independent scene contract for ContactGaussian-WM Stage II."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal


BodyRole = Literal["dynamic", "kinematic", "static"]
CollisionType = Literal["gaussian_union", "plane", "query_points", "none"]


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class RenderSpec:
    gaussian_ply: Path
    foreground_threshold: float | None = None
    opacity_threshold: float | None = None
    object_id: int | None = None
    recenter: bool = False


@dataclass(frozen=True)
class CollisionSpec:
    type: CollisionType
    gaussian_ply: Path | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BodySpec:
    id: str
    role: BodyRole
    render: RenderSpec | None
    collision: CollisionSpec
    physics: dict[str, Any]
    initialization: dict[str, Any]
    trajectory: Path | None = None


@dataclass(frozen=True)
class ObservationSpec:
    rgb_dir: Path
    instance_mask_dir: Path | None
    camera_manifest: Path | None
    fps: float | None
    timestamps: tuple[float, ...] | None


@dataclass(frozen=True)
class ContactPairSpec:
    body_a: str
    body_b: str
    model: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class SceneManifest:
    version: int
    scene_id: str
    path: Path
    bodies: tuple[BodySpec, ...]
    environment: tuple[BodySpec, ...]
    observations: ObservationSpec
    contact_pairs: tuple[ContactPairSpec, ...]
    actions: dict[str, Any]
    simulation: dict[str, Any]
    training: dict[str, Any]
    evaluation_trajectory: Path | None

    @property
    def all_bodies(self) -> tuple[BodySpec, ...]:
        return self.bodies + self.environment

    def body(self, body_id: str) -> BodySpec:
        matches = [body for body in self.all_bodies if body.id == body_id]
        if len(matches) != 1:
            raise KeyError(body_id)
        return matches[0]


def _parse_render(payload: dict[str, Any] | None, base: Path) -> RenderSpec | None:
    if payload is None:
        return None
    gaussian_ply = _resolve(base, payload.get("gaussian_ply"))
    if gaussian_ply is None:
        raise ValueError("render.gaussian_ply is required when render is present")
    return RenderSpec(
        gaussian_ply=gaussian_ply,
        foreground_threshold=payload.get("foreground_threshold"),
        opacity_threshold=payload.get("opacity_threshold"),
        object_id=None if payload.get("object_id") is None else int(payload["object_id"]),
        recenter=bool(payload.get("recenter", False)),
    )


def _parse_collision(payload: dict[str, Any] | None, base: Path) -> CollisionSpec:
    payload = payload or {"type": "none"}
    collision_type = str(payload.get("type", "none"))
    parameters = {
        key: value for key, value in payload.items() if key not in {"type", "gaussian_ply"}
    }
    return CollisionSpec(
        type=collision_type,  # validated separately for clearer aggregated errors
        gaussian_ply=_resolve(base, payload.get("gaussian_ply")),
        parameters=parameters,
    )


def _parse_body(payload: dict[str, Any], base: Path) -> BodySpec:
    return BodySpec(
        id=str(payload.get("id", "")).strip(),
        role=str(payload.get("role", "")).strip(),
        render=_parse_render(payload.get("render"), base),
        collision=_parse_collision(payload.get("collision"), base),
        physics=dict(payload.get("physics") or {}),
        initialization=dict(payload.get("initialization") or {}),
        trajectory=_resolve(base, payload.get("trajectory")),
    )


def load_scene_manifest(path: str | Path, *, validate: bool = True, check_paths: bool = True) -> SceneManifest:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    base = path.parent
    observations_payload = dict(payload.get("observations") or {})
    rgb_dir = _resolve(base, observations_payload.get("rgb_dir"))
    if rgb_dir is None:
        rgb_dir = (base / "observations" / "rgb").resolve()
    timestamps = observations_payload.get("timestamps")
    evaluation = payload.get("evaluation") or {}
    manifest = SceneManifest(
        version=int(payload.get("version", 1)),
        scene_id=str(payload.get("scene_id", "")).strip(),
        path=path,
        bodies=tuple(_parse_body(item, base) for item in payload.get("bodies", [])),
        environment=tuple(_parse_body(item, base) for item in payload.get("environment", [])),
        observations=ObservationSpec(
            rgb_dir=rgb_dir,
            instance_mask_dir=_resolve(base, observations_payload.get("instance_mask_dir")),
            camera_manifest=_resolve(base, observations_payload.get("camera_manifest")),
            fps=None if observations_payload.get("fps") is None else float(observations_payload["fps"]),
            timestamps=None if timestamps is None else tuple(float(value) for value in timestamps),
        ),
        contact_pairs=tuple(
            ContactPairSpec(
                body_a=str(item.get("body_a", "")).strip(),
                body_b=str(item.get("body_b", "")).strip(),
                model=str(item.get("model", "dual_cone")),
                parameters={
                    key: value for key, value in item.items() if key not in {"body_a", "body_b", "model"}
                },
            )
            for item in payload.get("contact_pairs", [])
        ),
        actions=dict(payload.get("actions") or {}),
        simulation=dict(payload.get("simulation") or {}),
        training=dict(payload.get("training") or {}),
        evaluation_trajectory=_resolve(base, evaluation.get("trajectory")),
    )
    if validate:
        errors = validate_scene_manifest(manifest, check_paths=check_paths)
        if errors:
            raise ValueError("Invalid scene manifest:\n- " + "\n- ".join(errors))
    return manifest


def validate_scene_manifest(manifest: SceneManifest, *, check_paths: bool = True) -> list[str]:
    errors: list[str] = []
    if manifest.version != 1:
        errors.append(f"unsupported version {manifest.version}; expected 1")
    if not manifest.scene_id:
        errors.append("scene_id must be non-empty")
    if not manifest.bodies:
        errors.append("bodies must contain at least one modeled rigid body")

    ids = [body.id for body in manifest.all_bodies]
    for body_id in ids:
        if not body_id:
            errors.append("every body/environment entry requires a non-empty id")
    duplicates = sorted({body_id for body_id in ids if ids.count(body_id) > 1})
    if duplicates:
        errors.append(f"body ids must be unique: {duplicates}")

    allowed_roles = {"dynamic", "kinematic", "static"}
    allowed_collisions = {"gaussian_union", "plane", "query_points", "none"}
    for body in manifest.all_bodies:
        prefix = f"body {body.id or '<missing>'}"
        if body.role not in allowed_roles:
            errors.append(f"{prefix}: role must be dynamic, kinematic, or static")
        if body.collision.type not in allowed_collisions:
            errors.append(f"{prefix}: unsupported collision type {body.collision.type!r}")
        if body.role == "dynamic" and body.render is None:
            errors.append(f"{prefix}: dynamic bodies require a render Gaussian asset")
        if body.role == "kinematic" and body.trajectory is None:
            errors.append(f"{prefix}: kinematic bodies require trajectory")
        if body.collision.type == "gaussian_union":
            collision_ply = body.collision.gaussian_ply or (
                body.render.gaussian_ply if body.render is not None else None
            )
            if collision_ply is None:
                errors.append(f"{prefix}: gaussian_union requires collision.gaussian_ply or render.gaussian_ply")
        if body.collision.type == "plane":
            normal = body.collision.parameters.get("normal", [0.0, 0.0, 1.0])
            if not isinstance(normal, list) or len(normal) != 3:
                errors.append(f"{prefix}: plane normal must contain three values")
            elif sum(float(value) ** 2 for value in normal) <= 1e-12:
                errors.append(f"{prefix}: plane normal must be non-zero")
        for key in ("foreground_threshold", "opacity_threshold"):
            value = getattr(body.render, key) if body.render is not None else None
            if value is not None and not 0.0 <= float(value) <= 1.0:
                errors.append(f"{prefix}: render.{key} must be in [0, 1]")

    if manifest.observations.fps is None and manifest.observations.timestamps is None:
        errors.append("observations requires fps or timestamps")
    if manifest.observations.fps is not None and manifest.observations.fps <= 0.0:
        errors.append("observations.fps must be positive")
    if manifest.observations.timestamps is not None:
        timestamps = manifest.observations.timestamps
        if len(timestamps) < 3:
            errors.append("observations.timestamps must contain at least three frames")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            errors.append("observations.timestamps must be strictly increasing")

    physics_timestep = manifest.simulation.get("physics_timestep")
    if physics_timestep is not None and float(physics_timestep) <= 0.0:
        errors.append("simulation.physics_timestep must be positive")
    steps_per_frame = manifest.simulation.get("steps_per_frame")
    if steps_per_frame is not None:
        try:
            valid_steps = (
                not isinstance(steps_per_frame, bool)
                and int(steps_per_frame) == float(steps_per_frame)
                and int(steps_per_frame) >= 1
            )
        except (TypeError, ValueError, OverflowError):
            valid_steps = False
        if not valid_steps:
            errors.append("simulation.steps_per_frame must be a positive integer")

    known_ids = set(ids)
    seen_pairs: set[tuple[str, str]] = set()
    for pair in manifest.contact_pairs:
        if pair.body_a not in known_ids:
            errors.append(f"contact pair references unknown body_a {pair.body_a!r}")
        if pair.body_b not in known_ids:
            errors.append(f"contact pair references unknown body_b {pair.body_b!r}")
        if pair.body_a == pair.body_b:
            errors.append(f"contact pair cannot reference the same body twice: {pair.body_a!r}")
        canonical = tuple(sorted((pair.body_a, pair.body_b)))
        if canonical in seen_pairs:
            errors.append(f"duplicate contact pair: {canonical}")
        seen_pairs.add(canonical)

    if check_paths:
        required_paths: list[tuple[str, Path | None]] = [("observations.rgb_dir", manifest.observations.rgb_dir)]
        if manifest.observations.camera_manifest is not None:
            required_paths.append(("observations.camera_manifest", manifest.observations.camera_manifest))
        for body in manifest.all_bodies:
            if body.render is not None:
                required_paths.append((f"body {body.id} render.gaussian_ply", body.render.gaussian_ply))
            if body.collision.gaussian_ply is not None:
                required_paths.append((f"body {body.id} collision.gaussian_ply", body.collision.gaussian_ply))
            if body.trajectory is not None:
                required_paths.append((f"body {body.id} trajectory", body.trajectory))
        for label, required_path in required_paths:
            if required_path is None or not required_path.exists():
                errors.append(f"{label} does not exist: {required_path}")
        if manifest.observations.rgb_dir.exists() and not any(
            path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            for path in manifest.observations.rgb_dir.iterdir()
        ):
            errors.append(f"observations.rgb_dir contains no image frames: {manifest.observations.rgb_dir}")
    return errors


def manifest_summary(manifest: SceneManifest) -> dict[str, Any]:
    return {
        "version": manifest.version,
        "scene_id": manifest.scene_id,
        "manifest": str(manifest.path),
        "body_ids": [body.id for body in manifest.bodies],
        "environment_ids": [body.id for body in manifest.environment],
        "dynamic_body_ids": [body.id for body in manifest.all_bodies if body.role == "dynamic"],
        "contact_pairs": [
            {"body_a": pair.body_a, "body_b": pair.body_b, "model": pair.model}
            for pair in manifest.contact_pairs
        ],
        "rgb_dir": str(manifest.observations.rgb_dir),
        "camera_manifest": (
            None if manifest.observations.camera_manifest is None else str(manifest.observations.camera_manifest)
        ),
        "fps": manifest.observations.fps,
        "physics_timestep": manifest.simulation.get("physics_timestep"),
        "steps_per_frame": manifest.simulation.get("steps_per_frame"),
        "has_timestamps": manifest.observations.timestamps is not None,
        "evaluation_trajectory": (
            None if manifest.evaluation_trajectory is None else str(manifest.evaluation_trajectory)
        ),
        "uses_object_shape_presets": False,
    }
