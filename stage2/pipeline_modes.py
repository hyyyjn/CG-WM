"""Execution-mode contracts for the staged ContactGaussian-WM implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Stage2PipelineMode(str, Enum):
    PAPER_COMPATIBLE = "paper_compatible"
    IMAGE_ONLY = "image_only"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class Stage2ModeContract:
    mode: Stage2PipelineMode
    description: str
    initial_state_source: str
    action_source: str
    supervision: str
    allow_initial_state_prefit: bool
    allow_temporal_windows: bool
    allow_nonpaper_geometry_gradient: bool

    def to_serializable(self) -> dict:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


MODE_CONTRACTS = {
    Stage2PipelineMode.PAPER_COMPATIBLE: Stage2ModeContract(
        mode=Stage2PipelineMode.PAPER_COMPATIBLE,
        description="Paper-reference path; components are replaced and verified step by step.",
        initial_state_source="known_manifest_state",
        action_source="manifest_or_zero_action",
        supervision="full_image_l1_plus_loftr",
        allow_initial_state_prefit=False,
        allow_temporal_windows=False,
        allow_nonpaper_geometry_gradient=False,
    ),
    Stage2PipelineMode.IMAGE_ONLY: Stage2ModeContract(
        mode=Stage2PipelineMode.IMAGE_ONLY,
        description="Video-only extension with estimated initial state.",
        initial_state_source="image_estimation",
        action_source="manifest_or_zero_action",
        supervision="configurable_image_loss",
        allow_initial_state_prefit=True,
        allow_temporal_windows=False,
        allow_nonpaper_geometry_gradient=False,
    ),
    Stage2PipelineMode.EXPERIMENTAL: Stage2ModeContract(
        mode=Stage2PipelineMode.EXPERIMENTAL,
        description="Ablations and memory-oriented extensions outside the paper contract.",
        initial_state_source="configurable",
        action_source="manifest_or_zero_action",
        supervision="configurable_image_loss",
        allow_initial_state_prefit=True,
        allow_temporal_windows=True,
        allow_nonpaper_geometry_gradient=True,
    ),
}


def resolve_stage2_mode(value: str | Stage2PipelineMode) -> Stage2ModeContract:
    try:
        mode = value if isinstance(value, Stage2PipelineMode) else Stage2PipelineMode(str(value))
    except ValueError as error:
        choices = ", ".join(item.value for item in Stage2PipelineMode)
        raise ValueError(f"unknown Stage2 pipeline mode {value!r}; choose one of: {choices}") from error
    return MODE_CONTRACTS[mode]


def validate_stage2_mode_options(
    contract: Stage2ModeContract, *, prefit_initial_state: bool,
    temporal_window_frames: int, geometry_gradient_route: str,
) -> None:
    if prefit_initial_state and not contract.allow_initial_state_prefit:
        raise ValueError(f"{contract.mode.value} mode requires a known initial state; prefit is not allowed")
    if temporal_window_frames > 0 and not contract.allow_temporal_windows:
        raise ValueError(
            f"temporal windows belong to experimental mode, not {contract.mode.value}"
        )
    if geometry_gradient_route != "collision_only" and not contract.allow_nonpaper_geometry_gradient:
        raise ValueError(
            f"{geometry_gradient_route!r} geometry gradients belong to experimental mode"
        )
