"""Smoke test for multi-object pairwise contact graph construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    CollisionEngineConfig,
    GaussianCollisionBody,
    make_box_surface_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import RigidBodyState  # noqa: E402
from gaussian_initiailization.stage2.differentiable_contact_graph import build_pairwise_contact_graph  # noqa: E402

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "pairwise_contact_graph"


def make_box_body(half_extent: float = 0.1, resolution: int = 3) -> GaussianCollisionBody:
    query_points = make_box_surface_query_points(
        [half_extent, half_extent, half_extent],
        grid_resolution=resolution,
        dtype=torch.float32,
    )
    spacing = 2.0 * float(half_extent) / float(max(resolution - 1, 1))
    radii = torch.full((query_points.shape[0],), spacing * 0.25, dtype=torch.float32)
    return GaussianCollisionBody(query_points, radii, query_points)


def make_state(position: torch.Tensor) -> RigidBodyState:
    return RigidBodyState(
        position=position,
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=position.dtype, device=position.device),
        linear_velocity=torch.zeros(3, dtype=position.dtype, device=position.device),
        angular_velocity=torch.zeros(3, dtype=position.dtype, device=position.device),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contact_threshold = 0.05
    body = make_box_body()
    position_a = torch.tensor([-0.08, 0.0, 0.0], requires_grad=True)
    states = (
        make_state(position_a),
        make_state(torch.tensor([0.08, 0.0, 0.0])),
        make_state(torch.tensor([0.70, 0.0, 0.0])),
    )
    graph = build_pairwise_contact_graph(
        (body, body, body),
        states,
        names=("moving_box", "near_box", "far_box"),
        dynamic_flags=(True, False, False),
        collision_config=CollisionEngineConfig(
            softness=1e-3,
            smooth_min_temperature=1e-2,
            num_contact_patches=4,
            broad_phase_margin=0.02,
            broad_phase_mode="aabb",
        ),
        include_inactive=True,
    )
    hash_graph = build_pairwise_contact_graph(
        (body, body, body),
        states,
        names=("moving_box", "near_box", "far_box"),
        dynamic_flags=(True, False, False),
        candidate_pair_mode="spatial_hash",
        spatial_hash_cell_size=0.25,
        collision_config=CollisionEngineConfig(
            softness=1e-3,
            smooth_min_temperature=1e-2,
            num_contact_patches=4,
            broad_phase_margin=0.02,
            broad_phase_mode="aabb",
        ),
        include_inactive=False,
    )
    active_edges = graph.active_edges(contact_threshold=contact_threshold)
    if not active_edges:
        raise RuntimeError("Expected at least one active contact edge.")

    contact_loss = sum(edge.max_patch_weight + edge.max_penetration for edge in active_edges)
    contact_loss.backward()
    adjacency = graph.adjacency_matrix(contact_threshold=contact_threshold)
    active_pairs = [(edge.body_i, edge.body_j) for edge in active_edges]
    hash_active_pairs = [(edge.body_i, edge.body_j) for edge in hash_graph.active_edges(contact_threshold=contact_threshold)]
    summary = {
        **graph.to_serializable(contact_threshold=contact_threshold),
        "contact_threshold": contact_threshold,
        "broad_phase_mode": "aabb",
        "num_bodies": graph.num_bodies,
        "num_edges": graph.num_edges,
        "num_active_edges": len(active_edges),
        "active_pairs": active_pairs,
        "spatial_hash_num_edges": hash_graph.num_edges,
        "spatial_hash_active_pairs": hash_active_pairs,
        "spatial_hash_only_expected_pair": hash_active_pairs == [(0, 1)] and hash_graph.num_edges == 1,
        "expected_active_pair_present": (0, 1) in active_pairs,
        "far_pair_inactive": not bool(adjacency[1, 2].detach().cpu().item()) and not bool(adjacency[0, 2].detach().cpu().item()),
        "position_a_grad": position_a.grad.detach().cpu().tolist(),
        "gradient_is_finite": bool(torch.isfinite(position_a.grad).all().detach().cpu().item()),
    }
    output_path = OUT_DIR / "pairwise_contact_graph_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
