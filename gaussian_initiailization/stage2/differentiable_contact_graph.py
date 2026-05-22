"""Pairwise contact graph construction for multi-object Stage 2 scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .differentiable_collision_detection import (
    BodyPairContacts,
    CollisionEngineConfig,
    DifferentiableCollisionEngine,
    GaussianCollisionBody,
)


@dataclass(frozen=True)
class ContactGraphNode:
    """One rigid object in a multi-object contact graph."""

    name: str
    body: GaussianCollisionBody
    state: object
    dynamic: bool = True


@dataclass(frozen=True)
class PairwiseContactGraphEdge:
    """Contact edge between two rigid objects."""

    body_i: int
    body_j: int
    contacts: BodyPairContacts

    @property
    def max_patch_weight(self) -> torch.Tensor:
        return torch.max(self.contacts.patch_weights)

    @property
    def max_penetration(self) -> torch.Tensor:
        return torch.max(self.contacts.patch_penetrations)

    @property
    def min_signed_distance(self) -> torch.Tensor:
        return torch.min(self.contacts.patch_signed_distances)

    @property
    def broad_phase_overlap(self) -> torch.Tensor:
        return self.contacts.broad_phase_overlaps

    def is_active(self, contact_threshold: float = 0.5) -> torch.Tensor:
        return self.broad_phase_overlap & (self.max_patch_weight > float(contact_threshold))


@dataclass(frozen=True)
class PairwiseContactGraph:
    """Sparse graph of pairwise Gaussian-body contacts for one scene state."""

    nodes: tuple[ContactGraphNode, ...]
    edges: tuple[PairwiseContactGraphEdge, ...]

    @property
    def num_bodies(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def active_edges(self, contact_threshold: float = 0.5) -> tuple[PairwiseContactGraphEdge, ...]:
        active = []
        for edge in self.edges:
            if bool(edge.is_active(contact_threshold).detach().cpu().item()):
                active.append(edge)
        return tuple(active)

    def adjacency_matrix(self, contact_threshold: float = 0.5) -> torch.Tensor:
        if not self.edges:
            return torch.zeros((self.num_bodies, self.num_bodies), dtype=torch.bool)
        device = self.edges[0].contacts.patch_weights.device
        adjacency = torch.zeros((self.num_bodies, self.num_bodies), dtype=torch.bool, device=device)
        for edge in self.edges:
            active = edge.is_active(contact_threshold)
            adjacency[edge.body_i, edge.body_j] = active
            adjacency[edge.body_j, edge.body_i] = active
        return adjacency

    def to_serializable(self, contact_threshold: float = 0.5) -> dict:
        adjacency = self.adjacency_matrix(contact_threshold).detach().cpu().tolist()
        return {
            "nodes": [
                {
                    "index": idx,
                    "name": node.name,
                    "dynamic": bool(node.dynamic),
                }
                for idx, node in enumerate(self.nodes)
            ],
            "edges": [
                {
                    "body_i": int(edge.body_i),
                    "body_j": int(edge.body_j),
                    "name_i": self.nodes[edge.body_i].name,
                    "name_j": self.nodes[edge.body_j].name,
                    "broad_phase_mode": edge.contacts.broad_phase_mode,
                    "broad_phase_overlap": bool(edge.broad_phase_overlap.detach().cpu().item()),
                    "active": bool(edge.is_active(contact_threshold).detach().cpu().item()),
                    "max_patch_weight": float(edge.max_patch_weight.detach().cpu().item()),
                    "max_penetration": float(edge.max_penetration.detach().cpu().item()),
                    "min_signed_distance": float(edge.min_signed_distance.detach().cpu().item()),
                }
                for edge in self.edges
            ],
            "adjacency": adjacency,
        }


def _default_names(num_bodies: int) -> tuple[str, ...]:
    return tuple(f"body_{idx}" for idx in range(num_bodies))


def _all_candidate_pairs(num_bodies: int) -> tuple[tuple[int, int], ...]:
    return tuple((idx, jdx) for idx in range(num_bodies) for jdx in range(idx + 1, num_bodies))


def spatial_hash_candidate_pairs(
    bodies: Sequence[GaussianCollisionBody],
    states: Sequence[object],
    *,
    cell_size: float,
    margin: float = 0.0,
) -> tuple[tuple[int, int], ...]:
    """Return candidate body pairs whose world AABBs occupy at least one grid cell."""

    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive.")
    cells_to_bodies: dict[tuple[int, int, int], set[int]] = {}
    for body_idx, (body, state) in enumerate(zip(bodies, states)):
        min_corner, max_corner = body.world_aabb(
            state.position,
            quaternion_wxyz=state.quaternion_wxyz,
        )
        min_cell = torch.floor((min_corner.detach().cpu() - float(margin)) / float(cell_size)).to(torch.int64)
        max_cell = torch.floor((max_corner.detach().cpu() + float(margin)) / float(cell_size)).to(torch.int64)
        for x in range(int(min_cell[0].item()), int(max_cell[0].item()) + 1):
            for y in range(int(min_cell[1].item()), int(max_cell[1].item()) + 1):
                for z in range(int(min_cell[2].item()), int(max_cell[2].item()) + 1):
                    cells_to_bodies.setdefault((x, y, z), set()).add(body_idx)

    pairs: set[tuple[int, int]] = set()
    for body_indices in cells_to_bodies.values():
        ordered = sorted(body_indices)
        for local_i, body_i in enumerate(ordered):
            for body_j in ordered[local_i + 1:]:
                pairs.add((body_i, body_j))
    return tuple(sorted(pairs))


def _validate_scene_inputs(
    bodies: Sequence[GaussianCollisionBody],
    states: Sequence[object],
    names: Sequence[str] | None,
    dynamic_flags: Sequence[bool] | None,
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    if len(bodies) != len(states):
        raise ValueError(f"bodies and states must have equal length, got {len(bodies)} and {len(states)}.")
    if len(bodies) < 2:
        raise ValueError("Need at least two bodies to build a pairwise contact graph.")
    for idx, state in enumerate(states):
        for attribute in ("position", "quaternion_wxyz"):
            if not hasattr(state, attribute):
                raise ValueError(f"state {idx} is missing required attribute {attribute!r}.")

    resolved_names = _default_names(len(bodies)) if names is None else tuple(str(name) for name in names)
    if len(resolved_names) != len(bodies):
        raise ValueError(f"names must have length {len(bodies)}, got {len(resolved_names)}.")

    resolved_dynamic = tuple(True for _ in bodies) if dynamic_flags is None else tuple(bool(flag) for flag in dynamic_flags)
    if len(resolved_dynamic) != len(bodies):
        raise ValueError(f"dynamic_flags must have length {len(bodies)}, got {len(resolved_dynamic)}.")
    return resolved_names, resolved_dynamic


def build_pairwise_contact_graph(
    bodies: Sequence[GaussianCollisionBody],
    states: Sequence[object],
    *,
    names: Sequence[str] | None = None,
    dynamic_flags: Sequence[bool] | None = None,
    candidate_pairs: Iterable[tuple[int, int]] | None = None,
    candidate_pair_mode: str = "all",
    spatial_hash_cell_size: float | None = None,
    collision_config: CollisionEngineConfig | None = None,
    include_inactive: bool = False,
    contact_threshold: float = 0.5,
) -> PairwiseContactGraph:
    """Build a sparse pairwise contact graph for a multi-object scene.

    The graph topology is intentionally Python-side and sparse: every candidate
    pair gets a differentiable `BodyPairContacts` payload, then the edge is kept
    if its broad phase overlaps, its contact gate is active, or
    `include_inactive=True`.
    """

    resolved_names, resolved_dynamic = _validate_scene_inputs(bodies, states, names, dynamic_flags)
    nodes = tuple(
        ContactGraphNode(name=resolved_names[idx], body=body, state=states[idx], dynamic=resolved_dynamic[idx])
        for idx, body in enumerate(bodies)
    )
    config = collision_config or CollisionEngineConfig()
    if candidate_pairs is not None:
        pairs = tuple(candidate_pairs)
    elif candidate_pair_mode == "all":
        pairs = _all_candidate_pairs(len(bodies))
    elif candidate_pair_mode == "spatial_hash":
        if spatial_hash_cell_size is None:
            raise ValueError("spatial_hash_cell_size is required when candidate_pair_mode='spatial_hash'.")
        pairs = spatial_hash_candidate_pairs(
            bodies,
            states,
            cell_size=float(spatial_hash_cell_size),
            margin=float(config.broad_phase_margin),
        )
    else:
        raise ValueError("candidate_pair_mode must be one of: all, spatial_hash.")
    engine = DifferentiableCollisionEngine(config)
    edges = []
    for body_i, body_j in pairs:
        if body_i == body_j:
            raise ValueError("candidate_pairs must not contain self edges.")
        if body_i < 0 or body_j < 0 or body_i >= len(bodies) or body_j >= len(bodies):
            raise ValueError(f"candidate pair {(body_i, body_j)} is out of range for {len(bodies)} bodies.")
        if body_i > body_j:
            body_i, body_j = body_j, body_i

        state_i = states[body_i]
        state_j = states[body_j]
        contacts = engine.body_pair_contacts(
            bodies[body_i],
            state_i.position,
            bodies[body_j],
            state_j.position,
            quaternion_a_wxyz=state_i.quaternion_wxyz,
            quaternion_b_wxyz=state_j.quaternion_wxyz,
        )
        edge = PairwiseContactGraphEdge(body_i=body_i, body_j=body_j, contacts=contacts)
        keep_edge = include_inactive or bool(edge.broad_phase_overlap.detach().cpu().item())
        keep_edge = keep_edge or bool((edge.max_patch_weight > float(contact_threshold)).detach().cpu().item())
        if keep_edge:
            edges.append(edge)
    return PairwiseContactGraph(nodes=nodes, edges=tuple(edges))
