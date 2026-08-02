"""URDF/MJCF adapters for differentiable articulated Gaussian kinematics."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from .articulated_kinematics import (
    ArticulatedLink,
    forward_kinematics,
    link_velocities_from_poses,
)


def _floats(text: str | None, count: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return default
    values = tuple(float(value) for value in text.replace(",", " ").split())
    if len(values) != count:
        raise ValueError(f"Expected {count} values, got {len(values)} in {text!r}.")
    return values


def _rpy_quaternion(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (0.5 * value for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


@dataclass(frozen=True)
class LoadedArticulatedModel:
    links: list[ArticulatedLink]
    joint_names: list[str]
    source_path: Path
    source_format: str

    @property
    def movable_joint_names(self) -> list[str]:
        return [
            name for name, link in zip(self.joint_names, self.links)
            if link.joint_type != "fixed"
        ]


def load_urdf(path: Path) -> LoadedArticulatedModel:
    root = ET.parse(path).getroot()
    link_names = [node.attrib["name"] for node in root.findall("link")]
    joints_by_child = {}
    children_by_parent: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        joints_by_child[child] = joint
        children_by_parent.setdefault(parent, []).append(child)
    roots = [name for name in link_names if name not in joints_by_child]
    if len(roots) != 1:
        raise ValueError(f"URDF must have one root link, found {roots}.")
    ordered_names: list[str] = []

    def visit(name: str) -> None:
        ordered_names.append(name)
        for child in children_by_parent.get(name, []):
            visit(child)

    visit(roots[0])
    indices = {name: index for index, name in enumerate(ordered_names)}
    links, joint_names = [], []
    for name in ordered_names:
        joint = joints_by_child.get(name)
        if joint is None:
            links.append(ArticulatedLink(name, -1, "fixed", (0, 0, 1), (0, 0, 0), (1, 0, 0, 0)))
            joint_names.append(f"{name}__root")
            continue
        joint_type = joint.attrib.get("type", "fixed")
        if joint_type == "continuous":
            joint_type = "revolute"
        if joint_type not in ("fixed", "revolute", "prismatic"):
            raise ValueError(f"Unsupported URDF joint type {joint_type!r}.")
        origin = joint.find("origin")
        position = _floats(None if origin is None else origin.attrib.get("xyz"), 3, (0, 0, 0))
        quaternion = _rpy_quaternion(
            _floats(None if origin is None else origin.attrib.get("rpy"), 3, (0, 0, 0))
        )
        axis_node = joint.find("axis")
        axis = _floats(None if axis_node is None else axis_node.attrib.get("xyz"), 3, (1, 0, 0))
        parent_name = joint.find("parent").attrib["link"]
        links.append(ArticulatedLink(
            name, indices[parent_name], joint_type, axis, position, quaternion
        ))
        joint_names.append(joint.attrib["name"])
    return LoadedArticulatedModel(links, joint_names, path.resolve(), "urdf")


def load_mjcf(path: Path) -> LoadedArticulatedModel:
    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>.")
    links: list[ArticulatedLink] = []
    joint_names: list[str] = []

    def visit(body: ET.Element, parent: int) -> None:
        index = len(links)
        name = body.attrib.get("name", f"body_{index}")
        joints = body.findall("joint")
        if len(joints) > 1:
            raise ValueError(f"MJCF body {name!r} has multiple joints; split it into one body per joint.")
        joint = joints[0] if joints else None
        mj_type = "fixed" if joint is None else joint.attrib.get("type", "hinge")
        joint_type = {"hinge": "revolute", "slide": "prismatic", "fixed": "fixed"}.get(mj_type)
        if joint_type is None:
            raise ValueError(f"Unsupported MJCF joint type {mj_type!r}.")
        position = _floats(body.attrib.get("pos"), 3, (0, 0, 0))
        if "quat" in body.attrib:
            quaternion = _floats(body.attrib["quat"], 4, (1, 0, 0, 0))
        else:
            quaternion = _rpy_quaternion(_floats(body.attrib.get("euler"), 3, (0, 0, 0)))
        axis = _floats(None if joint is None else joint.attrib.get("axis"), 3, (0, 0, 1))
        pivot = _floats(None if joint is None else joint.attrib.get("pos"), 3, (0, 0, 0))
        links.append(ArticulatedLink(
            name, parent, joint_type, axis, position, quaternion, pivot
        ))
        joint_names.append(
            f"{name}__fixed" if joint is None else joint.attrib.get("name", f"{name}__joint")
        )
        for child in body.findall("body"):
            visit(child, index)

    for body in worldbody.findall("body"):
        visit(body, -1)
    return LoadedArticulatedModel(links, joint_names, path.resolve(), "mjcf")


def load_articulated_model(path: Path) -> LoadedArticulatedModel:
    suffix = path.suffix.lower()
    if suffix == ".urdf":
        return load_urdf(path)
    if suffix in (".xml", ".mjcf"):
        return load_mjcf(path)
    raise ValueError(f"Expected .urdf, .xml, or .mjcf, got {path}.")


def load_joint_trajectory(
    model: LoadedArticulatedModel,
    path: Path,
    *,
    dt: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    states = payload.get("states")
    names = payload.get("joint_names", model.movable_joint_names)
    if states is not None:
        rows = [state.get("joint_positions", state.get("qpos")) for state in states]
        times = [float(state.get("time", index)) for index, state in enumerate(states)]
        base_positions = [state.get("base_position", [0, 0, 0]) for state in states]
        base_quaternions = [state.get("base_quaternion_wxyz", [1, 0, 0, 0]) for state in states]
    else:
        rows = payload["joint_positions"]
        times = payload.get("times", list(range(len(rows))))
        base_positions = payload.get("base_positions", [[0, 0, 0]] * len(rows))
        base_quaternions = payload.get("base_quaternions_wxyz", [[1, 0, 0, 0]] * len(rows))
    if dt is None:
        differences = [b - a for a, b in zip(times[:-1], times[1:])]
        dt = sorted(differences)[len(differences) // 2]
    name_to_column = {name: index for index, name in enumerate(names)}
    full = torch.zeros((len(rows), len(model.links)), dtype=dtype, device=device)
    for link_index, (joint_name, link) in enumerate(zip(model.joint_names, model.links)):
        if link.joint_type != "fixed":
            if joint_name not in name_to_column:
                raise ValueError(f"Trajectory is missing joint {joint_name!r}.")
            full[:, link_index] = torch.tensor(
                [row[name_to_column[joint_name]] for row in rows], dtype=dtype, device=device
            )
    positions, quaternions = forward_kinematics(
        model.links,
        full,
        base_position=torch.tensor(base_positions, dtype=dtype, device=device),
        base_quaternion_wxyz=torch.tensor(base_quaternions, dtype=dtype, device=device),
    )
    linear, angular = link_velocities_from_poses(positions, quaternions, float(dt))
    return {
        "joint_positions": full,
        "positions": positions,
        "quaternions": quaternions,
        "linear_velocities": linear,
        "angular_velocities": angular,
        "dt": float(dt),
    }
