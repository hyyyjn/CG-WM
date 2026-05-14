"""Smoke test for Stage 2: sphere falling onto a floor.

This script does not rely on any Stage 1 PLY or MuJoCo episode on disk. It
synthesises a sphere (either as a single analytic primitive or as a tiny
"Gaussian cluster"), drops it under gravity, and reports trajectories +
contact diagnostics so we can eyeball whether the contact integrator behaves.

It also runs three sanity checks comparing the differentiable collision SDF
against analytic ground truth (hard min over spheres).

Outputs:
  - tools/_stage2_smoke_out/sphere_plane.png    matplotlib trajectory
  - tools/_stage2_smoke_out/sphere_gaussian.png matplotlib trajectory
  - tools/_stage2_smoke_out/smoke_summary.json  numbers (RMSE vs analytic, etc.)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (
    PlaneCollider,
    detect_gaussian_union_contacts,
    make_floor_disk_query_points,
    make_sphere_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (
    ComplementarityFreeContactDynamics,
    ContactDynamicsConfig,
    GaussianUnionFloorContactDynamics,
    ImpedanceContactDynamicsConfig,
    ImpedanceFloorContactDynamics,
    RigidState,
    rollout,
)

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "smoke"


def analytic_ball_drop(z0: float, vz0: float, g: float, e: float, r: float, dt: float, steps: int):
    """Closed-form bouncing-ball reference (z and vz)."""
    z = z0
    vz = vz0
    floor = r  # ball center sits at r when resting on z=0 floor
    traj_z = [z]
    for _ in range(steps - 1):
        vz = vz + g * dt
        z = z + vz * dt
        if z < floor:
            # reflect using restitution e, set position to floor + tiny overshoot mirror
            z = floor + (floor - z)
            vz = -e * vz
            if abs(vz) < 0.05 and abs(z - floor) < 5e-3:
                vz = 0.0
                z = floor
        traj_z.append(z)
    return np.array(traj_z, dtype=np.float64)


def sanity_check_gaussian_union_sdf():
    """Compare softmax-weighted-distance SDF against hard min over primitives.

    Paper specifies LogSumExp; current code uses softmax-weighted average. We
    verify how closely it tracks hard-min outside the geometry.
    """
    torch.manual_seed(0)
    device = "cpu"
    # cluster of 7 spheres approximating a ball of radius ~0.10
    centers_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, -0.05, 0.0],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, -0.05],
        ],
        dtype=np.float32,
    )
    radii_np = np.full((centers_np.shape[0],), 0.05, dtype=np.float32)
    centers = torch.tensor(centers_np, device=device)
    radii = torch.tensor(radii_np, device=device)

    # query a 3D lattice around the ball
    qs = np.stack(np.meshgrid(
        np.linspace(-0.3, 0.3, 13),
        np.linspace(-0.3, 0.3, 13),
        np.linspace(-0.3, 0.3, 13),
        indexing="ij",
    ), axis=-1).reshape(-1, 3).astype(np.float32)
    q = torch.tensor(qs, device=device)

    # hard min (paper's exact phi_Ggeo)
    diff = q[:, None, :] - centers[None, :, :]
    primitive = torch.linalg.norm(diff, dim=-1) - radii[None, :]
    hard = primitive.min(dim=-1).values

    # current code path (softmax-weighted average)
    contacts = detect_gaussian_union_contacts(
        q, centers, radii,
        collider_normal=torch.tensor([0.0, 0.0, 1.0]),
        softness=1e-3,
        smooth_min_temperature=2e-2,
    )
    soft = contacts.signed_distances

    # only compare outside the geometry (paper's sigmoid penalty handles inside)
    outside_mask = hard > 0.01
    diff_outside = (soft[outside_mask] - hard[outside_mask]).abs()

    return {
        "num_query_points": int(qs.shape[0]),
        "hard_min_range": [float(hard.min().item()), float(hard.max().item())],
        "soft_outside_mae": float(diff_outside.mean().item()),
        "soft_outside_max_err": float(diff_outside.max().item()),
        "soft_inside_min": float(soft[~outside_mask].min().item()) if (~outside_mask).any() else None,
        "hard_inside_min": float(hard[~outside_mask].min().item()) if (~outside_mask).any() else None,
    }


def run_plane_contact_drop():
    radius = 0.10
    z0 = 1.0
    dt = 1.0 / 120.0
    steps = 300
    restitution = 0.5
    g = -9.81

    collider = PlaneCollider.floor()
    local_query = make_sphere_query_points(radius, num_points=64, include_center=False)

    cfg = ContactDynamicsConfig(
        dt=dt,
        mass=1.0,
        restitution=restitution,
        acceleration=(0.0, 0.0, g),
        contact_softness=1e-3,
        contact_gate_softness=2e-3,
        smooth_max_temperature=1e-2,
        position_slop=1e-4,
    )
    dynamics = ComplementarityFreeContactDynamics(collider, local_query, cfg)
    state0 = RigidState(
        position=torch.tensor([0.0, 0.0, z0]),
        linear_velocity=torch.tensor([0.0, 0.0, 0.0]),
    )
    states, diagnostics = rollout(state0, dynamics, num_steps=steps - 1)

    z_pred = np.array([float(s.position[2].item()) for s in states])
    vz_pred = np.array([float(s.linear_velocity[2].item()) for s in states])
    z_ref = analytic_ball_drop(z0, 0.0, g, restitution, radius, dt, steps)

    contact_gates = np.array([float(d["contact_gate"].item()) for d in diagnostics])
    max_pen = np.array([float(d["max_penetration"].item()) for d in diagnostics])
    rmse = float(np.sqrt(np.mean((z_pred - z_ref) ** 2)))
    first_contact = int(np.argmax(contact_gates > 0.5)) if (contact_gates > 0.5).any() else -1
    settled_z = float(z_pred[-1])

    return {
        "z_pred": z_pred,
        "z_ref": z_ref,
        "vz_pred": vz_pred,
        "contact_gates": contact_gates,
        "max_pen": max_pen,
        "rmse": rmse,
        "first_contact_frame": first_contact,
        "settled_z": settled_z,
        "settled_target": radius,
        "dt": dt,
        "steps": steps,
        "restitution": restitution,
    }


def run_gaussian_union_drop():
    radius = 0.10
    z0 = 1.0
    dt = 1.0 / 120.0
    steps = 300
    restitution = 0.5
    g = -9.81

    # Sphere as a single Gaussian primitive at the body origin
    local_centers = torch.zeros((1, 3))
    radii = torch.tensor([radius])
    floor_offsets = make_floor_disk_query_points(radius * 1.1, num_rings=5, num_angles=24)
    collider = PlaneCollider.floor()

    cfg = ContactDynamicsConfig(
        dt=dt,
        mass=1.0,
        restitution=restitution,
        acceleration=(0.0, 0.0, g),
        contact_softness=1e-3,
        contact_gate_softness=2e-3,
        smooth_max_temperature=1e-2,
        position_slop=1e-4,
    )
    dynamics = GaussianUnionFloorContactDynamics(
        collider=collider,
        floor_query_offsets_xy=floor_offsets,
        local_gaussian_centers=local_centers,
        gaussian_radii=radii,
        config=cfg,
    )
    state0 = RigidState(
        position=torch.tensor([0.0, 0.0, z0]),
        linear_velocity=torch.tensor([0.0, 0.0, 0.0]),
    )
    states, diagnostics = rollout(state0, dynamics, num_steps=steps - 1)

    z_pred = np.array([float(s.position[2].item()) for s in states])
    vz_pred = np.array([float(s.linear_velocity[2].item()) for s in states])
    z_ref = analytic_ball_drop(z0, 0.0, g, restitution, radius, dt, steps)

    contact_gates = np.array([float(d["contact_gate"].item()) for d in diagnostics])
    max_pen = np.array([float(d["max_penetration"].item()) for d in diagnostics])
    rmse = float(np.sqrt(np.mean((z_pred - z_ref) ** 2)))
    first_contact = int(np.argmax(contact_gates > 0.5)) if (contact_gates > 0.5).any() else -1
    settled_z = float(z_pred[-1])
    return {
        "z_pred": z_pred,
        "z_ref": z_ref,
        "vz_pred": vz_pred,
        "contact_gates": contact_gates,
        "max_pen": max_pen,
        "rmse": rmse,
        "first_contact_frame": first_contact,
        "settled_z": settled_z,
        "settled_target": radius,
        "dt": dt,
        "steps": steps,
        "restitution": restitution,
    }


def run_impedance_drop():
    """Paper-style impedance dynamics with K, D (no restitution coefficient).

    Stiffness K and damping D are fixed here to plausible values; the fit-loop
    smoke test learns them from a trajectory instead.
    """
    radius = 0.10
    z0 = 1.0
    dt = 1.0 / 120.0
    steps = 300
    g = -9.81

    local_centers = torch.zeros((1, 3))
    radii = torch.tensor([radius])
    floor_offsets = make_floor_disk_query_points(radius * 1.1, num_rings=5, num_angles=24)
    collider = PlaneCollider.floor()

    # ImpedanceFloorContactDynamics applies SoftPlus internally; for raw>20 the
    # output is essentially identical to the raw value, so we pass targets in
    # directly. K=1000 gives a ~1 cm equilibrium overlap, D=30 mildly damps.
    stiffness = torch.tensor(1000.0)
    damping = torch.tensor(30.0)

    cfg = ImpedanceContactDynamicsConfig(
        dt=dt,
        mass=1.0,
        gravity=(0.0, 0.0, g),
        contact_softness=1e-3,
        smooth_min_temperature=1e-2,
        inside_penalty=0.02,
        inside_sharpness=50.0,
    )
    dynamics = ImpedanceFloorContactDynamics(
        collider=collider,
        floor_query_offsets_xy=floor_offsets,
        local_gaussian_centers=local_centers,
        gaussian_radii=radii,
        stiffness=stiffness,
        damping=damping,
        config=cfg,
    )
    state0 = RigidState(
        position=torch.tensor([0.0, 0.0, z0]),
        linear_velocity=torch.tensor([0.0, 0.0, 0.0]),
    )
    states, diagnostics = rollout(state0, dynamics, num_steps=steps - 1)

    z_pred = np.array([float(s.position[2].item()) for s in states])
    vz_pred = np.array([float(s.linear_velocity[2].item()) for s in states])
    # Use restitution=0.4 as a rough analytic baseline; impedance damping makes
    # the effective coefficient depend on K/D and isn't directly comparable.
    z_ref = analytic_ball_drop(z0, 0.0, g, 0.4, radius, dt, steps)

    lambdas = np.array([float(d["lambda"].item()) for d in diagnostics])
    phi_agg = np.array([float(d["phi_agg"].item()) for d in diagnostics])
    max_pen = np.array([float(d["max_penetration"].item()) for d in diagnostics])
    rmse = float(np.sqrt(np.mean((z_pred - z_ref) ** 2)))
    first_contact = int(np.argmax(lambdas > 0.5)) if (lambdas > 0.5).any() else -1
    settled_z = float(z_pred[-1])
    return {
        "z_pred": z_pred,
        "z_ref": z_ref,
        "vz_pred": vz_pred,
        "contact_gates": lambdas,
        "max_pen": max_pen,
        "phi_agg": phi_agg,
        "rmse": rmse,
        "first_contact_frame": first_contact,
        "settled_z": settled_z,
        "settled_target": radius,
        "dt": dt,
        "steps": steps,
        "stiffness": 8000.0,
        "damping": 60.0,
    }


def plot_trajectory(result, title, path):
    """Plot trajectory using PIL (no matplotlib dep in conda gs)."""
    from PIL import Image, ImageDraw

    z_pred = result["z_pred"]
    z_ref = result["z_ref"]
    vz_pred = result["vz_pred"]
    gates = result["contact_gates"]
    pens = result["max_pen"]
    dt = result["dt"]
    n = z_pred.shape[0]
    t = np.arange(n) * dt

    width, height = 920, 720
    margin_l = 78
    margin_r = 30
    margin_t = 70
    margin_b = 40
    panel_h = (height - margin_t - margin_b - 40) // 3
    panel_gap = 20

    img = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 48), fill=(28, 28, 30))
    draw.text((14, 14), title, fill=(255, 255, 245))
    draw.text(
        (14, 30),
        f"rmse={result['rmse']:.4f} m  settled_z={result['settled_z']:.3f} (target {result['settled_target']:.2f})"
        f"  first_contact_frame={result['first_contact_frame']}",
        fill=(220, 220, 220),
    )

    def panel(idx, title_text, series, colors, y_axis_label, y_range=None):
        y_top = margin_t + idx * (panel_h + panel_gap)
        y_bot = y_top + panel_h
        # background
        draw.rectangle((margin_l, y_top, width - margin_r, y_bot), fill=(255, 255, 255), outline=(160, 160, 160))
        draw.text((margin_l + 4, y_top + 4), title_text, fill=(40, 40, 40))
        draw.text((6, y_top + panel_h // 2 - 6), y_axis_label, fill=(60, 60, 60))

        if y_range is None:
            all_values = np.concatenate(series)
            ymin = float(all_values.min())
            ymax = float(all_values.max())
        else:
            ymin, ymax = y_range
        if ymax - ymin < 1e-6:
            ymax = ymin + 1.0
        span = ymax - ymin

        # zero baseline if visible
        if ymin <= 0.0 <= ymax:
            zy = int(y_bot - (0.0 - ymin) / span * (y_bot - y_top))
            draw.line((margin_l, zy, width - margin_r, zy), fill=(210, 210, 210), width=1)

        # x-axis ticks
        for k in range(5):
            tx = margin_l + int(k / 4 * (width - margin_l - margin_r))
            draw.line((tx, y_bot, tx, y_bot + 4), fill=(120, 120, 120))
            draw.text((tx - 14, y_bot + 6), f"{t[-1] * k / 4:.2f}s", fill=(80, 80, 80))

        # y-axis labels
        for k in range(4):
            yv = ymin + k / 3 * span
            yy = int(y_bot - (yv - ymin) / span * (y_bot - y_top))
            draw.line((margin_l - 4, yy, margin_l, yy), fill=(120, 120, 120))
            draw.text((margin_l - 70, yy - 6), f"{yv:+.3f}", fill=(80, 80, 80))

        for color, vals, label in zip(colors, series, ["pred", "ref", "extra"][: len(series)]):
            n_v = vals.shape[0]
            pts = []
            for i in range(n_v):
                px = margin_l + int(i / max(n - 1, 1) * (width - margin_l - margin_r))
                py = int(y_bot - (vals[i] - ymin) / span * (y_bot - y_top))
                pts.append((px, py))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)

    panel(
        0,
        "z (m): solid = stage2 pred, dashed = analytic bounce reference",
        [z_pred, z_ref],
        [(34, 87, 184), (200, 70, 40)],
        "z[m]",
    )
    # add dashed effect for ref by overdrawing white gaps
    # (skipped to keep PIL plot simple; legend in title is enough)

    panel(
        1,
        "vz (m/s)",
        [vz_pred],
        [(220, 130, 30)],
        "vz",
    )

    panel(
        2,
        "contact gate (red) & max penetration (purple)",
        [np.concatenate(([0.0], gates)), np.concatenate(([0.0], pens))],
        [(200, 50, 50), (130, 60, 180)],
        "",
    )

    img.save(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    sdf = sanity_check_gaussian_union_sdf()
    summary["gaussian_union_sdf_vs_hard_min"] = sdf

    plane = run_plane_contact_drop()
    summary["plane_contact_drop"] = {
        "rmse_vs_analytic": plane["rmse"],
        "first_contact_frame": plane["first_contact_frame"],
        "settled_z": plane["settled_z"],
        "settled_target": plane["settled_target"],
        "max_penetration_overall": float(plane["max_pen"].max()),
    }
    plot_trajectory(plane, "ComplementarityFreeContactDynamics (plane SDF + sphere query)", OUT_DIR / "sphere_plane.png")

    gu = run_gaussian_union_drop()
    summary["gaussian_union_drop"] = {
        "rmse_vs_analytic": gu["rmse"],
        "first_contact_frame": gu["first_contact_frame"],
        "settled_z": gu["settled_z"],
        "settled_target": gu["settled_target"],
        "max_penetration_overall": float(gu["max_pen"].max()),
    }
    plot_trajectory(gu, "GaussianUnionFloorContactDynamics (floor query + Gaussian SDF)", OUT_DIR / "sphere_gaussian.png")

    imp = run_impedance_drop()
    summary["impedance_drop"] = {
        "rmse_vs_analytic_e0p4": imp["rmse"],
        "first_contact_frame": imp["first_contact_frame"],
        "settled_z": imp["settled_z"],
        "settled_target": imp["settled_target"],
        "max_penetration_overall": float(imp["max_pen"].max()),
        "max_phi_negative": float((-imp["phi_agg"]).max()),
        "max_lambda": float(imp["contact_gates"].max()),
        "stiffness_softplus_in": imp["stiffness"],
        "damping_softplus_in": imp["damping"],
        "trajectory_nan": bool(np.isnan(imp["z_pred"]).any()),
    }
    plot_trajectory(imp, "ImpedanceFloorContactDynamics (paper SoftPlus K, D)", OUT_DIR / "sphere_impedance.png")

    with open(OUT_DIR / "smoke_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
