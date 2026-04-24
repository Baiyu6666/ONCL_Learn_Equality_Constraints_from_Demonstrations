from __future__ import annotations

import time
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from torch import nn

from evaluation.evaluator import DEFAULT_EVAL_CFG, eval_bounds_from_train
from datasets.constraint_datasets import generate_dataset
from datasets.ur5_pybullet_utils import (
    UR5_LINK_LENGTHS,
    UR5PyBulletKinematics,
    _make_pybullet_friendly_urdf,
    pick_default_ee_link_index,
    resolve_ur5_kinematics_cfg,
    resolve_ur5_render_cfg,
)
from models.kinematics import (
    is_arm_dataset,
    planar_fk,
    planar_fk as _planar_fk,
    spatial_fk,
    spatial_fk as _spatial_fk,
    spatial_tool_axis_n6,
    wrap_np_pi as _wrap_np_pi,
)
from models.projection import (
    project_points_tensor,
)

ZERO_EPS_QUANTILE_DEFAULT = float(DEFAULT_EVAL_CFG["zero_eps_quantile"])
_UR5_PLANNER_CHECKER: dict[str, Any] | None = None


def resolve_periodic_mode(
    *,
    periodic_joint: bool | None = None,
    dataset_name: str | None = None,
) -> bool:
    if periodic_joint is not None:
        return bool(periodic_joint)
    if dataset_name is None:
        return False
    return "arm_" in str(dataset_name)


def _wrap_torch_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + np.pi, 2.0 * np.pi) - np.pi


def _angle_delta_torch(a: torch.Tensor, b: torch.Tensor, periodic: bool) -> torch.Tensor:
    d = b - a
    return _wrap_torch_pi(d) if periodic else d


def _angle_delta_np(a: np.ndarray, b: np.ndarray, periodic: bool) -> np.ndarray:
    d = b - a
    if not periodic:
        return d.astype(np.float32)
    return ((d + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _interp_linear_np(a: np.ndarray, b: np.ndarray, s: np.ndarray, periodic: bool) -> np.ndarray:
    d = _angle_delta_np(a, b, periodic=periodic)
    x = a + s * d
    if periodic:
        x = ((x + np.pi) % (2.0 * np.pi) - np.pi)
    return x.astype(np.float32)


def build_linear_path(
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    n_waypoints: int,
    periodic: bool,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32).reshape(-1, 1)
    path = _interp_linear_np(
        x_start.reshape(1, -1).astype(np.float32),
        x_goal.reshape(1, -1).astype(np.float32),
        t,
        periodic=periodic,
    )
    path[0] = x_start.astype(np.float32)
    path[-1] = x_goal.astype(np.float32)
    return path


def _build_piecewise_linear_path(
    x_start: np.ndarray,
    x_mid: np.ndarray,
    x_goal: np.ndarray,
    *,
    n_waypoints: int,
    periodic: bool,
) -> np.ndarray:
    n = int(max(3, n_waypoints))
    n1 = max(2, n // 2 + 1)
    n2 = max(2, n - n1 + 1)
    p1 = build_linear_path(x_start, x_mid, n_waypoints=n1, periodic=periodic)
    p2 = build_linear_path(x_mid, x_goal, n_waypoints=n2, periodic=periodic)
    out = np.concatenate([p1[:-1], p2], axis=0).astype(np.float32)
    if len(out) != n:
        idx = np.linspace(0, len(out) - 1, n).astype(np.int32)
        out = out[idx].astype(np.float32)
    out[0] = x_start.astype(np.float32)
    out[-1] = x_goal.astype(np.float32)
    return out


def _build_obstacle_arc_init_path(
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    n_waypoints: int,
    periodic: bool,
    obstacle_center_xy: tuple[float, float],
    obstacle_radius: float,
    obstacle_margin: float,
    side_sign: float,
) -> np.ndarray:
    start = np.asarray(x_start, dtype=np.float32).copy()
    goal = np.asarray(x_goal, dtype=np.float32).copy()
    mid = 0.5 * (start + goal)
    s_xy = start[:2].astype(np.float32)
    g_xy = goal[:2].astype(np.float32)
    c_xy = np.asarray(obstacle_center_xy, dtype=np.float32).reshape(2)
    v = g_xy - s_xy
    vn = float(np.linalg.norm(v))
    if vn < 1e-8:
        v = np.asarray([1.0, 0.0], dtype=np.float32)
        vn = 1.0
    v = (v / vn).astype(np.float32)
    perp = np.asarray([-v[1], v[0]], dtype=np.float32)
    clearance = float(obstacle_radius) + float(obstacle_margin) + 0.04 * max(float(obstacle_radius), 1.0)
    base_mid_xy = 0.5 * (s_xy + g_xy)
    offset_xy = c_xy + float(side_sign) * clearance * perp
    # Bias the intermediate waypoint away from the obstacle while staying between start and goal.
    mid[:2] = (0.35 * base_mid_xy + 0.65 * offset_xy).astype(np.float32)
    return _build_piecewise_linear_path(
        start.astype(np.float32),
        mid.astype(np.float32),
        goal.astype(np.float32),
        n_waypoints=int(n_waypoints),
        periodic=bool(periodic),
    )


def _line_intersects_circle_xy(
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    obstacle_center_xy: tuple[float, float],
    obstacle_radius: float,
    obstacle_margin: float,
) -> bool:
    s = np.asarray(x_start, dtype=np.float32).reshape(-1)[:2]
    g = np.asarray(x_goal, dtype=np.float32).reshape(-1)[:2]
    c = np.asarray(obstacle_center_xy, dtype=np.float32).reshape(2)
    v = g - s
    vv = float(np.dot(v, v))
    if vv < 1e-12:
        return float(np.linalg.norm(s - c)) <= float(obstacle_radius + obstacle_margin)
    t = float(np.dot(c - s, v) / vv)
    t = max(0.0, min(1.0, t))
    foot = s + t * v
    d = float(np.linalg.norm(foot - c))
    return d <= float(obstacle_radius + obstacle_margin)


def _path_xy_quality_score(
    path: np.ndarray,
    *,
    obstacle_center_xy: tuple[float, float] | None = None,
    obstacle_radius: float = 0.0,
    obstacle_margin: float = 0.0,
) -> float:
    p = np.asarray(path, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2:
        return 1e9
    xy = p[:, :2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_len = float(np.sum(step))
    chord_vec = xy[-1] - xy[0]
    chord = float(np.linalg.norm(chord_vec))
    ratio = path_len / max(chord, 1e-8)
    backtrack = 0.0
    if chord > 1e-8:
        direction = chord_vec / chord
        progress = (xy - xy[0:1]) @ direction
        dprog = np.diff(progress.reshape(-1))
        backtrack = float(np.sum(np.maximum(-dprog, 0.0)))
    loopiness = max(0.0, ratio - 1.15)
    score = path_len + 1.8 * backtrack + 0.35 * chord * loopiness
    if obstacle_center_xy is not None and float(obstacle_radius) > 0.0:
        c_xy = np.asarray(obstacle_center_xy, dtype=np.float32).reshape(1, 2)
        dist = np.linalg.norm(xy - c_xy, axis=1)
        penetration = np.maximum(float(obstacle_radius) + float(obstacle_margin) - dist, 0.0)
        score += 50.0 * float(np.sum(penetration ** 2))
    return float(score)


def _build_perturbed_linear_init_paths(
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    n_waypoints: int,
    periodic: bool,
    obstacle_center_xy: tuple[float, float] | None = None,
    obstacle_radius: float = 0.0,
    obstacle_margin: float = 0.0,
) -> list[np.ndarray]:
    start = np.asarray(x_start, dtype=np.float32).reshape(-1)
    goal = np.asarray(x_goal, dtype=np.float32).reshape(-1)
    out = [build_linear_path(start, goal, n_waypoints=int(n_waypoints), periodic=bool(periodic)).astype(np.float32)]
    if start.shape[0] < 2:
        return out
    s_xy = start[:2].astype(np.float32)
    g_xy = goal[:2].astype(np.float32)
    chord = g_xy - s_xy
    chord_n = float(np.linalg.norm(chord))
    if chord_n < 1e-8:
        return out
    direction = chord / chord_n
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    base_mid = 0.5 * (start + goal)
    base_offset = max(0.06, 0.18 * chord_n)
    if obstacle_center_xy is not None and float(obstacle_radius) > 0.0:
        base_offset = max(base_offset, float(obstacle_radius) + float(obstacle_margin) + 0.03)
    t = np.linspace(0.0, 1.0, int(max(3, n_waypoints)), dtype=np.float32).reshape(-1, 1)
    envelope = np.sin(np.pi * t).astype(np.float32)
    line = out[0].copy()
    for sign in (-1.0, 1.0):
        for amp_scale in (0.8, 1.25):
            cand = line.copy()
            cand[:, :2] = (cand[:, :2] + float(sign * amp_scale * base_offset) * envelope * normal.reshape(1, 2)).astype(np.float32)
            cand[0] = start.astype(np.float32)
            cand[-1] = goal.astype(np.float32)
            out.append(cand.astype(np.float32))
        mid = base_mid.copy()
        mid[:2] = (base_mid[:2] + float(sign) * base_offset * normal).astype(np.float32)
        out.append(
            _build_piecewise_linear_path(
                start.astype(np.float32),
                mid.astype(np.float32),
                goal.astype(np.float32),
                n_waypoints=int(n_waypoints),
                periodic=bool(periodic),
            ).astype(np.float32)
        )
    return out


def pick_far_pair_workspace_planar(
    x: np.ndarray,
    lengths: list[float],
    min_ratio: float,
    max_ratio: float,
    tries: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    reach = float(sum(lengths))
    lo = float(min_ratio) * max(reach, 1e-6)
    hi = float(max_ratio) * max(reach, 1e-6)
    if hi < lo:
        hi = lo
    target = 0.5 * (lo + hi)
    best = None
    best_delta = 1e18

    q = x.astype(np.float32)
    ee = _planar_fk(q, lengths)[:, -1, :]
    n = len(q)
    for _ in range(max(1, int(tries))):
        i = int(np.random.randint(0, n))
        j = int(np.random.randint(0, n))
        if i == j:
            continue
        d = float(np.linalg.norm(ee[i] - ee[j]))
        if lo <= d <= hi:
            return q[i], q[j], d, reach
        delta = abs(d - target)
        if delta < best_delta:
            best_delta = delta
            best = (q[i], q[j], d)
    assert best is not None
    return best[0], best[1], float(best[2]), reach


def _smoothstep(s: np.ndarray) -> np.ndarray:
    return (s * s * (3.0 - 2.0 * s)).astype(np.float32)


def _angle_shortest_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _wrap_np_pi(b - a)


def _angle_interp_shortest(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    d = _angle_shortest_delta(a, b)
    return _wrap_np_pi(a + s * d)


def init_path_joint_spline(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    n_waypoints: int,
    mid_noise: float,
) -> np.ndarray:
    q_start = q_start.astype(np.float32)
    q_goal = q_goal.astype(np.float32)
    d = _angle_shortest_delta(q_start, q_goal)
    q_mid = _wrap_np_pi(
        q_start
        + 0.5 * d
        + float(mid_noise) * np.random.randn(*q_start.shape).astype(np.float32)
    )

    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32)
    q = np.zeros((int(n_waypoints), q_start.shape[0]), dtype=np.float32)
    left = t <= 0.5
    right = ~left
    if np.any(left):
        s = _smoothstep((t[left] / 0.5).reshape(-1, 1))
        q[left] = _angle_interp_shortest(q_start.reshape(1, -1), q_mid.reshape(1, -1), s)
    if np.any(right):
        s = _smoothstep(((t[right] - 0.5) / 0.5).reshape(-1, 1))
        q[right] = _angle_interp_shortest(q_mid.reshape(1, -1), q_goal.reshape(1, -1), s)
    q[0] = q_start
    q[-1] = q_goal
    return _wrap_np_pi(q)


def _fk_ee_torch(q: torch.Tensor, lengths: list[float], is_spatial: bool) -> torch.Tensor:
    if not is_spatial:
        ang = torch.cumsum(q, dim=1)
        x = torch.zeros((q.shape[0],), dtype=q.dtype, device=q.device)
        y = torch.zeros((q.shape[0],), dtype=q.dtype, device=q.device)
        for k in range(q.shape[1]):
            lk = float(lengths[k])
            x = x + lk * torch.cos(ang[:, k])
            y = y + lk * torch.sin(ang[:, k])
        return torch.stack([x, y], dim=1)

    if q.shape[1] == 3:
        l1, l2 = float(lengths[0]), float(lengths[1])
        q1 = q[:, 0]
        t2 = q[:, 1]
        t3 = q[:, 1] + q[:, 2]
        r1 = l1 * torch.cos(t2)
        z1 = l1 * torch.sin(t2)
        r2 = r1 + l2 * torch.cos(t3)
        z2 = z1 + l2 * torch.sin(t3)
        x = r2 * torch.cos(q1)
        y = r2 * torch.sin(q1)
        return torch.stack([x, y, z2], dim=1)

    if q.shape[1] == 4:
        l1, l2, l3 = float(lengths[0]), float(lengths[1]), float(lengths[2])
        q1 = q[:, 0]
        t2 = q[:, 1]
        t3 = q[:, 1] + q[:, 2]
        t4 = q[:, 1] + q[:, 2] + q[:, 3]
        r1 = l1 * torch.cos(t2)
        z1 = l1 * torch.sin(t2)
        r2 = r1 + l2 * torch.cos(t3)
        z2 = z1 + l2 * torch.sin(t3)
        r3 = r2 + l3 * torch.cos(t4)
        z3 = z2 + l3 * torch.sin(t4)
        x = r3 * torch.cos(q1)
        y = r3 * torch.sin(q1)
        return torch.stack([x, y, z3], dim=1)

    if q.shape[1] == 6:
        a1 = q[:, 0]
        a2 = q[:, 1]
        a3 = q[:, 1] + q[:, 2]
        a4 = q[:, 1] + q[:, 2] + q[:, 3]
        r1 = float(lengths[0]) * torch.cos(a2)
        z1 = float(lengths[0]) * torch.sin(a2)
        r2 = r1 + float(lengths[1]) * torch.cos(a3)
        z2 = z1 + float(lengths[1]) * torch.sin(a3)
        r3 = r2 + float(lengths[2]) * torch.cos(a4)
        z3 = z2 + float(lengths[2]) * torch.sin(a4)
        r4 = r3 + float(lengths[3]) * torch.cos(a4)
        z4 = z3 + float(lengths[3]) * torch.sin(a4)
        r5 = r4 + float(lengths[4]) * torch.cos(a4 + q[:, 4])
        z5 = z4 + float(lengths[4]) * torch.sin(a4 + q[:, 4])
        r6 = r5 + float(lengths[5]) * torch.cos(a4 + q[:, 4])
        z6 = z5 + float(lengths[5]) * torch.sin(a4 + q[:, 4])
        x = r6 * torch.cos(a1)
        y = r6 * torch.sin(a1)
        return torch.stack([x, y, z6], dim=1)

    raise ValueError(f"unsupported spatial dof={q.shape[1]}")


def init_path_via_workspace_ik(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    lengths: list[float],
    is_spatial: bool,
    use_pybullet_n6: bool,
    n_waypoints: int,
    device: str,
) -> np.ndarray:
    if is_spatial:
        ee_s = _spatial_fk(q_start.reshape(1, -1), lengths, use_pybullet_n6=use_pybullet_n6)[0, -1, :]
        ee_g = _spatial_fk(q_goal.reshape(1, -1), lengths, use_pybullet_n6=use_pybullet_n6)[0, -1, :]
    else:
        ee_s = _planar_fk(q_start.reshape(1, -1), lengths)[0, -1, :]
        ee_g = _planar_fk(q_goal.reshape(1, -1), lengths)[0, -1, :]

    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32).reshape(-1, 1)
    ee_targets = (1.0 - t) * ee_s.reshape(1, -1) + t * ee_g.reshape(1, -1)

    q_path = np.zeros((int(n_waypoints), q_start.shape[0]), dtype=np.float32)
    q_path[0] = q_start.astype(np.float32)
    q_prev = q_start.astype(np.float32).copy()
    for i in range(1, int(n_waypoints) - 1):
        target = torch.tensor(ee_targets[i : i + 1].astype(np.float32), device=device)
        qv = torch.tensor(q_prev.reshape(1, -1), device=device, requires_grad=True)
        opt = torch.optim.Adam([qv], lr=0.03)
        for _ in range(70):
            opt.zero_grad(set_to_none=True)
            ee = _fk_ee_torch(qv, lengths, is_spatial=is_spatial)
            reg = (qv - torch.tensor(q_prev.reshape(1, -1), device=device)).pow(2).mean()
            loss = (ee - target).pow(2).mean() + 0.01 * reg
            loss.backward()
            opt.step()
            with torch.no_grad():
                qv[:] = torch.remainder(qv + np.pi, 2.0 * np.pi) - np.pi
        q_prev = qv.detach().cpu().numpy().reshape(-1).astype(np.float32)
        q_path[i] = q_prev
    q_path[-1] = q_goal.astype(np.float32)
    return _wrap_np_pi(q_path)


def plan_linear_then_project(
    model: nn.Module,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    device: str,
    n_waypoints: int,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int = 0,
    f_abs_stop: float | None = None,
    keep_endpoints: bool = False,
    periodic: bool = False,
) -> np.ndarray:
    path = build_linear_path(
        x_start.astype(np.float32),
        x_goal.astype(np.float32),
        n_waypoints=int(n_waypoints),
        periodic=bool(periodic),
    )
    path_t = torch.from_numpy(path).to(device)
    proj_t, _ = project_points_tensor(
        model,
        path_t,
        proj_steps=int(proj_steps),
        proj_alpha=float(proj_alpha),
        proj_min_steps=int(proj_min_steps),
        f_abs_stop=f_abs_stop,
    )
    out = proj_t.detach().cpu().numpy().astype(np.float32)
    if periodic:
        out = ((out + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    if keep_endpoints:
        out[0] = x_start.astype(np.float32)
        out[-1] = x_goal.astype(np.float32)
    return out


def plan_linear_then_model_project(
    model: nn.Module,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    device: str,
    n_waypoints: int,
    keep_endpoints: bool = False,
    periodic: bool = False,
) -> np.ndarray:
    """Linear interpolation, then one-shot model projection per waypoint.

    This is used by projector models with explicit project_tensor(x) semantics
    (e.g., VAE D(E(x))) and intentionally does not use iterative projector
    hyper-parameters such as alpha/steps/min_steps.
    """
    path = build_linear_path(
        x_start.astype(np.float32),
        x_goal.astype(np.float32),
        n_waypoints=int(n_waypoints),
        periodic=bool(periodic),
    )
    if not hasattr(model, "project_tensor"):
        raise ValueError("model does not implement project_tensor required for one-shot projector planning")
    with torch.no_grad():
        xt = torch.from_numpy(path).to(device)
        proj_t = model.project_tensor(xt)
        out = proj_t.detach().cpu().numpy().astype(np.float32)
    if periodic:
        out = ((out + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    if keep_endpoints:
        out[0] = x_start.astype(np.float32)
        out[-1] = x_goal.astype(np.float32)
    return out


def plan_path_optimized(
    model: nn.Module,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    device: str,
    n_waypoints: int,
    opt_steps: int,
    opt_lr: float,
    lam_manifold: float,
    lam_len: float,
    lam_smooth: float,
    trust_scale: float,
    periodic: bool,
    init_path: np.ndarray | None = None,
    obstacle_center_xy: tuple[float, float] | None = None,
    obstacle_radius: float = 0.0,
    obstacle_margin: float = 0.0,
    lam_obstacle: float = 0.0,
    obstacle_exclude_endpoints: bool = True,
    ref_path_xy: np.ndarray | None = None,
    lam_ref_xy: float = 0.0,
    ref_path: np.ndarray | None = None,
    lam_ref_path: float = 0.0,
    ref_center_z: np.ndarray | None = None,
    lam_center_z_ref: float = 0.0,
    lam_center_z_smooth: float = 0.0,
    lam_backtrack_xy: float = 0.0,
    lam_detour_xy: float = 0.0,
    detour_soft_cap_xy: float = 0.0,
    bound_indices: list[int] | tuple[int, ...] | None = None,
    bound_lo: float | None = None,
    bound_hi: float | None = None,
) -> np.ndarray:
    if init_path is None:
        path0 = build_linear_path(
            x_start.astype(np.float32),
            x_goal.astype(np.float32),
            n_waypoints=int(n_waypoints),
            periodic=bool(periodic),
        )
    else:
        path0 = init_path.astype(np.float32)

    q = torch.tensor(path0, device=device, requires_grad=True)
    q0 = torch.tensor(x_start.astype(np.float32), device=device)
    qT = torch.tensor(x_goal.astype(np.float32), device=device)
    bound_idx_t = None
    bound_lo_f = None
    bound_hi_f = None
    if bound_indices is not None and bound_lo is not None and bound_hi is not None:
        idx = [int(i) for i in bound_indices if 0 <= int(i) < int(q.shape[1])]
        if idx:
            bound_idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            bound_lo_f = float(bound_lo)
            bound_hi_f = float(bound_hi)

    with torch.no_grad():
        v0 = _angle_delta_torch(
            torch.tensor(path0[:-1], device=device, dtype=torch.float32),
            torch.tensor(path0[1:], device=device, dtype=torch.float32),
            periodic=bool(periodic),
        )
        mean_step0 = float(torch.mean(torch.linalg.norm(v0, dim=1)).item()) if v0.shape[0] > 0 else 1.0
    trust_delta = float(max(1e-6, float(trust_scale) * max(mean_step0, 1e-6)))

    opt = torch.optim.Adam([q], lr=float(opt_lr))
    obs_center_t = None
    obs_rad = float(max(0.0, obstacle_radius))
    obs_margin = float(max(0.0, obstacle_margin))
    obs_weight = float(max(0.0, lam_obstacle))
    if obstacle_center_xy is not None and q.shape[1] >= 2 and (obs_rad > 0.0) and (obs_weight > 0.0):
        obs_center_t = torch.tensor(
            [float(obstacle_center_xy[0]), float(obstacle_center_xy[1])],
            device=device,
            dtype=torch.float32,
        ).reshape(1, 2)
    ref_xy_t = None
    if ref_path_xy is not None and q.shape[1] >= 2 and float(lam_ref_xy) > 0.0:
        ref_xy = np.asarray(ref_path_xy, dtype=np.float32)
        if ref_xy.shape[0] == path0.shape[0] and ref_xy.shape[1] >= 2:
            ref_xy_t = torch.tensor(ref_xy[:, :2], device=device, dtype=torch.float32)
    ref_path_t = None
    if ref_path is not None and float(lam_ref_path) > 0.0:
        ref_full = np.asarray(ref_path, dtype=np.float32)
        if ref_full.shape == path0.shape:
            ref_path_t = torch.tensor(ref_full, device=device, dtype=torch.float32)
    ref_center_z_t = None
    if ref_center_z is not None and q.shape[1] >= 9 and float(lam_center_z_ref) > 0.0:
        ref_cz = np.asarray(ref_center_z, dtype=np.float32).reshape(-1)
        if ref_cz.shape[0] == path0.shape[0]:
            ref_center_z_t = torch.tensor(ref_cz, device=device, dtype=torch.float32)
    for _ in range(int(opt_steps)):
        q_prev = q.detach().clone()
        opt.zero_grad(set_to_none=True)
        f = model(q)
        if f.dim() == 1:
            f = f.unsqueeze(1)
        loss_man = (f ** 2).mean()
        v = _angle_delta_torch(q[:-1], q[1:], periodic=bool(periodic))
        loss_len = (v ** 2).mean()
        if q.shape[0] >= 3:
            dv = _angle_delta_torch(v[:-1], v[1:], periodic=bool(periodic))
            loss_smooth = (dv ** 2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=q.device)
        if obs_center_t is not None:
            q_obs = q[1:-1, :] if (bool(obstacle_exclude_endpoints) and q.shape[0] > 2) else q
            if q_obs.shape[0] > 0:
                dxy = q_obs[:, :2] - obs_center_t
                dist = torch.sqrt(torch.sum(dxy * dxy, dim=1) + 1e-12)
                penetration = torch.relu((obs_rad + obs_margin) - dist)
                loss_obs = (penetration ** 2).mean()
            else:
                loss_obs = torch.tensor(0.0, device=q.device)
        else:
            loss_obs = torch.tensor(0.0, device=q.device)
        if ref_xy_t is not None:
            loss_ref = ((q[:, :2] - ref_xy_t) ** 2).mean()
        else:
            loss_ref = torch.tensor(0.0, device=q.device)
        if ref_path_t is not None:
            diff_ref = _angle_delta_torch(ref_path_t, q, periodic=bool(periodic))
            loss_ref_path = (diff_ref ** 2).mean()
        else:
            loss_ref_path = torch.tensor(0.0, device=q.device)
        if q.shape[1] >= 9:
            center_z = 0.5 * (q[:, 2] + q[:, 8])
            if ref_center_z_t is not None:
                loss_center_z_ref = ((center_z - ref_center_z_t) ** 2).mean()
            else:
                loss_center_z_ref = torch.tensor(0.0, device=q.device)
            if q.shape[0] >= 3 and float(lam_center_z_smooth) > 0.0:
                dcz = center_z[1:] - center_z[:-1]
                ddcz = dcz[1:] - dcz[:-1]
                loss_center_z_smooth = (ddcz ** 2).mean()
            else:
                loss_center_z_smooth = torch.tensor(0.0, device=q.device)
        else:
            loss_center_z_ref = torch.tensor(0.0, device=q.device)
            loss_center_z_smooth = torch.tensor(0.0, device=q.device)
        if q.shape[1] >= 2 and float(lam_backtrack_xy) > 0.0:
            chord = qT[:2] - q0[:2]
            chord_n = torch.linalg.norm(chord)
            if float(chord_n.detach().cpu().item()) > 1e-8:
                direction = chord / chord_n
                progress = torch.sum((q[:, :2] - q0[:2]) * direction.reshape(1, 2), dim=1)
                dprog = progress[1:] - progress[:-1]
                loss_backtrack = (torch.relu(-dprog) ** 2).mean()
            else:
                loss_backtrack = torch.tensor(0.0, device=q.device)
        else:
            loss_backtrack = torch.tensor(0.0, device=q.device)
        if q.shape[1] >= 2 and float(lam_detour_xy) > 0.0:
            chord = qT[:2] - q0[:2]
            chord_n = torch.linalg.norm(chord)
            if float(chord_n.detach().cpu().item()) > 1e-8:
                direction = chord / chord_n
                normal = torch.stack([-direction[1], direction[0]])
                perp = torch.sum((q[:, :2] - q0[:2]) * normal.reshape(1, 2), dim=1)
                if float(detour_soft_cap_xy) > 0.0:
                    excess = torch.relu(torch.abs(perp) - float(detour_soft_cap_xy))
                    loss_detour = (excess ** 2).mean()
                else:
                    loss_detour = (perp ** 2).mean()
            else:
                loss_detour = torch.tensor(0.0, device=q.device)
        else:
            loss_detour = torch.tensor(0.0, device=q.device)
        loss = (
            float(lam_manifold) * loss_man
            + float(lam_smooth) * loss_smooth
            + float(lam_len) * loss_len
            + float(obs_weight) * loss_obs
            + float(max(0.0, lam_ref_xy)) * loss_ref
            + float(max(0.0, lam_ref_path)) * loss_ref_path
            + float(max(0.0, lam_center_z_ref)) * loss_center_z_ref
            + float(max(0.0, lam_center_z_smooth)) * loss_center_z_smooth
            + float(max(0.0, lam_backtrack_xy)) * loss_backtrack
            + float(max(0.0, lam_detour_xy)) * loss_detour
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            dq = _angle_delta_torch(q_prev, q, periodic=bool(periodic))
            dn = torch.linalg.norm(dq, dim=1, keepdim=True).clamp_min(1e-9)
            scale = torch.minimum(torch.ones_like(dn), torch.full_like(dn, trust_delta) / dn)
            q[:] = q_prev + dq * scale
            if periodic:
                q[:] = _wrap_torch_pi(q)
            if bound_idx_t is not None:
                q[:, bound_idx_t] = torch.clamp(q[:, bound_idx_t], min=bound_lo_f, max=bound_hi_f)
            q[0] = q0
            q[-1] = qT
    out = q.detach().cpu().numpy().astype(np.float32)
    if periodic:
        out = ((out + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    return out


def _cfg_val(cfg: Any, names: list[str], default: float) -> float:
    for n in names:
        if n.startswith("projector."):
            proj = getattr(cfg, "projector", None)
            if isinstance(proj, dict):
                key = n.split(".", 1)[1]
                pv = proj.get(key, None)
                if pv is not None:
                    return float(pv)
            continue
        if n.startswith("planner."):
            pln = getattr(cfg, "planner", None)
            if isinstance(pln, dict):
                key = n.split(".", 1)[1]
                pv = pln.get(key, None)
                if pv is not None:
                    return float(pv)
            continue
        if hasattr(cfg, n):
            v = getattr(cfg, n)
            if v is not None:
                return float(v)
    return float(default)


def _planner_int(cfg: Any, key: str, default: int) -> int:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        try:
            return int(pln[key])
        except Exception:
            return int(default)
    return int(default)


def _planner_float(cfg: Any, key: str, default: float) -> float:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        try:
            return float(pln[key])
        except Exception:
            return float(default)
    return float(default)


def _planner_int_list(cfg: Any, key: str) -> list[int] | None:
    pln = getattr(cfg, "planner", None)
    if not isinstance(pln, dict) or key not in pln:
        return None
    v = pln.get(key)
    try:
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        return [int(v)]
    except Exception:
        return None


def _planner_bool(cfg: Any, key: str, default: bool) -> bool:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        try:
            return bool(pln[key])
        except Exception:
            return bool(default)
    return bool(default)


def _planner_str(cfg: Any, key: str, default: str) -> str:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        try:
            return str(pln[key])
        except Exception:
            return str(default)
    return str(default)


def _planner_center_xy(cfg: Any) -> tuple[float, float] | None:
    pln = getattr(cfg, "planner", None)
    if not isinstance(pln, dict):
        return None
    if "obstacle_center_xy" not in pln:
        return None
    v = pln.get("obstacle_center_xy")
    try:
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
    except Exception:
        return None
    return None


def _get_ur5_planner_checker() -> dict[str, Any] | None:
    global _UR5_PLANNER_CHECKER
    if _UR5_PLANNER_CHECKER is not None:
        return _UR5_PLANNER_CHECKER
    try:
        import pybullet as p  # type: ignore
    except Exception:
        return None
    try:
        kin = resolve_ur5_kinematics_cfg({})
        ur5 = UR5PyBulletKinematics.from_settings(
            urdf_path=str(kin["urdf_path"]),
            ee_link_index=(int(kin["ee_link_index"]) if int(kin["ee_link_index"]) >= 0 else None),
            tool_axis=str(kin["tool_axis"]),
        )
    except Exception:
        return None

    link_ids = sorted(set([int(v) for v in ur5.arm_joint_indices] + [int(ur5.ee_link_index)]))
    check_pairs: list[tuple[int, int]] = []
    for i in range(len(link_ids)):
        for j in range(i + 1, len(link_ids)):
            a = int(link_ids[i])
            b = int(link_ids[j])
            if abs(a - b) <= 1:
                continue
            check_pairs.append((a, b))

    _UR5_PLANNER_CHECKER = {
        "p": p,
        "ur5": ur5,
        "q_lo": ur5.joint_lower.astype(np.float32),
        "q_hi": ur5.joint_upper.astype(np.float32),
        "check_pairs": check_pairs,
    }
    return _UR5_PLANNER_CHECKER


def _ur5_clip_limits(path: np.ndarray, checker: dict[str, Any]) -> np.ndarray:
    lo = checker["q_lo"].reshape(1, -1)
    hi = checker["q_hi"].reshape(1, -1)
    return np.clip(path.astype(np.float32), lo, hi).astype(np.float32)


def _ur5_in_limit(q: np.ndarray, checker: dict[str, Any], margin: float = 0.0) -> bool:
    lo = checker["q_lo"]
    hi = checker["q_hi"]
    return bool(np.all(q >= (lo + float(margin))) and np.all(q <= (hi - float(margin))))


def _ur5_self_collision(q: np.ndarray, checker: dict[str, Any], margin: float = 0.0) -> bool:
    p = checker["p"]
    ur5 = checker["ur5"]
    ur5._set_q(q.astype(np.float32))
    for a, b in checker["check_pairs"]:
        pts = p.getClosestPoints(
            bodyA=ur5.robot_id,
            bodyB=ur5.robot_id,
            distance=float(margin),
            linkIndexA=int(a),
            linkIndexB=int(b),
            physicsClientId=ur5.client_id,
        )
        if len(pts) > 0:
            return True
    return False


def _ur5_validate_path(
    path: np.ndarray,
    checker: dict[str, Any],
    *,
    limit_margin: float,
    collision_margin: float,
    seg_substeps: int,
    periodic: bool,
) -> tuple[bool, dict[str, Any]]:
    bad: set[int] = set()
    limit_viol = 0
    coll_viol = 0
    n = int(path.shape[0])
    for i in range(n):
        qi = path[i].astype(np.float32)
        ok_lim = _ur5_in_limit(qi, checker, margin=limit_margin)
        if not ok_lim:
            limit_viol += 1
            bad.add(i)
            continue
        if _ur5_self_collision(qi, checker, margin=collision_margin):
            coll_viol += 1
            bad.add(i)
    if int(seg_substeps) > 0:
        ss = np.linspace(0.0, 1.0, int(seg_substeps) + 2, dtype=np.float32)[1:-1]
        for i in range(n - 1):
            a = path[i].astype(np.float32)
            b = path[i + 1].astype(np.float32)
            for s in ss:
                q = _interp_linear_np(a, b, np.array([[float(s)]], dtype=np.float32), periodic=periodic).reshape(-1)
                if (not _ur5_in_limit(q, checker, margin=limit_margin)) or _ur5_self_collision(
                    q, checker, margin=collision_margin
                ):
                    bad.add(i)
                    bad.add(i + 1)
                    coll_viol += 1
                    break
    return (len(bad) == 0), {"bad_ids": sorted(bad), "limit_viol": int(limit_viol), "coll_viol": int(coll_viol)}


def _plan_path_pybullet_guarded(
    model: nn.Module,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    cfg: Any,
    *,
    base_planner: str,
    n_waypoints: int,
    periodic: bool,
    init_path: np.ndarray | None,
    f_abs_stop: float | None,
    keep_endpoints: bool,
) -> np.ndarray:
    checker = _get_ur5_planner_checker()
    if checker is None:
        # pybullet unavailable: fallback to default planner path.
        if base_planner == "linear_proj":
            return plan_linear_then_project(
                model,
                x_start,
                x_goal,
                device=str(cfg.device),
                n_waypoints=int(n_waypoints),
                proj_steps=int(_cfg_val(cfg, ["projector.steps"], 100)),
                proj_alpha=float(_cfg_val(cfg, ["projector.alpha"], 0.3)),
                proj_min_steps=int(_cfg_val(cfg, ["projector.min_steps"], 0)),
                f_abs_stop=f_abs_stop,
                keep_endpoints=bool(keep_endpoints),
                periodic=bool(periodic),
            )
        return plan_path_optimized(
            model,
            x_start,
            x_goal,
            device=str(cfg.device),
            n_waypoints=int(n_waypoints),
            opt_steps=int(_cfg_val(cfg, ["planner.opt_steps"], 1240)),
            opt_lr=float(_cfg_val(cfg, ["planner.opt_lr"], 0.01)),
            lam_manifold=float(_cfg_val(cfg, ["planner.lam_manifold"], 1.0)),
            lam_len=float(_cfg_val(cfg, ["planner.lam_len_joint"], 0.40)),
            lam_smooth=float(_cfg_val(cfg, ["planner.opt_lam_smooth"], 0.2)),
            trust_scale=float(_cfg_val(cfg, ["planner.trust_scale"], 0.8)),
            periodic=bool(periodic),
            init_path=init_path,
        )

    restarts = max(1, _planner_int(cfg, "pyb_restarts", 6))
    repair_rounds = max(0, _planner_int(cfg, "pyb_repair_rounds", 3))
    repair_win = max(1, _planner_int(cfg, "pyb_repair_window", 7))
    repair_steps = max(20, _planner_int(cfg, "pyb_repair_opt_steps", 180))
    seg_substeps = max(0, _planner_int(cfg, "pyb_check_substeps", 4))
    limit_margin = max(0.0, _planner_float(cfg, "pyb_limit_margin", 0.0))
    collision_margin = max(0.0, _planner_float(cfg, "pyb_collision_margin", 0.0))
    restart_noise = max(0.0, _planner_float(cfg, "pyb_restart_noise_scale", 0.10))
    lo = checker["q_lo"]
    hi = checker["q_hi"]
    span = np.maximum(hi - lo, 1e-3)
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 2027)

    def _run_base(_init_path: np.ndarray | None, _opt_steps_override: int | None = None) -> np.ndarray:
        if base_planner == "linear_proj":
            out = plan_linear_then_project(
                model,
                x_start,
                x_goal,
                device=str(cfg.device),
                n_waypoints=int(n_waypoints),
                proj_steps=int(_cfg_val(cfg, ["projector.steps"], 100)),
                proj_alpha=float(_cfg_val(cfg, ["projector.alpha"], 0.3)),
                proj_min_steps=int(_cfg_val(cfg, ["projector.min_steps"], 0)),
                f_abs_stop=f_abs_stop,
                keep_endpoints=bool(keep_endpoints),
                periodic=bool(periodic),
            )
            return _ur5_clip_limits(out, checker)
        out = plan_path_optimized(
            model,
            x_start,
            x_goal,
            device=str(cfg.device),
            n_waypoints=int(n_waypoints),
            opt_steps=int(_opt_steps_override if _opt_steps_override is not None else _cfg_val(cfg, ["planner.opt_steps"], 1240)),
            opt_lr=float(_cfg_val(cfg, ["planner.opt_lr"], 0.01)),
            lam_manifold=float(_cfg_val(cfg, ["planner.lam_manifold"], 1.0)),
            lam_len=float(_cfg_val(cfg, ["planner.lam_len_joint"], 0.40)),
            lam_smooth=float(_cfg_val(cfg, ["planner.opt_lam_smooth"], 0.2)),
            trust_scale=float(_cfg_val(cfg, ["planner.trust_scale"], 0.8)),
            periodic=bool(periodic),
            init_path=_init_path,
        )
        return _ur5_clip_limits(out, checker)

    best_path = None
    best_score = 10**9
    for t in range(restarts):
        if t == 0 and init_path is not None:
            cur_init = _ur5_clip_limits(init_path.astype(np.float32), checker)
        else:
            if init_path is not None:
                cur_init = init_path.astype(np.float32).copy()
            else:
                cur_init = build_linear_path(
                    x_start.astype(np.float32),
                    x_goal.astype(np.float32),
                    n_waypoints=int(n_waypoints),
                    periodic=bool(periodic),
                ).astype(np.float32)
            if cur_init.shape[0] > 2:
                noise = rng.normal(size=cur_init[1:-1].shape).astype(np.float32)
                cur_init[1:-1] = cur_init[1:-1] + float(restart_noise) * noise * span.reshape(1, -1)
            cur_init = _ur5_clip_limits(cur_init, checker)
            if periodic:
                cur_init = _wrap_np_pi(cur_init)
            cur_init[0] = x_start.astype(np.float32)
            cur_init[-1] = x_goal.astype(np.float32)

        q_path = _run_base(cur_init)
        ok, rep = _ur5_validate_path(
            q_path,
            checker,
            limit_margin=limit_margin,
            collision_margin=collision_margin,
            seg_substeps=seg_substeps,
            periodic=bool(periodic),
        )
        score = int(rep["limit_viol"]) + int(rep["coll_viol"]) + len(rep["bad_ids"])
        if score < best_score:
            best_score = score
            best_path = q_path.copy()
        if ok:
            return q_path

        # Local repair around violating waypoints, then short traj-opt refinement.
        repaired = q_path.copy()
        for _ in range(repair_rounds):
            bad_ids = list(rep["bad_ids"])
            if not bad_ids:
                break
            for bi in bad_ids:
                if bi <= 0 or bi >= repaired.shape[0] - 1:
                    continue
                l = max(1, int(bi) - repair_win)
                r = min(repaired.shape[0] - 2, int(bi) + repair_win)
                repaired[l : r + 1] += (
                    float(restart_noise) * 0.75 * rng.normal(size=repaired[l : r + 1].shape).astype(np.float32) * span.reshape(1, -1)
                )
            repaired = _ur5_clip_limits(repaired, checker)
            if periodic:
                repaired = _wrap_np_pi(repaired)
            repaired[0] = x_start.astype(np.float32)
            repaired[-1] = x_goal.astype(np.float32)
            repaired = _run_base(repaired, _opt_steps_override=repair_steps)
            ok, rep = _ur5_validate_path(
                repaired,
                checker,
                limit_margin=limit_margin,
                collision_margin=collision_margin,
                seg_substeps=seg_substeps,
                periodic=bool(periodic),
            )
            score = int(rep["limit_viol"]) + int(rep["coll_viol"]) + len(rep["bad_ids"])
            if score < best_score:
                best_score = score
                best_path = repaired.copy()
            if ok:
                return repaired
    assert best_path is not None
    return best_path


def plan_path(
    model: nn.Module,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    cfg: Any,
    *,
    planner_name: str,
    n_waypoints: int,
    dataset_name: str | None = None,
    periodic_joint: bool | None = None,
    init_path: np.ndarray | None = None,
    f_abs_stop: float | None = None,
    keep_endpoints: bool = False,
) -> np.ndarray:
    periodic = resolve_periodic_mode(periodic_joint=periodic_joint, dataset_name=dataset_name)
    p = str(planner_name).strip().lower()
    use_ur5_pybullet_guard = str(dataset_name or "").strip() == "6d_spatial_arm_up_n6"
    if use_ur5_pybullet_guard and p in ("traj_opt", "linear_proj"):
        return _plan_path_pybullet_guarded(
            model,
            x_start,
            x_goal,
            cfg,
            base_planner=p,
            n_waypoints=int(n_waypoints),
            periodic=bool(periodic),
            init_path=init_path,
            f_abs_stop=f_abs_stop,
            keep_endpoints=bool(keep_endpoints),
        )
    if p == "linear_proj":
        return plan_linear_then_project(
            model,
            x_start,
            x_goal,
            device=str(cfg.device),
            n_waypoints=int(n_waypoints),
            proj_steps=int(_cfg_val(cfg, ["projector.steps"], 100)),
            proj_alpha=float(_cfg_val(cfg, ["projector.alpha"], 0.3)),
            proj_min_steps=int(_cfg_val(cfg, ["projector.min_steps"], 0)),
            f_abs_stop=f_abs_stop,
            keep_endpoints=bool(keep_endpoints),
            periodic=bool(periodic),
        )
    if p == "traj_opt":
        objective_mode = _planner_str(cfg, "traj_opt_objective", "full").strip().lower()
        minimal_objective = objective_mode in ("minimal", "minimal_zsmooth", "minimal_refinit", "minimal_refinit_zlinear")
        minimal_with_zsmooth = objective_mode == "minimal_zsmooth"
        minimal_with_refinit = objective_mode == "minimal_refinit"
        minimal_with_refinit_zlinear = objective_mode == "minimal_refinit_zlinear"
        obs_enabled = _planner_bool(cfg, "obstacle_enable", False)
        obs_center = _planner_center_xy(cfg) if obs_enabled else None
        obs_radius = _planner_float(cfg, "obstacle_radius", 0.0) if obs_enabled else 0.0
        obs_margin = _planner_float(cfg, "obstacle_margin", 0.0) if obs_enabled else 0.0
        obs_weight = _planner_float(cfg, "lam_obstacle", 0.0) if obs_enabled else 0.0
        obs_excl_ep = _planner_bool(cfg, "obstacle_exclude_endpoints", True)
        lam_ref_path = _planner_float(cfg, "lam_ref_path", 0.0)
        lam_center_z_ref = _planner_float(cfg, "lam_center_z_ref", 0.0)
        lam_center_z_smooth = _planner_float(cfg, "lam_center_z_smooth", 0.0)
        bound_indices = _planner_int_list(cfg, "bound_indices")
        bound_lo = _planner_float(cfg, "bound_lo", float("nan"))
        bound_hi = _planner_float(cfg, "bound_hi", float("nan"))
        if bound_indices is None or not np.isfinite(bound_lo) or not np.isfinite(bound_hi):
            bound_indices = None
            bound_lo = None
            bound_hi = None
        direct_path_blocked = False
        if (
            obs_enabled
            and obs_center is not None
            and float(obs_radius) > 0.0
            and not bool(periodic)
            and int(x_start.shape[0]) >= 2
        ):
            direct_path_blocked = _line_intersects_circle_xy(
                x_start,
                x_goal,
                obstacle_center_xy=obs_center,
                obstacle_radius=float(obs_radius),
                obstacle_margin=float(obs_margin),
            )
        if minimal_objective:
            obs_center = None
            obs_radius = 0.0
            obs_margin = 0.0
            obs_weight = 0.0
            lam_ref_path = (lam_ref_path if (minimal_with_refinit or minimal_with_refinit_zlinear) else 0.0)
            lam_center_z_ref = (lam_center_z_ref if minimal_with_refinit_zlinear else 0.0)
            lam_center_z_smooth = (lam_center_z_smooth if minimal_with_zsmooth else 0.0)
            bound_indices = None
            bound_lo = None
            bound_hi = None
            direct_path_blocked = False
        linear_ref_weight = 6.0 if not bool(direct_path_blocked) else 0.0
        detour_soft_cap = float(max(0.0, float(obs_radius) + float(obs_margin) + 0.03)) if obs_enabled else 0.0
        common_kwargs = dict(
            device=str(cfg.device),
            n_waypoints=int(n_waypoints),
            opt_steps=int(_cfg_val(cfg, ["planner.opt_steps"], 1240)),
            opt_lr=float(_cfg_val(cfg, ["planner.opt_lr"], 0.01)),
            lam_manifold=float(_cfg_val(cfg, ["planner.lam_manifold"], 1.0)),
            lam_len=(0.0 if minimal_objective else float(_cfg_val(cfg, ["planner.lam_len_joint"], 0.40))),
            lam_smooth=(0.0 if minimal_objective else float(_cfg_val(cfg, ["planner.opt_lam_smooth"], 0.2))),
            trust_scale=float(_cfg_val(cfg, ["planner.trust_scale"], 0.8)),
            periodic=bool(periodic),
            obstacle_center_xy=obs_center,
            obstacle_radius=float(obs_radius),
            obstacle_margin=float(obs_margin),
            lam_obstacle=float(obs_weight),
            obstacle_exclude_endpoints=bool(obs_excl_ep),
            lam_backtrack_xy=(0.0 if minimal_objective else (1.0 if bool(direct_path_blocked) else 0.6)),
            lam_detour_xy=(0.0 if minimal_objective else (1.4 if bool(direct_path_blocked) else 0.9)),
            detour_soft_cap_xy=detour_soft_cap,
            lam_ref_path=float(lam_ref_path),
            lam_center_z_ref=float(lam_center_z_ref),
            lam_center_z_smooth=float(lam_center_z_smooth),
            bound_indices=bound_indices,
            bound_lo=bound_lo,
            bound_hi=bound_hi,
        )
        if init_path is not None:
            ref_center_z = None
            if minimal_with_refinit_zlinear and init_path.ndim == 2 and init_path.shape[1] >= 9 and x_start.shape[0] >= 9:
                cz0 = 0.5 * float(x_start[2] + x_start[8])
                czT = 0.5 * float(x_goal[2] + x_goal[8])
                ref_center_z = np.linspace(cz0, czT, int(init_path.shape[0]), dtype=np.float32)
            elif (not minimal_objective) and init_path.ndim == 2 and init_path.shape[1] >= 9:
                ref_center_z = (0.5 * (init_path[:, 2] + init_path[:, 8])).astype(np.float32)
            return plan_path_optimized(
                model,
                x_start,
                x_goal,
                init_path=init_path,
                ref_path_xy=(None if minimal_objective else init_path.astype(np.float32)),
                ref_path=(init_path.astype(np.float32) if (minimal_with_refinit or (not minimal_objective)) else None),
                ref_center_z=ref_center_z,
                **common_kwargs,
            )
        linear_init = build_linear_path(
            x_start.astype(np.float32),
            x_goal.astype(np.float32),
            n_waypoints=int(n_waypoints),
            periodic=bool(periodic),
        )
        if bool(direct_path_blocked):
            candidates = _build_perturbed_linear_init_paths(
                x_start.astype(np.float32),
                x_goal.astype(np.float32),
                n_waypoints=int(n_waypoints),
                periodic=bool(periodic),
                obstacle_center_xy=obs_center,
                obstacle_radius=float(obs_radius),
                obstacle_margin=float(obs_margin),
            )
            best_path = None
            best_score = 1e18
            for cand in candidates:
                out = plan_path_optimized(
                    model,
                    x_start,
                    x_goal,
                    init_path=cand.astype(np.float32),
                    ref_path_xy=cand.astype(np.float32),
                    ref_path=cand.astype(np.float32),
                    ref_center_z=(0.5 * (cand[:, 2] + cand[:, 8])).astype(np.float32) if cand.shape[1] >= 9 else None,
                    lam_ref_xy=0.45,
                    **common_kwargs,
                )
                score = _path_xy_quality_score(
                    out,
                    obstacle_center_xy=obs_center,
                    obstacle_radius=float(obs_radius),
                    obstacle_margin=float(obs_margin),
                )
                if score < best_score:
                    best_score = score
                    best_path = out.astype(np.float32)
            assert best_path is not None
            return best_path
        return plan_path_optimized(
            model,
            x_start,
            x_goal,
            init_path=linear_init.astype(np.float32),
            ref_path_xy=linear_init.astype(np.float32) if not bool(direct_path_blocked) else None,
            ref_path=linear_init.astype(np.float32),
            ref_center_z=(0.5 * (linear_init[:, 2] + linear_init[:, 8])).astype(np.float32) if linear_init.shape[1] >= 9 else None,
            lam_ref_xy=linear_ref_weight,
            **common_kwargs,
        )
    raise ValueError(f"unknown planner_name '{planner_name}'")


def _pln(cfg: Any, key: str, default: Any) -> Any:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        return pln[key]
    return default

def _plot_planar_arm_planning(
    model: nn.Module,
    name: str,
    x_train: np.ndarray,
    out_path: str,
    cfg: Any,
    render_pybullet: bool = True,
) -> list[np.ndarray]:
    base_name = str(name[:-5] if str(name).endswith("_traj") else name)
    if base_name == "2d_planar_arm_line_n2":
        lengths = [1.0, 0.8]
        y_line = 0.3
        is_spatial = False
        use_pybullet_n6 = False
    elif base_name == "3d_planar_arm_line_n3":
        lengths = [1.0, 0.8, 0.6]
        y_line = 0.35
        is_spatial = False
        use_pybullet_n6 = False
    elif base_name == "3d_spatial_arm_plane_n3":
        lengths = [1.0, 0.8]
        y_line = 0.35  # here means z-plane value
        is_spatial = True
        use_pybullet_n6 = False
    elif base_name in ("3d_spatial_arm_ellip_n3", "3d_spatial_arm_circle_n3"):
        lengths = [1.0, 0.8]
        y_line = None
        is_spatial = True
        use_pybullet_n6 = False
    elif base_name == "6d_spatial_arm_up_n6":
        lengths = list(UR5_LINK_LENGTHS)
        y_line = None
        is_spatial = True
        use_pybullet_n6 = True
    elif base_name == "6d_spatial_arm_up_n6_py":
        lengths = list(UR5_LINK_LENGTHS)
        y_line = None
        is_spatial = True
        use_pybullet_n6 = False
    else:
        return []

    # Sample start/goal from a denser manifold candidate set, then enforce workspace distance.
    try:
        x_dense, grid_dense = generate_dataset(base_name, cfg)
        cand = grid_dense if (grid_dense is not None and len(grid_dense) >= 2) else x_dense
        if cand.shape[1] != x_train.shape[1]:
            cand = x_train
    except Exception:
        cand = x_train

    cases: list[tuple[np.ndarray, np.ndarray, float]] = []
    lo = float(_pln(cfg, "pair_min_ratio", 0.15)) * max(float(sum(lengths)), 1e-6)
    hi = float(_pln(cfg, "pair_max_ratio", 0.35)) * max(float(sum(lengths)), 1e-6)
    if is_spatial:
        q_c = cand.astype(np.float32)
        ee_c = spatial_fk(q_c, lengths, use_pybullet_n6=use_pybullet_n6)[:, -1, :]  # (N,3)
        target = 0.5 * (lo + hi)
    else:
        q_c = None
        ee_c = None
        target = 0.5 * (lo + hi)
    for _ in range(3):
        if not is_spatial:
            q_start, q_goal, ee_dist, _ = pick_far_pair_workspace_planar(
                x=cand.astype(np.float32),
                lengths=lengths,
                min_ratio=float(_pln(cfg, "pair_min_ratio", 0.15)),
                max_ratio=float(_pln(cfg, "pair_max_ratio", 0.35)),
                tries=int(_pln(cfg, "pair_tries", 1200)),
            )
        else:
            best = None
            best_delta = 1e18
            for _t in range(max(1, int(_pln(cfg, "pair_tries", 1200)))):
                i = int(np.random.randint(0, len(q_c)))
                j = int(np.random.randint(0, len(q_c)))
                if i == j:
                    continue
                d = float(np.linalg.norm(ee_c[i] - ee_c[j]))
                if lo <= d <= hi:
                    best = (q_c[i], q_c[j], d)
                    break
                delta = abs(d - target)
                if delta < best_delta:
                    best_delta = delta
                    best = (q_c[i], q_c[j], d)
            assert best is not None
            q_start, q_goal, ee_dist = best[0], best[1], float(best[2])
        cases.append((q_start, q_goal, ee_dist))

    # Precompute constraint 0-level overlay once.
    contour_2d = None
    zpts_3d = None
    if x_train.shape[1] == 2:
        mins, maxs = eval_bounds_from_train(x_train, cfg)
        mins = mins.copy()
        maxs = maxs.copy()
        mins[0] = min(float(mins[0]), -np.pi)
        maxs[0] = max(float(maxs[0]), np.pi)
        mins[1] = min(float(mins[1]), -np.pi)
        maxs[1] = max(float(maxs[1]), np.pi)
        q1g, q2g = np.meshgrid(
            np.linspace(float(mins[0]), float(maxs[0]), 260),
            np.linspace(float(mins[1]), float(maxs[1]), 260),
        )
        grid = np.stack([q1g, q2g], axis=2).reshape(-1, 2).astype(np.float32)
        with torch.no_grad():
            fg = model(torch.from_numpy(grid).to(cfg.device))
            if fg.dim() == 1:
                fg = fg.unsqueeze(1)
            f1 = fg[:, 0].detach().cpu().numpy().reshape(q1g.shape)
        contour_2d = (q1g, q2g, f1, mins, maxs)
    elif x_train.shape[1] == 3:
        mins3, maxs3 = eval_bounds_from_train(x_train, cfg)
        q1g, q2g, q3g = np.meshgrid(
            np.linspace(float(mins3[0]), float(maxs3[0]), 34),
            np.linspace(float(mins3[1]), float(maxs3[1]), 34),
            np.linspace(float(mins3[2]), float(maxs3[2]), 34),
            indexing="ij",
        )
        grid3 = np.stack([q1g.ravel(), q2g.ravel(), q3g.ravel()], axis=1).astype(np.float32)
        with torch.no_grad():
            f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(cfg.device))
            if f_on.dim() == 1:
                f_on = f_on.unsqueeze(1)
            eps_q = float(getattr(cfg, "zero_eps_quantile", ZERO_EPS_QUANTILE_DEFAULT))
            eps0 = float(np.percentile(np.abs(f_on[:, 0].detach().cpu().numpy()), eps_q))
            vals = []
            chunk = max(2048, int(getattr(cfg, "surface_eval_chunk", 8192)))
            for s in range(0, len(grid3), chunk):
                e = min(len(grid3), s + chunk)
                fg = model(torch.from_numpy(grid3[s:e]).to(cfg.device))
                if fg.dim() == 1:
                    fg = fg.unsqueeze(1)
                vals.append(fg[:, 0].detach().cpu().numpy())
            f1g = np.concatenate(vals, axis=0)
        zmask = np.abs(f1g) <= max(eps0, 1e-4)
        zpts_3d = grid3[zmask]
        if len(zpts_3d) > 4500:
            idx = np.random.choice(len(zpts_3d), size=4500, replace=False)
            zpts_3d = zpts_3d[idx]

    fig = plt.figure(figsize=(10.5, 12.0))
    q_paths_render: list[np.ndarray] = []
    for row, (q_start, q_goal, ee_dist) in enumerate(cases):
        if str(_pln(cfg, "init_mode", "joint_spline")).lower() == "workspace_ik":
            q_lin = init_path_via_workspace_ik(
                q_start=q_start.astype(np.float32),
                q_goal=q_goal.astype(np.float32),
                lengths=lengths,
                is_spatial=is_spatial,
                use_pybullet_n6=use_pybullet_n6,
                n_waypoints=140,
                device=str(cfg.device),
            )
        else:
            q_lin = init_path_joint_spline(
                q_start=q_start.astype(np.float32),
                q_goal=q_goal.astype(np.float32),
                n_waypoints=140,
                mid_noise=float(_pln(cfg, "joint_mid_noise", 0.0)),
            )
        planner_name = str(_pln(cfg, "method", "traj_opt"))
        p_name = planner_name.strip().lower()
        if p_name == "linear_proj" and hasattr(model, "project_tensor"):
            q_path = plan_linear_then_model_project(
                model=model,
                x_start=q_start.astype(np.float32),
                x_goal=q_goal.astype(np.float32),
                device=str(cfg.device),
                n_waypoints=int(q_lin.shape[0]),
                periodic=bool(is_arm_dataset(name)),
            )
        else:
            q_path = plan_path(
                model=model,
                x_start=q_start.astype(np.float32),
                x_goal=q_goal.astype(np.float32),
                cfg=cfg,
                planner_name=planner_name,
                n_waypoints=int(q_lin.shape[0]),
                dataset_name=str(name),
                periodic_joint=bool(is_arm_dataset(name)),
                init_path=q_lin.astype(np.float32),
            )
        q_paths_render.append(q_path.astype(np.float32))
        q_init = q_lin.astype(np.float32)
        if not is_spatial:
            joints = planar_fk(q_path, lengths)
            ee = joints[:, -1, :]
        else:
            joints = spatial_fk(q_path, lengths, use_pybullet_n6=use_pybullet_n6)
            ee = joints[:, -1, :]
            joints_init = spatial_fk(q_init, lengths, use_pybullet_n6=use_pybullet_n6)
            ee_init = joints_init[:, -1, :]

        left_idx = 2 * row + 1
        right_idx = 2 * row + 2
        if q_path.shape[1] == 2:
            ax1 = fig.add_subplot(3, 2, left_idx)
            assert contour_2d is not None
            q1g, q2g, f1, mins, maxs = contour_2d
            ax1.contour(q1g, q2g, f1, levels=[0.0], colors=["red"], linewidths=1.4, alpha=0.95)
            q_unw = np.unwrap(q_path, axis=0)
            ax1.plot(q_unw[:, 0], q_unw[:, 1], "-", color="#0ea5e9", lw=1.2, alpha=0.85, label="planned")
            ax1.scatter([q_unw[0, 0], q_unw[-1, 0]], [q_unw[0, 1], q_unw[-1, 1]], c=["blue", "red"], s=20, zorder=3)
            ax1.set_xlim(float(mins[0]), float(maxs[0]))
            ax1.set_ylim(float(mins[1]), float(maxs[1]))
            ax1.set_aspect("equal", adjustable="box")
            ax1.set_xlabel("q1")
            ax1.set_ylabel("q2")
            ax1.set_title(f"Joint Path #{row+1} (ee={ee_dist:.2f})", fontsize=10)
            ax1.grid(alpha=0.25)
            ax1.legend(loc="best", fontsize=8)
        elif q_path.shape[1] == 3:
            ax1 = fig.add_subplot(3, 2, left_idx, projection="3d")
            q_unw = np.unwrap(q_path, axis=0)
            if zpts_3d is not None and len(zpts_3d) > 0:
                ax1.scatter(zpts_3d[:, 0], zpts_3d[:, 1], zpts_3d[:, 2], s=1.0, c="red", alpha=0.18)
            ax1.plot(q_unw[:, 0], q_unw[:, 1], q_unw[:, 2], "-", color="#0ea5e9", lw=1.0, alpha=0.8, label="planned")
            ax1.scatter([q_unw[0, 0]], [q_unw[0, 1]], [q_unw[0, 2]], c="blue", s=20)
            ax1.scatter([q_unw[-1, 0]], [q_unw[-1, 1]], [q_unw[-1, 2]], c="red", s=20)
            ax1.set_xlabel("q1")
            ax1.set_ylabel("q2")
            ax1.set_zlabel("q3")
            ax1.set_title(f"Joint Path #{row+1} (ee={ee_dist:.2f})", fontsize=10)
            ax1.legend(loc="best", fontsize=8)
        else:
            # 4D joint path: PCA to 3D for visualization.
            ax1 = fig.add_subplot(3, 2, left_idx, projection="3d")
            q_ref = x_train.astype(np.float32)
            mu = q_ref.mean(axis=0, keepdims=True)
            q0 = q_ref - mu
            _, _, vt = np.linalg.svd(q0, full_matrices=False)
            basis = vt[:3].T  # (4,3)
            emb = (q_path - mu) @ basis
            q_unw = np.unwrap(q_path, axis=0)
            emb_unw = (q_unw - mu) @ basis
            q_init_unw = np.unwrap(q_init, axis=0)
            emb_init = (q_init_unw - mu) @ basis
            emb_s = emb_unw[0:1]
            emb_g = emb_unw[-1:]
            if base_name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
                ax1.plot(emb_init[:, 0], emb_init[:, 1], emb_init[:, 2], "--", color="gray", lw=1.1, alpha=0.85, label="init (pre-proj)")
            ax1.plot(emb_unw[:, 0], emb_unw[:, 1], emb_unw[:, 2], "-", color="#0ea5e9", lw=1.2, alpha=0.85, label="planned")
            ax1.scatter([emb_s[0, 0]], [emb_s[0, 1]], [emb_s[0, 2]], c="blue", s=20)
            ax1.scatter([emb_g[0, 0]], [emb_g[0, 1]], [emb_g[0, 2]], c="red", s=20)
            ax1.set_xlabel("pc1")
            ax1.set_ylabel("pc2")
            ax1.set_zlabel("pc3")
            ax1.set_title(f"Joint Path #{row+1} (4D path in PCA-3D)", fontsize=10)

        if not is_spatial:
            ax2 = fig.add_subplot(3, 2, right_idx)
            ax2.plot(ee[:, 0], ee[:, 1], "-", color="green", lw=1.6, label="ee trail")
            step = max(1, len(joints) // 12)
            idx_list = list(range(0, len(joints), step))
            if idx_list[-1] != len(joints) - 1:
                idx_list.append(len(joints) - 1)
            n_seg = max(1, len(idx_list) - 1)
            for k, i in enumerate(idx_list):
                c = 0.88 - 0.70 * (k / n_seg)
                arm_color = (c, c, c)
                ax2.plot(joints[i, :, 0], joints[i, :, 1], "-", color=arm_color, alpha=0.95, lw=1.1)
            ax2.plot(joints[0, :, 0], joints[0, :, 1], "-", color="blue", lw=1.3)
            ax2.plot(joints[-1, :, 0], joints[-1, :, 1], "-", color="red", lw=1.3)
            ax2.scatter([ee[0, 0]], [ee[0, 1]], c="blue", s=26, zorder=5)
            ax2.scatter([ee[-1, 0]], [ee[-1, 1]], c="red", s=26, zorder=5)
            ax2.axhline(float(y_line), color="orange", linestyle="--", linewidth=1.7, alpha=0.9, label="GT line")
            reach = float(sum(lengths)) + 0.15
            ax2.set_xlim(-reach, reach)
            ax2.set_ylim(-reach, reach)
            ax2.set_aspect("equal", adjustable="box")
            ax2.grid(alpha=0.25)
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_title(f"Workspace #{row+1}", fontsize=10)
        else:
            ax2 = fig.add_subplot(3, 2, right_idx, projection="3d")
            if base_name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
                ax2.plot(ee_init[:, 0], ee_init[:, 1], ee_init[:, 2], "--", color="gray", lw=1.2, alpha=0.85, label="init ee")
                ax2.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color="green", lw=1.8, label="planned ee")
            else:
                ax2.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color="green", lw=1.6, label="ee trail")
                step = max(1, len(joints) // 12)
                idx_list = list(range(0, len(joints), step))
                if idx_list[-1] != len(joints) - 1:
                    idx_list.append(len(joints) - 1)
                n_seg = max(1, len(idx_list) - 1)
                for k, i in enumerate(idx_list):
                    c = 0.88 - 0.70 * (k / n_seg)
                    arm_color = (c, c, c)
                    ax2.plot(joints[i, :, 0], joints[i, :, 1], joints[i, :, 2], "-", color=arm_color, alpha=0.95, lw=1.1)
            # GT plane z=z_plane for plane-constraint datasets.
            if y_line is not None:
                xr = np.linspace(np.min(ee[:, 0]) - 0.3, np.max(ee[:, 0]) + 0.3, 12)
                yr = np.linspace(np.min(ee[:, 1]) - 0.3, np.max(ee[:, 1]) + 0.3, 12)
                XX, YY = np.meshgrid(xr, yr)
                ZZ = np.full_like(XX, float(y_line))
                ax2.plot_surface(XX, YY, ZZ, alpha=0.12, color="orange", linewidth=0, shade=False)
            ax2.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], c="blue", s=26)
            ax2.scatter([ee[-1, 0]], [ee[-1, 1]], [ee[-1, 2]], c="red", s=26)
            if base_name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py") and joints.shape[1] >= 2:
                # Visualize true end-effector tool orientation.
                d = spatial_tool_axis_n6(q_path.astype(np.float32), use_pybullet=use_pybullet_n6)
                d0 = spatial_tool_axis_n6(q_init.astype(np.float32), use_pybullet=use_pybullet_n6)
                qstep = max(1, len(ee) // 14)
                qidx = np.arange(0, len(ee), qstep, dtype=int)
                if qidx[-1] != len(ee) - 1:
                    qidx = np.concatenate([qidx, np.array([len(ee) - 1], dtype=int)])
                scale = 0.22
                ax2.quiver(
                    ee_init[qidx, 0], ee_init[qidx, 1], ee_init[qidx, 2],
                    d0[qidx, 0], d0[qidx, 1], d0[qidx, 2],
                    length=scale, normalize=True, color="#9ca3af", linewidth=0.8, alpha=0.75
                )
                ax2.quiver(
                    ee[qidx, 0], ee[qidx, 1], ee[qidx, 2],
                    d[qidx, 0], d[qidx, 1], d[qidx, 2],
                    length=scale, normalize=True, color="#f59e0b", linewidth=1.0, alpha=0.9
                )
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("z")
            ax2.set_title(f"Workspace 3D #{row+1}", fontsize=10)
            if base_name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
                ax2.legend(loc="best", fontsize=8)

    fig.suptitle(f"{name}: planning on learned manifold (3 cases)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    if bool(getattr(cfg, "show_3d_plot", True)):
        plt.show()
    plt.close(fig)
    print(f"saved: {out_path}")
    if render_pybullet and base_name == "6d_spatial_arm_up_n6" and bool(_pln(cfg, "pybullet_render", False)):
        _render_ur5_pybullet_trajectories(q_paths_render, cfg)

    # Slow animation of workspace motion.
    if not bool(_pln(cfg, "save_gif", True)):
        return q_paths_render
    out_gif = out_path.replace(".png", "_anim.gif")
    if is_spatial:
        fig2 = plt.figure(figsize=(10.2, 8.4))
        ax = fig2.add_subplot(111, projection="3d")
        # Tighter limits around trajectory for better visibility.
        c = np.mean(ee, axis=0)
        span = np.max(np.ptp(ee, axis=0))
        span = float(max(span, 0.35))
        half = 0.58 * span
        reach = float(sum(lengths)) + 0.15
        ax.set_xlim(float(c[0] - half), float(c[0] + half))
        ax.set_ylim(float(c[1] - half), float(c[1] + half))
        ax.set_zlim(float(c[2] - half), float(c[2] + half))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{name}: workspace motion (animation)")
        # GT plane z=z_plane for plane-constraint datasets.
        if y_line is not None:
            xr = np.linspace(-reach, reach, 12)
            yr = np.linspace(-reach, reach, 12)
            XX, YY = np.meshgrid(xr, yr)
            ZZ = np.full_like(XX, float(y_line))
            ax.plot_surface(XX, YY, ZZ, alpha=0.10, color="orange", linewidth=0, shade=False)
        ax.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], c="blue", s=38)
        ax.scatter([ee[-1, 0]], [ee[-1, 1]], [ee[-1, 2]], c="red", s=38)
        arm_line, = ax.plot([], [], [], "-", color="black", lw=2.0, alpha=0.9)
        trail, = ax.plot([], [], [], "-", color="green", lw=2.0, alpha=0.9)
        dot, = ax.plot([], [], [], "o", color="green", ms=6)
        ori_line = None
        ori = None
        ori_scale = 0.20
        if base_name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
            ori = spatial_tool_axis_n6(q_path.astype(np.float32), use_pybullet=use_pybullet_n6)
            ori_line, = ax.plot([], [], [], "-", color="#f59e0b", lw=2.2, alpha=0.95)

        frame_idx = np.arange(0, len(joints), max(1, int(_pln(cfg, "anim_stride", 1))), dtype=int)
        if frame_idx[-1] != len(joints) - 1:
            frame_idx = np.concatenate([frame_idx, np.array([len(joints) - 1], dtype=int)])

        def _init3():
            arm_line.set_data([], [])
            arm_line.set_3d_properties([])
            trail.set_data([], [])
            trail.set_3d_properties([])
            dot.set_data([], [])
            dot.set_3d_properties([])
            if ori_line is not None:
                ori_line.set_data([], [])
                ori_line.set_3d_properties([])
                return arm_line, trail, dot, ori_line
            return arm_line, trail, dot

        def _update3(k):
            i = int(frame_idx[k])
            arm_line.set_data(joints[i, :, 0], joints[i, :, 1])
            arm_line.set_3d_properties(joints[i, :, 2])
            trail.set_data(ee[: i + 1, 0], ee[: i + 1, 1])
            trail.set_3d_properties(ee[: i + 1, 2])
            dot.set_data([ee[i, 0]], [ee[i, 1]])
            dot.set_3d_properties([ee[i, 2]])
            if ori_line is not None and ori is not None:
                p0 = ee[i]
                p1 = p0 + ori_scale * ori[i]
                ori_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
                ori_line.set_3d_properties([p0[2], p1[2]])
                return arm_line, trail, dot, ori_line
            return arm_line, trail, dot

        ani = animation.FuncAnimation(
            fig2,
            _update3,
            init_func=_init3,
            frames=len(frame_idx),
            interval=max(1, int(round(1000.0 / max(1, int(_pln(cfg, "anim_fps", 6)))))),
            blit=False,
        )
        try:
            ani.save(out_gif, writer=animation.PillowWriter(fps=max(1, int(_pln(cfg, "anim_fps", 6)))))
            print(f"saved: {out_gif}")
        except Exception as e:
            print(f"[warn] {name}: failed to save gif: {e}")
        plt.close(fig2)
        return q_paths_render

    fig2, ax = plt.subplots(figsize=(6.2, 6.2))
    reach = float(sum(lengths)) + 0.15
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{name}: workspace motion (animation)")
    ax.scatter([ee[0, 0]], [ee[0, 1]], c="blue", s=42, label="start ee")
    ax.scatter([ee[-1, 0]], [ee[-1, 1]], c="red", s=42, label="goal ee")
    arm_line, = ax.plot([], [], "-", color="black", lw=2.0, alpha=0.9, label="arm")
    trail, = ax.plot([], [], "-", color="green", lw=2.0, alpha=0.9, label="ee trail")
    dot, = ax.plot([], [], "o", color="green", ms=6)
    ax.legend(loc="best", fontsize=8)

    frame_idx = np.arange(0, len(joints), max(1, int(_pln(cfg, "anim_stride", 1))), dtype=int)
    if frame_idx[-1] != len(joints) - 1:
        frame_idx = np.concatenate([frame_idx, np.array([len(joints) - 1], dtype=int)])

    def _init():
        arm_line.set_data([], [])
        trail.set_data([], [])
        dot.set_data([], [])
        return arm_line, trail, dot

    def _update(k):
        i = int(frame_idx[k])
        arm_line.set_data(joints[i, :, 0], joints[i, :, 1])
        trail.set_data(ee[: i + 1, 0], ee[: i + 1, 1])
        dot.set_data([ee[i, 0]], [ee[i, 1]])
        return arm_line, trail, dot

    ani = animation.FuncAnimation(
        fig2,
        _update,
        init_func=_init,
        frames=len(frame_idx),
        interval=max(1, int(round(1000.0 / max(1, int(_pln(cfg, "anim_fps", 6)))))),
        blit=True,
    )
    try:
        ani.save(out_gif, writer=animation.PillowWriter(fps=max(1, int(_pln(cfg, "anim_fps", 6)))))
        print(f"saved: {out_gif}")
    except Exception as e:
        print(f"[warn] {name}: failed to save gif: {e}")
    plt.close(fig2)
    return q_paths_render
def _render_ur5_pybullet_trajectories(q_paths: list[np.ndarray], cfg: Any) -> None:
    if not q_paths:
        return
    kin_ov = {}
    for k, a in (("urdf_path", "ur5_urdf_path"), ("ee_link_index", "ur5_ee_link_index"), ("tool_axis", "ur5_tool_axis")):
        if hasattr(cfg, a):
            kin_ov[k] = getattr(cfg, a)
    ren_ov = {}
    for k, a in (
        ("grasp_offset", "ur5_grasp_offset"),
        ("grasp_axis_shift", "ur5_grasp_axis_shift"),
        ("debug_axes", "ur5_debug_axes"),
        ("cylinder_rotate_90", "ur5_cylinder_rotate_90"),
        ("gripper_close_ratio", "ur5_gripper_close_ratio"),
    ):
        if hasattr(cfg, a):
            ren_ov[k] = getattr(cfg, a)
    kin = resolve_ur5_kinematics_cfg(kin_ov)
    ren = resolve_ur5_render_cfg(ren_ov)

    urdf_path = str(kin["urdf_path"]).strip()
    if not urdf_path:
        print("[warn] skip pybullet render: urdf_path is empty")
        return
    try:
        import pybullet as p  # type: ignore
        import pybullet_data  # type: ignore
    except Exception as e:
        print(f"[warn] skip pybullet render: {e}")
        return

    cid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=cid)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.7, cameraYaw=48.0, cameraPitch=-28.0, cameraTargetPosition=[0.0, 0.0, 0.5], physicsClientId=cid
    )
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    floor = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.02], useFixedBase=True, physicsClientId=cid)
    p.changeVisualShape(floor, -1, rgbaColor=[0.96, 0.97, 0.99, 1.0], physicsClientId=cid)
    try:
        rid = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)
    except Exception:
        patched = _make_pybullet_friendly_urdf(urdf_path)
        rid = p.loadURDF(patched, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)

    rev = []
    nj = p.getNumJoints(rid, physicsClientId=cid)
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        if int(info[2]) == p.JOINT_REVOLUTE:
            rev.append(int(j))
    if len(rev) < 6:
        print(f"[warn] skip pybullet render: URDF has only {len(rev)} revolute joints")
        p.disconnect(physicsClientId=cid)
        return
    arm = rev[:6]
    ee_raw = int(kin["ee_link_index"])
    ee_idx = ee_raw if ee_raw >= 0 else pick_default_ee_link_index(rid, arm[-1], cid)
    try:
        ee_info = p.getJointInfo(rid, ee_idx, physicsClientId=cid)
        ee_name = ee_info[12].decode("utf-8", errors="ignore")
        print(f"[pybullet] ee link index={ee_idx}, name={ee_name}")
    except Exception:
        pass
    tool_axis_name = str(kin["tool_axis"]).strip().lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(tool_axis_name, 2)
    grasp_offset = float(ren["grasp_offset"])
    grasp_axis_shift = float(ren["grasp_axis_shift"])
    debug_axes = bool(ren["debug_axes"])
    cyl_rotate_90 = bool(ren["cylinder_rotate_90"])

    cyl_len = 0.22
    cyl_rad = 0.03
    cyl_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(cyl_rad),
        length=float(cyl_len),
        rgbaColor=[0.95, 0.75, 0.15, 0.9],
        physicsClientId=cid,
    )
    cyl_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=int(cyl_vis),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=cid,
    )
    top_line_id = -1
    # Try to anchor object by actual fingertip geometry (3-finger gripper).
    name_to_idx: dict[str, int] = {}
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        lname = info[12].decode("utf-8", errors="ignore")
        name_to_idx[lname] = int(j)
    tip_names = [
        "gripperfinger_1_link_3",
        "gripperfinger_2_link_3",
        "gripperfinger_middle_link_3",
    ]
    tip_idx = [name_to_idx[nm] for nm in tip_names if nm in name_to_idx]
    gripper_joint_idx = None
    gripper_joint_limits = (-1.0, 1.0)
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        jname = info[1].decode("utf-8", errors="ignore")
        if "gripperrobotiq_hand_joint" in jname:
            gripper_joint_idx = int(j)
            lo = float(info[8])
            hi = float(info[9])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                gripper_joint_limits = (lo, hi)
            break
    if debug_axes:
        print("[pybullet] link indices (subset):")
        for nm in ("base_link", "wrist_3_link", "tool0", "ee_link", "gripperpalm",
                   "gripperfinger_1_link_3", "gripperfinger_2_link_3", "gripperfinger_middle_link_3"):
            if nm in name_to_idx:
                print(f"  - {nm}: {name_to_idx[nm]}")
        print(f"[pybullet] using ee_idx={ee_idx}, tool_axis={tool_axis_name}")

    def _axis_vec_from_quat(quat_xyzw: list[float]) -> np.ndarray:
        mat = np.asarray(p.getMatrixFromQuaternion(quat_xyzw), dtype=np.float32).reshape(3, 3)
        v = mat[:, int(axis_idx)]
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _orient_cylinder_quat(ee_quat: list[float]) -> list[float]:
        # PyBullet cylinder local axis is +Z. Rotate it to requested tool axis, then apply ee orientation.
        if axis_idx == 2:  # z
            q_off = [0.0, 0.0, 0.0, 1.0]
        elif axis_idx == 0:  # x
            q_off = p.getQuaternionFromEuler([0.0, float(np.pi / 2.0), 0.0])
        else:  # y
            q_off = p.getQuaternionFromEuler([float(-np.pi / 2.0), 0.0, 0.0])
        _, q = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat, [0.0, 0.0, 0.0], q_off)
        return list(q)

    def _quat_from_axis(axis: np.ndarray, ref_axis: np.ndarray) -> list[float]:
        # Build quaternion whose local +Z aligns with axis, and x is stabilized by ref_axis.
        z = axis / max(float(np.linalg.norm(axis)), 1e-8)
        x0 = ref_axis - np.dot(ref_axis, z) * z
        if np.linalg.norm(x0) < 1e-8:
            x0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x0 = x0 - np.dot(x0, z) * z
        x = x0 / max(float(np.linalg.norm(x0)), 1e-8)
        y = np.cross(z, x)
        y = y / max(float(np.linalg.norm(y)), 1e-8)
        x = np.cross(y, z)
        R = np.stack([x, y, z], axis=1).astype(np.float32)
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return [float(qx), float(qy), float(qz), float(qw)]

    def _apply_cylinder_extra_rotation(quat_xyzw: np.ndarray) -> np.ndarray:
        if not cyl_rotate_90:
            return quat_xyzw
        # Rotate cylinder by +90 deg around its local X so gripper tends to contact curved side.
        q_extra = p.getQuaternionFromEuler([float(np.pi / 2.0), 0.0, 0.0])
        _, q_new = p.multiplyTransforms([0.0, 0.0, 0.0], quat_xyzw.tolist(), [0.0, 0.0, 0.0], q_extra)
        return np.asarray(q_new, dtype=np.float32)

    def _cyl_axis_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
        mat = np.asarray(p.getMatrixFromQuaternion(quat_xyzw.tolist()), dtype=np.float32).reshape(3, 3)
        v = mat[:, 2]  # cylinder local Z
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _tip_center() -> np.ndarray | None:
        if len(tip_idx) < 3:
            return None
        pts = []
        for li in tip_idx[:3]:
            ls = p.getLinkState(rid, li, computeForwardKinematics=True, physicsClientId=cid)
            pts.append(np.asarray(ls[4], dtype=np.float32))
        return (pts[0] + pts[1] + pts[2]) / 3.0

    def _cylinder_pose_from_gripper(ee_pos: np.ndarray, ee_quat: list[float]) -> tuple[np.ndarray, np.ndarray]:
        # Preferred: infer from fingertip geometry.
        tool = _axis_vec_from_quat(ee_quat)
        if len(tip_idx) >= 3:
            pts = []
            for li in tip_idx[:3]:
                ls = p.getLinkState(rid, li, computeForwardKinematics=True, physicsClientId=cid)
                pts.append(np.asarray(ls[4], dtype=np.float32))
            center = (pts[0] + pts[1] + pts[2]) / 3.0
            center = center + tool * float(grasp_axis_shift)
            q = np.asarray(_quat_from_axis(tool, tool), dtype=np.float32)
            q = _apply_cylinder_extra_rotation(q)
            return center, q
        # Fallback: ee frame + offset.
        center = ee_pos + tool * float(grasp_offset)
        q = np.asarray(_quat_from_axis(tool, tool), dtype=np.float32)
        q = _apply_cylinder_extra_rotation(q)
        return center, q

    dt = float(max(float(_pln(cfg, "pybullet_real_time_dt", 0.06)), 0.01))
    try:
        # Initialize to first frame before enabling rendering to avoid the startup "disconnected" flash.
        q0 = q_paths[0][0]
        for k, j in enumerate(arm):
            p.resetJointState(rid, j, float(q0[k]), targetVelocity=0.0, physicsClientId=cid)
        if gripper_joint_idx is not None:
            lo, hi = gripper_joint_limits
            r = float(np.clip(float(ren["gripper_close_ratio"]), 0.0, 1.0))
            qg = lo + r * (hi - lo)
            p.resetJointState(rid, gripper_joint_idx, qg, targetVelocity=0.0, physicsClientId=cid)
        ls0 = p.getLinkState(rid, ee_idx, computeForwardKinematics=True, physicsClientId=cid)
        ee_pos0 = np.asarray(ls0[4], dtype=np.float32)
        ee_quat0 = list(ls0[5])
        cyl_pos0, cyl_quat0 = _cylinder_pose_from_gripper(ee_pos0, ee_quat0)
        # Fix a rigid grasp offset in ee frame to avoid tiny frame-to-frame jitter.
        R0 = np.asarray(p.getMatrixFromQuaternion(ee_quat0), dtype=np.float32).reshape(3, 3)
        tip0 = _tip_center()
        if tip0 is not None:
            local_off = (R0.T @ (tip0 - ee_pos0)).astype(np.float32)
            local_off = local_off + np.array([0.0, 0.0, float(grasp_axis_shift)], dtype=np.float32)
        else:
            local_off = np.array([0.0, 0.0, float(grasp_offset)], dtype=np.float32)
        q_ref = np.asarray(_orient_cylinder_quat(ee_quat0), dtype=np.float32)
        q_ref = _apply_cylinder_extra_rotation(q_ref)
        _, ee_inv = p.invertTransform([0.0, 0.0, 0.0], ee_quat0)
        _, q_offset = p.multiplyTransforms([0.0, 0.0, 0.0], ee_inv, [0.0, 0.0, 0.0], q_ref.tolist())
        q_offset = np.asarray(q_offset, dtype=np.float32)
        cyl_pos0 = ee_pos0 + (R0 @ local_off).astype(np.float32)
        _, q0w = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat0, [0.0, 0.0, 0.0], q_offset.tolist())
        p.resetBasePositionAndOrientation(cyl_id, cyl_pos0.tolist(), list(q0w), physicsClientId=cid)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=cid)
        for cidx, q_path in enumerate(q_paths):
            print(f"[pybullet] rendering case {cidx + 1}/{len(q_paths)} ...")
            for i in range(q_path.shape[0]):
                q = q_path[i]
                for k, j in enumerate(arm):
                    p.resetJointState(rid, j, float(q[k]), targetVelocity=0.0, physicsClientId=cid)
                if gripper_joint_idx is not None:
                    lo, hi = gripper_joint_limits
                    r = float(np.clip(float(ren["gripper_close_ratio"]), 0.0, 1.0))
                    qg = lo + r * (hi - lo)
                    p.resetJointState(rid, gripper_joint_idx, qg, targetVelocity=0.0, physicsClientId=cid)
                ls = p.getLinkState(rid, ee_idx, computeForwardKinematics=True, physicsClientId=cid)
                ee_pos = np.asarray(ls[4], dtype=np.float32)
                ee_quat = list(ls[5])  # xyzw
                R = np.asarray(p.getMatrixFromQuaternion(ee_quat), dtype=np.float32).reshape(3, 3)
                cyl_pos = ee_pos + (R @ local_off).astype(np.float32)
                _, q_cur = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat, [0.0, 0.0, 0.0], q_offset.tolist())
                cyl_quat = np.asarray(q_cur, dtype=np.float32)
                axis = _cyl_axis_from_quat(cyl_quat)
                p.resetBasePositionAndOrientation(cyl_id, cyl_pos.tolist(), cyl_quat.tolist(), physicsClientId=cid)

                p0 = cyl_pos + axis * (0.5 * cyl_len)
                p1 = cyl_pos + axis * (0.5 * cyl_len + 0.18)
                top_line_id = p.addUserDebugLine(
                    p0.tolist(),
                    p1.tolist(),
                    [1.0, 0.1, 0.1],
                    lineWidth=4.0,
                    lifeTime=0.0,
                    replaceItemUniqueId=int(top_line_id),
                    physicsClientId=cid,
                )
                if debug_axes:
                    mat = np.asarray(p.getMatrixFromQuaternion(ee_quat), dtype=np.float32).reshape(3, 3)
                    o = ee_pos
                    lx = o + 0.10 * mat[:, 0]
                    ly = o + 0.10 * mat[:, 1]
                    lz = o + 0.10 * mat[:, 2]
                    p.addUserDebugLine(o.tolist(), lx.tolist(), [1, 0, 0], 2.0, 1e-3, physicsClientId=cid)
                    p.addUserDebugLine(o.tolist(), ly.tolist(), [0, 1, 0], 2.0, 1e-3, physicsClientId=cid)
                    p.addUserDebugLine(o.tolist(), lz.tolist(), [0, 0, 1], 2.0, 1e-3, physicsClientId=cid)
                time.sleep(dt)
            time.sleep(0.6)
    finally:
        p.disconnect(physicsClientId=cid)
