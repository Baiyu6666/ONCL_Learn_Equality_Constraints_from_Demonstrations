#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from experiments.dataset_resolve import resolve_dataset
from planning.kinematics import spatial_fk
from datasets.ur5_pybullet_utils import UR5_LINK_LENGTHS


TRAJ_DATASETS = [
    # "3d_paraboloid_traj",
    # "3d_twosphere_traj",
    # "3d_saddle_surface_traj",
    # "3d_sphere_surface_traj",
    # "3d_torus_surface_traj",
    # "3d_planar_arm_line_n3_traj",
    # "3d_spatial_arm_ellip_n3_traj",
    # "3d_vz_2d_ellipse_traj",
    "6d_spatial_arm_up_n6_py_traj",
]


def _project_root() -> str:
    # tools/preview_3d_trajectory_datasets.py -> repo root is ../
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_config_root(config_root: str) -> str:
    p = str(config_root).strip() or "configs"
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    # Fallback: resolve relative to project root (works when cwd != repo root).
    return os.path.join(_project_root(), p)


def _resolve_outdir(outdir: str) -> str:
    p = str(outdir).strip() or "outputs/analysis/traj_preview_3d"
    if os.path.isabs(p):
        return p
    return os.path.join(_project_root(), p)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview 3D traj datasets with GT surface/grid.")
    p.add_argument("--config-root", default="configs", help="Config root containing datasets/*.json")
    p.add_argument("--outdir", default="outputs/analysis/traj_preview_3d", help="Output directory")
    p.add_argument("--seed", type=int, default=127 , help="Override dataset seed for preview")
    p.add_argument("--max-grid", type=int, default=2500, help="Max GT grid points to draw")
    return p.parse_args()


def _load_cfg(config_root: str, dataset_name: str, seed: int) -> SimpleNamespace:
    path = os.path.join(config_root, "datasets", f"{dataset_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["seed"] = int(seed)
    return SimpleNamespace(**cfg)


def _segments_by_generation_rule(points: np.ndarray, traj_count: int, traj_len: int) -> list[np.ndarray]:
    p = np.asarray(points, dtype=np.float32)
    n = int(p.shape[0])
    k = max(1, int(traj_count))
    tlen = max(2, int(traj_len))
    segs: list[np.ndarray] = []
    for i in range(k):
        a = i * tlen
        if a >= n:
            break
        b = min(n, (i + 1) * tlen)
        if b - a < 2:
            continue
        segs.append(p[a:b].astype(np.float32))
    if not segs and n >= 2:
        segs = [p]
    return segs


def _jump_cut_indices(points: np.ndarray, jump_ratio: float = 3.0) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    if len(p) < 3:
        return np.zeros((0,), dtype=np.int64)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    med = float(np.median(step))
    if (not np.isfinite(med)) or med <= 1e-9:
        med = float(np.mean(step) + 1e-9)
    thr = max(float(med * jump_ratio), float(np.percentile(step, 95) * 1.5))
    return np.where(step > thr)[0].astype(np.int64)


def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0])
    kk = int(max(2, min(k, n)))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(pts.astype(np.float64))
        _, idx = tree.query(pts.astype(np.float64), k=kk)
        if idx.ndim == 1:
            idx = idx[:, None]
        return idx.astype(np.int64)
    except Exception:
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        return idx.astype(np.int64)


def _angle_embed(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float32)
    return np.concatenate([np.sin(qq), np.cos(qq)], axis=1).astype(np.float32)


def _nearest_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(b.astype(np.float64))
        _, idx = tree.query(a.astype(np.float64), k=1)
        return np.asarray(idx, dtype=np.int64)
    except Exception:
        aa = a.astype(np.float32)
        bb = b.astype(np.float32)
        d2 = np.sum((aa[:, None, :] - bb[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)


def _plot_one_3d(
    *,
    name: str,
    x_train: np.ndarray,
    grid: np.ndarray,
    traj_count: int,
    traj_len: int,
    out_path: str,
    max_grid: int,
) -> None:
    rng = np.random.default_rng(0)
    g_all = grid.astype(np.float32)
    g = g_all
    if len(g) > int(max_grid):
        idx = rng.choice(len(g), size=int(max_grid), replace=False)
        g = g[idx]

    fig = plt.figure(figsize=(10.5, 5.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Left: GT manifold surface (preferred) + dataset scatter overlay.
    # Fallback to GT scatter when triangulated surface is ill-conditioned.
    g_surf = grid.astype(np.float32)
    if len(g_surf) > 1800:
        idx = rng.choice(len(g_surf), size=1800, replace=False)
        g_surf = g_surf[idx]
    surface_drawn = False
    try:
        ax1.plot_trisurf(
            g_surf[:, 0],
            g_surf[:, 1],
            g_surf[:, 2],
            color="#A7C7E7",
            alpha=0.38,
            linewidth=0.12,
            edgecolor=(0.45, 0.55, 0.70, 0.35),
            antialiased=True,
        )
        surface_drawn = True
    except Exception:
        surface_drawn = False
    if not surface_drawn:
        ax1.scatter(g[:, 0], g[:, 1], g[:, 2], s=1.6, c="#9CA3AF", alpha=0.28, linewidths=0)
    ax1.scatter(
        x_train[:, 0], x_train[:, 1], x_train[:, 2],
        s=4.0, c="#1D4ED8", alpha=0.55, linewidths=0
    )
    ax1.set_title(f"{name} - GT manifold + dataset scatter")
    ax1.set_xlabel("x/q1")
    ax1.set_ylabel("y/q2")
    ax1.set_zlabel("z/q3")

    # Right: trajectory line plot
    # Draw full grid used for trajectory generation (no subsampling) for debugging.
    ax2.scatter(g_all[:, 0], g_all[:, 1], g_all[:, 2], s=3.2, c="#9CA3AF", alpha=0.32, linewidths=0)
    colors = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2"]
    shown_jump_label = False
    segs = _segments_by_generation_rule(x_train, traj_count=int(traj_count), traj_len=int(traj_len))
    for i, tr in enumerate(segs):
        if len(tr) < 2:
            continue
        c = colors[i % len(colors)]
        ax2.plot(tr[:, 0], tr[:, 1], tr[:, 2], "-", color=c, linewidth=1.3, alpha=0.9)
        ax2.scatter(tr[:, 0], tr[:, 1], tr[:, 2], s=9, c=c, alpha=0.85, linewidths=0)
        ax2.scatter([tr[0, 0]], [tr[0, 1]], [tr[0, 2]], s=14, c=c, alpha=0.95)
        jump_idx = _jump_cut_indices(tr, jump_ratio=3.0)
        for ji in jump_idx:
            p = tr[int(ji) + 1].astype(np.float32)
            ax2.scatter(
                [p[0]], [p[1]], [p[2]],
                s=46, c="#F59E0B", edgecolors="black", linewidths=0.7, zorder=8,
                label=("jump point" if not shown_jump_label else None),
            )
            shown_jump_label = True
    ax2.set_title(f"{name} - trajectory lines + full grid (n={len(g_all)})")
    ax2.set_xlabel("x/q1")
    ax2.set_ylabel("y/q2")
    ax2.set_zlabel("z/q3")
    if shown_jump_label:
        ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_one_6d_arm(
    *,
    name: str,
    x_train: np.ndarray,
    traj_count: int,
    traj_len: int,
    out_path: str,
) -> None:
    q = x_train.astype(np.float32)
    joints = spatial_fk(q, list(UR5_LINK_LENGTHS), use_pybullet_n6=False)
    ee = joints[:, -1, :]

    fig = plt.figure(figsize=(10.8, 5.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.scatter(ee[:, 0], ee[:, 1], ee[:, 2], s=4.0, c="#1D4ED8", alpha=0.55, linewidths=0)
    ax1.set_title(f"{name} - workspace EE scatter")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    colors = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2"]
    segs_ee = _segments_by_generation_rule(ee, traj_count=int(traj_count), traj_len=int(traj_len))
    for i, seg in enumerate(segs_ee):
        if len(seg) < 2:
            continue
        c = colors[i % len(colors)]
        ax2.plot(
            seg[:, 0], seg[:, 1], seg[:, 2], "-",
            color=c, lw=1.5, alpha=0.92,
        )
        ax2.scatter(seg[:, 0], seg[:, 1], seg[:, 2], s=10, c=c, alpha=0.88, linewidths=0)
        ax2.scatter([seg[0, 0]], [seg[0, 1]], [seg[0, 2]], s=14, c=c, alpha=0.95)

    ax2.set_title(f"{name} - workspace trajectories")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_6d_jump_knn_debug(
    *,
    name: str,
    x_train: np.ndarray,
    grid: np.ndarray,
    traj_count: int,
    traj_len: int,
    traj_knn: int,
    out_path: str,
) -> None:
    q = np.asarray(x_train, dtype=np.float32)
    gq = np.asarray(grid, dtype=np.float32)
    if len(q) < 2 or len(gq) < 2:
        return
    ee = spatial_fk(q, list(UR5_LINK_LENGTHS), use_pybullet_n6=False)[:, -1, :]
    ee_grid = spatial_fk(gq, list(UR5_LINK_LENGTHS), use_pybullet_n6=False)[:, -1, :]
    knn = _knn_indices(_angle_embed(gq), k=max(2, int(traj_knn)))
    q_to_g = _nearest_index(q, gq)

    segs_q = _segments_by_generation_rule(q, traj_count=int(traj_count), traj_len=int(traj_len))
    segs_ee = _segments_by_generation_rule(ee, traj_count=int(traj_count), traj_len=int(traj_len))
    step_d = []
    for seg in segs_ee:
        if len(seg) >= 2:
            step_d.append(np.linalg.norm(np.diff(seg, axis=0), axis=1))
    if not step_d:
        return
    step_all = np.concatenate(step_d, axis=0).astype(np.float32)
    med = float(np.median(step_all))
    if (not np.isfinite(med)) or med <= 1e-9:
        med = float(np.mean(step_all) + 1e-9)
    thr = max(float(med * 3.0), float(np.percentile(step_all, 95) * 1.5))

    fig = plt.figure(figsize=(9.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2"]
    for i, seg in enumerate(segs_ee):
        if len(seg) < 2:
            continue
        c = colors[i % len(colors)]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], "-", color=c, linewidth=1.0, alpha=0.40)

    events: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cursor = 0
    for seg_q, seg_ws in zip(segs_q, segs_ee):
        if len(seg_q) < 2:
            cursor += len(seg_q)
            continue
        prev_idx = -1
        seg_idx = q_to_g[cursor : cursor + len(seg_q)]
        for j in range(len(seg_q) - 1):
            cur_idx = int(seg_idx[j])
            nxt_ws = seg_ws[j + 1]
            cur_ws = seg_ws[j]
            d = float(np.linalg.norm(nxt_ws - cur_ws))
            nbr = knn[cur_idx]
            nbr = nbr[nbr != cur_idx]
            if prev_idx >= 0:
                nbr = nbr[nbr != prev_idx]
            cand = nbr[: min(14, len(nbr))]
            if d > thr:
                ws_c = ee_grid[cand] if len(cand) > 0 else np.zeros((0, 3), dtype=np.float32)
                events.append((cur_ws.astype(np.float32), nxt_ws.astype(np.float32), ws_c.astype(np.float32)))
            prev_idx = cur_idx
        cursor += len(seg_q)

    # Ensure visible debug markers even when no step exceeds threshold.
    if len(events) == 0:
        flat: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        cursor = 0
        for seg_q, seg_ws in zip(segs_q, segs_ee):
            if len(seg_q) < 2:
                cursor += len(seg_q)
                continue
            prev_idx = -1
            seg_idx = q_to_g[cursor : cursor + len(seg_q)]
            for j in range(len(seg_q) - 1):
                cur_idx = int(seg_idx[j])
                cur_ws = seg_ws[j]
                nxt_ws = seg_ws[j + 1]
                d = float(np.linalg.norm(nxt_ws - cur_ws))
                nbr = knn[cur_idx]
                nbr = nbr[nbr != cur_idx]
                if prev_idx >= 0:
                    nbr = nbr[nbr != prev_idx]
                cand = nbr[: min(14, len(nbr))]
                ws_c = ee_grid[cand] if len(cand) > 0 else np.zeros((0, 3), dtype=np.float32)
                flat.append((d, cur_ws.astype(np.float32), nxt_ws.astype(np.float32), ws_c.astype(np.float32)))
                prev_idx = cur_idx
            cursor += len(seg_q)
        flat.sort(key=lambda t: t[0], reverse=True)
        for _, cur_ws, nxt_ws, ws_c in flat[:6]:
            events.append((cur_ws, nxt_ws, ws_c))

    shown_jump = False
    shown_knn = False
    shown_edge = False
    for cur_ws, nxt_ws, ws_c in events[:48]:
        ax.scatter(
            [nxt_ws[0]], [nxt_ws[1]], [nxt_ws[2]],
            s=64, c="#EF4444", edgecolors="black", linewidths=0.9, zorder=8,
            label=("jump point" if not shown_jump else None),
        )
        shown_jump = True
        if len(ws_c) > 0:
            ax.scatter(
                ws_c[:, 0], ws_c[:, 1], ws_c[:, 2],
                s=36, c="#F59E0B", alpha=0.95, linewidths=0,
                label=("knn candidates" if not shown_knn else None),
            )
            shown_knn = True
        ax.plot(
            [cur_ws[0], nxt_ws[0]], [cur_ws[1], nxt_ws[1]], [cur_ws[2], nxt_ws[2]],
            "--", color="#EF4444", linewidth=1.4, alpha=0.95,
            label=("jump edge" if not shown_edge else None),
        )
        shown_edge = True

    ax.set_title(f"{name} - workspace jump+knn debug (k={int(traj_knn)}, n={len(events)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if shown_jump or shown_knn or shown_edge:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config_root = _resolve_config_root(args.config_root)
    outdir = _resolve_outdir(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    for ds_name in TRAJ_DATASETS:
        cfg = _load_cfg(config_root, ds_name, seed=int(args.seed))
        ds = resolve_dataset(ds_name, cfg, optimize_ur5_train_only=False)
        x_train = np.asarray(ds["x_train"], dtype=np.float32)
        grid = np.asarray(ds["grid"], dtype=np.float32)
        traj_count = int(getattr(cfg, "traj_count", max(8, int(cfg.n_train) // 24)))
        traj_len = int(getattr(cfg, "traj_len", max(8, int(math.ceil(len(x_train) / float(max(traj_count, 1)))))))
        out_path = os.path.join(outdir, f"{ds_name}_preview.png")
        if int(x_train.shape[1]) == 6 and str(ds_name) in ("6d_spatial_arm_up_n6_py_traj",):
            _plot_one_6d_arm(
                name=ds_name,
                x_train=x_train,
                traj_count=traj_count,
                traj_len=traj_len,
                out_path=out_path,
            )
            dbg_path = os.path.join(outdir, f"{ds_name}_jump_knn_debug.png")
            _plot_6d_jump_knn_debug(
                name=ds_name,
                x_train=x_train,
                grid=grid,
                traj_count=traj_count,
                traj_len=traj_len,
                traj_knn=int(getattr(cfg, "traj_knn", 24)),
                out_path=dbg_path,
            )
            print(f"[saved] {dbg_path}")
        else:
            _plot_one_3d(
                name=ds_name,
                x_train=x_train,
                grid=grid,
                traj_count=traj_count,
                traj_len=traj_len,
                out_path=out_path,
                max_grid=int(args.max_grid),
            )
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
