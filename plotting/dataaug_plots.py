from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from evaluation.evaluator import compute_eps_stop, eval_bounds_from_train


def _pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, copy=False).astype(np.float64)
    b = np.nan_to_num(b, copy=False).astype(np.float64)
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def _knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    n = int(x.shape[0])
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    k_eff = max(1, min(int(k), n - 1))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.asarray(x, dtype=np.float64))
        idx = tree.query(np.asarray(x, dtype=np.float64), k=k_eff + 1)[1]
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[:, None]
        return idx[:, 1:]
    except Exception:
        d2 = _pairwise_sqdist(x, x)
        return np.argsort(d2, axis=1)[:, 1 : k_eff + 1].astype(np.int64)


def _as_sigmas(sigmas: Tuple[float, ...] | float) -> np.ndarray:
    if isinstance(sigmas, (int, float)):
        return np.array([float(sigmas)], dtype=np.float32)
    return np.array(sigmas, dtype=np.float32)


def _effective_knn_norm_estimation_points(cfg: Any, n: int) -> int:
    return max(cfg.knn_norm_estimation_min_points, int(round(cfg.knn_norm_estimation_ratio * n)))


def _effective_knn_off_data_filter_points(cfg: Any, n: int) -> int:
    return max(cfg.knn_off_data_filter_min_points, int(round(cfg.knn_off_data_filter_ratio * n)))


def _local_pca_frame(neigh: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(neigh) == 0:
        d = center.shape[0]
        return np.zeros((d,), dtype=np.float64), np.eye(d, dtype=np.float64), np.ones((1,), dtype=np.float64), neigh
    mu = np.mean(neigh, axis=0, keepdims=True)
    xc = neigh - mu
    cov = (xc.T @ xc) / max(len(neigh), 1)
    evals, evecs = np.linalg.eigh(cov)
    w = np.full((len(neigh),), 1.0 / max(len(neigh), 1), dtype=np.float64)
    return evals, evecs, w, neigh


def _knn_normals(x: np.ndarray, k: int) -> np.ndarray:
    n, d = x.shape
    nbr_idx_all = _knn_indices(x, k)
    normals = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        nbr_idx = nbr_idx_all[i]
        nbrs = x[nbr_idx]
        _, evecs, _, _ = _local_pca_frame(nbrs, x[i])
        nvec = evecs[:, 0]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        normals[i] = nvec.astype(np.float32)
    return normals


def _true_distance(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        d2 = _pairwise_sqdist(x, grid)
        return np.sqrt(np.min(np.maximum(d2, 0.0), axis=1))
    tree = cKDTree(grid)
    d, _ = tree.query(x, k=1)
    return d


def plot_planned_paths(
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    plans_proj: List[np.ndarray],
    plans_constr: List[np.ndarray],
    cfg: Any,
    out_path: str,
    title: str,
    zero_level_eps: float,
) -> None:
    if x_train.shape[1] != 2:
        return
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    xx, yy = np.meshgrid(np.linspace(-4.0, 4.0, 300), np.linspace(-3.0, 3.0, 300))
    grid_eval = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid_eval).to(cfg.device)
    with torch.no_grad():
        f_raw = model(grid_t).detach().cpu().numpy()
        if f_raw.ndim == 1:
            f_grid = f_raw.reshape(xx.shape)
            near_levels = [-float(zero_level_eps), float(zero_level_eps)]
        elif f_raw.ndim == 2 and f_raw.shape[1] == 1:
            f_grid = f_raw[:, 0].reshape(xx.shape)
            near_levels = [-float(zero_level_eps), float(zero_level_eps)]
        else:
            # Vector-valued residual field (e.g. VAE): use residual norm for contouring.
            f_grid = np.linalg.norm(f_raw, axis=1).reshape(xx.shape)
            near_levels = [0.0, float(zero_level_eps)]
        v = 0.5 * (f_grid ** 2)
    v_max = float(np.percentile(v, 95))
    levels = (np.linspace(0.0, 1.0, 25) ** 3) * v_max

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    plt.contourf(xx, yy, f_grid, levels=near_levels, colors=["#ffa500"], alpha=0.55)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=4, alpha=0.6, label="data", zorder=3)
    cmap = plt.get_cmap("tab10")
    all_trajs = [("projected (on)", "-", t) for t in plans_proj] + [("constrained (on)", "--", t) for t in plans_constr]
    seen_labels: set[str] = set()
    for i, (label, style, traj) in enumerate(all_trajs):
        c = cmap(i % 10)
        leg = label if label not in seen_labels else None
        seen_labels.add(label)
        plt.plot(traj[:, 0], traj[:, 1], style, color=c, linewidth=2.0, label=leg)
        plt.scatter(traj[0, 0], traj[0, 1], c=[c], s=25, zorder=4)
        plt.scatter(traj[-1, 0], traj[-1, 1], c=[c], s=25, zorder=4)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_planned_paths_off(
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    plans_proj_off: List[np.ndarray],
    plans_constr_off: List[np.ndarray],
    cfg: Any,
    out_path: str,
    title: str,
    zero_level_eps: float,
) -> None:
    if x_train.shape[1] != 2:
        return
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    xx, yy = np.meshgrid(np.linspace(-4.0, 4.0, 300), np.linspace(-3.0, 3.0, 300))
    grid_eval = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid_eval).to(cfg.device)
    with torch.no_grad():
        f_raw = model(grid_t).detach().cpu().numpy()
        if f_raw.ndim == 1:
            f_grid = f_raw.reshape(xx.shape)
            near_levels = [-float(zero_level_eps), float(zero_level_eps)]
        elif f_raw.ndim == 2 and f_raw.shape[1] == 1:
            f_grid = f_raw[:, 0].reshape(xx.shape)
            near_levels = [-float(zero_level_eps), float(zero_level_eps)]
        else:
            # Vector-valued residual field (e.g. VAE): use residual norm for contouring.
            f_grid = np.linalg.norm(f_raw, axis=1).reshape(xx.shape)
            near_levels = [0.0, float(zero_level_eps)]
        v = 0.5 * (f_grid ** 2)
    v_max = float(np.percentile(v, 95))
    levels = (np.linspace(0.0, 1.0, 25) ** 3) * v_max

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    plt.contourf(xx, yy, f_grid, levels=near_levels, colors=["#ffa500"], alpha=0.55)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=4, alpha=0.6, label="data", zorder=3)
    for i, traj in enumerate(plans_proj_off):
        plt.plot(traj[:, 0], traj[:, 1], ":", color="gold", linewidth=2.0, label="projected (off)" if i == 0 else None)
        plt.scatter(traj[0, 0], traj[0, 1], c="gold", s=25, zorder=4, marker="o")
        plt.scatter(traj[-1, 0], traj[-1, 1], c="gold", s=40, zorder=4, marker="x")
    for i, traj in enumerate(plans_constr_off):
        plt.plot(traj[:, 0], traj[:, 1], "-.", color="purple", linewidth=2.0, label="constrained (off)" if i == 0 else None)
        plt.scatter(traj[0, 0], traj[0, 1], c="purple", s=25, zorder=4, marker="o")
        plt.scatter(traj[-1, 0], traj[-1, 1], c="purple", s=40, zorder=4, marker="x")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _draw_sigma_interval_segments_on_contour(x_train: np.ndarray, cfg: Any, n_segments: int = 15) -> None:
    if x_train.shape[1] != 2 or len(x_train) == 0:
        return
    n_train = len(x_train)
    k = _effective_knn_norm_estimation_points(cfg, n_train)
    n_hat = _knn_normals(x_train, k)
    base = float(np.max(_as_sigmas(cfg.sigmas)))
    r_pos = np.full(n_train, base, dtype=np.float32)
    r_neg = np.full(n_train, base, dtype=np.float32)
    idx_list = [min(n_train - 1, int(round((n_train - 1) * r))) for r in np.linspace(0.05, 0.95, max(1, int(n_segments)))]
    for idx in idx_list:
        nvec = n_hat[idx] / (np.linalg.norm(n_hat[idx]) + 1e-12)
        p_pos = x_train[idx] + nvec * float(r_pos[idx])
        p_neg = x_train[idx] - nvec * float(r_neg[idx])
        plt.plot([p_neg[0], p_pos[0]], [p_neg[1], p_pos[1]], color="gray", linewidth=1.0, alpha=0.95, zorder=2)
        plt.scatter([p_neg[0], p_pos[0]], [p_neg[1], p_pos[1]], s=4, color="black", alpha=0.95, zorder=2)


def plot_contour_only(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    out_path: str,
    title: str,
    grid: np.ndarray | None = None,
    zero_level_eps: float | None = None,
) -> None:
    if x_train.shape[1] != 2:
        return
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        f = model(torch.from_numpy(pts).to(cfg.device)).cpu().numpy().reshape(xx.shape)
    level = compute_eps_stop(model, x_train, cfg) if zero_level_eps is None else float(zero_level_eps)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, np.abs(f), levels=32, cmap="viridis")
    if grid is not None and grid.shape[1] == 2:
        plt.scatter(grid[:, 0], grid[:, 1], s=2, c="white", alpha=0.15)
    plt.contour(xx, yy, np.abs(f), levels=[level], colors="red", linewidths=2.0)
    _draw_sigma_interval_segments_on_contour(x_train, cfg, n_segments=15)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=6, c="gray", alpha=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_distance_curves(traj: np.ndarray, grid: np.ndarray, out_path: str, title: str) -> None:
    n_steps, n_pts, _ = traj.shape
    final = traj[-1]
    d_learn = []
    d_true = []
    for k in range(n_steps):
        xk = traj[k]
        d_true.append(_true_distance(xk, grid))
        d_learn.append(np.linalg.norm(xk - final, axis=1))
    d_true = np.stack(d_true, axis=0)
    d_learn = np.stack(d_learn, axis=0)

    plt.figure(figsize=(8, 5))
    for i in range(n_pts):
        plt.plot(d_true[:, i], label=f"true d (pt{i+1})")
    for i in range(n_pts):
        plt.plot(d_learn[:, i], "--", label=f"learned d (pt{i+1})")
    plt.xlabel("iteration")
    plt.ylabel("distance")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_worst_distance_contour(
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    x0: np.ndarray,
    traj: np.ndarray,
    cfg: Any,
    out_path: str,
    title: str,
    zero_level_eps: float,
    frac: float = 0.05,
) -> None:
    if x0.shape[1] == 3:
        return
    n_pts = traj.shape[1]
    k = max(1, int(math.ceil(n_pts * frac)))
    d_final = _true_distance(traj[-1], grid)
    idx = np.argsort(d_final)[-k:]

    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    xx, yy = np.meshgrid(np.linspace(-4.0, 4.0, 300), np.linspace(-3.0, 3.0, 300))
    grid_eval = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid_eval).to(cfg.device)
    with torch.no_grad():
        f_grid = model(grid_t).cpu().numpy().reshape(xx.shape)
        v = (0.5 * (f_grid ** 2)).reshape(xx.shape)
    v_max = float(np.percentile(v, 95))
    levels = np.linspace(0.0, math.sqrt(max(v_max, 1e-12)), 15) ** 2

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    eps = float(zero_level_eps)
    plt.contourf(xx, yy, f_grid, levels=[-eps, eps], colors=["#ffa500"], alpha=0.55)
    plt.scatter(grid[:, 0], grid[:, 1], s=8, alpha=0.4, label="grid", zorder=3)
    for i in idx:
        plt.plot(traj[:, i, 0], traj[:, i, 1], "-", color="red", linewidth=0.9)
        plt.scatter(traj[-1, i, 0], traj[-1, i, 1], c="red", s=25, zorder=4)
        plt.text(traj[-1, i, 0], traj[-1, i, 1], f"{d_final[i]:.3f}", fontsize=7, color="red")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_contour_and_trajectory(
    model: nn.Module,
    x_train: np.ndarray,
    x0: np.ndarray,
    traj: np.ndarray,
    cfg: Any,
    out_path: str,
    title: str,
    zero_level_eps: float,
    grid: np.ndarray | None = None,
    eval_points: np.ndarray | None = None,
    worst_frac: float = 0.002,
    project_trajectory_fn: Callable[..., Tuple[torch.Tensor, int]] | None = None,
) -> None:
    start_marker_size = 14
    worst_x0 = x0
    worst_traj = traj
    worst_idx = np.array([], dtype=np.int64)
    if grid is not None and len(grid) > 0 and eval_points is not None and len(eval_points) > 0 and project_trajectory_fn is not None:
        eval_t = torch.from_numpy(eval_points).to(cfg.device)
        worst_traj_t, _ = project_trajectory_fn(model, eval_t, cfg, f_abs_stop=float(zero_level_eps))
        worst_traj = worst_traj_t.detach().cpu().numpy()
        worst_x0 = eval_points
        n_worst = max(1, int(math.ceil(len(eval_points) * float(worst_frac))))
        d_final = _true_distance(worst_traj[-1], grid)
        worst_idx = np.argsort(d_final)[-n_worst:]

    if x_train.shape[1] == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        if grid is not None:
            m = int(math.sqrt(len(grid)))
            if m * m == len(grid):
                gx = grid[:, 0].reshape(m, m)
                gy = grid[:, 1].reshape(m, m)
                gz = grid[:, 2].reshape(m, m)
                ax.plot_surface(gx, gy, gz, color="lightgray", alpha=0.25, linewidth=0)
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=8, alpha=0.5, c="gray")
        for i in range(traj.shape[1]):
            ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], color="green", linewidth=0.8)
            ax.scatter(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], color="green", s=8, alpha=0.8)
            ax.scatter(x0[i, 0], x0[i, 1], x0[i, 2], color="green", s=start_marker_size)
        for i in worst_idx:
            ax.plot(worst_traj[:, i, 0], worst_traj[:, i, 1], worst_traj[:, i, 2], color="red", linewidth=0.9)
            ax.scatter(worst_traj[:, i, 0], worst_traj[:, i, 1], worst_traj[:, i, 2], color="red", s=8, alpha=0.9)
            ax.scatter(worst_x0[i, 0], worst_x0[i, 1], worst_x0[i, 2], color="red", s=start_marker_size)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    mins, maxs = eval_bounds_from_train(x_train, cfg)
    x_min, x_max = float(mins[0]), float(maxs[0])
    y_min, y_max = float(mins[1]), float(maxs[1])
    xx, yy = np.meshgrid(np.linspace(-4.0, 4.0, 300), np.linspace(-3.0, 3.0, 300))
    grid_eval = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid_eval).to(cfg.device)
    with torch.no_grad():
        f_grid = model(grid_t).cpu().numpy().reshape(xx.shape)
        v = (0.5 * (f_grid ** 2)).reshape(xx.shape)
    v_max = float(np.percentile(v, 95))
    levels = (np.linspace(0.0, 1.0, 25) ** 3) * v_max

    plt.figure(figsize=(8, 6))
    if levels.size > 1:
        cs = plt.contour(xx, yy, v, levels=levels[1:], linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=7, fmt="%.2f")
    eps = float(zero_level_eps)
    plt.contourf(xx, yy, f_grid, levels=[-eps, eps], colors=["#ffa500"], alpha=0.55)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=6, alpha=0.6, label="data", zorder=3)
    _draw_sigma_interval_segments_on_contour(x_train, cfg, n_segments=45)
    for i in range(traj.shape[1]):
        plt.scatter(x0[i, 0], x0[i, 1], c="green", s=start_marker_size, label="random starts" if i == 0 else None)
        plt.plot(traj[:, i, 0], traj[:, i, 1], "-", color="green", linewidth=0.8, label="random traj" if i == 0 else None)
        plt.scatter(traj[:, i, 0], traj[:, i, 1], c="green", s=4, alpha=0.8)
    for j, i in enumerate(worst_idx):
        plt.scatter(worst_x0[i, 0], worst_x0[i, 1], c="red", s=start_marker_size, label="worst starts (top 5%)" if j == 0 else None)
        plt.plot(worst_traj[:, i, 0], worst_traj[:, i, 1], "-", color="red", linewidth=0.9, label="worst traj (top 5%)" if j == 0 else None)
        plt.scatter(worst_traj[:, i, 0], worst_traj[:, i, 1], c="red", s=4, alpha=0.9)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    _, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_knn_normals(
    x: np.ndarray,
    idx_list: List[int],
    k: int,
    out_path: str,
    title: str,
    cfg: Any,
    grid: np.ndarray | None = None,
    n_basis: np.ndarray | None = None,
    off_bank: np.ndarray | None = None,
    off_bank_preview: np.ndarray | None = None,
    off_bank_preview_mask: np.ndarray | None = None,
) -> None:
    def _infer_structured_surface(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if g.ndim != 2 or g.shape[1] != 3 or len(g) < 16:
            return None
        # Only render when (x,y)->z behaves like a regular grid; otherwise skip to avoid noisy surfaces.
        xy = np.round(g[:, :2].astype(np.float64), 6)
        ux = np.unique(xy[:, 0])
        uy = np.unique(xy[:, 1])
        if len(ux) * len(uy) != len(g):
            return None
        if len(ux) < 3 or len(uy) < 3:
            return None
        ix = np.searchsorted(ux, xy[:, 0])
        iy = np.searchsorted(uy, xy[:, 1])
        zgrid = np.full((len(ux), len(uy)), np.nan, dtype=np.float64)
        for p, q, z in zip(ix, iy, g[:, 2].astype(np.float64)):
            if np.isnan(zgrid[p, q]):
                zgrid[p, q] = z
            elif abs(zgrid[p, q] - z) > 1e-6:
                return None
        if np.isnan(zgrid).any():
            return None
        xx, yy = np.meshgrid(ux, uy, indexing="ij")
        return xx.astype(np.float32), yy.astype(np.float32), zgrid.astype(np.float32)

    nbr_idx_all = _knn_indices(x, k)
    if n_basis is None or n_basis.shape[0] != len(x) or n_basis.shape[1] != x.shape[1]:
        raise ValueError("plot_knn_normals requires n_basis with shape (N, d, codim)")
    if x.shape[1] == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        if grid is not None:
            surf = _infer_structured_surface(grid)
            if surf is not None:
                gx, gy, gz = surf
                ax.plot_surface(gx, gy, gz, color="lightgray", alpha=0.2, linewidth=0)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=6, alpha=0.35, color="gray")
    else:
        plt.figure(figsize=(6, 5))
        plt.scatter(x[:, 0], x[:, 1], s=6, alpha=0.35, color="gray")
    basis_colors = ["#ef4444", "#06b6d4", "#22c55e", "#f59e0b"]
    for idx in idx_list:
        nn_idx = nbr_idx_all[idx]
        nbrs = x[nn_idx]
        basis = n_basis[idx]
        codim = int(basis.shape[1])

        if x.shape[1] == 3:
            ax.scatter(nbrs[:, 0], nbrs[:, 1], nbrs[:, 2], s=6, color="blue")
            ax.scatter(x[idx, 0], x[idx, 1], x[idx, 2], s=9, color="red")
        else:
            plt.scatter(nbrs[:, 0], nbrs[:, 1], s=6, color="blue")
            plt.scatter(x[idx, 0], x[idx, 1], s=9, color="red")

        for j in range(codim):
            nvec = basis[:, j] / (np.linalg.norm(basis[:, j]) + 1e-12)
            c = basis_colors[j % len(basis_colors)]
            if x.shape[1] == 3:
                ax.quiver(
                    x[idx, 0], x[idx, 1], x[idx, 2],
                    nvec[0], nvec[1], nvec[2],
                    length=0.35, normalize=True, color=c, linewidth=1.2,
                )
            else:
                p_pos = x[idx] + nvec * 0.35
                p_neg = x[idx] - nvec * 0.35
                plt.plot([p_neg[0], p_pos[0]], [p_neg[1], p_pos[1]], color=c, linewidth=1.1, alpha=0.95)

        bank_pts = None
        if off_bank is not None and off_bank.ndim == 3 and idx < off_bank.shape[0]:
            bank_pts = off_bank[idx]
            if bank_pts.ndim == 2 and bank_pts.shape[1] == x.shape[1]:
                bank_pts = bank_pts[np.isfinite(bank_pts).all(axis=1)]
        if bank_pts is None:
            bank_pts = np.zeros((0, x.shape[1]), dtype=np.float32)
        preview_pts = None
        preview_mask = None
        if (
            off_bank_preview is not None
            and off_bank_preview_mask is not None
            and off_bank_preview.ndim == 3
            and off_bank_preview_mask.ndim == 2
            and idx < off_bank_preview.shape[0]
            and idx < off_bank_preview_mask.shape[0]
        ):
            preview_pts = off_bank_preview[idx]
            preview_mask = off_bank_preview_mask[idx].astype(bool)
            if preview_pts.ndim == 2 and preview_pts.shape[1] == x.shape[1]:
                finite = np.isfinite(preview_pts).all(axis=1)
                preview_pts = preview_pts[finite]
                preview_mask = preview_mask[: len(finite)][finite]
        if preview_pts is not None and preview_mask is not None and len(preview_pts) > 0:
            keep_pts = preview_pts[preview_mask]
            drop_pts = preview_pts[~preview_mask]
            if x.shape[1] == 3:
                if len(keep_pts) > 0:
                    ax.scatter(keep_pts[:, 0], keep_pts[:, 1], keep_pts[:, 2], s=4, color="green", alpha=0.45)
                if len(drop_pts) > 0:
                    ax.scatter(drop_pts[:, 0], drop_pts[:, 1], drop_pts[:, 2], s=4, color="orange", alpha=0.65)
            else:
                if len(keep_pts) > 0:
                    plt.scatter(keep_pts[:, 0], keep_pts[:, 1], s=4, color="green", alpha=0.45)
                if len(drop_pts) > 0:
                    plt.scatter(drop_pts[:, 0], drop_pts[:, 1], s=4, color="orange", alpha=0.65)
        else:
            if x.shape[1] == 3:
                if len(bank_pts) > 0:
                    ax.scatter(bank_pts[:, 0], bank_pts[:, 1], bank_pts[:, 2], s=4, color="green", alpha=0.45)
            else:
                if len(bank_pts) > 0:
                    plt.scatter(bank_pts[:, 0], bank_pts[:, 1], s=4, color="green", alpha=0.45)
    if x.shape[1] == 3:
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.legend()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_loss_curves(history: Dict[str, List[float]], out_path: str, title: str, cfg: Any) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    alpha = cfg.loss_ema_alpha

    def ema_series(values: List[float]) -> List[float]:
        if not values:
            return []
        ema = [values[0]]
        for val in values[1:]:
            ema.append(alpha * ema[-1] + (1 - alpha) * val)
        return ema

    loss_keys = ["loss_on", "loss_off", "loss_eikonal", "loss_total"]
    for key in loss_keys:
        vals = history.get(key, [])
        if vals:
            axes[0].plot(vals, alpha=0.25, linewidth=0.8, label=f"{key} raw")
            axes[0].plot(ema_series(vals), linewidth=1.5, label=f"{key} ema")
    axes[0].set_title("Unweighted")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    w_keys = ["w_loss_on", "w_loss_off", "w_loss_eikonal"]
    for key in w_keys:
        vals = history.get(key, [])
        if vals:
            axes[1].plot(vals, alpha=0.25, linewidth=0.8, label=f"{key} raw")
            axes[1].plot(ema_series(vals), linewidth=1.5, label=f"{key} ema")
    axes[1].set_title("Weighted")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
