"""Data augmentation baselines (margin/dataaug) for equality-constraint manifolds."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets.constraint_datasets import set_seed
from evaluation.evaluator import (
    compute_eps_stop,
    resolve_eval_cfg,
    sample_eval_seed_points,
)
from experiments.dataset_resolve import resolve_dataset
from evaluation.eval_runner import run_eval_metrics
from models.kinematics import (
    is_arm_dataset as _is_arm_dataset,
    planar_fk as _planar_fk,
    spatial_fk_n3 as _spatial_fk_n3,
    spatial_fk_n4 as _spatial_fk_n4,
    workspace_embed_for_eval as shared_workspace_embed_for_eval,
    wrap_np_pi as _wrap_np_pi,
)
from models.feature_normalizer import FeatureNormalizer
from models.projection import (
    project_points_with_steps_numpy,
    project_trajectory_numpy,
    project_trajectory_tensor,
    true_distance,
    true_projection,
)
from models.mlp import MLP
from models.planner import plan_path
from plotting.dataaug_plots import (
    plot_contour_and_trajectory,
    plot_contour_only,
    plot_distance_curves,
    plot_knn_normals,
    plot_loss_curves,
    plot_planned_paths,
    plot_planned_paths_off,
    plot_worst_distance_contour,
)

@dataclass
class Config:
    seed: int = 721  # random seed
    device: str = "auto"  # "auto", "cpu", or "cuda"

    n_train: int = 512  # training points per dataset (dataset-layer config should override)
    traj_gene_n_grid: int = 4096  # grid points used for trajectory/data generation

    # KNN sizes
    knn_norm_estimation_ratio: float = 0.08  # ratio for normal-estimation knn points
    knn_norm_estimation_min_points: int = 4  # min points for normal-estimation knn

    sigmas: Tuple[float, ...] = (0.2)  # off-surface sigma list
    off_sigma_mode: str = "list"  # "list" or "max"
    off_bank_size: int = 16  # off-bank batches cached
    off_bank_oversample: int = 6  # oversample factor for off-bank
    off_bank_chunk: int = 64  # chunk size for off-bank
    off_bank_max_rounds: int = 6  # rounds for off-bank fill
    off_curriculum_enable: bool = False  # enable truncated-Gaussian curriculum on off-bank sampling
    off_curriculum_start_ratio: float = 0.2  # initial |s| cap ratio in [0,1]
    off_curriculum_power: float = 1.0  # progress exponent for cap schedule
    off_curriculum_epochs: int = 0  # curriculum epochs; 0 means use full training epochs

    use_knn_filter: bool = False  # filter off points by knn
    knn_off_data_filter_ratio: float = 0.03  # ratio for off-data filter knn points
    knn_off_data_filter_min_points: int = 1  # min points for off-data filter knn

    hidden: int = 128  # model width
    depth: int = 3  # model depth
    lr: float = 3e-4  # optimizer learning rate
    lr_decay_step: int = 0  # <=0 disables LR decay
    lr_decay_gamma: float = 1.0  # multiplicative decay factor
    epochs: int = 2000  # training epochs
    train_log_every: int = 50  # logging interval (epochs)
    batch_size: int = 128  # batch size
    baseline_margin_target: float = 1.0  # target |h(x_off)| for margin baseline
    lam_on: float = 1.0  # on-manifold penalty weight
    lam_off: float = 10.0  # off-manifold penalty weight

    # plot
    loss_ema_alpha: float = 0.85  # loss EMA alpha curve

    eps: float = 1e-8  # numeric eps

    projector: dict = field(default_factory=lambda: {"alpha": 0.3, "steps": 100, "min_steps": 30})
    planner: dict = field(
        default_factory=lambda: {
            "enable": True,
            "steps": 64,
            "method": "traj_opt",
            "opt_steps": 1240,
            "opt_lr": 0.01,
            "opt_lam_smooth": 0.2,
            "lam_manifold": 1.0,
            "lam_len_joint": 0.40,
            "trust_scale": 0.8,
            "off_dist": 0.6,
        }
    )

def _ur5_backend_from_dataset(name: str) -> str:
    return "pybullet" if str(name) == "6d_spatial_arm_up_n6" else "analytic"


def _planner_val(cfg: Config, key: str, default: float | int | str) -> float | int | str:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        return pln[key]
    return default


def _planner_bool(cfg: Config, key: str, default: bool) -> bool:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        return bool(pln[key])
    return bool(default)


def _projector_val(cfg: Config, key: str, default: float | int) -> float | int:
    proj = getattr(cfg, "projector", None)
    if isinstance(proj, dict) and key in proj:
        return proj[key]
    return default


def pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (N, d), b: (M, d)
    a = np.nan_to_num(a, copy=False).astype(np.float64)
    b = np.nan_to_num(b, copy=False).astype(np.float64)
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    d2 = np.maximum(d2, 0.0)
    return d2


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
        d2 = pairwise_sqdist(x, x)
        return np.argsort(d2, axis=1)[:, 1 : k_eff + 1].astype(np.int64)


def _as_sigmas(sigmas: Tuple[float, ...] | float) -> np.ndarray:
    if isinstance(sigmas, (int, float)):
        return np.array([float(sigmas)], dtype=np.float32)
    return np.array(sigmas, dtype=np.float32)


def effective_knn_norm_estimation_points(cfg: Config, n: int) -> int:
    return max(cfg.knn_norm_estimation_min_points, int(round(cfg.knn_norm_estimation_ratio * n)))


def effective_knn_off_data_filter_points(cfg: Config, n: int) -> int:
    return max(
        cfg.knn_off_data_filter_min_points,
        int(round(cfg.knn_off_data_filter_ratio * n)),
    )


def _local_pca_frame(
    neigh: np.ndarray,
    center: np.ndarray,
    cfg: Config | None = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(neigh) == 0:
        d = center.shape[0]
        return (
            np.zeros((d,), dtype=np.float64),
            np.eye(d, dtype=np.float64),
            np.ones((1,), dtype=np.float64),
            neigh,
        )
    mu = np.mean(neigh, axis=0, keepdims=True)
    xc = neigh - mu
    cov = (xc.T @ xc) / max(len(neigh), 1)
    evals, evecs = np.linalg.eigh(cov)
    w = np.full((len(neigh),), 1.0 / max(len(neigh), 1), dtype=np.float64)
    return evals, evecs, w, neigh


def knn_normals(
    x: np.ndarray, k: int, cfg: Config | None = None
) -> np.ndarray:
    n, d = x.shape
    nbr_idx_all = _knn_indices(x, k)
    normals = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        nbr_idx = nbr_idx_all[i]
        nbrs = x[nbr_idx]
        _, evecs, _, _ = _local_pca_frame(nbrs, x[i], cfg=cfg)
        nvec = evecs[:, 0]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        normals[i] = nvec.astype(np.float32)
    return normals


def knn_normal_bases(
    x: np.ndarray,
    k: int,
    codim: int,
    cfg: Config | None = None,
) -> np.ndarray:
    n, d = x.shape
    c = max(1, min(int(codim), int(d)))
    nbr_idx_all = _knn_indices(x, k)
    bases = np.zeros((n, d, c), dtype=np.float32)
    for i in range(n):
        nbr_idx = nbr_idx_all[i]
        nbrs = x[nbr_idx]
        _, evecs, _, _ = _local_pca_frame(nbrs, x[i], cfg=cfg)
        b = evecs[:, :c]
        for j in range(c):
            bj = b[:, j]
            b[:, j] = bj / (np.linalg.norm(bj) + 1e-12)
        bases[i] = b.astype(np.float32)
    return bases


def sample_off_manifold(
    x_on: torch.Tensor,
    n_basis: torch.Tensor,
    sigmas: Tuple[float, ...] | float,
    sigma_mode: str = "list",
    sigma_per_point: torch.Tensor | None = None,
    r_kappa_per_point: torch.Tensor | None = None,
    r_pos_per_point: torch.Tensor | None = None,
    r_neg_per_point: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sigmas = _as_sigmas(sigmas)
    if sigma_mode == "scale":
        raise ValueError("sigma_mode='scale' was removed; use 'max' or 'list'")

    # Sample off-distance radius rho from sigmas-defined distribution.
    if sigma_per_point is not None:
        sigma = sigma_per_point.to(device=x_on.device, dtype=x_on.dtype).view(-1, 1)
        rho = torch.abs(torch.randn_like(x_on[:, :1]) * sigma)
        if r_pos_per_point is not None and r_neg_per_point is not None:
            r_pos = r_pos_per_point.to(device=x_on.device, dtype=x_on.dtype).view(-1, 1)
            r_neg = r_neg_per_point.to(device=x_on.device, dtype=x_on.dtype).view(-1, 1)
            rho = torch.minimum(rho, torch.minimum(r_pos, r_neg))
        else:
            if r_kappa_per_point is not None:
                r_kappa = r_kappa_per_point.to(device=x_on.device, dtype=x_on.dtype).view(-1, 1)
                rho = torch.minimum(rho, r_kappa)
    elif sigma_mode == "max":
        sigma = torch.tensor(
            float(np.max(sigmas)), device=x_on.device, dtype=x_on.dtype
        ).reshape(1, 1)
        rho = torch.abs(torch.randn_like(x_on[:, :1]) * sigma)
    else:
        sigma_choices = torch.tensor(sigmas, device=x_on.device, dtype=x_on.dtype)
        idx = torch.randint(0, len(sigmas), (x_on.shape[0], 1), device=x_on.device)
        sigma = sigma_choices[idx]
        rho = torch.abs(torch.randn_like(x_on[:, :1]) * sigma)

    # Sample uniform directions in local normal subspace.
    # n_basis: (B, d, c), dir_local: (B, c), unit-norm.
    c = int(n_basis.shape[2])
    dir_local = torch.randn((x_on.shape[0], c), device=x_on.device, dtype=x_on.dtype)
    dir_local = dir_local / torch.linalg.norm(dir_local, dim=1, keepdim=True).clamp_min(1e-12)
    delta_local = dir_local * rho  # (B, c)
    delta = torch.einsum("bdc,bc->bd", n_basis, delta_local)
    x_off = x_on + delta
    return x_off, rho


def filter_off_by_knn(
    x_off: torch.Tensor, x_ref: torch.Tensor, idx_on: torch.Tensor, knn_off_data_filter_points: int
) -> torch.Tensor:
    n_off = int(x_off.shape[0])
    n_ref = int(x_ref.shape[0])
    if n_off <= 0 or n_ref <= 0:
        return torch.zeros((n_off,), dtype=torch.bool, device=x_off.device)
    k = max(1, min(int(knn_off_data_filter_points), n_ref))
    idx_on = idx_on.view(-1, 1)
    # Chunk pairwise distance to avoid OOM on large-N datasets.
    # Peak temporary tensor ~ chunk_size * n_ref.
    max_elems = 8_000_000
    chunk_size = max(1, min(n_off, max_elems // max(1, n_ref)))
    out_mask: list[torch.Tensor] = []
    for start in range(0, n_off, chunk_size):
        end = min(n_off, start + chunk_size)
        d2 = torch.cdist(x_off[start:end], x_ref) ** 2
        nn_idx = torch.topk(d2, k, largest=False).indices
        mask = (nn_idx == idx_on[start:end]).any(dim=1)
        out_mask.append(mask)
    return torch.cat(out_mask, dim=0)


def precompute_off_bank(
    x: np.ndarray,
    n_basis: np.ndarray,
    cfg: Config,
    sigma_per_point: np.ndarray | None,
    r_kappa_per_point: np.ndarray | None,
    r_pos_per_point: np.ndarray | None = None,
    r_neg_per_point: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n, dim = x.shape
    k = max(1, cfg.off_bank_size)
    off_bank = np.zeros((n, k, dim), dtype=np.float32)
    s_bank = np.zeros((n, k, 1), dtype=np.float32)
    x_t = torch.from_numpy(x)
    n_t = torch.from_numpy(n_basis)
    idx_t = torch.arange(n, dtype=torch.long)

    chunk = max(1, cfg.off_bank_chunk)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        x_chunk = x_t[start:end]
        n_chunk = n_t[start:end]
        idx_chunk = idx_t[start:end]
        filled = [0 for _ in range(end - start)]
        rounds = 0
        while min(filled) < k and rounds < cfg.off_bank_max_rounds:
            rounds += 1
            n_samples = k * cfg.off_bank_oversample
            x_rep = x_chunk.repeat_interleave(n_samples, dim=0)
            n_rep = n_chunk.repeat_interleave(n_samples, dim=0)
            idx_rep = idx_chunk.repeat_interleave(n_samples, dim=0)
            sigma_rep = None
            r_kappa_rep = None
            r_pos_rep = None
            r_neg_rep = None
            if sigma_per_point is not None:
                sigma_chunk = sigma_per_point[start:end]
                sigma_rep = torch.from_numpy(np.repeat(sigma_chunk, n_samples))
            if r_kappa_per_point is not None:
                r_chunk = r_kappa_per_point[start:end]
                r_kappa_rep = torch.from_numpy(np.repeat(r_chunk, n_samples))
            if r_pos_per_point is not None:
                rp_chunk = r_pos_per_point[start:end]
                r_pos_rep = torch.from_numpy(np.repeat(rp_chunk, n_samples))
            if r_neg_per_point is not None:
                rn_chunk = r_neg_per_point[start:end]
                r_neg_rep = torch.from_numpy(np.repeat(rn_chunk, n_samples))
            x_off_cand, s_cand = sample_off_manifold(
                x_rep,
                n_rep,
                cfg.sigmas,
                cfg.off_sigma_mode,
                sigma_rep,
                r_kappa_rep,
                r_pos_rep,
                r_neg_rep,
            )
            if cfg.use_knn_filter:
                mask = filter_off_by_knn(
                    x_off_cand,
                    x_t,
                    idx_rep,
                    effective_knn_off_data_filter_points(cfg, len(x)),
                )
            else:
                mask = torch.ones(
                    x_off_cand.shape[0],
                    dtype=torch.bool,
                    device=x_off_cand.device,
                )
            mask = mask.view(end - start, n_samples)
            x_off_cand = x_off_cand.view(end - start, n_samples, dim)
            s_cand = s_cand.view(end - start, n_samples, 1)
            for i in range(end - start):
                if filled[i] >= k:
                    continue
                keep = torch.where(mask[i])[0]
                if keep.numel() == 0:
                    continue
                for j in keep.tolist():
                    off_bank[start + i, filled[i]] = x_off_cand[i, j].numpy()
                    s_bank[start + i, filled[i]] = s_cand[i, j].numpy()
                    filled[i] += 1
                    if filled[i] >= k:
                        break
        for i in range(end - start):
            if filled[i] < k:
                need = k - filled[i]
                sigma_one = None
                r_one = None
                rp_one = None
                rn_one = None
                if sigma_per_point is not None:
                    sigma_one = torch.from_numpy(
                        np.array([sigma_per_point[start + i]], dtype=np.float32)
                    )
                if r_kappa_per_point is not None:
                    r_one = torch.from_numpy(
                        np.array([r_kappa_per_point[start + i]], dtype=np.float32)
                    )
                if r_pos_per_point is not None:
                    rp_one = torch.from_numpy(
                        np.array([r_pos_per_point[start + i]], dtype=np.float32)
                    )
                if r_neg_per_point is not None:
                    rn_one = torch.from_numpy(
                        np.array([r_neg_per_point[start + i]], dtype=np.float32)
                    )
                if cfg.use_knn_filter:
                    # Keep filter semantics in fallback: keep sampling until enough
                    # valid points are found.
                    tries = 0
                    max_tries = max(32, cfg.off_bank_max_rounds * 16)
                    while filled[i] < k and tries < max_tries:
                        tries += 1
                        x_off_fill, s_fill = sample_off_manifold(
                            x_chunk[i : i + 1],
                            n_chunk[i : i + 1],
                            cfg.sigmas,
                            cfg.off_sigma_mode,
                            sigma_one,
                            r_one,
                            rp_one,
                            rn_one,
                        )
                        mask_one = filter_off_by_knn(
                            x_off_fill,
                            x_t,
                            idx_chunk[i : i + 1],
                            effective_knn_off_data_filter_points(cfg, len(x)),
                        )
                        if bool(mask_one.item()):
                            off_bank[start + i, filled[i]] = x_off_fill[0].numpy()
                            s_bank[start + i, filled[i]] = s_fill[0].numpy()
                            filled[i] += 1
                    if filled[i] < k:
                        raise RuntimeError(
                            "off_bank fill failed under knn filter; "
                            "try lowering knn_off_data_filter_ratio/min_points "
                            "or increasing off_bank_oversample/off_bank_max_rounds."
                        )
                else:
                    x_off_fill, s_fill = sample_off_manifold(
                        x_chunk[i : i + 1],
                        n_chunk[i : i + 1],
                        cfg.sigmas,
                        cfg.off_sigma_mode,
                        sigma_one,
                        r_one,
                        rp_one,
                        rn_one,
                    )
                    off_bank[start + i, filled[i] :] = (
                        x_off_fill.repeat(need, 1).numpy()
                    )
                    s_bank[start + i, filled[i] :] = s_fill.repeat(need, 1).numpy()
    return off_bank, s_bank


def build_off_bank_preview(
    x: np.ndarray,
    n_basis: np.ndarray,
    cfg: Config,
    sigma_per_point: np.ndarray | None,
    r_kappa_per_point: np.ndarray | None,
    r_pos_per_point: np.ndarray | None = None,
    r_neg_per_point: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n, dim = x.shape
    n_preview = int(max(8, min(64, max(1, cfg.off_bank_size) * max(1, cfg.off_bank_oversample))))
    x_t = torch.from_numpy(x)
    n_t = torch.from_numpy(n_basis)
    idx_t = torch.arange(n, dtype=torch.long)
    x_rep = x_t.repeat_interleave(n_preview, dim=0)
    n_rep = n_t.repeat_interleave(n_preview, dim=0)
    idx_rep = idx_t.repeat_interleave(n_preview)
    sigma_rep = None
    r_kappa_rep = None
    r_pos_rep = None
    r_neg_rep = None
    if sigma_per_point is not None:
        sigma_rep = torch.from_numpy(np.repeat(sigma_per_point, n_preview))
    if r_kappa_per_point is not None:
        r_kappa_rep = torch.from_numpy(np.repeat(r_kappa_per_point, n_preview))
    if r_pos_per_point is not None:
        r_pos_rep = torch.from_numpy(np.repeat(r_pos_per_point, n_preview))
    if r_neg_per_point is not None:
        r_neg_rep = torch.from_numpy(np.repeat(r_neg_per_point, n_preview))

    x_off_cand, _ = sample_off_manifold(
        x_rep,
        n_rep,
        cfg.sigmas,
        cfg.off_sigma_mode,
        sigma_rep,
        r_kappa_rep,
        r_pos_rep,
        r_neg_rep,
    )
    if cfg.use_knn_filter:
        mask = filter_off_by_knn(
            x_off_cand,
            x_t,
            idx_rep,
            effective_knn_off_data_filter_points(cfg, len(x)),
        )
    else:
        mask = torch.ones(x_off_cand.shape[0], dtype=torch.bool)
    return (
        x_off_cand.view(n, n_preview, dim).numpy().astype(np.float32),
        mask.view(n, n_preview).numpy().astype(bool),
    )


def _off_curriculum_cap_ratio(cfg: Config, epoch: int) -> float | None:
    if not bool(cfg.off_curriculum_enable):
        return None
    total = int(cfg.off_curriculum_epochs) if int(cfg.off_curriculum_epochs) > 0 else int(cfg.epochs)
    total = max(1, total)
    progress = min(1.0, max(0.0, float(epoch + 1) / float(total)))
    power = max(1e-8, float(cfg.off_curriculum_power))
    start = float(np.clip(cfg.off_curriculum_start_ratio, 0.0, 1.0))
    return start + (1.0 - start) * (progress ** power)


def _sample_off_bank_indices(
    s_bank_rows: np.ndarray, cap_ratio: float | None
) -> np.ndarray:
    # s_bank_rows: (B, K, 1), pick one candidate index per row.
    bsz, ksz = s_bank_rows.shape[0], s_bank_rows.shape[1]
    if cap_ratio is None:
        return np.random.randint(0, ksz, size=bsz)

    s_abs = np.abs(s_bank_rows[..., 0])
    row_max = np.maximum(np.max(s_abs, axis=1), 1e-12)
    caps = row_max * float(cap_ratio)
    keep = s_abs <= caps[:, None]
    bank_idx = np.empty(bsz, dtype=np.int64)
    for i in range(bsz):
        cand = np.where(keep[i])[0]
        if cand.size == 0:
            # Fallback: choose closest-to-manifold candidate.
            bank_idx[i] = int(np.argmin(s_abs[i]))
        else:
            bank_idx[i] = int(cand[np.random.randint(0, cand.size)])
    return bank_idx

def _make_project_fn(cfg: Config):
    def _project_points_for_eval(
        model: nn.Module, x0: np.ndarray, eps_stop: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return project_points_with_steps_numpy(
            model,
            x0,
            device=str(cfg.device),
            proj_steps=int(_projector_val(cfg, "steps", 100)),
            proj_alpha=float(_projector_val(cfg, "alpha", 0.3)),
            proj_min_steps=int(_projector_val(cfg, "min_steps", 0)),
            f_abs_stop=eps_stop,
        )

    return _project_points_for_eval


def train_baseline(
    cfg: Config, mode: str, x: np.ndarray, n_basis: np.ndarray
) -> Tuple[nn.Module, Dict[str, float], Dict[str, List[float]], Dict[str, np.ndarray]]:
    x_t = torch.from_numpy(x)
    n_t = torch.from_numpy(n_basis)
    idx_t = torch.arange(len(x), dtype=torch.long)
    sigma_per_point = None
    r_kappa_per_point = None
    r_pos_per_point = None
    r_neg_per_point = None
    off_bank = None
    s_bank = None
    off_bank_preview = np.zeros((len(x), 0, x.shape[1]), dtype=np.float32)
    off_bank_preview_mask = np.zeros((len(x), 0), dtype=bool)
    if cfg.off_bank_size > 0:
        off_bank, s_bank = precompute_off_bank(
            x,
            n_basis,
            cfg,
            sigma_per_point,
            r_kappa_per_point,
            r_pos_per_point,
            r_neg_per_point,
        )
    off_bank_preview, off_bank_preview_mask = build_off_bank_preview(
        x,
        n_basis,
        cfg,
        sigma_per_point,
        r_kappa_per_point,
        r_pos_per_point,
        r_neg_per_point,
    )

    ds = TensorDataset(x_t, n_t, idx_t)
    use_cuda = cfg.device == "cuda"
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0,
    )

    model = MLP(in_dim=x.shape[1], hidden=cfg.hidden, depth=cfg.depth, out_dim=1).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = None
    if int(cfg.lr_decay_step) > 0 and float(cfg.lr_decay_gamma) < 1.0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=int(cfg.lr_decay_step),
            gamma=float(cfg.lr_decay_gamma),
        )

    history: Dict[str, List[float]] = {
        "loss_on": [],
        "loss_off": [],
        "loss_total": [],
        "w_loss_on": [],
        "w_loss_off": [],
    }
    last_stats: Dict[str, float] = {}
    x_ref = x_t.to(cfg.device)
    for epoch in range(cfg.epochs):
        for xb, nbasis, idxb in dl:
            xb = xb.to(cfg.device)
            nbasis = nbasis.to(cfg.device)
            idxb = idxb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            # On-manifold penalty: enforce h(x_on)=0.
            h_on = model(xb)
            loss_on = (h_on ** 2).mean()

            if off_bank is not None and s_bank is not None:
                idx_np = idxb.cpu().numpy()
                cap_ratio = _off_curriculum_cap_ratio(cfg, epoch)
                bank_idx = _sample_off_bank_indices(s_bank[idx_np], cap_ratio)
                x_off = torch.from_numpy(off_bank[idx_np, bank_idx]).to(cfg.device)
                s = torch.from_numpy(s_bank[idx_np, bank_idx]).to(cfg.device)
            else:
                sigma_batch = None
                if sigma_per_point is not None:
                    sigma_batch = torch.from_numpy(
                        sigma_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                r_kappa_batch = None
                if r_kappa_per_point is not None:
                    r_kappa_batch = torch.from_numpy(
                        r_kappa_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                r_pos_batch = None
                if r_pos_per_point is not None:
                    r_pos_batch = torch.from_numpy(
                        r_pos_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                r_neg_batch = None
                if r_neg_per_point is not None:
                    r_neg_batch = torch.from_numpy(
                        r_neg_per_point[idxb.cpu().numpy()].astype(np.float32)
                    ).to(cfg.device)
                x_off, s = sample_off_manifold(
                    xb,
                    nbasis,
                    cfg.sigmas,
                    cfg.off_sigma_mode,
                    sigma_batch,
                    r_kappa_batch,
                    r_pos_batch,
                    r_neg_batch,
                )
                if cfg.use_knn_filter:
                    knn_off_data_filter_points = effective_knn_off_data_filter_points(cfg, x_ref.shape[0])
                    mask = filter_off_by_knn(x_off, x_ref, idxb, knn_off_data_filter_points)
                    if mask.sum() > 0:
                        x_off = x_off[mask]
                        s = s[mask]
                    else:
                        # Keep filter semantics: no valid off-point this mini-batch.
                        # Still train on on-manifold term.
                        s = None
            if s is None:
                loss_off = torch.zeros((), device=xb.device)
            else:
                # Off-manifold penalty only (simple baseline):
                # - margin mode: |h(x_off)| -> 1
                # - dataaug mode: |h(x_off)| -> |delta|
                h_off = model(x_off)
                if mode == "margin":
                    target = torch.full_like(h_off, float(cfg.baseline_margin_target))
                    loss_off = ((torch.abs(h_off) - target) ** 2).mean()
                elif mode == "dataaug":
                    target = torch.abs(s)
                    loss_off = ((torch.abs(h_off) - target) ** 2).mean()
                else:
                    raise ValueError(f"unknown baseline mode: {mode}")

            loss = cfg.lam_on * loss_on + cfg.lam_off * loss_off
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_stats = {
                "loss_on": float(loss_on.detach().cpu()),
                "loss_off": float(loss_off.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
                "w_loss_on": float((cfg.lam_on * loss_on).detach().cpu()),
                "w_loss_off": float((cfg.lam_off * loss_off).detach().cpu()),
            }
        if last_stats:
            for k in history:
                history[k].append(last_stats[k])
        if ((epoch + 1) % max(1, int(cfg.train_log_every))) == 0 or epoch == 0 or (epoch + 1) == int(cfg.epochs):
            lr_now = float(opt.param_groups[0]["lr"])
            print(
                f"[train] method={mode} ep={epoch+1:4d}/{cfg.epochs} "
                f"| lr={lr_now:.2e} "
                f"| loss={last_stats['loss_total']:.6f} "
                f"| on={last_stats['loss_on']:.6f} "
                f"| off={last_stats['loss_off']:.6f}"
            )
        if scheduler is not None:
            scheduler.step()

    artifacts = {
        "off_bank": off_bank if off_bank is not None else np.zeros((0, 0, x.shape[1]), dtype=np.float32),
        "off_bank_preview": off_bank_preview,
        "off_bank_preview_mask": off_bank_preview_mask,
    }
    return model, last_stats, history, artifacts


def train_baseline_with_optional_normalization(
    cfg: Config,
    mode: str,
    x: np.ndarray,
    n_basis: np.ndarray | None = None,
    *,
    true_codim: int = 1,
    dataset_name: str | None = None,
    knn_k: int | None = None,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, List[float]], Dict[str, np.ndarray]]:
    normalizer = FeatureNormalizer.fit(x, dataset_name=dataset_name, enable=True)
    if not normalizer.enabled:
        if n_basis is None:
            k_eff = effective_knn_norm_estimation_points(cfg, len(x)) if knn_k is None else int(knn_k)
            n_basis = knn_normal_bases(x, k_eff, int(true_codim), cfg)
        model, stats, hist, artifacts = train_baseline(cfg, mode=mode, x=x, n_basis=n_basis)
        artifacts["feature_normalizer"] = normalizer
        artifacts["n_basis_plot"] = n_basis.astype(np.float32)
        return model, stats, hist, artifacts

    x_norm = normalizer.transform(x)
    cfg_norm = Config(**{k: getattr(cfg, k) for k in Config.__dataclass_fields__.keys()})
    k_eff = effective_knn_norm_estimation_points(cfg_norm, len(x_norm)) if knn_k is None else int(knn_k)
    n_basis_norm = knn_normal_bases(x_norm, k_eff, int(true_codim), cfg_norm)
    model, stats, hist, artifacts = train_baseline(cfg_norm, mode=mode, x=x_norm, n_basis=n_basis_norm)
    artifacts["feature_normalizer"] = normalizer
    artifacts["n_basis_plot"] = np.zeros((len(x), x.shape[1], int(true_codim)), dtype=np.float32)
    artifacts["x_train_model_space"] = x_norm.astype(np.float32)
    return model, stats, hist, artifacts


def main() -> None:
    cfg = Config()
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(7)
    datasets = [
        "2d_sharp_star",
        # "2d_figure_eight",
        # "2d_ellipse",
        # "2d_discontinuous",
        # "2d_noisy_sine",
        # "2d_sparse_sine",
        # "2d_sine",
        # "2d_looped_spiro",

        # "2d_hetero_noise",

    ]
    methods = ["margin", "dataaug"]
    output_root = "outputs_levelset_datasets"
    os.makedirs(output_root, exist_ok=True)
    eval_results = []

    for name in datasets:
        set_seed(cfg.seed)
        ds = resolve_dataset(
            name,
            cfg,
            optimize_ur5_train_only=True,
            ur5_backend=_ur5_backend_from_dataset(name),
        )
        x_train = ds["x_train"]
        grid = ds["grid"]
        eval_cfg_ref = resolve_eval_cfg(
            cfg,
            method_key="margin",
            dataset_name=name,
        )
        vis_vals_ref = dict(cfg.__dict__)
        vis_vals_ref.update(vars(eval_cfg_ref))
        vis_cfg_ref = SimpleNamespace(**vis_vals_ref)
        x_eval = sample_eval_seed_points(x_train, eval_cfg_ref)
        knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x_train))
        true_codim = int(ds.get("true_codim", 1))
        n_basis = knn_normal_bases(x_train, knn_norm_estimation_points, true_codim, cfg)

        margin_model = None
        margin_stats = {}
        margin_history = {}
        margin_artifacts: Dict[str, np.ndarray] = {}
        if "margin" in methods:
            margin_model, margin_stats, margin_history, margin_artifacts = train_baseline(
                cfg, mode="margin", x=x_train, n_basis=n_basis
            )

        delta_model = None
        delta_stats = {}
        delta_history = {}
        delta_artifacts: Dict[str, np.ndarray] = {}
        if "dataaug" in methods:
            delta_model, delta_stats, delta_history, delta_artifacts = train_baseline(
                cfg, mode="dataaug", x=x_train, n_basis=n_basis
            )

        out_dir = os.path.join(output_root, name)
        os.makedirs(out_dir, exist_ok=True)

        if x_train.shape[1] == 3:
            n_plot = 16
        else:
            n_plot = 128
        n_plot = min(n_plot, len(x_eval))
        x0 = x_eval[:n_plot]
        x0_t = torch.from_numpy(x0).to(cfg.device)
        plan_pairs = None
        plan_pairs_off = None
        if x_train.shape[1] == 2 and _planner_bool(cfg, "enable", True):
            plan_rng = np.random.default_rng(cfg.seed + 77)
            n_pairs = 4
            replace = len(x_eval) < 2 * n_pairs
            picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
            plan_pairs = [
                (x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]])
                for i in range(n_pairs)
            ]
            off_dirs = plan_rng.normal(size=(n_pairs, 2)).astype(np.float32)
            norms = np.linalg.norm(off_dirs, axis=1, keepdims=True) + 1e-8
            off_dirs = off_dirs / norms
            plan_pairs_off = []
            for i in range(n_pairs):
                x0_p = plan_pairs[i][0]
                x1_on = plan_pairs[i][1]
                x1_off = x1_on + off_dirs[i] * float(_planner_val(cfg, "off_dist", 0.6))
                plan_pairs_off.append((x0_p, x1_off))
        steps = 0
        knn_norm_estimation_points = effective_knn_norm_estimation_points(cfg, len(x_train))
        n_train = len(x_train)
        idx_list = [
            min(n_train - 1, int(round((n_train - 1) * r)))
            for r in np.linspace(0.05, 0.95, 15)
        ]
        plot_knn_normals(
            x_train,
            idx_list=idx_list,
            k=knn_norm_estimation_points,
            out_path=os.path.join(out_dir, "knn_normal.png"),
            title=f"{name}: KNN + Normal",
            cfg=vis_cfg_ref,
            grid=grid,
            n_basis=n_basis,
            off_bank=(
                delta_artifacts.get("off_bank")
                if len(delta_artifacts.get("off_bank", np.zeros((0, 0, x_train.shape[1]), dtype=np.float32))) > 0
                else margin_artifacts.get("off_bank")
            ),
            off_bank_preview=(
                delta_artifacts.get("off_bank_preview")
                if len(delta_artifacts.get("off_bank_preview", np.zeros((0, 0, x_train.shape[1]), dtype=np.float32))) > 0
                else margin_artifacts.get("off_bank_preview")
            ),
            off_bank_preview_mask=(
                delta_artifacts.get("off_bank_preview_mask")
                if len(delta_artifacts.get("off_bank_preview_mask", np.zeros((0, 0), dtype=bool))) > 0
                else margin_artifacts.get("off_bank_preview_mask")
            ),
        )
        if margin_model is not None:
            eval_cfg_margin = resolve_eval_cfg(cfg, method_key="margin", dataset_name=name)
            vis_vals_margin = dict(cfg.__dict__)
            vis_vals_margin.update(vars(eval_cfg_margin))
            vis_cfg_margin = SimpleNamespace(**vis_vals_margin)
            margin_eps_used = compute_eps_stop(margin_model, x_train, eval_cfg_margin)
            margin_traj, _ = project_trajectory_tensor(
                margin_model,
                x0_t,
                proj_steps=int(_projector_val(cfg, "steps", 100)),
                proj_alpha=float(_projector_val(cfg, "alpha", 0.3)),
                proj_min_steps=int(_projector_val(cfg, "min_steps", 0)),
                f_abs_stop=margin_eps_used,
            )
            margin_traj = margin_traj.cpu().numpy()
            plot_contour_and_trajectory(
                margin_model,
                x_train,
                x0,
                margin_traj,
                vis_cfg_margin,
                out_path=os.path.join(out_dir, "margin_contour_traj.png"),
                title=f"{name}: Margin Baseline",
                grid=grid,
                zero_level_eps=margin_eps_used,
                eval_points=x_eval,
                worst_frac=0.05,
                project_trajectory_fn=lambda m, x0_seed, c, f_abs_stop: (
                    torch.from_numpy(
                        project_trajectory_numpy(
                            m,
                            x0_seed.detach().cpu().numpy(),
                            device=str(c.device),
                            proj_steps=int(_projector_val(c, "steps", 100)),
                            proj_alpha=float(_projector_val(c, "alpha", 0.3)),
                            proj_min_steps=int(_projector_val(c, "min_steps", 0)),
                            f_abs_stop=f_abs_stop,
                        )
                    ),
                    int(c.proj_steps),
                ),
            )
            plot_distance_curves(
                margin_traj,
                grid,
                out_path=os.path.join(out_dir, "margin_distance_curves.png"),
                title=f"{name}: Margin Baseline Distances",
            )
            plot_worst_distance_contour(
                margin_model,
                x_train,
                grid,
                x0,
                margin_traj,
                vis_cfg_margin,
                out_path=os.path.join(out_dir, "margin_worst_distance_contour.png"),
                title=f"{name}: Margin Worst Distance (top 5%)",
                zero_level_eps=margin_eps_used,
            )
            plot_loss_curves(
                margin_history,
                out_path=os.path.join(out_dir, "margin_loss_curves.png"),
                title=f"{name}: Margin Baseline Losses",
                cfg=vis_cfg_margin,
            )

        if delta_model is not None:
            eval_cfg_delta = resolve_eval_cfg(cfg, method_key="dataaug", dataset_name=name)
            vis_vals_delta = dict(cfg.__dict__)
            vis_vals_delta.update(vars(eval_cfg_delta))
            vis_cfg_delta = SimpleNamespace(**vis_vals_delta)
            delta_eps_used = compute_eps_stop(delta_model, x_train, eval_cfg_delta)
            delta_traj, _ = project_trajectory_tensor(
                delta_model,
                x0_t,
                proj_steps=int(_projector_val(cfg, "steps", 100)),
                proj_alpha=float(_projector_val(cfg, "alpha", 0.3)),
                proj_min_steps=int(_projector_val(cfg, "min_steps", 0)),
                f_abs_stop=delta_eps_used,
            )
            delta_traj = delta_traj.cpu().numpy()
            plot_contour_and_trajectory(
                delta_model,
                x_train,
                x0,
                delta_traj,
                vis_cfg_delta,
                out_path=os.path.join(out_dir, "delta_contour_traj.png"),
                title=f"{name}: Delta Baseline",
                grid=grid,
                zero_level_eps=delta_eps_used,
                eval_points=x_eval,
                worst_frac=0.05,
                project_trajectory_fn=lambda m, x0_seed, c, f_abs_stop: (
                    torch.from_numpy(
                        project_trajectory_numpy(
                            m,
                            x0_seed.detach().cpu().numpy(),
                            device=str(c.device),
                            proj_steps=int(_projector_val(c, "steps", 100)),
                            proj_alpha=float(_projector_val(c, "alpha", 0.3)),
                            proj_min_steps=int(_projector_val(c, "min_steps", 0)),
                            f_abs_stop=f_abs_stop,
                        )
                    ),
                    int(c.proj_steps),
                ),
            )
            plot_distance_curves(
                delta_traj,
                grid,
                out_path=os.path.join(out_dir, "delta_distance_curves.png"),
                title=f"{name}: Delta Baseline Distances",
            )
            if plan_pairs is not None:
                plans_proj = []
                plans_constr = []
                plans_proj_off = []
                plans_constr_off = []
                planner_name = str(_planner_val(cfg, "method", "traj_opt"))
                n_waypoints = int(_planner_val(cfg, "steps", 64)) + 1
                use_linear = planner_name.lower() == "linear_proj"
                for x0_p, x1_p in plan_pairs:
                    x0_p = true_projection(x0_p[None, :], grid)[0][0]
                    x1_p = true_projection(x1_p[None, :], grid)[0][0]
                    planned = plan_path(
                        model=delta_model,
                        x_start=x0_p,
                        x_goal=x1_p,
                        cfg=cfg,
                        planner_name=planner_name,
                        n_waypoints=n_waypoints,
                        dataset_name=name,
                        periodic_joint=bool(ds.get("periodic_joint", False)),
                        f_abs_stop=delta_eps_used,
                    )
                    if use_linear:
                        plans_proj.append(planned)
                    else:
                        plans_constr.append(planned)
                if plan_pairs_off is not None:
                    for x0_p, x1_p in plan_pairs_off:
                        x0_p = true_projection(x0_p[None, :], grid)[0][0]
                        planned_off = plan_path(
                            model=delta_model,
                            x_start=x0_p,
                            x_goal=x1_p,
                            cfg=cfg,
                            planner_name=planner_name,
                            n_waypoints=n_waypoints,
                            dataset_name=name,
                            periodic_joint=bool(ds.get("periodic_joint", False)),
                            f_abs_stop=delta_eps_used,
                            keep_endpoints=False,
                        )
                        if use_linear:
                            plans_proj_off.append(planned_off)
                        else:
                            plans_constr_off.append(planned_off)
                plot_planned_paths(
                    delta_model,
                    x_train,
                    grid,
                    plans_proj,
                    plans_constr,
                    vis_cfg_delta,
                    out_path=os.path.join(out_dir, "delta_planner_paths.png"),
                    title=f"{name}: Delta Planned Paths",
                    zero_level_eps=delta_eps_used,
                )
                if plans_proj_off or plans_constr_off:
                    plot_planned_paths_off(
                        delta_model,
                        x_train,
                        grid,
                        plans_proj_off,
                        plans_constr_off,
                        vis_cfg_delta,
                        out_path=os.path.join(out_dir, "delta_planner_paths_off.png"),
                        title=f"{name}: Delta Planned Paths (Off-Manifold)",
                        zero_level_eps=delta_eps_used,
                    )
            plot_worst_distance_contour(
                delta_model,
                x_train,
                grid,
                x0,
                delta_traj,
                vis_cfg_delta,
                out_path=os.path.join(out_dir, "delta_worst_distance_contour.png"),
                title=f"{name}: Delta Worst Distance (top 5%)",
                zero_level_eps=delta_eps_used,
            )
            plot_loss_curves(
                delta_history,
                out_path=os.path.join(out_dir, "delta_loss_curves.png"),
                title=f"{name}: Delta Baseline Losses",
                cfg=vis_cfg_delta,
            )

        print(f"[{name}] projection steps:", steps)
        if margin_stats:
            print(f"[{name}] baseline margin losses:", margin_stats)
        if delta_stats:
            print(f"[{name}] dataaug losses:", delta_stats)

        if margin_model is not None:
            post_fn = _wrap_np_pi if _is_arm_dataset(name) else None
            use_pybullet_n6 = str(name) == "6d_spatial_arm_up_n6"
            embed_fn = (
                lambda q, _name=name: shared_workspace_embed_for_eval(
                    _name,
                    q,
                    ur5_use_pybullet_n6=use_pybullet_n6,
                )
            ) if _is_arm_dataset(name) else None
            project_fn_margin = _make_project_fn(cfg)
            metrics_map, _, _ = run_eval_metrics(
                cfg=cfg,
                method_key="margin",
                dataset_name=name,
                model=margin_model,
                x_train=x_train,
                project_fn=project_fn_margin,
                embed_fn=embed_fn,
                postprocess_fn=post_fn,
            )
            eval_results.append(
                {
                    "dataset": name,
                    "method": "margin",
                    "metrics": metrics_map,
                }
            )
        if delta_model is not None:
            post_fn = _wrap_np_pi if _is_arm_dataset(name) else None
            use_pybullet_n6 = str(name) == "6d_spatial_arm_up_n6"
            embed_fn = (
                lambda q, _name=name: shared_workspace_embed_for_eval(
                    _name,
                    q,
                    ur5_use_pybullet_n6=use_pybullet_n6,
                )
            ) if _is_arm_dataset(name) else None
            project_fn_delta = _make_project_fn(cfg)
            metrics_map, _, _ = run_eval_metrics(
                cfg=cfg,
                method_key="dataaug",
                dataset_name=name,
                model=delta_model,
                x_train=x_train,
                project_fn=project_fn_delta,
                embed_fn=embed_fn,
                postprocess_fn=post_fn,
            )
            eval_results.append(
                {
                    "dataset": name,
                    "method": "dataaug",
                    "metrics": metrics_map,
                }
            )

    if eval_results:
        metrics_out = os.path.join(output_root, "metrics.json")
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        metrics_txt = os.path.join(output_root, "metrics.txt")
        lines = []
        lines.append("=== Evaluation Metrics (per dataset) ===")
        print("\n=== Evaluation Metrics (per dataset) ===")
        for entry in eval_results:
            m = entry["metrics"]
            line1 = (
                f"[eval] {entry['dataset']} | method={entry['method']} "
                f"| proj_dist={m['proj_manifold_dist']:.6f} "
                f"| pred_recall={m['pred_recall']:.6f} "
                f"| pred_FPrate={m['pred_FPrate']:.6f} "
                f"| chamfer={m['bidirectional_chamfer']:.6f} "
                f"| gt->learned={m['gt_to_learned_mean']:.6f} "
                f"| learned->gt={m['learned_to_gt_mean']:.6f} "
                f"| space={m.get('dist_space', 'unknown')}"
            )
            line2 = (
                f"[eval] {entry['dataset']} | method={entry['method']} "
                f"| proj_steps={m['proj_steps']:.2f} "
                f"| proj_true_dist={m['proj_true_dist']:.6f} "
                f"| proj_v_residual={m['proj_v_residual']:.6f} "
                f"| eval_eps={m['eval_eps_used']:.6f} "
                f"| pred_precision={m['pred_precision']:.6f}"
            )
            lines.append(line1)
            lines.append(line2)
            print(line1)
            print(line2)
        lines.append("")
        lines.append("=== Evaluation Metrics (mean over datasets) ===")
        print("\n=== Evaluation Metrics (mean over datasets) ===")
        by_method: Dict[str, List[Dict[str, float]]] = {}
        for entry in eval_results:
            by_method.setdefault(entry["method"], []).append(entry["metrics"])
        for method, items in by_method.items():
            num_keys = []
            for k, v in items[0].items():
                if isinstance(v, (int, float, np.floating)):
                    num_keys.append(k)
            avg = {k: float(np.mean([float(it[k]) for it in items])) for k in num_keys}
            line1 = (
                f"[eval-avg] method={method} "
                f"| proj_dist={avg['proj_manifold_dist']:.6f} "
                f"| pred_recall={avg['pred_recall']:.6f} "
                f"| pred_FPrate={avg['pred_FPrate']:.6f} "
                f"| chamfer={avg['bidirectional_chamfer']:.6f} "
                f"| gt->learned={avg['gt_to_learned_mean']:.6f} "
                f"| learned->gt={avg['learned_to_gt_mean']:.6f}"
            )
            line2 = (
                f"[eval-avg] method={method} "
                f"| proj_steps={avg['proj_steps']:.2f} "
                f"| proj_true_dist={avg['proj_true_dist']:.6f} "
                f"| proj_v_residual={avg['proj_v_residual']:.6f} "
                f"| eval_eps={avg['eval_eps_used']:.6f} "
                f"| pred_precision={avg['pred_precision']:.6f}"
            )
            lines.append(line1)
            lines.append(line2)
            print(line1)
            print(line2)
        with open(metrics_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    from experiments.unified_experiment import run_one as _run_one_unified

    p = argparse.ArgumentParser(description="dataaug direct wrapper (unified runner)")
    p.add_argument("--method", default="margin,dataaug", help="margin|dataaug or comma-separated")
    p.add_argument("--dataset", default="2d_sharp_star", help="single or comma-separated datasets")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config-root", default="configs")
    p.add_argument("--override", action="append", default=[], help="dotted key=value override")
    args = p.parse_args()
    methods = [m.strip() for m in str(args.method).split(",") if m.strip()]
    datasets = [d.strip() for d in str(args.dataset).split(",") if d.strip()]
    for m in methods:
        for ds_name in datasets:
            print(f"[run] method={m} dataset={ds_name}")
            result, loaded_paths = _run_one_unified(
                method=m,
                dataset=ds_name,
                out_root=str(args.outdir),
                seed_override=args.seed,
                config_root=str(args.config_root),
                cli_overrides=list(args.override),
            )
            mm = result["metrics"]
            print(f"[cfg] loaded_layers={loaded_paths if loaded_paths else '[]'}")
            print(
                f"[done] method={m} dataset={ds_name} "
                f"proj_dist={mm.get('proj_manifold_dist', float('nan')):.6f} "
                f"recall={mm.get('pred_recall', float('nan')):.6f} "
                f"FPrate={mm.get('pred_FPrate', float('nan')):.6f}"
            )
