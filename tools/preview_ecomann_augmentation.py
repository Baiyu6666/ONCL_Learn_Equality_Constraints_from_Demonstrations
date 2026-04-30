#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import fields
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from experiments.config_loader import apply_overrides, load_layered_config
from experiments.dataset_resolve import resolve_dataset
from datasets.constraint_datasets import set_seed
from methods.dataaug import knn_normal_bases
from methods import ecomann as ecomann_base


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_outdir(outdir: str) -> str:
    p = str(outdir).strip() or "outputs/analysis/ecomann_aug_preview"
    if os.path.isabs(p):
        return p
    return os.path.join(_project_root(), p)


def _build_cfg(mapping: dict) -> ecomann_base.Config:
    allowed = {f.name for f in fields(ecomann_base.Config)}
    kwargs = {k: v for k, v in mapping.items() if k in allowed}
    return ecomann_base.Config(**kwargs)


def _make_loader(cfg: ecomann_base.Config, x_train: np.ndarray):
    smp_root = ecomann_base._add_smp_repo_to_path()
    from smp_manifold_learning.dataset_loader.ecmnn_dataset_loader import ECMNNDatasetLoader

    os.makedirs(os.path.join(smp_root, "data", "augmented"), exist_ok=True)
    tmp_dir = os.path.join(smp_root, "data", "augmented")
    ds_base = os.path.join(tmp_dir, f"preview_{int(cfg.seed)}")
    np.save(ds_base + ".npy", x_train.astype(np.float32))

    old = os.getcwd()
    try:
        os.chdir(smp_root)
        loader = ECMNNDatasetLoader(
            ds_base,
            is_performing_data_augmentation=bool(cfg.is_performing_data_augmentation),
            N_normal_space_traversal=int(cfg.n_normal_space_traversal),
            is_optimizing_signed_siamese_pairs=bool(cfg.is_optimizing_signed_siamese_pairs),
            clean_aug_data=bool(cfg.clean_aug_data),
            is_aligning_lpca_normal_space_eigvecs=bool(cfg.is_aligning_lpca_normal_space_eigvecs),
            is_augmenting_w_rand_comb_of_normaleigvecs=bool(cfg.is_augmenting_w_rand_comb_of_normaleigvecs),
            rand_seed=int(cfg.seed),
            N_local_neighborhood_mult=float(cfg.n_local_neighborhood_mult),
        )
    finally:
        os.chdir(old)

    return loader


def _plot_knn_normals_2d(x_train: np.ndarray, out_path: str) -> None:
    cfg_dummy = SimpleNamespace(knn_norm_estimation_ratio=0.08, knn_norm_estimation_min_points=4)
    k = max(int(round(float(cfg_dummy.knn_norm_estimation_ratio) * len(x_train))), int(cfg_dummy.knn_norm_estimation_min_points))
    n_basis = knn_normal_bases(x_train.astype(np.float32), k=k, codim=1, cfg=None)
    nvec = n_basis[:, :, 0]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_train[:, 0], x_train[:, 1], s=8, c="gray", alpha=0.8, label="on-manifold")
    n = len(x_train)
    idx = np.linspace(0, max(0, n - 1), num=min(64, n), dtype=int)
    scale = 0.15 * float(np.mean(np.max(x_train, axis=0) - np.min(x_train, axis=0)))
    plt.quiver(
        x_train[idx, 0],
        x_train[idx, 1],
        nvec[idx, 0],
        nvec[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / max(scale, 1e-6),
        width=0.002,
        color="tab:green",
        alpha=0.9,
        label="KNN normal",
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("EcoMaNN KNN Normals (2D)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_aug_points_2d(x_all: np.ndarray, level: np.ndarray, out_path: str) -> None:
    level = level.reshape(-1)
    on_mask = np.isclose(level, 0.0)
    off_mask = ~on_mask
    plt.figure(figsize=(8, 6))
    if np.any(off_mask):
        sc = plt.scatter(
            x_all[off_mask, 0],
            x_all[off_mask, 1],
            c=level[off_mask],
            s=7,
            cmap="viridis",
            alpha=0.7,
            label="augmented (off-manifold)",
        )
        cb = plt.colorbar(sc)
        cb.set_label("norm_level_data")
    plt.scatter(x_all[on_mask, 0], x_all[on_mask, 1], s=10, c="black", alpha=0.85, label="on-manifold")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("EcoMaNN Augmented Data (2D)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _reconstruct_filtered_aug_points_2d(
    x_train: np.ndarray,
    data: dict,
    cfg: ecomann_base.Config,
) -> np.ndarray:
    if x_train.shape[1] != 2:
        return np.zeros((0, x_train.shape[1]), dtype=np.float32)
    if not bool(cfg.clean_aug_data):
        return np.zeros((0, x_train.shape[1]), dtype=np.float32)
    if not bool(cfg.is_performing_data_augmentation):
        return np.zeros((0, x_train.shape[1]), dtype=np.float32)

    cov_null = np.asarray(data["cov_nullspace"], dtype=np.float32)
    cov_svd_s = np.asarray(data["cov_svd_s"], dtype=np.float32)
    n_on = int(x_train.shape[0])
    cov_null = cov_null[:n_on]
    cov_svd_s = cov_svd_s[:n_on]

    dim_ambient = int(x_train.shape[1])
    dim_normal = int(cov_null.shape[2])
    dim_tangent = int(dim_ambient - dim_normal)
    mean_tangent_eig = float(np.mean(cov_svd_s[:, :dim_tangent]))
    eps = float(cfg.n_local_neighborhood_mult) * float(np.sqrt(max(mean_tangent_eig, 0.0)))
    aug_clean_thresh = 1e-1
    kd_tree = cKDTree(data=x_train)

    filtered_pts: list[np.ndarray] = []
    for i in range(1, int(cfg.n_normal_space_traversal) + 1):
        unsigned_level_mult = eps * i
        level_mult_eigvec_list: list[np.ndarray] = []
        for d in range(dim_normal):
            for sign in (-1.0, 1.0):
                level_mult_eigvec = sign * unsigned_level_mult * cov_null[:, :, d]
                level_mult_eigvec_list.append(level_mult_eigvec)

        for level_mult_eigvec in level_mult_eigvec_list:
            new_data = x_train + level_mult_eigvec
            for idx, x in enumerate(new_data):
                _, idx_near = kd_tree.query(x)
                if np.linalg.norm(new_data[idx_near] - x) > (aug_clean_thresh * unsigned_level_mult):
                    filtered_pts.append(new_data[idx])

    if not filtered_pts:
        return np.zeros((0, x_train.shape[1]), dtype=np.float32)
    return np.asarray(filtered_pts, dtype=np.float32)


def _plot_aug_points_2d_with_filtered(
    x_all: np.ndarray,
    level: np.ndarray,
    filtered_pts: np.ndarray,
    out_path: str,
) -> None:
    level = level.reshape(-1)
    on_mask = np.isclose(level, 0.0)
    off_mask = ~on_mask
    plt.figure(figsize=(8, 6))
    if np.any(off_mask):
        sc = plt.scatter(
            x_all[off_mask, 0],
            x_all[off_mask, 1],
            c=level[off_mask],
            s=7,
            cmap="viridis",
            alpha=0.7,
            label="augmented (accepted)",
        )
        cb = plt.colorbar(sc)
        cb.set_label("norm_level_data")
    plt.scatter(x_all[on_mask, 0], x_all[on_mask, 1], s=10, c="black", alpha=0.85, label="on-manifold")
    if filtered_pts.size > 0:
        plt.scatter(
            filtered_pts[:, 0],
            filtered_pts[:, 1],
            s=30,
            facecolors="none",
            edgecolors="red",
            linewidths=1.0,
            alpha=0.9,
            label="filtered by KNN clean",
        )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("EcoMaNN Augmented Data (2D)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview EcoMaNN augmentation quality.")
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--seed", default=173, type=int)
    ap.add_argument("--config-root", default="configs", type=str)
    ap.add_argument("--outdir", default="outputs/analysis/ecomann_aug_preview", type=str)
    ap.add_argument("--override", action="append", default=[], help="dotted key=value override")
    args = ap.parse_args()

    cfg_map, _ = load_layered_config(args.config_root, "ecomann", args.dataset)
    cfg_map = apply_overrides(cfg_map, args.override)
    cfg = _build_cfg(cfg_map)
    cfg.seed = int(args.seed)

    set_seed(int(cfg.seed))
    ds = resolve_dataset(
        args.dataset,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=("pybullet" if str(args.dataset) == "6d_spatial_arm_up_n6" else "analytic"),
    )
    x_train = np.asarray(ds["x_train"], dtype=np.float32)
    if x_train.ndim != 2 or x_train.shape[1] != 2:
        raise ValueError(f"preview script currently supports 2D datasets only, got shape={x_train.shape}")

    loader = _make_loader(cfg, x_train)
    data = loader.dataset.data
    x_all = np.asarray(data["data"], dtype=np.float32)
    level = np.asarray(data["norm_level_data"], dtype=np.float32).reshape(-1)
    filtered_pts = _reconstruct_filtered_aug_points_2d(x_train, data, cfg)

    outdir = _resolve_outdir(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out_knn = os.path.join(outdir, f"{args.dataset}_seed{int(cfg.seed)}_knn_normals.png")
    out_aug = os.path.join(outdir, f"{args.dataset}_seed{int(cfg.seed)}_augmented_points.png")

    _plot_knn_normals_2d(x_train, out_knn)
    _plot_aug_points_2d_with_filtered(x_all, level, filtered_pts, out_aug)

    print(f"[saved] {out_knn}")
    print(f"[saved] {out_aug}")
    print(f"[stats] on={int(np.sum(np.isclose(level,0.0)))} off={int(np.sum(~np.isclose(level,0.0)))} total={int(len(level))}")
    print(f"[stats] filtered={int(filtered_pts.shape[0])}")


if __name__ == "__main__":
    main()
