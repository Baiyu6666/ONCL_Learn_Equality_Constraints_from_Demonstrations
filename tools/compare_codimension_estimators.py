from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from experiments.config_loader import load_layered_config
from experiments.dataset_resolve import (
    BASE_2D_DATASETS,
    BASE_3D_DATASETS,
    BASE_4D_DATASETS,
    BASE_6D_DATASETS,
    BASE_12D_DATASETS,
    resolve_dataset,
)
from planning.kinematics import wrap_np_pi as _wrap_np_pi


def _all_base_datasets() -> list[str]:
    # Intentionally exclude 2D datasets.
    return list(BASE_3D_DATASETS) + list(BASE_4D_DATASETS) + list(BASE_6D_DATASETS) + list(BASE_12D_DATASETS)


def _knn_indices(x: np.ndarray, i: int, k: int) -> np.ndarray:
    k = int(max(2, min(k, len(x) - 1)))
    c = x[i]
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(x.astype(np.float64))
        _, nbr = tree.query(c.astype(np.float64), k=k + 1)
        nbr = np.asarray(nbr, dtype=np.int64).reshape(-1)
        nbr = nbr[nbr != i]
        if len(nbr) >= k:
            return nbr[:k]
    except Exception:
        pass
    d2 = np.sum((x - c.reshape(1, -1)) ** 2, axis=1)
    order = np.argsort(d2)
    return order[1 : k + 1].astype(np.int64)


def _knn_indices_periodic(x: np.ndarray, i: int, k: int, periodic_joint: bool) -> np.ndarray:
    if not periodic_joint:
        return _knn_indices(x, i, k)
    k = int(max(2, min(k, len(x) - 1)))
    c = x[i]
    d2 = np.sum(_wrap_np_pi(x - c.reshape(1, -1)) ** 2, axis=1)
    order = np.argsort(d2)
    return order[1 : k + 1].astype(np.int64)


def estimate_codim_oncl_details(
    x: np.ndarray,
    *,
    sample_ratio: float = 0.2,
    k_neighbors: int = 0,
    periodic_joint: bool = False,
    seed: int = 0,
    const_axis_std_ratio: float = 0.0,
) -> dict[str, object]:
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    std = np.std(x.astype(np.float64), axis=0)
    std_ref = max(float(np.max(std)), 1e-12)
    std_tol = max(1e-10, float(const_axis_std_ratio) * std_ref)
    keep_mask = std > std_tol
    n_const = int(np.sum(~keep_mask))
    x_eff = x[:, keep_mask] if np.any(keep_mask) else x
    d_eff = int(x_eff.shape[1])

    n_sample = int(max(16, min(n, round(float(sample_ratio) * n))))
    if int(k_neighbors) > 0:
        k = int(k_neighbors)
    else:
        k = int(np.clip(round(np.sqrt(n)), max(3, d_eff + 2), min(96, n - 1)))

    if d_eff < 2:
        est = int(np.clip(max(1, n_const), 1, max(1, d - 1)))
        return {
            "estimated_codim": est,
            "sample_idx": np.zeros((0,), dtype=np.int64),
            "local_codim": np.zeros((0,), dtype=np.int64),
            "eigengaps": np.zeros((0, 0), dtype=np.float64),
            "d_eff": d_eff,
            "n_const_axes": n_const,
            "k_neighbors": int(max(2, min(k, max(2, n - 1)))),
        }

    rng = np.random.default_rng(int(seed))
    sample_idx = rng.choice(n, size=n_sample, replace=False)
    local_codim = np.zeros((n_sample,), dtype=np.int64)
    eigengaps = np.zeros((n_sample, d_eff - 1), dtype=np.float64)
    eps = 1e-10

    for i, idx in enumerate(sample_idx):
        nbr = _knn_indices_periodic(x_eff, int(idx), int(k), periodic_joint)
        center = x_eff[int(idx)]
        neigh = x_eff[nbr]
        if periodic_joint:
            neigh = center.reshape(1, -1) + _wrap_np_pi(neigh - center.reshape(1, -1))
        mu = np.mean(neigh, axis=0, keepdims=True)
        xc = neigh - mu
        cov = (xc.T @ xc) / max(1, len(neigh))
        evals = np.linalg.eigvalsh(cov.astype(np.float64))
        evals = np.maximum(evals, eps)
        gaps = np.log(evals[1:] + eps) - np.log(evals[:-1] + eps)
        eigengaps[i] = gaps
        codim_eff = int(np.clip(np.argmax(gaps) + 1, 1, d_eff - 1))
        local_codim[i] = int(np.clip(codim_eff + n_const, 1, d - 1))

    counts = np.bincount(local_codim, minlength=d + 1)
    est = int(np.argmax(counts[1:]) + 1)
    return {
        "estimated_codim": est,
        "sample_idx": np.asarray(sample_idx, dtype=np.int64),
        "local_codim": np.asarray(local_codim, dtype=np.int64),
        "eigengaps": np.asarray(eigengaps, dtype=np.float64),
        "d_eff": d_eff,
        "n_const_axes": n_const,
        "k_neighbors": int(k),
    }


def _plot_oncl_mismatch_details(
    outdir: str,
    dataset: str,
    true_codim: int,
    details: dict[str, object],
    seed: int,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    local_codim = np.asarray(details["local_codim"], dtype=np.int64)
    eigengaps = np.asarray(details["eigengaps"], dtype=np.float64)
    if local_codim.size == 0 or eigengaps.size == 0:
        return
    d_eff = int(details["d_eff"])
    est_codim = int(details["estimated_codim"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8))

    cmin = int(np.min(local_codim))
    cmax = int(np.max(local_codim))
    bins = np.arange(cmin, cmax + 2)
    ax0.hist(local_codim, bins=bins, color="#4C78A8", edgecolor="white", rwidth=0.88)
    ax0.axvline(float(true_codim), color="#2E8B57", ls="--", lw=2.0, label=f"true={true_codim}")
    ax0.axvline(float(est_codim), color="#E45756", ls="-", lw=2.0, label=f"mode={est_codim}")
    ax0.set_xlabel("Local Estimated Codim")
    ax0.set_ylabel("Count")
    ax0.set_title(f"{dataset}: Local Codim Distribution")
    ax0.legend(frameon=False, fontsize=9)

    rng = np.random.default_rng(int(seed) + 12345)
    idx_ok = np.where(local_codim == int(true_codim))[0]
    idx_bad = np.where(local_codim != int(true_codim))[0]
    n_ok = min(4, len(idx_ok))
    n_bad = min(4, len(idx_bad))
    pick_ok = rng.choice(idx_ok, size=n_ok, replace=False) if n_ok > 0 else np.array([], dtype=np.int64)
    pick_bad = rng.choice(idx_bad, size=n_bad, replace=False) if n_bad > 0 else np.array([], dtype=np.int64)

    xdim = np.arange(1, d_eff, dtype=np.int64)
    for j, ii in enumerate(pick_ok):
        y = eigengaps[int(ii)]
        ax1.plot(xdim, y, color="#2A9D8F", lw=1.7, alpha=0.9, label=("correct" if j == 0 else None))
        ax1.scatter([int(np.argmax(y) + 1)], [float(np.max(y))], color="#2A9D8F", s=18)
    for j, ii in enumerate(pick_bad):
        y = eigengaps[int(ii)]
        ax1.plot(xdim, y, color="#D1495B", lw=1.7, alpha=0.9, label=("wrong" if j == 0 else None))
        ax1.scatter([int(np.argmax(y) + 1)], [float(np.max(y))], color="#D1495B", s=18)
    ax1.axvline(float(true_codim), color="#2E8B57", ls="--", lw=1.8, label="true codim")
    ax1.set_xlabel("Split Dimension Index")
    ax1.set_ylabel("Eigen Gap (log λ[j+1] - log λ[j])")
    ax1.set_title(f"{dataset}: Eigengap Curves (Correct vs Wrong)")
    ax1.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(outdir, f"{dataset}_oncl_codim_mismatch_analysis.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def estimate_codim_ecomann_style(
    x: np.ndarray,
    *,
    n_local_neighborhood_mult: float = 1.0,
) -> int:
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    if n < 4 or d < 2:
        return max(1, d - 1)

    k_float = float(n_local_neighborhood_mult) * 2.0 * (2.0 ** float(d))
    k = int(max(2, min(n - 1, round(k_float))))

    cov_list = []
    for i in range(n):
        nbr = _knn_indices(x, i, k)
        x_nb = x[nbr] - x[i].reshape(1, -1)
        cov = (x_nb.T @ x_nb) / max(1, k - 1)
        cov_list.append(cov.astype(np.float64))
    cov_stack = np.stack(cov_list, axis=0)

    # Match EcoMaNN logic in ecmnn_dataset_loader.py:
    # dim_tangent = mode(argmax(s_j - s_{j+1})) + 1, dim_normal = d - dim_tangent
    svals = np.linalg.svd(cov_stack, compute_uv=False, full_matrices=False)
    if d < 2:
        return 1
    diff = svals[:, : (d - 1)] - svals[:, 1:]
    argmax_drop = np.argmax(diff, axis=1)
    counts = np.bincount(argmax_drop, minlength=max(1, d - 1))
    mode_idx = int(np.argmax(counts))
    dim_tangent = int(mode_idx + 1)
    dim_normal = int(d - dim_tangent)
    return max(1, min(d - 1, dim_normal))


def _format_table(rows: list[dict[str, object]]) -> str:
    headers = ["dataset", "dim", "true_codim", "oncl_est", "ecomann_est"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r[h])))
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    body = [" | ".join(str(r[h]).ljust(widths[h]) for h in headers) for r in rows]
    return "\n".join([line, sep] + body)


def _confusion_matrix(
    rows: list[dict[str, object]],
    est_key: str,
) -> tuple[list[int], np.ndarray]:
    vals_true = sorted({int(r["true_codim"]) for r in rows})
    vals_est = sorted({int(r[est_key]) for r in rows})
    vals = sorted(set(vals_true) | set(vals_est))
    idx = {v: i for i, v in enumerate(vals)}
    m = np.zeros((len(vals), len(vals)), dtype=np.int64)
    for r in rows:
        i = idx[int(r["true_codim"])]
        j = idx[int(r[est_key])]
        m[i, j] += 1
    return vals, m


def _format_confusion(title: str, vals: list[int], m: np.ndarray) -> str:
    col_w = max(5, max(len(str(v)) for v in vals))
    header = "true\\est".ljust(8) + " " + " ".join(str(v).rjust(col_w) for v in vals)
    lines = [title, header]
    for i, v in enumerate(vals):
        row = str(v).ljust(8) + " " + " ".join(str(int(m[i, j])).rjust(col_w) for j in range(len(vals)))
        lines.append(row)
    return "\n".join(lines)


def _iter_datasets(cli_datasets: str | None) -> Iterable[str]:
    if cli_datasets is None or str(cli_datasets).strip() == "":
        return _all_base_datasets()
    out = []
    for s in [v.strip() for v in str(cli_datasets).split(",") if v.strip()]:
        if s in BASE_2D_DATASETS:
            print(f"[skip] 2D dataset ignored: {s}")
            continue
        out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare true codim with oncl/ecomann codim estimators using current layered configs."
    )
    parser.add_argument("--config-root", type=str, default="configs")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names. Default: all base datasets.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for both methods. Default: keep config seed.")
    parser.add_argument("--n-train", type=int, default=None, help="Override n_train for both methods.")
    parser.add_argument("--n-grid", type=int, default=None, help="Override n_grid for both methods.")
    parser.add_argument("--outdir", type=str, default="outputs/analysis/codim_estimators")
    parser.add_argument(
        "--ur5-backend",
        type=str,
        default="auto",
        choices=["auto", "pybullet", "analytic"],
        help="UR5 backend for 6d_spatial_arm_up_* datasets.",
    )
    parser.add_argument(
        "--ur5-train-only",
        action="store_true",
        default=True,
        help="Use optimize_ur5_train_only=True to enable UR5 cache reuse by n_train/seed.",
    )
    parser.add_argument(
        "--no-ur5-train-only",
        dest="ur5_train_only",
        action="store_false",
        help="Disable optimize_ur5_train_only for UR5 datasets.",
    )
    args = parser.parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_root = str(args.config_root)
    if not os.path.isabs(config_root):
        cand = os.path.join(project_root, config_root)
        if os.path.isdir(cand):
            config_root = cand
        else:
            config_root = os.path.abspath(config_root)

    rows: list[dict[str, object]] = []
    outdir_arg = str(args.outdir).strip()
    if os.path.isabs(outdir_arg):
        outdir = outdir_arg
    else:
        outdir = os.path.join(project_root, outdir_arg)
    os.makedirs(outdir, exist_ok=True)
    for name in _iter_datasets(args.datasets):
        eik_map, _ = load_layered_config(config_root, "oncl", str(name))
        eco_map, _ = load_layered_config(config_root, "ecomann", str(name))
        if not eik_map:
            raise ValueError(
                f"missing layered config for method=oncl dataset={name} under config_root={config_root}"
            )
        if not eco_map:
            raise ValueError(
                f"missing layered config for method=ecomann dataset={name} under config_root={config_root}"
            )

        if args.seed is not None:
            eik_map["seed"] = int(args.seed)
            eco_map["seed"] = int(args.seed)
        if args.n_train is not None:
            eik_map["n_train"] = int(args.n_train)
            eco_map["n_train"] = int(args.n_train)
        if args.n_grid is not None:
            eik_map["n_grid"] = int(args.n_grid)
            eco_map["n_grid"] = int(args.n_grid)

        # Resolve dataset only once so both estimators use identical data.
        data_map = dict(eik_map)
        if args.seed is not None:
            data_map["seed"] = int(args.seed)
        if args.n_train is not None:
            data_map["n_train"] = int(args.n_train)
        if args.n_grid is not None:
            data_map["n_grid"] = int(args.n_grid)

        ds_name = str(name)
        if args.ur5_backend == "auto":
            ur5_backend = "pybullet" if ds_name == "6d_spatial_arm_up_n6" else "analytic"
        else:
            ur5_backend = str(args.ur5_backend)
        ds = resolve_dataset(
            ds_name,
            SimpleNamespace(**data_map),
            optimize_ur5_train_only=bool(args.ur5_train_only),
            ur5_backend=ur5_backend,
        )
        x_data = np.asarray(ds["x_train"], dtype=np.float32)
        d = int(x_data.shape[1])
        true_c = int(ds.get("true_codim", max(1, d - 1)))
        periodic = bool(ds.get("periodic_joint", False))
        eik = estimate_codim_oncl_details(
            x_data,
            sample_ratio=float(eik_map.get("codim_auto_sample_ratio", 0.2)),
            k_neighbors=int(eik_map.get("codim_auto_k_neighbors", 0)),
            periodic_joint=periodic,
            seed=int(eik_map.get("seed", 0)) + 1000,
            const_axis_std_ratio=float(eik_map.get("codim_auto_const_axis_std_ratio", 0.0)),
        )
        eik_c = int(eik["estimated_codim"])
        if eik_c != true_c:
            _plot_oncl_mismatch_details(
                outdir=outdir,
                dataset=ds_name,
                true_codim=true_c,
                details=eik,
                seed=int(eik_map.get("seed", 0)),
            )

        eco_c = int(
            estimate_codim_ecomann_style(
                x_data,
                n_local_neighborhood_mult=float(eco_map.get("n_local_neighborhood_mult", 1.0)),
            )
        )
        rows.append(
            {
                "dataset": str(name),
                "dim": d,
                "true_codim": true_c,
                "oncl_est": eik_c,
                "ecomann_est": eco_c,
            }
        )

    rows.sort(key=lambda r: (int(r["dim"]), str(r["dataset"])))
    print(_format_table(rows))
    print()

    vals_eik, m_eik = _confusion_matrix(rows, "oncl_est")
    print(_format_confusion("Confusion Matrix: true vs oncl_est", vals_eik, m_eik))
    print()
    vals_eco, m_eco = _confusion_matrix(rows, "ecomann_est")
    print(_format_confusion("Confusion Matrix: true vs ecomann_est", vals_eco, m_eco))
    print()
    print(f"[saved] mismatch plots: {outdir}")


if __name__ == "__main__":
    main()
