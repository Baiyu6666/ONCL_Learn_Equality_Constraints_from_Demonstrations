from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Iterable

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_METHODS = ["dataaug", "oncl", "vae", "ecomann"]
DEFAULT_METRICS = [
    "proj_manifold_dist",
    "pred_precision",
    "train_seconds",
    "gt_to_learned_mean",
    "learned_to_gt_mean",
    "bidirectional_chamfer",
]


def _split_csv(text: str | None, default: Iterable[str] | None = None) -> list[str]:
    if text is None or str(text).strip() == "":
        return list(default) if default is not None else []
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _resolve_bench_dir(bench: str) -> str:
    b = str(bench).strip()
    if not b:
        raise ValueError("--bench is required")
    if os.path.isabs(b):
        return b
    p1 = os.path.abspath(os.path.join(_PROJECT_ROOT, "outputs", "bench", b))
    if os.path.isdir(p1):
        return p1
    p2 = os.path.abspath(os.path.join(_PROJECT_ROOT, b))
    if os.path.isdir(p2):
        return p2
    return os.path.abspath(b)


def _to_float(v: object, default: float = np.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


def _load_from_per_case_csv(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "dataset": str(r.get("dataset", "")).strip(),
                    "method": str(r.get("method", "")).strip(),
                    "seed": str(r.get("seed", "")).strip() or "0",
                    "bidirectional_chamfer": _to_float(r.get("bidirectional_chamfer", "")),
                    "proj_manifold_dist": _to_float(r.get("proj_manifold_dist", "")),
                    "pred_precision": _to_float(r.get("pred_precision", "")),
                    "train_seconds": _to_float(r.get("train_seconds", "")),
                    "gt_to_learned_mean": _to_float(r.get("gt_to_learned_mean", "")),
                    "learned_to_gt_mean": _to_float(r.get("learned_to_gt_mean", "")),
                }
            )
    return rows


def _load_from_per_run_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            metrics = r.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            rows.append(
                {
                    "dataset": str(r.get("dataset", "")).strip(),
                    "method": str(r.get("method", "")).strip(),
                    "seed": str(r.get("seed", "")).strip() or "0",
                    "bidirectional_chamfer": _to_float(metrics.get("bidirectional_chamfer", r.get("bidirectional_chamfer", ""))),
                    "proj_manifold_dist": _to_float(metrics.get("proj_manifold_dist", r.get("proj_manifold_dist", ""))),
                    "pred_precision": _to_float(metrics.get("pred_precision", r.get("pred_precision", ""))),
                    "train_seconds": _to_float(metrics.get("train_seconds", r.get("train_seconds", ""))),
                    "gt_to_learned_mean": _to_float(metrics.get("gt_to_learned_mean", r.get("gt_to_learned_mean", ""))),
                    "learned_to_gt_mean": _to_float(metrics.get("learned_to_gt_mean", r.get("learned_to_gt_mean", ""))),
                }
            )
    return rows


def _load_from_eval_jsons(bench_dir: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_dir in sorted(glob.glob(os.path.join(bench_dir, "*"))):
        if not os.path.isdir(method_dir):
            continue
        method = os.path.basename(method_dir)
        for p in sorted(glob.glob(os.path.join(method_dir, "*_eval.json"))):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            dataset = os.path.basename(p)
            if dataset.endswith("_eval.json"):
                dataset = dataset[: -len("_eval.json")]
            if dataset.endswith(f"_{method}"):
                dataset = dataset[: -(len(method) + 1)]
            dataset = dataset.replace("_oncl", "")
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": "last",
                    "bidirectional_chamfer": _to_float(d.get("bidirectional_chamfer", "")),
                    "proj_manifold_dist": _to_float(d.get("proj_manifold_dist", "")),
                    "pred_precision": _to_float(d.get("pred_precision", "")),
                    "train_seconds": _to_float(d.get("train_seconds", "")),
                    "gt_to_learned_mean": _to_float(d.get("gt_to_learned_mean", "")),
                    "learned_to_gt_mean": _to_float(d.get("learned_to_gt_mean", "")),
                }
            )
    return rows


def _load_rows(bench_dir: str) -> tuple[list[dict[str, object]], str]:
    per_run = os.path.join(bench_dir, "per_run_metrics.jsonl")
    if os.path.isfile(per_run):
        return _load_from_per_run_jsonl(per_run), "per_run_metrics.jsonl"
    per_case = os.path.join(bench_dir, "per_case_metrics.csv")
    if os.path.isfile(per_case):
        return _load_from_per_case_csv(per_case), "per_case_metrics.csv"
    return _load_from_eval_jsons(bench_dir), "*_eval.json"


def _build_samples(
    rows: list[dict[str, object]],
    datasets: list[str],
    methods: list[str],
    metric: str,
    fill_missing_zero: bool,
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {d: {m: [] for m in methods} for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        m = str(r.get("method", "")).strip()
        if d not in out or m not in out[d]:
            continue
        v = _to_float(r.get(metric, np.nan))
        if np.isfinite(v):
            out[d][m].append(float(v))
    if fill_missing_zero:
        for d in datasets:
            for m in methods:
                if len(out[d][m]) == 0:
                    out[d][m] = [0.0]
    return out


def _compute_mean_std(samples: dict[str, dict[str, list[float]]], datasets: list[str], methods: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    stds = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    for mi, m in enumerate(methods):
        for di, d in enumerate(datasets):
            vals = np.asarray(samples[d][m], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                means[mi, di] = 0.0
                stds[mi, di] = 0.0
            else:
                means[mi, di] = float(np.mean(vals))
                stds[mi, di] = float(np.std(vals, ddof=0))
    return means, stds


def _plot_grouped_bar_with_error(
    out_path: str,
    title: str,
    ylabel: str,
    datasets: list[str],
    methods: list[str],
    samples: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    means, stds = _compute_mean_std(samples, datasets=datasets, methods=methods)

    n_ds = len(datasets)
    n_m = max(1, len(methods))
    fig_w = max(12, 1.1 * n_ds)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    x = np.arange(n_ds, dtype=np.float32)
    width = 0.8 / n_m

    for mi, m in enumerate(methods):
        pos = x + (mi - (n_m - 1) / 2.0) * width
        ax.bar(pos, means[mi], width=width * 0.90, color=color_map[m], alpha=0.72, edgecolor=color_map[m], linewidth=1.0, label=m if mi == 0 else None, zorder=2)
        ax.errorbar(pos, means[mi], yerr=stds[mi], fmt="none", ecolor="#1F2937", elinewidth=1.2, capsize=2.8, capthick=1.0, alpha=0.95, zorder=3)

    for i in range(n_ds - 1):
        ax.axvline(x=float(i) + 0.5, color="#6B7280", linestyle="-", linewidth=1.2, alpha=0.5, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_xlim(-0.6, float(n_ds) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    handles = [plt.Line2D([0], [0], color=color_map[m], lw=6, alpha=0.7, label=m) for m in methods]
    ax.legend(handles=handles, frameon=False, ncol=min(4, len(methods)), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_matrix_csv(path: str, datasets: list[str], methods: list[str], samples: dict[str, dict[str, list[float]]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "n", "mean", "median", "std", "min", "max"])
        for d in datasets:
            for m in methods:
                vals = np.asarray(samples[d][m], dtype=np.float64)
                if vals.size == 0:
                    continue
                w.writerow([d, m, int(vals.size), float(np.mean(vals)), float(np.median(vals)), float(np.std(vals, ddof=0)), float(np.min(vals)), float(np.max(vals))])


def _normalize_per_dataset(
    rows: list[dict[str, object]],
    *,
    datasets: list[str],
    methods: list[str],
    metric: str,
    fill_missing_zero: bool,
) -> dict[str, dict[str, list[float]]]:
    per_ds_vals: dict[str, list[float]] = {d: [] for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        if d not in per_ds_vals:
            continue
        v = _to_float(r.get(metric, np.nan))
        if np.isfinite(v):
            per_ds_vals[d].append(float(v))
    out: dict[str, dict[str, list[float]]] = {d: {m: [] for m in methods} for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        m = str(r.get("method", "")).strip()
        if d not in out or m not in out[d]:
            continue
        v = _to_float(r.get(metric, np.nan))
        if not np.isfinite(v):
            continue
        ds_pool = np.asarray(per_ds_vals[d], dtype=np.float64)
        if ds_pool.size == 0:
            nv = 0.0
        else:
            lo = float(np.min(ds_pool))
            hi = float(np.max(ds_pool))
            nv = 0.5 if hi <= lo + 1e-12 else (float(v) - lo) / (hi - lo)
        out[d][m].append(float(np.clip(nv, 0.0, 1.0)))
    if fill_missing_zero:
        for d in datasets:
            for m in methods:
                if len(out[d][m]) == 0:
                    out[d][m] = [0.0]
    return out


def _save_cross_dataset_csv(path: str, methods: list[str], datasets: list[str], norm_samples: dict[str, dict[str, list[float]]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_over_datasets", "std_over_datasets", "n_datasets"])
        for m in methods:
            ds_means: list[float] = []
            for d in datasets:
                vals = np.asarray(norm_samples[d][m], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                ds_means.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
            arr = np.asarray(ds_means, dtype=np.float64)
            w.writerow([m, float(np.mean(arr)), float(np.std(arr, ddof=0)), int(arr.size)])


PAPER_DATASETS = [
    "2d_ellipse",
    "2d_planar_arm_line_n2",
    "2d_sparse_sine",
    "3d_planar_arm_line_n3_traj",
    "3d_spatial_arm_ellip_n3_traj",
    "3d_twosphere_traj",
    "3d_vz_2d_ellipse_traj",
    "3d_torus_surface_traj",
    "6d_spatial_arm_up_n6_py_traj",
    "6d_workspace_sine_surface_pose_traj",
    "12d_dual_arm_traj"
]

PAPER_DATASET_LABELS = {
    "2d_ellipse": "2DEllipse",
    "2d_planar_arm_line_n2": "2DPlanarArmLine",
    "2d_sparse_sine": "2DSineSparse",
    "3d_planar_arm_line_n3_traj": "3DPlanarArmLine",
    "3d_spatial_arm_ellip_n3_traj": "3DArmEllipse",
    "3d_twosphere_traj": "3DTwoSphere",
    "3d_vz_2d_ellipse_traj": "3DTwistEllipse",
    "3d_torus_surface_traj": "3DTorus",
    "6d_spatial_arm_up_n6_py_traj": "6DArmUp",
    "6d_workspace_sine_surface_pose_traj": "6DSinePose",
    "12d_dual_arm_traj": "12DDualArm",
}

METRIC_DISPLAY_LABELS = {
    "proj_manifold_dist": r"Distance Error $\epsilon_l$",
    "gt_to_learned_mean": r"Coverage Error $\epsilon_g$",
    "learned_to_gt_mean": r"Distance Error $\epsilon_l$",
    "bidirectional_chamfer": r"Bi-Chamfer $\epsilon_{\mathrm{bi}}$",
    "train_seconds": "Training Time",
}

METRIC_FILE_ALIASES = {
    "proj_manifold_dist": "distance_error",
    "gt_to_learned_mean": "coverage_error",
    "train_seconds": "training_time",
}

METRIC_INPUT_ALIASES = {
    "gt_to_learned": "gt_to_learned_mean",
    "learned_to_gt": "learned_to_gt_mean",
    "chamfer": "bidirectional_chamfer",
}

TRAJ_COMPARE_PAIRS = [
    # ("3d_planar_arm_line_n3", "3d_planar_arm_line_n3_traj", "3DPlanarArmLine"),
    ("3d_spatial_arm_ellip_n3", "3d_spatial_arm_ellip_n3_traj", "3DArmEllipse"),
    # ("3d_twosphere", "3d_twosphere_traj", "3DTwoSphere"),
    # ("3d_vz_2d_ellipse", "3d_vz_2d_ellipse_traj", "3DTwistedEliip"),
    ("3d_torus_surface", "3d_torus_surface_traj", "3DTorus"),
    ("6d_workspace_sine_surface_pose", "6d_workspace_sine_surface_pose_traj", "6DSinePose"),
]

TRAIN_TIME_TABLE_DATASETS = [
    "6d_workspace_sine_surface_pose_traj",
    "3d_spatial_arm_ellip_n3_traj",
    "6d_spatial_arm_up_n6_py_traj",
]

CODIM_COMPARE_DATASETS = [
    "3d_spatial_arm_ellip_n3_traj",
    # "3d_vz_2d_ellipse_traj",
    "6d_workspace_sine_surface_pose_traj",
    "6d_spatial_arm_up_n6_py_traj",
]

CODIM_HINTS = {
    "2d_ellipse": 1,
    "2d_planar_arm_line_n2": 1,
    "2d_sparse_sine": 1,
    "3d_planar_arm_line_n3": 1,
    "3d_planar_arm_line_n3_traj": 1,
    "3d_spatial_arm_ellip_n3": 2,
    "3d_spatial_arm_ellip_n3_traj": 2,
    "3d_twosphere": 1,
    "3d_twosphere_traj": 1,
    "3d_vz_2d_ellipse": 2,
    "3d_vz_2d_ellipse_traj": 2,
    "3d_torus_surface": 1,
    "3d_torus_surface_traj": 1,
    "6d_workspace_sine_surface_pose": 2,
    "6d_workspace_sine_surface_pose_traj": 2,
    "6d_spatial_arm_up_n6_py": 2,
    "6d_spatial_arm_up_n6_py_traj": 2,
    "12d_dual_arm_traj": 10,
}

METHOD_DISPLAY_LABELS = {
    "oncl": "ONCL (Ours)",
    "dataaug": "DataAug",
    "igr": "IGR",
    "vae": "VAE",
    "ecomann": "ECoMaNN",
}

METHOD_PRIORITY = ["oncl", "dataaug", "igr", "vae", "ecomann"]


def _to_display_dataset(dataset: str) -> str:
    return PAPER_DATASET_LABELS.get(str(dataset), str(dataset))


def _to_compact_display_dataset(dataset: str) -> str:
    label = _to_display_dataset(dataset)
    compact = {
        "3DArmEllipse": "3DArmEllip",
        "3DTwistEllipse": "3DTwistEllip",
        "3DPlanarArmLine": "3DPlanarArm",
        "2DPlanarArmLine": "2DPlanarArm",
        "6DSinePose": "6DSinePose",
        "12DDualArm": "12DDualArm",
    }
    return compact.get(label, label)


def _to_compact_display_dataset_multiline(dataset: str) -> str:
    label = _to_display_dataset(dataset)
    compact = {
        "3DArmEllipse": "3DArm\nEllip",
        "3DTwistEllipse": "3DTwist\nEllip",
        "6DArmUp": "6DArm\nUp",
        "6DSinePose": "6DSine\nPose",
    }
    return compact.get(label, _to_compact_display_dataset(dataset))


def _to_display_metric(metric: str) -> str:
    return METRIC_DISPLAY_LABELS.get(str(metric), str(metric))


def _to_metric_file_stem(metric: str) -> str:
    return METRIC_FILE_ALIASES.get(str(metric), str(metric))


def _normalize_metrics(metrics: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in metrics:
        mm = METRIC_INPUT_ALIASES.get(str(m), str(m))
        if mm in seen:
            continue
        seen.add(mm)
        out.append(mm)
    return out


def _to_display_method(method: str) -> str:
    return METHOD_DISPLAY_LABELS.get(str(method), str(method))


def _sort_methods_for_paper(methods: list[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(METHOD_PRIORITY)}
    return sorted(methods, key=lambda m: (rank.get(m, 10_000), m))


def _remap_samples(
    samples: dict[str, dict[str, list[float]]],
    datasets: list[str],
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    out: dict[str, dict[str, list[float]]] = {}
    display_datasets: list[str] = []
    for d in datasets:
        dd = _to_display_dataset(d)
        out[dd] = samples[d]
        display_datasets.append(dd)
    return out, display_datasets


def _remap_method_samples(
    samples: dict[str, dict[str, list[float]]],
    methods: list[str],
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    out: dict[str, dict[str, list[float]]] = {}
    display_methods: list[str] = []
    for m in methods:
        dm = _to_display_method(m)
        display_methods.append(dm)
    for d, m_map in samples.items():
        out[d] = {}
        for m in methods:
            dm = _to_display_method(m)
            out[d][dm] = m_map.get(m, [])
    return out, display_methods


def _vals_from_rows(rows: list[dict[str, object]], dataset: str, method: str, metric: str) -> np.ndarray:
    vals: list[float] = []
    for r in rows:
        if str(r.get("dataset", "")).strip() != str(dataset):
            continue
        if str(r.get("method", "")).strip() != str(method):
            continue
        v = _to_float(r.get(metric, np.nan))
        if np.isfinite(v):
            vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def _plot_traj_overlay_compare(
    *,
    out_path: str,
    title: str,
    ylabel: str,
    rows: list[dict[str, object]],
    methods: list[str],
    metric: str,
) -> None:
    import matplotlib.pyplot as plt

    default_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: default_colors[i % len(default_colors)] for i, m in enumerate(methods)}
    # Keep the overlay figure visually aligned with the codim comparison plot:
    # ECoMaNN should use the same red as the IGR bar in distance_error_codim_bar.
    color_map["ecomann"] = "#E45756"
    n_groups = len(TRAJ_COMPARE_PAIRS)
    n_m = max(1, len(methods))
    # Narrower single-column subfigure size for side-by-side layout.
    fig, ax = plt.subplots(figsize=(2.45, 1.5))
    centers = np.arange(n_groups, dtype=np.float64)
    width = 0.8 / n_m

    for gi, (base_ds, traj_ds, _) in enumerate(TRAJ_COMPARE_PAIRS):
        group_center = centers[gi]
        for mi, m in enumerate(methods):
            x = group_center + (mi - (n_m - 1) / 2.0) * width
            bar_w = width * 0.86
            vals_traj = _vals_from_rows(rows, traj_ds, m, metric)
            vals_base = _vals_from_rows(rows, base_ds, m, metric)
            mean_traj = float(np.mean(vals_traj)) if vals_traj.size > 0 else np.nan
            std_traj = float(np.std(vals_traj, ddof=0)) if vals_traj.size > 0 else 0.0
            mean_base = float(np.mean(vals_base)) if vals_base.size > 0 else np.nan
            std_base = float(np.std(vals_base, ddof=0)) if vals_base.size > 0 else 0.0

            if np.isfinite(mean_traj):
                ax.bar(
                    x,
                    mean_traj,
                    width=bar_w,
                    color=color_map[m],
                    edgecolor=color_map[m],
                    linewidth=1.0,
                    alpha=0.82,
                    zorder=2,
                )
            if np.isfinite(mean_base):
                ax.bar(
                    x,
                    mean_base,
                    width=bar_w,
                    color="none",
                    edgecolor="#6B7280",
                    linewidth=1.6,
                    hatch="///",
                    zorder=4,
                )

    for i in range(n_groups - 1):
        ax.axvline(float(i) + 0.5, color="#6B7280", linestyle="-", linewidth=1.1, alpha=0.45, zorder=0)

    ax.set_xticks(centers)
    ax.set_xticklabels([_to_compact_display_dataset(traj_ds) for _, traj_ds, _ in TRAJ_COMPARE_PAIRS])
    ax.tick_params(axis="x", labelsize=7.4, pad=3)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.yaxis.label.set_size(10)
    ax.grid(axis="y", alpha=0.22, zorder=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[m], edgecolor=color_map[m], alpha=0.85, label=_to_display_method(m))
        for m in methods
    ] + [
        plt.Rectangle((0, 0), 1, 1, facecolor="#9CA3AF", edgecolor="#4B5563", alpha=0.75, label="Trajectory demonstrations"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#4B5563", hatch="///", linewidth=1.4, label="Scatter data"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.4, -0.18),
        fontsize=6.2,
        columnspacing=0.9,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.21, right=0.99, top=0.93, bottom=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0.01)
    plt.close(fig)


def _plot_paper_disterror_grouped_bar(
    out_path: str,
    ylabel: str,
    datasets: list[str],
    methods: list[str],
    samples: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    means, stds = _compute_mean_std(samples, datasets=datasets, methods=methods)

    n_ds = len(datasets)
    n_m = max(1, len(methods))
    x = np.arange(n_ds, dtype=np.float32)
    width = 0.82 / n_m
    # Paper double-column friendly size, close to traj-vs-scatter styling.
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    for mi, m in enumerate(methods):
        pos = x + (mi - (n_m - 1) / 2.0) * width
        ax.bar(
            pos,
            means[mi],
            width=width * 0.90,
            color=color_map[m],
            alpha=0.82,
            edgecolor=color_map[m],
            linewidth=1.0,
            zorder=2,
            label=m,
        )
        ax.errorbar(
            pos,
            means[mi],
            yerr=stds[mi],
            fmt="none",
            ecolor="#1F2937",
            elinewidth=1.0,
            capsize=2.0,
            capthick=1.0,
            zorder=3,
        )

    for i in range(n_ds - 1):
        ax.axvline(float(i) + 0.5, color="#6B7280", linestyle="-", linewidth=1.0, alpha=0.35, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=18, ha="right")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(frameon=False, ncol=min(4, len(methods)), loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=7)

    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0), pad=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_paper_cross_dataset_average(
    out_path: str,
    methods: list[str],
    datasets: list[str],
    norm_samples: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    ds_method_mean = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    for mi, m in enumerate(methods):
        for di, d in enumerate(datasets):
            vals = np.asarray(norm_samples[d][m], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            ds_method_mean[mi, di] = float(np.mean(vals)) if vals.size > 0 else 0.0

    method_mean = np.mean(ds_method_mean, axis=1)
    method_std = np.std(ds_method_mean, axis=1, ddof=0)

    x = np.arange(len(methods), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(2.75, 1.35))
    bar_colors = [color_map[m] for m in methods]
    ax.bar(
        x,
        method_mean,
        width=0.66,
        color=bar_colors,
        alpha=0.82,
        edgecolor=bar_colors,
        linewidth=1.0,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(r"Normalized DistError $\bar{\epsilon}_d$", fontsize=9)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_codim_gt1_oncl_compare(
    *,
    out_path: str,
    ylabel: str,
    rows: list[dict[str, object]],
    datasets: list[str],
    metric: str,
) -> None:
    import matplotlib.pyplot as plt

    methods = ["oncl", "igr"]
    seen_methods = {str(r.get("method", "")).strip() for r in rows}
    if any(m not in seen_methods for m in methods):
        return

    allowed = set(CODIM_COMPARE_DATASETS)
    ds_gt1 = [
        d for d in datasets
        if d in allowed and int(CODIM_HINTS.get(str(d), 1)) > 1
    ]
    if not ds_gt1:
        return

    labels = [_to_compact_display_dataset(d) for d in ds_gt1]
    centers = np.arange(len(ds_gt1), dtype=np.float64)
    width = 0.34
    colors = {"oncl": "#4C78A8", "igr": "#E45756"}

    fig, ax = plt.subplots(figsize=(2.45, 1.5))
    for mi, m in enumerate(methods):
        xs = centers + (-0.5 if mi == 0 else 0.5) * width
        vals = []
        for d in ds_gt1:
            arr = _vals_from_rows(rows, d, m, metric)
            vals.append(float(np.mean(arr)) if arr.size > 0 else np.nan)
        vals_np = np.asarray(vals, dtype=np.float64)
        ok = np.isfinite(vals_np)
        if np.any(ok):
            ax.bar(
                xs[ok],
                vals_np[ok],
                width=width * 0.88,
                color=colors[m],
                edgecolor=colors[m],
                linewidth=1.0,
                alpha=0.85,
                label=_to_display_method(m),
                zorder=2,
            )

    for i in range(len(ds_gt1) - 1):
        ax.axvline(float(i) + 0.5, color="#6B7280", linestyle="-", linewidth=1.1, alpha=0.45, zorder=0)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=6.8, pad=4)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=2)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.4, borderaxespad=0.0)
    fig.subplots_adjust(left=0.21, right=0.99, top=0.93, bottom=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0.01)
    plt.close(fig)


def _write_training_time_table(
    *,
    outdir: str,
    rows: list[dict[str, object]],
    methods: list[str],
) -> None:
    ds_ids = list(TRAIN_TIME_TABLE_DATASETS)
    ds_labels = [_to_display_dataset(d) for d in ds_ids]
    mean_map: dict[tuple[str, str], float] = {}
    std_map: dict[tuple[str, str], float] = {}
    n_map: dict[tuple[str, str], int] = {}
    for d in ds_ids:
        for m in methods:
            arr = _vals_from_rows(rows, d, m, "train_seconds")
            arr = _remove_outliers_iqr(arr)
            n = int(arr.size)
            n_map[(d, m)] = n
            if n <= 0:
                mean_map[(d, m)] = float("nan")
                std_map[(d, m)] = float("nan")
            else:
                mean_map[(d, m)] = float(np.mean(arr))
                std_map[(d, m)] = float(np.std(arr, ddof=0))

    csv_path = os.path.join(outdir, "training_time_table.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["method"] + ds_labels
        w.writerow(header)
        for m in methods:
            row = [_to_display_method(m)]
            for d in ds_ids:
                mu = mean_map[(d, m)]
                sd = std_map[(d, m)]
                n = n_map[(d, m)]
                if n <= 0 or (not np.isfinite(mu)) or (not np.isfinite(sd)):
                    row.append("-")
                else:
                    row.append(f"{mu:.4f} ± {sd:.4f}")
            w.writerow(row)

    md_path = os.path.join(outdir, "training_time_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Method | " + " | ".join(ds_labels) + " |\n")
        f.write("|---" * (len(ds_labels) + 1) + "|\n")
        for m in methods:
            cells = []
            for d in ds_ids:
                mu = mean_map[(d, m)]
                sd = std_map[(d, m)]
                n = n_map[(d, m)]
                if n <= 0 or (not np.isfinite(mu)) or (not np.isfinite(sd)):
                    cells.append("-")
                else:
                    cells.append(f"{mu:.4f} ± {sd:.4f}")
            f.write("| " + _to_display_method(m) + " | " + " | ".join(cells) + " |\n")

    print(f"[saved] {csv_path}")
    print(f"[saved] {md_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-style benchmark grouped bar charts with display dataset names.")
    p.add_argument("--bench", type=str, default="paper_mix_2d_3d6d_traj_vs_nontraj_7seed", help="Benchmark name (under outputs/bench) or absolute path.")
    p.add_argument("--methods", type=str, default="dataaug,oncl,vae,ecomann", help="Comma-separated methods. Default: dataaug,oncl,vae,ecomann")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset ids. Default: paper preset (2D non-traj + 3D/6D traj).")
    p.add_argument("--metrics", type=str, default="proj_manifold_dist,gt_to_learned", help="Comma-separated metrics to plot.")
    p.add_argument("--analysis-dir", type=str, default="analysis_bars_paper", help="Output dir under bench dir.")
    p.add_argument("--fill-missing-zero", action="store_true", default=True)
    p.add_argument("--no-fill-missing-zero", dest="fill_missing_zero", action="store_false")
    p.add_argument("--traj-overlay-compare", default=True, help="Also plot traj vs non-traj overlay bars for selected dataset pairs.")
    p.add_argument("--traj-overlay-methods", type=str, default="oncl,ecomann", help="Comma-separated methods for traj/non-traj overlay plot only. Default: use --methods.")
    p.add_argument("--codim-gt1-compare", default=True, help="Also plot codim>1 dataset comparison between oncl and igr.")
    args = p.parse_args()

    bench_dir = _resolve_bench_dir(args.bench)
    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(f"benchmark directory not found: {bench_dir}")

    rows, source_name = _load_rows(bench_dir)
    if not rows:
        raise RuntimeError(f"no benchmark rows found under: {bench_dir}")

    methods = _split_csv(args.methods, DEFAULT_METHODS)
    metrics = _normalize_metrics(_split_csv(args.metrics, DEFAULT_METRICS))
    all_methods_in_rows = sorted({str(r.get("method", "")) for r in rows if str(r.get("method", ""))})
    if args.methods is None:
        methods = [m for m in DEFAULT_METHODS if m in all_methods_in_rows] + [m for m in all_methods_in_rows if m not in DEFAULT_METHODS]
    else:
        methods = [m for m in methods if m in all_methods_in_rows]
    methods = _sort_methods_for_paper(methods)
    overlay_methods = _split_csv(args.traj_overlay_methods, methods)
    overlay_methods = [m for m in overlay_methods if m in all_methods_in_rows]
    overlay_methods = _sort_methods_for_paper(overlay_methods)

    datasets = _split_csv(args.datasets, None)
    if not datasets:
        datasets = list(PAPER_DATASETS)

    outdir = os.path.join(bench_dir, args.analysis_dir)
    os.makedirs(outdir, exist_ok=True)

    for metric in metrics:
        metric_label = _to_display_metric(metric)
        metric_datasets = list(TRAIN_TIME_TABLE_DATASETS) if str(metric) == "train_seconds" else list(datasets)
        metric_rows = rows
        if str(metric) == "train_seconds":
            metric_rows = _rows_with_train_seconds_outliers_removed(rows, dataset_allow=set(metric_datasets))
        samples_raw = _build_samples(
            metric_rows,
            metric_datasets,
            methods,
            metric=metric,
            fill_missing_zero=bool(args.fill_missing_zero),
        )
        samples_ds, display_datasets = _remap_samples(samples_raw, metric_datasets)
        samples, display_methods = _remap_method_samples(samples_ds, methods)
        metric_stem = _to_metric_file_stem(metric)
        png_name = f"{metric_stem}_bar.png"
        csv_name = f"{metric_stem}_bar_stats.csv"
        out_png = os.path.join(outdir, png_name)
        out_csv = os.path.join(outdir, csv_name)
        if str(metric) in {
            "proj_manifold_dist",
            "gt_to_learned_mean",
            "learned_to_gt_mean",
            "bidirectional_chamfer",
            "train_seconds",
        }:
            _plot_paper_disterror_grouped_bar(
                out_path=out_png,
                ylabel=metric_label,
                datasets=display_datasets,
                methods=display_methods,
                samples=samples,
            )
        else:
            _plot_grouped_bar_with_error(
                out_png,
                title=f"Paper Benchmark ({metric_label})",
                ylabel=metric_label,
                datasets=display_datasets,
                methods=display_methods,
                samples=samples,
            )
        _save_matrix_csv(out_csv, datasets=display_datasets, methods=display_methods, samples=samples)
        print(f"[saved] {out_png}")
        print(f"[saved] {out_csv}")

        if str(metric) == "train_seconds":
            continue

        norm_samples_raw = _normalize_per_dataset(
            metric_rows,
            datasets=metric_datasets,
            methods=methods,
            metric=metric,
            fill_missing_zero=bool(args.fill_missing_zero),
        )
        norm_samples_ds, norm_display_datasets = _remap_samples(norm_samples_raw, metric_datasets)
        norm_samples, norm_display_methods = _remap_method_samples(norm_samples_ds, methods)
        out_png_cross = os.path.join(outdir, f"{metric_stem}_normalized_bar.png")
        out_csv_cross = os.path.join(outdir, f"{metric_stem}_normalized_bar_stats.csv")
        _plot_paper_cross_dataset_average(
            out_png_cross,
            methods=norm_display_methods,
            datasets=norm_display_datasets,
            norm_samples=norm_samples,
        )
        _save_cross_dataset_csv(out_csv_cross, methods=norm_display_methods, datasets=norm_display_datasets, norm_samples=norm_samples)
        print(f"[saved] {out_png_cross}")
        print(f"[saved] {out_csv_cross}")

        if bool(args.traj_overlay_compare):
            out_overlay = os.path.join(outdir, f"{metric_stem}_traj_vs_scatter.png")
            _plot_traj_overlay_compare(
                out_path=out_overlay,
                title=f"Traj vs Non-traj Overlay ({metric_label})",
                ylabel=metric_label,
                rows=rows,
                methods=overlay_methods,
                metric=metric,
            )
            print(f"[saved] {out_overlay}")
        if bool(args.codim_gt1_compare):
            out_codim = os.path.join(outdir, f"{metric_stem}_codim_bar.png")
            _plot_codim_gt1_oncl_compare(
                out_path=out_codim,
                ylabel=metric_label,
                rows=rows,
                datasets=datasets,
                metric=metric,
            )
            if os.path.isfile(out_codim):
                print(f"[saved] {out_codim}")

    print(f"[info] bench={bench_dir}")
    print(f"[info] source={source_name}")
    print(f"[info] methods={','.join([_to_display_method(m) for m in methods])}")
    print(f"[info] datasets={','.join([_to_display_dataset(d) for d in datasets])}")
    _write_training_time_table(outdir=outdir, rows=rows, methods=methods)


def _remove_outliers_iqr(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return x
    q1 = float(np.percentile(x, 25.0))
    q3 = float(np.percentile(x, 75.0))
    iqr = q3 - q1
    if iqr <= 1e-12:
        return x
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    y = x[(x >= lo) & (x <= hi)]
    if y.size == 0:
        return x
    return y


def _rows_with_train_seconds_outliers_removed(
    rows: list[dict[str, object]],
    dataset_allow: set[str] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        m = str(r.get("method", "")).strip()
        grouped.setdefault((d, m), []).append(r)

    out: list[dict[str, object]] = []
    for (d, m), items in grouped.items():
        if dataset_allow is not None and d not in dataset_allow:
            out.extend(items)
            continue
        vals = []
        keep_idx = []
        for i, r in enumerate(items):
            v = _to_float(r.get("train_seconds", np.nan))
            if np.isfinite(v):
                vals.append(float(v))
                keep_idx.append(i)
        if len(vals) < 4:
            out.extend(items)
            continue
        arr = np.asarray(vals, dtype=np.float64)
        q1 = float(np.percentile(arr, 25.0))
        q3 = float(np.percentile(arr, 75.0))
        iqr = q3 - q1
        if iqr <= 1e-12:
            out.extend(items)
            continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        removed = 0
        for i, r in enumerate(items):
            v = _to_float(r.get("train_seconds", np.nan))
            if np.isfinite(v) and (v < lo or v > hi):
                removed += 1
                continue
            out.append(r)
        if removed > 0:
            print(f"[info][training_time] removed outliers for {d}/{m}: {removed}")
    return out


if __name__ == "__main__":
    main()
