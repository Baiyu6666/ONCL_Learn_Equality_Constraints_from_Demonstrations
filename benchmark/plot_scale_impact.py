#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


DEFAULT_EXPERIMENTS = [
    "paper_scale_020_3env_7seed",
    "paper_scale_050_3env_7seed",
    "paper_scale_075_3env_7seed",
    "paper_scale_150_3env_7seed",
    "paper_scale_300_3env_7seed",
]

DEFAULT_SCALE1_EXPERIMENT = "paper_mix_2d_3d6d_traj_vs_nontraj_7seed"

DEFAULT_METRICS = [
    "proj_true_dist",
    "proj_manifold_dist",
    "bidirectional_chamfer",
    "pred_recall",
    "pred_precision",
    "pred_FPrate",
    "train_seconds",
]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root(), path))


@dataclass(frozen=True)
class Record:
    exp_name: str
    scale: float
    method: str
    dataset: str
    seed: int
    metrics: Dict[str, float]


def _safe_float(x: str) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _parse_scale(exp_name: str) -> float:
    parts = exp_name.split("_")
    for i, p in enumerate(parts[:-1]):
        if p == "scale":
            raw = parts[i + 1]
            if raw.isdigit():
                return float(raw) / 100.0
            return float(raw)
    raise ValueError(f"Cannot parse scale from experiment name: {exp_name}")


def _read_per_case_csv(
    bench_root: str,
    exp_name: str,
    metrics: Sequence[str],
    scale_override: Optional[float] = None,
) -> List[Record]:
    csv_path = os.path.join(bench_root, exp_name, "per_case_metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    scale = float(scale_override) if scale_override is not None else _parse_scale(exp_name)
    rows: List[Record] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method", "")).strip()
            dataset = str(row.get("dataset", "")).strip()
            seed_raw = row.get("seed", "")
            if not method or not dataset:
                continue
            try:
                seed = int(float(seed_raw))
            except Exception:
                continue
            metric_vals: Dict[str, float] = {}
            for m in metrics:
                v = _safe_float(str(row.get(m, "")))
                if v is not None:
                    metric_vals[m] = v
            if not metric_vals:
                continue
            rows.append(
                Record(
                    exp_name=exp_name,
                    scale=scale,
                    method=method,
                    dataset=dataset,
                    seed=seed,
                    metrics=metric_vals,
                )
            )
    return rows


def _mean_std(xs: Iterable[float]) -> Tuple[float, float, int]:
    vals = list(xs)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return mean, math.sqrt(max(0.0, var)), n


def _pct_change(cur: float, base: float) -> Optional[float]:
    if not math.isfinite(cur) or not math.isfinite(base):
        return None
    if abs(base) < 1e-12:
        return None
    return (cur - base) / abs(base) * 100.0


def _write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _method_label(method: str) -> str:
    if method == "oncl":
        return "ONCL (Ours)"
    if method == "ecomann":
        return "ECoMaNN"
    return method


def _metric_label(metric: str) -> str:
    mapping = {
        "proj_manifold_dist": r"Distance Error",
        "gt_to_learned_mean": r"Coverage Error $\epsilon_g$",
        "learned_to_gt_mean": r"Distance Error $\epsilon_l$",
        "bidirectional_chamfer": r"Bi-Chamfer $\epsilon_{\mathrm{bi}}$",
        "train_seconds": "Training Time",
        "proj_true_dist": r"Projection Error $\epsilon_p$",
        "pred_recall": "Recall",
        "pred_precision": "Precision",
        "pred_FPrate": "FP Rate",
    }
    return mapping.get(metric, metric)


def _metric_file_stem(metric: str) -> str:
    mapping = {
        "proj_manifold_dist": "distance_error",
        "gt_to_learned_mean": "coverage_error",
        "train_seconds": "training_time",
    }
    return mapping.get(metric, metric)


def _plot_dataset_bar(
    dataset: str,
    metric: str,
    methods: Sequence[str],
    scales: Sequence[float],
    detail_values: Dict[Tuple[str, str, str, float], List[float]],
    outdir: str,
) -> None:
    # Style aligned to analyze/benchmark_paper_boxplots.py
    fig, ax = plt.subplots(figsize=(3.45, 2.2))

    group_gap = 1.95
    bar_gap = 0.34
    bar_w = 0.26
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"][: len(scales)]

    xticks: List[float] = []
    xlabels: List[str] = []
    legend_handles: List[mpatches.Patch] = []

    for si, scale in enumerate(scales):
        legend_handles.append(
            mpatches.Patch(facecolor=colors[si], edgecolor="#333333", label=f"{scale:g}x")
        )

    for mi, method in enumerate(methods):
        group_center = mi * group_gap
        xticks.append(group_center)
        xlabels.append(_method_label(method))

        for si, scale in enumerate(scales):
            x = group_center + (si - (len(scales) - 1) / 2.0) * bar_gap
            vals = detail_values.get((dataset, method, metric, scale), [])
            mean, std, n = _mean_std(vals)
            if n == 0:
                continue
            ax.bar(
                x,
                mean,
                width=bar_w,
                color=colors[si],
                alpha=0.82,
                edgecolor=colors[si],
                linewidth=1.0,
                zorder=3,
            )
            ax.errorbar(
                x,
                mean,
                yerr=std,
                fmt="none",
                ecolor="#1F2937",
                elinewidth=1.0,
                capsize=2.0,
                capthick=1.0,
                zorder=4,
            )

    for i in range(len(methods) - 1):
        ax.axvline(float(i) * group_gap + group_gap / 2.0, color="#6B7280", linestyle="-", linewidth=1.0, alpha=0.35, zorder=0)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(_metric_label(metric), fontsize=9)
    # Match paper figure style: no per-subplot title.
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=legend_handles,
        title="Number of Training Data",
        title_fontsize=7,
        frameon=False,
        ncol=len(scales),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=7,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0), pad=0.25)

    os.makedirs(outdir, exist_ok=True)
    metric_stem = _metric_file_stem(metric)
    out_png = os.path.join(outdir, f"{dataset}_{metric_stem}_bar.png")
    out_pdf = os.path.join(outdir, f"{dataset}_{metric_stem}_bar.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] plot: {out_png}")


def _plot_paper_single_panel_line(
    *,
    metric: str,
    scales: Sequence[float],
    detail_values: Dict[Tuple[str, str, str, float], List[float]],
    outdir: str,
) -> None:
    target_datasets = [
        "3d_torus_surface_traj",
        "6d_workspace_sine_surface_pose_traj",
    ]
    target_methods = ["oncl", "ecomann"]
    if len(scales) == 0:
        return

    # Single-column friendly figure with dual y-axis:
    # left axis for Torus, right axis for SinePose.
    fig, ax = plt.subplots(figsize=(5, 1.8))
    ax_r = ax.twinx()

    # Dataset by color; method by linestyle.
    dataset_style = {
        "3d_torus_surface_traj": ("3DTorus", "#1f77b4"),
        "6d_workspace_sine_surface_pose_traj": ("6DSinePose", "#d62728"),
    }
    method_style = {"oncl": ("ONCL (Ours)", "-"), "ecomann": ("ECoMaNN", "--")}

    x = list(range(len(scales)))
    xlabels = [f"{s:g}x" for s in scales]

    any_drawn = False
    legend_handles: List[Line2D] = []
    legend_seen: set[str] = set()
    for ds in target_datasets:
        if ds not in dataset_style:
            continue
        ds_name, ds_color = dataset_style[ds]
        target_ax = ax if ds == "3d_torus_surface_traj" else ax_r
        for m in target_methods:
            m_name, ls = method_style.get(m, (m, "-"))
            means: List[float] = []
            stds: List[float] = []
            valid_x: List[int] = []
            for i, s in enumerate(scales):
                vals = detail_values.get((ds, m, metric, s), [])
                mean, std, n = _mean_std(vals)
                if n == 0:
                    continue
                valid_x.append(i)
                means.append(mean)
                stds.append(std)
            if not valid_x:
                continue
            any_drawn = True
            legend_label = f"{ds_name} - {m_name}"
            line_w = 1.4
            target_ax.errorbar(
                valid_x,
                means,
                yerr=stds,
                color=ds_color,
                linestyle=ls,
                linewidth=line_w,
                marker="o" if m == "oncl" else "s",
                markersize=3.4,
                capsize=2.0,
                elinewidth=1.0,
                zorder=3,
            )
            if legend_label not in legend_seen:
                legend_seen.add(legend_label)
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=ds_color,
                        linestyle=ls,
                        linewidth=line_w,
                        marker="o" if m == "oncl" else "s",
                        markersize=3.4,
                        label=legend_label,
                    )
                )

    if not any_drawn:
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8, colors=dataset_style["3d_torus_surface_traj"][1])
    ax_r.tick_params(axis="y", labelsize=8, colors=dataset_style["6d_workspace_sine_surface_pose_traj"][1])
    ax.set_xlabel("Number of training data", fontsize=8)
    ax.set_ylabel(
        f"3DTorus {_metric_label(metric)} ",
        fontsize=8,
        color=dataset_style["3d_torus_surface_traj"][1],
    )
    ax_r.set_ylabel(
        f"6DSinePose {_metric_label(metric)}",
        fontsize=8,
        color=dataset_style["6d_workspace_sine_surface_pose_traj"][1],
    )
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
        fontsize=6.7,
        handlelength=3.0,
        handletextpad=0.45,
        columnspacing=0.8,
        labelspacing=0.8,
    )
    # Keep extra margins for dual y-labels and reserve the right side for legend.
    fig.subplots_adjust(left=0.1, right=0.55, bottom=0.25, top=0.91)

    os.makedirs(outdir, exist_ok=True)
    metric_stem = _metric_file_stem(metric)
    out_png = os.path.join(outdir, f"{metric_stem}_torus_sinepose_single_panel_line.png")
    out_pdf = os.path.join(outdir, f"{metric_stem}_torus_sinepose_single_panel_line.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[OK] plot: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze scale impact and generate unified CSV + paper-style bar plots."
    )
    parser.add_argument(
        "--bench-root",
        default="outputs/bench",
        help="Root directory containing paper_scale_* benchmark folders.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment folder names under --bench-root.",
    )
    parser.add_argument(
        "--scale1-experiment",
        default=DEFAULT_SCALE1_EXPERIMENT,
        help="Benchmark folder used as scale=1.0 source.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to include in CSV output.",
    )
    parser.add_argument(
        "--plot-metrics",
        nargs="+",
        default=["proj_manifold_dist", "train_seconds"],
        help="Metrics used for per-dataset bar plots.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["oncl", "ecomann"],
        help="Method order in grouped bars.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/analysis/scale_impact_analysis",
        help="Unified output directory for CSV and plots.",
    )
    args = parser.parse_args()

    all_metrics = list(dict.fromkeys(list(args.metrics) + list(args.plot_metrics)))
    bench_root = _resolve_path(args.bench_root)
    outdir = _resolve_path(args.outdir)
    primary_records: List[Record] = []
    for exp in args.experiments:
        primary_records.extend(_read_per_case_csv(bench_root, exp, all_metrics))
    if not primary_records:
        raise RuntimeError("No valid rows found. Check experiment names and metric keys.")
    target_pairs = {(r.dataset, r.method) for r in primary_records}
    target_methods = {r.method for r in primary_records}
    target_datasets = {r.dataset for r in primary_records}

    scale1_records: List[Record] = []
    if str(args.scale1_experiment).strip():
        scale1_all = _read_per_case_csv(
            bench_root=bench_root,
            exp_name=str(args.scale1_experiment).strip(),
            metrics=all_metrics,
            scale_override=1.0,
        )
        for r in scale1_all:
            if (r.dataset, r.method) in target_pairs:
                scale1_records.append(r)

    all_records: List[Record] = list(primary_records) + list(scale1_records)

    detail_values: Dict[Tuple[str, str, str, float], List[float]] = defaultdict(list)
    summary_values: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)

    scales = sorted({r.scale for r in all_records})
    datasets = sorted(target_datasets)
    methods_found = sorted(target_methods)
    methods = [m for m in args.methods if m in methods_found]
    if not methods:
        methods = methods_found

    for r in all_records:
        for metric, value in r.metrics.items():
            detail_values[(r.dataset, r.method, metric, r.scale)].append(value)
            summary_values[(r.method, metric, r.scale)].append(value)

    detail_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for method in methods_found:
            for metric in args.metrics:
                base_scale = scales[0]
                base_key = (dataset, method, metric, base_scale)
                base_mean, _, _ = _mean_std(detail_values.get(base_key, []))
                for scale in scales:
                    key = (dataset, method, metric, scale)
                    mean, std, n = _mean_std(detail_values.get(key, []))
                    if n == 0:
                        continue
                    pct = _pct_change(mean, base_mean)
                    detail_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "metric": metric,
                            "scale": scale,
                            "n": n,
                            "mean": mean,
                            "std": std,
                            "baseline_scale": base_scale,
                            "baseline_mean": base_mean,
                            "delta_vs_baseline": mean - base_mean if math.isfinite(base_mean) else "",
                            "pct_change_vs_baseline": pct if pct is not None else "",
                        }
                    )

    summary_rows: List[Dict[str, object]] = []
    for method in methods_found:
        for metric in args.metrics:
            base_scale = scales[0]
            base_key = (method, metric, base_scale)
            base_mean, _, _ = _mean_std(summary_values.get(base_key, []))
            for scale in scales:
                key = (method, metric, scale)
                mean, std, n = _mean_std(summary_values.get(key, []))
                if n == 0:
                    continue
                pct = _pct_change(mean, base_mean)
                summary_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "scale": scale,
                        "n": n,
                        "mean": mean,
                        "std": std,
                        "baseline_scale": base_scale,
                        "baseline_mean": base_mean,
                        "delta_vs_baseline": mean - base_mean if math.isfinite(base_mean) else "",
                        "pct_change_vs_baseline": pct if pct is not None else "",
                    }
                )

    detail_csv = os.path.join(outdir, "scale_impact_detail.csv")
    summary_csv = os.path.join(outdir, "scale_impact_summary.csv")
    _write_csv(
        detail_csv,
        [
            "dataset",
            "method",
            "metric",
            "scale",
            "n",
            "mean",
            "std",
            "baseline_scale",
            "baseline_mean",
            "delta_vs_baseline",
            "pct_change_vs_baseline",
        ],
        detail_rows,
    )
    _write_csv(
        summary_csv,
        [
            "method",
            "metric",
            "scale",
            "n",
            "mean",
            "std",
            "baseline_scale",
            "baseline_mean",
            "delta_vs_baseline",
            "pct_change_vs_baseline",
        ],
        summary_rows,
    )
    print(f"[OK] csv:  {detail_csv}")
    print(f"[OK] csv:  {summary_csv}")

    for metric in args.plot_metrics:
        for dataset in datasets:
            _plot_dataset_bar(
                dataset=dataset,
                metric=metric,
                methods=methods,
                scales=scales,
                detail_values=detail_values,
                outdir=outdir,
            )
        _plot_paper_single_panel_line(
            metric=metric,
            scales=scales,
            detail_values=detail_values,
            outdir=outdir,
        )

    print(f"[Info] outdir: {outdir}")
    print(f"[Info] plot metrics: {args.plot_metrics}")
    print(f"[Info] datasets: {datasets}")
    print(f"[Info] scales: {scales}")
    print(f"[Info] scale=1 source: {args.scale1_experiment}")
    print(f"[Info] scale=1 matched rows: {len(scale1_records)}")


if __name__ == "__main__":
    main()
