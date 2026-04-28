from __future__ import annotations

import argparse
import csv
import os
import sys
import numpy as np
from scipy import stats

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analyze import benchmark_paper_boxplots as paper
from analyze import old_benchmark_boxplots as base


LOWER_IS_BETTER_METRICS = {
    "proj_manifold_dist",
    "gt_to_learned_mean",
    "learned_to_gt_mean",
    "bidirectional_chamfer",
    "train_seconds",
}

HIGHER_IS_BETTER_METRICS = {
    "pred_precision",
}


def _independent_arrays(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    method_a: str,
    method_b: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    vals_a: list[float] = []
    vals_b: list[float] = []
    for r in rows:
        if str(r.get("dataset", "")).strip() != str(dataset):
            continue
        method = str(r.get("method", "")).strip()
        v = base._to_float(r.get(metric, np.nan))
        if not np.isfinite(v):
            continue
        if method == str(method_a):
            vals_a.append(float(v))
        elif method == str(method_b):
            vals_b.append(float(v))
    return np.asarray(vals_a, dtype=np.float64), np.asarray(vals_b, dtype=np.float64)


def _holm_bonferroni(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(np.asarray(pvals, dtype=np.float64))
    ranked = [float(pvals[i]) for i in order]
    adj_ranked = [0.0] * n
    running = 0.0
    for k, p in enumerate(ranked):
        adj = (n - k) * p
        running = max(running, adj)
        adj_ranked[k] = min(1.0, running)
    out = [0.0] * n
    for idx, adj in zip(order, adj_ranked):
        out[int(idx)] = float(adj)
    return out


def _cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return np.nan
    nx = int(x.size)
    ny = int(y.size)
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def _metric_alternative(metric: str) -> str:
    m = str(metric)
    if m in LOWER_IS_BETTER_METRICS:
        return "less"
    if m in HIGHER_IS_BETTER_METRICS:
        return "greater"
    raise ValueError(f"unknown metric direction for t-test: {metric}")


def _mean_std_str(vals: np.ndarray) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return "-"
    return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=0):.4f}"


def _save_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_md(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    cols = [
        "Dataset",
        "Baseline",
        "n_oncl",
        "n_base",
        "ONCL",
        "Baseline mean±std",
        "alternative",
        "t",
        "p",
        "p_holm",
        "significant_0.05",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(r["Dataset"]),
                        str(r["Baseline"]),
                        str(r["n_oncl"]),
                        str(r["n_base"]),
                        str(r["ONCL"]),
                        str(r["Baseline mean±std"]),
                        str(r["alternative"]),
                        str(r["t"]),
                        str(r["p"]),
                        str(r["p_holm"]),
                        str(r["significant_0.05"]),
                    ]
                )
                + " |\n"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Independent Welch t-tests for ONCL vs baselines on paper benchmark results.")
    p.add_argument("--bench", type=str, default="paper_mix_2d_3d6d_traj_vs_nontraj_7seed", help="Benchmark name or absolute path.")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset ids. Default: paper preset.")
    p.add_argument("--methods", type=str, default="dataaug,vae,ecomann", help="Baselines to compare against ONCL.")
    p.add_argument("--metrics", type=str, default="proj_manifold_dist,gt_to_learned", help="Comma-separated metrics.")
    p.add_argument("--analysis-dir", type=str, default="analysis_bars_paper", help="Output dir under bench dir.")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    args = p.parse_args()

    bench_dir = base._resolve_bench_dir(args.bench)
    rows, _source = base._load_rows(bench_dir)
    if not rows:
        raise RuntimeError(f"no benchmark rows found under: {bench_dir}")

    datasets = base._split_csv(args.datasets, None)
    if not datasets:
        datasets = list(paper.PAPER_DATASETS)
    metrics = paper._normalize_metrics(base._split_csv(args.metrics, base.DEFAULT_METRICS))
    baselines = paper._sort_methods_for_paper(base._split_csv(args.methods, []))

    all_methods = sorted({str(r.get("method", "")).strip() for r in rows if str(r.get("method", "")).strip()})
    baselines = [m for m in baselines if m in all_methods and m != "oncl"]
    if "oncl" not in all_methods:
        raise RuntimeError("ONCL results not found in benchmark rows")

    outdir = os.path.join(bench_dir, str(args.analysis_dir))
    os.makedirs(outdir, exist_ok=True)

    for metric in metrics:
        alternative = _metric_alternative(metric)
        raw_rows: list[dict[str, object]] = []
        holm_groups: list[float] = []

        for dataset in datasets:
            for baseline in baselines:
                oncl_vals, base_vals = _independent_arrays(
                    rows,
                    dataset=dataset,
                    method_a="oncl",
                    method_b=baseline,
                    metric=metric,
                )
                if oncl_vals.size < 2 or base_vals.size < 2:
                    continue
                test = stats.ttest_ind(oncl_vals, base_vals, equal_var=False, alternative=alternative)
                if metric in LOWER_IS_BETTER_METRICS:
                    # Positive diff means ONCL is better by this amount.
                    mean_diff = float(np.mean(base_vals) - np.mean(oncl_vals))
                    effect = _cohens_d_independent(base_vals, oncl_vals)
                else:
                    mean_diff = float(np.mean(oncl_vals) - np.mean(base_vals))
                    effect = _cohens_d_independent(oncl_vals, base_vals)
                row = {
                    "dataset_id": dataset,
                    "Dataset": paper._to_display_dataset(dataset),
                    "baseline_id": baseline,
                    "Baseline": paper._to_display_method(baseline),
                    "metric": metric,
                    "metric_label": paper._to_display_metric(metric),
                    "n_oncl": int(oncl_vals.size),
                    "n_base": int(base_vals.size),
                    "oncl_mean": float(np.mean(oncl_vals)),
                    "oncl_std": float(np.std(oncl_vals, ddof=0)),
                    "baseline_mean": float(np.mean(base_vals)),
                    "baseline_std": float(np.std(base_vals, ddof=0)),
                    "mean_diff_good_direction": mean_diff,
                    "cohens_d_paired": effect,
                    "test_type": "welch_independent",
                    "alternative": "ONCL < baseline" if alternative == "less" else "ONCL > baseline",
                    "t_stat": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "ONCL": _mean_std_str(oncl_vals),
                    "Baseline mean±std": _mean_std_str(base_vals),
                }
                raw_rows.append(row)
                holm_groups.append(float(test.pvalue))

        adj = _holm_bonferroni(holm_groups)
        for row, p_adj in zip(raw_rows, adj):
            row["p_holm"] = float(p_adj)
            row["significant_0.05"] = bool(float(p_adj) < float(args.alpha))
            row["t"] = f"{row['t_stat']:.4f}"
            row["p"] = f"{row['p_value']:.4g}"
            row["p_holm_fmt"] = f"{row['p_holm']:.4g}"

        csv_rows: list[dict[str, object]] = []
        md_rows: list[dict[str, object]] = []
        for row in raw_rows:
            csv_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "Dataset": row["Dataset"],
                    "baseline_id": row["baseline_id"],
                    "Baseline": row["Baseline"],
                    "metric": row["metric"],
                    "metric_label": row["metric_label"],
                    "n_oncl": row["n_oncl"],
                    "n_base": row["n_base"],
                    "oncl_mean": row["oncl_mean"],
                    "oncl_std": row["oncl_std"],
                    "baseline_mean": row["baseline_mean"],
                    "baseline_std": row["baseline_std"],
                    "mean_diff_good_direction": row["mean_diff_good_direction"],
                    "cohens_d_independent": row["cohens_d_paired"],
                    "test_type": row["test_type"],
                    "alternative": row["alternative"],
                    "t_stat": row["t_stat"],
                    "p_value": row["p_value"],
                    "p_holm": row["p_holm"],
                    "significant_0.05": row["significant_0.05"],
                }
            )
            md_rows.append(
                {
                    "Dataset": row["Dataset"],
                    "Baseline": row["Baseline"],
                    "n_oncl": row["n_oncl"],
                    "n_base": row["n_base"],
                    "ONCL": row["ONCL"],
                    "Baseline mean±std": row["Baseline mean±std"],
                    "alternative": row["alternative"],
                    "t": row["t"],
                    "p": row["p"],
                    "p_holm": row["p_holm_fmt"],
                    "significant_0.05": "yes" if row["significant_0.05"] else "no",
                }
            )

        stem = paper._to_metric_file_stem(metric)
        csv_path = os.path.join(outdir, f"{stem}_welch_ttests.csv")
        md_path = os.path.join(outdir, f"{stem}_welch_ttests.md")
        _save_csv(csv_path, csv_rows)
        _save_md(md_path, md_rows)
        print(f"[saved] {csv_path}")
        print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
