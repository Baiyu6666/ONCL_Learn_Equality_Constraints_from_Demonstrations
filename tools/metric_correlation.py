from __future__ import annotations

import argparse
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_LEGACY_SCRIPT = os.path.join(_PROJECT_ROOT, "outputs", "bench", "all_2d3d6d_5seed", "analyze_dataset_corr_cov_proj.py")


def _resolve_bench_dir(bench: str) -> str:
    b = str(bench).strip()
    if os.path.isabs(b):
        return b
    p1 = os.path.join(_PROJECT_ROOT, "outputs", "bench", b)
    if os.path.isdir(p1):
        return p1
    return os.path.abspath(b)


def main() -> None:
    p = argparse.ArgumentParser(description="Build corr/cov plots for one benchmark by bench name.")
    p.add_argument(
        "--bench",
        default="paper_mix_2d_3d6d_traj_vs_nontraj_7seed",
        help="Benchmark name (under outputs/bench) or absolute path",
    )
    p.add_argument("--analysis-dir", default="analysis_corr_cov_proj_manifold_dist", help="Output dir name under benchmark dir")
    args = p.parse_args()

    bench_dir = _resolve_bench_dir(args.bench)
    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(f"benchmark directory not found: {bench_dir}")
    input_csv = os.path.join(bench_dir, "per_case_metrics.csv")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"missing per_case_metrics.csv: {input_csv}")
    outdir = os.path.join(bench_dir, args.analysis_dir)
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(_LEGACY_SCRIPT):
        raise FileNotFoundError(f"legacy script missing: {_LEGACY_SCRIPT}")

    cmd = [sys.executable, _LEGACY_SCRIPT, "--input-csv", input_csv, "--outdir", outdir]
    subprocess.check_call(cmd, cwd=_PROJECT_ROOT)


if __name__ == "__main__":
    main()
