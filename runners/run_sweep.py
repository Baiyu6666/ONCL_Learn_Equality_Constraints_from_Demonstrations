#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Keep local W&B files under repo-root/wandb regardless of launch cwd.
os.environ["WANDB_DIR"] = os.path.join(_PROJECT_ROOT, "wandb")

from experiments.unified_experiment import VALID_METHODS, run_one

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

def _resolve_config_root(config_root: str) -> str:
    p = str(config_root).strip()
    if not p:
        p = "configs"
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    fallback = os.path.join(_PROJECT_ROOT, p)
    return fallback


def _resolve_outdir(outdir: str) -> str:
    p = str(outdir).strip()
    if not p:
        p = "outputs/sweeps"
    if os.path.isabs(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


_RESERVED_CFG_KEYS = {
    "method",
    "methods",
    "datasets",
    "seed",
    "seeds",
    "config_root",
    "outdir",
    "method_tag",
    "method_tag_prefix",
    "override",
}


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _to_override(k: str, v: Any) -> str:
    return f"{k}={json.dumps(v, ensure_ascii=False)}"


def _override_keys(overrides: list[str]) -> set[str]:
    keys: set[str] = set()
    for expr in overrides:
        if "=" not in str(expr):
            continue
        k = str(expr).split("=", 1)[0].strip()
        if k:
            keys.add(k)
    return keys


def _with_default_non_gif_overrides(overrides: list[str]) -> list[str]:
    out = list(overrides)
    keys = _override_keys(out)
    if "planner.save_gif" not in keys:
        out.append("planner.save_gif=false")
    if "planner.pybullet_render" not in keys:
        out.append("planner.pybullet_render=false")
    return out


def _log_dataset_plots_to_wandb(
    *,
    outdir: str,
    method: str,
    dataset: str,
    min_mtime: float | None = None,
) -> None:
    method_dir = os.path.join(outdir, method)
    if not os.path.isdir(method_dir):
        return
    patterns = [
        os.path.join(method_dir, f"{dataset}*.png"),
        os.path.join(method_dir, f"{dataset}*.jpg"),
        os.path.join(method_dir, f"{dataset}*.jpeg"),
    ]
    plot_paths: list[str] = []
    for pat in patterns:
        plot_paths.extend(glob.glob(pat))
    if min_mtime is not None:
        plot_paths = [p for p in plot_paths if os.path.getmtime(p) >= float(min_mtime)]
    if not plot_paths:
        return
    for p in sorted(set(plot_paths)):
        key = os.path.splitext(os.path.basename(p))[0]
        try:
            wandb.log({f"{dataset}/{method}/plot/{key}": wandb.Image(p)})
        except Exception as e:
            print(f"[warn] wandb image log failed: {p} ({e})")


def _sanitize_wandb_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _sanitize_wandb_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_wandb_value(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _flatten_dict_for_wandb(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}__{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict_for_wandb(v, key))
        else:
            out[key] = _sanitize_wandb_value(v)
    return out


def _sync_resolved_config_to_wandb(
    *,
    run_obj: Any,
    method: str,
    dataset: str,
    seed: int | None,
    resolved_config: dict[str, Any],
    loaded_paths: list[str],
) -> None:
    prefix = f"resolved__{method}__{dataset}"
    payload = _flatten_dict_for_wandb(dict(resolved_config), f"{prefix}__cfg")
    payload[f"{prefix}__seed"] = seed
    payload[f"{prefix}__loaded_config_paths"] = [str(p) for p in loaded_paths]
    payload[f"{prefix}__loaded_layers_count"] = int(len(loaded_paths))
    try:
        run_obj.config.update(payload, allow_val_change=True)
    except Exception as e:
        print(f"[warn] wandb config sync failed for {method}/{dataset}: {e}")


def _resolve_method(args_method: str, cfg_dict: dict[str, Any]) -> str:
    method = str(args_method).strip()
    if method:
        return method
    m_cfg = cfg_dict.get("method")
    if isinstance(m_cfg, str) and m_cfg.strip():
        return str(m_cfg).strip()
    raise ValueError(
        "method is required: pass --method or provide sweep parameter 'method'"
    )


def _resolve_seed(args_seed: int | None, cfg_dict: dict[str, Any]) -> int:
    if args_seed is not None:
        return int(args_seed)
    s_cfg = cfg_dict.get("seed")
    if isinstance(s_cfg, (int, float)):
        return int(s_cfg)
    if isinstance(s_cfg, str) and s_cfg.strip():
        return int(str(s_cfg).strip())
    if "seeds" in cfg_dict:
        raise ValueError("use single 'seed' for sweep trials, not 'seeds'")
    raise ValueError("seed is required: pass --seed or provide sweep parameter 'seed'")


def _collect_dynamic_overrides(cfg_dict: dict[str, Any], method: str) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, list[str]]]:
    global_overrides: list[str] = []
    method_overrides: list[str] = []
    dataset_overrides: dict[str, list[str]] = {}
    method_dataset_overrides: dict[str, list[str]] = {}

    for k, v in cfg_dict.items():
        if k in _RESERVED_CFG_KEYS:
            continue
        if k.startswith("_"):
            continue

        if k.startswith("ds__"):
            parts = str(k).split("__", 2)
            if len(parts) == 3 and parts[1] and parts[2]:
                ds = parts[1]
                param = parts[2]
                dataset_overrides.setdefault(ds, []).append(_to_override(param, v))
            continue

        if k.startswith("m__"):
            parts = str(k).split("__", 2)
            if len(parts) < 3:
                continue
            m_name = parts[1]
            rest = parts[2]
            if str(m_name) != str(method):
                continue
            if rest.startswith("ds__"):
                sub = rest.split("__", 2)
                if len(sub) == 3 and sub[1] and sub[2]:
                    ds = sub[1]
                    param = sub[2]
                    method_dataset_overrides.setdefault(ds, []).append(_to_override(param, v))
                continue
            if rest:
                method_overrides.append(_to_override(rest, v))
            continue

        global_overrides.append(_to_override(k, v))

    return global_overrides, method_overrides, dataset_overrides, method_dataset_overrides


def _aggregate_metrics(results: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    metric_keys: set[str] = set()
    for r in results:
        metric_keys.update(r.get("metrics", {}).keys())

    avg_all: dict[str, float] = {}
    std_all: dict[str, float] = {}
    for k in sorted(metric_keys):
        vals: list[float] = []
        for r in results:
            v = r.get("metrics", {}).get(k)
            if isinstance(v, (int, float)):
                fv = float(v)
                if math.isfinite(fv):
                    vals.append(fv)
        if vals:
            avg_all[k] = float(sum(vals) / len(vals))
            std_all[k] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return avg_all, std_all


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="W&B sweep trial runner: run one method over multiple datasets for one trial seed."
    )
    p.add_argument("--method", default="")
    p.add_argument("--datasets", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config-root", default="configs")
    p.add_argument("--outdir", default="outputs/sweeps")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--method-tag-prefix", default="method", help="wandb tag prefix, e.g. method:oncl")

    return p


def main() -> None:
    args = _build_parser().parse_args()
    config_root = _resolve_config_root(str(args.config_root))
    outdir = _resolve_outdir(str(args.outdir))

    if wandb is None:
        raise RuntimeError("wandb is required for run_sweep.py but is not installed in current environment")

    run = wandb.init()
    cfg_dict = dict(wandb.config)

    method = _resolve_method(args.method, cfg_dict)
    if method not in VALID_METHODS:
        raise ValueError(f"unsupported method '{method}', valid={sorted(VALID_METHODS)}")

    datasets = _split_csv(args.datasets)
    if not datasets:
        raise ValueError("no datasets provided")

    seed = _resolve_seed(args.seed, cfg_dict)

    method_tag = f"{args.method_tag_prefix}:{method}" if str(args.method_tag_prefix) else method
    try:
        old_tags = list(getattr(run, "tags", ()) or ())
        if method_tag not in old_tags:
            run.tags = tuple(old_tags + [method_tag])
    except Exception:
        pass

    global_overrides, method_overrides, ds_overrides, mds_overrides = _collect_dynamic_overrides(cfg_dict, method)

    os.makedirs(outdir, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_start_ts = time.time()
        dynamic_overrides = (
            list(global_overrides)
            + list(method_overrides)
            + list(ds_overrides.get(dataset, []))
            + list(mds_overrides.get(dataset, []))
            + list(args.override)
        )
        dynamic_overrides = _with_default_non_gif_overrides(dynamic_overrides)
        print(f"[run] method={method} dataset={dataset} seed={seed}")
        result, loaded_paths = run_one(
            method=method,
            dataset=dataset,
            out_root=outdir,
            seed_override=seed,
            config_root=config_root,
            cli_overrides=dynamic_overrides,
        )
        result["seed"] = int(seed)
        result["loaded_config_paths"] = loaded_paths
        all_results.append(result)
        m = result["metrics"]
        print(
            f"[done] method={method} dataset={dataset} seed={seed} "
            f"proj_dist={m.get('proj_manifold_dist', float('nan')):.6f} "
            f"recall={m.get('pred_recall', float('nan')):.6f} "
            f"FPrate={m.get('pred_FPrate', float('nan')):.6f}"
        )
        wandb.log({f"{dataset}/{method}/{k}": v for k, v in m.items()})
        wandb.log({f"{dataset}/loaded_layers_count": float(len(loaded_paths))})
        _sync_resolved_config_to_wandb(
            run_obj=run,
            method=str(method),
            dataset=str(dataset),
            seed=int(seed),
            resolved_config=dict(result.get("config", {})),
            loaded_paths=list(loaded_paths),
        )
        _log_dataset_plots_to_wandb(
            outdir=outdir,
            method=method,
            dataset=dataset,
            min_mtime=dataset_start_ts,
        )

    avg_all, std_all = _aggregate_metrics(all_results)
    if avg_all:
        wandb.log({f"sweep/avg/{k}": v for k, v in avg_all.items()})
    if std_all:
        wandb.log({f"sweep/std/{k}": v for k, v in std_all.items()})
    wandb.log(
        {
            "sweep/n_datasets": float(len(datasets)),
            "sweep/n_seeds": 1.0,
            "sweep/n_cases": float(len(all_results)),
        }
    )

    summary = {
        "method": method,
        "datasets": datasets,
        "seed": int(seed),
        "avg_metrics": avg_all,
        "std_metrics": std_all,
        "results": all_results,
        "global_overrides": global_overrides,
        "method_overrides": method_overrides,
        "dataset_overrides": ds_overrides,
        "method_dataset_overrides": mds_overrides,
        "cli_overrides": list(args.override),
    }
    out_path = os.path.join(outdir, f"sweep_summary_{method}_seed{int(seed)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path}")
    print(f"[sweep] method={method} avg_metrics={len(avg_all)} over cases={len(all_results)}")

    run.finish()


if __name__ == "__main__":
    main()
