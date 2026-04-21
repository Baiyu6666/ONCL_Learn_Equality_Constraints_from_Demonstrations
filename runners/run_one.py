#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

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
        p = "outputs"
    if os.path.isabs(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


def _clear_directory(path: str) -> None:
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def _prepare_outdir(path: str, *, retrain: bool) -> None:
    if os.path.exists(path) and not os.path.isdir(path):
        raise ValueError(f"outdir exists but is not a directory: {path}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return
    has_existing = any(True for _ in os.scandir(path))
    if not has_existing:
        return
    if retrain:
        print(f"[retrain] clearing existing outdir: {path}")
        _clear_directory(path)
        return
    raise ValueError(
        "outdir already exists and is not empty; refusing to mix old/new results. "
        "Use --retrain (or -retrain) to clear it first, or choose a new --outdir."
    )


def _log_dataset_plots_to_wandb(*, outdir: str, method: str, dataset: str) -> None:
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
    if not plot_paths:
        return
    for p in sorted(set(plot_paths)):
        key = os.path.splitext(os.path.basename(p))[0]
        try:
            wandb.log({f"{dataset}/{method}/plot/{key}": wandb.Image(p)})
        except Exception as e:
            print(f"[warn] wandb image log failed: {p} ({e})")


def _sanitize_wandb_value(v: object) -> object:
    if isinstance(v, dict):
        return {str(k): _sanitize_wandb_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_wandb_value(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _flatten_dict_for_wandb(d: dict[str, object], prefix: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}__{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict_for_wandb(v, key))
        else:
            out[key] = _sanitize_wandb_value(v)
    return out


def _sync_resolved_config_to_wandb(
    *,
    wb_run: object,
    method: str,
    dataset: str,
    seed: int | None,
    resolved_config: dict[str, object],
    loaded_paths: list[str],
) -> None:
    prefix = f"resolved__{method}__{dataset}"
    payload = _flatten_dict_for_wandb(dict(resolved_config), f"{prefix}__cfg")
    payload[f"{prefix}__seed"] = seed
    payload[f"{prefix}__loaded_config_paths"] = [str(p) for p in loaded_paths]
    payload[f"{prefix}__loaded_layers_count"] = int(len(loaded_paths))
    try:
        wb_run.config.update(payload, allow_val_change=True)
    except Exception as e:
        print(f"[warn] wandb config sync failed for {method}/{dataset}: {e}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one method-dataset experiment")
    p.add_argument("--method", required=True, choices=sorted(VALID_METHODS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("-retrain", "--retrain", action="store_true", help="clear non-empty outdir before running")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config-root", default="configs")
    p.add_argument("--override", action="append", default=[], help="dotted key=value override")

    p.add_argument("--wandb-enable", action="store_true")
    p.add_argument("--wandb-project", default="LearnEqConstraints")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-run-name", default="")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    outdir = _resolve_outdir(str(args.outdir))
    _prepare_outdir(outdir, retrain=bool(args.retrain))
    config_root = _resolve_config_root(str(args.config_root))

    wb_run = None
    if args.wandb_enable:
        if wandb is None:
            print("[warn] wandb requested but not available; continuing without wandb")
        else:
            wb_run = wandb.init(
                project=str(args.wandb_project),
                entity=(str(args.wandb_entity) or None),
                name=(str(args.wandb_run_name) or None),
                config={
                    "method": args.method,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "config_root": config_root,
                    "override": args.override,
                },
            )

    result, loaded_paths = run_one(
        method=args.method,
        dataset=args.dataset,
        out_root=outdir,
        seed_override=args.seed,
        config_root=config_root,
        cli_overrides=args.override,
    )

    summary_path = os.path.join(outdir, f"summary_{args.method}_{args.dataset}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[done] method={args.method} dataset={args.dataset}")
    print(f"[cfg] loaded_layers={loaded_paths if loaded_paths else '[]'}")
    print(f"[saved] {summary_path}")

    if wb_run is not None:
        m = result["metrics"]
        _sync_resolved_config_to_wandb(
            wb_run=wb_run,
            method=str(args.method),
            dataset=str(args.dataset),
            seed=(int(args.seed) if args.seed is not None else None),
            resolved_config=dict(result.get("config", {})),
            loaded_paths=list(loaded_paths),
        )
        wandb.log({f"{args.dataset}/{args.method}/{k}": v for k, v in m.items()})
        _log_dataset_plots_to_wandb(
            outdir=outdir,
            method=str(args.method),
            dataset=str(args.dataset),
        )
        wandb.finish()


if __name__ == "__main__":
    main()
