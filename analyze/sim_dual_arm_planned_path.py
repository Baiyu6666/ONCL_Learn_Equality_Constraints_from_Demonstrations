#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.ur5_pybullet_utils import resolve_dual_ur5_base_cfg
from analyze.sim_dual_arm_ur5_demo import (
    DualUR5DemoSim,
    _load_planned_path_npz,
    _resolve_path,
)
from evaluation.evaluator import dual_arm_pose_true_constraint_error_arrays


def _summary_stats(vals: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _parse_snapshot_steps(text: str) -> list[int]:
    raw = str(text).strip()
    if not raw:
        return []
    vals: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        vals.append(int(token))
    return vals


def _parse_crop_frac(text: str) -> tuple[float, float, float, float] | None:
    raw = str(text).strip()
    if not raw:
        return None
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError("snapshot crop must have four comma-separated fractions: x0,y0,x1,y1")
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render a planned 12D dual-arm path in PyBullet.")
    ap.add_argument("--gui", type=int, default=1, help="0: no video, 1: DIRECT render video, 2: GUI render only")
    ap.add_argument("--dataset-config", type=str, default="configs/datasets/12d_dual_arm_traj.json")
    ap.add_argument("--plan-outdir", type=str, default="", help="Planner output dir containing dual_arm_learned_plans.npz; render outputs are also written here.")
    ap.add_argument("--planned-path-npz", type=str, default="", help="Optional explicit .npz path; overrides loading from --plan-outdir")
    ap.add_argument("--planned-path-key", type=str, default="path_0", help="Path key inside planned-path-npz")
    ap.add_argument(
        "--orientation-mode",
        type=str,
        default="overhand",
        choices=("none", "task", "overhand"),
        help="IK orientation target for tracking the planned task-space path.",
    )
    ap.add_argument("--ik-iters", type=int, default=160)
    ap.add_argument("--sim-dt", type=float, default=1.0 / 180.0)
    ap.add_argument("--max-joint-speed", type=float, default=0.75)
    ap.add_argument("--min-segment-time", type=float, default=0.06)
    ap.add_argument("--terminal-hold-time", type=float, default=0.45)
    ap.add_argument("--max-force", type=float, default=260.0)
    ap.add_argument("--video-width", type=int, default=1280)
    ap.add_argument("--video-height", type=int, default=900)
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--video-slowdown", type=float, default=1.2)
    ap.add_argument("--trace-stride", type=int, default=18)
    ap.add_argument("--arm-trace-radius", type=float, default=0.0030, help="Rendered radius for the two arm end-effector traces.")
    ap.add_argument("--save-snapshots", type=int, default=0, help="1 to save PNG snapshots using the same camera as gui=1 video rendering.")
    ap.add_argument("--snapshot-steps", type=str, default="", help="Comma-separated control-step indices for snapshots, e.g. 40,120,240.")
    ap.add_argument("--snapshot-dir", type=str, default="", help="Optional snapshot output directory. Defaults to <plan-outdir>/snapshots.")
    ap.add_argument(
        "--snapshot-crop-frac",
        type=str,
        default="0.10,0.08,0.86,0.80",
        help="Relative crop box for saved snapshots only, formatted as x0,y0,x1,y1.",
    )
    ap.add_argument("--realtime", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    with open(_resolve_path(str(args.dataset_config)), "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict)
    base_cfg = resolve_dual_ur5_base_cfg(cfg)

    npz_path = str(args.planned_path_npz).strip()
    plan_outdir = _resolve_path(str(args.plan_outdir))
    if not npz_path:
        if not str(args.plan_outdir).strip():
            raise ValueError("Provide either --planned-path-npz or --plan-outdir.")
        npz_path = os.path.join(plan_outdir, "dual_arm_learned_plans.npz")
    else:
        if not str(args.plan_outdir).strip():
            plan_outdir = os.path.dirname(_resolve_path(npz_path))
    npz_path = _resolve_path(npz_path)

    path = _load_planned_path_npz(npz_path, str(args.planned_path_key))
    os.makedirs(plan_outdir, exist_ok=True)
    snapshot_steps = _parse_snapshot_steps(str(args.snapshot_steps))
    snapshot_dir = _resolve_path(str(args.snapshot_dir)) if str(args.snapshot_dir).strip() else os.path.join(plan_outdir, "snapshots")
    snapshot_crop_frac = _parse_crop_frac(str(args.snapshot_crop_frac))

    video_path = None
    if int(args.gui) == 1:
        video_path = os.path.join(plan_outdir, "dual_arm_planned_path.mp4")

    sim = DualUR5DemoSim(
        gui=(int(args.gui) == 2),
        urdf_path=None,
        ee_link_index=None,
        tool_axis=None,
        base_cfg=base_cfg,
        sim_dt=float(args.sim_dt),
    )
    try:
        sim.build_guide_visuals(path["center"], path["p_left"], path["p_right"], cfg=cfg, draw_center_guide=False)
        q_left, ik_left = sim.solve_ik_path(
            sim.left,
            path["pose_left"],
            orientation_mode=str(args.orientation_mode),
            side="left",
            max_iters=int(args.ik_iters),
            seed_offset=0,
        )
        q_right, ik_right = sim.solve_ik_path(
            sim.right,
            path["pose_right"],
            orientation_mode=str(args.orientation_mode),
            side="right",
            max_iters=int(args.ik_iters),
            seed_offset=101,
        )
        t, ql_ref, qr_ref, qdl_ref, qdr_ref = sim.time_parameterize(
            q_left,
            q_right,
            max_joint_speed=float(args.max_joint_speed),
            min_segment_time=float(args.min_segment_time),
            terminal_hold_time=float(args.terminal_hold_time),
        )
        track = sim.track(
            ql_ref,
            qr_ref,
            qdl_ref,
            qdr_ref,
            video_path=video_path,
            video_width=int(args.video_width),
            video_height=int(args.video_height),
            video_fps=int(args.video_fps),
            video_slowdown=float(args.video_slowdown),
            realtime=bool(int(args.realtime)),
            trace_stride=int(args.trace_stride),
            max_force=float(args.max_force),
            arm_trace_radius=float(args.arm_trace_radius),
            draw_object_trace=True,
            object_trace_radius=0.003,
            snapshot_steps=(snapshot_steps if bool(int(args.save_snapshots)) else None),
            snapshot_dir=snapshot_dir,
            snapshot_crop_frac=(snapshot_crop_frac if bool(int(args.save_snapshots)) else None),
        )
        ref_left_pos = path["pose_left"][:, 0:3].astype(float)
        ref_right_pos = path["pose_right"][:, 0:3].astype(float)
        exec_left_pos = np.asarray(track["ee_left"], dtype=float)
        exec_right_pos = np.asarray(track["ee_right"], dtype=float)
        if len(exec_left_pos) != len(ref_left_pos):
            t_ref = np.linspace(0.0, 1.0, len(ref_left_pos), dtype=float)
            t_exec = np.linspace(0.0, 1.0, len(exec_left_pos), dtype=float)
            ref_left_pos = np.stack([np.interp(t_exec, t_ref, ref_left_pos[:, j]) for j in range(3)], axis=1)
            ref_right_pos = np.stack([np.interp(t_exec, t_ref, ref_right_pos[:, j]) for j in range(3)], axis=1)
        exec_left_err = np.linalg.norm(exec_left_pos - ref_left_pos, axis=1)
        exec_right_err = np.linalg.norm(exec_right_pos - ref_right_pos, axis=1)
        exec_pose = np.concatenate(
            [
                np.asarray(track["ee_left"], dtype=np.float32),
                np.asarray(track["ee_left_rpy"], dtype=np.float32),
                np.asarray(track["ee_right"], dtype=np.float32),
                np.asarray(track["ee_right_rpy"], dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        exec_true_err = dual_arm_pose_true_constraint_error_arrays(exec_pose, cfg)
        summary = {
            "planned_path_npz": npz_path,
            "planned_path_key": str(args.planned_path_key),
            "orientation_mode": str(args.orientation_mode),
            "sim_dt": float(args.sim_dt),
            "n_task_waypoints": int(len(path["pose_left"])),
            "n_control_steps": int(len(t)),
            "duration_s": float(t[-1]) if len(t) else 0.0,
            "left_ik": ik_left,
            "right_ik": ik_right,
            "exec_pos_err_left_mean": float(np.mean(exec_left_err)),
            "exec_pos_err_right_mean": float(np.mean(exec_right_err)),
            "exec_pos_err_left_max": float(np.max(exec_left_err)),
            "exec_pos_err_right_max": float(np.max(exec_right_err)),
            "exec_true_constraint": {
                "left_pos_err": _summary_stats(exec_true_err["left_pos_err"]),
                "right_pos_err": _summary_stats(exec_true_err["right_pos_err"]),
                "mean_pos_err": _summary_stats(exec_true_err["mean_pos_err"]),
                "left_ori_err_deg": _summary_stats(exec_true_err["left_ori_err_deg"]),
                "right_ori_err_deg": _summary_stats(exec_true_err["right_ori_err_deg"]),
                "mean_ori_err_deg": _summary_stats(exec_true_err["mean_ori_err_deg"]),
                "analytic_vector_dist": _summary_stats(exec_true_err["analytic_vector_dist"]),
                "center_err": _summary_stats(exec_true_err["center_err"]),
                "span_err": _summary_stats(exec_true_err["span_err"]),
            },
            "mean_joint_err_left": track["mean_joint_err_left"],
            "mean_joint_err_right": track["mean_joint_err_right"],
            "max_joint_err_left": track["max_joint_err_left"],
            "max_joint_err_right": track["max_joint_err_right"],
            "arm_trace_radius": float(args.arm_trace_radius),
            "save_snapshots": bool(int(args.save_snapshots)),
            "snapshot_steps": [int(v) for v in snapshot_steps],
            "snapshot_dir": (snapshot_dir if bool(int(args.save_snapshots)) else None),
            "snapshot_crop_frac": (list(snapshot_crop_frac) if snapshot_crop_frac is not None else None),
            "snapshot_paths": [str(v) for v in track.get("snapshot_paths", [])],
            "base_cfg": base_cfg,
            "video_path": video_path,
        }
        os.makedirs(plan_outdir, exist_ok=True)
        summary_path = os.path.join(plan_outdir, "dual_arm_planned_path_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(
            "[summary] "
            f"left_ik={ik_left['mean_pos_err']:.4f}/{ik_left['max_pos_err']:.4f}, "
            f"right_ik={ik_right['mean_pos_err']:.4f}/{ik_right['max_pos_err']:.4f}, "
            f"exec_pos={float(np.mean(exec_left_err)):.4f}/{float(np.mean(exec_right_err)):.4f}, "
            f"exec_true_pos={summary['exec_true_constraint']['mean_pos_err']['mean']:.4f}, "
            f"exec_true_ori_deg={summary['exec_true_constraint']['mean_ori_err_deg']['mean']:.2f}, "
            f"joint_err={track['mean_joint_err_left']:.4f}/{track['mean_joint_err_right']:.4f}"
        )
        print(f"[saved] {summary_path}")
        if video_path:
            print(f"[saved] {video_path}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
