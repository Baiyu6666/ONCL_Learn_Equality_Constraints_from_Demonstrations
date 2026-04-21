from __future__ import annotations

import os
import math
import re
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_UR5_URDF = os.path.join(_THIS_DIR, "assets", "UR5+gripper", "ur5_gripper.urdf")
UR5_LINK_LENGTHS = [0.425, 0.39225, 0.10915, 0.09465, 0.0823, 0.10]

DEFAULT_UR5_KINEMATICS_CFG = {
    # Default to vendored URDF inside this repository.
    "urdf_path": _REPO_UR5_URDF,
    "ee_link_index": -1,  # <0 means auto-pick
    "tool_axis": "x",
}

DEFAULT_UR5_RENDER_CFG = {
    "grasp_offset": 0.11,
    "grasp_axis_shift": 0.0,
    "debug_axes": False,
    "cylinder_rotate_90": True,
    "gripper_close_ratio": 0.78,
}

DEFAULT_DUAL_UR5_BASE_CFG = {
    "dual_arm_left_base_xyz": [-0.24, -0.70, 0.0],
    "dual_arm_left_base_rpy": [0.0, 0.0, math.pi / 2.0],
    "dual_arm_right_base_xyz": [0.24, 0.70, 0.0],
    "dual_arm_right_base_rpy": [0.0, 0.0, -math.pi / 2.0],
}


def resolve_ur5_kinematics_cfg(overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULT_UR5_KINEMATICS_CFG)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if k in cfg and v is not None})
    return cfg


def resolve_ur5_render_cfg(overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULT_UR5_RENDER_CFG)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if k in cfg and v is not None})
    return cfg


def resolve_dual_ur5_base_cfg(overrides: Any | None = None) -> dict:
    cfg = {k: list(v) for k, v in DEFAULT_DUAL_UR5_BASE_CFG.items()}
    if overrides is None:
        return cfg
    src = overrides if isinstance(overrides, dict) else vars(overrides)
    for k in cfg:
        v = src.get(k, None)
        if v is not None:
            cfg[k] = [float(x) for x in v]
    return cfg


@dataclass
class UR5PyBulletKinematics:
    client_id: int
    robot_id: int
    arm_joint_indices: list[int]
    ee_link_index: int
    tool_axis_index: int
    joint_lower: np.ndarray
    joint_upper: np.ndarray

    @staticmethod
    def from_env() -> "UR5PyBulletKinematics":
        urdf_path = os.environ.get("UR5_URDF_PATH", "").strip()
        ee_link_env = os.environ.get("UR5_EE_LINK_INDEX", "").strip()
        ee_idx = int(ee_link_env) if ee_link_env else None
        tool_axis = os.environ.get("UR5_TOOL_AXIS", "x").strip().lower()
        return UR5PyBulletKinematics.from_settings(urdf_path, ee_link_index=ee_idx, tool_axis=tool_axis)

    @staticmethod
    def from_settings(
        urdf_path: str,
        ee_link_index: int | None = None,
        tool_axis: str = "x",
    ) -> "UR5PyBulletKinematics":
        try:
            import pybullet as p  # type: ignore
            import pybullet_data  # type: ignore
        except Exception as e:
            raise RuntimeError(f"pybullet unavailable: {e}")

        if not urdf_path:
            raise RuntimeError("UR5_URDF_PATH is not set")
        if not os.path.exists(urdf_path):
            raise RuntimeError(f"UR5_URDF_PATH does not exist: {urdf_path}")

        cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        urdf_text = open(urdf_path, "r", encoding="utf-8").read()
        if "package://" in urdf_text:
            patched = _make_pybullet_friendly_urdf(urdf_path)
            rid = p.loadURDF(
                patched,
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=cid,
            )
        else:
            rid = p.loadURDF(
                urdf_path,
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=cid,
            )

        rev = []
        lo = []
        hi = []
        nj = p.getNumJoints(rid, physicsClientId=cid)
        for j in range(nj):
            info = p.getJointInfo(rid, j, physicsClientId=cid)
            if int(info[2]) == p.JOINT_REVOLUTE:
                rev.append(int(j))
                lj = float(info[8])
                hj = float(info[9])
                if (not np.isfinite(lj)) or (not np.isfinite(hj)) or (hj <= lj):
                    lj, hj = -math.pi, math.pi
                lo.append(lj)
                hi.append(hj)

        if len(rev) < 6:
            raise RuntimeError(f"UR5 model has fewer than 6 revolute joints: {len(rev)}")
        arm = rev[:6]
        lower = np.asarray(lo[:6], dtype=np.float32)
        upper = np.asarray(hi[:6], dtype=np.float32)

        ee_idx = int(ee_link_index) if ee_link_index is not None else pick_default_ee_link_index(rid, arm[-1], cid)

        tool_axis = str(tool_axis).strip().lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        tool_axis_idx = axis_map.get(tool_axis, 2)

        return UR5PyBulletKinematics(
            client_id=cid,
            robot_id=rid,
            arm_joint_indices=arm,
            ee_link_index=ee_idx,
            tool_axis_index=tool_axis_idx,
            joint_lower=lower,
            joint_upper=upper,
        )

    def _set_q(self, q: np.ndarray) -> None:
        import pybullet as p  # type: ignore

        for i, j in enumerate(self.arm_joint_indices):
            p.resetJointState(
                self.robot_id,
                j,
                targetValue=float(q[i]),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )

    def fk_batch(self, q_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import pybullet as p  # type: ignore

        q_batch = np.asarray(q_batch, dtype=np.float32)
        n = q_batch.shape[0]
        chain = np.zeros((n, 7, 3), dtype=np.float32)
        ee_pos = np.zeros((n, 3), dtype=np.float32)
        tool_axis = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            self._set_q(q_batch[i])
            chain[i, 0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            for k, jidx in enumerate(self.arm_joint_indices):
                ls = p.getLinkState(self.robot_id, jidx, computeForwardKinematics=True, physicsClientId=self.client_id)
                chain[i, k + 1] = np.asarray(ls[4], dtype=np.float32)

            ls_ee = p.getLinkState(
                self.robot_id, self.ee_link_index, computeForwardKinematics=True, physicsClientId=self.client_id
            )
            ee_pos[i] = np.asarray(ls_ee[4], dtype=np.float32)
            rot = np.asarray(p.getMatrixFromQuaternion(ls_ee[5]), dtype=np.float32).reshape(3, 3)
            axis = rot[:, self.tool_axis_index]
            axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
            tool_axis[i] = axis.astype(np.float32)

        return chain, ee_pos, tool_axis

    def tool_axis_batch(self, q_batch: np.ndarray) -> np.ndarray:
        _, _, axis = self.fk_batch(q_batch)
        return axis


def _candidate_mesh_paths(urdf_dir: str, basename: str) -> list[str]:
    stem, ext = os.path.splitext(basename)
    names = [
        basename,
        basename.lower(),
        basename.upper(),
        stem + ".stl",
        stem + ".STL",
        stem + ".dae",
        stem + ".DAE",
    ]
    roots = [
        os.path.join(urdf_dir, "mesh"),
        os.path.join(urdf_dir, "mesh", "visual"),
        os.path.join(urdf_dir, "meshes"),
        os.path.join(urdf_dir, "meshes", "visual"),
    ]
    out: list[str] = []
    for r in roots:
        for n in names:
            out.append(os.path.join(r, n))
    return out


def _resolve_mesh_uri(urdf_dir: str, uri: str) -> str:
    if not uri.startswith("package://"):
        return uri
    b = os.path.basename(uri)
    for cand in _candidate_mesh_paths(urdf_dir, b):
        if os.path.exists(cand):
            return cand
    return uri


def _make_pybullet_friendly_urdf(urdf_path: str) -> str:
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    text = open(urdf_path, "r", encoding="utf-8").read()

    def _sub(m: re.Match[str]) -> str:
        pre, uri, post = m.group(1), m.group(2), m.group(3)
        new_uri = _resolve_mesh_uri(urdf_dir, uri)
        return f'{pre}{new_uri}{post}'

    patched = re.sub(r'(filename\s*=\s*")([^"]+)(")', _sub, text)
    fd, tmp = tempfile.mkstemp(prefix="ur5_patch_", suffix=".urdf")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(patched)
    return tmp


def pick_default_ee_link_index(robot_id: int, fallback_joint_index: int, client_id: int) -> int:
    import pybullet as p  # type: ignore

    prefer = (
        "tcp",
        "tool0",
        "ee_link",
        "gripper",
        "wrist_3_link",
        "wrist3_link",
        "wrist3",
    )
    nj = p.getNumJoints(robot_id, physicsClientId=client_id)
    best = None
    best_rank = 10**9
    for j in range(nj):
        info = p.getJointInfo(robot_id, j, physicsClientId=client_id)
        link_name = info[12].decode("utf-8", errors="ignore").lower()
        for r, key in enumerate(prefer):
            if key in link_name and r < best_rank:
                best_rank = r
                best = int(j)
    if best is not None:
        return int(best)
    return int(fallback_joint_index)
