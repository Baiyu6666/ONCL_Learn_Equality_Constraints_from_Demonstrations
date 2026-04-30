from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from datasets.ur5_pybullet_utils import (
    _make_pybullet_friendly_urdf,
    pick_default_ee_link_index,
    resolve_ur5_kinematics_cfg,
)


@dataclass
class IKConfig:
    method: str = "pybullet"
    max_iters: int = 64
    damping: float = 0.05
    step_size: float = 0.6
    max_delta_norm: float = 0.18
    pos_tol: float = 0.004
    ori_tol_rad: float = math.radians(2.5)
    warm_start_weight: float = 0.01
    rest_pose_weight: float = 0.002
    search_first_seed: bool = True
    seed_search_samples: int = 256
    seed_search_seed: int = 0
    seed_search_ori_weight: float = 0.06
    seed_search_home_std: float = 0.28
    tool_frame_offset_rpy: tuple[float, float, float] = (0.0, math.pi / 2.0, 0.0)
    continuity_repair: bool = True
    continuity_jump_abs: float = 0.08
    continuity_jump_factor: float = 4.0
    pybullet_rest_home_blend: float = 0.35


@dataclass
class JointTrackConfig:
    sim_dt: float = 1.0 / 240.0
    max_joint_speed: float = 0.9
    min_segment_time: float = 0.04
    trajectory_time_scale: float = 1.0
    endpoint_ramp_time: float = 0.20
    reference_smooth_passes: int = 12
    terminal_hold_time: float = 0.35
    position_gain: float = 0.32
    velocity_gain: float = 1.05
    max_force: float = 140.0
    settle_steps: int = 24
    realtime: bool = False
    video_fps: int = 30
    video_slowdown: float = 3.0
    video_width: int = 1024
    video_height: int = 768
    draw_ee_trace: bool = True
    draw_ref_trace: bool = False
    draw_surface_wireframe: bool = True
    keep_visuals_on_finish: bool = False
    preserve_trace_history: bool = False
    enable_keyboard_pause: bool = False
    pause_poll_dt: float = 1.0 / 60.0
    trace_stride: int = 16
    trace_width: float = 3.0
    surface_line_stride: int = 4
    surface_line_width: float = 1.2
    waypoint_marker_radius: float = 0.006


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    roll = float(rpy[0])
    pitch = float(rpy[1])
    yaw = float(rpy[2])
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    q = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return q


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_orientation_error(desired_xyzw: np.ndarray, current_xyzw: np.ndarray) -> np.ndarray:
    q_err = _quat_multiply(desired_xyzw, _quat_conjugate(current_xyzw))
    if float(q_err[3]) < 0.0:
        q_err = -q_err
    vec = q_err[:3].astype(np.float32)
    sin_half = float(np.linalg.norm(vec))
    if sin_half < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    axis = vec / sin_half
    angle = 2.0 * math.atan2(sin_half, max(float(q_err[3]), 1e-8))
    return (axis * angle).astype(np.float32)


def _joint_array(val: float | list[float] | np.ndarray, n: int) -> list[float]:
    if isinstance(val, np.ndarray):
        arr = val.astype(np.float32).reshape(-1)
        if len(arr) == 1:
            return [float(arr[0])] * n
        if len(arr) != n:
            raise ValueError(f"expected {n} joint values, got {len(arr)}")
        return [float(v) for v in arr]
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            return [float(val[0])] * n
        if len(val) != n:
            raise ValueError(f"expected {n} joint values, got {len(val)}")
        return [float(v) for v in val]
    return [float(val)] * n


class _FFmpegVideoWriter:
    def __init__(self, *, out_path: str, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg binary not found in PATH")
        self.out_path = os.path.abspath(out_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(0.1, fps))
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            self.out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def append_data(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.shape != (self.height, self.width, 3):
            raise ValueError(
                f"video frame has shape {arr.shape}, expected {(self.height, self.width, 3)}"
            )
        self.proc.stdin.write(arr.tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.wait(timeout=30)
        if self.proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg exited with code {self.proc.returncode}")


def _retime_video_ffmpeg(*, src_path: str, dst_path: str, slowdown: float) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary not found in PATH")
    scale = float(max(1e-3, slowdown))
    if abs(scale - 1.0) <= 1e-6:
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.move(src_path, dst_path)
        return os.path.abspath(dst_path)
    tmp_out = os.path.abspath(dst_path)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        os.path.abspath(src_path),
        "-filter:v",
        f"setpts={scale:.6f}*PTS",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        tmp_out,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg retime failed with code {proc.returncode}")
    if os.path.exists(src_path) and os.path.abspath(src_path) != tmp_out:
        try:
            os.remove(src_path)
        except Exception:
            pass
    return tmp_out


class UR5TrajectoryController:
    def __init__(
        self,
        *,
        gui: bool = False,
        hide_gripper: bool = True,
        urdf_path: str | None = None,
        ee_link_index: int | None = None,
        tool_axis: str | None = None,
        sim_dt: float = 1.0 / 240.0,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        try:
            import pybullet as p  # type: ignore
            import pybullet_data  # type: ignore
        except Exception as e:
            raise RuntimeError(f"pybullet unavailable: {e}")

        self._p = p
        self._pybullet_data = pybullet_data
        self._gui = bool(gui)
        self._hide_gripper = bool(hide_gripper)
        self._surface_body_ids: list[int] = []
        self._surface_mesh_path: str | None = None
        self._surface_mesh_paths: list[str] = []
        self._surface_tex_path: str | None = None
        self._obstacle_body_ids: list[int] = []
        self._obstacle_mesh_path: str | None = None
        self._trace_body_ids: list[int] = []
        self._trace_history_segment_idx: int = 0
        self._marker_body_ids: list[int] = []
        self._scene_body_ids: list[int] = []
        self._scene_tex_paths: list[str] = []
        kin_cfg = resolve_ur5_kinematics_cfg(
            {
                "urdf_path": urdf_path,
                "ee_link_index": ee_link_index,
                "tool_axis": tool_axis,
            }
        )
        self.urdf_path = str(kin_cfg["urdf_path"])
        self.ee_link_index_override = kin_cfg.get("ee_link_index")
        self.tool_axis = str(kin_cfg["tool_axis"])
        mode = p.GUI if self._gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setTimeStep(float(sim_dt), physicsClientId=self.client_id)
        p.setGravity(float(gravity[0]), float(gravity[1]), float(gravity[2]), physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self._build_scene_backdrop()

        load_path = self.urdf_path
        try:
            txt = open(self.urdf_path, "r", encoding="utf-8").read()
        except Exception:
            txt = ""
        if "package://" in txt:
            load_path = _make_pybullet_friendly_urdf(self.urdf_path)
        self.robot_id = p.loadURDF(
            load_path,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id,
        )
        self._patched_urdf = load_path if load_path != self.urdf_path else None

        self.arm_joint_indices: list[int] = []
        self.ik_joint_indices: list[int] = []
        lo: list[float] = []
        hi: list[float] = []
        nj = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        for j in range(nj):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            if int(info[2]) != p.JOINT_FIXED:
                self.ik_joint_indices.append(int(j))
            if int(info[2]) == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(int(j))
                lj = float(info[8])
                hj = float(info[9])
                if (not np.isfinite(lj)) or (not np.isfinite(hj)) or (hj <= lj):
                    lj, hj = -math.pi, math.pi
                lo.append(lj)
                hi.append(hj)
        if len(self.arm_joint_indices) < 6:
            raise RuntimeError(f"UR5 model has fewer than 6 revolute joints: {len(self.arm_joint_indices)}")
        self.arm_joint_indices = self.arm_joint_indices[:6]
        self.q_lo = np.asarray(lo[:6], dtype=np.float32)
        self.q_hi = np.asarray(hi[:6], dtype=np.float32)
        self.arm_joint_ranges = np.maximum(self.q_hi - self.q_lo, 1e-3).astype(np.float32)
        self.arm_ik_positions = [int(self.ik_joint_indices.index(j)) for j in self.arm_joint_indices]
        ee_idx_cfg = None if int(self.ee_link_index_override) < 0 else int(self.ee_link_index_override)
        self.ee_link_index = (
            int(ee_idx_cfg)
            if ee_idx_cfg is not None
            else pick_default_ee_link_index(self.robot_id, self.arm_joint_indices[-1], self.client_id)
        )

        self.gripper_joint_indices: list[int] = []
        for j in range(nj):
            if j in self.arm_joint_indices:
                continue
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            if int(info[2]) in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.gripper_joint_indices.append(int(j))
        self.home_q = self._default_overhand_home_q()
        self.camera_distance = 2.3
        self.camera_yaw = 32.0
        self.camera_pitch = -30.0
        self.camera_fov = 60.0
        self.camera_target = np.asarray([0.15, -0.35, 0.35], dtype=np.float32)
        if self._hide_gripper:
            self.hide_gripper_links()
        self.disable_default_motors()

    def _default_overhand_home_q(self) -> np.ndarray:
        # Bias IK toward an overhand posture so the arm approaches the surface from above.
        q_nom = np.asarray([0.0, -1.25, 1.85, -2.10, -1.57, 0.0], dtype=np.float32)
        return np.clip(q_nom, self.q_lo, self.q_hi).astype(np.float32)

    def close(self) -> None:
        self._clear_surface_visuals()
        self._clear_obstacle_visual()
        self._clear_trace_visuals()
        self._clear_scene_backdrop()
        if getattr(self, "client_id", None) is not None:
            try:
                self._p.disconnect(physicsClientId=self.client_id)
            except Exception:
                pass
            self.client_id = None
        if getattr(self, "_patched_urdf", None):
            try:
                os.remove(self._patched_urdf)
            except Exception:
                pass
            self._patched_urdf = None

    def __enter__(self) -> "UR5TrajectoryController":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def disable_default_motors(self) -> None:
        p = self._p
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(self.arm_joint_indices),
            forces=[0.0] * len(self.arm_joint_indices),
            physicsClientId=self.client_id,
        )

    def hide_gripper_links(self) -> None:
        p = self._p
        nj = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        for j in range(nj):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            link_name = info[12].decode("utf-8", errors="ignore").lower()
            if "gripper" not in link_name:
                continue
            try:
                p.changeVisualShape(
                    self.robot_id,
                    j,
                    rgbaColor=[0.0, 0.0, 0.0, 0.0],
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass
            try:
                p.setCollisionFilterGroupMask(
                    self.robot_id,
                    j,
                    collisionFilterGroup=0,
                    collisionFilterMask=0,
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

    def reset_joint_state(self, q: np.ndarray) -> None:
        p = self._p
        qv = np.clip(np.asarray(q, dtype=np.float32).reshape(-1), self.q_lo, self.q_hi)
        for i, j in enumerate(self.arm_joint_indices):
            p.resetJointState(
                self.robot_id,
                int(j),
                targetValue=float(qv[i]),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        p = self._p
        sts = p.getJointStates(self.robot_id, self.arm_joint_indices, physicsClientId=self.client_id)
        q = np.asarray([float(s[0]) for s in sts], dtype=np.float32)
        qd = np.asarray([float(s[1]) for s in sts], dtype=np.float32)
        return q, qd

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        p = self._p
        ls = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        pos = np.asarray(ls[4], dtype=np.float32)
        quat = np.asarray(ls[5], dtype=np.float32)
        quat /= max(float(np.linalg.norm(quat)), 1e-8)
        return pos, quat

    def set_camera(
        self,
        *,
        distance: float,
        yaw: float,
        pitch: float,
        target_position: list[float] | tuple[float, float, float] | np.ndarray,
        fov: float | None = None,
    ) -> None:
        self.camera_distance = float(distance)
        self.camera_yaw = float(yaw)
        self.camera_pitch = float(pitch)
        if fov is not None:
            self.camera_fov = float(fov)
        self.camera_target = np.asarray(target_position, dtype=np.float32).reshape(3)
        if self._gui:
            try:
                self._p.resetDebugVisualizerCamera(
                    cameraDistance=self.camera_distance,
                    cameraYaw=self.camera_yaw,
                    cameraPitch=self.camera_pitch,
                    cameraTargetPosition=[float(v) for v in self.camera_target],
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

    def capture_frame(self, *, width: int, height: int) -> np.ndarray:
        p = self._p
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[float(v) for v in self.camera_target],
            distance=float(self.camera_distance),
            yaw=float(self.camera_yaw),
            pitch=float(self.camera_pitch),
            roll=0.0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=float(self.camera_fov),
            aspect=float(width) / max(float(height), 1.0),
            nearVal=0.02,
            farVal=6.0,
        )
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self._gui else p.ER_TINY_RENDERER
        _, _, rgba, _, _ = p.getCameraImage(
            width=int(width),
            height=int(height),
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=renderer,
            physicsClientId=self.client_id,
        )
        frame = np.asarray(rgba, dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3].copy()
        if not self._gui:
            crop_top = int(round(0.12 * float(height)))
            crop_top = max(0, min(int(height) - 2, crop_top))
            if crop_top > 0:
                cropped = frame[crop_top:, :, :]
                src_h, src_w = cropped.shape[:2]
                yy = np.linspace(0, src_h - 1, int(height)).astype(np.int32)
                xx = np.linspace(0, src_w - 1, int(width)).astype(np.int32)
                frame = cropped[yy][:, xx]
            # Replace the renderer's near-white empty background with a softer
            # neutral tone so the far field does not look like a white void.
            bg_mask = (
                (frame[..., 0] >= 245)
                & (frame[..., 1] >= 245)
                & (frame[..., 2] >= 245)
            )
            if np.any(bg_mask):
                frame[bg_mask] = np.asarray([220, 228, 232], dtype=np.uint8)
        return frame

    def _clear_surface_visuals(self) -> None:
        p = self._p
        for bid in self._surface_body_ids:
            try:
                p.removeBody(int(bid), physicsClientId=self.client_id)
            except Exception:
                pass
        self._surface_body_ids = []
        for mesh_path in list(getattr(self, "_surface_mesh_paths", [])):
            try:
                os.remove(mesh_path)
            except Exception:
                pass
        self._surface_mesh_paths = []
        if self._surface_mesh_path:
            try:
                os.remove(self._surface_mesh_path)
            except Exception:
                pass
            self._surface_mesh_path = None
        if self._surface_tex_path:
            try:
                os.remove(self._surface_tex_path)
            except Exception:
                pass
            self._surface_tex_path = None

    def _clear_obstacle_visual(self) -> None:
        p = self._p
        for bid in self._obstacle_body_ids:
            try:
                p.removeBody(int(bid), physicsClientId=self.client_id)
            except Exception:
                pass
        self._obstacle_body_ids = []
        if self._obstacle_mesh_path:
            try:
                os.remove(self._obstacle_mesh_path)
            except Exception:
                pass
            self._obstacle_mesh_path = None

    def _clear_trace_visuals(self) -> None:
        p = self._p
        for bid in self._trace_body_ids:
            try:
                p.removeBody(int(bid), physicsClientId=self.client_id)
            except Exception:
                pass
        self._trace_body_ids = []
        self._trace_history_segment_idx = 0

    def _clear_marker_visuals(self) -> None:
        p = self._p
        for bid in self._marker_body_ids:
            try:
                p.removeBody(int(bid), physicsClientId=self.client_id)
            except Exception:
                pass
        self._marker_body_ids = []

    def _clear_scene_backdrop(self) -> None:
        p = self._p
        for bid in self._scene_body_ids:
            try:
                p.removeBody(int(bid), physicsClientId=self.client_id)
            except Exception:
                pass
        self._scene_body_ids = []
        for tex_path in list(getattr(self, "_scene_tex_paths", [])):
            try:
                os.remove(tex_path)
            except Exception:
                pass
        self._scene_tex_paths = []

    def _add_visual_box(
        self,
        *,
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
        specular: tuple[float, float, float] = (0.04, 0.04, 0.04),
        body_list: list[int] | None = None,
    ) -> int:
        p = self._p
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[float(v) for v in half_extents],
            rgbaColor=[float(v) for v in rgba],
            specularColor=[float(v) for v in specular],
            physicsClientId=self.client_id,
        )
        bid = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(vis),
            baseCollisionShapeIndex=-1,
            basePosition=[float(v) for v in center],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        if body_list is None:
            self._scene_body_ids.append(int(bid))
        else:
            body_list.append(int(bid))
        return int(bid)

    def _build_scene_backdrop(
        self,
        surface_bounds: tuple[float, float, float, float] | None = None,
        *,
        table_top_z: float = 0.90,
    ) -> None:
        self._clear_scene_backdrop()
        if surface_bounds is None:
            return
        x_min, x_max, y_min, y_max = [float(v) for v in surface_bounds]
        obj_cx = 0.5 * (x_min + x_max)
        obj_cy = 0.5 * (y_min + y_max)
        table_w = max(float(x_max - x_min) + 0.42, 0.90)
        table_d = max(float(y_max - y_min) + 0.44, 1.00)
        table_x_min = obj_cx - 0.5 * table_w
        table_x_max = obj_cx + 0.5 * table_w
        table_y_min = obj_cy - 0.5 * table_d
        table_y_max = obj_cy + 0.5 * table_d
        cx = 0.5 * (table_x_min + table_x_max)
        cy = 0.5 * (table_y_min + table_y_max)
        hx = 0.5 * (table_x_max - table_x_min)
        hy = 0.5 * (table_y_max - table_y_min)

        top_thickness = 0.055
        leg_h = max(0.12, float(table_top_z) - 0.5 * top_thickness)
        top_center_z = float(table_top_z) - 0.5 * top_thickness
        self._add_visual_box(
            center=(cx, cy, top_center_z),
            half_extents=(hx, hy, 0.5 * top_thickness),
            rgba=(0.58, 0.60, 0.60, 1.0),
            specular=(0.04, 0.04, 0.04),
        )
        apron_z = float(table_top_z) - top_thickness - 0.035
        apron_h = 0.032
        self._add_visual_box(
            center=(cx, table_y_min + 0.035, apron_z),
            half_extents=(hx, 0.025, apron_h),
            rgba=(0.43, 0.44, 0.44, 1.0),
            specular=(0.03, 0.03, 0.03),
        )
        self._add_visual_box(
            center=(cx, table_y_max - 0.035, apron_z),
            half_extents=(hx, 0.025, apron_h),
            rgba=(0.43, 0.44, 0.44, 1.0),
            specular=(0.03, 0.03, 0.03),
        )
        leg_half = 0.035
        leg_z = 0.5 * leg_h
        for lx in (table_x_min + 0.09, table_x_max - 0.09):
            for ly in (table_y_min + 0.09, table_y_max - 0.09):
                self._add_visual_box(
                    center=(lx, ly, leg_z),
                    half_extents=(leg_half, leg_half, 0.5 * leg_h),
                    rgba=(0.36, 0.37, 0.37, 1.0),
                    specular=(0.025, 0.025, 0.025),
                )

    def _write_surface_top_mesh_obj(self, surf: np.ndarray) -> str | None:
        surf = np.asarray(surf, dtype=np.float32)
        if surf.ndim != 3 or surf.shape[2] != 3:
            return None
        h, w = int(surf.shape[0]), int(surf.shape[1])
        if h < 2 or w < 2:
            return None
        tmp = tempfile.NamedTemporaryFile(prefix="sine_surface_top_", suffix=".obj", delete=False)
        tmp_path = tmp.name
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("# auto-generated textured sine-surface top\n")
            for i in range(h):
                for j in range(w):
                    x, y, z = [float(v) for v in surf[i, j]]
                    f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
            for i in range(h):
                v = float(i) / max(h - 1, 1)
                for j in range(w):
                    u = float(j) / max(w - 1, 1)
                    f.write(f"vt {u:.8f} {v:.8f}\n")
            for i in range(h - 1):
                for j in range(w - 1):
                    v00 = i * w + j + 1
                    v01 = v00 + 1
                    v10 = (i + 1) * w + j + 1
                    v11 = v10 + 1
                    f.write(f"f {v00}/{v00} {v11}/{v11} {v10}/{v10}\n")
                    f.write(f"f {v00}/{v00} {v01}/{v01} {v11}/{v11}\n")
                    f.write(f"f {v10}/{v10} {v11}/{v11} {v00}/{v00}\n")
                    f.write(f"f {v11}/{v11} {v01}/{v01} {v00}/{v00}\n")
        return tmp_path

    def _write_surface_body_mesh_obj(self, surf: np.ndarray, *, bottom_z: float | None = None) -> str | None:
        surf = np.asarray(surf, dtype=np.float32)
        if surf.ndim != 3 or surf.shape[2] != 3:
            return None
        h, w = int(surf.shape[0]), int(surf.shape[1])
        if h < 2 or w < 2:
            return None
        z_min = float(np.nanmin(surf[..., 2]))
        if bottom_z is None:
            bottom_z = z_min - 0.075
        bottom_z = min(float(bottom_z), z_min - 0.018)
        tmp = tempfile.NamedTemporaryFile(prefix="sine_surface_body_", suffix=".obj", delete=False)
        tmp_path = tmp.name
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("# auto-generated solid sine-surface body\n")
            for i in range(h):
                for j in range(w):
                    x, y, z = [float(v) for v in surf[i, j]]
                    f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
            for i in range(h):
                for j in range(w):
                    x, y = [float(v) for v in surf[i, j, :2]]
                    f.write(f"v {x:.8f} {y:.8f} {float(bottom_z):.8f}\n")
            top_offset = 0
            bottom_offset = h * w
            for i in range(h - 1):
                for j in range(w - 1):
                    b00 = bottom_offset + i * w + j + 1
                    b01 = b00 + 1
                    b10 = bottom_offset + (i + 1) * w + j + 1
                    b11 = b10 + 1
                    f.write(f"f {b11} {b10} {b00}\n")
                    f.write(f"f {b01} {b11} {b00}\n")

            def _face(a_top: int, b_top: int, a_bot: int, b_bot: int) -> None:
                f.write(f"f {a_top} {a_bot} {b_bot}\n")
                f.write(f"f {a_top} {b_bot} {b_top}\n")

            for j in range(w - 1):
                t0 = top_offset + j + 1
                t1 = top_offset + j + 2
                b0 = bottom_offset + j + 1
                b1 = bottom_offset + j + 2
                _face(t0, t1, b0, b1)
                t0 = top_offset + (h - 1) * w + j + 1
                t1 = top_offset + (h - 1) * w + j + 2
                b0 = bottom_offset + (h - 1) * w + j + 1
                b1 = bottom_offset + (h - 1) * w + j + 2
                _face(t1, t0, b1, b0)
            for i in range(h - 1):
                t0 = top_offset + i * w + 1
                t1 = top_offset + (i + 1) * w + 1
                b0 = bottom_offset + i * w + 1
                b1 = bottom_offset + (i + 1) * w + 1
                _face(t1, t0, b1, b0)
                t0 = top_offset + i * w + w
                t1 = top_offset + (i + 1) * w + w
                b0 = bottom_offset + i * w + w
                b1 = bottom_offset + (i + 1) * w + w
                _face(t0, t1, b0, b1)
        return tmp_path

    def _write_surface_texture_png(self, width: int = 256, height: int = 256) -> str:
        w = int(max(32, width))
        h = int(max(32, height))
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        rng = np.random.default_rng(17)
        base = np.asarray([182.0, 176.0, 166.0], dtype=np.float32).reshape(1, 1, 3)
        warm = 7.0 * np.sin(2.0 * np.pi * (xx * 2.1 + yy * 0.4))[..., None]
        grain = rng.normal(0.0, 5.5, size=(h, w, 1)).astype(np.float32)
        speck = ((rng.random((h, w, 1), dtype=np.float32) > 0.986).astype(np.float32) * -26.0)
        brushed = 4.0 * np.sin(2.0 * np.pi * yy * 18.0)[..., None]
        rgb = np.clip(base + warm + grain + speck + brushed, 112.0, 232.0).astype(np.uint8)
        tmp = tempfile.NamedTemporaryFile(prefix="sine_surface_tex_", suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        Image.fromarray(rgb, mode="RGB").save(tmp_path, format="PNG")
        return tmp_path

    def _preferred_surface_texture_path(self) -> str | None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates = [
            # Poly Haven, CC0:
            # https://polyhaven.com/a/plaster_grey_04
            os.path.join(root, "datasets", "assets", "textures", "polyhaven_plaster_grey_04_diff_1k.png"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _prepare_texture_for_pybullet(self, src_path: str) -> str:
        src = os.path.abspath(src_path)
        # PyBullet often rejects 16-bit/RGBA PNGs. Re-encode to 8-bit RGB PNG.
        img = Image.open(src).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(prefix="sine_surface_tex_ext_", suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        img.save(tmp_path, format="PNG")
        return tmp_path

    def _build_surface_visuals(
        self,
        surf: np.ndarray,
        *,
        stride: int,
        line_width: float,
    ) -> None:
        self._clear_surface_visuals()
        surf_np = np.asarray(surf, dtype=np.float32)
        z_min = float(np.nanmin(surf_np[..., 2]))
        bottom_z = z_min - 0.075
        x_min = float(np.nanmin(surf_np[..., 0]))
        x_max = float(np.nanmax(surf_np[..., 0]))
        y_min = float(np.nanmin(surf_np[..., 1]))
        y_max = float(np.nanmax(surf_np[..., 1]))
        self._build_scene_backdrop(
            (x_min, x_max, y_min, y_max),
            table_top_z=float(bottom_z),
        )
        top_mesh_path = self._write_surface_top_mesh_obj(surf_np)
        body_mesh_path = self._write_surface_body_mesh_obj(surf_np, bottom_z=bottom_z)
        if top_mesh_path is None or body_mesh_path is None:
            for mesh_path in (top_mesh_path, body_mesh_path):
                if mesh_path:
                    try:
                        os.remove(mesh_path)
                    except Exception:
                        pass
            return
        p = self._p
        body_vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=body_mesh_path,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[0.38, 0.43, 0.40, 1.0],
            specularColor=[0.025, 0.025, 0.022],
            physicsClientId=self.client_id,
        )
        body_bid = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(body_vis),
            baseCollisionShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        top_vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=top_mesh_path,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[0.96, 0.96, 0.96, 1.0],
            specularColor=[0.03, 0.03, 0.03],
            physicsClientId=self.client_id,
        )
        top_bid = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(top_vis),
            baseCollisionShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        try:
            preferred_tex = self._preferred_surface_texture_path()
            if preferred_tex is not None:
                tex_path = self._prepare_texture_for_pybullet(preferred_tex)
            else:
                tex_path = self._write_surface_texture_png(width=320, height=320)
            tex_id = p.loadTexture(tex_path, physicsClientId=self.client_id)
            p.changeVisualShape(
                int(top_bid),
                -1,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                textureUniqueId=int(tex_id),
                specularColor=[0.03, 0.03, 0.03],
                physicsClientId=self.client_id,
            )
            self._surface_tex_path = tex_path
        except Exception:
            if 'tex_path' in locals():
                try:
                    if os.path.isfile(str(tex_path)):
                        os.remove(tex_path)
                except Exception:
                    pass
        self._surface_body_ids.extend([int(body_bid), int(top_bid)])
        self._surface_mesh_paths.extend([body_mesh_path, top_mesh_path])

    def _add_visual_cylinder_segment(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        radius: float,
        rgba: tuple[float, float, float, float],
        specular: tuple[float, float, float] = (0.06, 0.05, 0.05),
        body_list: list[int] | None = None,
    ) -> None:
        p = self._p
        a3 = np.asarray(a, dtype=np.float32).reshape(3)
        b3 = np.asarray(b, dtype=np.float32).reshape(3)
        d = b3 - a3
        length = float(np.linalg.norm(d))
        if not np.isfinite(length) or length <= 1e-6:
            return
        mid = 0.5 * (a3 + b3)
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        dir_u = d / length
        dot = float(np.clip(np.dot(z_axis, dir_u), -1.0, 1.0))
        if dot > 1.0 - 1e-7:
            quat = [0.0, 0.0, 0.0, 1.0]
        elif dot < -1.0 + 1e-7:
            quat = [1.0, 0.0, 0.0, 0.0]
        else:
            axis = np.cross(z_axis, dir_u)
            axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
            ang = math.acos(dot)
            s = math.sin(0.5 * ang)
            quat = [float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), float(math.cos(0.5 * ang))]
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=float(radius),
            length=float(length),
            rgbaColor=[float(v) for v in rgba],
            specularColor=[float(v) for v in specular],
            physicsClientId=self.client_id,
        )
        bid = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(vis),
            baseCollisionShapeIndex=-1,
            basePosition=[float(v) for v in mid],
            baseOrientation=quat,
            physicsClientId=self.client_id,
        )
        if body_list is None:
            self._obstacle_body_ids.append(int(bid))
        else:
            body_list.append(int(bid))

    def _add_visual_sphere_marker(
        self,
        center: np.ndarray,
        *,
        radius: float,
        rgba: tuple[float, float, float, float],
        specular: tuple[float, float, float] = (0.04, 0.03, 0.03),
        body_list: list[int] | None = None,
    ) -> None:
        p = self._p
        c = np.asarray(center, dtype=np.float32).reshape(3)
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=float(radius),
            rgbaColor=[float(v) for v in rgba],
            specularColor=[float(v) for v in specular],
            physicsClientId=self.client_id,
        )
        bid = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(vis),
            baseCollisionShapeIndex=-1,
            basePosition=[float(v) for v in c],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        if body_list is None:
            self._obstacle_body_ids.append(int(bid))
        else:
            body_list.append(int(bid))

    def build_waypoint_markers(
        self,
        waypoints_xyz: np.ndarray,
        *,
        radius: float = 0.010,
        rgba: tuple[float, float, float, float] = (0.09, 0.10, 0.12, 0.98),
        lift: float = 0.010,
        connect: bool = False,
        connect_rgba: tuple[float, float, float, float] = (0.25, 0.29, 0.35, 0.78),
        connect_radius: float = 0.0013,
    ) -> None:
        self._clear_marker_visuals()
        pts = np.asarray(waypoints_xyz, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return
        vis_pts = pts.copy()
        vis_pts[:, 2] += float(max(0.0, lift))
        for i in range(len(vis_pts)):
            self._add_visual_sphere_marker(
                vis_pts[i],
                radius=float(radius),
                rgba=rgba,
                body_list=self._marker_body_ids,
            )
        if bool(connect) and len(vis_pts) >= 2:
            for i in range(1, len(vis_pts)):
                self._add_visual_cylinder_segment(
                    vis_pts[i - 1],
                    vis_pts[i],
                    radius=float(connect_radius),
                    rgba=connect_rgba,
                    specular=(0.02, 0.02, 0.02),
                    body_list=self._marker_body_ids,
                )

    def _damage_radius(self, theta: np.ndarray, radius: float) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float32)
        return (
            float(radius)
            * (0.80 + 0.11 * np.sin(3.0 * th + 0.4) + 0.07 * np.cos(5.0 * th - 0.25))
        ).astype(np.float32)

    def _surface_z_from_grid_xy(self, surf: np.ndarray, x: float, y: float) -> float:
        pts = np.asarray(surf, dtype=np.float32)
        xy = pts[..., :2].reshape(-1, 2)
        z = pts[..., 2].reshape(-1)
        d2 = np.sum((xy - np.asarray([x, y], dtype=np.float32)[None, :]) ** 2, axis=1)
        return float(z[int(np.argmin(d2))])

    def _build_obstacle_visual(
        self,
        *,
        center_xy: np.ndarray,
        radius: float,
        z_min: float,
        z_max: float,
        surf: np.ndarray | None = None,
    ) -> None:
        self._clear_obstacle_visual()
        if surf is None:
            return
        surf_np = np.asarray(surf, dtype=np.float32)
        center_xy = np.asarray(center_xy, dtype=np.float32)

        # Sparse, irregular damage blotches slightly above the surface.
        blob_templates = [
            (-0.34, -0.22, 0.26, (0.46, 0.22, 0.14, 0.96)),
            (0.00, -0.08, 0.34, (0.42, 0.19, 0.12, 0.96)),
            (0.30, 0.08, 0.22, (0.50, 0.24, 0.16, 0.96)),
            (-0.16, 0.26, 0.20, (0.38, 0.16, 0.11, 0.96)),
            (0.20, -0.36, 0.17, (0.54, 0.28, 0.17, 0.94)),
            (-0.42, 0.10, 0.16, (0.34, 0.14, 0.10, 0.94)),
            (0.48, -0.06, 0.13, (0.44, 0.20, 0.13, 0.92)),
            (-0.06, 0.48, 0.12, (0.40, 0.17, 0.12, 0.92)),
        ]
        for ox, oy, rr, rgba in blob_templates:
            px = float(center_xy[0] + float(radius) * ox)
            py = float(center_xy[1] + float(radius) * oy)
            pz = self._surface_z_from_grid_xy(surf_np, px, py) + 0.012
            self._add_visual_sphere_marker(
                np.asarray([px, py, pz], dtype=np.float32),
                radius=max(0.006, float(radius) * rr * 1.15),
                rgba=rgba,
            )

        crack_templates = [
            np.asarray([[-0.52, -0.10], [-0.18, -0.04], [0.10, 0.05], [0.42, 0.12]], dtype=np.float32),
            np.asarray([[-0.18, -0.54], [-0.07, -0.18], [0.02, 0.08], [0.10, 0.34]], dtype=np.float32),
            np.asarray([[-0.34, 0.24], [-0.12, 0.10], [0.08, -0.02], [0.28, -0.14]], dtype=np.float32),
        ]
        crack_scale = float(radius) * 0.82
        for pts in crack_templates:
            pts_xy = np.asarray(center_xy, dtype=np.float32)[None, :] + crack_scale * pts
            pts_xyz = np.zeros((len(pts_xy), 3), dtype=np.float32)
            pts_xyz[:, :2] = pts_xy
            for i in range(len(pts_xy)):
                pts_xyz[i, 2] = self._surface_z_from_grid_xy(surf_np, float(pts_xy[i, 0]), float(pts_xy[i, 1])) + 0.014
            for i in range(len(pts_xyz) - 1):
                self._add_visual_cylinder_segment(
                    pts_xyz[i],
                    pts_xyz[i + 1],
                    radius=0.0033,
                    rgba=(0.16, 0.06, 0.05, 0.98),
                    specular=(0.02, 0.02, 0.02),
                )

        # Add an explicit outer keep-out boundary at the exact planner radius.
        ring_n = 28
        theta = np.linspace(0.0, 2.0 * math.pi, num=ring_n + 1, dtype=np.float32)
        ring_xy = np.stack(
            [
                center_xy[0] + float(radius) * np.cos(theta),
                center_xy[1] + float(radius) * np.sin(theta),
            ],
            axis=1,
        ).astype(np.float32)
        ring_xyz = np.zeros((len(ring_xy), 3), dtype=np.float32)
        ring_xyz[:, :2] = ring_xy
        for i in range(len(ring_xy)):
            ring_xyz[i, 2] = self._surface_z_from_grid_xy(surf_np, float(ring_xy[i, 0]), float(ring_xy[i, 1])) + 0.013
        for i in range(len(ring_xyz) - 1):
            self._add_visual_cylinder_segment(
                ring_xyz[i],
                ring_xyz[i + 1],
                radius=0.0017,
                rgba=(0.86, 0.34, 0.08, 0.96),
                specular=(0.05, 0.03, 0.02),
            )

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        p = self._p
        qv = np.asarray(q, dtype=np.float32).reshape(-1)
        q_full = np.zeros((len(self.ik_joint_indices),), dtype=np.float32)
        for arm_i, full_i in enumerate(self.arm_ik_positions):
            q_full[full_i] = qv[arm_i]
        zeros = [0.0] * len(q_full)
        jac_t, jac_r = p.calculateJacobian(
            self.robot_id,
            self.ee_link_index,
            [0.0, 0.0, 0.0],
            [float(v) for v in q_full],
            zeros,
            zeros,
            physicsClientId=self.client_id,
        )
        jt = np.asarray(jac_t, dtype=np.float32)[:, self.arm_ik_positions]
        jr = np.asarray(jac_r, dtype=np.float32)[:, self.arm_ik_positions]
        return np.concatenate([jt, jr], axis=0).astype(np.float32)

    def _target_from_pose_row(self, pose_row: np.ndarray, ik_cfg: IKConfig) -> tuple[np.ndarray, np.ndarray]:
        row = np.asarray(pose_row, dtype=np.float32).reshape(-1)
        target_pos = row[:3].astype(np.float32)
        target_quat = _rpy_to_quat_xyzw(row[3:6].astype(np.float32))
        off_rpy = np.asarray(getattr(ik_cfg, "tool_frame_offset_rpy", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(-1)
        if len(off_rpy) >= 3 and float(np.linalg.norm(off_rpy[:3])) > 1e-8:
            q_off = _rpy_to_quat_xyzw(off_rpy[:3].astype(np.float32))
            target_quat = _quat_multiply(target_quat, q_off).astype(np.float32)
            target_quat /= max(float(np.linalg.norm(target_quat)), 1e-8)
        return target_pos, target_quat

    def _evaluate_seed_score(
        self,
        q: np.ndarray,
        *,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        ori_weight: float,
    ) -> tuple[float, float, float]:
        qv = np.clip(np.asarray(q, dtype=np.float32).reshape(-1), self.q_lo, self.q_hi)
        self.reset_joint_state(qv)
        cur_pos, cur_quat = self.get_ee_pose()
        pos_err = float(np.linalg.norm(target_pos - cur_pos))
        ori_err = float(np.linalg.norm(_quat_orientation_error(target_quat, cur_quat)))
        score = pos_err + float(max(0.0, ori_weight)) * ori_err
        return score, pos_err, ori_err

    def _bootstrap_first_seed(self, *, target_pos: np.ndarray, target_quat: np.ndarray, ik_cfg: IKConfig) -> np.ndarray:
        p = self._p
        rng = np.random.default_rng(int(max(0, ik_cfg.seed_search_seed)))
        candidates: list[np.ndarray] = [
            self.home_q.copy(),
            np.clip(self.home_q + np.asarray([0.0, -0.12, 0.10, -0.10, 0.0, 0.0], dtype=np.float32), self.q_lo, self.q_hi),
            np.clip(self.home_q + np.asarray([0.0, 0.10, -0.08, 0.12, 0.0, 0.0], dtype=np.float32), self.q_lo, self.q_hi),
            np.asarray([0.0, -1.57, 1.57, -1.75, -1.57, 0.0], dtype=np.float32),
        ]
        try:
            sol_full = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                targetPosition=[float(v) for v in target_pos],
                lowerLimits=[float(v) for v in self.q_lo],
                upperLimits=[float(v) for v in self.q_hi],
                jointRanges=[float(v) for v in self.arm_joint_ranges],
                restPoses=[float(v) for v in self.home_q],
                solver=p.IK_DLS,
                maxNumIterations=80,
                residualThreshold=float(max(1e-6, ik_cfg.pos_tol * 0.5)),
                physicsClientId=self.client_id,
            )
            sol_full = np.asarray(sol_full, dtype=np.float32).reshape(-1)
            if len(sol_full) >= len(self.ik_joint_indices):
                q_pos = np.asarray([sol_full[idx] for idx in self.arm_ik_positions], dtype=np.float32)
                candidates.append(np.clip(q_pos, self.q_lo, self.q_hi).astype(np.float32))
        except Exception:
            pass

        n_rand = int(max(0, ik_cfg.seed_search_samples))
        if n_rand > 0:
            std = float(max(1e-3, ik_cfg.seed_search_home_std))
            scales = np.asarray([0.55, 0.75, 0.75, 0.85, 0.65, 0.85], dtype=np.float32) * std
            rand_q = rng.normal(loc=self.home_q[None, :], scale=scales[None, :], size=(n_rand, 6)).astype(np.float32)
            rand_q = np.clip(rand_q, self.q_lo[None, :], self.q_hi[None, :])
            candidates.extend([q.astype(np.float32) for q in rand_q])

        best_q = np.clip(candidates[0], self.q_lo, self.q_hi).astype(np.float32)
        best_score = float("inf")
        ori_weight = float(max(0.0, ik_cfg.seed_search_ori_weight))
        for q in candidates:
            score, _, _ = self._evaluate_seed_score(
                q,
                target_pos=target_pos,
                target_quat=target_quat,
                ori_weight=ori_weight,
            )
            if score < best_score:
                best_score = score
                best_q = np.clip(np.asarray(q, dtype=np.float32), self.q_lo, self.q_hi).astype(np.float32)

        jitter_n = min(96, max(0, n_rand // 2))
        if jitter_n > 0:
            scales = np.asarray([0.35, 0.45, 0.45, 0.55, 0.55, 0.55], dtype=np.float32)
            jitter = rng.normal(loc=0.0, scale=scales[None, :], size=(jitter_n, 6)).astype(np.float32)
            for q in np.clip(best_q[None, :] + jitter, self.q_lo[None, :], self.q_hi[None, :]):
                score, _, _ = self._evaluate_seed_score(
                    q,
                    target_pos=target_pos,
                    target_quat=target_quat,
                    ori_weight=ori_weight,
                )
                if score < best_score:
                    best_score = score
                    best_q = np.asarray(q, dtype=np.float32)
        return best_q.astype(np.float32)

    def solve_pose_path_ik(
        self,
        pose_path: np.ndarray,
        cfg: IKConfig | None = None,
        q_seed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ik_cfg = cfg or IKConfig()
        method = str(getattr(ik_cfg, "method", "pybullet")).strip().lower()
        if method == "pybullet":
            return self._solve_pose_path_ik_pybullet(pose_path, ik_cfg=ik_cfg, q_seed=q_seed)
        if method == "dls":
            return self._solve_pose_path_ik_dls(pose_path, ik_cfg=ik_cfg, q_seed=q_seed)
        raise ValueError(f"unknown IK method '{method}', expected 'pybullet' or 'dls'")

    def _summarize_ik_solution(
        self,
        pose_path: np.ndarray,
        q_out: np.ndarray,
        *,
        ik_cfg: IKConfig,
        method: str,
        iters_used: np.ndarray | None = None,
    ) -> dict[str, Any]:
        poses = np.asarray(pose_path, dtype=np.float32)
        q_path = np.asarray(q_out, dtype=np.float32)
        pos_errs = np.zeros((len(q_path),), dtype=np.float32)
        ori_errs = np.zeros((len(q_path),), dtype=np.float32)
        for i in range(len(q_path)):
            target_pos, target_quat = self._target_from_pose_row(poses[i], ik_cfg)
            self.reset_joint_state(q_path[i])
            cur_pos, cur_quat = self.get_ee_pose()
            pos_errs[i] = float(np.linalg.norm(target_pos - cur_pos))
            ori_errs[i] = float(np.linalg.norm(_quat_orientation_error(target_quat, cur_quat)))
        if iters_used is None:
            iters = np.ones((len(q_path),), dtype=np.int32)
        else:
            iters = np.asarray(iters_used, dtype=np.int32)
        return {
            "method": method,
            "mean_pos_err": float(np.mean(pos_errs)),
            "max_pos_err": float(np.max(pos_errs)),
            "mean_ori_err_rad": float(np.mean(ori_errs)),
            "max_ori_err_rad": float(np.max(ori_errs)),
            "mean_iters": float(np.mean(iters)),
            "first_pos_err": float(pos_errs[0]) if len(pos_errs) > 0 else 0.0,
            "first_ori_err_rad": float(ori_errs[0]) if len(ori_errs) > 0 else 0.0,
            "pos_errs": pos_errs.astype(np.float32),
            "ori_errs_rad": ori_errs.astype(np.float32),
            "iters_used": iters.astype(np.int32),
        }

    def _repair_pybullet_ik_discontinuity(
        self,
        pose_path: np.ndarray,
        q_out: np.ndarray,
        *,
        ik_cfg: IKConfig,
    ) -> tuple[np.ndarray, bool]:
        q_path = np.asarray(q_out, dtype=np.float32).copy()
        if len(q_path) < 4 or not bool(ik_cfg.continuity_repair):
            return q_path.astype(np.float32), False
        step_norms = np.linalg.norm(np.diff(q_path, axis=0), axis=1)
        if len(step_norms) < 3:
            return q_path.astype(np.float32), False
        baseline = float(np.median(step_norms))
        jump_thresh = max(float(ik_cfg.continuity_jump_abs), float(ik_cfg.continuity_jump_factor) * max(baseline, 1e-6))
        bad_idx = np.where(step_norms > jump_thresh)[0]
        if len(bad_idx) == 0:
            return q_path.astype(np.float32), False
        start_idx = int(max(1, bad_idx[0]))
        q_seed = q_path[start_idx - 1].copy()
        dls_cfg = IKConfig(
            method="dls",
            max_iters=max(int(ik_cfg.max_iters), 96),
            damping=float(ik_cfg.damping),
            step_size=float(ik_cfg.step_size),
            max_delta_norm=float(ik_cfg.max_delta_norm),
            pos_tol=float(ik_cfg.pos_tol),
            ori_tol_rad=float(ik_cfg.ori_tol_rad),
            warm_start_weight=float(ik_cfg.warm_start_weight),
            rest_pose_weight=float(ik_cfg.rest_pose_weight),
            search_first_seed=False,
            tool_frame_offset_rpy=tuple(float(v) for v in ik_cfg.tool_frame_offset_rpy),
            continuity_repair=False,
        )
        repaired_tail, _ = self._solve_pose_path_ik_dls(pose_path[start_idx:], ik_cfg=dls_cfg, q_seed=q_seed)
        q_path[start_idx:] = repaired_tail.astype(np.float32)
        return q_path.astype(np.float32), True

    def _solve_pose_path_ik_pybullet(
        self,
        pose_path: np.ndarray,
        *,
        ik_cfg: IKConfig,
        q_seed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        p = self._p
        poses = np.asarray(pose_path, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] < 6:
            raise ValueError("pose_path must have shape (N, 6)")
        if q_seed is not None:
            q_prev = np.clip(np.asarray(q_seed, dtype=np.float32).reshape(-1), self.q_lo, self.q_hi)
        elif len(poses) > 0 and bool(ik_cfg.search_first_seed):
            seed_pos, seed_quat = self._target_from_pose_row(poses[0], ik_cfg)
            q_prev = self._bootstrap_first_seed(target_pos=seed_pos, target_quat=seed_quat, ik_cfg=ik_cfg)
        else:
            q_prev = self.home_q.copy()
        q_out = np.zeros((poses.shape[0], 6), dtype=np.float32)
        iters_used = np.full((poses.shape[0],), int(max(1, ik_cfg.max_iters)), dtype=np.int32)
        joint_damping = [float(max(1e-6, ik_cfg.damping))] * len(self.ik_joint_indices)
        for i in range(poses.shape[0]):
            target_pos, target_quat = self._target_from_pose_row(poses[i], ik_cfg)
            home_blend = float(np.clip(ik_cfg.pybullet_rest_home_blend, 0.0, 1.0))
            rest_q = ((1.0 - home_blend) * q_prev + home_blend * self.home_q).astype(np.float32)
            rest_q = np.clip(rest_q, self.q_lo, self.q_hi).astype(np.float32)
            sol_full = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                targetPosition=[float(v) for v in target_pos],
                targetOrientation=[float(v) for v in target_quat],
                lowerLimits=[float(v) for v in self.q_lo],
                upperLimits=[float(v) for v in self.q_hi],
                jointRanges=[float(v) for v in self.arm_joint_ranges],
                restPoses=[float(v) for v in rest_q],
                jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=int(max(1, ik_cfg.max_iters)),
                residualThreshold=float(max(1e-6, ik_cfg.pos_tol * 0.5)),
                physicsClientId=self.client_id,
            )
            sol_full = np.asarray(sol_full, dtype=np.float32).reshape(-1)
            if len(sol_full) < len(self.ik_joint_indices):
                raise RuntimeError(
                    f"pybullet IK returned {len(sol_full)} joints, expected at least {len(self.ik_joint_indices)}"
                )
            q = np.asarray([sol_full[idx] for idx in self.arm_ik_positions], dtype=np.float32)
            q = np.clip(q, self.q_lo, self.q_hi).astype(np.float32)
            q_out[i] = q
            q_prev = q.copy()
        repaired_q, repaired = self._repair_pybullet_ik_discontinuity(poses, q_out, ik_cfg=ik_cfg)
        q_out = repaired_q.astype(np.float32)
        method_name = "pybullet+dls_repair" if repaired else "pybullet"
        summary = self._summarize_ik_solution(poses, q_out, ik_cfg=ik_cfg, method=method_name, iters_used=iters_used)
        return q_out.astype(np.float32), summary

    def _solve_pose_path_ik_dls(
        self,
        pose_path: np.ndarray,
        *,
        ik_cfg: IKConfig,
        q_seed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        poses = np.asarray(pose_path, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] < 6:
            raise ValueError("pose_path must have shape (N, 6)")
        if q_seed is not None:
            q_prev = np.clip(np.asarray(q_seed, dtype=np.float32).reshape(-1), self.q_lo, self.q_hi)
        elif len(poses) > 0 and bool(ik_cfg.search_first_seed):
            seed_pos, seed_quat = self._target_from_pose_row(poses[0], ik_cfg)
            q_prev = self._bootstrap_first_seed(target_pos=seed_pos, target_quat=seed_quat, ik_cfg=ik_cfg)
        else:
            q_prev = self.home_q.copy()
        q_out = np.zeros((poses.shape[0], 6), dtype=np.float32)
        iters_used = np.zeros((poses.shape[0],), dtype=np.int32)
        for i in range(poses.shape[0]):
            q = q_prev.copy()
            target_pos, target_quat = self._target_from_pose_row(poses[i], ik_cfg)
            ok = False
            for it in range(int(max(1, ik_cfg.max_iters))):
                self.reset_joint_state(q)
                cur_pos, cur_quat = self.get_ee_pose()
                e_pos = (target_pos - cur_pos).astype(np.float32)
                e_ori = _quat_orientation_error(target_quat, cur_quat).astype(np.float32)
                pos_n = float(np.linalg.norm(e_pos))
                ori_n = float(np.linalg.norm(e_ori))
                if pos_n <= float(ik_cfg.pos_tol) and ori_n <= float(ik_cfg.ori_tol_rad):
                    ok = True
                    iters_used[i] = it + 1
                    break
                err = np.concatenate([e_pos, e_ori], axis=0).astype(np.float32)
                J = self._jacobian(q)
                lam2 = float(ik_cfg.damping) ** 2
                A = J @ J.T + lam2 * np.eye(6, dtype=np.float32)
                dq = J.T @ np.linalg.solve(A, err)
                dq = dq.astype(np.float32)
                if float(ik_cfg.warm_start_weight) > 0.0:
                    dq += float(ik_cfg.warm_start_weight) * (q_prev - q)
                if float(ik_cfg.rest_pose_weight) > 0.0:
                    dq += float(ik_cfg.rest_pose_weight) * (self.home_q - q)
                dq *= float(ik_cfg.step_size)
                dq_norm = float(np.linalg.norm(dq))
                if dq_norm > float(ik_cfg.max_delta_norm):
                    dq = dq * (float(ik_cfg.max_delta_norm) / max(dq_norm, 1e-8))
                q = np.clip(q + dq, self.q_lo, self.q_hi).astype(np.float32)
            if not ok:
                iters_used[i] = int(max(1, ik_cfg.max_iters))
            q_out[i] = q
            q_prev = q.copy()
        summary = self._summarize_ik_solution(poses, q_out, ik_cfg=ik_cfg, method="dls", iters_used=iters_used)
        return q_out.astype(np.float32), summary

    def time_parameterize_joint_path(
        self,
        joint_path: np.ndarray,
        cfg: JointTrackConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        track_cfg = cfg or JointTrackConfig()
        q = np.asarray(joint_path, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 6:
            raise ValueError("joint_path must have shape (N, 6)")
        if len(q) <= 1:
            t = np.zeros((len(q),), dtype=np.float32)
            qd = np.zeros_like(q)
            return t, q.astype(np.float32), qd.astype(np.float32)
        seg_dt: list[float] = [0.0]
        vmax = float(max(track_cfg.max_joint_speed, 1e-3))
        min_dt = float(max(track_cfg.min_segment_time, 1e-3))
        time_scale = float(max(track_cfg.trajectory_time_scale, 1e-3))
        for i in range(1, len(q)):
            dq = np.abs(q[i] - q[i - 1])
            dt = max(float(np.max(dq) / vmax), min_dt) * time_scale
            seg_dt.append(dt)
        t_src = np.cumsum(np.asarray(seg_dt, dtype=np.float32))
        total_t = float(t_src[-1])
        n_out = int(max(2, math.ceil(total_t / float(track_cfg.sim_dt)))) + 1
        t_out = np.linspace(0.0, total_t, num=n_out, dtype=np.float32)
        q_out = np.zeros((len(t_out), q.shape[1]), dtype=np.float32)
        for j in range(q.shape[1]):
            q_out[:, j] = np.interp(t_out, t_src, q[:, j]).astype(np.float32)
        ramp_time = float(max(track_cfg.endpoint_ramp_time, 0.0))
        if total_t > 1e-8 and ramp_time > 1e-8:
            ramp_time = float(min(ramp_time, 0.45 * total_t))
            if ramp_time > 2.0 * float(track_cfg.sim_dt):
                q_linear = q_out.copy()
                start_mask = t_out < ramp_time
                if np.any(start_mask):
                    u = np.clip(t_out[start_mask] / ramp_time, 0.0, 1.0).astype(np.float32)
                    s = (u * u * (3.0 - 2.0 * u)).reshape(-1, 1).astype(np.float32)
                    q_out[start_mask] = q[0:1] + s * (q_linear[start_mask] - q[0:1])
                end_mask = t_out > (total_t - ramp_time)
                if np.any(end_mask):
                    u = np.clip((total_t - t_out[end_mask]) / ramp_time, 0.0, 1.0).astype(np.float32)
                    s = (u * u * (3.0 - 2.0 * u)).reshape(-1, 1).astype(np.float32)
                    q_out[end_mask] = q[-1:] + s * (q_linear[end_mask] - q[-1:])
        smooth_passes = int(max(0, track_cfg.reference_smooth_passes))
        if smooth_passes > 0 and len(q_out) >= 4:
            q_start = q_out[0].copy()
            q_goal = q_out[-1].copy()
            for _ in range(smooth_passes):
                q_prev = q_out.copy()
                q_out[1:-1] = (
                    0.25 * q_prev[:-2]
                    + 0.50 * q_prev[1:-1]
                    + 0.25 * q_prev[2:]
                ).astype(np.float32)
                q_out[0] = q_start
                q_out[-1] = q_goal
        hold_time = float(max(track_cfg.terminal_hold_time, 0.0))
        if hold_time > 0.0:
            dt = float(track_cfg.sim_dt)
            n_hold = int(max(1, round(hold_time / max(dt, 1e-8))))
            t_hold = t_out[-1] + dt * np.arange(1, n_hold + 1, dtype=np.float32)
            q_hold = np.repeat(q_out[-1:, :], n_hold, axis=0).astype(np.float32)
            t_out = np.concatenate([t_out, t_hold], axis=0).astype(np.float32)
            q_out = np.concatenate([q_out, q_hold], axis=0).astype(np.float32)
        qd_out = np.zeros_like(q_out)
        if len(q_out) >= 2:
            dt = float(track_cfg.sim_dt)
            qd_out[1:-1] = (q_out[2:] - q_out[:-2]) / max(2.0 * dt, 1e-8)
            qd_out[0] = (q_out[1] - q_out[0]) / max(dt, 1e-8)
            qd_out[-1] = (q_out[-1] - q_out[-2]) / max(dt, 1e-8)
            qd_lim = float(max(track_cfg.max_joint_speed, 1e-3))
            qd_out = np.clip(qd_out, -qd_lim, qd_lim).astype(np.float32)
        if hold_time > 0.0:
            qd_out[-n_hold:] = 0.0
        return t_out.astype(np.float32), q_out.astype(np.float32), qd_out.astype(np.float32)

    def track_joint_trajectory(
        self,
        q_ref: np.ndarray,
        qd_ref: np.ndarray | None = None,
        cfg: JointTrackConfig | None = None,
        video_path: str | None = None,
        ee_ref_pos: np.ndarray | None = None,
        surface_xyz_grid: np.ndarray | None = None,
        obstacle_center_xy: np.ndarray | None = None,
        obstacle_radius: float | None = None,
        waypoint_xyz: np.ndarray | None = None,
        reuse_static_scene: bool = False,
    ) -> dict[str, Any]:
        track_cfg = cfg or JointTrackConfig()
        q_cmd = np.asarray(q_ref, dtype=np.float32)
        if q_cmd.ndim != 2 or q_cmd.shape[1] != 6:
            raise ValueError("q_ref must have shape (N, 6)")
        if qd_ref is None:
            qd_cmd = np.zeros_like(q_cmd, dtype=np.float32)
        else:
            qd_cmd = np.asarray(qd_ref, dtype=np.float32)
            if qd_cmd.shape != q_cmd.shape:
                raise ValueError("qd_ref must match q_ref shape")
        if ee_ref_pos is not None:
            ee_ref = np.asarray(ee_ref_pos, dtype=np.float32)
            if ee_ref.shape != (len(q_cmd), 3):
                raise ValueError("ee_ref_pos must have shape (N, 3) matching q_ref")
        else:
            ee_ref = None
        if surface_xyz_grid is not None:
            surf = np.asarray(surface_xyz_grid, dtype=np.float32)
            if surf.ndim != 3 or surf.shape[2] != 3:
                raise ValueError("surface_xyz_grid must have shape (H, W, 3)")
        else:
            surf = None
        if obstacle_center_xy is not None:
            obs_center = np.asarray(obstacle_center_xy, dtype=np.float32).reshape(2)
            obs_radius = float(obstacle_radius if obstacle_radius is not None else 0.0)
        else:
            obs_center = None
            obs_radius = None
        if waypoint_xyz is not None:
            wp_xyz = np.asarray(waypoint_xyz, dtype=np.float32)
            if wp_xyz.ndim != 2 or wp_xyz.shape[1] != 3:
                raise ValueError("waypoint_xyz must have shape (K, 3)")
        else:
            wp_xyz = None

        p = self._p
        writer = None
        video_out_path = os.path.abspath(video_path) if video_path else None
        gui_video_log_id: int | None = None
        gui_video_raw_path: str | None = None
        capture_every = max(1, int(round(1.0 / max(float(track_cfg.sim_dt) * float(track_cfg.video_fps), 1e-8))))
        if video_path:
            os.makedirs(os.path.dirname(video_out_path), exist_ok=True)
            if self._gui:
                try:
                    raw_path = video_out_path
                    if abs(float(track_cfg.video_slowdown) - 1.0) > 1e-6:
                        root, ext = os.path.splitext(video_out_path)
                        raw_path = root + ".raw_gui" + (ext or ".mp4")
                    gui_video_log_id = int(
                        p.startStateLogging(
                            p.STATE_LOGGING_VIDEO_MP4,
                            raw_path,
                            physicsClientId=self.client_id,
                        )
                    )
                    gui_video_raw_path = os.path.abspath(raw_path)
                except Exception:
                    gui_video_log_id = None
                    gui_video_raw_path = None
                    video_out_path = None
            else:
                try:
                    out_fps = float(track_cfg.video_fps) / max(float(track_cfg.video_slowdown), 1e-3)
                    writer = _FFmpegVideoWriter(
                        out_path=video_out_path,
                        width=int(track_cfg.video_width),
                        height=int(track_cfg.video_height),
                        fps=out_fps,
                    )
                except Exception:
                    writer = None
                    video_out_path = None

        pause_key = 32  # ASCII for space.
        paused = False

        def _handle_pause_toggle() -> None:
            nonlocal paused
            if (not self._gui) or (not bool(track_cfg.enable_keyboard_pause)):
                return
            try:
                events = p.getKeyboardEvents()
            except Exception:
                return
            ev = int(events.get(pause_key, 0))
            if ev & int(getattr(p, "KEY_WAS_TRIGGERED", 0x1)):
                paused = not paused

        def _pause_if_requested() -> None:
            nonlocal paused
            while paused:
                _handle_pause_toggle()
                if not paused:
                    break
                time.sleep(float(max(1e-3, track_cfg.pause_poll_dt)))

        self.reset_joint_state(q_cmd[0])
        if not bool(track_cfg.preserve_trace_history):
            self._clear_trace_visuals()
        if not bool(reuse_static_scene):
            self._clear_surface_visuals()
        if bool(track_cfg.draw_surface_wireframe) and surf is not None and (not bool(reuse_static_scene) or len(self._surface_body_ids) == 0):
            self._build_surface_visuals(
                surf,
                stride=int(max(1, track_cfg.surface_line_stride)),
                line_width=float(max(0.5, track_cfg.surface_line_width)),
            )
        if not bool(reuse_static_scene):
            self._clear_obstacle_visual()
        if (
            obs_center is not None
            and obs_radius is not None
            and obs_radius > 0.0
            and (not bool(reuse_static_scene) or len(self._obstacle_body_ids) == 0)
        ):
            if surf is not None:
                z_min = float(np.nanmin(surf[..., 2]) - 0.12)
                z_max = float(np.nanmax(surf[..., 2]) + 0.12)
            elif ee_ref is not None:
                z_min = float(np.min(ee_ref[:, 2]) - 0.12)
                z_max = float(np.max(ee_ref[:, 2]) + 0.12)
            else:
                z_min, z_max = 0.0, 1.0
            self._build_obstacle_visual(
                center_xy=obs_center,
                radius=float(obs_radius),
                z_min=z_min,
                z_max=z_max,
                surf=surf,
            )
        if wp_xyz is not None and (not bool(reuse_static_scene) or len(self._marker_body_ids) == 0):
            self.build_waypoint_markers(
                wp_xyz,
                radius=float(max(0.001, track_cfg.waypoint_marker_radius)),
            )
        pos_gains = _joint_array(track_cfg.position_gain, len(self.arm_joint_indices))
        vel_gains = _joint_array(track_cfg.velocity_gain, len(self.arm_joint_indices))
        forces = _joint_array(track_cfg.max_force, len(self.arm_joint_indices))
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[float(v) for v in q_cmd[0]],
            targetVelocities=[float(v) for v in qd_cmd[0]],
            positionGains=pos_gains,
            velocityGains=vel_gains,
            forces=forces,
            physicsClientId=self.client_id,
        )
        for _ in range(int(max(0, track_cfg.settle_steps))):
            p.stepSimulation(physicsClientId=self.client_id)
        if writer is not None:
            writer.append_data(
                self.capture_frame(width=int(track_cfg.video_width), height=int(track_cfg.video_height))
            )

        n_ref = len(q_cmd)
        n = n_ref
        q_meas = np.zeros((n, q_cmd.shape[1]), dtype=np.float32)
        qd_meas = np.zeros((n, q_cmd.shape[1]), dtype=np.float32)
        ee_pos = np.zeros((n, 3), dtype=np.float32)
        ee_quat = np.zeros((n, 4), dtype=np.float32)
        q_err_norm = np.zeros((n,), dtype=np.float32)
        q_ref_log = np.zeros((n, q_cmd.shape[1]), dtype=np.float32)
        qd_ref_log = np.zeros((n, q_cmd.shape[1]), dtype=np.float32)

        trace_stride = int(max(1, track_cfg.trace_stride))
        trace_width = float(max(0.5, track_cfg.trace_width))
        if bool(track_cfg.preserve_trace_history):
            exec_palette = [
                (0.10, 0.70, 0.25, 0.92),
                (0.17, 0.48, 0.91, 0.92),
                (0.88, 0.45, 0.05, 0.92),
                (0.58, 0.23, 0.88, 0.92),
                (0.84, 0.18, 0.40, 0.92),
                (0.00, 0.65, 0.65, 0.92),
            ]
            ref_palette = [
                (0.95, 0.35, 0.10, 0.78),
                (0.25, 0.62, 0.98, 0.78),
                (0.98, 0.60, 0.16, 0.78),
                (0.72, 0.44, 0.98, 0.78),
                (0.96, 0.44, 0.60, 0.78),
                (0.25, 0.80, 0.80, 0.78),
            ]
            palette_idx = int(self._trace_history_segment_idx % len(exec_palette))
            exec_trace_rgba = exec_palette[palette_idx]
            ref_trace_rgba = ref_palette[palette_idx % len(ref_palette)]
        else:
            exec_trace_rgba = (0.10, 0.70, 0.25, 0.92)
            ref_trace_rgba = (0.95, 0.35, 0.10, 0.78)
        trace_radius = float(max(0.0012, 0.0008 * trace_width))
        # Visualization only; this offsets rendered trace geometry slightly above the scene
        # and does not affect execution data, error metrics, or saved state logs.
        trace_lift = np.asarray([0.0, 0.0, max(0.008, 3.0 * trace_radius)], dtype=np.float32)
        prev_exec_pos: np.ndarray | None = None
        prev_ref_pos: np.ndarray | None = None

        t0 = time.time()
        for i_ref in range(n_ref):
            _handle_pause_toggle()
            _pause_if_requested()
            p.setJointMotorControlArray(
                self.robot_id,
                self.arm_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[float(v) for v in q_cmd[i_ref]],
                targetVelocities=[float(v) for v in qd_cmd[i_ref]],
                positionGains=pos_gains,
                velocityGains=vel_gains,
                forces=forces,
                physicsClientId=self.client_id,
            )
            p.stepSimulation(physicsClientId=self.client_id)
            if bool(track_cfg.realtime):
                time.sleep(float(track_cfg.sim_dt))
            q_i, qd_i = self.get_joint_state()
            pos_i, quat_i = self.get_ee_pose()
            q_meas[i_ref] = q_i
            qd_meas[i_ref] = qd_i
            ee_pos[i_ref] = pos_i
            ee_quat[i_ref] = quat_i
            q_ref_log[i_ref] = q_cmd[i_ref]
            qd_ref_log[i_ref] = qd_cmd[i_ref]
            q_err_norm[i_ref] = float(np.linalg.norm(q_cmd[i_ref] - q_i))
            if i_ref % trace_stride == 0:
                if bool(track_cfg.draw_ee_trace):
                    if prev_exec_pos is not None:
                        pos_vis = pos_i + trace_lift
                        prev_exec_vis = prev_exec_pos + trace_lift
                        self._add_visual_cylinder_segment(
                            prev_exec_vis,
                            pos_vis,
                            radius=trace_radius,
                            rgba=exec_trace_rgba,
                            specular=(0.03, 0.05, 0.03),
                            body_list=self._trace_body_ids,
                        )
                    prev_exec_pos = pos_i.copy()
                if bool(track_cfg.draw_ref_trace) and ee_ref is not None:
                    ref_pos = ee_ref[i_ref].astype(np.float32)
                    if prev_ref_pos is not None:
                        ref_vis = ref_pos + trace_lift
                        prev_ref_vis = prev_ref_pos + trace_lift
                        self._add_visual_cylinder_segment(
                            prev_ref_vis,
                            ref_vis,
                            radius=max(0.0010, trace_radius * 0.82),
                            rgba=ref_trace_rgba,
                            specular=(0.08, 0.03, 0.02),
                            body_list=self._trace_body_ids,
                        )
                    prev_ref_pos = ref_pos.copy()
            if writer is not None and ((i_ref + 1) % capture_every == 0 or i_ref == n_ref - 1):
                writer.append_data(
                    self.capture_frame(width=int(track_cfg.video_width), height=int(track_cfg.video_height))
                )

        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if gui_video_log_id is not None:
            try:
                p.stopStateLogging(int(gui_video_log_id), physicsClientId=self.client_id)
            except Exception:
                pass
            if gui_video_raw_path is not None and video_out_path is not None:
                try:
                    video_out_path = _retime_video_ffmpeg(
                        src_path=gui_video_raw_path,
                        dst_path=video_out_path,
                        slowdown=float(track_cfg.video_slowdown),
                    )
                except Exception:
                    video_out_path = gui_video_raw_path if os.path.exists(gui_video_raw_path) else video_out_path
        if not bool(reuse_static_scene) and not bool(track_cfg.keep_visuals_on_finish):
            self._clear_surface_visuals()
            self._clear_obstacle_visual()
            self._clear_marker_visuals()
        if not bool(track_cfg.keep_visuals_on_finish) and not bool(track_cfg.preserve_trace_history):
            self._clear_trace_visuals()
        elif bool(track_cfg.preserve_trace_history):
            self._trace_history_segment_idx += 1

        return {
            "sim_dt": float(track_cfg.sim_dt),
            "trajectory_time_scale": float(track_cfg.trajectory_time_scale),
            "wall_seconds": float(time.time() - t0),
            "q_ref": q_ref_log.astype(np.float32),
            "qd_ref": qd_ref_log.astype(np.float32),
            "q_meas": q_meas.astype(np.float32),
            "qd_meas": qd_meas.astype(np.float32),
            "ee_pos": ee_pos.astype(np.float32),
            "ee_quat_xyzw": ee_quat.astype(np.float32),
            "joint_err_norm": q_err_norm.astype(np.float32),
            "mean_joint_err_norm": float(np.mean(q_err_norm)),
            "max_joint_err_norm": float(np.max(q_err_norm)),
            "video_path": video_out_path,
        }
