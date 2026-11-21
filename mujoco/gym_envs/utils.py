"""Kinematics, math, and controller utilities for WISERLI-ATAL Gymnasium environments."""

from __future__ import annotations
import sys
import os
import numpy as np
import mujoco


def resolve_mujoco_path(path: str) -> str:
    """Ensures paths are compatible with MuJoCo C parser (e.g. converting UNC paths on Windows)."""
    abs_path = os.path.abspath(path)
    if sys.platform == "win32" and abs_path.startswith(r"\\"):
        unc_prefixes = [
            r"\\B3D\Project H1\WiserCobot\WISERLI-ATAL-C1.1",
            r"//B3D/Project H1/WiserCobot/WISERLI-ATAL-C1.1",
        ]
        for prefix in unc_prefixes:
            if abs_path.replace("/", "\\").lower().startswith(prefix.replace("/", "\\").lower()):
                rel = abs_path.replace("/", "\\")[len(prefix):].lstrip("\\")
                candidate = os.path.join("W:\\", rel)
                if os.path.exists(candidate):
                    return candidate
    return abs_path


def enable_robot_gravcomp(model: mujoco.MjModel) -> None:
    """Enables gravity compensation on all robot bodies without affecting free objects."""
    excluded = {"world", "grasp_cube", "target", "grasp_object", "drop_goal"}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name not in excluded:
            model.body_gravcomp[body_id] = 1.0


def diffik_solve(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    delta_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    dof_ids: np.ndarray | None = None,
    damping: float = 1e-4,
    integration_dt: float = 0.1,
    max_angvel: float = 2.0,
) -> np.ndarray:
    """Computes joint velocity / displacement using Damped Least Squares IK.
    
    If target_quat is None, performs agile 3-DOF position tracking across the full reachable workspace.
    """
    if dof_ids is None:
        dof_ids = np.arange(6)

    if target_quat is not None:
        # 6-DOF task space (position + orientation)
        twist = np.zeros(6)
        twist[:3] = delta_pos / integration_dt

        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
        mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        twist[3:] *= 0.5 / integration_dt  # compliant orientation gain

        jac = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:, dof_ids]

        diag = damping * np.eye(6)
        dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, twist)

    else:
        # Agile 3-DOF position DLS tracking
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, dof_ids]  # 3 x 6

        A = J @ J.T + damping * np.eye(3)
        v = delta_pos / integration_dt
        dq = J.T @ np.linalg.solve(A, v)

    if max_angvel > 0:
        dq_max = np.abs(dq).max()
        if dq_max > max_angvel:
            dq *= max_angvel / dq_max

    return dq * integration_dt


def clip_joint_limits(
    model: mujoco.MjModel,
    q_target: np.ndarray,
    joint_indices: list[int] | None = None,
) -> np.ndarray:
    """Clips joint targets strictly inside joint ranges defined in the model."""
    if joint_indices is None:
        joint_indices = list(range(6))
    q_clipped = q_target.copy()
    for idx, j_id in enumerate(joint_indices):
        if model.jnt_limited[j_id]:
            lo, hi = model.jnt_range[j_id]
            q_clipped[idx] = np.clip(q_clipped[idx], lo, hi)
    return q_clipped


def action_to_gripper_ctrl(action_val: float) -> float:
    """Maps normalized action in [-1.0, 1.0] to Robotiq 2F-85 control [0, 255]."""
    norm = np.clip(action_val, -1.0, 1.0)
    return float((norm + 1.0) * 0.5 * 255.0)


def gripper_ctrl_to_action(ctrl_val: float) -> float:
    """Maps Robotiq 2F-85 control [0, 255] to normalized [-1.0, 1.0]."""
    return float(np.clip((ctrl_val / 127.5) - 1.0, -1.0, 1.0))
