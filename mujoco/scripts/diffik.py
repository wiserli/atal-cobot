#!/usr/bin/env python3
"""
Differential Inverse Kinematics (DiffIK) for WISERLI-ATAL-C1.1.
Adapted from kevinzakka/mjctrl (diffik.py) for WISERLI-ATAL-C1.1 with and without Robotiq 2F-85.

Usage:
  # Track a circular trajectory with Robotiq 2F-85 gripper (default):
  python mujoco/scripts/diffik.py

  # Interactive mocap target (double-click red box, Ctrl + drag in viewer):
  python mujoco/scripts/diffik.py --mode mocap

  # Bare 6-DOF arm without gripper:
  python mujoco/scripts/diffik.py --no-gripper
"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

# Integration timestep in seconds.
integration_dt: float = 0.1

# Damping term for the damped least-squares pseudoinverse.
damping: float = 1e-4

# Gains for twist computation (between 0 and 1).
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation on robot links (excludes free objects like grasp_cube).
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint velocity in rad/s (0 to disable).
max_angvel: float = 2.0

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ARM_ACTUATORS = [f"{name}_act" for name in ARM_JOINTS]


def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return (x, y) coordinates of a circle of radius r centered at (h, k)."""
    x = r * np.cos(2.0 * np.pi * f * t) + h
    y = r * np.sin(2.0 * np.pi * f * t) + k
    return np.array([x, y])


def enable_robot_gravcomp(model: mujoco.MjModel) -> None:
    """Enable gravity compensation on all robot bodies without affecting free objects."""
    excluded = {"world", "grasp_cube", "target"}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name not in excluded:
            model.body_gravcomp[body_id] = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffIK controller for WISERLI-ATAL-C1.1")
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Use bare arm scene.xml instead of scene_2f85.xml.",
    )
    parser.add_argument(
        "--mode",
        choices=["circle", "mocap"],
        default="circle",
        help="'circle' follows a circular trajectory; 'mocap' tracks the draggable target box.",
    )
    parser.add_argument(
        "--gripper",
        type=float,
        default=0.0,
        help="Robotiq 2F-85 command (0=open .. 255=closed) when gripper is attached.",
    )
    args = parser.parse_args()

    mujoco_dir = Path(__file__).resolve().parent.parent
    xml_path = mujoco_dir / ("scene.xml" if args.no_gripper else "scene_2f85.xml")
    site_name = "attachment_site" if args.no_gripper else "pinch"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    if gravity_compensation:
        enable_robot_gravcomp(model)

    site_id = model.site(site_name).id
    jnt_ids = np.array([model.joint(name).id for name in ARM_JOINTS])
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in jnt_ids])
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in jnt_ids])
    actuator_ids = np.array([model.actuator(name).id for name in ARM_ACTUATORS])
    jnt_lo = model.jnt_range[jnt_ids, 0]
    jnt_hi = model.jnt_range[jnt_ids, 1]

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    key_id = model.key("home").id
    mocap_id = model.body("target").mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    circle_z = 0.30 if args.no_gripper else 0.22

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        # Initialize mocap target cleanly at current site pose.
        data.mocap_pos[mocap_id] = data.site(site_id).xpos.copy()
        mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)

        if gripper_act_id >= 0:
            data.ctrl[gripper_act_id] = np.clip(args.gripper, 0.0, 255.0)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            if args.mode == "circle":
                data.mocap_pos[mocap_id, 0:2] = circle(data.time, r=0.08, h=0.45, k=0.0, f=0.35)
                data.mocap_pos[mocap_id, 2] = circle_z

            # Position & orientation error -> desired spatial twist.
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Compute 6x6 Jacobian for the 6 arm DOFs.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            jac_arm = jac[:, dof_ids]

            # Damped least-squares: J @ dq = twist.
            dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, twist)

            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate & clamp arm joint positions.
            q_des = np.clip(data.qpos[qpos_ids] + dq * integration_dt, jnt_lo, jnt_hi)
            data.ctrl[actuator_ids] = q_des

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
