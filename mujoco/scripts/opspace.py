#!/usr/bin/env python3
"""
Operational Space Control (OSC / Cartesian Impedance Torque Control) for WISERLI-ATAL-C1.1.
Adapted from kevinzakka/mjctrl (opspace.py).

Converts the 6 arm actuators to direct torque control in-memory while preserving the
Robotiq 2F-85 gripper actuator, computes task-space inertia decoupling (Mx), Cartesian
stiffness/damping, nullspace posture impedance, and exact Coriolis/gravity compensation.

Usage:
  python mujoco/scripts/opspace.py
  python mujoco/scripts/opspace.py --mode circle
  python mujoco/scripts/opspace.py --no-gripper
"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

# Cartesian impedance control gains (tuned for ~2.5 kg WISERLI-ATAL-C1.1 cobot).
impedance_pos = np.asarray([120.0, 120.0, 120.0])  # [N/m]
impedance_ori = np.asarray([35.0, 35.0, 35.0])  # [Nm/rad]

# Nullspace joint impedance gains for joint_1..joint_6.
Kp_null = np.asarray([25.0, 25.0, 20.0, 12.0, 10.0, 6.0])

# Critical damping ratio.
damping_ratio: float = 1.0

# Twist scaling gains.
Kpos: float = 0.95
Kori: float = 0.95
integration_dt: float = 1.0
gravity_compensation: bool = True
dt: float = 0.002

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ARM_ACTUATORS = [f"{name}_act" for name in ARM_JOINTS]


def convert_arm_actuators_to_torque(model: mujoco.MjModel, actuator_ids: np.ndarray) -> None:
    """Switch the 6 arm position actuators to pure torque (motor) mode in-memory."""
    for aid in actuator_ids:
        model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_gainprm[aid, 0] = 1.0
        model.actuator_biasprm[aid, :] = 0.0
        model.actuator_ctrlrange[aid, :] = model.actuator_forcerange[aid, :]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operational Space Controller (OSC) for WISERLI-ATAL-C1.1"
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Use bare arm scene.xml instead of scene_2f85.xml.",
    )
    parser.add_argument(
        "--mode",
        choices=["mocap", "circle"],
        default="mocap",
        help="'mocap' (default) tracks the draggable target box; 'circle' runs a circular path.",
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

    site_id = model.site(site_name).id
    jnt_ids = np.array([model.joint(name).id for name in ARM_JOINTS])
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in jnt_ids])
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in jnt_ids])
    actuator_ids = np.array([model.actuator(name).id for name in ARM_ACTUATORS])

    # Switch arm actuators from position servos to direct torque control.
    convert_arm_actuators_to_torque(model, actuator_ids)
    tau_lo = model.actuator_ctrlrange[actuator_ids, 0]
    tau_hi = model.actuator_ctrlrange[actuator_ids, 1]

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    key_id = model.key("home").id
    q0 = model.key("home").qpos[qpos_ids].copy()
    mocap_id = model.body("target").mocapid[0]

    # Compute stiffness and critical damping matrices.
    damping_pos = damping_ratio * 2.0 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2.0 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2.0 * np.sqrt(Kp_null)

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    eye_nv = np.eye(model.nv)
    eye_arm = np.eye(len(ARM_JOINTS))
    circle_z = 0.30 if args.no_gripper else 0.22

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        data.mocap_pos[mocap_id] = data.site(site_id).xpos.copy()
        mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)

        if gripper_act_id >= 0:
            data.ctrl[gripper_act_id] = np.clip(args.gripper, 0.0, 255.0)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            if args.mode == "circle":
                w = 2.0 * np.pi * 0.35 * data.time
                data.mocap_pos[mocap_id] = [0.45 + 0.08 * np.cos(w), 0.08 * np.sin(w), circle_z]

            # Spatial error twist.
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # 6x6 Arm Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            jac_arm = jac[:, dof_ids]

            # 6x6 Arm inverse mass matrix and operational-space inertia matrix Mx.
            mujoco.mj_solveM(model, data, M_inv, eye_nv)
            M_arm_inv = M_inv[np.ix_(dof_ids, dof_ids)]
            M_arm = np.linalg.inv(M_arm_inv)

            Mx_inv = jac_arm @ M_arm_inv @ jac_arm.T
            if abs(np.linalg.det(Mx_inv)) >= 1e-4:
                Mx = np.linalg.inv(Mx_inv)
            else:
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

            # Task-space impedance torques.
            qvel_arm = data.qvel[dof_ids]
            tau = jac_arm.T @ Mx @ (Kp * twist - Kd * (jac_arm @ qvel_arm))

            # Dynamically consistent nullspace posture control.
            Jbar = M_arm_inv @ jac_arm.T @ Mx
            ddq = Kp_null * (q0 - data.qpos[qpos_ids]) - Kd_null * qvel_arm
            tau += (eye_arm - jac_arm.T @ Jbar.T) @ (M_arm @ ddq)

            # Gravity + Coriolis compensation.
            if gravity_compensation:
                tau += data.qfrc_bias[dof_ids]

            data.ctrl[actuator_ids] = np.clip(tau, tau_lo, tau_hi)
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
