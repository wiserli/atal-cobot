#!/usr/bin/env python3
"""
Autonomous Pick-and-Place Demo using Differential IK + Robotiq 2F-85 in MuJoCo.

Uses damped least-squares Differential IK on the 'pinch' TCP site to pick up
'grasp_cube', lift it, move it across the workspace, place it down, and repeat.

Usage:
  python mujoco/scripts/pick_and_place_diffik.py
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

integration_dt: float = 0.1
damping: float = 1e-3
Kpos: float = 0.90
Kori: float = 0.85
dt: float = 0.002
max_angvel: float = 1.6

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ARM_ACTUATORS = [f"{name}_act" for name in ARM_JOINTS]


def enable_robot_gravcomp(model: mujoco.MjModel) -> None:
    excluded = {"world", "grasp_cube", "target"}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name not in excluded:
            model.body_gravcomp[body_id] = 1.0


def main() -> None:
    mujoco_dir = Path(__file__).resolve().parent.parent
    xml_path = mujoco_dir / "scene_2f85.xml"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    enable_robot_gravcomp(model)

    site_id = model.site("pinch").id
    jnt_ids = np.array([model.joint(name).id for name in ARM_JOINTS])
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in jnt_ids])
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in jnt_ids])
    actuator_ids = np.array([model.actuator(name).id for name in ARM_ACTUATORS])
    jnt_lo = model.jnt_range[jnt_ids, 0]
    jnt_hi = model.jnt_range[jnt_ids, 1]

    gripper_act_id = model.actuator("fingers_actuator").id
    cube_body_id = model.body("grasp_cube").id
    key_id = model.key("home").id
    mocap_id = model.body("target").mocapid[0]

    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Top-down grasp orientation (aligned with home pinch orientation facing workspace).
    grasp_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

    # Place target alternates between +Y and -Y sides.
    place_xy = [np.array([0.38, 0.16]), np.array([0.38, -0.16])]
    cycle_idx = 0

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        # Capture natural downward-pointing home orientation as clean grasp orientation.
        mujoco.mju_mat2Quat(grasp_quat, data.site(site_id).xmat)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # State machine: (phase_name, duration_s, gripper_cmd)
        phase = 0
        phase_t0 = data.time
        pick_xy = data.xpos[cube_body_id, :2].copy()

        while viewer.is_running():
            step_start = time.time()
            elapsed = data.time - phase_t0
            dst_xy = place_xy[cycle_idx % len(place_xy)]

            if phase == 0:  # Hover above cube (open)
                pick_xy = data.xpos[cube_body_id, :2].copy()
                target_pos = np.array([pick_xy[0], pick_xy[1], 0.13])
                gripper_cmd = 0.0
                if elapsed > 1.8:
                    phase, phase_t0 = 1, data.time
            elif phase == 1:  # Descend onto cube (open)
                target_pos = np.array([pick_xy[0], pick_xy[1], 0.024])
                gripper_cmd = 0.0
                if elapsed > 1.5:
                    phase, phase_t0 = 2, data.time
            elif phase == 2:  # Close Robotiq 2F-85 firmly around cube
                target_pos = np.array([pick_xy[0], pick_xy[1], 0.024])
                gripper_cmd = 225.0
                if elapsed > 1.0:
                    phase, phase_t0 = 3, data.time
            elif phase == 3:  # Lift cube
                target_pos = np.array([pick_xy[0], pick_xy[1], 0.16])
                gripper_cmd = 225.0
                if elapsed > 1.4:
                    phase, phase_t0 = 4, data.time
            elif phase == 4:  # Carry above place location
                target_pos = np.array([dst_xy[0], dst_xy[1], 0.16])
                gripper_cmd = 225.0
                if elapsed > 1.8:
                    phase, phase_t0 = 5, data.time
            elif phase == 5:  # Lower to place location
                target_pos = np.array([dst_xy[0], dst_xy[1], 0.032])
                gripper_cmd = 225.0
                if elapsed > 1.4:
                    phase, phase_t0 = 6, data.time
            elif phase == 6:  # Release cube
                target_pos = np.array([dst_xy[0], dst_xy[1], 0.032])
                gripper_cmd = 0.0
                if elapsed > 0.9:
                    phase, phase_t0 = 7, data.time
            else:  # Retreat upward and start next pick cycle
                target_pos = np.array([dst_xy[0], dst_xy[1], 0.16])
                gripper_cmd = 0.0
                if elapsed > 1.2:
                    cycle_idx += 1
                    phase, phase_t0 = 0, data.time

            data.mocap_pos[mocap_id] = target_pos
            data.mocap_quat[mocap_id] = grasp_quat

            # Differential IK step.
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            jac_arm = jac[:, dof_ids]
            dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, twist)

            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            q_des = np.clip(data.qpos[qpos_ids] + dq * integration_dt, jnt_lo, jnt_hi)
            data.ctrl[actuator_ids] = q_des
            data.ctrl[gripper_act_id] = gripper_cmd

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
