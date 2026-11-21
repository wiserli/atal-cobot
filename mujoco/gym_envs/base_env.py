"""Base Farama Gymnasium environment for WISERLI-ATAL-C1.1 Cobot with Robotiq 2F-85."""

from __future__ import annotations
import os
from typing import Any, Literal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .utils import (
    diffik_solve,
    clip_joint_limits,
    action_to_gripper_ctrl,
    gripper_ctrl_to_action,
    resolve_mujoco_path,
    enable_robot_gravcomp,
)


class WiserliAtalBaseEnv(gym.Env):
    """Abstract Base Class for WISERLI-ATAL-C1.1 MuJoCo Gymnasium environments."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_path: str,
        action_mode: Literal["cartesian", "joint_position"] = "cartesian",
        frame_skip: int = 25,
        max_episode_steps: int = 150,
        render_mode: str | None = None,
        camera_name: str | None = None,
        width: int = 480,
        height: int = 360,
    ):
        super().__init__()

        self.action_mode = action_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.width = width
        self.height = height

        # Resolve path
        if not os.path.isabs(xml_path):
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(curr_dir, "..", xml_path)
        xml_path = resolve_mujoco_path(xml_path)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Enable internal gravity compensation on cobot bodies (standard for cobot controllers)
        enable_robot_gravcomp(self.model)

        # Cache IDs
        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(1, 7)]
        self.arm_qpos_indices = [self.model.jnt_qposadr[j] for j in self.arm_joint_ids]
        self.arm_qvel_indices = [self.model.jnt_dofadr[j] for j in self.arm_joint_ids]
        self.arm_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}_act") for i in range(1, 7)]

        # Set realistic industrial stiffness gains (kp=800, kv=80) to eliminate gravity droop
        for act_id in self.arm_act_ids:
            self.model.actuator_gainprm[act_id, 0] = 800.0
            self.model.actuator_biasprm[act_id, 1] = -800.0
            self.model.actuator_biasprm[act_id, 2] = -80.0

        self.gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        self.pinch_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.attachment_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        # Wrist Force / Torque sensor IDs
        self.force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_force")
        self.torque_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_torque")

        # Scale factors
        self.max_cartesian_step = 0.03  # 3 cm per step
        self.max_yaw_step = 0.08        # ~4.5 deg per step
        self.max_joint_step = 0.05      # radians per step

        # Define Action Space
        if self.action_mode == "cartesian":
            # [dx, dy, dz, dyaw, gripper_cmd] all in [-1, 1]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        elif self.action_mode == "joint_position":
            # [dq1..dq6, gripper_cmd] all in [-1, 1]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action mode: {self.action_mode}")

        # Rendering helpers
        self._viewer = None
        self._renderer = None
        self._step_count = 0

    def _get_arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_indices].copy()

    def _get_arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self.arm_qvel_indices].copy()

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.pinch_site_id].copy()

    def _get_ee_mat(self) -> np.ndarray:
        return self.data.site_xmat[self.pinch_site_id].reshape(3, 3).copy()

    def _get_wrist_ft(self) -> np.ndarray:
        """Returns 6-DoF wrist force/torque sensor readings [Fx, Fy, Fz, Tx, Ty, Tz]."""
        if self.force_sensor_id >= 0 and self.torque_sensor_id >= 0:
            force_adr = self.model.sensor_adr[self.force_sensor_id]
            torque_adr = self.model.sensor_adr[self.torque_sensor_id]
            force = self.data.sensordata[force_adr : force_adr + 3]
            torque = self.data.sensordata[torque_adr : torque_adr + 3]
            return np.concatenate([force, torque], dtype=np.float32)
        return np.zeros(6, dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        """Applies the policy action to actuators."""
        action = np.clip(action, -1.0, 1.0)

        if self.action_mode == "cartesian":
            dx, dy, dz, dyaw, gripper_val = action
            current_ee_pos = self._get_ee_pos()
            delta_pos = np.array([dx, dy, dz], dtype=np.float64) * self.max_cartesian_step

            # Floor safety clearance: prevent commanding into ground (< -0.30m)
            if (current_ee_pos[2] + delta_pos[2]) < -0.30:
                delta_pos[2] = max(0.0, delta_pos[2])

            dq = diffik_solve(
                model=self.model,
                data=self.data,
                site_id=self.pinch_site_id,
                delta_pos=delta_pos,
                target_quat=None,  # Agile 3D position reach without orientation lock
                dof_ids=np.array(self.arm_qvel_indices),
                damping=1e-4,
                integration_dt=0.1,
                max_angvel=2.0,
            )

            current_q = self._get_arm_qpos()
            target_q = clip_joint_limits(self.model, current_q + dq, self.arm_joint_ids)

            # Apply yaw update to wrist joint 6
            target_q[5] += dyaw * self.max_yaw_step

            # Set arm actuators
            for i, act_id in enumerate(self.arm_act_ids):
                self.data.ctrl[act_id] = target_q[i]

            # Set gripper actuator
            self.data.ctrl[self.gripper_act_id] = action_to_gripper_ctrl(gripper_val)

        elif self.action_mode == "joint_position":
            dq = action[:6] * self.max_joint_step
            gripper_val = action[6]
            current_q = self._get_arm_qpos()
            target_q = clip_joint_limits(self.model, current_q + dq, self.arm_joint_ids)

            for i, act_id in enumerate(self.arm_act_ids):
                self.data.ctrl[act_id] = target_q[i]

            self.data.ctrl[self.gripper_act_id] = action_to_gripper_ctrl(gripper_val)

    def _step_simulation(self) -> None:
        """Steps MuJoCo physics forward by frame_skip sub-steps."""
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def render(self) -> np.ndarray | None:
        """Renders current simulation frame according to render_mode."""
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        elif self.render_mode in ("rgb_array", "depth_array"):
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

            cam_id = -1
            if self.camera_name is not None:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

            self._renderer.update_scene(self.data, camera=cam_id)

            if self.render_mode == "rgb_array":
                return self._renderer.render()
            else:
                self._renderer.enable_depth_rendering()
                depth = self._renderer.render()
                self._renderer.disable_depth_rendering()
                return depth

        return None

    def close(self) -> None:
        """Safely cleans up environment resources without crashing GLFW on Linux/WSL."""
        if self._renderer is not None:
            try:
                del self._renderer
            except Exception:
                pass
            self._renderer = None

        if self._viewer is not None:
            try:
                # Do not force immediate C-level window destroy to avoid GLFW atexit segfault on Linux
                if hasattr(self._viewer, "is_running") and self._viewer.is_running():
                    self._viewer.close()
            except Exception:
                pass
            self._viewer = None
