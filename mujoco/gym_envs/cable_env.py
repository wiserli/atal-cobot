"""Deformable flexible cable manipulation environment for WISERLI-ATAL-C1.1 with Robotiq 2F-85."""

from __future__ import annotations
import numpy as np
from gymnasium import spaces
import mujoco

from .base_env import WiserliAtalBaseEnv
from .utils import gripper_ctrl_to_action


class WiserliAtalCableEnv(WiserliAtalBaseEnv):
    """Deformable cable routing and manipulation task for WISERLI-ATAL-C1.1 Cobot.
    
    The robot must:
      1. Approach the flexible cable on the table
      2. Grasp the free segment / tip with its Robotiq 2F-85 parallel gripper
      3. Manipulate and pull/route the flexible cable to the designated goal site
      4. Observe contact and tension through the wrist 6-axis Force/Torque sensor
    """

    def __init__(
        self,
        xml_path: str = "scene_rl_cable.xml",
        action_mode: str = "cartesian",
        reward_type: str = "dense",
        distance_threshold: float = 0.045,  # 4.5 cm
        render_mode: str | None = None,
        max_episode_steps: int = 150,
        **kwargs,
    ):
        super().__init__(
            xml_path=xml_path,
            action_mode=action_mode,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        # Cable body IDs: resolve to cable_plug if present, otherwise wire:B_last
        plug_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cable_plug")
        if plug_id >= 0:
            self.cable_tip_id = plug_id
        else:
            self.cable_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wire:B_last")

        mid_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wire:B_5")
        if mid_id >= 0:
            self.cable_mid_id = mid_id
        else:
            self.cable_mid_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wire:B_6")

        # Goal mocap body ID
        self.goal_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cable_goal")
        ]

        # Workspace sampling bounds for the cable goal site (x, y, z)
        # Centered around [0.38, 0.12, 0.035] on tabletop
        self.goal_bounds_low = np.array([0.36, 0.10, 0.025], dtype=np.float32)
        self.goal_bounds_high = np.array([0.42, 0.14, 0.045], dtype=np.float32)

        self.current_goal_pos = np.zeros(3, dtype=np.float32)

        # Observation Space (39-dim vector):
        # [qpos(6), qvel(6), ee_pos(3), gripper_state(1), cable_tip_pos(3), cable_mid_pos(3),
        #  goal_pos(3), ee_to_tip(3), tip_to_goal(3), ee_to_tip_dist(1), tip_to_goal_dist(1), wrist_ft(6)]
        obs_dim = 6 + 6 + 3 + 1 + 3 + 3 + 3 + 3 + 3 + 1 + 1 + 6  # 39
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_cable_tip_pos(self) -> np.ndarray:
        return self.data.xpos[self.cable_tip_id].copy()

    def _get_cable_mid_pos(self) -> np.ndarray:
        return self.data.xpos[self.cable_mid_id].copy()

    def _set_goal_pos(self, pos: np.ndarray) -> None:
        self.current_goal_pos = pos.copy()
        if self.goal_mocap_id >= 0:
            self.data.mocap_pos[self.goal_mocap_id] = pos

    def _get_obs(self) -> np.ndarray:
        qpos = self._get_arm_qpos()
        qvel = self._get_arm_qvel()
        ee_pos = self._get_ee_pos()
        gripper_state = np.array([gripper_ctrl_to_action(self.data.ctrl[self.gripper_act_id])], dtype=np.float32)

        cable_tip_pos = self._get_cable_tip_pos()
        cable_mid_pos = self._get_cable_mid_pos()
        goal_pos = self.current_goal_pos.copy()

        ee_to_tip = cable_tip_pos - ee_pos
        tip_to_goal = goal_pos - cable_tip_pos

        ee_to_tip_dist = np.linalg.norm(ee_to_tip, keepdims=True).astype(np.float32)
        tip_to_goal_dist = np.linalg.norm(tip_to_goal, keepdims=True).astype(np.float32)

        wrist_ft = self._get_wrist_ft()

        return np.concatenate(
            [
                qpos,
                qvel,
                ee_pos,
                gripper_state,
                cable_tip_pos,
                cable_mid_pos,
                goal_pos,
                ee_to_tip,
                tip_to_goal,
                ee_to_tip_dist,
                tip_to_goal_dist,
                wrist_ft,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0

        # Reset MuJoCo physics to initial keyframe 'home'
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Randomize target goal position on the right side of the table
        rand_goal = self.np_random.uniform(self.goal_bounds_low, self.goal_bounds_high)
        self._set_goal_pos(rand_goal)

        # Settle simulation so the deformable cable rests naturally on the table
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        tip_to_goal_dist = float(np.linalg.norm(self._get_cable_tip_pos() - self.current_goal_pos))
        ee_to_tip_dist = float(np.linalg.norm(self._get_ee_pos() - self._get_cable_tip_pos()))

        info = {
            "is_success": False,
            "dist_tip_to_goal": tip_to_goal_dist,
            "dist_ee_to_cable": ee_to_tip_dist,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        self._apply_action(action)
        self._step_simulation()

        obs = self._get_obs()
        ee_pos = self._get_ee_pos()
        cable_tip = self._get_cable_tip_pos()
        goal_pos = self.current_goal_pos

        dist_ee_tip = float(np.linalg.norm(ee_pos - cable_tip))
        dist_tip_goal = float(np.linalg.norm(cable_tip - goal_pos))
        is_success = bool(dist_tip_goal < self.distance_threshold)

        # Tension sensing: check wrist force magnitude
        wrist_ft = self._get_wrist_ft()
        wrist_force_mag = float(np.linalg.norm(wrist_ft[:3]))

        if self.reward_type == "dense":
            # 1. Approach distance to cable tip
            r_approach = -dist_ee_tip

            # 2. Cable tip to goal distance
            r_goal = -2.0 * dist_tip_goal

            # 3. Excessive tension penalty (encourages gentle, compliant pulling)
            r_tension = -0.02 * max(0.0, wrist_force_mag - 25.0)

            # 4. Action regularization cost
            r_ctrl = -0.01 * float(np.sum(np.square(action)))

            reward = r_approach + r_goal + r_tension + r_ctrl
            if is_success:
                reward += 5.0
        else:
            reward = 1.0 if is_success else 0.0

        terminated = is_success
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "is_success": is_success,
            "dist_tip_to_goal": dist_tip_goal,
            "dist_ee_to_cable": dist_ee_tip,
            "wrist_force_mag": wrist_force_mag,
            "step_count": self._step_count,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

