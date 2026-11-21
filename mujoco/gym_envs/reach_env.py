"""Reach task environment for WISERLI-ATAL-C1.1 Cobot with Robotiq 2F-85."""

from __future__ import annotations
import numpy as np
from gymnasium import spaces
import mujoco

from .base_env import WiserliAtalBaseEnv


class WiserliAtalReachEnv(WiserliAtalBaseEnv):
    """End-effector reaching task for WISERLI-ATAL-C1.1 Cobot.
    
    The agent must control the robot arm to bring the gripper fingertip center
    ('pinch' site) within a target distance of a 3D goal marker.
    """

    def __init__(
        self,
        xml_path: str = "scene_rl_reach.xml",
        action_mode: str = "cartesian",
        reward_type: str = "dense",
        distance_threshold: float = 0.03,  # 3 cm
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

        # Target mocap body ID
        self.target_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        ]

        # Workspace bounding box for random target generation (x, y, z)
        # Frontal reachable space: x in [0.30, 0.52], y in [-0.22, 0.22], z in [0.10, 0.35]
        self.target_bounds_low = np.array([0.30, -0.22, 0.10], dtype=np.float32)
        self.target_bounds_high = np.array([0.52, 0.22, 0.35], dtype=np.float32)

        self.current_target_pos = np.zeros(3, dtype=np.float32)

        # Observation Space (25-dim vector):
        # [qpos(6), qvel(6), ee_pos(3), target_pos(3), ee_to_target(3), ee_dist(1), wrist_ft(6)] -> 28 total
        obs_dim = 6 + 6 + 3 + 3 + 3 + 1 + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _sample_target_pos(self) -> np.ndarray:
        return self.np_random.uniform(self.target_bounds_low, self.target_bounds_high)

    def _set_target_pos(self, pos: np.ndarray) -> None:
        self.current_target_pos = pos.copy()
        if self.target_mocap_id >= 0:
            self.data.mocap_pos[self.target_mocap_id] = pos

    def _get_obs(self) -> np.ndarray:
        qpos = self._get_arm_qpos()
        qvel = self._get_arm_qvel()
        ee_pos = self._get_ee_pos()
        target_pos = self.current_target_pos.copy()
        rel_pos = target_pos - ee_pos
        dist = np.linalg.norm(rel_pos, keepdims=True).astype(np.float32)
        wrist_ft = self._get_wrist_ft()

        return np.concatenate(
            [qpos, qvel, ee_pos, target_pos, rel_pos, dist, wrist_ft],
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

        # Reset MuJoCo physics to initial keyframe 'ready' or 'home'
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Add small random noise to initial joint angles
        noise = self.np_random.uniform(-0.05, 0.05, size=6)
        for i, idx in enumerate(self.arm_qpos_indices):
            self.data.qpos[idx] += noise[i]

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Sample and set new target position
        new_target = self._sample_target_pos()
        self._set_target_pos(new_target)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {
            "is_success": False,
            "distance": float(np.linalg.norm(self._get_ee_pos() - self.current_target_pos)),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # Apply action and simulate
        self._apply_action(action)
        self._step_simulation()

        obs = self._get_obs()
        ee_pos = self._get_ee_pos()
        dist = float(np.linalg.norm(ee_pos - self.current_target_pos))
        is_success = dist < self.distance_threshold

        # Compute reward
        if self.reward_type == "dense":
            # Distance reward + action penalty + success bonus
            ctrl_cost = 0.01 * float(np.sum(np.square(action)))
            reward = -dist - ctrl_cost
            if is_success:
                reward += 2.0
        else:
            # Sparse 0 / -1 reward
            reward = 0.0 if is_success else -1.0

        terminated = is_success
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "is_success": is_success,
            "distance": dist,
            "step_count": self._step_count,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
