"""Pick-and-place task environment for WISERLI-ATAL-C1.1 Cobot with Robotiq 2F-85."""

from __future__ import annotations
import numpy as np
from gymnasium import spaces
import mujoco

from .base_env import WiserliAtalBaseEnv
from .utils import gripper_ctrl_to_action


class WiserliAtalPickAndPlaceEnv(WiserliAtalBaseEnv):
    """Pick-and-Place manipulation task for WISERLI-ATAL-C1.1 Cobot.
    
    The robot must:
      1. Reach the cube on the table
      2. Close its Robotiq 2F-85 gripper to grasp the cube
      3. Lift the cube above the table
      4. Transport the cube and place it inside the drop goal marker
    """

    def __init__(
        self,
        xml_path: str = "scene_rl_pick_and_place.xml",
        action_mode: str = "cartesian",
        reward_type: str = "dense",
        render_mode: str | None = None,
        max_episode_steps: int = 200,
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

        # Object & Goal IDs
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
        self.obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        self.obj_qpos_adr = self.model.jnt_qposadr[self.obj_joint_id]

        self.goal_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drop_goal")
        ]

        # Table surface height (tabletop at Z=0.0)
        self.table_height = 0.0
        self.cube_half_size = 0.0175

        # Workspace sampling bounds on the tabletop
        # Cube spawn area (left side of table): x in [0.38, 0.46], y in [-0.16, -0.06]
        self.obj_bounds_low = np.array([0.38, -0.16], dtype=np.float32)
        self.obj_bounds_high = np.array([0.46, -0.06], dtype=np.float32)

        # Goal spawn area (right side of table): x in [0.38, 0.46], y in [0.06, 0.16]
        self.goal_bounds_low = np.array([0.38, 0.06], dtype=np.float32)
        self.goal_bounds_high = np.array([0.46, 0.16], dtype=np.float32)

        self.current_goal_pos = np.zeros(3, dtype=np.float32)

        # Observation Space (36-dim vector):
        # [qpos(6), qvel(6), ee_pos(3), gripper_state(1), obj_pos(3), obj_quat(4),
        #  goal_pos(3), ee_to_obj(3), obj_to_goal(3), ee_to_obj_dist(1), obj_to_goal_dist(1), wrist_ft(6)]
        obs_dim = 6 + 6 + 3 + 1 + 3 + 4 + 3 + 3 + 3 + 1 + 1 + 6  # 40
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_obj_pos(self) -> np.ndarray:
        return self.data.qpos[self.obj_qpos_adr : self.obj_qpos_adr + 3].copy()

    def _get_obj_quat(self) -> np.ndarray:
        return self.data.qpos[self.obj_qpos_adr + 3 : self.obj_qpos_adr + 7].copy()

    def _set_goal_pos(self, pos: np.ndarray) -> None:
        self.current_goal_pos = pos.copy()
        if self.goal_mocap_id >= 0:
            self.data.mocap_pos[self.goal_mocap_id] = pos

    def _get_obs(self) -> np.ndarray:
        qpos = self._get_arm_qpos()
        qvel = self._get_arm_qvel()
        ee_pos = self._get_ee_pos()
        gripper_state = np.array([gripper_ctrl_to_action(self.data.ctrl[self.gripper_act_id])], dtype=np.float32)

        obj_pos = self._get_obj_pos()
        obj_quat = self._get_obj_quat()
        goal_pos = self.current_goal_pos.copy()

        ee_to_obj = obj_pos - ee_pos
        obj_to_goal = goal_pos - obj_pos

        ee_to_obj_dist = np.linalg.norm(ee_to_obj, keepdims=True).astype(np.float32)
        obj_to_goal_dist = np.linalg.norm(obj_to_goal, keepdims=True).astype(np.float32)

        wrist_ft = self._get_wrist_ft()

        return np.concatenate(
            [
                qpos,
                qvel,
                ee_pos,
                gripper_state,
                obj_pos,
                obj_quat,
                goal_pos,
                ee_to_obj,
                obj_to_goal,
                ee_to_obj_dist,
                obj_to_goal_dist,
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

        # Randomize initial object position on table
        rand_obj_xy = self.np_random.uniform(self.obj_bounds_low, self.obj_bounds_high)
        self.data.qpos[self.obj_qpos_adr] = rand_obj_xy[0]
        self.data.qpos[self.obj_qpos_adr + 1] = rand_obj_xy[1]
        self.data.qpos[self.obj_qpos_adr + 2] = self.table_height + self.cube_half_size + 0.001
        self.data.qpos[self.obj_qpos_adr + 3 : self.obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Randomize drop goal position
        rand_goal_xy = self.np_random.uniform(self.goal_bounds_low, self.goal_bounds_high)
        goal_pos = np.array([rand_goal_xy[0], rand_goal_xy[1], self.table_height + 0.002], dtype=np.float32)
        self._set_goal_pos(goal_pos)

        # Settle simulation
        for _ in range(15):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = {
            "is_success": False,
            "is_lifted": False,
            "dist_to_obj": float(np.linalg.norm(self._get_ee_pos() - self._get_obj_pos())),
            "dist_obj_to_goal": float(np.linalg.norm(self._get_obj_pos() - self.current_goal_pos)),
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
        obj_pos = self._get_obj_pos()
        goal_pos = self.current_goal_pos

        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        dist_obj_goal = float(np.linalg.norm(obj_pos - goal_pos))
        is_lifted = bool(obj_pos[2] > (self.table_height + 0.035))
        is_success = bool(dist_obj_goal < 0.045)

        # Multi-stage shaped reward
        if self.reward_type == "dense":
            # 1. Reach penalty
            r_reach = -dist_ee_obj

            # 2. Lift bonus
            r_lift = 1.0 if is_lifted else 0.0

            # 3. Transport reward (active when lifted)
            r_transport = -dist_obj_goal if is_lifted else 0.0

            # 4. Action cost
            r_ctrl = -0.01 * float(np.sum(np.square(action)))

            reward = r_reach + r_lift + r_transport + r_ctrl
            if is_success:
                reward += 5.0
        else:
            reward = 1.0 if is_success else 0.0

        terminated = is_success
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "is_success": is_success,
            "is_lifted": is_lifted,
            "dist_to_obj": dist_ee_obj,
            "dist_obj_to_goal": dist_obj_goal,
            "step_count": self._step_count,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
