#!/usr/bin/env python3
"""Interactive test and demonstration script for WISERLI-ATAL-C1.1 Gymnasium environments.

Usage:
    # Test Cartesian Reach environment:
    python mujoco/scripts/test_gym_env.py --env WiserliAtalReach-v0 --policy heuristic --render

    # Test Joint-Space Reach environment:
    python mujoco/scripts/test_gym_env.py --env WiserliAtalReachJoint-v0 --policy heuristic --render

    # Test Pick-and-Place environment:
    python mujoco/scripts/test_gym_env.py --env WiserliAtalPickAndPlace-v0 --policy heuristic --render

    # Test Deformable Cable Manipulation environment:
    python mujoco/scripts/test_gym_env.py --env WiserliAtalCable-v0 --policy heuristic --render
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure mujoco package directory is in sys.path
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MUJOCO_DIR = os.path.dirname(CURR_DIR)
if MUJOCO_DIR not in sys.path:
    sys.path.insert(0, MUJOCO_DIR)

import gymnasium as gym
import mujoco
import gym_envs  # Registers WiserliAtal environments


def heuristic_reach_policy(obs: np.ndarray) -> np.ndarray:
    """Proportional task-space controller that drives the gripper directly to the target."""
    # In WiserliAtalReachEnv: rel_pos = target_pos - ee_pos is at obs[18:21]
    rel_pos = obs[18:21]
    dist = np.linalg.norm(rel_pos)
    if dist > 1e-4:
        dir_vec = rel_pos / dist
    else:
        dir_vec = np.zeros(3)

    action_xyz = np.clip(dir_vec, -1.0, 1.0)
    dyaw = 0.0
    gripper = -1.0  # Open gripper
    return np.array([action_xyz[0], action_xyz[1], action_xyz[2], dyaw, gripper], dtype=np.float32)


def heuristic_reach_joint_policy(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """Computes 7-DoF joint actions using Jacobian inverse for WiserliAtalReachJoint-v0."""
    rel_pos = obs[18:21]
    raw_env = env.unwrapped
    jacp = np.zeros((3, raw_env.model.nv))
    mujoco.mj_jacSite(raw_env.model, raw_env.data, jacp, None, raw_env.pinch_site_id)
    J = jacp[:, raw_env.arm_qvel_indices]

    A = J @ J.T + 1e-4 * np.eye(3)
    dq = J.T @ np.linalg.solve(A, rel_pos)

    action_joints = np.clip(dq / raw_env.max_joint_step, -1.0, 1.0)
    gripper = -1.0
    return np.append(action_joints, gripper).astype(np.float32)


class HeuristicPickAndPlaceController:
    """State-machine heuristic policy that reaches, grasps, lifts, and places the cube."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "APPROACH"
        self.grasp_timer = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ee = obs[12:15]
        obj = obs[16:19]
        goal = obs[23:26]

        target = np.copy(ee)
        gripper = -1.0

        if self.state == "APPROACH":
            # 8cm above cube
            target = np.array([obj[0], obj[1], 0.08])
            if np.linalg.norm(ee[:2] - obj[:2]) < 0.03 and np.abs(ee[2] - 0.08) < 0.03:
                self.state = "DESCEND"

        elif self.state == "DESCEND":
            # Descend around cube
            target = np.array([obj[0], obj[1], 0.016])
            if np.linalg.norm(ee - target) < 0.025:
                self.state = "GRASP"

        elif self.state == "GRASP":
            # Close gripper firmly
            target = np.array([obj[0], obj[1], 0.016])
            gripper = 1.0
            self.grasp_timer += 1
            if self.grasp_timer >= 12:
                self.state = "LIFT"

        elif self.state == "LIFT":
            # Lift up
            gripper = 1.0
            target = np.array([ee[0], ee[1], 0.14])
            if ee[2] > 0.10:
                self.state = "TRANSPORT"

        elif self.state == "TRANSPORT":
            # Carry across table to goal
            gripper = 1.0
            target = np.array([goal[0], goal[1], 0.12])
            if np.linalg.norm(ee[:2] - goal[:2]) < 0.04:
                self.state = "LOWER"

        elif self.state == "LOWER":
            target = np.array([goal[0], goal[1], 0.03])
            if ee[2] < 0.05:
                gripper = -1.0  # release
            else:
                gripper = 1.0

        diff = target - ee
        dist = np.linalg.norm(diff)
        action_xyz = np.clip(diff / max(dist, 1e-4), -1.0, 1.0)
        return np.array([action_xyz[0], action_xyz[1], action_xyz[2], 0.0, gripper], dtype=np.float32)


class HeuristicCableController:
    """State-machine heuristic controller that reaches, grasps, and routes the deformable cable."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "APPROACH"
        self.grasp_timer = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ee = obs[12:15]
        cable_tip = obs[16:19]
        goal = obs[22:25]

        target = np.copy(ee)
        gripper = -1.0

        if self.state == "APPROACH":
            # Hover 6cm above cable tip
            target = np.array([cable_tip[0], cable_tip[1], cable_tip[2] + 0.06])
            if np.linalg.norm(ee[:2] - cable_tip[:2]) < 0.03 and np.abs(ee[2] - (cable_tip[2] + 0.06)) < 0.03:
                self.state = "DESCEND"

        elif self.state == "DESCEND":
            # Descend around cable tip
            target = np.array([cable_tip[0], cable_tip[1], cable_tip[2] + 0.006])
            if np.linalg.norm(ee - target) < 0.022:
                self.state = "GRASP"

        elif self.state == "GRASP":
            # Clasp Robotiq 2F-85 fingers firmly around the cable
            target = np.array([cable_tip[0], cable_tip[1], cable_tip[2] + 0.006])
            gripper = 1.0
            self.grasp_timer += 1
            if self.grasp_timer >= 14:
                self.state = "LIFT"

        elif self.state == "LIFT":
            # Lift cable slightly off the table
            gripper = 1.0
            target = np.array([ee[0], ee[1], 0.08])
            if ee[2] > 0.06:
                self.state = "PULL_TO_GOAL"

        elif self.state == "PULL_TO_GOAL":
            # Pull cable tip towards goal
            gripper = 1.0
            target = np.array([goal[0], goal[1], 0.05])
            if np.linalg.norm(cable_tip - goal) < 0.045:
                self.state = "RELEASE"

        elif self.state == "RELEASE":
            target = np.array([goal[0], goal[1], 0.025])
            gripper = -1.0  # release cable at goal

        diff = target - ee
        dist = np.linalg.norm(diff)
        action_xyz = np.clip(diff / max(dist, 1e-4), -1.0, 1.0)
        return np.array([action_xyz[0], action_xyz[1], action_xyz[2], 0.0, gripper], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Test WISERLI-ATAL Gymnasium environments.")
    parser.add_argument(
        "--env",
        type=str,
        default="WiserliAtalReach-v0",
        choices=[
            "WiserliAtalReach-v0",
            "WiserliAtalPickAndPlace-v0",
            "WiserliAtalCable-v0",
            "WiserliAtalReachJoint-v0",
            "WiserliAtalPickAndPlaceJoint-v0",
            "WiserliAtalCableJoint-v0",
        ],
        help="Gymnasium environment ID to test.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="heuristic",
        choices=["heuristic", "random"],
        help="Action selection policy (heuristic solves the task; random samples actions).",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument("--max_steps", type=int, default=120, help="Max steps per episode.")
    parser.add_argument("--render", action="store_true", help="Launch interactive 3D MuJoCo viewer.")
    parser.add_argument("--fps", type=float, default=20.0, help="Viewer playback frame rate.")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    print(f"\n===========================================================")
    print(f"Creating Gymnasium Environment: {args.env}")
    print(f"Render Mode: {render_mode} | Policy: {args.policy}")
    print(f"===========================================================\n")

    env = gym.make(args.env, render_mode=render_mode, max_episode_steps=args.max_steps)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}\n")

    sleep_dt = 1.0 / args.fps if args.render else 0.0
    pnp_controller = HeuristicPickAndPlaceController()
    cable_controller = HeuristicCableController()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep * 100 + 42)
        pnp_controller.reset()
        cable_controller.reset()
        ep_reward = 0.0
        success = False

        print(f"--- Starting Episode {ep + 1}/{args.episodes} ---")

        for step in range(args.max_steps):
            if args.policy == "heuristic":
                if args.env == "WiserliAtalReach-v0":
                    action = heuristic_reach_policy(obs)
                elif args.env == "WiserliAtalReachJoint-v0":
                    action = heuristic_reach_joint_policy(env, obs)
                elif "PickAndPlace" in args.env:
                    action = pnp_controller.get_action(obs)
                elif "Cable" in args.env:
                    action = cable_controller.get_action(obs)
                else:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if info.get("is_success", False):
                success = True

            if (step + 1) % 20 == 0 or terminated or truncated:
                dist_val = info.get("distance", info.get("dist_obj_to_goal", info.get("dist_tip_to_goal", 0.0)))
                print(f"  Step {step + 1:3d} | Reward: {reward:+.3f} | Total: {ep_reward:+.2f} | dist={dist_val:.4f}m")

            if args.render:
                time.sleep(sleep_dt)

            if terminated or truncated:
                break

        status = "SUCCESS" if success else "TRUNCATED"
        print(f"Episode {ep + 1} finished: {status} | Total Reward: {ep_reward:.2f} | Steps: {step + 1}\n")

    env.close()
    print("Environment closed cleanly.")

    # Clean OS-level exit to prevent Linux/WSL GLFW Mesa teardown segfault
    if args.render:
        os._exit(0)


if __name__ == "__main__":
    main()
