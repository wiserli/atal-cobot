# WISERLI-ATAL-C1.1 Gymnasium Reinforcement Learning Environments

Standard [Farama Gymnasium](https://gymnasium.farama.org/) environments for the **WISERLI-ATAL-C1.1 6-DOF Cobot** equipped with the **Robotiq 2F-85 parallel gripper**, **wrist 6-axis Force/Torque sensor**, and **MuJoCo deformable cable elasticity physics**.

Designed for reinforcement learning (PPO, SAC, TD3), imitation learning (ACT, Diffusion Policy), and robotic manipulation research. Compatible with **Stable-Baselines3**, **Gymnasium**, and **Hugging Face LeRobot**.

---

## 1. Registered Environments

| Environment ID | Task Description | Action Space | Default Max Steps |
| :--- | :--- | :--- | :--- |
| `WiserliAtalReach-v0` | Reach randomized 3D goal marker | Task-Space $\Delta x, \Delta y, \Delta z, \Delta \text{yaw}, \text{gripper}$ (5D) | 150 |
| `WiserliAtalPickAndPlace-v0` | Reach, grasp 35mm cube on table, lift and place at goal | Task-Space $\Delta x, \Delta y, \Delta z, \Delta \text{yaw}, \text{gripper}$ (5D) | 200 |
| `WiserliAtalCable-v0` | Deformable cable grasping, routing, and pulling with tension sensing | Task-Space $\Delta x, \Delta y, \Delta z, \Delta \text{yaw}, \text{gripper}$ (5D) | 150 |
| `WiserliAtalReachJoint-v0` | Reaching task | Joint displacements $\Delta q_{1..6}, \text{gripper}$ (7D) | 150 |
| `WiserliAtalPickAndPlaceJoint-v0` | Pick-and-place task | Joint displacements $\Delta q_{1..6}, \text{gripper}$ (7D) | 200 |
| `WiserliAtalCableJoint-v0` | Deformable cable manipulation | Joint displacements $\Delta q_{1..6}, \text{gripper}$ (7D) | 150 |

---

## 2. Quickstart

### Test Environment with Interactive Viewer
```bash
# Test Reach task with heuristic target-tracking policy:
python mujoco/scripts/test_gym_env.py --env WiserliAtalReach-v0 --policy heuristic --render

# Test Pick-and-Place task:
python mujoco/scripts/test_gym_env.py --env WiserliAtalPickAndPlace-v0 --policy heuristic --render

# Test Deformable Cable manipulation task:
python mujoco/scripts/test_gym_env.py --env WiserliAtalCable-v0 --policy heuristic --render
```

### Python API Usage
```python
import sys
sys.path.insert(0, "mujoco")

import gymnasium as gym
import gym_envs

# Create deformable cable manipulation environment with 3D viewer
env = gym.make("WiserliAtalCable-v0", render_mode="human")

obs, info = env.reset(seed=42)
for _ in range(150):
    action = env.action_space.sample()  # Replace with trained policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## 3. Deformable Cable Physics (`mujoco.elasticity.cable`)

The cable environment uses MuJoCo's native cable plugin:
```xml
<extension>
  <plugin plugin="mujoco.elasticity.cable"/>
</extension>
```
- **Physics**: Real-time finite-element Cosserat rod formulation computing bending, torsion (twist), and axial elongation forces.
- **Tension & Contact Sensing**: The wrist 6-axis Force/Torque sensor detects tension while pulling the wire, applying a gentle compliance penalty if pulling with excessive force ($> 25\text{ N}$) to train realistic manipulation.

---

## 4. Reinforcement Learning Training

Train an RL agent with Stable-Baselines3:
```bash
# Train PPO on the reaching task:
python mujoco/scripts/train_reach_rl.py --algo ppo --timesteps 50000

# Train on the cable manipulation task:
python mujoco/scripts/train_reach_rl.py --env WiserliAtalCable-v0 --algo ppo --timesteps 50000
```
