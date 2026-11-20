# WISERLI-ATAL-COBOTs

A complete, self-contained **ROS 2 (Jazzy + Gazebo Harmonic)** and **MuJoCo (3.x)** simulation & control package for the **WISERLI-ATAL-C1.1** 6-DOF (other versions to be added soon) collaborative robot arm, with an integrated **Robotiq 2F-85** parallel gripper and a wrist-mounted **RGB-D camera** (Intel RealSense D435i form factor).

Demo video of **WISERLI-ATAL-C1.1** 

<video src="media/wiserli-atal-c1.1-demo.m4v" width="600" controls>
  Your browser does not support the video tag.
</video>

## Key Features

- **Dual Simulator Support**: Runs out of the box in **MuJoCo 3.x** (Python / Linux / WSL / Windows) AND **ROS 2 Jazzy + Gazebo Harmonic** (`ros_gz_sim` + `gz_ros2_control`).
- **With & Without Gripper**: Every workflow (URDF, Xacro, ROS 2 launch, MuJoCo models, GUI, and IK scripts) supports both the bare 6-DOF arm and the full arm + Robotiq 2F-85 + wrist RGB-D setup.
- **Self-Contained Robotiq 2F-85**:
  - **ROS 2 / Gazebo**: Official visual `.dae` & collision `.stl` meshes, 5 synchronized `<mimic>` joints, and `parallel_gripper_action_controller/GripperActionController`.
  - **MuJoCo**: Official closed-loop 4-bar parallel linkage (`<equality>` + `<tendon>`), silicone pad contact physics, and single `0–255` `fingers_actuator`.
- **Wrist-Mounted RGB-D Camera**: RealSense D435i-style camera on `tool0` with live RGB + metric Depth rendering in MuJoCo and `rgbd_camera` + `ros_gz_bridge` (`/wrist_camera/image`, `/wrist_camera/depth_image`, `/wrist_camera/points`) in ROS 2.
- **Interactive Live Control GUI**: Unified Python GUI (`mujoco/interactive_gui.py`) with 3D orbit/pan/zoom viewport, live side-by-side wrist RGB + click-to-probe Depth map, and real-time sliders for all 6 joints + gripper.
- **Task-Space Control Scripts (`mjctrl`-style)**: Ready-to-run Differential IK (`diffik.py`), Nullspace DiffIK (`diffik_nullspace.py`), Operational Space Control (`opspace.py`), and an autonomous pick-and-place demo (`pick_and_place_diffik.py`).

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Cloning the Repository (Git LFS)](#1-cloning-the-repository-git-lfs)
3. [Quick Start: MuJoCo (Beginner Friendly)](#2-quick-start-mujoco-no-ros-required)
4. [Quick Start: ROS 2 Jazzy + Gazebo Harmonic](#3-quick-start-ros-2-jazzy--gazebo-harmonic)
5. [URDF / Xacro & MJCF Files Reference](#4-urdf--xacro--mjcf-files-reference)
6. [Pushing This Workspace to GitHub (First Time)](#5-pushing-this-workspace-to-github-first-time)
7. [Troubleshooting](#6-troubleshooting)
8. [License & Attribution](#license--attribution)

---

## Repository Structure

```text
atal_cobot/
├── CMakeLists.txt                  # ROS 2 ament_cmake build configuration
├── package.xml                     # ROS 2 package manifest (format 3)
├── requirements.txt                # Python dependencies for MuJoCo & GUI
├── .gitattributes                  # Git LFS rules for 3D mesh files (*.STL, *.stl, *.dae)
├── .gitignore
├── LICENSE                         # BSD-3-Clause
├── config/
│   ├── ros2_controllers.yaml       # joint_state_broadcaster, arm_controller, gripper_controller
│   ├── gz_bridge.yaml              # Gazebo Harmonic <-> ROS 2 bridge (/clock + RGB-D camera)
│   └── joint_names_WISERLI-ATAL-C1.1.yaml
├── env-hooks/
│   └── wiserli_atal_description.dsv.in  # Auto-sets GZ_SIM_RESOURCE_PATH
├── launch/
│   ├── display.launch.py           # RViz2 + joint_state_publisher_gui
│   ├── control.launch.py           # ros2_control mock hardware + RViz2 (no Gazebo)
│   └── gazebo.launch.py            # Gazebo Harmonic + gz_ros2_control + RGB-D bridge + RViz2
├── meshes/
│   ├── *.STL                       # Original SolidWorks arm meshes
│   └── robotiq_2f_85/              # Bundled Robotiq 2F-85 visual (.dae) & collision (.stl)
├── mujoco/
│   ├── wiserli_atal.xml            # MJCF: 6-DOF arm only (without gripper)
│   ├── scene.xml                   # Scene: arm only + mocap target
│   ├── wiserli_atal_2f85.xml       # MJCF: arm + Robotiq 2F-85 + wrist RGB-D camera
│   ├── scene_2f85.xml              # Scene: arm + gripper + wrist camera + grasp cube + target
│   ├── robotiq_2f85.xml            # Standalone Robotiq 2F-85 MJCF
│   ├── interactive_gui.py          # Live 3D + RGB + Depth + joint/gripper GUI
│   ├── scripts/
│   │   ├── diffik.py               # Differential IK (circle & interactive mocap)
│   │   ├── diffik_nullspace.py     # Nullspace posture DiffIK
│   │   ├── opspace.py              # Operational Space Control (Cartesian impedance torque)
│   │   └── pick_and_place_diffik.py # Autonomous DiffIK pick-and-place demo
│   └── assets/
│       ├── arm/                    # MuJoCo-ready Binary STLs (< 50k faces each)
│       └── robotiq_2f85/           # Official DeepMind mujoco_menagerie 2F-85 STLs
├── rviz/
│   └── display.rviz                # Pre-configured RViz2 layout
├── urdf/
│   ├── WISERLI-ATAL-C1.1.urdf      # Original unmodified SolidWorks URDF (arm only)
│   ├── wiserli_atal_with_gripper.urdf  # Standalone ROS 2 URDF (arm + gripper + RGB-D)
│   ├── wiserli_atal.urdf.xacro     # Main modular ROS 2 Xacro entry point
│   ├── wiserli_atal_arm.xacro
│   ├── robotiq_2f_85_macro.xacro
│   ├── wrist_rgbd_camera.xacro
│   ├── wiserli_atal.ros2_control.xacro
│   └── wiserli_atal.gazebo.xacro
└── worlds/
    └── empty.sdf                   # Gazebo Harmonic world with sensors plugin & grasp cube
```

---

## 1. Cloning the Repository (Git LFS)

Because the original SolidWorks actuator meshes (`meshes/Actuator_4..6.STL`) are > 100 MB, this repository uses **Git LFS** for `.STL`, `.stl`, and `.dae` files.

Install Git LFS once, then clone:

```bash
# Ubuntu / Debian / WSL
sudo apt update && sudo apt install -y git-lfs

# Initialize Git LFS and clone
git lfs install
git clone https://github.com/wiserli/atal-cobot.git atal_cobot
cd atal_cobot
```

*(If you already cloned without Git LFS, run `git lfs pull` inside the repo to download the actual 3D meshes.)*

---

## 2. Quick Start: MuJoCo (No ROS Required)

Works on **Linux, WSL2 (WSLg), Windows, and macOS** with Python 3.10+.

### Step 1: Create a Python Environment & Install Requirements

```bash
# Using Conda (recommended)
conda create -n wiserli python=3.11 -y
conda activate wiserli
pip install -r requirements.txt

# OR using standard venv
python3 -m venv .venv
source .venv/bin/activate        # On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*(Ubuntu/WSL system Python users: install Tkinter if needed via `sudo apt install -y python3-tk`.)*

### Step 2: Launch the Interactive Live GUI (3D + RGB + Depth + Joint Sliders)

```bash
# Full robot: 6-DOF arm + Robotiq 2F-85 gripper + live wrist RGB & Depth camera feeds
python mujoco/interactive_gui.py

# Bare 6-DOF arm without gripper
python mujoco/interactive_gui.py --no-gripper
```

**GUI Controls:**
- **3D Viewport**: Left-drag = Orbit, Right-drag (or Shift+Left) = Pan, Scroll = Zoom.
- **Depth Feed**: Click anywhere on the depth image to measure exact distance in meters.
- **Right Panel**: Move `joint_1`–`joint_6` live, open/close the Robotiq 2F-85 (`0`–`255`), or test 1-click **Pre-Grasp** → **Grasp Pose** → **Close (230)** on the test cube.

### Step 3: Run Native MuJoCo Viewer

```bash
# With Robotiq 2F-85 + wrist RGB-D camera (press '[' or ']' to switch to wrist_rgbd view)
python -m mujoco.viewer --mjcf=mujoco/scene_2f85.xml

# Without gripper
python -m mujoco.viewer --mjcf=mujoco/scene.xml
```

### Step 4: Run Task-Space Control Scripts (`mujoco/scripts/`)

```bash
# Differential IK following a circle (default: with gripper)
python mujoco/scripts/diffik.py

# Interactive mocap tracking (double-click red target box, then Ctrl + drag in viewer)
python mujoco/scripts/diffik.py --mode mocap

# Nullspace Differential IK (add --no-gripper for bare arm)
python mujoco/scripts/diffik_nullspace.py
python mujoco/scripts/diffik_nullspace.py --no-gripper

# Operational Space Control (Cartesian impedance torque control)
python mujoco/scripts/opspace.py
python mujoco/scripts/opspace.py --mode circle

# Autonomous DiffIK + Robotiq 2F-85 Pick-and-Place Demo
python mujoco/scripts/pick_and_place_diffik.py
```

---

## 3. Quick Start: ROS 2 Jazzy + Gazebo Harmonic

Requires **Ubuntu 24.04 + ROS 2 Jazzy Jalisco**.

### Step 1: Place Inside a Colcon Workspace & Build

```bash
mkdir -p ~/ros2_ws/src
cp -r /path/to/atal_cobot ~/ros2_ws/src/atal_cobot
cd ~/ros2_ws

# Install all ROS 2 / Gazebo Harmonic / ros2_control dependencies
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y

# Build and source
colcon build --packages-select atal_cobot --symlink-install
source install/setup.bash
```

### Step 2: Launch Visualization, Mock Control, or Gazebo Harmonic

```bash
# 1. RViz2 + interactive joint sliders (arm + synchronized gripper linkage)
ros2 launch atal_cobot display.launch.py

# 2. ros2_control mock hardware + RViz2 (no Gazebo required)
ros2 launch atal_cobot control.launch.py

# 3. Full Gazebo Harmonic simulation + gz_ros2_control + wrist RGB-D camera + RViz2
ros2 launch atal_cobot gazebo.launch.py
```

Useful launch arguments:
- `include_gripper:=false` — Load the bare 6-DOF arm without the Robotiq 2F-85.
- `rviz:=false` — Disable RViz2.
- `gz_args:="-r -s"` — Run Gazebo Harmonic headless.

### Step 3: Send ROS 2 Action Commands (Arm & Gripper)

```bash
# Close Robotiq 2F-85 gripper (0.0 = open 85 mm, 0.79 = closed)
ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/ParallelGripperCommand \
  "{command: {name: ['robotiq_85_left_knuckle_joint'], position: [0.79]}}"

# Open Robotiq 2F-85 gripper
ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/ParallelGripperCommand \
  "{command: {name: ['robotiq_85_left_knuckle_joint'], position: [0.0]}}"

# Move the 6-DOF arm via JointTrajectoryController
ros2 action send_goal /arm_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'], points: [{positions: [0.3, -0.4, 0.5, 0.0, 0.4, 0.0], time_from_start: {sec: 3, nanosec: 0}}]}}"
```

### Bridged Wrist RGB-D ROS 2 Topics (in Gazebo Harmonic)

- `/wrist_camera/image` (`sensor_msgs/msg/Image`)
- `/wrist_camera/depth_image` (`sensor_msgs/msg/Image`)
- `/wrist_camera/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/wrist_camera/points` (`sensor_msgs/msg/PointCloud2`, frame: `wrist_camera_optical_frame`)

---

## 4. URDF / Xacro & MJCF Files Reference

| File | Purpose |
| --- | --- |
| `urdf/WISERLI-ATAL-C1.1.urdf` | Original unmodified SolidWorks-exported URDF (arm only, 13 links) |
| `urdf/wiserli_atal_with_gripper.urdf` | Standalone ROS 2 URDF with arm + Robotiq 2F-85 + wrist RGB-D camera (28 links) |
| `urdf/wiserli_atal.urdf.xacro` | Main modular ROS 2 Xacro (`include_gripper`, `include_camera`, `use_gazebo`) |
| `mujoco/scene.xml` / `wiserli_atal.xml` | MuJoCo 6-DOF arm only (with `tool0` / `attachment_site`) |
| `mujoco/scene_2f85.xml` / `wiserli_atal_2f85.xml` | MuJoCo 6-DOF arm + Robotiq 2F-85 + wrist RGB-D camera + grasp cube |

---

## 5. Pushing This Workspace to GitHub (First Time)

If you are publishing this folder to a new GitHub repository, make sure **Git LFS is initialized before your first commit** so the large `.STL` files upload cleanly:

```bash
git lfs install
git init -b main
git add .gitattributes .gitignore
git add .
git commit -m "Initial commit: WISERLI-ATAL-C1.1 ROS 2 Jazzy + MuJoCo + Robotiq 2F-85 + Wrist RGB-D"
git remote add origin https://github.com/wiserli/atal-cobot.git
git push -u origin main
```

---

## 6. Troubleshooting

- **`Error: number of faces should be between 1 and 200000 in STL file` in MuJoCo**: Always load the models under `mujoco/` (which use decimated `< 50k`-face Binary STLs in `mujoco/assets/arm/`), not the raw SolidWorks STLs in `meshes/`.
- **Meshes missing after `git clone`**: Run `git lfs install && git lfs pull` to fetch the LFS-tracked `.STL` / `.stl` / `.dae` binaries.
- **`ModuleNotFoundError: No module named 'tkinter'` on Ubuntu/WSL**: Install via `sudo apt install -y python3-tk` (or use a Conda environment).
- **Gazebo Harmonic cannot find meshes**: Make sure you ran `source install/setup.bash` after `colcon build` so `env-hooks/wiserli_atal_description.dsv.in` exports `GZ_SIM_RESOURCE_PATH`.

---

## License & Attribution

Released under the **BSD-3-Clause License** (see [LICENSE](LICENSE)).

- Robotiq 2F-85 ROS 2 assets adapted from PickNik Robotics [`ros2_robotiq_gripper`](https://github.com/PickNikRobotics/ros2_robotiq_gripper) (`BSD-3-Clause`).
- Robotiq 2F-85 MuJoCo MJCF & STL assets adapted from Google DeepMind [`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie) (`BSD-3-Clause`).
- Task-space control scripts in `mujoco/scripts/` adapted from Kevin Zakka [`mjctrl`](https://github.com/kevinzakka/mjctrl) (`MIT`).
