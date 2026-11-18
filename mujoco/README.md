# MuJoCo Models for WISERLI-ATAL-C1.1 (With & Without Robotiq 2F-85)

Native MuJoCo (MJCF) models and MuJoCo-compatible Binary STL assets (`< 50,000` faces per mesh, well under MuJoCo's `200,000`-face STL limit).

## Files

| File | Description |
| --- | --- |
| `wiserli_atal.xml` | 6-DOF WISERLI-ATAL-C1.1 arm **without** gripper (includes `tool0` / `attachment_site`) |
| `scene.xml` | Ready-to-run scene for the arm **without** gripper |
| `wiserli_atal_2f85.xml` | 6-DOF arm **with** fully functional closed-loop 4-bar linkage Robotiq 2F-85 gripper |
| `scene_2f85.xml` | Ready-to-run scene for the arm **with** Robotiq 2F-85 + graspable 40 mm test cube |
| `robotiq_2f85.xml` | Standalone Robotiq 2F-85 gripper MJCF model |
| `interactive_gui.py` | Live interactive GUI with 3D orbit view, side-by-side wrist RGB + Depth feeds, and all joint/gripper sliders |
| `scripts/` | `mjctrl`-style task-space controllers (`diffik.py`, `diffik_nullspace.py`, `opspace.py`, `pick_and_place_diffik.py`) |
| `assets/arm/` | MuJoCo-compatible Binary STL meshes for the arm (`< 50k` faces each) |
| `assets/robotiq_2f85/` | Official DeepMind `mujoco_menagerie` Robotiq 2F-85 STL assets (`BSD-3-Clause`) |

## Interactive Control + Live RGB-D GUI

Launch the unified live control dashboard (interactive 3D view + side-by-side wrist RGB & metric depth views + live sliders for `joint_1`..`joint_6` and the Robotiq 2F-85 gripper):

```bash
# Arm WITH Robotiq 2F-85 gripper + live wrist RGB & Depth feeds
python mujoco/interactive_gui.py

# Arm WITHOUT gripper
python mujoco/interactive_gui.py --no-gripper
```

## Quick Launch in Native MuJoCo Viewer

Install MuJoCo (`pip install mujoco`) and launch either scene directly:

```bash
# Arm WITHOUT gripper
python -m mujoco.viewer --mjcf=mujoco/scene.xml

# Arm WITH Robotiq 2F-85 gripper + grasp cube
python -m mujoco.viewer --mjcf=mujoco/scene_2f85.xml
```

In the interactive viewer (`Control` panel on the right):
- `joint_1_act` .. `joint_6_act`: Target joint angles in radians.
- `fingers_actuator`: `0` = fully open (85 mm), `255` = fully closed.

## Python Example

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mujoco/scene_2f85.xml")
data = mujoco.MjData(model)

# Reset to home keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)

# Command arm pose + close Robotiq 2F-85 gripper
data.ctrl[:6] = [0.2, -0.3, 0.4, 0.0, 0.3, 0.0]
data.ctrl[6] = 200.0  # 0 (open) .. 255 (closed)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```
