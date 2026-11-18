# MuJoCo Control Scripts (`mjctrl`-style) for WISERLI-ATAL-C1.1

Task-space control scripts inspired by [kevinzakka/mjctrl](https://github.com/kevinzakka/mjctrl), adapted for the 6-DOF **WISERLI-ATAL-C1.1** robot arm both **with** (`scene_2f85.xml`, controlling `pinch` TCP) and **without** (`scene.xml`, controlling `attachment_site`) the Robotiq 2F-85 gripper.

## Scripts

| Script | Controller | Default Mode |
| --- | --- | --- |
| `diffik.py` | Damped least-squares Differential IK | Circular trajectory (`--mode circle`) |
| `diffik_nullspace.py` | Differential IK + nullspace posture control toward `home` | Draggable mocap target (`--mode mocap`) |
| `opspace.py` | Operational Space Control (Cartesian impedance torque control + gravity/Coriolis comp) | Draggable mocap target (`--mode mocap`) |
| `pick_and_place_diffik.py` | Autonomous DiffIK + Robotiq 2F-85 pick-and-place state machine | Picks & places `grasp_cube` |

## Usage

```bash
# 1. Differential IK circular tracking (with Robotiq 2F-85 gripper)
python mujoco/scripts/diffik.py

# 2. Differential IK tracking the draggable red mocap box (double-click box, Ctrl + drag)
python mujoco/scripts/diffik.py --mode mocap

# 3. Nullspace Differential IK (with gripper, or add --no-gripper for bare arm)
python mujoco/scripts/diffik_nullspace.py
python mujoco/scripts/diffik_nullspace.py --no-gripper

# 4. Operational Space Control (Cartesian impedance torque control)
python mujoco/scripts/opspace.py
python mujoco/scripts/opspace.py --mode circle

# 5. Autonomous Pick-and-Place with Robotiq 2F-85
python mujoco/scripts/pick_and_place_diffik.py
```

### Interactive Mocap Target Tip
In `mujoco.viewer`, **double-click** the translucent red `target` cube, then hold **`Ctrl` + Right-drag** to translate it or **`Ctrl` + Left-drag** to rotate it in 3D space.
