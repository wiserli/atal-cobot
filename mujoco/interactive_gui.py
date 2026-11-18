#!/usr/bin/env python3
"""
Interactive Live MuJoCo GUI for WISERLI-ATAL-C1.1 + Robotiq 2F-85 + Wrist RGB-D Camera.

Features:
  - Live interactive 3D MuJoCo viewport (Left-drag: Orbit, Right/Shift-drag: Pan, Scroll: Zoom)
  - Live side-by-side Wrist Camera feeds: Normal RGB view + Colorized Metric Depth view
  - Click anywhere on the Depth feed to inspect exact metric distance at that pixel
  - Live sliders for all 6 arm joints (joint_1..joint_6) + Robotiq 2F-85 gripper (0..255)
  - 1-click presets (Home, Pre-Grasp, Grasp, Reset Scene) + Open/Half/Close gripper + Demo Wave
  - Supports both full robot with gripper/camera (scene_2f85.xml) and bare arm (--no-gripper)

Usage:
  python mujoco/interactive_gui.py
  python mujoco/interactive_gui.py --no-gripper
  python mujoco/interactive_gui.py --mjcf path/to/scene_2f85.xml
"""

import argparse
import math
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageTk


# Color palette (modern dark robotics UI)
BG_DARK = "#16181d"
BG_PANEL = "#1f2229"
BG_CARD = "#262a33"
FG_TEXT = "#e6eaf0"
FG_MUTED = "#9aa3b2"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#22c55e"
ACCENT_ORANGE = "#f59e0b"
ACCENT_RED = "#ef4444"


class WiserliMuJoCoGUI:
    def __init__(self, mjcf_path: str, target_fps: int = 30):
        self.mjcf_path = os.path.abspath(mjcf_path)
        self.target_fps = max(10, min(60, target_fps))
        self.frame_ms = int(1000 / self.target_fps)

        # Load MuJoCo model & data
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # Detect optional features (gripper + wrist camera + TCP site)
        self.wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgbd"
        )
        self.has_wrist_cam = self.wrist_cam_id >= 0

        self.gripper_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
        )
        self.has_gripper = self.gripper_act_id >= 0

        self.tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch"
        )
        if self.tcp_site_id < 0:
            self.tcp_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
            )

        # Collect 6 arm joint metadata
        self.arm_joints = []
        for idx in range(1, 7):
            jname = f"joint_{idx}"
            aname = f"joint_{idx}_act"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if jid >= 0 and aid >= 0:
                lo, hi = self.model.jnt_range[jid]
                self.arm_joints.append(
                    {"name": jname, "jid": jid, "aid": aid, "min": float(lo), "max": float(hi)}
                )

        # Render dimensions
        self.main_w, self.main_h = 640, 440
        self.cam_w, self.cam_h = 316, 237

        # Ensure offscreen framebuffer is large enough
        self.main_w = min(self.main_w, self.model.vis.global_.offwidth)
        self.main_h = min(self.main_h, self.model.vis.global_.offheight)

        # Create MuJoCo offscreen renderers (single-threaded on main loop -> WSLg safe)
        self.main_renderer = mujoco.Renderer(
            self.model, height=self.main_h, width=self.main_w
        )
        self.wrist_renderer = (
            mujoco.Renderer(self.model, height=self.cam_h, width=self.cam_w)
            if self.has_wrist_cam
            else None
        )

        # Interactive 3D free camera & scene option
        self.free_cam = mujoco.MjvCamera()
        self.scene_opt = mujoco.MjvOption()
        self._set_camera_preset("3/4")

        # Mouse drag state for 3D viewport
        self._drag_mode = None
        self._last_mouse = (0, 0)

        # Simulation state
        self.paused = False
        self.demo_wave = False
        self.demo_t0 = 0.0
        self.active_main_cam = "free"  # "free" or "wrist_rgbd"
        self.depth_max_m = 1.5
        self.depth_probe_px = (self.cam_w // 2, self.cam_h // 2)
        self.last_rgb = None
        self.last_depth_vis = None
        self.last_depth_raw = None

        # FPS tracking
        self._fps_t = time.perf_counter()
        self._fps_frames = 0
        self.fps_value = 0.0

        # PhotoImage references (kept alive to prevent GC flicker)
        self._img_main = None
        self._img_rgb = None
        self._img_depth = None
        self._after_id = None
        self._closing = False

        # Build Tkinter UI
        self.root = tk.Tk()
        self.root.title("WISERLI-ATAL-C1.1 + Robotiq 2F-85 — Live MuJoCo Control & RGB-D GUI")
        self.root.configure(bg=BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._setup_styles()
        self._build_ui()
        self._sync_sliders_from_ctrl()

    def _setup_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("TLabel", background=BG_DARK, foreground=FG_TEXT, font=("DejaVu Sans", 9))
        style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG_TEXT, font=("DejaVu Sans", 9))
        style.configure("Card.TLabel", background=BG_CARD, foreground=FG_TEXT, font=("DejaVu Sans", 9))
        style.configure("Muted.TLabel", background=BG_CARD, foreground=FG_MUTED, font=("DejaVu Sans", 8))
        style.configure("Header.TLabel", background=BG_PANEL, foreground=FG_TEXT, font=("DejaVu Sans", 10, "bold"))
        style.configure("Title.TLabel", background=BG_DARK, foreground=FG_TEXT, font=("DejaVu Sans", 11, "bold"))
        style.configure(
            "Horizontal.TScale",
            background=BG_CARD,
            troughcolor="#121419",
            sliderlength=16,
        )

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        # ---- LEFT COLUMN: 3D View + Dual RGB-D Wrist Camera Feeds ----
        left_col = ttk.Frame(container)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # 3D Viewport toolbar
        tb = ttk.Frame(left_col, style="Panel.TFrame", padding=(8, 5))
        tb.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(tb, text="3D Simulation View", style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        for label, key in [("3/4", "3/4"), ("Front", "front"), ("Side", "side"), ("Top", "top")]:
            self._make_btn(tb, label, lambda k=key: self._select_main_cam(k), width=6).pack(
                side=tk.LEFT, padx=2
            )
        if self.has_wrist_cam:
            self._make_btn(
                tb, "Wrist", lambda: self._select_main_cam("wrist_rgbd"), width=6, bg="#2563eb"
            ).pack(side=tk.LEFT, padx=2)

        self.var_contacts = tk.BooleanVar(value=False)
        self.var_frames = tk.BooleanVar(value=False)
        tk.Checkbutton(
            tb,
            text="Contacts",
            variable=self.var_contacts,
            bg=BG_PANEL,
            fg=FG_TEXT,
            selectcolor=BG_DARK,
            activebackground=BG_PANEL,
            activeforeground=FG_TEXT,
            font=("DejaVu Sans", 8),
        ).pack(side=tk.RIGHT, padx=4)
        tk.Checkbutton(
            tb,
            text="Frames",
            variable=self.var_frames,
            bg=BG_PANEL,
            fg=FG_TEXT,
            selectcolor=BG_DARK,
            activebackground=BG_PANEL,
            activeforeground=FG_TEXT,
            font=("DejaVu Sans", 8),
        ).pack(side=tk.RIGHT, padx=4)

        # 3D Viewport image label
        self.lbl_main = tk.Label(
            left_col,
            bg="#0d0f14",
            width=self.main_w,
            height=self.main_h,
            bd=1,
            relief=tk.SOLID,
            cursor="fleur",
        )
        self.lbl_main.pack(pady=(0, 6))
        self.lbl_main.bind("<ButtonPress-1>", self._on_mouse_down)
        self.lbl_main.bind("<ButtonPress-3>", self._on_mouse_down)
        self.lbl_main.bind("<B1-Motion>", self._on_mouse_drag)
        self.lbl_main.bind("<B3-Motion>", self._on_mouse_drag)
        self.lbl_main.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.lbl_main.bind("<ButtonRelease-3>", self._on_mouse_up)
        self.lbl_main.bind("<MouseWheel>", self._on_mouse_wheel)
        self.lbl_main.bind("<Button-4>", self._on_mouse_wheel_linux)
        self.lbl_main.bind("<Button-5>", self._on_mouse_wheel_linux)

        # Dual Wrist Camera Feed Row (RGB + Metric Depth)
        if self.has_wrist_cam:
            cam_row = ttk.Frame(left_col)
            cam_row.pack(fill=tk.X)

            # RGB feed card
            rgb_card = ttk.Frame(cam_row, style="Panel.TFrame", padding=4)
            rgb_card.pack(side=tk.LEFT, padx=(0, 6))
            ttk.Label(rgb_card, text="Wrist Camera — Normal RGB", style="Header.TLabel").pack(
                anchor=tk.W, pady=(0, 3)
            )
            self.lbl_rgb = tk.Label(
                rgb_card, bg="#0d0f14", width=self.cam_w, height=self.cam_h, bd=1, relief=tk.SOLID
            )
            self.lbl_rgb.pack()

            # Depth feed card
            depth_card = ttk.Frame(cam_row, style="Panel.TFrame", padding=4)
            depth_card.pack(side=tk.LEFT)
            d_hdr = ttk.Frame(depth_card, style="Panel.TFrame")
            d_hdr.pack(fill=tk.X, pady=(0, 3))
            ttk.Label(d_hdr, text="Wrist Camera — Depth (click to probe)", style="Header.TLabel").pack(
                side=tk.LEFT
            )
            self.lbl_depth = tk.Label(
                depth_card,
                bg="#0d0f14",
                width=self.cam_w,
                height=self.cam_h,
                bd=1,
                relief=tk.SOLID,
                cursor="crosshair",
            )
            self.lbl_depth.pack()
            self.lbl_depth.bind("<Button-1>", self._on_depth_click)

        # ---- RIGHT COLUMN: Joint Sliders + Gripper + Presets + Telemetry ----
        right_col = ttk.Frame(container, style="Panel.TFrame", padding=10, width=360)
        right_col.pack(side=tk.RIGHT, fill=tk.Y)
        right_col.pack_propagate(False)

        ttk.Label(right_col, text="Robot Joint Controls (Live)", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, 6)
        )

        # 6 Arm Joint Sliders
        self.joint_vars = {}
        self.joint_readouts = {}
        joints_card = ttk.Frame(right_col, style="Card.TFrame", padding=8)
        joints_card.pack(fill=tk.X, pady=(0, 8))

        for jinfo in self.arm_joints:
            jname = jinfo["name"]
            row = ttk.Frame(joints_card, style="Card.TFrame")
            row.pack(fill=tk.X, pady=3)

            hdr = ttk.Frame(row, style="Card.TFrame")
            hdr.pack(fill=tk.X)
            ttk.Label(hdr, text=jname, style="Card.TLabel", font=("DejaVu Sans", 9, "bold")).pack(
                side=tk.LEFT
            )
            ro = ttk.Label(hdr, text="cmd: +0.00 | act: +0.00 rad", style="Muted.TLabel")
            ro.pack(side=tk.RIGHT)
            self.joint_readouts[jname] = ro

            var = tk.DoubleVar(value=0.0)
            self.joint_vars[jname] = var
            scale = ttk.Scale(
                row,
                from_=jinfo["min"],
                to=jinfo["max"],
                variable=var,
                orient=tk.HORIZONTAL,
                command=lambda val, aid=jinfo["aid"]: self._on_joint_slider(aid, val),
            )
            scale.pack(fill=tk.X, pady=(1, 0))

        # Robotiq 2F-85 Gripper Control Card
        self.gripper_var = tk.DoubleVar(value=0.0)
        self.gripper_readout = None
        if self.has_gripper:
            ttk.Label(right_col, text="Robotiq 2F-85 Gripper", style="Header.TLabel").pack(
                anchor=tk.W, pady=(4, 4)
            )
            g_card = ttk.Frame(right_col, style="Card.TFrame", padding=8)
            g_card.pack(fill=tk.X, pady=(0, 8))

            g_hdr = ttk.Frame(g_card, style="Card.TFrame")
            g_hdr.pack(fill=tk.X)
            ttk.Label(g_hdr, text="Gripper (0=Open, 255=Closed)", style="Card.TLabel").pack(
                side=tk.LEFT
            )
            self.gripper_readout = ttk.Label(g_hdr, text="0", style="Muted.TLabel")
            self.gripper_readout.pack(side=tk.RIGHT)

            ttk.Scale(
                g_card,
                from_=0.0,
                to=255.0,
                variable=self.gripper_var,
                orient=tk.HORIZONTAL,
                command=self._on_gripper_slider,
            ).pack(fill=tk.X, pady=(3, 6))

            g_btns = ttk.Frame(g_card, style="Card.TFrame")
            g_btns.pack(fill=tk.X)
            self._make_btn(g_btns, "Open (0)", lambda: self._set_gripper(0.0), bg="#15803d").pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3)
            )
            self._make_btn(g_btns, "Half (128)", lambda: self._set_gripper(128.0), bg="#b45309").pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=3
            )
            self._make_btn(g_btns, "Close (230)", lambda: self._set_gripper(230.0), bg="#b91c1c").pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0)
            )

        # Depth Range Slider (if wrist camera present)
        if self.has_wrist_cam:
            d_card = ttk.Frame(right_col, style="Card.TFrame", padding=8)
            d_card.pack(fill=tk.X, pady=(0, 8))
            dh = ttk.Frame(d_card, style="Card.TFrame")
            dh.pack(fill=tk.X)
            ttk.Label(dh, text="Depth Colormap Max Range", style="Card.TLabel").pack(side=tk.LEFT)
            self.lbl_depth_max = ttk.Label(dh, text="1.50 m", style="Muted.TLabel")
            self.lbl_depth_max.pack(side=tk.RIGHT)
            self.depth_max_var = tk.DoubleVar(value=self.depth_max_m)
            ttk.Scale(
                d_card,
                from_=0.3,
                to=3.0,
                variable=self.depth_max_var,
                orient=tk.HORIZONTAL,
                command=self._on_depth_range,
            ).pack(fill=tk.X, pady=(2, 0))

        # Pose Presets & Simulation Controls
        ttk.Label(right_col, text="Presets & Simulation", style="Header.TLabel").pack(
            anchor=tk.W, pady=(2, 4)
        )
        p_card = ttk.Frame(right_col, style="Card.TFrame", padding=8)
        p_card.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(p_card, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=(0, 4))
        self._make_btn(row1, "Home", self._preset_home, bg="#334155").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3)
        )
        if self.has_gripper:
            self._make_btn(row1, "Pre-Grasp", self._preset_pregrasp, bg="#1d4ed8").pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=3
            )
            self._make_btn(row1, "Grasp Pose", self._preset_grasp, bg="#7c3aed").pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0)
            )

        row2 = ttk.Frame(p_card, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=(0, 4))
        self.btn_pause = self._make_btn(row2, "Pause Sim", self._toggle_pause, bg="#475569")
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.btn_demo = self._make_btn(row2, "Demo Wave", self._toggle_demo, bg="#0f766e")
        self.btn_demo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self._make_btn(row2, "Reset All", self._reset_scene, bg="#991b1b").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0)
        )

        if self.has_wrist_cam:
            self._make_btn(p_card, "Save RGB + Depth Snapshot", self._save_snapshot, bg="#374151").pack(
                fill=tk.X
            )

        # Live Telemetry Box
        t_card = ttk.Frame(right_col, style="Card.TFrame", padding=8)
        t_card.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(t_card, text="Live Telemetry", style="Card.TLabel", font=("DejaVu Sans", 9, "bold")).pack(
            anchor=tk.W
        )
        self.lbl_telemetry = ttk.Label(
            t_card,
            text="Sim: 0.00 s | FPS: 0.0\nTCP: (0.000, 0.000, 0.000) m\nDepth probe: --",
            style="Muted.TLabel",
            justify=tk.LEFT,
        )
        self.lbl_telemetry.pack(anchor=tk.W, pady=(3, 0))

        # Status/help footer
        ttk.Label(
            right_col,
            text="3D View: Left-drag=Orbit | Right-drag=Pan | Scroll=Zoom",
            style="Panel.TLabel",
            font=("DejaVu Sans", 7),
        ).pack(side=tk.BOTTOM, anchor=tk.W)

    def _make_btn(self, parent, text, cmd, width=None, bg="#334155"):
        kwargs = {
            "text": text,
            "command": cmd,
            "bg": bg,
            "fg": FG_TEXT,
            "activebackground": "#475569",
            "activeforeground": "#ffffff",
            "relief": tk.FLAT,
            "font": ("DejaVu Sans", 8, "bold"),
            "padx": 6,
            "pady": 4,
            "cursor": "hand2",
        }
        if width is not None:
            kwargs["width"] = width
        return tk.Button(parent, **kwargs)

    # ---- Camera & Mouse Handlers ----
    def _set_camera_preset(self, preset: str):
        self.free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if preset == "3/4":
            self.free_cam.lookat[:] = [0.22, 0.0, 0.25]
            self.free_cam.distance = 1.25
            self.free_cam.azimuth = 135.0
            self.free_cam.elevation = -22.0
        elif preset == "front":
            self.free_cam.lookat[:] = [0.20, 0.0, 0.25]
            self.free_cam.distance = 1.20
            self.free_cam.azimuth = 180.0
            self.free_cam.elevation = -15.0
        elif preset == "side":
            self.free_cam.lookat[:] = [0.25, 0.0, 0.25]
            self.free_cam.distance = 1.15
            self.free_cam.azimuth = 90.0
            self.free_cam.elevation = -15.0
        elif preset == "top":
            self.free_cam.lookat[:] = [0.25, 0.0, 0.15]
            self.free_cam.distance = 1.25
            self.free_cam.azimuth = 90.0
            self.free_cam.elevation = -85.0

    def _select_main_cam(self, key: str):
        if key == "wrist_rgbd" and self.has_wrist_cam:
            self.active_main_cam = "wrist_rgbd"
        else:
            self.active_main_cam = "free"
            self._set_camera_preset(key)

    def _on_mouse_down(self, event):
        if self.active_main_cam != "free":
            return
        self._last_mouse = (event.x, event.y)
        shift_held = (event.state & 0x0001) != 0
        if event.num == 3 or shift_held:
            self._drag_mode = "pan"
        else:
            self._drag_mode = "orbit"

    def _on_mouse_drag(self, event):
        if self.active_main_cam != "free" or self._drag_mode is None:
            return
        dx = event.x - self._last_mouse[0]
        dy = event.y - self._last_mouse[1]
        self._last_mouse = (event.x, event.y)

        if self._drag_mode == "orbit":
            self.free_cam.azimuth -= dx * 0.35
            self.free_cam.elevation = max(-89.0, min(89.0, self.free_cam.elevation - dy * 0.35))
        elif self._drag_mode == "pan":
            scale = 0.0018 * self.free_cam.distance
            az = math.radians(self.free_cam.azimuth)
            el = math.radians(self.free_cam.elevation)
            right = np.array([-math.sin(az), math.cos(az), 0.0])
            up = np.array(
                [-math.cos(az) * math.sin(el), -math.sin(az) * math.sin(el), math.cos(el)]
            )
            self.free_cam.lookat[:] += (-dx * right + dy * up) * scale

    def _on_mouse_up(self, _event):
        self._drag_mode = None

    def _on_mouse_wheel(self, event):
        if self.active_main_cam != "free":
            return
        factor = 0.9 if event.delta > 0 else 1.11
        self.free_cam.distance = max(0.2, min(5.0, self.free_cam.distance * factor))

    def _on_mouse_wheel_linux(self, event):
        if self.active_main_cam != "free":
            return
        factor = 0.9 if event.num == 4 else 1.11
        self.free_cam.distance = max(0.2, min(5.0, self.free_cam.distance * factor))

    def _on_depth_click(self, event):
        x = max(0, min(self.cam_w - 1, int(event.x)))
        y = max(0, min(self.cam_h - 1, int(event.y)))
        self.depth_probe_px = (x, y)

    # ---- Slider & Button Callbacks ----
    def _on_joint_slider(self, aid: int, val: str):
        if self.demo_wave:
            self.demo_wave = False
            self.btn_demo.configure(text="Demo Wave", bg="#0f766e")
        self.data.ctrl[aid] = float(val)

    def _on_gripper_slider(self, val: str):
        if self.has_gripper:
            self.data.ctrl[self.gripper_act_id] = float(val)

    def _on_depth_range(self, val: str):
        self.depth_max_m = float(val)
        self.lbl_depth_max.configure(text=f"{self.depth_max_m:.2f} m")

    def _set_gripper(self, val: float):
        if not self.has_gripper:
            return
        self.gripper_var.set(val)
        self.data.ctrl[self.gripper_act_id] = val

    def _set_arm_targets(self, targets):
        self.demo_wave = False
        self.btn_demo.configure(text="Demo Wave", bg="#0f766e")
        for jinfo, val in zip(self.arm_joints, targets):
            clamped = max(jinfo["min"], min(jinfo["max"], float(val)))
            self.data.ctrl[jinfo["aid"]] = clamped
            self.joint_vars[jinfo["name"]].set(clamped)

    def _sync_sliders_from_ctrl(self):
        for jinfo in self.arm_joints:
            self.joint_vars[jinfo["name"]].set(float(self.data.ctrl[jinfo["aid"]]))
        if self.has_gripper:
            self.gripper_var.set(float(self.data.ctrl[self.gripper_act_id]))

    def _preset_home(self):
        self._set_arm_targets([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if self.has_gripper:
            self._set_gripper(0.0)

    def _preset_pregrasp(self):
        # IK-verified pose hovering right above grasp_cube at [0.40, 0.0, 0.09]
        self._set_arm_targets([-0.066, 0.072, -0.313, -0.004, -0.410, 0.0])
        if self.has_gripper:
            self._set_gripper(0.0)

    def _preset_grasp(self):
        # IK-verified pose centered around grasp_cube at [0.40, 0.0, 0.026]
        self._set_arm_targets([-0.069, -0.063, -0.340, -0.004, -0.339, 0.0])

    def _reset_scene(self):
        self.demo_wave = False
        self.btn_demo.configure(text="Demo Wave", bg="#0f766e")
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._sync_sliders_from_ctrl()

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.configure(text="Resume Sim", bg=ACCENT_ORANGE)
        else:
            self.btn_pause.configure(text="Pause Sim", bg="#475569")

    def _toggle_demo(self):
        self.demo_wave = not self.demo_wave
        if self.demo_wave:
            self.demo_t0 = self.data.time
            self.btn_demo.configure(text="Stop Wave", bg=ACCENT_ORANGE)
        else:
            self.btn_demo.configure(text="Demo Wave", bg="#0f766e")

    def _save_snapshot(self):
        if self.last_rgb is None or self.last_depth_vis is None:
            return
        out_dir = os.path.join(os.path.dirname(self.mjcf_path), "snapshots")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        rgb_path = os.path.join(out_dir, f"wrist_rgb_{ts}.png")
        depth_path = os.path.join(out_dir, f"wrist_depth_{ts}.png")
        raw_path = os.path.join(out_dir, f"wrist_depth_m_{ts}.npy")
        Image.fromarray(self.last_rgb).save(rgb_path)
        Image.fromarray(self.last_depth_vis).save(depth_path)
        if self.last_depth_raw is not None:
            np.save(raw_path, self.last_depth_raw)
        print(f"[Snapshot saved] {rgb_path} | {depth_path}")

    # ---- Depth Rendering Helper ----
    def _colorize_depth(self, depth_m: np.ndarray):
        d_min = 0.05
        d_max = max(d_min + 0.05, self.depth_max_m)
        clipped = np.clip(depth_m, d_min, d_max)
        norm = ((clipped - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        # Invert so closer objects appear warmer (red/yellow) and farther cooler (blue)
        bgr = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Draw probe crosshair + metric distance label
        px, py = self.depth_probe_px
        probe_dist = float(depth_m[py, px])
        cv2.drawMarker(
            rgb, (px, py), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1
        )
        txt = f"d = {probe_dist:.3f} m"
        cv2.rectangle(rgb, (6, 6), (132, 26), (15, 18, 24), -1)
        cv2.putText(
            rgb, txt, (11, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Draw compact vertical colorbar legend on right edge
        bar_w = 10
        bar_h = rgb.shape[0] - 24
        grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
        bar_bgr = cv2.applyColorMap(np.repeat(grad, bar_w, axis=1), cv2.COLORMAP_TURBO)
        bar_rgb = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB)
        x0 = rgb.shape[1] - bar_w - 6
        y0 = 12
        rgb[y0 : y0 + bar_h, x0 : x0 + bar_w] = bar_rgb
        cv2.putText(
            rgb, f"{d_min:.2f}m", (x0 - 44, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1
        )
        cv2.putText(
            rgb, f"{d_max:.1f}m", (x0 - 38, y0 + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1
        )
        return rgb, probe_dist

    # ---- Main Live Simulation & Render Tick ----
    def _tick(self):
        if self._closing:
            return
        # 1. Optional gentle demo wave
        if self.demo_wave and not self.paused:
            dt_wave = self.data.time - self.demo_t0
            wave = [
                0.35 * math.sin(0.8 * dt_wave),
                0.20 * math.sin(0.6 * dt_wave),
                -0.25 + 0.20 * math.cos(0.7 * dt_wave),
                0.30 * math.sin(1.1 * dt_wave),
                -0.30 + 0.25 * math.sin(0.9 * dt_wave),
                0.50 * math.sin(1.2 * dt_wave),
            ]
            for jinfo, val in zip(self.arm_joints, wave):
                self.data.ctrl[jinfo["aid"]] = val
                self.joint_vars[jinfo["name"]].set(val)
            if self.has_gripper:
                gval = 127.5 * (1.0 - math.cos(1.2 * dt_wave))
                self.data.ctrl[self.gripper_act_id] = gval
                self.gripper_var.set(gval)

        # 2. Step MuJoCo physics to match real-time frame interval
        if not self.paused:
            n_steps = max(1, int(round((self.frame_ms / 1000.0) / self.model.opt.timestep)))
            for _ in range(n_steps):
                mujoco.mj_step(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)

        # 3. Update MuJoCo visual options
        self.scene_opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.var_contacts.get())
        self.scene_opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(self.var_contacts.get())
        self.scene_opt.frame = (
            mujoco.mjtFrame.mjFRAME_BODY if self.var_frames.get() else mujoco.mjtFrame.mjFRAME_NONE
        )

        # 4. Render 3D Main Viewport
        cam_arg = "wrist_rgbd" if self.active_main_cam == "wrist_rgbd" else self.free_cam
        self.main_renderer.update_scene(self.data, camera=cam_arg, scene_option=self.scene_opt)
        main_rgb = self.main_renderer.render()
        self._img_main = ImageTk.PhotoImage(Image.fromarray(main_rgb))
        self.lbl_main.configure(image=self._img_main)

        # 5. Render Wrist Camera RGB & Metric Depth
        probe_dist = None
        if self.has_wrist_cam and self.wrist_renderer is not None:
            self.wrist_renderer.disable_depth_rendering()
            self.wrist_renderer.update_scene(self.data, camera="wrist_rgbd")
            wrist_rgb = self.wrist_renderer.render().copy()
            # Draw subtle center reticle on RGB feed
            cx, cy = self.cam_w // 2, self.cam_h // 2
            cv2.drawMarker(
                wrist_rgb, (cx, cy), (80, 255, 120), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1
            )
            self.last_rgb = wrist_rgb
            self._img_rgb = ImageTk.PhotoImage(Image.fromarray(wrist_rgb))
            self.lbl_rgb.configure(image=self._img_rgb)

            self.wrist_renderer.enable_depth_rendering()
            self.wrist_renderer.update_scene(self.data, camera="wrist_rgbd")
            depth_raw = self.wrist_renderer.render().copy()
            self.wrist_renderer.disable_depth_rendering()
            self.last_depth_raw = depth_raw

            depth_vis, probe_dist = self._colorize_depth(depth_raw)
            self.last_depth_vis = depth_vis
            self._img_depth = ImageTk.PhotoImage(Image.fromarray(depth_vis))
            self.lbl_depth.configure(image=self._img_depth)

        # 6. Update joint & gripper live readouts
        for jinfo in self.arm_joints:
            jname = jinfo["name"]
            cmd = float(self.data.ctrl[jinfo["aid"]])
            qaddr = self.model.jnt_qposadr[jinfo["jid"]]
            act = float(self.data.qpos[qaddr])
            self.joint_readouts[jname].configure(
                text=f"cmd:{cmd:+.2f} | act:{act:+.2f} rad ({math.degrees(act):+5.1f}\u00b0)"
            )

        if self.has_gripper and self.gripper_readout is not None:
            g_cmd = float(self.data.ctrl[self.gripper_act_id])
            r_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
            g_rad = float(self.data.qpos[self.model.jnt_qposadr[r_jid]]) if r_jid >= 0 else 0.0
            self.gripper_readout.configure(text=f"cmd:{g_cmd:5.1f} | knuckle:{g_rad:.3f} rad")

        # 7. Update FPS & Telemetry
        self._fps_frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 0.5:
            self.fps_value = self._fps_frames / (now - self._fps_t)
            self._fps_frames = 0
            self._fps_t = now

        tcp_txt = "n/a"
        if self.tcp_site_id >= 0:
            p = self.data.site_xpos[self.tcp_site_id]
            tcp_txt = f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m"
        depth_txt = (
            f"{probe_dist:.3f} m @ px{self.depth_probe_px}" if probe_dist is not None else "n/a"
        )
        self.lbl_telemetry.configure(
            text=f"Sim: {self.data.time:6.2f} s  |  GUI: {self.fps_value:4.1f} FPS\n"
            f"TCP: {tcp_txt}\n"
            f"Depth probe: {depth_txt}"
        )

        # Schedule next live frame
        if not self._closing:
            self._after_id = self.root.after(self.frame_ms, self._tick)

    def _on_close(self):
        self._closing = True
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        try:
            self.main_renderer.close()
        except Exception:
            pass
        if self.wrist_renderer is not None:
            try:
                self.wrist_renderer.close()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self._after_id = self.root.after(10, self._tick)
        self.root.mainloop()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Live Interactive MuJoCo Control & RGB-D Camera GUI for WISERLI-ATAL-C1.1"
    )
    parser.add_argument(
        "--mjcf",
        type=str,
        default=None,
        help="Path to MJCF scene XML (defaults to scene_2f85.xml).",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Load arm-only scene.xml without gripper/wrist camera.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target GUI refresh rate (default: 30).",
    )
    args = parser.parse_args()

    if args.mjcf:
        mjcf_path = args.mjcf
    elif args.no_gripper:
        mjcf_path = os.path.join(script_dir, "scene.xml")
    else:
        mjcf_path = os.path.join(script_dir, "scene_2f85.xml")

    if not os.path.isfile(mjcf_path):
        print(f"Error: MJCF file not found: {mjcf_path}", file=sys.stderr)
        sys.exit(1)

    app = WiserliMuJoCoGUI(mjcf_path, target_fps=args.fps)
    app.run()


if __name__ == "__main__":
    main()
