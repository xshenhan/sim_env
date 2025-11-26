#!/usr/bin/env python3
"""
Get detailed camera configuration from RLBench
Run with: xvfb-run -a python check_camera_detailed.py
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks.close_box import CloseBox
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np

print("="*80)
print("RLBench 相机详细配置")
print("="*80)

obs_config = ObservationConfig()
obs_config.left_shoulder_camera.rgb = True

action_mode = MoveArmThenGripper(
    arm_action_mode=JointPosition(),
    gripper_action_mode=Discrete()
)

env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True)
env.launch()
task_env = env.get_task(CloseBox)

# Reset to initialize scene
desc, obs = task_env.reset()

# Get workspace position
workspace_pos = task_env._scene._workspace.get_position()
print(f"\n【Workspace 中心位置】")
print(f"  Position: [{workspace_pos[0]:.4f}, {workspace_pos[1]:.4f}, {workspace_pos[2]:.4f}]")

# Get detailed camera info
print(f"\n【RLBench 默认相机详细配置】")
print(f"{'='*80}")

cameras = [
    'cam_over_shoulder_left',
    'cam_over_shoulder_right',
    'cam_overhead',
    'cam_front',
    'cam_wrist'
]

for cam_name in cameras:
    try:
        cam = VisionSensor(cam_name)

        # Get position
        pos = cam.get_position()

        # Get orientation (Euler angles in radians)
        orient = cam.get_orientation()
        orient_deg = np.degrees(orient)

        # Get FOV
        fov = cam.get_perspective_angle()
        fov_deg = np.degrees(fov)

        # Get resolution
        res = cam.get_resolution()

        # Calculate distance to workspace
        dist_to_workspace = np.sqrt(
            (pos[0] - workspace_pos[0])**2 +
            (pos[1] - workspace_pos[1])**2 +
            (pos[2] - workspace_pos[2])**2
        )

        # Calculate height above workspace
        height_above = pos[2] - workspace_pos[2]

        print(f"\n{cam_name}:")
        print(f"  Position:    [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
        print(f"  Orientation: [{orient_deg[0]:7.2f}°, {orient_deg[1]:7.2f}°, {orient_deg[2]:7.2f}°] (α, β, γ)")
        print(f"  FOV:         {fov_deg:.2f}°")
        print(f"  Resolution:  {res[0]} x {res[1]}")
        print(f"  Distance to workspace: {dist_to_workspace:.4f}m")
        print(f"  Height above workspace: {height_above:+.4f}m")

    except Exception as e:
        print(f"\n{cam_name}: 获取失败 - {e}")

print(f"\n{'='*80}")
print(f"\n【总结】")
print(f"  如果要设置多相机的 target_point，应该使用:")
print(f"  target_point = [{workspace_pos[0]:.4f}, {workspace_pos[1]:.4f}, {workspace_pos[2]:.4f}]")

env.shutdown()
