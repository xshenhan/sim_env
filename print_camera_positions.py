#!/usr/bin/env python3
"""
Print generated camera positions
Run with: python print_camera_positions.py
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from policy_research.env.rlbench.camera_utils import generate_camera_positions
import numpy as np

print("="*70)
print("生成的相机位置")
print("="*70)

# Use same settings as our current implementation
target_point = [0.25, 0.0, 0.75]
radius = 1.5
num_cameras = 40

print(f"\n参数:")
print(f"  target_point: {target_point}")
print(f"  radius: {radius}m")
print(f"  num_cameras: {num_cameras}")

# Generate camera positions
camera_configs = generate_camera_positions(
    num_cameras=num_cameras,
    target_point=target_point,
    radius=radius
)

print(f"\n生成的{num_cameras}个相机位置:")
print("="*70)
print(f"{'ID':>3} | {'X':>8} | {'Y':>8} | {'Z':>8} | {'高度':>8} | {'距离':>8} | {'仰角':>8}")
print("-"*70)

min_z = float('inf')
max_z = float('-inf')
min_elevation = float('inf')
max_elevation = float('-inf')

for i, (position, target) in enumerate(camera_configs):
    x, y, z = position

    # Calculate height above target
    height_above_target = z - target_point[2]

    # Calculate distance from target
    dist = np.sqrt((x - target_point[0])**2 +
                   (y - target_point[1])**2 +
                   (z - target_point[2])**2)

    # Calculate elevation angle (仰角)
    # Elevation = arcsin((z - target_z) / distance)
    elevation_rad = np.arcsin(height_above_target / dist) if dist > 0 else 0
    elevation_deg = np.degrees(elevation_rad)

    print(f"{i:3d} | {x:8.3f} | {y:8.3f} | {z:8.3f} | "
          f"{height_above_target:+8.3f} | {dist:8.3f} | {elevation_deg:7.1f}°")

    # Track min/max
    min_z = min(min_z, z)
    max_z = max(max_z, z)
    min_elevation = min(min_elevation, elevation_deg)
    max_elevation = max(max_elevation, elevation_deg)

print("="*70)
print(f"\n统计:")
print(f"  Z坐标范围: {min_z:.3f}m - {max_z:.3f}m")
print(f"  相对target高度范围: {min_z - target_point[2]:+.3f}m - {max_z - target_point[2]:+.3f}m")
print(f"  仰角范围: {min_elevation:.1f}° - {max_elevation:.1f}°")

# Count cameras by height
below_target = sum(1 for pos, _ in camera_configs if pos[2] < target_point[2])
near_target = sum(1 for pos, _ in camera_configs
                  if abs(pos[2] - target_point[2]) < 0.3)
above_target = sum(1 for pos, _ in camera_configs if pos[2] > target_point[2])

print(f"\n  低于target (z < {target_point[2]:.2f}): {below_target} 个相机")
print(f"  接近target (±0.3m): {near_target} 个相机")
print(f"  高于target (z > {target_point[2]:.2f}): {above_target} 个相机")

# Count cameras by elevation angle
low_angle = sum(1 for pos, _ in camera_configs
                if np.degrees(np.arcsin((pos[2] - target_point[2]) / radius)) < 15)
high_angle = sum(1 for pos, _ in camera_configs
                 if np.degrees(np.arcsin((pos[2] - target_point[2]) / radius)) > 45)

print(f"\n  低仰角 (< 15°): {low_angle} 个相机")
print(f"  高仰角 (> 45°): {high_angle} 个相机")

if min_elevation < 10:
    print(f"\n⚠️  警告: 有相机仰角过低 (最低{min_elevation:.1f}°)")
    print(f"   可能看不到桌面/机器人")

print("="*70)
