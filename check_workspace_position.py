#!/usr/bin/env python3
"""
Check RLBench workspace position and compare with our camera target
Run with: xvfb-run -a python check_workspace_position.py
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks.close_box import CloseBox
from pyrep.objects.vision_sensor import VisionSensor


def euler_to_rotation_matrix(alpha, beta, gamma):
    """
    将ZYX欧拉角转换为旋转矩阵
    alpha: 绕Z轴旋转 (yaw)
    beta: 绕Y轴旋转 (pitch)
    gamma: 绕X轴旋转 (roll)
    """
    # 绕Z轴旋转
    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
    # 绕Y轴旋转
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    # 绕X轴旋转
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])
    # ZYX顺序：先Z，再Y，最后X
    R = Rz @ Ry @ Rx
    return R


def compute_ray_intersection(positions, directions):
    """
    计算多条射线的最接近交点（最小二乘法）
    
    对于每条射线 i: r_i(t) = p_i + t * d_i
    找到点 x，使得 sum_i min_t ||x - r_i(t)||^2 最小
    
    Args:
        positions: 射线起点列表，每个是 [x, y, z]
        directions: 射线方向列表，每个是归一化的方向向量 [dx, dy, dz]
    
    Returns:
        intersection_point: 最接近的交点 [x, y, z]
        avg_distance: 平均距离误差
    """
    positions = np.array(positions)
    directions = np.array(directions)
    
    n = len(positions)
    
    # 构建线性系统: 最小化 sum_i ||(I - d_i * d_i^T)(x - p_i)||^2
    # 对x求导并令其等于0，得到: sum_i P_i * x = sum_i P_i * p_i
    # 其中 P_i = I - d_i * d_i^T 是投影矩阵
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for i in range(n):
        d = directions[i]
        p = positions[i]
        # 投影矩阵: P_i = I - d * d^T
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ p
    
    # 求解 A * intersection = b
    try:
        intersection_point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        intersection_point = np.linalg.pinv(A) @ b
    
    # 计算平均距离误差
    distances = []
    for i in range(n):
        p = positions[i]
        d = directions[i]
        # 计算射线到交点的最近点
        t = np.dot(intersection_point - p, d)
        closest_point_on_ray = p + t * d
        dist = np.linalg.norm(intersection_point - closest_point_on_ray)
        distances.append(dist)
    
    avg_distance = np.mean(distances)
    
    return intersection_point, avg_distance

print("="*70)
print("查询RLBench workspace和相机位置")
print("="*70)

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
print(f"\n1. RLBench workspace中心位置:")
print(f"   [{workspace_pos[0]:.3f}, {workspace_pos[1]:.3f}, {workspace_pos[2]:.3f}]")

# Get RLBench default cameras positions
print(f"\n2. RLBench默认相机位置:")
cameras = [
    'cam_over_shoulder_left',
    'cam_over_shoulder_right',
    'cam_overhead',
    'cam_front',
    'cam_wrist'
]

camera_data = []  # 存储相机数据用于计算交点

for cam_name in cameras:
    try:
        cam = VisionSensor(cam_name)
        pos = cam.get_position()
        rot = cam.get_orientation()  # Euler angles in radians
        rot_deg = np.degrees(rot)  # Convert to degrees
        
        # 计算相机朝向向量（局部Z轴方向）
        R = euler_to_rotation_matrix(rot[0], rot[1], rot[2])
        forward = R[:, 2]  # 局部Z轴在世界坐标系中的方向
        
        camera_data.append({
            'name': cam_name,
            'position': np.array(pos),
            'rotation': rot,
            'forward': forward
        })
        
        print(f"   {cam_name:25s}:")
        print(f"      Position: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
        print(f"      Rotation: [{rot[0]:7.3f}, {rot[1]:7.3f}, {rot[2]:7.3f}] rad")
        print(f"                [{rot_deg[0]:7.2f}°, {rot_deg[1]:7.2f}°, {rot_deg[2]:7.2f}°]")
        print(f"      Forward:  [{forward[0]:7.3f}, {forward[1]:7.3f}, {forward[2]:7.3f}]")
    except Exception as e:
        print(f"   {cam_name:25s}: 不存在 ({e})")

# 计算射线交点
if len(camera_data) >= 2:
    print(f"\n2.1. 计算相机射线交点:")
    positions = [data['position'] for data in camera_data]
    directions = [data['forward'] for data in camera_data]
    
    intersection_point, avg_distance = compute_ray_intersection(positions, directions)
    
    print(f"   最接近交点位置: [{intersection_point[0]:7.3f}, {intersection_point[1]:7.3f}, {intersection_point[2]:7.3f}]")
    print(f"   平均距离误差: {avg_distance:.6f}m")
    
    # 计算每条射线到交点的距离
    print(f"\n   各射线到交点的距离:")
    for i, data in enumerate(camera_data):
        pos = data['position']
        forward = data['forward']
        # 计算射线到交点的最近点参数t
        t = np.dot(intersection_point - pos, forward)
        closest_point_on_ray = pos + t * forward
        dist = np.linalg.norm(intersection_point - closest_point_on_ray)
        print(f"      {data['name']:25s}: 距离={dist:.6f}m, 参数t={t:.3f}")
    
    # 与workspace位置对比
    dist_to_workspace = np.linalg.norm(intersection_point - np.array(workspace_pos))
    print(f"\n   交点与workspace中心的距离: {dist_to_workspace:.6f}m")
else:
    print(f"\n2.1. 相机数量不足，无法计算交点（需要至少2个相机）")

# Calculate average height of cameras above workspace
print(f"\n3. 默认相机与workspace的关系:")
cam_left = VisionSensor('cam_over_shoulder_left')
cam_right = VisionSensor('cam_over_shoulder_right')
cam_overhead = VisionSensor('cam_overhead')

heights = []
for cam in [cam_left, cam_right, cam_overhead]:
    pos = cam.get_position()
    height_above_workspace = pos[2] - workspace_pos[2]
    heights.append(height_above_workspace)

avg_height = sum(heights) / len(heights)
print(f"   相机平均高度（相对workspace）: {avg_height:.3f}m")

# Our current settings
our_target = [0.25, 0.0, 0.75]
our_radius = 1.5

print(f"\n4. 我们当前的设置:")
print(f"   camera_target: [{our_target[0]:.3f}, {our_target[1]:.3f}, {our_target[2]:.3f}]")
print(f"   camera_radius: {our_radius:.3f}m")

# Compare
print(f"\n5. 对比:")
diff_x = our_target[0] - workspace_pos[0]
diff_y = our_target[1] - workspace_pos[1]
diff_z = our_target[2] - workspace_pos[2]

print(f"   X方向差异: {diff_x:+.3f}m")
print(f"   Y方向差异: {diff_y:+.3f}m")
print(f"   Z方向差异: {diff_z:+.3f}m")

if abs(diff_x) < 0.1 and abs(diff_y) < 0.1 and abs(diff_z) < 0.1:
    print(f"\n   ✅ 我们的target与workspace中心接近")
else:
    print(f"\n   ⚠️  我们的target与workspace中心有较大差异")
    print(f"   建议使用: [{workspace_pos[0]:.2f}, {workspace_pos[1]:.2f}, {workspace_pos[2]:.2f}]")

print(f"{'='*70}")
env.shutdown()
