#!/usr/bin/env python3
"""
可视化相机位置的独立脚本
使用 Open3D 绘制相机位置，每个相机用小坐标系表示，并显示世界坐标系
"""

import sys
import os

# 添加当前目录到路径，确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d
from typing import List, Tuple


def generate_camera_positions(
    num_cameras: int = 40,
    target_point: List[float] = [0.25, 0.0, 0.75],
    radius: float = 1.5
) -> List[Tuple[List[float], List[float]]]:
    """
    Generate camera positions uniformly distributed on a sphere using Hammersley sequence

    Args:
        num_cameras: Number of cameras to generate
        target_point: Center point that all cameras look at (workspace center)
        radius: Distance from cameras to target point

    Returns:
        List of (position, target) tuples
    """
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def radical_inverse(base, n):
        val = 0
        inv_base = 1.0 / base
        inv_base_n = inv_base
        while n > 0:
            digit = n % base
            val += digit * inv_base_n
            n //= base
            inv_base_n *= inv_base
        return val

    def halton_sequence(dim, n):
        return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

    def hammersley_sequence(dim, n, num_samples):
        return [n / num_samples] + halton_sequence(dim - 1, n)

    def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
        u, v = hammersley_sequence(2, n, num_samples)
        u += offset[0] / num_samples
        v += offset[1]
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
        theta = np.arccos(1 - 2 * u) - np.pi / 2
        phi = v * 2 * np.pi
        return [phi, theta]

    camera_configs = []
    offset = (np.random.rand(), np.random.rand())

    for i in range(num_cameras):
        y, p = sphere_hammersley_sequence(i, num_cameras, offset)
        # Only keep cameras above xy-plane
        if p < 0:
            p = -p

        cam_pos = [
            radius * np.cos(p) * np.cos(y) + target_point[0],
            radius * np.cos(p) * np.sin(y) + target_point[1],
            radius * np.sin(p) + target_point[2],
        ]

        camera_configs.append((cam_pos, target_point))

    return camera_configs


def compute_camera_rotation_matrix(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    计算相机的旋转矩阵，使其Y轴指向目标（PyRep/CoppeliaSim 坐标系）
    
    PyRep/CoppeliaSim 相机局部坐标系:
        +X = right (图像水平方向右方)
        +Y = forward (相机视线方向，相机看向 +Y)
        +Z = up (图像上方)
    
    Returns:
        3x3 旋转矩阵
    """
    # 前向向量 (从相机指向目标)
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    # 世界向上
    world_up = np.array([0, 0, 1])
    
    # 右向量
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm
    
    # 上向量
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # 旋转矩阵: 列向量是 [X, Y, Z] = [right, forward, up]
    # 相机看向 +Y 方向
    R = np.column_stack([right, forward, up])
    
    return R


def create_camera_frustum(position: np.ndarray, rotation: np.ndarray, 
                          scale: float = 0.1, color: List[float] = [1, 1, 1]) -> o3d.geometry.LineSet:
    """
    创建一个相机视锥体线框
    """
    # 相机视锥体的顶点（相机局部坐标系）
    # 相机在原点，看向+Z方向
    w, h, d = scale * 0.6, scale * 0.4, scale  # 宽度、高度、深度
    
    vertices_local = np.array([
        [0, 0, 0],           # 相机中心
        [-w, -h, d],         # 左下
        [w, -h, d],          # 右下
        [w, h, d],           # 右上
        [-w, h, d],          # 左上
    ])
    
    # 转换到世界坐标系
    vertices_world = (rotation @ vertices_local.T).T + position
    
    # 定义边
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 从相机中心到四角
        [1, 2], [2, 3], [3, 4], [4, 1],  # 视锥体底边
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set


def create_coordinate_frame(position: np.ndarray, rotation: np.ndarray = None, 
                            size: float = 0.1) -> o3d.geometry.TriangleMesh:
    """
    创建一个坐标系
    """
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    if rotation is not None:
        coord_frame.rotate(rotation, center=[0, 0, 0])
    
    coord_frame.translate(position)
    
    return coord_frame


def visualize_cameras(
    num_cameras: int = 40,
    target_point: List[float] = [0.25, 0.0, 0.75],
    radius: float = 1.5
):
    """
    使用 Open3D 可视化相机位置
    """
    # 生成相机位置
    camera_configs = generate_camera_positions(num_cameras, target_point, radius)
    
    print("=" * 50)
    print("相机位置可视化 (Open3D)")
    print("=" * 50)
    print(f"相机数量: {num_cameras}")
    print(f"目标点: {target_point}")
    print(f"半径: {radius}")
    print("-" * 50)
    print("控制说明:")
    print("  鼠标左键拖动: 旋转视角")
    print("  鼠标滚轮: 缩放")
    print("  鼠标中键拖动: 平移")
    print("  R: 重置视角")
    print("  Q: 退出")
    print("=" * 50)
    
    geometries = []
    
    # 1. 创建世界坐标系（大的，在原点）
    world_frame = create_coordinate_frame(
        position=np.array([0, 0, 0]),
        rotation=None,
        size=0.5
    )
    geometries.append(world_frame)
    
    # 2. 创建目标点球体
    target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    target_sphere.translate(target_point)
    target_sphere.paint_uniform_color([1, 1, 0])  # 黄色
    geometries.append(target_sphere)
    
    # 3. 为每个相机创建坐标系和视锥体
    for i, (cam_pos, cam_target) in enumerate(camera_configs):
        position = np.array(cam_pos)
        target_pt = np.array(cam_target)
        
        # 计算相机旋转矩阵
        rotation = compute_camera_rotation_matrix(position, target_pt)
        
        # 创建相机坐标系（小的）
        cam_frame = create_coordinate_frame(
            position=position,
            rotation=rotation,
            size=0.08
        )
        geometries.append(cam_frame)
        
        # 创建相机视锥体
        frustum = create_camera_frustum(
            position=position,
            rotation=rotation,
            scale=0.1,
            color=[0.7, 0.7, 0.7]  # 浅灰色
        )
        geometries.append(frustum)
        
        # 创建从相机到目标的连线
        line_points = np.array([position, target_pt])
        line_indices = np.array([[0, 1]])
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(line_points)
        line.lines = o3d.utility.Vector2iVector(line_indices)
        line.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3]])  # 深灰色
        geometries.append(line)
    
    # 4. 创建一个包围球的线框（帮助理解相机分布）
    sphere_lineset = o3d.geometry.LineSet.create_from_triangle_mesh(
        o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    )
    sphere_lineset.translate(target_point)
    sphere_lineset.paint_uniform_color([0.2, 0.2, 0.5])  # 深蓝色
    geometries.append(sphere_lineset)
    
    # 可视化
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Camera Positions ({num_cameras} cameras)",
        width=1280,
        height=720,
        left=100,
        top=100
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化相机位置分布 (Open3D)")
    parser.add_argument("--num_cameras", type=int, default=40, help="相机数量")
    parser.add_argument("--target_x", type=float, default=0.25, help="目标点X坐标")
    parser.add_argument("--target_y", type=float, default=0.0, help="目标点Y坐标")
    parser.add_argument("--target_z", type=float, default=0.75, help="目标点Z坐标")
    parser.add_argument("--radius", type=float, default=1.5, help="相机到目标点的距离")
    
    args = parser.parse_args()
    
    target_point = [args.target_x, args.target_y, args.target_z]
    
    visualize_cameras(
        num_cameras=args.num_cameras,
        target_point=target_point,
        radius=args.radius
    )
