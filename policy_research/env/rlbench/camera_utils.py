"""
Camera utilities for multi-camera setup in RLBench

Provides sphere sampling and camera configuration utilities for
dynamically creating multiple cameras around the workspace.
"""

import numpy as np
from typing import List, Tuple, Optional
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode
from scipy.spatial.transform import Rotation as R


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


def compute_look_at_orientation(position: List[float], target: List[float]) -> List[float]:
    """
    Compute Euler angles (alpha, beta, gamma) for a camera looking at target

    Args:
        position: Camera position [x, y, z]
        target: Target point to look at [x, y, z]

    Returns:
        Euler angles [alpha, beta, gamma] in radians for PyRep/CoppeliaSim
    """
    # Compute forward direction (from camera to target)
    forward = np.array(target) - np.array(position)
    forward = forward / np.linalg.norm(forward)

    # World up vector
    world_up = np.array([0, 0, 1])

    # Compute right vector
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)

    # Handle edge case: camera pointing straight up or down
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm

    # Compute actual up vector
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix (camera coordinate system)
    # In CoppeliaSim/PyRep, camera looks along +Y axis in its local frame
    # Local coordinate system:
    #   +X = right (图像水平方向右方)
    #   +Y = forward (相机视线方向)
    #   +Z = up (图像上方)
    Rot = np.zeros((3, 3))
    Rot[:, 0] = -right      # local X (right)
    Rot[:, 1] = up    # local Y (forward, camera looks along +Y)
    Rot[:, 2] = forward         # local Z (up)


    return R.from_matrix(Rot).as_euler('xyz', degrees=False)


def create_cameras(
    num_cameras: int,
    image_size: int,
    target_point: List[float] = [0.25, 0.0, 0.75],
    radius: float = 1.5,
    fov: float = 40.0,
    template_camera: Optional[VisionSensor] = None
) -> List[VisionSensor]:
    """
    Create multiple cameras uniformly distributed on a sphere

    Args:
        num_cameras: Number of cameras to create
        image_size: Resolution of camera images (square)
        target_point: Center point that all cameras look at
        radius: Distance from cameras to target point
        fov: Field of view in degrees
        template_camera: Template camera to copy from (if None, creates new cameras)

    Returns:
        List of VisionSensor objects
    """
    camera_configs = generate_camera_positions(num_cameras, target_point, radius)
    cameras = []

    for i, (position, target) in enumerate(camera_configs):
        # Create camera - use copy() if template provided, otherwise create new
        if template_camera is not None:
            cam = template_camera.copy()
        else:
            cam = VisionSensor.create([image_size, image_size])
            cam.set_render_mode(RenderMode.OPENGL3)

        # Set position
        cam.set_position(position)

        # Set orientation to look at target
        orientation = compute_look_at_orientation(position, target)
        cam.set_orientation(orientation)

        # Set FOV (set_perspective_angle expects degrees, not radians)
        cam.set_perspective_angle(fov)

        # Set resolution if using copied camera
        if template_camera is not None:
            cam.set_resolution([image_size, image_size])

        # IMPORTANT: Enable explicit handling permanently
        # This is required for handle_explicitly() to work
        cam.set_explicit_handling(1)

        cameras.append(cam)

    return cameras


def get_camera_intrinsics(
    image_size: int,
    fov: float = 40.0
) -> np.ndarray:
    """
    Compute camera intrinsic matrix

    Args:
        image_size: Resolution of camera images (square)
        fov: Field of view in degrees

    Returns:
        3x3 intrinsic matrix
    """
    focal_length = (image_size / 2.0) / np.tan(np.deg2rad(fov / 2.0))

    intrinsic_matrix = np.array([
        [focal_length, 0, image_size / 2.0],
        [0, focal_length, image_size / 2.0],
        [0, 0, 1]
    ])

    return intrinsic_matrix


def get_camera_extrinsics(camera: VisionSensor) -> np.ndarray:
    """
    Get camera extrinsic matrix (world to camera transform)

    Args:
        camera: VisionSensor object

    Returns:
        4x4 extrinsic matrix
    """
    # Get camera pose (position + quaternion)
    pose = camera.get_pose()
    position = pose[:3]
    quaternion = pose[3:]  # [qx, qy, qz, qw]

    # Convert quaternion to rotation matrix
    qx, qy, qz, qw = quaternion

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

    # Build extrinsic matrix (world to camera)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R.T
    extrinsic_matrix[:3, 3] = -R.T @ position

    return extrinsic_matrix


def unproject_depth_to_pointcloud(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    bbox: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth image to 3D point cloud in world coordinates

    Args:
        depth: Depth image (H, W)
        intrinsic: 3x3 intrinsic matrix
        extrinsic: 4x4 extrinsic matrix (world to camera)
        bbox: Optional bounding box [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        (points, valid_mask) where points is (H*W, 3) and valid_mask is (H*W,)
    """
    h, w = depth.shape

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([u.flatten(), v.flatten(), np.ones(h * w)], axis=0)  # [3, H*W]

    # Inverse of intrinsic matrix
    K_inv = np.linalg.inv(intrinsic)

    # Camera coordinates
    xyz_cam = K_inv @ pixels  # [3, H*W]
    xyz_cam = xyz_cam * depth.flatten()  # [3, H*W]
    xyz_cam = xyz_cam.T  # [H*W, 3]

    # Convert to world coordinates
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3:4].T  # [1, 3]

    # Extrinsic is world to camera, need to invert
    R_cam_to_world = R.T
    t_cam_to_world = -R_cam_to_world @ t.T

    xyz_world = (xyz_cam @ R_cam_to_world.T) + t_cam_to_world.T

    # Filter invalid depths
    valid_mask = (depth.flatten() > 0) & (depth.flatten() < 10.0)

    # Apply bounding box filter if provided
    if bbox is not None:
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        bbox_mask = (
            (xyz_world[:, 0] >= x_min) & (xyz_world[:, 0] <= x_max) &
            (xyz_world[:, 1] >= y_min) & (xyz_world[:, 1] <= y_max) &
            (xyz_world[:, 2] >= z_min) & (xyz_world[:, 2] <= z_max)
        )
        valid_mask = valid_mask & bbox_mask

    return xyz_world, valid_mask


def merge_pointclouds(
    cameras: List[VisionSensor],
    image_size: int,
    fov: float = 40.0,
    bbox: Optional[List[float]] = None
) -> np.ndarray:
    """
    Merge point clouds from multiple cameras

    Args:
        cameras: List of VisionSensor objects
        image_size: Resolution of camera images
        fov: Field of view in degrees
        bbox: Optional bounding box for filtering

    Returns:
        Merged point cloud (N, 3)
    """
    intrinsic = get_camera_intrinsics(image_size, fov)
    pointclouds = []

    for cam in cameras:
        # Capture depth
        depth = cam.capture_depth()

        # Get extrinsic
        extrinsic = get_camera_extrinsics(cam)

        # Unproject to world coordinates
        xyz_world, valid_mask = unproject_depth_to_pointcloud(
            depth, intrinsic, extrinsic, bbox
        )

        pointclouds.append(xyz_world[valid_mask])

    # Merge all point clouds
    if pointclouds:
        return np.concatenate(pointclouds, axis=0)
    else:
        return np.zeros((0, 3))
