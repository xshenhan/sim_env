import os
import string
import numpy as np
import gymnasium
import hydra
import gc
from PIL import Image
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.environment import Environment
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.const import RenderMode

from policy_research.env.rlbench.action_modes import (
    AbsoluteJointPositionActionMode, 
    AbsoluteEndEffectorPoseActionMode,
    DeltaJointPositionActionMode,
)

from typing import Optional, List, Literal

from policy_research.env.rlbench.camera_utils import (
    create_cameras,
    get_camera_intrinsics,
    get_camera_extrinsics,
    merge_pointclouds,
)



class RlbenchEnv(gymnasium.Env):
    metadata = {
        "render_modes": ['rgb_array'],
        "render_fps": 10
    }

    def __init__(self,
        task_name: Optional[str] = None,
        image_size: int = 128,
        seed: Optional[int] = None,
        camera_names: List[str] = ['left_shoulder','right_shoulder','overhead','wrist','front'],
        state_ports: List[str] = [
            'joint_positions', 'joint_velocities', 'joint_forces',
            'gripper_open', 'gripper_pose', 'gripper_joint_positions',
            'gripper_touch_forces'
        ],
        video_resolution: int = 512,
        max_episode_steps: int = 250,
        control_mode: Literal['ee_pose', 'qpos', 'delta_qpos'] = 'qpos',
        use_multi_cam: bool = False,
        num_cameras: int = 40,
        camera_target: List[float] = [0.25, 0.0, 0.75],
        camera_radius: float = 1.5,
        camera_fov: float = 40.0,
        collect_pointcloud: bool = True,
        pointcloud_bbox: Optional[List[float]] = None,
    ):
        super().__init__()

        # Multi-camera configuration
        self.use_multi_cam = use_multi_cam
        self.num_cameras = num_cameras
        self.camera_target = camera_target
        self.camera_radius = camera_radius
        self.camera_fov = camera_fov
        self.collect_pointcloud = collect_pointcloud
        self.pointcloud_bbox = pointcloud_bbox
        self.multi_cameras = None
        
        # Debug: save first frame during get_demos
        self.debug_save_first_frame = False
        self.debug_dir = None
        self.debug_episode_idx = 0
        self._debug_frame_count = 0

        # If using multi-camera, disable default cameras
        if use_multi_cam:
            camera_names = []

        self.camera_names = camera_names
        self.state_ports = state_ports
        self.image_size = image_size
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False

        # observation config
        obs_config = ObservationConfig()
        for name in camera_names:
            obs_config.__dict__[f"{name}_camera"].rgb = True
            obs_config.__dict__[f"{name}_camera"].depth = False
            obs_config.__dict__[f"{name}_camera"].point_cloud = False
            obs_config.__dict__[f"{name}_camera"].image_size = (image_size, image_size)
        for name in state_ports:
            obs_config.__dict__[name] = True

        # coppelia engine setup
        if control_mode == 'ee_pose':
            self.action_mode = AbsoluteEndEffectorPoseActionMode()
        elif control_mode == 'qpos':
            self.action_mode = AbsoluteJointPositionActionMode()
        elif control_mode == 'delta_qpos':
            self.action_mode = DeltaJointPositionActionMode()
        else:
            raise ValueError(f"Unknown control mode: {control_mode}")
        self.rlbench_env = Environment(
            action_mode=self.action_mode,
            obs_config=obs_config,
            headless=True
        )
        self.rlbench_env.launch()

        # task setup
        if task_name is not None:
            self.set_task(task_name, seed=seed)

    
    def _reset_task_vars(self):
        self.task_name = None
        # self.task_prompt = None
        self.task_env = None
        self.recording_camera = None

        # Clean up multi-cameras
        if self.multi_cameras is not None:
            for cam in self.multi_cameras:
                try:
                    cam.remove()
                except:
                    pass
            self.multi_cameras = None

        self.observation_space = None
        self.action_space = None
        self.cur_step = 0
        self.done = False
        gc.collect()

    
    def set_task(self, task_name: str, seed: Optional[int] = None):
        # clean up if any
        self._reset_task_vars()

        self.task_name = task_name
        self.task_env = self.rlbench_env.get_task(
            hydra.utils.get_class(
                f"rlbench.tasks.{task_name}."
                f"{''.join([word.capitalize() for word in task_name.split('_')])}"
            )
        )
        if seed is not None:
            np.random.seed(seed)

        # video recording camera setup
        dummy_placeholder = Dummy("cam_cinematic_placeholder")
        self.recording_camera = VisionSensor.create(
            [self.video_resolution, self.video_resolution])
        self.recording_camera.set_pose(dummy_placeholder.get_pose())
        self.recording_camera.set_render_mode(RenderMode.OPENGL3)

        # Multi-camera setup
        if self.use_multi_cam:
            # Get RLBench's existing camera to reuse (move and capture at each position)
            # This works because native RLBench cameras properly sync with scene state
            self.reusable_camera = VisionSensor('cam_over_shoulder_left')

            # Store original camera state for restoration
            self.reusable_camera_original_pos = self.reusable_camera.get_position()
            self.reusable_camera_original_orient = self.reusable_camera.get_orientation()
            self.reusable_camera_original_fov = self.reusable_camera.get_perspective_angle()

            # Generate multi-camera positions
            from policy_research.env.rlbench.camera_utils import generate_camera_positions, compute_look_at_orientation
            self.camera_configs = generate_camera_positions(
                num_cameras=self.num_cameras,
                target_point=self.camera_target,
                radius=self.camera_radius
            )

            # Monkey patch Scene.get_observation to capture multi-cameras
            self._patch_get_observation()

        _, obs = self.task_env.reset()
        # description is a list of prompts (not currently used)

        # setup gym spaces 
        sample_obs_dict = self._extract_obs(obs)
        # sample_obs_dict.pop('prompt', None)  # remove prompt from observation space
        self.observation_space = gymnasium.spaces.Dict({
            key: gymnasium.spaces.Box(
                low=0, high=255, 
                shape=value.shape, dtype=value.dtype
            ) if 'rgb' in key else gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=value.shape, dtype=value.dtype
            ) for key, value in sample_obs_dict.items()
        })
        # self.observation_space.spaces['prompt'] = gymnasium.spaces.Text(
        #     min_length=0, max_length=512,
        #     charset=string.printable
        # )
        action_bounds = self.action_mode.action_bounds()
        self.action_space = gymnasium.spaces.Box(
            low=np.float32(action_bounds[0]), high=np.float32(action_bounds[1]),
            shape=self.rlbench_env.action_shape, dtype=np.float32
        )

    def _patch_get_observation(self):
        """Monkey patch Scene.get_observation to capture multi-cameras by moving reusable camera"""
        scene = self.task_env._scene
        original_get_observation = scene.get_observation
        reusable_cam = self.reusable_camera
        camera_configs = self.camera_configs
        image_size = self.image_size
        fov = self.camera_fov
        env_self = self  # Reference to self for use in patched function

        # Import here to avoid circular import
        from policy_research.env.rlbench.camera_utils import compute_look_at_orientation

        def patched_get_observation():
            # Get original observation first
            obs = original_get_observation()
            
            # Increment frame counter and print progress
            env_self._debug_frame_count += 1
            print(f"\r  [get_demos] Capturing frame {env_self._debug_frame_count}, rendering {len(camera_configs)} cameras...", end="", flush=True)

            # Capture multi-camera images by moving the reusable camera
            all_rgb_images = {}
            for i, (position, target) in enumerate(camera_configs):
                # Move camera to position i
                reusable_cam.set_position(position)

                # Set orientation to look at target
                orientation = compute_look_at_orientation(position, target)
                reusable_cam.set_orientation(orientation)

                # Set FOV
                reusable_cam.set_perspective_angle(fov)

                # Set resolution
                reusable_cam.set_resolution([image_size, image_size])

                # Capture at current scene state
                reusable_cam.handle_explicitly()
                rgb = reusable_cam.capture_rgb()
                # Convert to uint8
                rgb_uint8 = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                obs.misc[f'camera_{i:02d}_rgb'] = rgb_uint8
                all_rgb_images[f'camera_{i:02d}_rgb'] = rgb_uint8

                # Also capture depth
                depth = reusable_cam.capture_depth()
                obs.misc[f'camera_{i:02d}_depth'] = np.float32(depth)

            # Save first frame if debug mode is enabled
            if env_self.debug_save_first_frame and env_self._debug_frame_count == 1:
                env_self._save_debug_images(all_rgb_images)

            return obs

        # Replace the method
        scene.get_observation = patched_get_observation
    
    def _save_debug_images(self, rgb_images: dict):
        """Save debug images during get_demos"""
        if self.debug_dir is None:
            return
        
        os.makedirs(self.debug_dir, exist_ok=True)
        print(f"\n  [Debug] Saving first frame images to {self.debug_dir}")
        
        # Save individual images
        for key, img in rgb_images.items():
            camera_name = key.replace('_rgb', '')
            filename = os.path.join(self.debug_dir, f"episode_{self.debug_episode_idx:03d}_{camera_name}.png")
            pil_img = Image.fromarray(img)
            pil_img.save(filename)
        
        # Create grid image
        self._create_camera_grid(rgb_images)
        print(f"  [Debug] Saved {len(rgb_images)} images + grid")
    
    def _create_camera_grid(self, rgb_images: dict, grid_cols: int = 8):
        """Create a grid of all camera images"""
        from PIL import ImageDraw
        
        sorted_keys = sorted(rgb_images.keys())
        pil_images = [Image.fromarray(rgb_images[k]) for k in sorted_keys]
        
        if not pil_images:
            return
        
        num_images = len(pil_images)
        grid_rows = (num_images + grid_cols - 1) // grid_cols
        
        img_w, img_h = pil_images[0].size
        scale = 0.5
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        grid_w = grid_cols * new_w
        grid_h = grid_rows * new_h
        grid = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(grid)
        
        for idx, pil_img in enumerate(pil_images):
            row = idx // grid_cols
            col = idx % grid_cols
            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x1 = col * new_w
            y1 = row * new_h
            grid.paste(resized, (x1, y1))
            draw.text((x1 + 5, y1 + 5), f"{idx:02d}", fill=(255, 255, 255))
        
        grid_filename = os.path.join(self.debug_dir, f"episode_{self.debug_episode_idx:03d}_grid.png")
        grid.save(grid_filename)

    def _extract_obs(self, rlbench_obs: Observation):
        obs_dict = {}

        # state
        for port_name in self.state_ports:
            state_data = getattr(rlbench_obs, port_name, None)
            if state_data is not None:
                state_data = np.float32(state_data)
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                obs_dict[port_name] = state_data

        # images from cameras
        if self.use_multi_cam:
            # Get multi-camera images from obs.misc (already captured during demo recording)
            for i in range(self.num_cameras):
                rgb_key = f"camera_{i:02d}_rgb"
                depth_key = f"camera_{i:02d}_depth"

                # Images were captured by our patched get_observation() and stored in misc
                if rgb_key in rlbench_obs.misc:
                    obs_dict[rgb_key] = rlbench_obs.misc[rgb_key]
                if depth_key in rlbench_obs.misc:
                    obs_dict[depth_key] = rlbench_obs.misc[depth_key]

            # TODO: Merge point clouds from depth images in obs.misc
            # Currently disabled because we use reusable camera approach
            if self.collect_pointcloud:
                # Pointcloud merging not yet implemented for reusable camera approach
                pass
        else:
            # Use default RLBench cameras
            for name in self.camera_names:
                obs_dict[f"{name}_rgb"] = getattr(rlbench_obs, f"{name}_rgb")

        # # prompt
        # obs_dict['prompt'] = self.task_prompt

        return obs_dict

    def get_camera_params(self):
        """
        Get camera parameters (intrinsics and extrinsics) for all multi-cameras

        Returns:
            List of dicts containing camera parameters, or None if not using multi-cam
        """
        if not self.use_multi_cam or self.multi_cameras is None:
            return None

        intrinsic = get_camera_intrinsics(self.image_size, self.camera_fov)
        camera_params = []

        for i, cam in enumerate(self.multi_cameras):
            extrinsic = get_camera_extrinsics(cam)
            pose = cam.get_pose()
            position = pose[:3]
            quaternion = pose[3:]

            params = {
                'camera_name': f'camera_{i:02d}',
                'camera_id': i,
                'intrinsic_matrix': intrinsic.tolist(),
                'extrinsic_matrix': extrinsic.tolist(),
                'position': position.tolist(),
                'quaternion': quaternion.tolist(),  # [qx, qy, qz, qw]
                'fov': float(self.camera_fov),
                'image_size': int(self.image_size),
            }
            camera_params.append(params)

        return camera_params
    

    def render(self, mode='rgb_array'):
        # NOTE: only for video recording wrapper
        assert mode == 'rgb_array'
        frame = self.recording_camera.capture_rgb()
        frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
        return frame
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        _, obs = self.task_env.reset()
        self.cur_step = 0
        self.done = False
        # return self._extract_obs(obs), {'prompt': self.task_prompt}
        return self._extract_obs(obs), {}
    
    
    def step(self, action: np.ndarray):
        obs, reward, terminated = self.task_env.step(action)
        self.cur_step += 1
        self.done = self.done or terminated or (reward >= 1) \
            or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, {}


    def close(self) -> None:
        # Restore reusable camera to original state if using multi-cam
        if self.use_multi_cam and hasattr(self, 'reusable_camera'):
            try:
                self.reusable_camera.set_position(self.reusable_camera_original_pos)
                self.reusable_camera.set_orientation(self.reusable_camera_original_orient)
                self.reusable_camera.set_perspective_angle(self.reusable_camera_original_fov)
            except:
                pass

        self.rlbench_env.shutdown()
