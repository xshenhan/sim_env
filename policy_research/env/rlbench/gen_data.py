if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import click
import numpy as np
import copy
import zarr
import multiprocessing as mp
import json
from PIL import Image
from tqdm import tqdm

from policy_research.common.replay_buffer import ReplayBuffer
from policy_research.common.input_util import wait_user_input
from policy_research.env.rlbench.env import RlbenchEnv
from policy_research.env.rlbench.factory import get_subtasks

from typing import List, Dict, Optional


def generate_one_episode_data(
    env: RlbenchEnv, 
    pbar: Optional[tqdm] = None,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    episode_idx: int = 0,
):
    # 提示用户正在生成 demo（这一步可能很慢）
    if pbar:
        pbar.set_postfix({'status': 'generating demo...'})
        pbar.refresh()
    
    # 设置 env 的 debug 属性，这样在 get_demos 期间可以实时保存第一帧
    if debug and debug_dir is not None:
        env.debug_save_first_frame = True
        env.debug_dir = debug_dir
        env.debug_episode_idx = episode_idx
        env._debug_frame_count = 0  # 重置帧计数器
    
    demo = env.task_env.get_demos(1, live_demos=True, max_attempts=1)
    print()  # 换行，因为 patched_get_observation 使用了 \r
    
    # 关闭 debug 模式
    env.debug_save_first_frame = False
    
    demo = demo[0]      # only one episode
    this_total_count = len(demo)
    this_data_collected = {'ee_action': [], 'joint_action': []}

    if pbar:
        pbar.set_postfix({'status': f'processing 0/{this_total_count} steps'})
        pbar.refresh()

    for step_idx, obs in enumerate(demo):
        obs_dict = env._extract_obs(obs)
        for k, v in obs_dict.items():
            if k not in this_data_collected:
                this_data_collected[k] = []
            this_data_collected[k].append(v)
        # absolute joint position control
        action = np.append(obs.joint_positions, obs.gripper_joint_positions)
        this_data_collected['joint_action'].append(action)
        # absolute ee pose control
        action = np.append(obs.gripper_pose, obs.gripper_joint_positions)
        this_data_collected['ee_action'].append(action)

        # Update progress bar with step progress
        if pbar:
            pbar.set_postfix({'status': f'processing {step_idx+1}/{this_total_count} steps'})
            pbar.refresh()

    this_data_collected['delta_joint_action'] = np.diff(
        this_data_collected['joint_action'], axis=0,
        append=np.array(this_data_collected['joint_action'])[-1:]
    )

    return this_total_count, this_data_collected


def save_debug_first_frame(
    obs_dict: Dict,
    debug_dir: str,
    episode_idx: int,
    use_multi_cam: bool = False,
):
    """
    保存第一帧的所有相机图像用于调试（使用 PIL 避免 Qt 依赖）
    """
    os.makedirs(debug_dir, exist_ok=True)
    
    # 找出所有 RGB 图像
    rgb_keys = [k for k in obs_dict.keys() if k.endswith('_rgb')]
    
    if not rgb_keys:
        print(f"Warning: No RGB images found in obs_dict for episode {episode_idx}")
        return
    
    # 保存每个相机的图像
    for key in rgb_keys:
        img = obs_dict[key]
        # 确保是 uint8 格式
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # 使用 PIL 保存图像
        pil_img = Image.fromarray(img)
        camera_name = key.replace('_rgb', '')
        filename = os.path.join(debug_dir, f"episode_{episode_idx:03d}_{camera_name}.png")
        pil_img.save(filename)
    
    # 如果是多相机模式，额外创建一个拼接图
    if use_multi_cam and len(rgb_keys) > 1:
        create_camera_grid(obs_dict, rgb_keys, debug_dir, episode_idx)
    
    print(f"  [Debug] Saved {len(rgb_keys)} first-frame images to {debug_dir}")


def create_camera_grid(
    obs_dict: Dict,
    rgb_keys: List[str],
    debug_dir: str,
    episode_idx: int,
    grid_cols: int = 8,
):
    """
    将多个相机图像拼接成网格图（使用 PIL 避免 Qt 依赖）
    """
    from PIL import ImageDraw, ImageFont
    
    pil_images = []
    for key in sorted(rgb_keys):
        img = obs_dict[key]
        # 确保是 uint8 格式
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        pil_images.append(Image.fromarray(img))
    
    if not pil_images:
        return
    
    # 计算网格布局
    num_images = len(pil_images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # 获取图像尺寸（假设所有图像大小相同）
    img_w, img_h = pil_images[0].size
    
    # 缩小图像以便于查看
    scale = 0.5
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    
    # 创建网格画布
    grid_w = grid_cols * new_w
    grid_h = grid_rows * new_h
    grid = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid)
    
    # 填充图像
    for idx, pil_img in enumerate(pil_images):
        row = idx // grid_cols
        col = idx % grid_cols
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x1 = col * new_w
        y1 = row * new_h
        grid.paste(resized, (x1, y1))
        
        # 添加相机编号标签
        draw.text((x1 + 5, y1 + 5), f"{idx:02d}", fill=(255, 255, 255))
    
    # 保存网格图
    grid_filename = os.path.join(debug_dir, f"episode_{episode_idx:03d}_grid.png")
    grid.save(grid_filename)


def worker(
    worker_id: int,
    task_names: List[str],
    task_idices: List[int],
    sensors: List[str],
    data_dtype: Dict,
    save_subtasks: bool,
    use_multi_cam: bool = False,
    num_cameras: int = 40,
    camera_target: List[float] = [0.218, 0.0, 1.917],
    camera_radius: float = 1.5,
    camera_fov: float = 45.0,
    debug: bool = False,
    debug_dir: Optional[str] = None,
) -> List[ReplayBuffer]:
    seed = os.getpid()
    env = RlbenchEnv(
        camera_names=sensors if not use_multi_cam else [],
        seed=seed,
        use_multi_cam=use_multi_cam,
        num_cameras=num_cameras,
        camera_target=camera_target,
        camera_radius=camera_radius,
        camera_fov=camera_fov,
    )
    replay_buffers = [
        ReplayBuffer.create_empty_zarr() for _ in range(
            (len(task_names) + 1) if save_subtasks else 1
        )
    ]

    global_episode_idx = 0
    idx = 0
    while idx < len(task_idices):
        task_idx = task_idices[idx]
        task_name = task_names[task_idx]
        env.set_task(task_name)
        
        try:
            this_total_count, this_data_collected = generate_one_episode_data(
                env,
                debug=debug,
                debug_dir=debug_dir,
                episode_idx=global_episode_idx,
            )
        except RuntimeError as e:
            if str(e) == 'Could not collect demos. Maybe a problem with the task?':
                # skip this episode
                print(f"Worker {worker_id} - [{idx+1}/{len(task_idices)}]: task: {env.task_name}, failed")
                idx += 1  # 即使失败也要增加索引，避免无限循环
                continue
            else:
                raise e

        idx += 1
        global_episode_idx += 1
        # dtype conversion
        for k, v in this_data_collected.items():
            this_data_collected[k] = np.array(v, dtype=data_dtype.get(k, 'float32'))
        # copy trial data to total data collected
        replay_buffers[-1].add_episode(copy.deepcopy(this_data_collected))
        # copy trial data to subtask data collected if needed
        if save_subtasks:
            replay_buffers[task_idx].add_episode(copy.deepcopy(this_data_collected))
        print(
            f"Worker {worker_id} - [{idx}/{len(task_idices)}]: "
            f"task: {env.task_name}, steps: {this_total_count}"
        )

    env.close()
    return replay_buffers


def multiprocess_core(
    num_workers: int,
    task_name: str,
    sensors: str,
    save_subtasks: bool,
    num_tasks: int,
    num_eps_per_task: int,
    use_multi_cam: bool = False,
    num_cameras: int = 40,
    camera_target: List[float] = [0.218, 0.0, 1.917],
    camera_radius: float = 1.5,
    camera_fov: float = 45.0,
    debug: bool = False,
    debug_dir: Optional[str] = None,
) -> List[ReplayBuffer]:
    replay_buffers = [
        ReplayBuffer.create_empty_zarr() for _ in range(
            (num_tasks + 1) if save_subtasks else 1
        )
    ]

    # Update dtype for multi-camera
    if use_multi_cam:
        data_dtype = {
            **{f"camera_{i:02d}_rgb": 'uint8' for i in range(num_cameras)},
            **{f"camera_{i:02d}_depth": 'float32' for i in range(num_cameras)},
            'pointcloud': 'float32',
        }
    else:
        data_dtype = {
            **{f"{sensor}_rgb": 'uint8' for sensor in sensors},
            **{f"{sensor}_depth": 'float32' for sensor in sensors},
        }

    # collect data
    task_indices = list(range(num_tasks)) * num_eps_per_task
    with mp.Pool(num_workers) as pool:
        # 使用轮询分配方式，确保任务均匀分配
        args = [
            (
                i,
                get_subtasks(task_name),
                task_indices[i::num_workers],  # 轮询分配：worker 0 处理索引 0,4,8...，worker 1 处理索引 1,5,9...
                sensors,
                data_dtype,
                save_subtasks,
                use_multi_cam,
                num_cameras,
                camera_target,
                camera_radius,
                camera_fov,
                debug,
                os.path.join(debug_dir, f"worker_{i}") if debug_dir else None,
            ) for i in range(num_workers)
        ]
        results = pool.starmap(worker, args)
        pool.close()
        pool.join()

    # merge data
    for buff_idx in range(len(replay_buffers)):
        for worker_idx in range(num_workers):
            for eps_idx in range(results[worker_idx][buff_idx].n_episodes):
                replay_buffers[buff_idx].add_episode(
                    results[worker_idx][buff_idx].get_episode(eps_idx)
                )
    return replay_buffers


def uniprocess_core(
    task_name: str,
    sensors: str,
    save_subtasks: bool,
    num_tasks: int,
    num_eps_per_task: int,
    use_multi_cam: bool = False,
    num_cameras: int = 40,
    camera_target: List[float] = [0.218, 0.0, 1.917],
    camera_radius: float = 1.5,
    camera_fov: float = 45.0,
    debug: bool = False,
    debug_dir: Optional[str] = None,
) -> List[ReplayBuffer]:
    replay_buffers = [
        ReplayBuffer.create_empty_zarr() for _ in range(
            (num_tasks + 1) if save_subtasks else 1
        )
    ]

    # Update dtype for multi-camera
    if use_multi_cam:
        data_dtype = {
            **{f"camera_{i:02d}_rgb": 'uint8' for i in range(num_cameras)},
            **{f"camera_{i:02d}_depth": 'float32' for i in range(num_cameras)},
            'pointcloud': 'float32',
        }
    else:
        data_dtype = {
            **{f"{sensor}_rgb": 'uint8' for sensor in sensors},
            **{f"{sensor}_depth": 'float32' for sensor in sensors},
        }

    env = RlbenchEnv(
        camera_names=sensors if not use_multi_cam else [],
        use_multi_cam=use_multi_cam,
        num_cameras=num_cameras,
        camera_target=camera_target,
        camera_radius=camera_radius,
        camera_fov=camera_fov,
    )

    # Iterate through subtasks without shadowing the task_name parameter
    total_episodes = num_tasks * num_eps_per_task
    pbar = tqdm(total=total_episodes, desc="Collecting data", unit="episode")

    global_episode_idx = 0
    for task_idx, subtask_name in enumerate(get_subtasks(task_name)):
        env.set_task(subtask_name)

        episode_idx = 0
        while episode_idx < num_eps_per_task:
            try:
                this_total_count, this_data_collected = generate_one_episode_data(
                    env, 
                    pbar,
                    debug=debug,
                    debug_dir=debug_dir,
                    episode_idx=global_episode_idx,
                )
            except RuntimeError as e:
                if str(e) == 'Could not collect demos. Maybe a problem with the task?':
                    # skip this episode
                    pbar.write(f"Episode {episode_idx + num_eps_per_task * task_idx + 1} failed. Skip")
                    continue
                else:
                    raise e

            episode_idx += 1
            global_episode_idx += 1
            # dtype conversion
            for k, v in this_data_collected.items():
                this_data_collected[k] = np.array(v, dtype=data_dtype.get(k, 'float32'))
            # copy trial data to total data collected
            replay_buffers[-1].add_episode(copy.deepcopy(this_data_collected))
            # copy trial data to subtask data collected if needed
            if save_subtasks:
                replay_buffers[task_idx].add_episode(copy.deepcopy(this_data_collected))

            pbar.set_description(f"Task: {env.task_name}, steps: {this_total_count}")
            pbar.update(1)

    pbar.close()

    env.close()
    return replay_buffers


@click.command()
@click.option('-t', '--task_name', type=str, required=True)
@click.option('-d', '--root_data_dir', type=str, default='data/rlbench')
@click.option('-s', '--save_dir', type=str, default=None)
@click.option('-c', '--num_episodes', type=int, default=10)
@click.option('-o', '--sensors', multiple=True, type=str, default=(
    'left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'))
@click.option('-w', '--num_workers', type=int, default=1)
@click.option('--save_subtasks', is_flag=True)
@click.option('--use_multi_cam', is_flag=True, help='Use multi-camera setup (disables default cameras)')
@click.option('--num_cameras', type=int, default=40, help='Number of cameras in multi-camera mode')
@click.option('--camera_target', type=float, nargs=3, default=[0.2500, 0.0000, 0.7520],
              help='Camera look-at target point [x, y, z]')
@click.option('--camera_radius', type=float, default=.5, help='Camera distance from target')
@click.option('--camera_fov', type=float, default=40.0, help='Camera field of view (degrees)')
@click.option('--debug', is_flag=True, help='Save first frame images for debugging')
def gen_rlbench_data(
    task_name: str,
    root_data_dir: str,
    save_dir: str,
    num_episodes: int,
    sensors: str,
    num_workers: int,
    save_subtasks: bool,
    use_multi_cam: bool,
    num_cameras: int,
    camera_target: tuple,
    camera_radius: float,
    camera_fov: float,
    debug: bool,
):  
    assert num_workers > 0

    # Convert camera_target to list
    camera_target = list(camera_target)

    # Update save_dir name for multi-camera mode
    if save_dir is None:
        suffix = f'_multicam{num_cameras}' if use_multi_cam else ''
        save_dir = os.path.join(root_data_dir, f'{task_name}_N{num_episodes}{suffix}.zarr')

    # Print configuration
    if use_multi_cam:
        print(f"\n{'='*60}")
        print(f"Multi-Camera Configuration:")
        print(f"  Number of cameras: {num_cameras}")
        print(f"  Camera target: {camera_target}")
        print(f"  Camera radius: {camera_radius}m")
        print(f"  Camera FOV: {camera_fov}°")
        print(f"{'='*60}\n")

    if os.path.exists(save_dir):
        keypress = wait_user_input(
            valid_input=lambda key: key in ['', 'y', 'n'],
            prompt=f"{save_dir} already exists. Overwrite? [y/`n`]: ",
            default='n'
        )
        if keypress == 'n':
            print("Abort")
            return
        else:
            os.system(f"rm -rf {save_dir}")
    pathlib.Path(save_dir).mkdir(parents=True)
    save_dir = [save_dir,]

    # distribute episodes across different environments
    num_tasks = len(get_subtasks(task_name))
    assert num_episodes % num_tasks == 0
    num_eps_per_task = num_episodes // num_tasks
    assert not save_subtasks or num_tasks > 1

    # create subtask directories
    if save_subtasks:
        for sname in get_subtasks(task_name):
            sdir = os.path.join(root_data_dir, f'{sname}_N{num_eps_per_task}.zarr')
            if os.path.exists(sdir):
                keypress = wait_user_input(
                    valid_input=lambda key: key in ['', 'y', 'n'],
                    prompt=f"{sdir} already exists. Overwrite? [y/`n`]: ",
                    default='n'
                )
                if keypress == 'n':
                    print("Abort")
                    return
                else:
                    os.system(f"rm -rf {sdir}")
            os.mkdir(sdir)
            save_dir.append(sdir)
    save_dir = save_dir[1:] + save_dir[:1]

    # Setup debug directory
    debug_dir = None
    if debug:
        debug_dir = os.path.join(root_data_dir, f'{task_name}_debug_frames')
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[Debug mode] First frame images will be saved to: {debug_dir}")

    # collect data
    if num_workers == 1:
        replay_buffers = uniprocess_core(
            task_name,
            sensors,
            save_subtasks,
            num_tasks,
            num_eps_per_task,
            use_multi_cam,
            num_cameras,
            camera_target,
            camera_radius,
            camera_fov,
            debug,
            debug_dir,
        )
    elif num_workers > 1:
        mp.set_start_method('spawn', force=True)
        replay_buffers = multiprocess_core(
            num_workers,
            task_name,
            sensors,
            save_subtasks,
            num_tasks,
            num_eps_per_task,
            use_multi_cam,
            num_cameras,
            camera_target,
            camera_radius,
            camera_fov,
            debug,
            debug_dir,
        )

    # report
    for buff, sdir in zip(replay_buffers, save_dir):
        print('-' * 50)
        print(f"{sdir}: \n{buff}")

    # save data collected
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    for i, buff in enumerate(replay_buffers):
        buff.save_to_path(save_dir[i], compressors=compressor)

    # Save camera parameters if using multi-camera
    if use_multi_cam:
        # Create a temporary env to get camera params
        temp_env = RlbenchEnv(
            task_name=task_name,
            use_multi_cam=True,
            num_cameras=num_cameras,
            camera_target=camera_target,
            camera_radius=camera_radius,
            camera_fov=camera_fov,
        )
        camera_params = temp_env.get_camera_params()
        temp_env.close()

        # Save camera params to JSON
        camera_params_file = save_dir[0].replace('.zarr', '_camera_params.json')
        with open(camera_params_file, 'w') as f:
            json.dump({
                'num_cameras': num_cameras,
                'camera_target': camera_target,
                'camera_radius': camera_radius,
                'camera_fov': camera_fov,
                'cameras': camera_params
            }, f, indent=2)
        print(f"\nSaved camera parameters to: {camera_params_file}")


if __name__ == '__main__':
    gen_rlbench_data()
