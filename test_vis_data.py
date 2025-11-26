import numpy as np
import imageio.v3 as iio
from pathlib import Path
from policy_research.common.replay_buffer import ReplayBuffer

def extract_images_from_zarr(zarr_path: str, output_dir: str, fps: int = 30):
    """
    从 zarr 文件中提取图像并使用 imageio.v3 保存为 mp4 视频
    
    Args:
        zarr_path: zarr 文件路径
        output_dir: 输出文件夹路径
        fps: 视频帧率，默认 30
    """
    # 打开 zarr 文件
    print(f"正在打开 zarr 文件: {zarr_path}")
    buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
    
    # 找到所有图像键（以 _rgb 结尾）
    image_keys = [key for key in buffer.keys() if key.endswith('_rgb')]
    
    if len(image_keys) == 0:
        print("警告: 未找到任何图像数据（以 _rgb 结尾的键）")
        return
    
    print(f"找到 {len(image_keys)} 个图像键: {image_keys}")
    print(f"总共有 {buffer.n_episodes} 个 episode, {buffer.n_steps} 个时间步")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取 episode 边界
    episode_ends = buffer.episode_ends[:]
    
    # 遍历每个 episode
    for ep_idx in range(buffer.n_episodes):
        print(f"\n处理 episode {ep_idx + 1}/{buffer.n_episodes}")
        
        # 获取当前 episode 的数据范围
        start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        end_idx = episode_ends[ep_idx]
        
        # 为每个 episode 创建子文件夹
        ep_dir = output_path / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(exist_ok=True)
        
        # 遍历每个图像键（相机）
        for img_key in image_keys:
            # 获取当前 episode 的图像数据
            images = buffer.data[img_key][start_idx:end_idx]
            
            # 准备图像数组
            image_list = []
            for t_idx in range(len(images)):
                img = images[t_idx]
                
                # 确保图像是 uint8 格式
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                
                image_list.append(img)
            
            # 将图像列表转换为 numpy 数组 (T, H, W, C)
            video_array = np.array(image_list)
            
            # 使用 imageio.v3 保存为 mp4 视频
            video_filename = ep_dir / f"{img_key}.mp4"
            iio.imwrite(video_filename, video_array, fps=fps)
            print(f"  已保存: {video_filename} ({len(image_list)} 帧)")
        
        print(f"  Episode {ep_idx + 1} 完成: {end_idx - start_idx} 个时间步")
    
    print(f"\n所有视频已保存到: {output_path}")

if __name__ == "__main__":
    zarr_path = "test/close_box_N1_multicam40.zarr"
    output_dir = "test/close_box_N1_multicam40_data"
    
    extract_images_from_zarr(zarr_path, output_dir, fps=30)

