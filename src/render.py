#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to render novel views from a trained Dynamic NeRF model.
"""

import os
import argparse
import torch
import numpy as np
import imageio
from tqdm import tqdm

from models.nerf import NeRF
from models.dynamic_nerf import DynamicNeRF
from data.dataset import DynamicNeRFDataset
from utils.config import parse_args_and_config
from utils.ray_utils import get_rays


def create_camera_trajectory(dataset, trajectory_type='circle', num_frames=60, radius=4.0):
    """
    Create a camera trajectory for rendering novel views.
    
    Args:
        dataset: Dataset object.
        trajectory_type: Type of trajectory ('circle', 'spiral', etc.).
        num_frames: Number of frames to generate.
        radius: Radius of the trajectory.
        
    Returns:
        List of camera poses.
    """
    if trajectory_type == 'circle':
        # Create a circular trajectory around the center
        poses = []
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.0  # Fixed height
            
            # Look-at matrix
            center = np.array([0, 0, 0])
            eye = np.array([x, y, z])
            up = np.array([0, 1, 0])
            
            f = center - eye
            f = f / np.linalg.norm(f)
            r = np.cross(up, f)
            r = r / np.linalg.norm(r)
            u = np.cross(f, r)
            
            pose = np.eye(4)
            pose[:3, 0] = r
            pose[:3, 1] = u
            pose[:3, 2] = f
            pose[:3, 3] = eye
            
            poses.append(pose)
    
    elif trajectory_type == 'spiral':
        # Create a spiral trajectory
        poses = []
        for i in range(num_frames):
            angle = 2 * np.pi * i / (num_frames // 2)  # Two rotations
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = -0.5 + i / num_frames  # Move upward
            
            # Look-at matrix
            center = np.array([0, 0, 0])
            eye = np.array([x, y, z])
            up = np.array([0, 1, 0])
            
            f = center - eye
            f = f / np.linalg.norm(f)
            r = np.cross(up, f)
            r = r / np.linalg.norm(r)
            u = np.cross(f, r)
            
            pose = np.eye(4)
            pose[:3, 0] = r
            pose[:3, 1] = u
            pose[:3, 2] = f
            pose[:3, 3] = eye
            
            poses.append(pose)
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    return poses


def render_novel_views(model, poses, hwf, time_steps=None, device='cpu', chunk_size=4096):
    """
    Render novel views from a trajectory of camera poses.
    
    Args:
        model: NeRF or DynamicNeRF model.
        poses: List of camera poses.
        hwf: Tuple of (height, width, focal).
        time_steps: Time steps for dynamic rendering.
        device: Device to render on.
        chunk_size: Chunk size for batched inference.
        
    Returns:
        List of rendered images.
    """
    H, W, focal = hwf
    rendered_imgs = []
    rendered_depths = []
    is_dynamic = time_steps is not None
    
    for i, pose in enumerate(tqdm(poses, desc="Rendering views")):
        # Get time step for dynamic model
        time_step = None
        if is_dynamic:
            # Use provided time steps or interpolate between 0 and 1
            if isinstance(time_steps, (list, np.ndarray)):
                time_step = time_steps[i % len(time_steps)]
            else:
                time_step = i / (len(poses) - 1)
        
        # Generate rays for this camera pose
        rays_o, rays_d = get_rays(H, W, focal, pose)
        
        # Flatten rays for batched inference
        rays_o = rays_o.reshape(-1, 3).to(device)
        rays_d = rays_d.reshape(-1, 3).to(device)
        
        # Render in chunks to avoid OOM
        rgb_chunks = []
        depth_chunks = []
        
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_rays_o = rays_o[i:i+chunk_size]
                chunk_rays_d = rays_d[i:i+chunk_size]
                
                if is_dynamic:
                    # Prepare time input (same time for all rays in the chunk)
                    chunk_time = torch.full((chunk_rays_o.shape[0], 1), time_step, device=device)
                    rgb_chunk, depth_chunk = model(chunk_rays_o, chunk_rays_d, chunk_time)
                else:
                    rgb_chunk, depth_chunk = model(chunk_rays_o, chunk_rays_d)
                
                rgb_chunks.append(rgb_chunk.cpu())
                depth_chunks.append(depth_chunk.cpu())
        
        # Combine chunks
        rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3).numpy()
        depth = torch.cat(depth_chunks, dim=0).reshape(H, W).numpy()
        
        # Clip and convert to image format
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        
        # Normalize depth for visualization
        depth_viz = depth / (depth.max() + 1e-10)
        depth_viz = (depth_viz * 255).astype(np.uint8)
        
        rendered_imgs.append(rgb)
        rendered_depths.append(depth_viz)
    
    return rendered_imgs, rendered_depths


def render_novel_times(model, pose, hwf, time_steps, device='cpu', chunk_size=4096):
    """
    Render the same view at different time steps.
    
    Args:
        model: DynamicNeRF model.
        pose: Camera pose.
        hwf: Tuple of (height, width, focal).
        time_steps: List of time steps to render.
        device: Device to render on.
        chunk_size: Chunk size for batched inference.
        
    Returns:
        List of rendered images at different time steps.
    """
    H, W, focal = hwf
    rendered_imgs = []
    rendered_depths = []
    
    # Generate rays for this camera pose
    rays_o, rays_d = get_rays(H, W, focal, pose)
    
    # Flatten rays for batched inference
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)
    
    for time_step in tqdm(time_steps, desc="Rendering time steps"):
        # Render in chunks to avoid OOM
        rgb_chunks = []
        depth_chunks = []
        
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_rays_o = rays_o[i:i+chunk_size]
                chunk_rays_d = rays_d[i:i+chunk_size]
                
                # Prepare time input (same time for all rays in the chunk)
                chunk_time = torch.full((chunk_rays_o.shape[0], 1), time_step, device=device)
                rgb_chunk, depth_chunk = model(chunk_rays_o, chunk_rays_d, chunk_time)
                
                rgb_chunks.append(rgb_chunk.cpu())
                depth_chunks.append(depth_chunk.cpu())
        
        # Combine chunks
        rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3).numpy()
        depth = torch.cat(depth_chunks, dim=0).reshape(H, W).numpy()
        
        # Clip and convert to image format
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        
        # Normalize depth for visualization
        depth_viz = depth / (depth.max() + 1e-10)
        depth_viz = (depth_viz * 255).astype(np.uint8)
        
        rendered_imgs.append(rgb)
        rendered_depths.append(depth_viz)
    
    return rendered_imgs, rendered_depths


def main():
    """Main function for rendering novel views from a trained model."""
    parser = argparse.ArgumentParser(description="Render novel views from a trained NeRF model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the model configuration file")
    parser.add_argument('--output_dir', type=str, default='renders', help="Directory to save renders")
    parser.add_argument('--trajectory', type=str, default='circle', choices=['circle', 'spiral'], help="Camera trajectory type")
    parser.add_argument('--num_frames', type=int, default=60, help="Number of frames to render")
    parser.add_argument('--time_steps', type=int, default=None, help="Number of time steps to render (for dynamic NeRF)")
    parser.add_argument('--render_depth', action='store_true', help="Render depth maps")
    parser.add_argument('--render_video', action='store_true', help="Create video from rendered frames")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = argparse.Namespace(**yaml.safe_load(f))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if config.model == 'nerf':
        model = NeRF(
            pos_encoder_dims=config.pos_encoder_dims,
            dir_encoder_dims=config.dir_encoder_dims,
            hidden_dims=config.hidden_dims,
        ).to(device)
    elif config.model == 'dynamic_nerf':
        model = DynamicNeRF(
            pos_encoder_dims=config.pos_encoder_dims,
            dir_encoder_dims=config.dir_encoder_dims,
            time_encoder_dims=config.time_encoder_dims,
            hidden_dims=config.hidden_dims,
            use_attention=config.use_attention,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config.model}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset for camera intrinsics
    dataset = DynamicNeRFDataset(
        root_dir=config.data_dir,
        scene_name=config.scene,
        split='val',
        img_size=config.img_size,
    )
    hwf = dataset.hwf  # height, width, focal length
    
    # Create camera trajectory
    poses = create_camera_trajectory(
        dataset, 
        trajectory_type=args.trajectory,
        num_frames=args.num_frames,
    )
    
    # Determine time steps
    time_steps = None
    if config.model == 'dynamic_nerf':
        if args.time_steps is None:
            # Use a single time step in the middle of the sequence
            time_steps = 0.5
        else:
            # Create evenly spaced time steps from 0 to 1
            time_steps = np.linspace(0, 1, args.time_steps)
    
    # Render novel views
    print("Rendering novel views...")
    rendered_imgs, rendered_depths = render_novel_views(
        model, poses, hwf, time_steps, device
    )
    
    # Save rendered images
    print(f"Saving renders to {args.output_dir}...")
    for i, img in enumerate(rendered_imgs):
        imageio.imwrite(os.path.join(args.output_dir, f'rgb_{i:04d}.png'), img)
    
    if args.render_depth:
        for i, depth in enumerate(rendered_depths):
            imageio.imwrite(os.path.join(args.output_dir, f'depth_{i:04d}.png'), depth)
    
    # Create video if requested
    if args.render_video:
        print("Creating video...")
        imageio.mimwrite(os.path.join(args.output_dir, 'rgb_video.mp4'), rendered_imgs, fps=30, quality=8)
        
        if args.render_depth:
            imageio.mimwrite(os.path.join(args.output_dir, 'depth_video.mp4'), rendered_depths, fps=30, quality=8)
    
    # For dynamic NeRF, also render time sequence
    if config.model == 'dynamic_nerf' and args.time_steps > 1:
        # Pick a reference pose (middle of the trajectory)
        ref_pose = poses[len(poses) // 2]
        
        # Render the same view at different time steps
        rendered_time_imgs, rendered_time_depths = render_novel_times(
            model, ref_pose, hwf, np.linspace(0, 1, args.time_steps), device
        )
        
        # Save time sequence images
        time_dir = os.path.join(args.output_dir, 'time_sequence')
        os.makedirs(time_dir, exist_ok=True)
        
        for i, img in enumerate(rendered_time_imgs):
            imageio.imwrite(os.path.join(time_dir, f'time_{i:04d}.png'), img)
        
        if args.render_depth:
            for i, depth in enumerate(rendered_time_depths):
                imageio.imwrite(os.path.join(time_dir, f'time_depth_{i:04d}.png'), depth)
        
        # Create time sequence video
        if args.render_video:
            imageio.mimwrite(os.path.join(time_dir, 'time_video.mp4'), rendered_time_imgs, fps=10, quality=8)
            
            if args.render_depth:
                imageio.mimwrite(os.path.join(time_dir, 'time_depth_video.mp4'), rendered_time_depths, fps=10, quality=8)
    
    print("Rendering completed!")


if __name__ == "__main__":
    main() 