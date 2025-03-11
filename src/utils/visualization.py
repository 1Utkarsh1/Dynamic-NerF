#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for Dynamic NeRF.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from tqdm import tqdm

from utils.ray_utils import get_rays, render_rays, create_spiral_poses


def visualize_results(model, dataset, device, output_dir='viz', num_views=5, 
                     num_time_steps=10, dynamic=True):
    """
    Visualize results by rendering novel views and time steps.
    
    Args:
        model: NeRF or DynamicNeRF model.
        dataset: Dataset object.
        device: Device to render on.
        output_dir (str): Directory to save visualizations.
        num_views (int): Number of novel views to render.
        num_time_steps (int): Number of time steps to render.
        dynamic (bool): Whether the model is dynamic.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get camera parameters
    H, W, focal = dataset.hwf
    
    # Create spiral path for camera
    center_pose = dataset.poses[len(dataset.poses) // 2]
    radii = (0.5, 0.5, 0.5)  # Radii of spiral path
    focus_depth = 4.0  # Focus depth
    novel_poses = create_spiral_poses(radii, focus_depth, n_poses=num_views)
    
    # Create time steps
    if dynamic:
        time_steps = torch.linspace(0, 1, num_time_steps, device=device).unsqueeze(1)  # (T, 1)
    
    # Render from fixed viewpoint at different time steps
    if dynamic:
        render_temporal_sequence(model, novel_poses[0], H, W, focal, time_steps, 
                                device, os.path.join(output_dir, 'temporal'))
    
    # Render from different viewpoints at fixed time steps
    if dynamic:
        # Use middle time step
        fixed_time = torch.tensor([[0.5]], device=device)
        render_spatial_sequence(model, novel_poses, H, W, focal, fixed_time, 
                               device, os.path.join(output_dir, 'spatial_dynamic'))
    else:
        render_spatial_sequence(model, novel_poses, H, W, focal, None, 
                               device, os.path.join(output_dir, 'spatial_static'))
    
    # Create comparison with ground truth
    if len(dataset) > 0:
        compare_with_ground_truth(model, dataset, device, os.path.join(output_dir, 'comparison'))


def render_image(model, pose, H, W, focal, time_step=None, device='cpu', chunk_size=4096):
    """
    Render a full image using the model.
    
    Args:
        model: NeRF or DynamicNeRF model.
        pose: Camera pose matrix.
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length.
        time_step (torch.Tensor, optional): Time step tensor of shape (1, 1).
        device (str): Device to render on.
        chunk_size (int): Chunk size for batched inference.
        
    Returns:
        dict: Dictionary with rendered outputs (RGB, depth).
    """
    # Generate rays for this camera pose
    rays_o, rays_d = get_rays(H, W, focal, pose)
    
    # Flatten rays
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)
    
    # Expand time step if provided
    if time_step is not None:
        time_indices = time_step.expand(rays_o.shape[0], -1).to(device)
    else:
        time_indices = None
    
    # Render rays
    results = render_rays(model, rays_o, rays_d, time_indices, chunk_size=chunk_size)
    
    # Reshape results to image dimensions
    rgb = results['rgb'].reshape(H, W, 3).cpu().numpy()
    depth = results['depth'].reshape(H, W).cpu().numpy()
    
    return {
        'rgb': rgb,
        'depth': depth,
    }


def render_temporal_sequence(model, pose, H, W, focal, time_steps, device, output_dir):
    """
    Render a temporal sequence from a fixed viewpoint.
    
    Args:
        model: DynamicNeRF model.
        pose: Camera pose matrix.
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length.
        time_steps (torch.Tensor): Time steps of shape (T, 1).
        device (str): Device to render on.
        output_dir (str): Directory to save sequence.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert pose to tensor if needed
    if isinstance(pose, np.ndarray):
        pose = torch.FloatTensor(pose).to(device)
    
    # Render at each time step
    rgb_frames = []
    depth_frames = []
    
    for i, time_step in enumerate(tqdm(time_steps, desc="Rendering temporal sequence")):
        # Render image
        results = render_image(model, pose, H, W, focal, time_step, device)
        
        # Convert to 8-bit RGB
        rgb = (results['rgb'] * 255).astype(np.uint8)
        
        # Normalize depth for visualization
        depth = results['depth']
        depth_viz = (depth - depth.min()) / (depth.max() - depth.min())
        depth_viz = (depth_viz * 255).astype(np.uint8)
        
        # Save images
        imageio.imwrite(os.path.join(output_dir, f'rgb_{i:03d}.png'), rgb)
        imageio.imwrite(os.path.join(output_dir, f'depth_{i:03d}.png'), depth_viz)
        
        # Append to frames for video
        rgb_frames.append(rgb)
        depth_frames.append(depth_viz)
    
    # Create videos
    imageio.mimwrite(os.path.join(output_dir, 'rgb_video.mp4'), rgb_frames, fps=10, quality=8)
    imageio.mimwrite(os.path.join(output_dir, 'depth_video.mp4'), depth_frames, fps=10, quality=8)


def render_spatial_sequence(model, poses, H, W, focal, time_step, device, output_dir):
    """
    Render a spatial sequence from different viewpoints at a fixed time step.
    
    Args:
        model: NeRF or DynamicNeRF model.
        poses (list): List of camera pose matrices.
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length.
        time_step (torch.Tensor, optional): Fixed time step of shape (1, 1).
        device (str): Device to render on.
        output_dir (str): Directory to save sequence.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Render at each pose
    rgb_frames = []
    depth_frames = []
    
    for i, pose in enumerate(tqdm(poses, desc="Rendering spatial sequence")):
        # Convert pose to tensor if needed
        if isinstance(pose, np.ndarray):
            pose = torch.FloatTensor(pose).to(device)
        
        # Render image
        results = render_image(model, pose, H, W, focal, time_step, device)
        
        # Convert to 8-bit RGB
        rgb = (results['rgb'] * 255).astype(np.uint8)
        
        # Normalize depth for visualization
        depth = results['depth']
        depth_viz = (depth - depth.min()) / (depth.max() - depth.min())
        depth_viz = (depth_viz * 255).astype(np.uint8)
        
        # Save images
        imageio.imwrite(os.path.join(output_dir, f'rgb_{i:03d}.png'), rgb)
        imageio.imwrite(os.path.join(output_dir, f'depth_{i:03d}.png'), depth_viz)
        
        # Append to frames for video
        rgb_frames.append(rgb)
        depth_frames.append(depth_viz)
    
    # Create videos
    imageio.mimwrite(os.path.join(output_dir, 'rgb_video.mp4'), rgb_frames, fps=10, quality=8)
    imageio.mimwrite(os.path.join(output_dir, 'depth_video.mp4'), depth_frames, fps=10, quality=8)


def compare_with_ground_truth(model, dataset, device, output_dir):
    """
    Compare rendered results with ground truth images.
    
    Args:
        model: NeRF or DynamicNeRF model.
        dataset: Dataset object.
        device (str): Device to render on.
        output_dir (str): Directory to save comparisons.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of examples to visualize
    num_examples = min(5, len(dataset))
    
    # Sample indices
    indices = np.linspace(0, len(dataset) - 1, num_examples, dtype=int)
    
    # Process each example
    for i, idx in enumerate(indices):
        # Get data from dataset
        rays_o, rays_d, time, gt_img = dataset[idx]
        
        # Convert to device
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        
        # Get scene dimensions
        H, W = rays_o.shape[:2]
        
        # For training dataset, rays are flattened, need to get original pose
        if len(rays_o.shape) == 2:
            # Use dataset pose and regenerate rays
            pose = dataset.poses[idx]
            H, W, focal = dataset.hwf
            rays_o, rays_d = get_rays(H, W, focal, pose)
        
        # Time step
        if time is not None:
            time_step = time.unsqueeze(0).to(device)
        else:
            time_step = None
        
        # Render image
        results = render_image(model, pose, H, W, focal, time_step, device)
        
        # Convert ground truth to numpy
        gt_img = gt_img.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot ground truth
        axes[0].imshow(gt_img)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Plot rendered image
        axes[1].imshow(results['rgb'])
        axes[1].set_title('Rendered')
        axes[1].axis('off')
        
        # Plot depth
        depth_viz = (results['depth'] - results['depth'].min()) / (results['depth'].max() - results['depth'].min())
        axes[2].imshow(depth_viz, cmap='viridis')
        axes[2].set_title('Depth')
        axes[2].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{i:03d}.png'))
        plt.close()


def create_plots(loss_dict, output_dir):
    """
    Create training plots.
    
    Args:
        loss_dict (dict): Dictionary with loss values.
        output_dir (str): Directory to save plots.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    for key, values in loss_dict.items():
        if 'loss' in key:
            plt.plot(values, label=key)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    # Plot metrics (PSNR, SSIM, etc.)
    plt.figure(figsize=(10, 5))
    for key, values in loss_dict.items():
        if 'psnr' in key or 'ssim' in key:
            plt.plot(values, label=key)
    
    plt.xlabel('Iteration')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    plt.close()


def interpolate_time_views(model, pose, H, W, focal, num_frames=60, device='cpu'):
    """
    Render a smooth interpolation through time at a fixed viewpoint.
    
    Args:
        model: DynamicNeRF model.
        pose: Camera pose matrix.
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length.
        num_frames (int): Number of frames to generate.
        device (str): Device to render on.
        
    Returns:
        list: List of rendered frames.
    """
    # Generate interpolated time steps
    time_steps = torch.linspace(0, 1, num_frames, device=device).unsqueeze(1)
    
    # Render at each time step
    frames = []
    
    for time_step in tqdm(time_steps, desc="Rendering time interpolation"):
        # Render image
        results = render_image(model, pose, H, W, focal, time_step, device)
        
        # Convert to 8-bit RGB
        rgb = (results['rgb'] * 255).astype(np.uint8)
        frames.append(rgb)
    
    return frames


def create_4d_visualization(model, dataset, device, output_path, num_views=30, num_time_steps=30):
    """
    Create a 4D visualization with interpolation in both space and time.
    
    Args:
        model: DynamicNeRF model.
        dataset: Dataset object.
        device: Device to render on.
        output_path (str): Path to save the resulting video.
        num_views (int): Number of views to interpolate.
        num_time_steps (int): Number of time steps to interpolate.
    """
    # Get camera parameters
    H, W, focal = dataset.hwf
    
    # Create spiral path for camera
    center_pose = dataset.poses[len(dataset.poses) // 2]
    radii = (0.5, 0.5, 0.5)  # Radii of spiral path
    focus_depth = 4.0  # Focus depth
    poses = create_spiral_poses(radii, focus_depth, n_poses=num_views)
    
    # Create time steps
    time_steps = torch.linspace(0, 1, num_time_steps, device=device).unsqueeze(1)
    
    # Render each view at each time step
    all_frames = []
    
    for i, pose in enumerate(tqdm(poses, desc="Rendering 4D visualization")):
        # Convert pose to tensor
        if isinstance(pose, np.ndarray):
            pose = torch.FloatTensor(pose).to(device)
        
        for j, time_step in enumerate(time_steps):
            # Render image
            results = render_image(model, pose, H, W, focal, time_step, device)
            
            # Convert to 8-bit RGB
            rgb = (results['rgb'] * 255).astype(np.uint8)
            all_frames.append(rgb)
    
    # Create video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(output_path, all_frames, fps=20, quality=8)


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities for Dynamic NeRF") 