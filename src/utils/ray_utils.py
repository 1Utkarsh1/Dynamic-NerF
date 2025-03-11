#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for ray generation and manipulation.
"""

import torch
import numpy as np


def get_rays(H, W, focal, c2w):
    """
    Generate rays for a camera with given parameters.
    
    Args:
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length.
        c2w (torch.Tensor or np.ndarray): Camera-to-world transformation matrix.
        
    Returns:
        tuple: (ray_origins, ray_directions)
    """
    # Convert numpy array to torch tensor if needed
    if isinstance(c2w, np.ndarray):
        c2w = torch.FloatTensor(c2w)
    
    # Create meshgrid for pixel coordinates
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    i = i.float()
    j = j.float()
    
    # Convert pixel coordinates to normalized device coordinates
    x = (i - W/2) / focal
    y = -(j - H/2) / focal  # Negative because y is down in image
    z = -torch.ones_like(x)  # Camera looks along negative z-axis
    
    # Stack to get directions in camera frame
    dirs = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    
    # Transform directions to world space
    rays_d = dirs.unsqueeze(-2) @ c2w[:3, :3].T  # (H, W, 1, 3) @ (3, 3) -> (H, W, 1, 3)
    rays_d = rays_d.squeeze(-2)  # (H, W, 3)
    
    # Get ray origin (camera position in world space)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
    
    return rays_o, rays_d


def get_ray_batch(rays_o, rays_d, time_indices=None, batch_size=1024, shuffle=True):
    """
    Create batches of rays for training.
    
    Args:
        rays_o (torch.Tensor): Ray origins of shape (N, 3).
        rays_d (torch.Tensor): Ray directions of shape (N, 3).
        time_indices (torch.Tensor, optional): Time indices of shape (N, 1).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle rays.
        
    Returns:
        generator: Yields batches of (ray_origins, ray_directions, [time_indices]).
    """
    N = rays_o.shape[0]
    indices = torch.arange(N)
    
    # Shuffle indices if specified
    if shuffle:
        indices = indices[torch.randperm(N)]
    
    # Generate batches
    for i in range(0, N, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_rays_o = rays_o[batch_indices]
        batch_rays_d = rays_d[batch_indices]
        
        if time_indices is not None:
            batch_time = time_indices[batch_indices]
            yield batch_rays_o, batch_rays_d, batch_time
        else:
            yield batch_rays_o, batch_rays_d


def sample_pdf(bins, weights, N_samples, det=False):
    """
    Sample from a probability density function (PDF) defined by weights.
    Used for hierarchical sampling in NeRF.
    
    Args:
        bins (torch.Tensor): Bin edges of shape (batch_size, N_bins + 1).
        weights (torch.Tensor): Weights of shape (batch_size, N_bins).
        N_samples (int): Number of samples to generate.
        det (bool): Whether to use deterministic sampling.
        
    Returns:
        torch.Tensor: Sampled points of shape (batch_size, N_samples).
    """
    device = weights.device
    
    # Get PDF and CDF
    weights = weights + 1e-5  # Prevent division by zero
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # (batch_size, N_bins)
    cdf = torch.cumsum(pdf, dim=-1)  # (batch_size, N_bins)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (batch_size, N_bins + 1)
    
    # Take samples
    if det:
        # Uniform sampling
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # Random sampling
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)
    
    # Invert CDF to find sample locations
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 2)
    above = torch.clamp(inds, 0, cdf.shape[-1] - 1)
    
    inds_g = torch.stack([below, above], dim=-1)  # (batch_size, N_samples, 2)
    
    # Get the upper and lower CDF values and bin edges
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
    # Linear interpolation
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples


def render_rays(model, rays_o, rays_d, time_indices=None, near=2.0, far=6.0, 
                N_samples=64, N_importance=0, chunk_size=1024, use_hierarchical=False):
    """
    Render rays using a NeRF model.
    
    Args:
        model: NeRF or DynamicNeRF model.
        rays_o (torch.Tensor): Ray origins of shape (N, 3).
        rays_d (torch.Tensor): Ray directions of shape (N, 3).
        time_indices (torch.Tensor, optional): Time indices of shape (N, 1).
        near (float): Near plane distance.
        far (float): Far plane distance.
        N_samples (int): Number of coarse samples per ray.
        N_importance (int): Number of fine samples per ray.
        chunk_size (int): Maximum number of rays to process at once.
        use_hierarchical (bool): Whether to use hierarchical sampling.
        
    Returns:
        dict: Dictionary with rendered outputs.
    """
    device = rays_o.device
    N_rays = rays_o.shape[0]
    
    # Sample points along rays
    t_vals = torch.linspace(near, far, N_samples, device=device)
    z_vals = t_vals.expand(N_rays, N_samples)
    
    # Perturb sampling points
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., :1], mids], dim=-1)
    t_rand = torch.rand(z_vals.shape, device=device)
    z_vals = lower + (upper - lower) * t_rand
    
    # Points in space to evaluate model at
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # (N_rays, N_samples, 3)
    
    # Evaluate model in chunks
    rgb_chunks = []
    sigma_chunks = []
    
    for i in range(0, N_rays, chunk_size):
        chunk_pts = pts[i:i+chunk_size].reshape(-1, 3)  # (chunk_size * N_samples, 3)
        chunk_dirs = rays_d[i:i+chunk_size].unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)
        
        if time_indices is not None:
            chunk_time = time_indices[i:i+chunk_size].unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 1)
            with torch.no_grad():
                chunk_rgb, chunk_sigma = model(chunk_pts, chunk_dirs, chunk_time)
        else:
            with torch.no_grad():
                chunk_rgb, chunk_sigma = model(chunk_pts, chunk_dirs)
        
        rgb_chunks.append(chunk_rgb)
        sigma_chunks.append(chunk_sigma)
    
    # Combine chunks
    rgb = torch.cat(rgb_chunks, dim=0).reshape(N_rays, N_samples, 3)
    sigma = torch.cat(sigma_chunks, dim=0).reshape(N_rays, N_samples, 1)
    
    # Volume rendering
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], dim=-1)  # (N_rays, N_samples)
    dists = dists * torch.norm(rays_d.unsqueeze(-2), dim=-1)
    
    # Compute alpha
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # (N_rays, N_samples)
    
    # Compute weights
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)  # (N_rays, N_samples)
    weights = alpha * T  # (N_rays, N_samples)
    
    # Hierarchical sampling (optional)
    if use_hierarchical and N_importance > 0:
        # Sample additional points
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=False)
        z_samples = z_samples.detach()
        
        # Combine with existing samples and sort
        z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        
        # Evaluate model at new points
        pts_combined = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals_combined.unsqueeze(-1)
        
        # Evaluate model in chunks
        rgb_chunks = []
        sigma_chunks = []
        
        for i in range(0, N_rays, chunk_size):
            chunk_pts = pts_combined[i:i+chunk_size].reshape(-1, 3)
            chunk_dirs = rays_d[i:i+chunk_size].unsqueeze(1).expand(-1, N_samples + N_importance, -1).reshape(-1, 3)
            
            if time_indices is not None:
                chunk_time = time_indices[i:i+chunk_size].unsqueeze(1).expand(-1, N_samples + N_importance, -1).reshape(-1, 1)
                with torch.no_grad():
                    chunk_rgb, chunk_sigma = model(chunk_pts, chunk_dirs, chunk_time)
            else:
                with torch.no_grad():
                    chunk_rgb, chunk_sigma = model(chunk_pts, chunk_dirs)
            
            rgb_chunks.append(chunk_rgb)
            sigma_chunks.append(chunk_sigma)
        
        # Combine chunks
        rgb = torch.cat(rgb_chunks, dim=0).reshape(N_rays, N_samples + N_importance, 3)
        sigma = torch.cat(sigma_chunks, dim=0).reshape(N_rays, N_samples + N_importance, 1)
        
        # Update z_vals and recompute weights
        z_vals = z_vals_combined
        
        # Volume rendering for combined samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], dim=-1)
        dists = dists * torch.norm(rays_d.unsqueeze(-2), dim=-1)
        
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
        weights = alpha * T
    
    # Composite RGB and depth
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)  # (N_rays, 3)
    depth = torch.sum(weights * z_vals, dim=-1)  # (N_rays)
    
    # Return results
    return {
        'rgb': rgb_final,
        'depth': depth,
        'weights': weights,
        'z_vals': z_vals,
    }


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Generate a spiral path of camera poses for rendering.
    
    Args:
        radii (tuple): Radii of spiral path in (x, y, z).
        focus_depth (float): Depth that the spiral path focuses on.
        n_poses (int): Number of poses to generate.
        
    Returns:
        list: List of camera pose matrices.
    """
    poses = []
    for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        # Spiral path
        center = np.array([
            radii[0] * np.cos(theta),
            radii[1] * np.sin(theta),
            radii[2] * np.sin(theta * 0.5)
        ])
        
        # Look-at point
        forward_vector = np.array([0, 0, focus_depth]) - center
        forward_vector = forward_vector / np.linalg.norm(forward_vector)
        
        # Up vector (approximately [0, 1, 0])
        up_vector = np.array([0, 1, 0])
        
        # Compute camera axes
        right_vector = np.cross(forward_vector, up_vector)
        right_vector = right_vector / np.linalg.norm(right_vector)
        
        # Recompute up vector to ensure orthogonality
        up_vector = np.cross(right_vector, forward_vector)
        
        # Camera-to-world matrix
        pose = np.eye(4)
        pose[:3, 0] = right_vector
        pose[:3, 1] = -up_vector  # Negative because y is down in image
        pose[:3, 2] = -forward_vector  # Negative because camera looks along negative z
        pose[:3, 3] = center
        
        poses.append(pose)
    
    return poses 