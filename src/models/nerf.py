#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of the original Neural Radiance Fields (NeRF) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoder(nn.Module):
    """
    Positional encoding for input coordinates as described in the NeRF paper.
    
    Encodes each coordinate with sin and cos functions at different frequencies.
    """
    def __init__(self, input_dims=3, num_freqs=10, include_input=True):
        """
        Initialize the positional encoder.
        
        Args:
            input_dims (int): Dimensionality of input coordinates (default 3 for xyz).
            num_freqs (int): Number of frequency bands to use.
            include_input (bool): Whether to include the original input in the encoding.
        """
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.max_freq = num_freqs - 1
        self.num_output_dims = input_dims * (1 + 2 * num_freqs) if include_input else input_dims * 2 * num_freqs
        
        # Frequency bands
        self.freq_bands = 2.0 ** torch.linspace(0, self.max_freq, num_freqs)
    
    def forward(self, x):
        """
        Apply positional encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dims).
            
        Returns:
            torch.Tensor: Encoded tensor of shape (..., num_output_dims).
        """
        # Ensure x has at least 2 dimensions
        orig_shape = x.shape
        x = x.view(-1, self.input_dims)
        
        # Create output tensor
        out = []
        
        # Include original input if specified
        if self.include_input:
            out.append(x)
        
        # Apply encoding
        for freq in self.freq_bands:
            # sin(2^i * π * x)
            out.append(torch.sin(x * freq * np.pi))
            # cos(2^i * π * x)
            out.append(torch.cos(x * freq * np.pi))
        
        # Concatenate all terms
        out = torch.cat(out, dim=-1)
        
        # Reshape to original batch shape
        out = out.view(*orig_shape[:-1], self.num_output_dims)
        
        return out


class NeRF(nn.Module):
    """
    Neural Radiance Field (NeRF) model.
    
    Maps 5D coordinates (position + viewing direction) to RGB color and density.
    """
    def __init__(self, pos_encoder_dims=60, dir_encoder_dims=24, hidden_dims=256):
        """
        Initialize the NeRF network.
        
        Args:
            pos_encoder_dims (int): Dimensionality of the position encoding.
            dir_encoder_dims (int): Dimensionality of the direction encoding.
            hidden_dims (int): Width of the hidden layers.
        """
        super().__init__()
        self.pos_encoder_dims = pos_encoder_dims
        self.dir_encoder_dims = dir_encoder_dims
        self.hidden_dims = hidden_dims
        
        # Position encoder (for xyz coordinates)
        self.pos_encoder = PositionalEncoder(input_dims=3, num_freqs=10)
        # Direction encoder (for viewing direction)
        self.dir_encoder = PositionalEncoder(input_dims=3, num_freqs=4)
        
        # Main MLP for density prediction
        self.main_net = nn.Sequential(
            nn.Linear(pos_encoder_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
        )
        
        # Density head
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dims, 1),
            nn.ReLU(True),  # Density should be non-negative
        )
        
        # Feature head for the following MLP
        self.feature_head = nn.Linear(hidden_dims, hidden_dims)
        
        # Direction-dependent MLP for color prediction
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dims + dir_encoder_dims, hidden_dims // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dims // 2, 3),
            nn.Sigmoid(),  # RGB values in [0, 1]
        )
    
    def forward(self, rays_o, rays_d):
        """
        Forward pass through the NeRF model.
        
        Args:
            rays_o (torch.Tensor): Ray origins of shape (N, 3).
            rays_d (torch.Tensor): Ray directions of shape (N, 3).
            
        Returns:
            tuple: (rgb, density) where rgb is a tensor of shape (N, 3) and
                  density is a tensor of shape (N, 1).
        """
        # For simplicity, assuming a single point per ray (not using the full pipeline)
        # In a real implementation, we would sample points along rays and do volume rendering
        
        # Get positions along rays (simplified, just using origins for now)
        positions = rays_o  # In actual impl, this would be origins + t * directions
        
        # Normalize directions for the encoding
        directions = F.normalize(rays_d, p=2, dim=-1)
        
        # Encode positions and directions
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)
        
        # Get features from main network
        features = self.main_net(pos_encoded)
        
        # Predict density
        density = self.density_head(features)
        
        # Get intermediate features for color prediction
        color_features = self.feature_head(features)
        
        # Concatenate features with encoded directions for view-dependent color
        color_input = torch.cat([color_features, dir_encoded], dim=-1)
        
        # Predict RGB color
        rgb = self.color_net(color_input)
        
        return rgb, density
    
    def volume_rendering(self, rays_o, rays_d, near=2.0, far=6.0, num_samples=64, rand=True):
        """
        Perform volume rendering for rays.
        
        Note: This is a simplified implementation for demonstration.
        
        Args:
            rays_o (torch.Tensor): Ray origins of shape (N, 3).
            rays_d (torch.Tensor): Ray directions of shape (N, 3).
            near (float): Near plane distance.
            far (float): Far plane distance.
            num_samples (int): Number of samples per ray.
            rand (bool): Whether to use randomized sampling.
            
        Returns:
            tuple: (rgb, depth, weights) for rendered color, depth, and weights.
        """
        device = rays_o.device
        batch_size = rays_o.shape[0]
        
        # Generate sample points along each ray
        t_vals = torch.linspace(near, far, num_samples, device=device)
        
        # Add randomization to sampling if specified
        if rand:
            # Add noise to sample points (stratified sampling)
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(batch_size, num_samples, device=device)
            t_vals = lower + (upper - lower) * t_rand
        
        # Expand rays for each sample
        rays_o_expand = rays_o.unsqueeze(1).expand(-1, num_samples, -1)  # (N, S, 3)
        rays_d_expand = rays_d.unsqueeze(1).expand(-1, num_samples, -1)  # (N, S, 3)
        
        # Calculate sample positions: o + td
        sample_points = rays_o_expand + rays_d_expand * t_vals.unsqueeze(-1)  # (N, S, 3)
        
        # Flatten for network evaluation
        flat_points = sample_points.reshape(-1, 3)  # (N*S, 3)
        flat_dirs = rays_d_expand.reshape(-1, 3)  # (N*S, 3)
        
        # Forward pass through network
        rgb, density = self(flat_points, flat_dirs)
        
        # Reshape outputs
        rgb = rgb.reshape(batch_size, num_samples, 3)  # (N, S, 3)
        density = density.reshape(batch_size, num_samples, 1)  # (N, S, 1)
        
        # Calculate distances between samples
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-10], dim=-1)  # (N, S)
        dists = dists.unsqueeze(-1)  # (N, S, 1)
        
        # Volume rendering computations
        alpha = 1.0 - torch.exp(-density * dists)  # (N, S, 1)
        
        # Calculate transmittance (accumulated transparency)
        trans = torch.cat([
            torch.ones_like(alpha[:, :1, :]),  # For the first sample
            torch.cumprod(1.0 - alpha + 1e-10, dim=1)[:, :-1, :]  # For remaining samples
        ], dim=1)  # (N, S, 1)
        
        # Calculate weights for each sample
        weights = alpha * trans  # (N, S, 1)
        
        # Calculate RGB, depth, and accumulated weights
        rgb_final = torch.sum(weights * rgb, dim=1)  # (N, 3)
        depth = torch.sum(weights * t_vals.unsqueeze(-1), dim=1)  # (N, 1)
        acc_weights = torch.sum(weights, dim=1)  # (N, 1)
        
        return rgb_final, depth, acc_weights


if __name__ == "__main__":
    # Simple test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeRF().to(device)
    
    # Create some test rays
    rays_o = torch.rand(4, 3, device=device)  # 4 rays, 3D origin
    rays_d = torch.rand(4, 3, device=device)  # 4 rays, 3D direction
    
    # Normalize ray directions
    rays_d = F.normalize(rays_d, p=2, dim=-1)
    
    # Forward pass
    rgb, density = model(rays_o, rays_d)
    
    print(f"RGB shape: {rgb.shape}")
    print(f"Density shape: {density.shape}")
    
    # Test volume rendering
    rgb, depth, weights = model.volume_rendering(rays_o, rays_d)
    
    print(f"Rendered RGB shape: {rgb.shape}")
    print(f"Rendered depth shape: {depth.shape}")
    print(f"Weights shape: {weights.shape}") 