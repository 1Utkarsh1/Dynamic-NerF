#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Dynamic Neural Radiance Fields (Dynamic NeRF) model.
Extends the original NeRF to handle temporal variations in scenes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.nerf import NeRF, PositionalEncoder


class TemporalEncoder(PositionalEncoder):
    """
    Temporal encoding for time inputs.
    Extends the PositionalEncoder to specifically encode time dimensions.
    """
    def __init__(self, input_dims=1, num_freqs=6, include_input=True):
        """
        Initialize the temporal encoder.
        
        Args:
            input_dims (int): Dimensionality of time input (usually 1).
            num_freqs (int): Number of frequency bands to use.
            include_input (bool): Whether to include the original input in the encoding.
        """
        super().__init__(input_dims=input_dims, num_freqs=num_freqs, include_input=include_input)


class SpatioTemporalAttention(nn.Module):
    """
    Attention mechanism for focusing on dynamic regions of the scene.
    """
    def __init__(self, in_dims, hidden_dims=128, heads=4):
        """
        Initialize the spatio-temporal attention module.
        
        Args:
            in_dims (int): Input feature dimensionality.
            hidden_dims (int): Hidden layer dimensionality.
            heads (int): Number of attention heads.
        """
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.heads = heads
        self.head_dims = hidden_dims // heads
        assert hidden_dims % heads == 0, "Hidden dimensionality must be divisible by number of heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_dims, hidden_dims)
        self.k_proj = nn.Linear(in_dims, hidden_dims)
        self.v_proj = nn.Linear(in_dims, hidden_dims)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dims, in_dims)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(in_dims)
    
    def forward(self, x):
        """
        Forward pass through the attention module.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, seq_len, in_dims).
            
        Returns:
            torch.Tensor: Attention-weighted features of shape (batch_size, seq_len, in_dims).
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer normalization
        x_norm = self.norm(x)
        
        # Project to queries, keys, and values
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.heads, self.head_dims)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.heads, self.head_dims)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.heads, self.head_dims)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, heads, seq_len, head_dims)
        k = k.transpose(1, 2)  # (batch_size, heads, seq_len, head_dims)
        v = v.transpose(1, 2)  # (batch_size, heads, seq_len, head_dims)
        
        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dims)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, heads, seq_len, head_dims)
        
        # Transpose and reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dims)
        
        # Project to output space
        output = self.out_proj(attn_output)
        
        # Residual connection
        return x + output


class DynamicNeRF(nn.Module):
    """
    Dynamic Neural Radiance Field (Dynamic NeRF) model.
    
    Maps 6D coordinates (position + viewing direction + time) to RGB color and density.
    """
    def __init__(self, pos_encoder_dims=60, dir_encoder_dims=24, time_encoder_dims=12, hidden_dims=256, use_attention=True):
        """
        Initialize the Dynamic NeRF network.
        
        Args:
            pos_encoder_dims (int): Dimensionality of the position encoding.
            dir_encoder_dims (int): Dimensionality of the direction encoding.
            time_encoder_dims (int): Dimensionality of the time encoding.
            hidden_dims (int): Width of the hidden layers.
            use_attention (bool): Whether to use spatio-temporal attention.
        """
        super().__init__()
        self.pos_encoder_dims = pos_encoder_dims
        self.dir_encoder_dims = dir_encoder_dims
        self.time_encoder_dims = time_encoder_dims
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention
        
        # Position encoder (for xyz coordinates)
        self.pos_encoder = PositionalEncoder(input_dims=3, num_freqs=10)
        # Direction encoder (for viewing direction)
        self.dir_encoder = PositionalEncoder(input_dims=3, num_freqs=4)
        # Time encoder (for temporal coordinate)
        self.time_encoder = TemporalEncoder(input_dims=1, num_freqs=6)
        
        # Main MLP for density prediction
        # Takes position and time as input
        self.main_net = nn.Sequential(
            nn.Linear(pos_encoder_dims + time_encoder_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
        )
        
        # Optional attention mechanism
        if use_attention:
            self.attention = SpatioTemporalAttention(hidden_dims)
        
        # Additional layers after attention
        self.post_attention = nn.Sequential(
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
        # Takes features and encoded directions as input
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dims + dir_encoder_dims, hidden_dims // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dims // 2, 3),
            nn.Sigmoid(),  # RGB values in [0, 1]
        )
    
    def forward(self, rays_o, rays_d, time_indices):
        """
        Forward pass through the Dynamic NeRF model.
        
        Args:
            rays_o (torch.Tensor): Ray origins of shape (N, 3).
            rays_d (torch.Tensor): Ray directions of shape (N, 3).
            time_indices (torch.Tensor): Time indices of shape (N, 1), normalized between 0 and 1.
            
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
        
        # Encode positions, directions, and time
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)
        time_encoded = self.time_encoder(time_indices)
        
        # Concatenate position and time encodings
        pos_time_encoded = torch.cat([pos_encoded, time_encoded], dim=-1)
        
        # Get features from main network with time input
        features = self.main_net(pos_time_encoded)
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention (add sequence dimension)
            features = features.unsqueeze(1)  # (N, 1, hidden_dims)
            features = self.attention(features)
            features = features.squeeze(1)  # (N, hidden_dims)
        
        # Apply post-attention layers
        features = self.post_attention(features)
        
        # Predict density
        density = self.density_head(features)
        
        # Get intermediate features for color prediction
        color_features = self.feature_head(features)
        
        # Concatenate features with encoded directions for view-dependent color
        color_input = torch.cat([color_features, dir_encoded], dim=-1)
        
        # Predict RGB color
        rgb = self.color_net(color_input)
        
        return rgb, density
    
    def volume_rendering(self, rays_o, rays_d, time_indices, near=2.0, far=6.0, num_samples=64, rand=True):
        """
        Perform volume rendering for rays at a specific time step.
        
        Note: This is a simplified implementation for demonstration.
        
        Args:
            rays_o (torch.Tensor): Ray origins of shape (N, 3).
            rays_d (torch.Tensor): Ray directions of shape (N, 3).
            time_indices (torch.Tensor): Time indices of shape (N, 1).
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
        
        # Expand time indices for each sample
        time_indices_expand = time_indices.unsqueeze(1).expand(-1, num_samples, -1)  # (N, S, 1)
        
        # Calculate sample positions: o + td
        sample_points = rays_o_expand + rays_d_expand * t_vals.unsqueeze(-1)  # (N, S, 3)
        
        # Flatten for network evaluation
        flat_points = sample_points.reshape(-1, 3)  # (N*S, 3)
        flat_dirs = rays_d_expand.reshape(-1, 3)  # (N*S, 3)
        flat_times = time_indices_expand.reshape(-1, 1)  # (N*S, 1)
        
        # Forward pass through network
        rgb, density = self(flat_points, flat_dirs, flat_times)
        
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
    model = DynamicNeRF().to(device)
    
    # Create some test rays
    rays_o = torch.rand(4, 3, device=device)  # 4 rays, 3D origin
    rays_d = torch.rand(4, 3, device=device)  # 4 rays, 3D direction
    time_indices = torch.rand(4, 1, device=device)  # 4 time indices between 0 and 1
    
    # Normalize ray directions
    rays_d = F.normalize(rays_d, p=2, dim=-1)
    
    # Forward pass
    rgb, density = model(rays_o, rays_d, time_indices)
    
    print(f"RGB shape: {rgb.shape}")
    print(f"Density shape: {density.shape}")
    
    # Test volume rendering
    rgb, depth, weights = model.volume_rendering(rays_o, rays_d, time_indices)
    
    print(f"Rendered RGB shape: {rgb.shape}")
    print(f"Rendered depth shape: {depth.shape}")
    print(f"Weights shape: {weights.shape}") 