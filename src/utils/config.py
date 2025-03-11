#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration utilities for Dynamic NeRF.
"""

import os
import argparse
import yaml
import torch
from datetime import datetime


def parse_args_and_config():
    """
    Parse command line arguments and configuration files.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dynamic Neural Radiance Fields")
    
    # Basic parameters
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default='dynamic_nerf',
                      help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run on (cuda or cpu), default uses CUDA if available')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='synthetic',
                      help='Dataset type (synthetic or real)')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Data directory')
    parser.add_argument('--scene', type=str, default='lego',
                      help='Scene name')
    parser.add_argument('--img_size', type=int, nargs=2, default=None,
                      help='Image dimensions [height, width]')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='dynamic_nerf',
                      choices=['nerf', 'dynamic_nerf'],
                      help='Model type')
    parser.add_argument('--hidden_dims', type=int, default=256,
                      help='Dimensions of hidden layers')
    parser.add_argument('--pos_encoder_dims', type=int, default=60,
                      help='Dimensions of positional encoding')
    parser.add_argument('--dir_encoder_dims', type=int, default=24,
                      help='Dimensions of direction encoding')
    parser.add_argument('--time_encoder_dims', type=int, default=12,
                      help='Dimensions of time encoding')
    parser.add_argument('--use_attention', action='store_true',
                      help='Use spatio-temporal attention')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='Batch size (rays per batch)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                      help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                      help='Learning rate decay factor')
    parser.add_argument('--lr_decay_steps', type=int, default=50000,
                      help='Learning rate decay steps')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Rendering parameters
    parser.add_argument('--near', type=float, default=2.0,
                      help='Near plane distance')
    parser.add_argument('--far', type=float, default=6.0,
                      help='Far plane distance')
    parser.add_argument('--n_samples', type=int, default=64,
                      help='Number of coarse samples per ray')
    parser.add_argument('--n_importance', type=int, default=128,
                      help='Number of fine samples per ray')
    parser.add_argument('--use_hierarchical', action='store_true',
                      help='Use hierarchical sampling')
    parser.add_argument('--chunk_size', type=int, default=1024,
                      help='Chunk size for rendering')
    
    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Log directory')
    parser.add_argument('--save_freq', type=int, default=10,
                      help='Save frequency in epochs')
    parser.add_argument('--eval_freq', type=int, default=5,
                      help='Evaluation frequency in epochs')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize results during training')
    parser.add_argument('--num_viz_views', type=int, default=5,
                      help='Number of views to visualize')
    parser.add_argument('--num_viz_time_steps', type=int, default=10,
                      help='Number of time steps to visualize')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration from file
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update arguments with values from config file (if not explicitly set)
        for key, value in config_dict.items():
            if key in args.__dict__ and args.__dict__[key] is None:
                setattr(args, key, value)
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_name = f"{args.experiment_name}_{args.scene}_{timestamp}"
    
    # Set random seed
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    return args


def create_config_file(path='configs/default.yaml'):
    """
    Create a default configuration file.
    
    Args:
        path (str): Path to save the config file.
    """
    config = {
        # Dataset parameters
        'dataset': 'synthetic',
        'data_dir': 'data',
        'scene': 'lego',
        'img_size': [400, 400],
        
        # Model parameters
        'model': 'dynamic_nerf',
        'hidden_dims': 256,
        'pos_encoder_dims': 60,
        'dir_encoder_dims': 24,
        'time_encoder_dims': 12,
        'use_attention': True,
        
        # Training parameters
        'epochs': 100,
        'batch_size': 1024,
        'learning_rate': 5e-4,
        'lr_decay': 0.1,
        'lr_decay_steps': 50000,
        'num_workers': 4,
        
        # Rendering parameters
        'near': 2.0,
        'far': 6.0,
        'n_samples': 64,
        'n_importance': 128,
        'use_hierarchical': True,
        'chunk_size': 1024,
        
        # Logging parameters
        'log_dir': 'logs',
        'save_freq': 10,
        'eval_freq': 5,
        'visualize': True,
        'num_viz_views': 5,
        'num_viz_time_steps': 10,
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save configuration
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created default configuration file at {path}")


if __name__ == "__main__":
    # Create default configuration
    create_config_file() 