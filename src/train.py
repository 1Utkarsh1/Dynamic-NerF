#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training script for Dynamic NeRF.
"""

import os
import time
import argparse
import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.nerf import NeRF
from models.dynamic_nerf import DynamicNeRF
from data.dataset import DynamicNeRFDataset
from utils.config import parse_args_and_config
from utils.visualization import visualize_results


def train(config):
    """
    Main training function.
    
    Args:
        config: Configuration object with hyperparameters.
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment directory
    experiment_dir = os.path.join(config.log_dir, f"{config.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(config), f)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=experiment_dir)
    
    # Create dataset
    print(f"Loading dataset {config.dataset}...")
    dataset = DynamicNeRFDataset(
        root_dir=config.data_dir,
        scene_name=config.scene,
        split='train',
        img_size=config.img_size,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    
    # Create validation dataset
    val_dataset = DynamicNeRFDataset(
        root_dir=config.data_dir,
        scene_name=config.scene,
        split='val',
        img_size=config.img_size,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    # Initialize model
    print("Initializing model...")
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
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.lr_decay
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}", leave=False) as pbar:
            for batch in pbar:
                # Move data to device
                rays_o, rays_d, time_indices, target_rgb = [
                    b.to(device) for b in batch
                ]
                
                # Forward pass
                optimizer.zero_grad()
                
                if config.model == 'nerf':
                    rgb, depth = model(rays_o, rays_d)
                else:  # dynamic_nerf
                    rgb, depth = model(rays_o, rays_d, time_indices)
                
                # Compute loss
                loss = torch.mean((rgb - target_rgb) ** 2)  # MSE loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                
                # Log to tensorboard
                writer.add_scalar('train/loss', loss.item(), global_step)
                global_step += 1
        
        # Update learning rate
        scheduler.step()
        
        # Compute average epoch loss
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch}/{config.epochs}, Loss: {epoch_loss:.6f}")
        
        # Validation
        if epoch % config.eval_freq == 0:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    # Move data to device
                    rays_o, rays_d, time_indices, target_rgb = [
                        b.to(device) for b in batch
                    ]
                    
                    # Forward pass
                    if config.model == 'nerf':
                        rgb, depth = model(rays_o, rays_d)
                    else:  # dynamic_nerf
                        rgb, depth = model(rays_o, rays_d, time_indices)
                    
                    # Compute loss
                    loss = torch.mean((rgb - target_rgb) ** 2)  # MSE loss
                    val_loss += loss.item()
            
            # Compute average validation loss
            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Log to tensorboard
            writer.add_scalar('val/loss', val_loss, global_step)
            
            # Save model if it has the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(experiment_dir, 'best_model.pt'))
                print(f"Saved new best model at epoch {epoch}")
            
            # Visualize results
            if config.visualize:
                visualize_results(
                    model, 
                    val_dataset, 
                    device,
                    output_dir=os.path.join(experiment_dir, f'viz_epoch_{epoch}'),
                    num_views=config.num_viz_views,
                    num_time_steps=config.num_viz_time_steps,
                    dynamic=config.model == 'dynamic_nerf',
                )
        
        # Save checkpoint
        if epoch % config.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, os.path.join(experiment_dir, 'final_model.pt'))
    
    print(f"Training completed! Models saved to {experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    # Parse arguments
    config = parse_args_and_config()
    
    # Start training
    train(config) 