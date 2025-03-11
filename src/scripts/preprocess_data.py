#!/usr/bin/env python
"""
Preprocess raw data for Dynamic NeRF training.

This script converts various dataset formats into a standardized format
suitable for Dynamic NeRF training. It supports Blender synthetic datasets
and custom video sequences.

Usage:
    python preprocess_data.py --input_dir /path/to/raw/data --output_dir /path/to/processed/data --dataset_type [blender|custom_video]
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import cv2
import shutil
from tqdm import tqdm

def preprocess_blender_data(input_dir, output_dir):
    """
    Preprocess Blender synthetic dataset.
    
    Args:
        input_dir: Path to raw blender data
        output_dir: Path to save processed data
    """
    print(f"Preprocessing Blender data from {input_dir} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split (train/val/test)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(input_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping.")
            continue
            
        # Create output directory for this split
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        
        # Load transform.json which contains camera parameters
        with open(os.path.join(split_dir, 'transforms.json'), 'r') as f:
            meta = json.load(f)
        
        frames = meta['frames']
        print(f"Processing {len(frames)} frames for {split} split")
        
        # Process each frame
        for i, frame in enumerate(tqdm(frames)):
            # Get file path for rgb image
            file_path = os.path.join(input_dir, frame['file_path'] + '.png')
            if not os.path.exists(file_path):
                file_path = os.path.join(input_dir, frame['file_path'][1:] + '.png')  # Handle paths with leading dot
            
            # Copy image to output directory
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save rgb image
            output_path = os.path.join(output_dir, split, f"{i:06d}.png")
            cv2.imwrite(output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Save camera parameters
            camera_params = {
                'transform_matrix': frame['transform_matrix'],
                'time': i / len(frames),  # Normalize time to [0, 1]
            }
            
            with open(os.path.join(output_dir, split, f"{i:06d}.json"), 'w') as f:
                json.dump(camera_params, f, indent=2)
    
    # Copy dataset metadata
    if os.path.exists(os.path.join(input_dir, 'transforms.json')):
        with open(os.path.join(input_dir, 'transforms.json'), 'r') as f:
            meta = json.load(f)
        
        # Save global dataset parameters
        dataset_params = {
            'camera_angle_x': meta.get('camera_angle_x', 0.6911112070083618),
            'frames': len(meta['frames']),
            'w': meta.get('w', 800),
            'h': meta.get('h', 800),
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(dataset_params, f, indent=2)
    
    print(f"Preprocessing complete. Data saved to {output_dir}")

def preprocess_video(input_dir, output_dir, fps=None):
    """
    Preprocess custom video data.
    
    Args:
        input_dir: Path to video file or directory of frames
        output_dir: Path to save processed data
        fps: Frames per second to extract (if input is video)
    """
    print(f"Preprocessing video data from {input_dir} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    
    # Check if input is a video file
    if os.path.isfile(input_dir) and input_dir.endswith(('.mp4', '.avi', '.mov')):
        # Extract frames from video
        cap = cv2.VideoCapture(input_dir)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_dir}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame sampling rate
        sample_rate = 1
        if fps is not None and fps < video_fps:
            sample_rate = int(video_fps / fps)
        
        print(f"Video properties: {width}x{height}, {video_fps} FPS, {total_frames} frames")
        print(f"Extracting frames with sample rate: {sample_rate}")
        
        # Extract frames
        frame_count = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Save the frame
                output_path = os.path.join(output_dir, 'train', f"{frame_idx:06d}.png")
                cv2.imwrite(output_path, frame)
                
                # Create basic camera parameters (assumed fixed camera)
                camera_params = {
                    'transform_matrix': [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ],
                    'time': frame_idx / (total_frames // sample_rate)  # Normalize time to [0, 1]
                }
                
                with open(os.path.join(output_dir, 'train', f"{frame_idx:06d}.json"), 'w') as f:
                    json.dump(camera_params, f, indent=2)
                
                frame_idx += 1
            
            frame_count += 1
        
        cap.release()
        
        # Save global dataset parameters
        dataset_params = {
            'camera_angle_x': 0.6911112070083618,  # Default, approximately 40 degrees FOV
            'frames': frame_idx,
            'w': width,
            'h': height,
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(dataset_params, f, indent=2)
        
    else:
        # Assume the input is a directory of frames
        image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"Error: No image files found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} image files")
        
        # Process each image
        for i, img_file in enumerate(tqdm(image_files)):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue
            
            # Save the frame
            output_path = os.path.join(output_dir, 'train', f"{i:06d}.png")
            cv2.imwrite(output_path, img)
            
            # Create basic camera parameters (assumed fixed camera)
            camera_params = {
                'transform_matrix': [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ],
                'time': i / len(image_files)  # Normalize time to [0, 1]
            }
            
            with open(os.path.join(output_dir, 'train', f"{i:06d}.json"), 'w') as f:
                json.dump(camera_params, f, indent=2)
        
        # Get image dimensions from the first image
        first_img = cv2.imread(os.path.join(input_dir, image_files[0]))
        height, width = first_img.shape[:2]
        
        # Save global dataset parameters
        dataset_params = {
            'camera_angle_x': 0.6911112070083618,  # Default, approximately 40 degrees FOV
            'frames': len(image_files),
            'w': width,
            'h': height,
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(dataset_params, f, indent=2)
    
    print(f"Preprocessing complete. Data saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for Dynamic NeRF")
    parser.add_argument("--input_dir", required=True, help="Path to input data directory")
    parser.add_argument("--output_dir", required=True, help="Path to output processed data directory")
    parser.add_argument("--dataset_type", required=True, choices=['blender', 'custom_video'], 
                        help="Type of dataset to preprocess")
    parser.add_argument("--fps", type=int, default=None, 
                        help="Frames per second to extract (for video input)")
    
    args = parser.parse_args()
    
    if args.dataset_type == 'blender':
        preprocess_blender_data(args.input_dir, args.output_dir)
    elif args.dataset_type == 'custom_video':
        preprocess_video(args.input_dir, args.output_dir, args.fps)
    else:
        print(f"Unsupported dataset type: {args.dataset_type}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 