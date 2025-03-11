# Dynamic NeRF

Neural Radiance Fields for Dynamic Scenes

![Dynamic NeRF Example](docs/images/dynamic_nerf_example.png)

## Overview

Dynamic NeRF extends Neural Radiance Fields (NeRF) to handle dynamic scenes. While traditional NeRF excels at recreating static 3D scenes from multiple viewpoints, Dynamic NeRF incorporates temporal information to model scenes with moving objects or changing environmental conditions.

This implementation includes:
- Temporal encoding for handling time-varying scenes
- Spatio-temporal attention mechanism to focus on dynamic elements
- Efficient ray sampling and volume rendering
- Comprehensive training and evaluation pipeline

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA (for GPU acceleration, recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dynamic-nerf.git
cd dynamic-nerf
```

2. Install the package:
```bash
pip install -e .
```

This will install all required dependencies and make the package available for import.

## Dataset Preparation

Dynamic NeRF supports several dataset formats:

### Blender Synthetic Dataset

For Blender-rendered synthetic scenes:

```bash
python -m src.scripts.preprocess_data --input_dir /path/to/blender/data --output_dir /path/to/output --dataset_type blender
```

### Custom Video Data

For processing video files or image sequences:

```bash
python -m src.scripts.preprocess_data --input_dir /path/to/video.mp4 --output_dir /path/to/output --dataset_type custom_video --fps 24
```

## Training

To train a Dynamic NeRF model:

```bash
python -m src.train \
    --config configs/default.yaml \
    --data_path /path/to/processed/data \
    --output_dir /path/to/save/model \
    --experiment_name my_dynamic_scene
```

Key training parameters can be modified in the config file or via command-line arguments:
- Learning rate and scheduler
- Batch size and number of iterations
- Network architecture (encoding dimensions, network depth)
- Sampling strategy (number of coarse/fine samples)

## Rendering

To render novel views from a trained model:

```bash
python -m src.render \
    --config configs/default.yaml \
    --checkpoint /path/to/model/checkpoint.pt \
    --output_dir /path/to/renderings \
    --render_video \
    --time_range 0 1 60  # Start time, end time, number of frames
```

## Model Architecture

The Dynamic NeRF model consists of several key components:

1. **Spatial Encoding**: Position encoding for 3D coordinates.
2. **Temporal Encoding**: Encodes the time dimension to handle dynamics.
3. **Spatio-Temporal Attention**: Allows the model to focus on relevant parts of the scene at different times.
4. **Hierarchical Sampling**: Coarse-to-fine sampling strategy for efficient rendering.

## Results

Example renderings:

![Dynamic NeRF Results](docs/images/results_comparison.png)

## Citation

If you find this implementation useful, please consider citing:

```
@misc{dynamicnerf2023,
  author = {Your Name},
  title = {Dynamic NeRF: Neural Radiance Fields for Dynamic Scenes},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/dynamic-nerf}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The original NeRF paper: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [Dynamic NeRF concepts](https://arxiv.org/abs/2011.13961) for handling time-varying scenes 