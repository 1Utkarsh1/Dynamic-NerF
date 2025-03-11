# Dynamic Neural Radiance Fields (Dynamic-NeRF)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()

A state-of-the-art implementation of Dynamic Neural Radiance Fields for high-quality novel view synthesis from multi-view video or image sequences.

## 🔄 Overview

Dynamic-NeRF extends traditional Neural Radiance Fields (NeRF) to capture and render dynamic 3D scenes with temporal variations. While standard NeRF excels at reconstructing static scenes, Dynamic-NeRF adds the ability to model moving objects, changing lighting conditions, and other time-dependent phenomena.


## 🌟 Key Features

- **Temporal Encoding**: Models time as an additional dimension for capturing scene dynamics
- **Spatio-Temporal Attention**: Advanced attention mechanisms to focus on moving regions
- **Static-Dynamic Decomposition**: Separate handling of static backgrounds and dynamic foregrounds
- **Novel View Synthesis**: Generate high-quality novel viewpoints at arbitrary time steps
- **Temporal Consistency**: Smooth transitions between time steps for realistic rendering

## 🧩 Model Architecture

Our Dynamic-NeRF implementation builds upon the original NeRF with several key innovations:

- **Time-Conditioned MLP**: Neural network conditioned on spatial position, viewing direction, and time
- **Temporal Embedding**: Specialized embedding functions to encode temporal information
- **Attention Mechanisms**: Spatio-temporal attention for focusing on dynamic regions
- **Optional Flow Integration**: Incorporation of optical flow for improved temporal coherence

```
[Architecture Diagram Coming Soon]
```

## 📊 Datasets

The project supports several datasets for training and evaluation:

- **D-NeRF Dataset**: Synthetic sequences with controlled object motion
- **Custom Blender Sequences**: Rendered scenes with ground truth camera parameters
- **Real-World Multi-View Video**: Captured sequences of dynamic scenes

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dynamic-nerf.git
cd dynamic-nerf

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (will be added later)
pip install -r requirements.txt
```

## 📈 Usage

### Quick Start

```bash
# Example command for training (to be implemented)
python src/train.py --config configs/d_nerf.yaml

# Example command for rendering novel views (to be implemented)
python src/render.py --model_path checkpoints/model.pt --time_steps 10
```

## 📝 Project Structure

```
dynamic-nerf/
├── src/                    # Source code
│   ├── models/             # Neural network architectures
│   ├── data/               # Data loading and preprocessing
│   ├── utils/              # Utility functions
│   ├── train.py            # Training script
│   └── render.py           # Rendering script
├── experiments/            # Jupyter notebooks for experiments
├── configs/                # Configuration files
├── data/                   # Dataset storage
│   └── README.md           # Dataset instructions
├── docs/                   # Documentation
└── results/                # Saved results and visualizations
```

## 📊 Results



## 🛠️ Roadmap

- [x] Project setup and repository structure
- [x] Implementation of baseline static NeRF
- [x] Extension to dynamic scenes with temporal encoding
- [x] Integration of spatio-temporal attention mechanisms
- [ ] Experimentation with various temporal embeddings
- [ ] Quantitative and qualitative evaluation
- [ ] Documentation and result visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

1. Mildenhall, B. et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" - *ECCV* (2020)
2. Pumarola, A. et al. "D-NeRF: Neural Radiance Fields for Dynamic Scenes" - *CVPR* (2021)
3. Li, Z. et al. "Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes" - *CVPR* (2021)
4. Park, K. et al. "Nerfies: Deformable Neural Radiance Fields" - *ICCV* (2021)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
    <b>Made with ❤️ by Utkarsh Rajput</b>
</div> 
