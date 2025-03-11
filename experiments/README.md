# Experiments Directory

This directory is for Jupyter notebooks that demonstrate various aspects of the Dynamic NeRF implementation.

## Notebook Files

The following notebooks should be created in this directory:

1. `dynamic_nerf_exploration.ipynb` - Basic exploration of the Dynamic NeRF model
2. `dataset_visualization.ipynb` - Visualization of the dataset and preprocessing steps
3. `hyperparameter_tuning.ipynb` - Experiments with different hyperparameters
4. `results_analysis.ipynb` - Analysis of training results and model performance

## Creating Notebooks

To create a new notebook, you can use Jupyter:

```bash
jupyter notebook
```

Navigate to this directory and click "New" -> "Python 3" to create a new notebook.

## Using the Dynamic NeRF Package in Notebooks

To use the Dynamic NeRF package in your notebooks, add the following code at the beginning:

```python
import os
import sys

# Add the project root to the path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Now you can import from the package
from src.models.nerf import NeRF
from src.models.dynamic_nerf import DynamicNeRF
# ... other imports
```

## Example Notebook Structure

A typical notebook should include:

1. **Introduction** - Explain the purpose of the notebook
2. **Setup** - Import necessary modules and set up the environment
3. **Data Loading** - Load and preprocess data
4. **Model Initialization** - Initialize the Dynamic NeRF model
5. **Experiments** - Run experiments and visualize results
6. **Conclusion** - Summarize findings and next steps

## Tips for Effective Notebooks

- Use markdown cells to document your code and explain your reasoning
- Include visualizations to help understand the data and results
- Keep code cells focused on a single task
- Use section headers to organize your notebook
- Save intermediate results to avoid rerunning long computations
- Include parameter explorations to understand model behavior 