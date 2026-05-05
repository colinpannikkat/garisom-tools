"""
# Model Interface and Sperry Interface Implementation

This module provides abstract base classes and concrete implementations for running
stomatal-optimization models.

## Classes

- `Model`: Abstract base class defining the interface for all models
- `SperryModel`: Concrete interface implementation for the Sperry model
- `PotkayModel`: Concrete interface implementation for the Potkay model

## Key Features

- **Parallel Execution**: Support for running multiple model instances concurrently
- **Flexible Configuration**: Customizable run and evaluation parameters
- **Model Evaluation**: Built-in metrics calculation and comparison with ground truth data
- **Ray Tune Integration**: Seamless integration with hyperparameter optimization
- **SALib Integration**: Easy sensitivity analysis and custom configs.
- **Monte Carlo Simulations**: Integrated Monte Carlo simulation for generating prediction
    uncertainty intervals.

## Example Usage

```python
from garisom_tools import SperryModel
from garisom_tools.config import MetricConfig
import pandas as pd
from datetime import datetime

# Load parameters and configuration
params = pd.read_csv("parameters.csv")
config_file = "model_config.csv"

# Create model instance
model = SperryModel(
    run_kwargs={
        'params': params,
        'config_file': config_file,
        'population': 1,
        'model_dir': '/path/to/model'
    },
    eval_kwargs={
        'ground': ground_truth_data,
        'start_date': datetime(2023, 7, 20),
        'end_date': datetime(2023, 8, 24)
    }
)

# Run single simulation
result = model.run(X={'param1': 0.5, 'param2': 1.2})

# Run parallel simulations
param_sets = [{'param1': 0.5, 'param2': 1.2}, {'param1': 0.7, 'param2': 1.0}]
results = model.run_parallel(param_sets, workers=4)
```
"""

from .base import Model
from .sperry import SperryModel
from .potkay import PotkayModel
