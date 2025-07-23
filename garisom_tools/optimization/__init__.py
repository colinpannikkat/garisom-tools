"""
# Optimization Framework

This module provides Bayesian optimization and hyperparameter tuning capabilities
for ecological models using Ray Tune and Optuna.

## Components

- `Optimizer`: Main optimization class using Ray Tune with Optuna search
- `OptimizationConfig`: Configuration for optimization experiments
- `GarisomOptimizationConfig`: GARISOM-specific optimization configuration

## Example Usage

```python
from garisom_tools.optimization import Optimizer, OptimizationConfig
from garisom_tools import GarisomModel

# Load configuration
config = OptimizationConfig.from_json('optimization_config.json')

# Create model instance
model = GarisomModel(
    run_kwargs={...},
    eval_kwargs={...}
)

# Setup optimizer
optimizer = Optimizer(
    model=model,
    config=config,
    verbosity=1
)

# Run optimization
best_results = optimizer.run()

# Access best parameters for each metric
best_rmse_params = best_results['rmse'].parameters
best_r2_params = best_results['r2'].parameters

print(f"Best RMSE: {best_results['rmse'].scores['rmse']}")
print(f"Best parameters: {best_rmse_params}")
```
"""

from .optimizer import *
from .config import *