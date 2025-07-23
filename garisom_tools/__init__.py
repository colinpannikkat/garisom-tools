"""
# Garisom Tools

A comprehensive toolkit for working with the GARISOM ecological model, providing functionality for:

- **Model Interface**: Abstract base classes and concrete implementations for running GARISOM simulations
- **Monte Carlo Simulations**: Tools for uncertainty quantification and sensitivity analysis
- **Optimization**: Bayesian optimization and hyperparameter tuning capabilities
- **Configuration Management**: Flexible configuration systems for metrics, parameter spaces, and distributions
- **Results Analysis**: Data structures and utilities for processing simulation outputs

## Main Components

- `Model`: Base classes for model execution and evaluation
- `GarisomModel`: Concrete implementation for GARISOM ecological model
- `montecarlo`: Monte Carlo simulation framework
- `optimization`: Optimization and parameter tuning tools
- `config`: Configuration management for metrics and parameter spaces
- `utils`: Utility functions for metrics, distributions, and results handling

## Example Usage

```python
from garisom_tools import GarisomModel
from garisom_tools.montecarlo import Sim, MonteCarloConfig
from garisom_tools.optimization import Optimizer, OptimizationConfig

# Load configurations
mc_config = MonteCarloConfig.from_json("mc_config.json")
opt_config = OptimizationConfig.from_json("opt_config.json")

# Create model instance
model = GarisomModel(run_kwargs={...}, eval_kwargs={...})

# Run Monte Carlo simulation
sim = Sim(model, mc_config, run_kwargs={...})
results = sim.run(n=1000, parallel=True)

# Run optimization
optimizer = Optimizer(model, opt_config)
best_params = optimizer.run()
```
"""

from .model import *