"""
# GARISOM Tools

A comprehensive toolkit for working with the GARISOM/Sperry model [**[1][2]**](#citation), providing functionality for:

- **Model Interface**: Abstract base classes and concrete implementation for running GARISOM simulations
- **Monte Carlo Simulations**: Tools for uncertainty quantification and sensitivity analysis
- **Optimization**: Bayesian optimization and hyperparameter tuning capabilities
- **Configuration Management**: Flexible configuration systems for metrics, parameter spaces, and distributions
- **Sensitivity Analysis**: Easy Sobol sensitivity analysis across any parameter input and model output
- **Results Analysis**: Data structures and utilities for processing simulation outputs

## Main Components

- `Model`: Base classes for model execution and evaluation. Enables users to easily
    derive new model classes and use the existing API with their custom models.
- `GarisomModel`: Concrete implementation for interfacing with the GARISOM/Sperry model
- `montecarlo`: Monte Carlo simulation framework
- `optimization`: Optimization and parameter tuning tools
- `config`: Configuration management for metrics and parameter spaces
- `sa`: Sensitivity analysis tools
- `utils`: Utility functions for metrics, distributions, and results handling

## Planned Additions
- More stomatal optimization models
- Supporting multiple years
- Better tests
- Build in default parameter and configuration files

## Example Usage

```python
from garisom_tools import GarisomModel
from garisom_tools.montecarlo import Sim, MonteCarloConfig
from garisom_tools.optimization import Optimizer, OptimizationConfig
from garisom_tools.sa import SensitivityAnalysis, SensitivityAnalysisConfig

# Load configurations
mc_config = MonteCarloConfig.from_json("mc_config.json")
opt_config = OptimizationConfig.from_json("opt_config.json")
sa_config = SensitivityAnalysisConfig.from_json("sa_config.json")

# Create model instance
model = GarisomModel(run_kwargs={...}, eval_kwargs={...})

# Run Sensitivity Analysis
sa = SensitivityAnalysis(model, sa_config)
sa.run("./results")

# Run Monte Carlo simulation
sim = Sim(model, mc_config, run_kwargs={...})
results = sim.run(n=1000, parallel=True)

# Run optimization
optimizer = Optimizer(model, opt_config)
best_params = optimizer.run()
```

## Citations

The GARISOM/Sperry model was originally published in the following papers. If you use
this package in your work, please cite the following.

[1] Sperry JS, Venturas MD, Anderegg WRL, Mencuccini M, Mackay DS, Wang Y, Love DM. 2017.
    Predicting stomatal responses to the environment from the optimization of photosynthetic gain and hydraulic cost.
    Plant, Cell & Environment 40: 816-830.

[2] Venturas MD, Sperry JS, Love DM, Frehner EH, Allred MG, Wang Y, Anderegg WRL. 2018.
    A stomatal control model based on optimization of carbon gain versus hydraulic risk predicts aspen sapling
    responses to drought. New Phytologist 220: 836-850.
"""

from .model import *
