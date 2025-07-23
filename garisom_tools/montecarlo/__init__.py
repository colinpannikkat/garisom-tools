"""
# Monte Carlo Simulations

This module provides functionality for running Monte Carlo simulations with
ecological models, including uncertainty quantification and sensitivity analysis.

## Components

- `Sim`: Main simulation class for running Monte Carlo experiments
- `MonteCarloConfig`: Configuration for Monte Carlo simulations
- `GarisomMonteCarloConfig`: GARISOM-specific Monte Carlo configuration

## Example Usage

```python
from garisom_tools.montecarlo import Sim, MonteCarloConfig
from garisom_tools import GarisomModel

# Load configuration
config = MonteCarloConfig.from_json('mc_config.json')

# Create model instance
model = GarisomModel(run_kwargs={...})

# Setup simulation
sim = Sim(
    model=model,
    config=config,
    run_kwargs={...},
    engine='sobol',  # Quasi-random sampling
    seed=42
)

# Run Monte Carlo simulation
results = sim.run(n=1000, parallel=True, workers=8)

# Analyze results
stats = sim.analyze(results, index_columns=['julian_day', 'hour'])
stats.save('/path/to/results/')
```
"""

from .sim import *
from .config import *