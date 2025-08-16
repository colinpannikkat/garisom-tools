"""
# Sensitivity Analysis

This module provides functionality for conducting sensitivity analysis on
ecological models to understand parameter importance and model behavior.

## Status

This module is currently under development. The sensitivity analysis
functionality is being ported from the original SA.py implementation
into a reusable class-based structure.

## Planned Components

- `SensitivityAnalysis`: Main class for conducting sensitivity analysis
- `SobolAnalysis`: Sobol sensitivity indices calculation
- `SAConfig`: Configuration for sensitivity analysis experiments

## Example Usage (Planned)

```python
from garisom_tools.sa import SensitivityAnalysis, SAConfig
from garisom_tools import GarisomModel

# Load configuration
sa_config = SAConfig.from_json('sa_config.json')

# Create model instance
model = GarisomModel(run_kwargs={...})

# Setup sensitivity analysis
sa = SensitivityAnalysis(
    model=model,
    config=sa_config,
    method='sobol'
)

# Run analysis
results = sa.run(n_samples=1000)

# Get sensitivity indices
first_order = results.first_order_indices
total_order = results.total_order_indices
```
"""

from .sa import *
from .config import *
