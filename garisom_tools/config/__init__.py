"""
# Configuration Management

This module provides configuration classes for managing metrics, parameter spaces,
and optimization settings used throughout the garisom_tools package.

## Components

- **MetricConfig**: Configuration for evaluation metrics and optimization modes
- **SpaceConfig**: Configuration for parameter sampling spaces and distributions

## Example Usage

```python
from garisom_tools.config import MetricConfig, SpaceConfig

# Create metric configuration
metric_config = MetricConfig.from_dict({
    'metrics': ['rmse', 'r2'],
    'modes': ['min', 'max'],
    'params': ['leaf_temp', 'leaf_temp']
})

# Create parameter space configuration
space_config = SpaceConfig.from_dict(
    mapping={'uniform': FloatDistribution, 'normal': NormalDistribution},
    data={
        'growth_rate': ['uniform', [0.01, 0.1]],
        'water_efficiency': ['normal', [0.8, 0.1]]
    }
)
```
"""

from .metric import *
from .space import *
