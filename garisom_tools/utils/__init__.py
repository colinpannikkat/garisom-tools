"""
# Utilities

This module provides utility functions and classes for working with metrics,
probability distributions, and results management in the garisom_tools package.

## Components

- **metric**: Evaluation metrics and performance measures
- **distributions**: Custom probability distributions for sampling
- **results**: Data structures for storing and managing simulation results

## Example Usage

```python
from garisom_tools.utils.metric import Metric, nash_sutcliffe_efficiency
from garisom_tools.utils.distributions import NormalDistribution
from garisom_tools.utils.results import EvalResults, MetricResult

# Create custom metric
nse_metric = Metric('nse', 'streamflow', nash_sutcliffe_efficiency)

# Use custom distribution
normal_dist = NormalDistribution(mu=0.5, sigma=0.1)
sample = normal_dist._sample(rng)

# Store results
results = {'rmse': 0.15, 'r2': 0.85}
eval_results = EvalResults(results)
```
"""
