"""
# Metric Configuration

This module provides configuration classes for managing evaluation metrics
and their optimization modes in model evaluation and optimization workflows.

## Classes

- `MetricConfig`: Main configuration class for organizing metrics and modes

## Example Usage

```python
from garisom_tools.config.metric import MetricConfig

# Create from dictionary
config = MetricConfig.from_dict({
    'metrics': ['rmse', 'r2', 'nse'],
    'modes': ['min', 'max', 'max'],
    'params': ['leaf_temp', 'transpiration', 'leaf_temp.nse']
})

# Access configured metrics
for metric, mode in zip(config.metrics, config.modes):
    print(f"{metric.name}: optimize {mode}")
```
"""

from garisom_tools.utils.metric import Metric, Mode
from dataclasses import dataclass


@dataclass
class MetricConfig:
    """
    Configuration for evaluation metrics and optimization modes.

    This class organizes multiple evaluation metrics with their corresponding
    optimization modes (minimize or maximize) for use in model evaluation
    and optimization workflows.

    Attributes:
        metrics (list[Metric]): List of metric instances for evaluation.
        modes (list[Mode]): List of optimization modes ('min' or 'max') for each metric.

    Example:
        ```python
        from garisom_tools.config.metric import MetricConfig

        # Create configuration
        config = MetricConfig.from_dict({
            'metrics': ['rmse', 'r2'],
            'modes': ['min', 'max'],
            'params': ['leaf_temp', 'leaf_temp']
        })

        # Use in evaluation
        for metric, mode in zip(config.metrics, config.modes):
            print(f"Metric: {metric.name}, Mode: {mode}, Output: {metric.output_name}")
        ```
    """
    metrics: list[Metric]
    modes: list[Mode]

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a MetricConfig from a dictionary specification.

        Args:
            data (dict): Configuration dictionary with keys:
                - 'params' (list[str]): Parameter/output names for each metric
                - 'metrics' (list[str]): Metric type names (e.g., 'rmse', 'r2')
                - 'modes' (list[str]): Optimization modes ('min' or 'max')

        Returns:
            MetricConfig: Configured instance with metrics and modes.

        Example:
            ```python
            config_dict = {
                'metrics': ['rmse', 'r2', 'nse'],
                'modes': ['min', 'max', 'max'],
                'params': ['leaf_temp', 'transpiration', 'leaf_temp.alt']
            }

            config = MetricConfig.from_dict(config_dict)

            # Results in:
            # - RMSE on leaf_temp (minimize)
            # - R² on transpiration (maximize)
            # - NSE on leaf_temp with alt name (maximize)
            ```

        Note:
            - 'params' can include suffixes (e.g., 'var.A') for multiple metrics on same output
            - Metric names must be supported by the Metric.from_name() method
            - Mode names must be 'min' or 'max'
        """
        params = data.get("params", [])
        metrics = data.get("metrics", [])
        metrics = [Metric.from_name(m, p) for m, p in zip(metrics, params)]
        modes = [Mode.from_name(m) for m in data.get("modes", [])]
        return cls(metrics=metrics, modes=modes)
