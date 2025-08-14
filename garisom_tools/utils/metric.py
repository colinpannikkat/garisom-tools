"""
# Evaluation Metrics

This module provides evaluation metrics and performance measures for comparing
model predictions with ground truth observations. It includes standard metrics
from scikit-learn as well as custom ecological modeling metrics.

## Functions

- `nash_sutcliffe_efficiency`: Nash-Sutcliffe model efficiency coefficient
- `normalized_nash_sutcliffe_efficiency`: Normalized Nash-Sutcliffe efficiency

## Classes

- `Metric`: Base metric class with name, output variable, and evaluation function
- `MSE`, `RMSE`, `R2`, `MAPE`, `MADE`, `NNSE`: Specific metric implementations
- `Mode`: Enumeration for optimization modes (min/max)

## Example Usage

```python
from garisom_tools.utils.metric import Metric, nash_sutcliffe_efficiency, Mode
import numpy as np

# Create custom metric
nse_metric = Metric('nse', 'leaftemp', nash_sutcliffe_efficiency)

# Create from name
rmse_metric = Metric.from_name('rmse', 'leaftemp')

# Use metric function directly
predictions = np.array([1.0, 2.0, 3.0])
observations = np.array([1.1, 1.9, 3.2])
nse_value = nash_sutcliffe_efficiency(predictions, observations)

# Optimization modes
min_mode = Mode.from_name('min')  # For RMSE, MSE
max_mode = Mode.from_name('max')  # For R², NSE
```
"""

from enum import Enum
import numpy as np
from dataclasses import dataclass
from typing import Callable

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error
)


def nash_sutcliffe_efficiency(predictions, targets):
    """
    Nash-Sutcliffe model efficiency coefficient.

    The Nash-Sutcliffe efficiency (NSE) is a normalized statistic that determines
    the relative magnitude of the residual variance compared to the measured data
    variance. NSE ranges from -∞ to 1, where 1 indicates perfect agreement.

    Args:
        predictions (array-like): Model predicted values.
        targets (array-like): Observed/ground truth values.

    Returns:
        float: Nash-Sutcliffe efficiency coefficient.
            - NSE = 1: Perfect agreement
            - NSE = 0: Model predictions as accurate as mean of observations
            - NSE < 0: Model worse than using mean of observations

    Formula:
        NSE = 1 - (Σ(targets - predictions)²) / (Σ(targets - mean(targets))²)

    Example:
        ```python
        import numpy as np

        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        nse = nash_sutcliffe_efficiency(predicted, observed)
        print(f"Nash-Sutcliffe Efficiency: {nse:.3f}")
        ```

    Reference:
        Nash, J. E. and Sutcliffe, J. V. (1970). River flow forecasting through
        conceptual models part I — A discussion of principles. Journal of Hydrology,
        10(3), 282-290.
    """
    return 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))


def normalized_nash_sutcliffe_efficiency(predictions, targets):
    """
    Normalized Nash-Sutcliffe model efficiency coefficient.

    The normalized NSE transforms the standard NSE to a range of 0 to 1,
    making it easier to interpret and compare across different models and datasets.

    Args:
        predictions (array-like): Model predicted values.
        targets (array-like): Observed/ground truth values.

    Returns:
        float: Normalized Nash-Sutcliffe efficiency coefficient (0 to 1).
            - NNSE = 1: Perfect agreement
            - NNSE = 0.5: Model predictions as accurate as mean of observations
            - NNSE < 0.5: Model worse than using mean of observations

    Formula:
        NNSE = 1 / (2 - NSE)

    Example:
        ```python
        import numpy as np

        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        nnse = normalized_nash_sutcliffe_efficiency(predicted, observed)
        print(f"Normalized NSE: {nnse:.3f}")
        ```

    Note:
        This normalization is useful when NSE values are very negative,
        as it constrains the metric to a bounded range.
    """
    return 1 / (2 - nash_sutcliffe_efficiency(predictions, targets))


@dataclass
class Metric:
    """
    Base class for evaluation metrics.

    This class encapsulates an evaluation metric with its name, target output variable,
    and evaluation function. It provides a standardized interface for different
    types of performance measures used in model evaluation.

    Attributes:
        name (str): Unique identifier for the metric (used in optimization).
        output_name (str): Name of the model output variable to evaluate.
        func (Callable): Function that computes the metric value.

    Example:
        ```python
        from sklearn.metrics import mean_squared_error

        # Create custom metric
        mse_metric = Metric(
            name='mse_temp',
            output_name='temperature',
            func=mean_squared_error
        )

        # Use metric
        predictions = [1.0, 2.0, 3.0]
        targets = [1.1, 1.9, 3.2]
        score = mse_metric.func(targets, predictions)
        ```
    """
    name: str
    output_name: str
    func: Callable

    @staticmethod
    def from_name(metric_name: str, optim_name: str) -> "Metric":
        """
        Create a Metric instance from string identifiers.

        This factory method creates the appropriate metric subclass based on
        the metric name and handles output name parsing for duplicate metrics.

        Args:
            metric_name (str): Type of metric to create. Supported values:
                - 'mse': Mean Squared Error
                - 'rmse': Root Mean Squared Error
                - 'r2': R-squared coefficient
                - 'mape': Mean Absolute Percentage Error
                - 'made': Median Absolute Error
                - 'nnse': Normalized Nash-Sutcliffe Efficiency
            optim_name (str): Optimization name that may include suffix for disambiguation.
                Format: 'output_name' or 'output_name.suffix'

        Returns:
            Metric: Appropriate metric subclass instance.

        Raises:
            ValueError: If metric_name is not recognized.

        Example:
            ```python
            # Simple metric
            rmse_metric = Metric.from_name('rmse', 'leaf_temp')
            # rmse_metric.name = 'leaf_temp'
            # rmse_metric.output_name = 'leaf_temp'

            # Metric with suffix for disambiguation
            rmse_a = Metric.from_name('rmse', 'leaf_temp.morning')
            # rmse_a.name = 'leaf_temp.morning'
            # rmse_a.output_name = 'leaf_temp'

            rmse_b = Metric.from_name('rmse', 'leaf_temp.evening')
            # rmse_b.name = 'leaf_temp.evening'
            # rmse_b.output_name = 'leaf_temp'
            ```

        Note:
            The suffix mechanism allows multiple metrics on the same output variable,
            which is useful when optimizing multiple aspects of model performance
            or when using different evaluation periods.
        """
        mapping = {
            "mse": MSE,
            "rmse": RMSE,
            "r2": R2,
            "mape": MAPE,
            "made": MADE,
            "nnse": NNSE,
            "nse": NSE
        }

        # Get metric class
        metric_cls = mapping.get(metric_name.lower())
        if metric_cls is None:
            raise ValueError(f"Unknown metric name: {metric_name}")

        # Check if name is hyphenated, indicating name is different from output
        # Optuna doesn't allow for 'metric' names to be the same, so we need to
        # differentiate between duplicates of outputs when using different
        # metrics. Ex: P-PD.A and P-PD.B must both be mapped to P-PD.
        output_name = optim_name
        if (i := optim_name.rfind(".")) != -1:
            output_name = optim_name[:i]

        return metric_cls(output_name, optim_name)


# Evaluation Metrics
class MSE(Metric):
    """
    Mean Squared Error metric.

    Computes the mean of the squared differences between predictions and targets.
    Lower values indicate better model performance.

    Formula: MSE = (1/n) * Σ(target - prediction)²

    Args:
        output_name (str): Name of the model output variable.
        name (str, optional): Metric identifier. Defaults to "mse".
    """
    def __init__(self, output_name: str, name: str = "mse"):
        super().__init__(name=name, output_name=output_name, func=mean_squared_error)


class RMSE(Metric):
    """
    Root Mean Squared Error metric.

    Computes the square root of the mean squared differences between predictions
    and targets. Has the same units as the target variable.

    Formula: RMSE = √(MSE) = √((1/n) * Σ(target - prediction)²)

    Args:
        output_name (str): Name of the model output variable.
        name (str, optional): Metric identifier. Defaults to "rmse".
    """
    def __init__(self, output_name: str, name: str = "rmse"):
        super().__init__(name=name, output_name=output_name, func=root_mean_squared_error)


class R2(Metric):
    """
    R-squared (coefficient of determination) metric.

    Measures the proportion of variance in the target variable that is predictable
    from the model. Higher values (closer to 1) indicate better performance.

    Formula: R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(target - prediction)² and SS_tot = Σ(target - mean(target))²

    Args:
        output_name (str): Name of the model output variable.
        name (str, optional): Metric identifier. Defaults to "r2".
    """
    def __init__(self, output_name: str, name: str = "r2"):
        super().__init__(name=name, output_name=output_name, func=r2_score)


class MAPE(Metric):
    """
    Mean Absolute Percentage Error metric.

    Computes the mean of the absolute percentage differences between predictions
    and targets. Expressed as a percentage.

    Formula: MAPE = (100/n) * Σ(|target - prediction| / |target|)

    Args:
        output_name (str): Name of the model output variable.
        name (str, optional): Metric identifier. Defaults to "mape".

    Note:
        Undefined when target values are zero. Use with caution for data
        containing zero or near-zero values.
    """
    def __init__(self, output_name: str, name: str = "mape"):
        super().__init__(name=name, output_name=output_name, func=mean_absolute_percentage_error)


class MADE(Metric):
    """
    Median Absolute Error metric.

    Computes the median of the absolute differences between predictions and targets.
    More robust to outliers than mean-based metrics.

    Formula: MADE = median(|target - prediction|)

    Args:
        output_name (str): Name of the model output variable.
        name (str, optional): Metric identifier. Defaults to "made".
    """
    def __init__(self, output_name: str, name: str = "made"):
        super().__init__(name=name, output_name=output_name, func=median_absolute_error)


class NNSE(Metric):
    """
    Normalized Nash-Sutcliffe Efficiency metric.

    Computes the normalized Nash-Sutcliffe efficiency coefficient, which is
    bounded between 0 and 1. Higher values indicate better performance.

    Formula: NNSE = 1 / (2 - NSE)
    where NSE = 1 - (Σ(target - prediction)² / Σ(target - mean(target))²)

    Args:
        output_name (str): Name of the model output variable.
        name (str, optional): Metric identifier. Defaults to "nnse".
    """
    def __init__(self, output_name: str, name: str = "nnse"):
        super().__init__(name=name, output_name=output_name, func=normalized_nash_sutcliffe_efficiency)


class NSE(Metric):
    """
    Nash-Sutcliffe Efficiency metric.

    Computes the Nash-Sutcliffe efficiency coefficient, which is
    bounded between -inf and 1. Higher values indicate better performance.

    Formula: NSE = 1 - (Σ(target - prediction)² / Σ(target - mean(target))²)

    Args:
        output_name (str): Name of the model output variable.
        name (str, optional): Metric identifier. Defaults to "nse".
    """
    def __init__(self, output_name: str, name: str = "nse"):
        super().__init__(name=name, output_name=output_name, func=normalized_nash_sutcliffe_efficiency)


# Evaluation Modes
class Mode(str, Enum):
    """
    Enumeration for optimization modes.

    Defines whether a metric should be minimized or maximized during optimization.
    This is used by optimization frameworks to determine the direction of improvement.

    Values:
        MAX: Maximize the metric (higher values are better)
        MIN: Minimize the metric (lower values are better)

    Example:
        ```python
        from garisom_tools.utils.metric import Mode

        # Metrics that should be minimized
        rmse_mode = Mode.MIN  # Lower RMSE is better
        mse_mode = Mode.MIN   # Lower MSE is better

        # Metrics that should be maximized
        r2_mode = Mode.MAX    # Higher R² is better
        nse_mode = Mode.MAX   # Higher NSE is better

        # Create from string
        min_mode = Mode.from_name('min')
        max_mode = Mode.from_name('max')
        ```
    """
    MAX = "max"
    MIN = "min"

    @staticmethod
    def from_name(name: str) -> "Mode":
        """
        Create a Mode instance from a string name.

        Args:
            name (str): Mode name, case-insensitive. Must be 'min' or 'max'.

        Returns:
            Mode: Corresponding Mode enumeration value.

        Raises:
            ValueError: If name is not 'min' or 'max'.

        Example:
            ```python
            min_mode = Mode.from_name('MIN')    # Case insensitive
            max_mode = Mode.from_name('max')

            # Use in configuration
            modes = [Mode.from_name(m) for m in ['min', 'max', 'min']]
            ```
        """
        try:
            return Mode(name.lower())
        except ValueError:
            raise ValueError(f"Unknown mode: {name}")
