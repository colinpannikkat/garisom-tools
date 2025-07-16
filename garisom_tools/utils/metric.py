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
    """Nash-Sutcliffe model efficiency coefficient"""
    return 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))


def normalized_nash_sutcliffe_efficiency(predictions, targets):
    """Normalized Nash-Sutcliffe model efficiency coefficient"""
    return 1 / (2 - nash_sutcliffe_efficiency(predictions, targets))


@dataclass
class Metric:
    name: str
    output_name: str
    func: Callable

    @staticmethod
    def from_name(metric_name: str, optim_name: str) -> "Metric":
        mapping = {
            "mse": MSE,
            "rmse": RMSE,
            "r2": R2,
            "mape": MAPE,
            "made": MADE,
            "nnse": NNSE
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
    def __init__(self, output_name: str, name: str = "mse"):
        super().__init__(name=name, output_name=output_name, func=mean_squared_error)


class RMSE(Metric):
    def __init__(self, output_name: str, name: str = "rmse"):
        super().__init__(name=name, output_name=output_name, func=root_mean_squared_error)


class R2(Metric):
    def __init__(self, output_name: str, name: str = "r2"):
        super().__init__(name=name, output_name=output_name, func=r2_score)


class MAPE(Metric):
    def __init__(self, output_name: str, name: str = "mape"):
        super().__init__(name=name, output_name=output_name, func=mean_absolute_percentage_error)


class MADE(Metric):
    def __init__(self, output_name: str, name: str = "made"):
        super().__init__(name=name, output_name=output_name, func=median_absolute_error)


class NNSE(Metric):
    def __init__(self, output_name: str, name: str = "nnse"):
        super().__init__(name=name, output_name=output_name, func=normalized_nash_sutcliffe_efficiency)


# Evaluation Modes
class Mode(str, Enum):
    MAX = "max"
    MIN = "min"

    @staticmethod
    def from_name(name: str) -> "Mode":
        try:
            return Mode(name.lower())
        except ValueError:
            raise ValueError(f"Unknown mode: {name}")
