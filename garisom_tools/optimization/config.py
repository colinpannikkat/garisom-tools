"""Configuration classes for optimization settings.

This module provides configuration classes for managing optimization parameters
and settings in the Garisom optimization framework.
"""

from ..config.space import SpaceConfig
from ..config.metric import MetricConfig
from garisom_tools.utils.distributions import (
    FloatDistribution,
    NormalDistribution,
    TruncatedNormalDistribution
)

import json
from dataclasses import dataclass, asdict


@dataclass
class OptimizationConfig:
    """Configuration class for optimization settings.

    This class provides a structured way to configure optimization parameters
    including search space, metrics, and execution settings. It supports
    serialization to and from JSON format for easy persistence and loading.

    Attributes:
        space (SpaceConfig, optional): Configuration for the search space. Contains
            parameter definitions and bounds for optimization variables. Defaults to None.
        metric (MetricConfig, optional): Configuration for the optimization metric.
            Defines how to evaluate and compare optimization results. Defaults to None.
        num_worker (int): Number of parallel workers to use during optimization.
            More workers can speed up evaluation but require more resources. Defaults to 4.
        num_samples (int): Number of samples to draw during optimization. Higher values
            typically lead to better results but take longer to compute. Defaults to 100.

    Example:
        Basic usage:

        ```python
        config = OptimizationConfig(
            num_worker=8,
            num_samples=200
        )
        config.to_json("optimization_config.json")

        # Load from file
        loaded_config = OptimizationConfig.from_json("optimization_config.json")
        ```
    """

    space: SpaceConfig = None
    metric: MetricConfig = None
    num_worker: int = 4
    num_samples: int = 100

    @classmethod
    def from_json(cls, infile: str):
        """Create an OptimizationConfig instance from a JSON file.

        Loads configuration data from a JSON file and creates a new instance
        with the specified parameters. Handles conversion of distribution
        strings to appropriate distribution classes.

        Args:
            infile (str): Path to the JSON file containing the configuration data.
                The file should contain a valid JSON object with configuration
                parameters.

        Returns:
            OptimizationConfig: A new OptimizationConfig instance initialized
                with data from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            TypeError: If the loaded data doesn't match the expected structure.
            KeyError: If required configuration keys are missing.

        Example:
            ```python
            config = OptimizationConfig.from_json("my_config.json")
            print(f"Using {config.num_worker} workers")
            ```
        """

        mapping = {
            "uniform": FloatDistribution,
            "norm": NormalDistribution,
            "truncnorm": TruncatedNormalDistribution,
        }

        with open(infile, "r") as f:
            data = json.load(f)
        data['metric'] = MetricConfig.from_dict(data["metric"]) if "metric" in data else None
        data['space'] = SpaceConfig.from_dict(mapping, data['space']) if 'space' in data else None
        return cls(**data)

    def to_json(self, outfile: str):
        """Serialize the configuration to a JSON file.

        Converts the current configuration instance to JSON format and saves
        it to the specified file. All instance attributes will be included
        in the serialized output.

        Args:
            outfile (str): Path to the output file where the JSON will be saved.
                The file will be created if it doesn't exist.

        Returns:
            None

        Raises:
            PermissionError: If there are insufficient permissions to write the file.
            OSError: If there are filesystem-related errors during writing.
            FileExistsError: If the file already exists and cannot be overwritten.

        Note:
            This method uses the dataclass `asdict()` function for serialization,
            which recursively converts all dataclass fields to dictionaries.

        Example:
            ```python
            config = OptimizationConfig(num_worker=8, num_samples=200)
            config.to_json("output_config.json")
            ```
        """
        with open(outfile, "+x") as f:
            json.dump(asdict(self), f)


@dataclass
class GarisomOptimizationConfig(OptimizationConfig):
    """Specialized optimization configuration for Garisom model optimization.

    Extends the base OptimizationConfig with additional parameters specific
    to Garisom ecosystem model optimization. Includes population-level settings
    and temporal constraints for optimization runs.

    Attributes:
        population (int): Population identifier or size parameter for the optimization.
            Used to specify which population or the population size to optimize. Defaults to 1.
        start_day (int): Starting day of year for the optimization period.
            Day 201 corresponds to approximately July 20th. Defaults to 201.
        end_day (int): Ending day of year for the optimization period.
            Day 236 corresponds to approximately August 24th. Defaults to 236.
        space (SpaceConfig, optional): Inherited from OptimizationConfig. Configuration
            for the search space.
        metric (MetricConfig, optional): Inherited from OptimizationConfig. Configuration
            for the optimization metric.
        num_worker (int): Inherited from OptimizationConfig. Number of parallel workers.
        num_samples (int): Inherited from OptimizationConfig. Number of optimization samples.

    Example:
        ```python
        garisom_config = GarisomOptimizationConfig(
            population=2,
            start_day=180,
            end_day=250,
            num_worker=6,
            num_samples=150
        )
        garisom_config.to_json("garisom_optimization.json")
        ```
    """

    population: int = 1
    start_day: int = 201
    end_day: int = 236
