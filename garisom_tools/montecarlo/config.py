"""Configuration classes for Monte Carlo simulation settings.

This module provides configuration classes for managing Monte Carlo simulation
parameters including search space definitions, output specifications, and
execution settings. It supports serialization to and from JSON format for
easy persistence and loading of simulation configurations.

The module includes both a base configuration class and a specialized version
for GARISOM model simulations with additional population-level
parameters.

Typical usage example:

    from garisom_tools.montecarlo import MonteCarloConfig

    config = MonteCarloConfig.from_json("mc_config.json")
    config.num_samples = 2048
    config.to_json("updated_config.json")
"""

from ..config.space import SpaceConfig
from garisom_tools.utils.distributions import (
    get_scipy_normal,
    get_scipy_truncated_normal,
    get_scipy_uniform
)

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MonteCarloConfig:
    """Configuration class for Monte Carlo simulation settings.

    This class provides a structured way to configure Monte Carlo simulation
    parameters including parameter space definitions, output specifications,
    and execution settings. It supports serialization to and from JSON format
    for easy persistence and configuration management.

    The configuration handles the mapping of distribution types to appropriate
    scipy distribution functions and manages the search space for parameter
    sampling during Monte Carlo simulations.

    Attributes:
        space (SpaceConfig, optional): Configuration for the parameter search
            space. Contains parameter definitions and their probability
            distributions for sampling. Defaults to None.
        outputs (list[str], optional): List of output variable names to collect
            from simulation runs. Specifies which model outputs to track and
            analyze. Defaults to None.
        num_worker (int): Number of parallel workers to use during simulation
            execution. More workers can speed up computation but require more
            system resources. Defaults to 4.
        num_samples (int): Number of Monte Carlo samples to generate during
            simulation. Higher values typically lead to more accurate statistics
            but take longer to compute. Defaults to 100.

    Example:
        Basic usage:

        ```python
        config = MonteCarloConfig(
            outputs=["P-PD", "leaftemp"],
            num_worker=8,
            num_samples=1024
        )
        config.to_json("mc_config.json")

        # Load from file
        loaded_config = MonteCarloConfig.from_json("mc_config.json")
        ```
    """

    space: SpaceConfig = None
    outputs: list[str] = None
    num_worker: int = 4
    num_samples: int = 100

    @classmethod
    def from_json(cls, infile: str):
        """Create a MonteCarloConfig instance from a JSON file.

        Loads configuration data from a JSON file and creates a new instance
        with the specified parameters. Handles conversion of distribution
        type strings to appropriate scipy distribution functions for the
        search space configuration.

        Args:
            infile (str): Path to the JSON file containing the configuration
                data. The file should contain a valid JSON object with
                configuration parameters including space, outputs, and
                execution settings.

        Returns:
            MonteCarloConfig: A new MonteCarloConfig instance initialized
                with data from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            KeyError: If required configuration keys are missing.
            TypeError: If the loaded data doesn't match the expected structure.

        Note:
            The method automatically maps distribution type strings ("normal",
            "truncnorm", "uniform") to their corresponding scipy distribution
            functions for proper parameter space configuration.

        Example:
            ```python
            config = MonteCarloConfig.from_json("simulation_config.json")
            print(f"Using {config.num_samples} samples with {config.num_worker} workers")
            ```
        """
        with open(infile, "r") as f:
            data = json.load(f)

        # Map distribution type strings to scipy distribution functions
        mapping = {
            "normal": get_scipy_normal,
            "truncnorm": get_scipy_truncated_normal,
            "uniform": get_scipy_uniform
        }

        # Convert space configuration using distribution mapping
        data['space'] = SpaceConfig.from_dict(mapping, data['space'])
        return cls(**data)

    def to_json(self, outfile: str):
        """Serialize the configuration to a JSON file.

        Converts the current configuration instance to JSON format and saves
        it to the specified file. All instance attributes will be included
        in the serialized output using the dataclass asdict() function.

        Args:
            outfile (str): Path to the output file where the JSON will be saved.
                The file will be created if it doesn't exist, but will raise
                an error if the file already exists.

        Returns:
            None

        Raises:
            FileExistsError: If the specified file already exists.
            PermissionError: If there are insufficient permissions to write
                the file.
            OSError: If there are filesystem-related errors during writing.

        Note:
            This method uses the dataclass `asdict()` function for serialization,
            which recursively converts all dataclass fields to dictionaries.
            The file is opened in exclusive creation mode ("+x") to prevent
            accidental overwrites.

        Example:
            ```python
            config = MonteCarloConfig(num_samples=2048, num_worker=8)
            config.to_json("new_simulation_config.json")
            ```
        """
        with open(outfile, "+x") as f:
            json.dump(asdict(self), f)


@dataclass
class GarisomMonteCarloConfig(MonteCarloConfig):
    """Specialized Monte Carlo configuration for Garisom ecosystem models.

    Extends the base MonteCarloConfig with additional parameters specific
    to Garisom ecosystem model simulations. Includes population-level settings
    and parameter overrides for GARISOM Monte Carlo analysis.

    This configuration class is designed for use with Garisom ecosystem models
    where population-level parameters and custom parameter overrides are
    frequently needed for uncertainty quantification and sensitivity analysis.

    Attributes:
        population (int): Population identifier or index for the simulation.
            Used to specify which population within the ecosystem model to
            simulate. Defaults to 1.
        parameters (dict[str, float], optional): Dictionary of parameter
            overrides to apply during simulation. These values will take
            precedence over sampled values for matching parameter names.
            Useful for fixing certain parameters while allowing others to
            vary. Defaults to None.
        space (SpaceConfig, optional): Inherited from MonteCarloConfig.
            Parameter search space configuration.
        outputs (list[str], optional): Inherited from MonteCarloConfig.
            List of output variables to collect.
        num_worker (int): Inherited from MonteCarloConfig. Number of parallel
            workers. Defaults to 4.
        num_samples (int): Inherited from MonteCarloConfig. Number of Monte
            Carlo samples. Defaults to 100.

    Example:
        ```python
        garisom_config = GarisomMonteCarloConfig(
            population=2,
            parameters={"water_stress": 0.5, "temperature_max": 35.0},
            num_worker=6,
            num_samples=512
        )
        garisom_config.to_json("garisom_mc_config.json")
        ```
    """

    population: int = 1
    parameters: Optional[dict[str, float]] = None
