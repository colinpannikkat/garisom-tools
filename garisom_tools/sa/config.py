"""Configuration classes for sensitivity analysis settings.

This module provides configuration classes for managing sensitivity analysis
parameters including problem definitions, metric configurations, and
execution settings. It supports serialization to and from JSON format for
easy persistence and loading of sensitivity analysis configurations.

The module includes problem definition classes for SALib compatibility and
comprehensive configuration management for Sobol sensitivity analysis.

Typical usage example:

    from garisom_tools.sa import SensitivityAnalysisConfig

    config = SensitivityAnalysisConfig.from_json("sa_config.json")
    config.samples = 4096
    config.to_json("updated_sa_config.json")
"""

from dataclasses import dataclass, asdict
from ..config.metric import MetricConfig
import json


@dataclass
class SensitivityAnalysisProblem:
    """Problem definition for sensitivity analysis using SALib.

    This class defines the parameter space and bounds for sensitivity analysis
    compatible with the SALib library. It encapsulates the problem structure
    required for Sobol sequence sampling and sensitivity index computation.

    The problem definition includes parameter names, bounds, and dimensionality
    information needed for proper sensitivity analysis setup.

    Attributes:
        num_vars (int): Number of variables (parameters) in the problem.
            Must match the length of names and bounds lists.
        names (list[str]): List of parameter names corresponding to each
            variable in the problem. Used for labeling and identification
            in sensitivity analysis results.
        bounds (list[list[float]]): List of [min, max] bounds for each
            parameter. Each inner list contains exactly two floats representing
            the lower and upper bounds for the corresponding parameter.

    Example:
        ```python
        problem = SensitivityAnalysisProblem(
            num_vars=3,
            names=["temperature", "humidity", "pressure"],
            bounds=[[15.0, 35.0], [0.3, 0.9], [900.0, 1100.0]]
        )
        problem_dict = problem.to_dict()
        ```
    """

    num_vars: int
    names: list[str]
    bounds: list[list[float]]

    def to_dict(self):
        """Convert the problem definition to a dictionary format.

        Converts the problem instance to a dictionary format compatible
        with SALib functions. This method is used internally to interface
        with SALib's sampling and analysis functions.

        Returns:
            dict: Dictionary representation of the problem with keys
                'num_vars', 'names', and 'bounds' suitable for SALib functions.

        Example:
            ```python
            problem = SensitivityAnalysisProblem(num_vars=2, names=["x", "y"],
                                               bounds=[[0, 1], [0, 1]])
            salib_dict = problem.to_dict()
            ```
        """
        return asdict(self)


@dataclass
class SensitivityAnalysisConfig:
    """Configuration class for sensitivity analysis execution settings.

    This class provides a comprehensive configuration structure for running
    sensitivity analysis experiments. It combines problem definitions, metric
    configurations, and execution parameters to enable systematic sensitivity
    analysis of model parameters.

    The configuration supports time-series analysis with configurable start
    and end days, population-specific analysis, and parallel execution settings.

    Attributes:
        problem (SensitivityAnalysisProblem): Problem definition containing
            parameter names, bounds, and dimensionality for sensitivity analysis.
        metric (MetricConfig): Configuration for evaluation metrics used to
            assess model performance and compute sensitivity indices.
        workers (int): Number of parallel workers to use during model execution.
            More workers can significantly speed up computation for large sample
            sizes.
        samples (int): Number of Sobol sequence samples to generate. Should be
            a power of 2 for optimal space-filling properties. Larger values
            provide more accurate sensitivity indices but require more computation.
        start_day (int): Starting day of year for the analysis period. Used to
            focus sensitivity analysis on specific temporal windows.
        end_day (int): Ending day of year for the analysis period. Combined with
            start_day to define the temporal scope of analysis.
        pop (int): Population identifier or index for population-specific
            sensitivity analysis. Useful for ecosystem models with multiple
            populations or demographic groups.

    Example:
        ```python
        problem = SensitivityAnalysisProblem(
            num_vars=3,
            names=["param1", "param2", "param3"],
            bounds=[[0, 1], [10, 50], [0.1, 0.9]]
        )
        config = SensitivityAnalysisConfig(
            problem=problem,
            metric=metric_config,
            workers=8,
            samples=1024,
            start_day=150,
            end_day=250,
            pop=1
        )
        ```
    """

    problem: SensitivityAnalysisProblem
    metric: MetricConfig
    workers: int
    samples: int
    start_day: int
    end_day: int
    pop: int

    @classmethod
    def from_json(cls, infile: str):
        """Create a SensitivityAnalysisConfig instance from a JSON file.

        Loads configuration data from a JSON file and creates a new instance
        with the specified parameters. Handles conversion of nested problem
        and metric configurations from their dictionary representations.

        Args:
            infile (str): Path to the JSON file containing the configuration
                data. The file should contain a valid JSON object with all
                required sensitivity analysis configuration parameters.

        Returns:
            SensitivityAnalysisConfig: A new SensitivityAnalysisConfig instance
                initialized with data from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            TypeError: If the loaded data doesn't match the expected structure.
            KeyError: If required configuration keys are missing from the JSON.

        Note:
            The method automatically converts the 'problem' dictionary to a
            SensitivityAnalysisProblem instance and the 'metric' dictionary
            to a MetricConfig instance for proper type safety.

        Example:
            ```python
            config = SensitivityAnalysisConfig.from_json("sa_config.json")
            print(f"Running SA with {config.samples} samples using {config.workers} workers")
            ```
        """

        with open(infile, "r") as f:
            data = json.load(f)

        # Convert nested dictionaries to proper dataclass instances
        data['problem'] = SensitivityAnalysisProblem(**data['problem'])
        data['metric'] = MetricConfig.from_dict(data["metric"])
        return cls(**data)

    def to_json(self, outfile: str):
        """Serialize the configuration to a JSON file.

        Converts the current configuration instance to JSON format and saves
        it to the specified file with proper indentation for readability.
        All instance attributes will be included in the serialized output.

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
            config = SensitivityAnalysisConfig(problem=prob, metric=met, ...)
            config.to_json("new_sa_config.json")
            ```
        """
        with open(outfile, "+x") as f:
            json.dump(asdict(self), f, indent=4)
