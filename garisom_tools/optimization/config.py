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
    space: SpaceConfig = None
    metric: MetricConfig = None
    num_worker: int = 4
    num_samples: int = 100

    @classmethod
    def from_json(cls, infile: str):
        """
        Creates an instance of the class from a JSON file.

        Args:
            infile (str): Path to the JSON file containing the configuration data.

        Returns:
            cls: An instance of the class initialized with data loaded from the JSON file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file is not a valid JSON.
            TypeError: If the data loaded from the file does not match the expected class signature.
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
        """
        Serializes the object to a JSON file.

        Args:
            outfile (str): The path to the output file where the JSON representation will be saved.

        Notes:
            The method uses the object's __dict__ attribute for serialization,
            which means all instance attributes will be included in the JSON output.
        """
        with open(outfile, "+x") as f:
            json.dump(asdict(self), f)


@dataclass
class GarisomOptimizationConfig(OptimizationConfig):
    population: int = 1
    start_day: int = 201
    end_day: int = 236
