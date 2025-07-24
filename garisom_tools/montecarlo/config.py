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
    space: SpaceConfig = None
    outputs: list[str] = None
    num_worker: int = 4
    num_samples: int = 100

    @classmethod
    def from_json(cls, infile: str):
        with open(infile, "r") as f:
            data = json.load(f)

        mapping = {
            "normal": get_scipy_normal,
            "truncnorm": get_scipy_truncated_normal,
            "uniform": get_scipy_uniform
        }

        data['space'] = SpaceConfig.from_dict(mapping, data['space'])
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
class GarisomMonteCarloConfig(MonteCarloConfig):
    population: int = 1
    parameters: Optional[dict[str, float]] = None
