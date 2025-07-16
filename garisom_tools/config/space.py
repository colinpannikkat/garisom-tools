from optuna.distributions import BaseDistribution
from typing import Callable

from dataclasses import dataclass


@dataclass
class SampleSpace:
    distribution: BaseDistribution
    parameters: tuple[float]

    def unpack(self):
        return (self.distribution, self.parameters)


class SpaceConfig(dict[str, SampleSpace]):

    @classmethod
    def from_dict(cls, mapping: dict[str, Callable], data: dict):
        space_config = {}
        for k, v in data.items():
            dist_type = v[0].lower()
            params = v[1]
            if dist_type not in mapping:
                raise ValueError(f"Unknown distribution type: {dist_type}")
            space_config[k] = SampleSpace(
                distribution=mapping[dist_type],
                parameters=tuple(params)
            )
        return cls(space_config)

    def get_search_space(self):
        space = {}
        for param_name, samplespace in self.items():
            sampler, parameters = samplespace.unpack()
            space[param_name] = sampler(*parameters)

        return space
