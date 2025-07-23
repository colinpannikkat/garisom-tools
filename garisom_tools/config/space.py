"""
# Parameter Space Configuration

This module provides configuration classes for defining parameter sampling spaces
and probability distributions used in Monte Carlo simulations and optimization.

## Classes

- `SampleSpace`: Container for a distribution and its parameters
- `SpaceConfig`: Configuration for multiple parameter sampling spaces

## Example Usage

```python
from garisom_tools.config.space import SpaceConfig
from optuna.distributions import FloatDistribution

# Define distribution mapping
mapping = {
    'uniform': FloatDistribution,
    'normal': NormalDistribution
}

# Create space configuration
space_config = SpaceConfig.from_dict(mapping, {
    'growth_rate': ['uniform', [0.01, 0.1]],
    'water_efficiency': ['normal', [0.8, 0.1]]
})

# Get search space for optimization
search_space = space_config.get_search_space()
```
"""

from optuna.distributions import BaseDistribution
from typing import Callable

from dataclasses import dataclass


@dataclass
class SampleSpace:
    """
    Container for a probability distribution and its parameters.

    This class encapsulates a distribution type along with its initialization
    parameters, providing a convenient way to store and unpack distribution
    configurations.

    Attributes:
        distribution (BaseDistribution): The distribution class (not instance).
        parameters (tuple[float]): Parameters for initializing the distribution.

    Example:
        ```python
        from optuna.distributions import FloatDistribution

        # Create a sample space for uniform distribution
        space = SampleSpace(
            distribution=FloatDistribution,
            parameters=(0.0, 1.0)
        )

        # Unpack for use
        dist_class, params = space.unpack()
        distribution = dist_class(*params)  # FloatDistribution(0.0, 1.0)
        ```
    """
    distribution: BaseDistribution
    parameters: tuple[float]

    def unpack(self):
        """
        Unpack the distribution and parameters.

        Returns:
            tuple: (distribution_class, parameters_tuple) ready for instantiation.

        Example:
            ```python
            space = SampleSpace(FloatDistribution, (0.0, 1.0))
            dist_class, params = space.unpack()
            distribution = dist_class(*params)
            ```
        """
        return (self.distribution, self.parameters)


class SpaceConfig(dict[str, SampleSpace]):
    """
    Configuration for multiple parameter sampling spaces.

    This class extends dict to provide a convenient interface for managing
    multiple parameter distributions with their associated sampling spaces.
    It supports creation from configuration dictionaries and conversion to
    search spaces for optimization frameworks.

    Inherits from dict[str, SampleSpace] where:
    - Keys are parameter names
    - Values are SampleSpace instances

    Example:
        ```python
        from garisom_tools.config.space import SpaceConfig
        from optuna.distributions import FloatDistribution
        from garisom_tools.utils.distributions import NormalDistribution

        # Define mapping of distribution names to classes
        mapping = {
            'uniform': FloatDistribution,
            'normal': NormalDistribution
        }

        # Create from configuration dictionary
        config_data = {
            'growth_rate': ['uniform', [0.01, 0.1]],
            'water_efficiency': ['normal', [0.8, 0.1]],
            'root_depth': ['uniform', [0.5, 2.0]]
        }

        space_config = SpaceConfig.from_dict(mapping, config_data)

        # Use with optimization
        search_space = space_config.get_search_space()
        
        # Access individual spaces
        growth_space = space_config['growth_rate']
        print(f"Growth rate distribution: {growth_space.distribution}")
        ```
    """

    @classmethod
    def from_dict(cls, mapping: dict[str, Callable], data: dict):
        """
        Create a SpaceConfig from a distribution mapping and configuration data.

        Args:
            mapping (dict[str, Callable]): Dictionary mapping distribution names
                to distribution classes. Keys are string names (e.g., 'uniform'),
                values are distribution classes (e.g., FloatDistribution).
            data (dict): Configuration data where keys are parameter names and
                values are lists of [distribution_name, parameters].

        Returns:
            SpaceConfig: Configured instance with parameter spaces.

        Raises:
            ValueError: If a distribution type in data is not found in mapping.

        Example:
            ```python
            mapping = {
                'uniform': FloatDistribution,
                'normal': NormalDistribution,
                'truncnorm': TruncatedNormalDistribution
            }

            data = {
                'param1': ['uniform', [0.0, 1.0]],
                'param2': ['normal', [0.5, 0.1]],
                'param3': ['truncnorm', [0.3, 0.05, 0.0, 1.0]]
            }

            config = SpaceConfig.from_dict(mapping, data)
            ```

        Note:
            - Distribution names in data must match keys in mapping (case-insensitive)
            - Parameters must be appropriate for the chosen distribution type
            - The resulting config can be used with various optimization frameworks
        """
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
        """
        Convert the space configuration to a search space for optimization.

        This method creates distribution instances from the stored configuration,
        making them ready for use with optimization frameworks like Optuna or Ray Tune.

        Returns:
            dict[str, BaseDistribution]: Dictionary mapping parameter names to
                instantiated distribution objects.

        Example:
            ```python
            # Create space config
            config = SpaceConfig.from_dict(mapping, {
                'param1': ['uniform', [0.0, 1.0]],
                'param2': ['normal', [0.5, 0.1]]
            })

            # Get search space for optimizer
            search_space = config.get_search_space()

            # Use with Optuna
            import optuna
            study = optuna.create_study()
            study.optimize(objective, n_trials=100, search_space=search_space)
            ```

        Note:
            - Each call creates new distribution instances
            - The returned dictionary is suitable for use with optimization frameworks
            - Distribution instances are created using the stored parameters
        """
        space = {}
        for param_name, samplespace in self.items():
            sampler, parameters = samplespace.unpack()
            space[param_name] = sampler(*parameters)

        return space
