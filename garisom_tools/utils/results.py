"""
# Results Management

This module provides data structures for storing, managing, and serializing
simulation and optimization results from the garisom_tools package.

## Type Aliases

- `EvalResults`: Dictionary mapping metric names to computed values

## Classes

- `MetricResult`: Container for metric scores and parameter values
- `ParamResults`: Collection of MetricResult instances by parameter set
- `StatsResults`: Collection of statistical summaries from Monte Carlo simulations

## Example Usage

```python
from garisom_tools.utils.results import MetricResult, ParamResults, StatsResults
import pandas as pd

# Store optimization results
metric_result = MetricResult(
    scores={'rmse': 0.15, 'r2': 0.85},
    parameters={'growth_rate': 0.05, 'water_efficiency': 0.8}
)

# Collect multiple results
param_results = ParamResults({
    'best_rmse': metric_result,
    'best_r2': another_result
})

# Save results
param_results.to_json('optimization_results.json')

# Store simulation statistics
stats_results = StatsResults({
    'mean': mean_df,
    'ci_low': ci_low_df,
    'ci_high': ci_high_df
})
stats_results.save('/results/directory')
```
"""

from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import json
import os


EvalResults = dict[np.ndarray]
"""Type alias for evaluation results dictionary mapping metric names to values."""


@dataclass
class MetricResult:
    """
    Container for metric scores and corresponding parameter values.

    This class stores the results of a model evaluation, including both the
    computed metric scores and the parameter values that produced those scores.
    It's typically used to store optimization results or benchmark comparisons.

    Attributes:
        scores (dict[str, float]): Dictionary mapping metric names to computed values.
        parameters (dict[str, float]): Dictionary mapping parameter names to values.

    Example:
        ```python
        result = MetricResult(
            scores={
                'rmse': 0.15,
                'r2': 0.85,
                'nse': 0.82
            },
            parameters={
                'growth_rate': 0.05,
                'water_efficiency': 0.8,
                'root_depth': 1.5
            }
        )

        print(f"RMSE: {result.scores['rmse']}")
        print(f"Growth rate: {result.parameters['growth_rate']}")

        # Convert to dictionary for serialization
        result_dict = result.to_dict()
        ```
    """
    scores: dict[str, float]
    parameters: dict[str, float]

    def to_dict(self):
        """
        Convert the MetricResult to a dictionary.

        Returns:
            dict: Dictionary representation suitable for JSON serialization.
        """
        return asdict(self)


class ParamResults(dict[str, MetricResult]):
    """
    Collection of MetricResult instances organized by identifier.

    This class extends dict to provide specialized functionality for managing
    multiple optimization or evaluation results. Each key represents a unique
    identifier (e.g., 'best_rmse', 'trial_001') and each value is a MetricResult.

    Example:
        ```python
        results = ParamResults()

        # Add results for different optimization criteria
        results['best_rmse'] = MetricResult(
            scores={'rmse': 0.12, 'r2': 0.88},
            parameters={'param1': 0.5, 'param2': 1.2}
        )

        results['best_r2'] = MetricResult(
            scores={'rmse': 0.15, 'r2': 0.92},
            parameters={'param1': 0.6, 'param2': 1.0}
        )

        # Save all results
        results.to_json('optimization_results.json')

        # Access specific result
        best_rmse_params = results['best_rmse'].parameters
        ```
    """

    def to_dict(self):
        """
        Convert all MetricResult instances to dictionaries.

        Returns:
            dict: Nested dictionary with all results converted to dict format.
        """
        return {k: v.to_dict() for k, v in self.items()}

    def to_json(self, outfile: str):
        """
        Save the results to a JSON file.

        Args:
            outfile (str): Path to the output JSON file.

        Example:
            ```python
            results = ParamResults({
                'trial_1': MetricResult(...),
                'trial_2': MetricResult(...)
            })
            
            results.to_json('experiment_results.json')
            ```

        Note:
            Uses the "+x" mode to create a new file, will fail if file already exists.
        """
        with open(outfile, "+x") as f:
            json.dump(self.to_dict(), f)


class StatsResults(dict[str, pd.DataFrame]):
    """
    Collection of statistical summary DataFrames from Monte Carlo simulations.

    This class manages multiple statistical summaries (e.g., mean, confidence intervals)
    computed from Monte Carlo simulation results. Each key represents a statistic type
    and each value is a DataFrame with the computed values.

    Example:
        ```python
        import pandas as pd

        # Create statistical summaries
        stats = StatsResults({
            'mean': pd.DataFrame({...}),
            'ci_low': pd.DataFrame({...}),
            'ci_high': pd.DataFrame({...}),
            'min_val': pd.DataFrame({...}),
            'max_val': pd.DataFrame({...})
        })

        # Save all statistics to CSV files
        stats.save('/results/monte_carlo/')

        # Access specific statistic
        mean_values = stats['mean']
        confidence_interval = (stats['ci_low'], stats['ci_high'])
        ```
    """

    def save(self, directory: str):
        """
        Save all statistical DataFrames to CSV files in the specified directory.

        Each statistic is saved as a separate CSV file named after its key.
        For example, if the dictionary contains 'mean' and 'ci_low', the files
        'mean.csv' and 'ci_low.csv' will be created.

        Args:
            directory (str): Path to the directory where CSV files will be saved.
                The directory must already exist.

        Example:
            ```python
            stats = StatsResults({
                'mean': mean_df,
                'std': std_df,
                'ci_low': ci_low_df,
                'ci_high': ci_high_df
            })

            # Save to directory (creates mean.csv, std.csv, ci_low.csv, ci_high.csv)
            stats.save('/path/to/results/monte_carlo/')
            ```

        Note:
            - Files are saved without row indices (index=False)
            - Existing files with the same names will be overwritten
            - The directory must exist before calling this method
        """
        [
            data.to_csv(
                os.path.join(directory, f"{stat}.csv"),
                index=False
            ) for stat, data in self.items()
        ]