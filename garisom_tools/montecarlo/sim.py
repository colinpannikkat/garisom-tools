# Model running
from garisom_tools import Model
from .config import MonteCarloConfig
from garisom_tools.utils.results import StatsResults

# Data
import pandas as pd
import numpy as np

# Distributions and Sampling
from scipy.stats import qmc

# Typing
from typing import Literal, Callable

# Plotting
import matplotlib.pyplot as plt


class Sim:
    """
    Sim class for running Monte Carlo simulations with configurable sampling engines and search spaces.

    Attributes:
        space (dict[str, Callable]): Dictionary mapping parameter names to their sampling distributions.
        model (Model): The model to be executed for each sample.
        seed (int): Random seed for reproducibility.
        rng (np.random.Generator): Random number generator instance.
        engine: Sampling engine instance (Sobol, LatinHypercube, or Halton).

        space_dict (dict): Dictionary defining the parameter search space.
        run_kwargs (dict): Arguments for initializing the Model.
        engine_kwargs (dict): Arguments for initializing the sampling engine.
        engine (Literal['sobol', 'latin', 'halton'], optional): Sampling engine type. Defaults to 'sobol'.
        **kwargs: Additional keyword arguments (e.g., 'seed').

    Methods:
        run(n: int = 1000, parallel: bool = True, workers: int = 4) -> list[pd.DataFrame]:
            Supports parallel execution.
    """
    def __init__(
        self,
        model: Model,
        config: MonteCarloConfig,
        run_kwargs: dict,
        engine_kwargs: dict = {},
        engine: Literal['sobol', 'latin', 'halton'] = 'sobol',
        **kwargs
    ):
        self.space: dict[str, Callable] = config.space.get_search_space()
        self.outputs: list = config.outputs
        self.model: Model = model
        self.run_kwargs: dict = run_kwargs

        # Ensure return_on_fail is passed in kwargs
        self.run_kwargs["return_on_fail"] = run_kwargs.get("return_on_fail", True)

        self.seed: int = kwargs.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self.engine = self._get_engine(
            engine,
            d=len(self.space),
            rng=self.rng,
            **engine_kwargs
        )

    @staticmethod
    def _get_engine(engine: str, **kwargs):
        match engine:
            case 'sobol':
                return qmc.Sobol(**kwargs)
            case 'latin':
                return qmc.LatinHypercube(**kwargs)
            case 'halton':
                return qmc.Halton(**kwargs)

    def _sample_from_space(self, n: int, workers: int) -> list[dict[str, float]]:
        """
        Generates a list of parameter samples from the defined parameter space using
        random sampling.

        For each parameter in the space, draws `n` random samples using the engine's
        random number generator, then applies the inverse cumulative distribution function
        (ppf) to map uniform samples to the parameter's distribution.

        Args:
            n (int): Number of samples to generate.
            workers (int): Number of worker threads/processes to use for sampling.

        Returns:
            list[dict[str, float]]: A list of dictionaries, each containing s
                ampled values for all parameters.
        """
        samples = self.engine.random(n, workers=workers)  # (n, dim)

        # For each parameter, compute inverse distribution from [0, 1) samples
        params = list(self.space.keys())
        param_values = {
            param: self.space[param].ppf(samples[:, i])
            for i, param in enumerate(params)
        }  # (dim, n)

        # Iterate through each parameter group to get samples from each parameter
        samples = [
            {name: param_values[name][i] for name in params}
            for i in range(n)
        ]  # (n, dim)

        return samples

    def run(self, n: int = 1000, parallel: bool = True, workers: int = 4) -> list[pd.DataFrame]:
        """
        Runs the simulation by generating samples and executing the model on each sample.

        Args:
            n (int, optional): Number of samples to generate. Defaults to 1000.
                parallel (bool, default=True): Whether to run the model in parallel.
            workers (int, default=4): Number of workers. Used for sampling, and
                also for parallel computation is parallel=True.

        Returns:
            list[pd.DataFrame]: Outputs for every parameter sample.
        """

        samples = self._sample_from_space(n, workers)  # (n, dim)

        if parallel:
            results = self.model.run_parallel(samples, workers=workers, **self.run_kwargs)
        else:
            results = []
            for sample in samples:
                results.append(self.model.run(sample, **self.run_kwargs))

        return results

    def analyze(
        self,
        results: list[pd.DataFrame],
        index_columns: list[str]
    ) -> StatsResults:
        """
        Analyze simulation results to compute min, max, average, median, and 95% confidence interval
        for each output across first dimension.

        Args:
            results (list[pd.DataFrame]): List of DataFrames from simulation runs.
            index_columns (list[str]): List of columns to treat as index columns (e.g., timestamps, weather inputs).

        Returns:
            dict: Dictionary with Dataframe for each statistic
        """
        # Put results into an ndarray for easy indexing
        data = np.array(results)  # N x T x D
        stats = {}
        columns = results[0].columns  # assume all results output same format

        # Get shapes
        T = data.shape[1]
        D = data.shape[2]

        # Make index_columns a set for easy lookup
        index_columns = set(index_columns)

        stats = {
            "ci_low": np.ndarray((T, D)),
            "ci_high": np.ndarray((T, D)),
            "mean": np.ndarray((T, D)),
            "min_val": np.ndarray((T, D)),
            "max_val": np.ndarray((T, D)),
        }

        # If data is 1D, reshape to (N, 1, 1) for consistency
        if data.ndim == 1:
            data = data[:, None, None]
        elif data.ndim == 2:
            data = data[:, :, None]

        for i, output in enumerate(results[0].columns):

            for t in range(T):

                # Add index_columns data, assume this is the same for each sample
                # These are things like timestamps, weather inputs, etc...
                if output in index_columns:

                    for key in stats.keys():
                        stats[key][t, i] = data[0, t, i]

                else:

                    vals = data[:, t, i]
                    vals = vals[~np.isnan(vals)]

                    stats["ci_low"][t, i] = np.quantile(vals, 0.025)
                    stats["ci_high"][t, i] = np.quantile(vals, 0.975)
                    stats["mean"][t, i] = np.mean(vals)
                    stats["min_val"][t, i] = np.min(vals)
                    stats["max_val"][t, i] = np.max(vals)

        stats_res = {
            key: pd.DataFrame(val, columns=columns)
            for key, val in stats.items()
        }

        return StatsResults(stats_res)

    @staticmethod
    def plot_ci(stats: dict[str, list[dict]]):

        for output, stat_list in stats.items():
            ci = [s["95%_ci"] for s in stat_list]
            mean = [s["mean"] for s in stat_list]
            t = range(len(stat_list))
            ci_low = [c[0] for c in ci]
            ci_high = [c[1] for c in ci]

            plt.figure()
            plt.plot(t, mean, label="Mean")
            plt.fill_between(t, ci_low, ci_high, color="lightblue", alpha=0.5, label="95% CI")
            plt.title(f"95% Confidence Interval for {output}")
            plt.xlabel("Time Step")
            plt.ylabel(output)
            plt.legend()
            plt.show()
