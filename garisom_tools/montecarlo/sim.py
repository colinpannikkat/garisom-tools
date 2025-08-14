"""Monte Carlo simulation framework with quasi-random sampling.

This module provides the Sim class for running Monte Carlo simulations using
various quasi-random sampling engines including Sobol sequences, Latin
Hypercube sampling, and Halton sequences. It supports parallel execution
and provides statistical analysis of simulation results.

The module integrates with the Garisom modeling framework to enable
uncertainty quantification and sensitivity analysis through systematic
parameter space exploration.

Typical usage example:

```python
    from garisom_tools import Model
    from garisom_tools.montecarlo import MonteCarloConfig, Sim

    model = Model()
    config = MonteCarloConfig.from_json("mc_config.json")
    sim = Sim(model, config, run_kwargs={})
    results = sim.run(n=1024, parallel=True)
    stats = sim.analyze(results, index_columns=["time"])
```
"""

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
    """Monte Carlo simulation runner with configurable sampling engines.

    This class orchestrates Monte Carlo simulations by combining quasi-random
    sampling methods with model execution. It supports multiple sampling
    engines for exploring parameter space and provides parallel execution
    capabilities for efficient computation.

    The class generates parameter samples using quasi-random sequences,
    executes the model for each sample, and provides statistical analysis
    of the results including confidence intervals and summary statistics.

    Attributes:
        space (dict[str, Callable]): Dictionary mapping parameter names to
            their sampling distributions.
        outputs (list): List of output variables to collect from simulations.
        model (Model): The model instance to be executed for each sample.
        run_kwargs (dict): Arguments passed to the model during execution.
        seed (int): Random seed for reproducibility.
        rng (np.random.Generator): Random number generator instance.
        engine: Quasi-random sampling engine instance (Sobol, LatinHypercube,
            or Halton).
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
        """Initializes the Sim with model, configuration, and sampling settings.

        Args:
            model (Model): The model instance to run simulations on. Must
                implement run() and run_parallel() methods.
            config (MonteCarloConfig): Configuration object containing the
                parameter space definition and output specifications.
            run_kwargs (dict): Keyword arguments to pass to the model's run
                method. Should include any model-specific parameters.
            engine_kwargs (dict, optional): Additional arguments for the
                sampling engine initialization. Defaults to {}.
            engine (Literal['sobol', 'latin', 'halton'], optional): Type of
                quasi-random sampling engine to use. Defaults to 'sobol'.
            **kwargs: Additional keyword arguments including:
                seed (int): Random seed for reproducibility. Defaults to 42.
        """
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
        """Creates and returns the specified quasi-random sampling engine.

        Factory method that instantiates the appropriate sampling engine
        based on the engine type. Supports Sobol sequences, Latin Hypercube
        sampling, and Halton sequences.

        Args:
            engine (str): Type of sampling engine to create. Must be one of
                'sobol', 'latin', or 'halton'.
            **kwargs: Additional arguments passed to the engine constructor.
                Common arguments include 'd' (dimensionality) and 'rng'
                (random number generator).

        Returns:
            qmc.QMCEngine: An instance of the specified quasi-random sampling
                engine ready for generating samples.

        Raises:
            ValueError: If the specified engine type is not supported.
        """
        match engine:
            case 'sobol':
                return qmc.Sobol(**kwargs)
            case 'latin':
                return qmc.LatinHypercube(**kwargs)
            case 'halton':
                return qmc.Halton(**kwargs)

    def _sample_from_space(self, n: int, workers: int) -> list[dict[str, float]]:
        """Generates parameter samples from the defined search space.

        Creates n samples from the parameter space using the configured
        quasi-random sampling engine. For each parameter, uniform samples
        from [0,1) are generated and then transformed using the parameter's
        inverse cumulative distribution function (ppf).

        For Sobol sequences, the number of samples must be a power of 2 to
        ensure proper space-filling properties.

        Args:
            n (int): Number of samples to generate. For Sobol engine, must
                be a power of 2.
            workers (int): Number of worker threads/processes to use for
                parallel sampling (where supported by the engine).

        Returns:
            list[dict[str, float]]: A list of parameter dictionaries, each
                containing sampled values for all parameters in the space.

        Raises:
            AssertionError: If using Sobol engine and n is not a power of 2.
        """
        if not isinstance(self.engine, qmc.Sobol):
            samples = self.engine.random(n, workers=workers)  # (n, dim)
        else:
            assert np.log2(n) % 1 == 0  # Ensure n is a power of 2 for Sobol
            samples = self.engine.random_base2(m=int(np.log2(n)))  # (n, dim)

        # Transform uniform samples to parameter distributions using inverse CDF
        params = list(self.space.keys())
        param_values = {
            param: self.space[param].ppf(samples[:, i])
            for i, param in enumerate(params)
        }  # (dim, n)

        # Reorganize into list of parameter dictionaries for model execution
        samples = [
            {name: param_values[name][i] for name in params}
            for i in range(n)
        ]  # (n, dim)

        return samples

    def run(
        self,
        n: int = 1000,
        parallel: bool = True,
        workers: int = 4,
        X: dict[str, float] = None
    ) -> list[pd.DataFrame]:
        """Executes the Monte Carlo simulation.

        Generates parameter samples and runs the model for each sample,
        optionally in parallel. The method supports overriding default
        parameter values and returns the complete simulation results.

        Args:
            n (int, optional): Number of Monte Carlo samples to generate.
                For Sobol engine, should be a power of 2. Defaults to 1000.
            parallel (bool, optional): Whether to execute model runs in
                parallel for faster computation. Defaults to True.
            workers (int, optional): Number of parallel workers to use for
                both sampling and model execution. Defaults to 4.
            X (dict[str, float], optional): Default parameter values to
                apply across all samples. These will override sampled
                values for matching parameter names. Defaults to None.

        Returns:
            list[pd.DataFrame]: List of DataFrames containing model outputs
                for each parameter sample. Length equals the number of
                successful model runs (may be less than n if some fail).

        Example:
            >>> sim = Sim(model, config, run_kwargs={})
            >>> results = sim.run(n=512, parallel=True, workers=8)
            >>> print(f"Completed {len(results)} simulations")
        """

        samples = self._sample_from_space(n, workers)  # (n, dim)

        if X:  # Apply default parameter overrides if provided
            samples = [dict(X, **sample) for sample in samples]

        if parallel:
            results = self.model.run_parallel(X=samples, workers=workers, **self.run_kwargs)
        else:
            results = []
            for sample in samples:
                results.append(self.model.run(X=sample, **self.run_kwargs))

        return results

    def analyze(
        self,
        results: list[pd.DataFrame | None],
        index_columns: list[str]
    ) -> StatsResults:
        """Analyzes simulation results to compute summary statistics.

        Computes comprehensive statistical summaries including confidence
        intervals, means, standard deviations, and extrema for each output
        variable across all simulation runs. Index columns (such as timestamps
        or weather inputs) are preserved from the first successful run.

        Args:
            results (list[pd.DataFrame | None]): List of simulation result
                DataFrames. None values (from failed runs) are automatically
                excluded from analysis.
            index_columns (list[str]): Column names to treat as index variables
                that should be preserved unchanged from the first result.
                Typically includes time, weather inputs, or other non-stochastic
                variables.

        Returns:
            StatsResults: Object containing DataFrames with statistical
                summaries including:
                - ci_low, ci_high: 95% confidence interval bounds
                - mean: Sample means
                - stddev: Sample standard deviations
                - stderr: Standard errors of the means
                - min_val, max_val: Minimum and maximum values

        Note:
            The final sample size may be smaller than the original n if some
            model runs failed and returned None. Index columns are copied
            from the first successful result under the assumption that they
            are identical across all runs.

        Example:
            >>> stats = sim.analyze(results, index_columns=["time", "weather"])
            >>> mean_df = stats.mean
            >>> ci_low_df = stats.ci_low
        """
        # Filter out None results from failed model runs
        # Final N may be smaller than originally requested due to failures
        data = np.array([result for result in results if result is not None])  # N? x T x D

        stats = {}
        columns = results[0].columns  # Assume all results have same format

        # Extract array dimensions for statistics computation
        T = data.shape[1]  # Time steps
        D = data.shape[2]  # Variables

        # Convert to set for efficient membership testing
        index_columns = set(index_columns)

        stats = {
            "ci_low": np.ndarray((T, D)),
            "ci_high": np.ndarray((T, D)),
            "mean": np.ndarray((T, D)),
            "stddev": np.ndarray((T, D)),
            "stderr": np.ndarray((T, D)),
            "min_val": np.ndarray((T, D)),
            "max_val": np.ndarray((T, D)),
        }

        # Ensure data has consistent 3D shape for processing
        if data.ndim == 1:
            data = data[:, None, None]
        elif data.ndim == 2:
            data = data[:, :, None]

        for i, output in enumerate(results[0].columns):

            # Copy index column data from first result (assumed constant)
            if output in index_columns:

                for key in stats.keys():
                    stats[key][:, i] = data[0, :, i]

            else:
                # Compute statistics across simulation runs (axis=0)
                vals = data[:, :, i]

                stats["ci_low"][:, i] = np.quantile(vals, 0.025, axis=0)
                stats["ci_high"][:, i] = np.quantile(vals, 0.975, axis=0)
                stats["mean"][:, i] = np.mean(vals, axis=0)
                stats["stddev"][:, i] = np.std(vals, axis=0, ddof=1)  # Sample std dev
                stats["stderr"][:, i] = stats["stddev"][:, i] / np.sqrt(vals.shape[0])
                stats["min_val"][:, i] = np.min(vals, axis=0)
                stats["max_val"][:, i] = np.max(vals, axis=0)

        stats_res = {
            key: pd.DataFrame(val, columns=columns)
            for key, val in stats.items()
        }

        return StatsResults(stats_res)

    @staticmethod
    def plot_ci(stats: dict[str, list[dict]]):
        """Plots confidence intervals for simulation outputs over time.

        Creates time series plots showing the mean values and 95% confidence
        intervals for each output variable. Each plot displays the mean as
        a line with the confidence interval as a shaded region.

        Args:
            stats (dict[str, list[dict]]): Dictionary mapping output variable
                names to lists of statistical dictionaries. Each dictionary
                should contain "mean" and "95%_ci" keys with corresponding
                values for each time step.

        Note:
            This method assumes that the statistics are ordered by time and
            that each output has consistent time indexing. The plots are
            displayed using matplotlib.pyplot.show().

        Example:
            >>> stats_dict = {"biomass": [{"mean": 10, "95%_ci": [8, 12]}, ...]}
            >>> Sim.plot_ci(stats_dict)
        """

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
