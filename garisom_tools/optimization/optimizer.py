"""Hyperparameter optimization using Ray Tune and Optuna.

This module provides the Optimizer class for hyperparameter tuning of machine
learning models using Ray Tune's distributed optimization framework combined
with Optuna's tree-structured Parzen estimator sampler. It supports
multi-objective optimization and returns the best results for each metric.

Typical usage example:

    from garisom_tools import Model
    from garisom_tools.optimization import OptimizationConfig, Optimizer

    model = Model()
    config = OptimizationConfig.from_json("config.json")
    optimizer = Optimizer(model, config)
    results = optimizer.run()
"""

# Config and model class
from garisom_tools.utils.results import ParamResults, MetricResult
from .config import OptimizationConfig
from garisom_tools import Model

# Using Optuna to allow for multiple objective optimization
from ray import tune
from ray.tune.search.optuna import OptunaSearch

import os
from optuna.samplers import TPESampler


class Optimizer():
    """Hyperparameter tuning optimizer using Ray Tune and Optuna.

    This class encapsulates the logic for setting up and running hyperparameter
    optimization experiments using Ray Tune's Tuner and Optuna's search
    algorithm. It supports multiple metrics and modes, and returns the best
    results for each metric.

    The optimizer uses Optuna's TPE (Tree-structured Parzen Estimator) sampler
    for efficient hyperparameter search and Ray Tune for distributed execution.

    Attributes:
        config (OptimizationConfig): Configuration object containing search
            space, metrics, and other optimization options.
        seed (int): Random seed for reproducibility. Defaults to 42.
        verbosity (int): Verbosity level for Ray Tune output.
        space (dict): Search space definition extracted from config.
        search (OptunaSearch): Optuna search algorithm instance.
        tuner (tune.Tuner): Ray Tune Tuner instance.
        results: Results object after running the tuner.
    """

    def __init__(self, model: Model, config: OptimizationConfig, verbosity: int = 1):
        """Initializes the Optimizer with model and configuration.

        Args:
            model (Model): The model to be optimized. Must implement
                `setup_model_and_return_callable` method.
            config (OptimizationConfig): Configuration for the optimization
                process including search space and metrics.
            verbosity (int, optional): Verbosity level for Ray Tune output.
                Defaults to 1.
        """
        self.config = config
        self.seed = 42
        self.verbosity = verbosity
        self.space = self.config.space.get_search_space()
        self.search = self._get_search_alg()
        self.tuner = self._get_tuner(model)

    def _get_search_alg(self):
        """Creates and configures the Optuna search algorithm.

        Initializes an OptunaSearch instance with the configured search space,
        metrics, and TPE sampler. The TPE sampler is configured with zero
        startup trials to ensure proper sampling from normal distributions.

        Returns:
            OptunaSearch: Configured Optuna search algorithm for hyperparameter
                optimization.
        """
        return OptunaSearch(
            space=self.space,
            metric=[metric.name for metric in self.config.metric.metrics],
            mode=self.config.metric.modes,
            sampler=TPESampler(
                n_startup_trials=0,  # Disable random sampling startup phase
                                     # to ensure proper normal distribution sampling
                seed=self.seed
            )
        )

    def _get_tuner(self, model: Model):
        """Creates and configures the Ray Tune Tuner.

        Sets up a Ray Tune Tuner instance with the model's callable function,
        search algorithm, and run configuration. The tuner is configured to
        save results in the current working directory.

        Args:
            model (Model): The model instance that provides the trainable
                function via `setup_model_and_return_callable`.

        Returns:
            tune.Tuner: Configured Ray Tune Tuner instance ready for
                hyperparameter search execution.
        """
        return tune.Tuner(
            model.setup_model_and_return_callable(self.config.metric),
            tune_config=tune.TuneConfig(
                search_alg=self.search,
                num_samples=self.config.num_samples
            ),
            run_config=tune.RunConfig(
                name="raytune_hyperparam_search",
                storage_path=os.getcwd(),
                verbose=self.verbosity
            )
        )

    def run(self) -> ParamResults:
        """Executes the hyperparameter optimization and returns results.

        Runs the Ray Tune hyperparameter search using the configured tuner
        and extracts the best results for each metric. For each metric, the
        method retrieves the best trial based on the metric's optimization
        mode (minimize or maximize) and compiles the scores and parameters.

        Returns:
            ParamResults: An object containing the best results for each metric.
                Each metric result includes the best scores across all metrics
                and the corresponding hyperparameters that achieved those scores.

        Example:
            >>> optimizer = Optimizer(model, config)
            >>> results = optimizer.run()
            >>> best_params = results.get_best_for_metric("accuracy")
        """
        self.results = self.tuner.fit()
        metric_and_modes = zip(
            self.config.metric.metrics,
            self.config.metric.modes
        )

        res = {}

        # Extract best results for each metric and compile scores and parameters
        for metric, mode in metric_and_modes:
            best_res = self.results.get_best_result(metric.name, mode)

            # Filter scores to include only the configured metrics
            scores = {
                name: score for name, score in best_res.metrics.items()
                if name in [m.name for m in self.config.metric.metrics]
            }
            parameters = best_res.config
            res[metric.name] = MetricResult(scores=scores, parameters=parameters)

        return ParamResults(res)
