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
    def __init__(self, model: Model, config: OptimizationConfig, verbosity: int = 1):
        self.config = config
        self.seed = 42
        self.verbosity = verbosity
        self.space = self.config.space.get_search_space()
        self.search = self._get_search_alg()
        self.tuner = self._get_tuner(model)

    def _get_search_alg(self):
        return OptunaSearch(
            space=self.space,
            metric=[metric.name for metric in self.config.metric.metrics],
            mode=self.config.metric.modes,
            sampler=TPESampler(
                n_startup_trials=0,  # Without this it uses randsampler for n trials,
                                     # which doesn't sample from normal distribution properly
                seed=self.seed
            )
        )

    def _get_tuner(self, model: Model):
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
        self.results = self.tuner.fit()
        metric_and_modes = zip(
            self.config.metric.metrics,
            self.config.metric.modes
        )

        res = {}

        # Get best results for each metric and save the corresponding scores
        # and parameters
        for metric, mode in metric_and_modes:
            best_res = self.results.get_best_result(metric.name, mode)
            scores = {
                name: score for name, score in best_res.metrics.items()
                if name in [m.name for m in self.config.metric.metrics]
            }
            parameters = best_res.config
            res[metric.name] = MetricResult(scores=scores, parameters=parameters)

        return ParamResults(res)
