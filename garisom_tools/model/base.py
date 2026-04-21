# Raytune
from ray import tune

# Basic data utils
import pandas as pd
from typing import Callable, Any
from abc import abstractmethod, ABC

# For model evaluation
from functools import partial

# Optimizer stuff
from garisom_tools.config import MetricConfig
from garisom_tools.utils.results import EvalResults


class Model(ABC):
    """
    Abstract base class for models.

    This class defines the interface that all models must implement,
    providing a standardized way to run simulations, evaluate results, and
    integrate with optimization frameworks like Ray Tune.

    Attributes:
        run_kwargs (dict): Keyword arguments passed to model execution methods.
        eval_kwargs (dict): Keyword arguments passed to model evaluation methods.

    Example:
        ```python
        class MyModel(Model):
            def __init__(self, run_kwargs=None, eval_kwargs=None):
                super().__init__(run_kwargs, eval_kwargs)

            @staticmethod
            def run(X=None, *args, **kwargs):
                # Implementation here
                pass

            # Implement other abstract methods...
        ```
    """

    def __init__(
            self,
            run_kwargs: dict[str, Any] = {},
            eval_kwargs: dict[str, Any] = {},
    ):
        """
        Initialize the Model instance.

        Args:
            run_kwargs (dict, optional): Keyword arguments for model execution.
                Defaults to empty dict.
            eval_kwargs (dict, optional): Keyword arguments for model evaluation.
                Defaults to empty dict.
        """
        self.run_kwargs = run_kwargs
        self.eval_kwargs = eval_kwargs

    @classmethod
    @abstractmethod
    def run_parallel(cls, X: list[dict[str, Any]] | None = None, *args, **kwargs) -> list[pd.DataFrame | None]:
        """
        Execute the model with multiple parameter sets in parallel.

        Args:
            X (list[dict[str, Any]], optional): List of parameter dictionaries.
                Each dictionary contains parameter names as keys and values as floats.
            *args: Variable length argument list passed to individual model runs.
            **kwargs: Arbitrary keyword arguments passed to individual model runs.

        Returns:
            list[pd.DataFrame | None]: List of model outputs, one per parameter set.
                None values indicate failed model runs.

        Note:
            This method should handle parallel execution internally and return
            results in the same order as the input parameter sets.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def run(cls, X: dict[str, Any] | None = None, *args, **kwargs) -> pd.DataFrame | None:
        """
        Execute the model with a single parameter set.

        Args:
            X (dict[str, Any], optional): Dictionary of parameter values.
                Keys are parameter names, values are numeric parameter values.
            *args: Variable length argument list for additional model inputs.
            **kwargs: Arbitrary keyword arguments for model configuration.

        Returns:
            pd.DataFrame | None: Model output as an array-like object (e.g., pandas DataFrame,
                numpy array). Returns None if the model run fails.

        Note:
            This method should handle a single model execution and return the
            complete time series or output data.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def launch_model(cls, *args, **kwargs) -> pd.DataFrame | None:
        """
        Low-level model execution method.

        This method handles the actual model subprocess execution or direct
        model computation. It should be called by the `run` method.

        Args:
            *args: Variable length argument list for model execution.
            **kwargs: Arbitrary keyword arguments for model configuration.

        Returns:
            pd.DataFrame | None: Raw model output. Returns None if execution fails.

        Note:
            This method typically handles file I/O, subprocess management,
            or direct model computation depending on the model implementation.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def evaluate_model(cls, *args, **kwargs) -> EvalResults:
        """
        Evaluate model output against ground truth data.

        This method computes various metrics comparing model predictions
        to observed data.

        Args:
            *args: Variable length argument list including model output and ground truth.
            **kwargs: Arbitrary keyword arguments for evaluation configuration.

        Returns:
            EvalResults: Dictionary mapping metric names to computed values.

        Note:
            The exact signature depends on the specific model implementation.
            Common arguments include model output, ground truth data, and
            evaluation period specifications.
        """
        raise NotImplementedError

    def get_objective(
        self
    ) -> Callable:
        """
        Create a partial function for model execution with predefined kwargs.

        This method creates a callable that can be used by optimization
        algorithms, with run_kwargs already bound to the model's run method.

        Returns:
            Callable: Partial function with run_kwargs applied to the run method.

        Example:
            ```python
            model = Model(run_kwargs={'config_file': 'config.csv'})
            objective = model.get_objective()
            result = objective(X={'param1': 0.5})  # config_file is automatically passed
            ```
        """
        return partial(
            self.run,
            **self.run_kwargs
        )

    def setup_model_and_return_callable(self, metric: MetricConfig) -> Callable:
        """
        Create a callable function for use with Ray Tune optimization.

        This method wraps the model execution and evaluation into a single
        callable that Ray Tune can use for hyperparameter optimization.

        Args:
            metric (MetricConfig): Metric configuration specifying which metrics
                to compute and their optimization modes (min/max).

        Returns:
            Callable: Function that takes a config dict and reports results to Ray Tune.

        Example:
            ```python
            from garisom_tools.config import MetricConfig
            from ray import tune

            # Setup metric configuration
            metric_config = MetricConfig.from_dict({
                'metrics': ['rmse', 'r2'],
                'modes': ['min', 'max'],
                'params': ['output_var', 'output_var']
            })

            # Create callable for Ray Tune
            model = Model(run_kwargs={...}, eval_kwargs={...})
            trainable = model.setup_model_and_return_callable(metric_config)

            # Use with Ray Tune
            tuner = tune.Tuner(trainable, param_space={...})
            results = tuner.fit()
            ```

        Note:
            The returned function automatically calls tune.report() with the
            computed metrics, making it compatible with Ray Tune's optimization
            framework.
        """

        objective = self.get_objective()

        def wrapped_model(config: dict) -> None:
            out = objective(X=config)
            errs = self.evaluate_model(
                out,
                metric_config=metric,
                **self.eval_kwargs,
            )
            tune.report(errs)

        return wrapped_model
