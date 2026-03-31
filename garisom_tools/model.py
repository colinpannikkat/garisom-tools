"""
# Model Interface and Sperry Interface Implementation

This module provides abstract base classes and concrete implementations for running
stomatal-optimization models, specifically the Sperry (Gain-Risk Stomatal Optimization Model from Sperry et al. 2017) 
model.

## Classes

- `Model`: Abstract base class defining the interface for all models
- `SperryModel`: Concrete interface implementation for the Sperry model

## Key Features

- **Parallel Execution**: Support for running multiple model instances concurrently
- **Flexible Configuration**: Customizable run and evaluation parameters
- **Model Evaluation**: Built-in metrics calculation and comparison with ground truth data
- **Ray Tune Integration**: Seamless integration with hyperparameter optimization
- **SALib Integration**: Easy sensitivity analysis and custom configs.
- **Monte Carlo Simulations**: Integrated Monte Carlo simulation for generating prediction
    uncertainty intervals.

## Example Usage

```python
from garisom_tools import SperryModel
from garisom_tools.config import MetricConfig
import pandas as pd
from datetime import datetime

# Load parameters and configuration
params = pd.read_csv("parameters.csv")
config_file = "model_config.csv"

# Create model instance
model = SperryModel(
    run_kwargs={
        'params': params,
        'config_file': config_file,
        'population': 1,
        'model_dir': '/path/to/model'
    },
    eval_kwargs={
        'ground': ground_truth_data,
        'start_date': datetime(2023, 7, 20),
        'end_date': datetime(2023, 8, 24)
    }
)

# Run single simulation
result = model.run(X={'param1': 0.5, 'param2': 1.2})

# Run parallel simulations
param_sets = [{'param1': 0.5, 'param2': 1.2}, {'param1': 0.7, 'param2': 1.0}]
results = model.run_parallel(param_sets, workers=4)
```
"""

# Raytune
from ray import tune

# Basic data utils
import pandas as pd
import numpy as np
from typing import Callable, Any
from abc import abstractmethod, ABC

# For model evaluation
import os
import subprocess
from tempfile import TemporaryDirectory
from functools import partial
from datetime import datetime

# Optimizer stuff
from garisom_tools.config import MetricConfig
from garisom_tools.utils.results import EvalResults

# Parallel runs
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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


class SperryModel(Model):
    """
    Concrete implementation of the Model interface for the Sperry model.

    The SperryModel class provides functionality to run the Sperry (Gain Risk Stomatal Optimization)
    model. Information on the model can be found in:

    - Sperry JS, Venturas MD, Anderegg WRL, Mencuccini M, Mackay DS, Wang Y, Love DM. 2017.
        Predicting stomatal responses to the environment from the optimization of photosynthetic gain and
        hydraulic cost. Plant, Cell & Environment 40: 816-830.
    - Venturas MD, Sperry JS, Love DM, Frehner EH, Allred MG, Wang Y, Anderegg WRL. 2018.
        A stomatal control model based on optimization of carbon gain versus hydraulic risk predicts aspen sapling
        responses to drought. New Phytologist 220: 836-850.

    This implementation:
    - Runs Sperry as a subprocess using parameter and configuration files
    - Supports parallel execution of multiple parameter sets
    - Handles temporary file management for model inputs/outputs
    - Provides comprehensive error handling and logging
    - Evaluates model outputs against ground truth observations

    Attributes:
    - run_kwargs (dict): Arguments for model execution
    - eval_kwargs (dict): Arguments for model evaluation

    Example:
        ```python
        import pandas as pd
        from garisom_tools import SperryModel

        # Load base parameters
        params = pd.read_csv("base_parameters.csv")

        # Create model instance
        model = SperryModel(
            run_kwargs={
                'params': params,
                'config_file': 'model_config.csv',
                'population': 1,
                'model_dir': '/path/to/Sperry/executable'
            },
            eval_kwargs={
                'ground': ground_truth_data,
                'start_day': 180,
                'end_day': 250
            }
        )

        # Run with custom parameters
        result = model.run(X={'i_kmaxTree': 230, 'i_rootBeta': 0.8})

        # Evaluate against ground truth
        metrics = model.evaluate_model(
            result,
            ground_truth_data,
            metric_config,
            start_day=180,
            end_day=250
        )
        ```
    """
    def __init__(
            self,
            run_kwargs: dict[str, Any] = {},
            eval_kwargs: dict[str, Any] = {}
    ):
        """
        Initialize the SperryModel instance.

        Args:
            run_kwargs (dict, optional): Keyword arguments for model execution.
                Expected keys include:
                - 'params': pandas.DataFrame with base parameter values
                - 'config_file': str path to model configuration file
                - 'population': int population index to use from parameters
                - 'model_dir': str path to directory containing Sperry executable
                - 'verbose': bool whether to print detailed output
                - 'return_on_fail': bool whether to return None on model failure
            eval_kwargs (dict, optional): Keyword arguments for model evaluation.
                Expected keys include:
                - 'ground': pandas.DataFrame with ground truth observations
                - 'start_day': int julian day to start evaluation period
                - 'end_day': int julian day to end evaluation period
        """
        super().__init__(run_kwargs=run_kwargs, eval_kwargs=eval_kwargs)

    @classmethod
    def run_parallel(
        cls,
        params: pd.DataFrame,
        config_file: str,
        population: int,
        model_dir: str,
        workers: int = 4,
        X: list[dict[str, float]] | None = None,
        **kwargs
    ) -> list[pd.DataFrame | None]:
        """
        Execute Sperry model runs in parallel for multiple parameter sets.

        This method uses ThreadPoolExecutor to run multiple Sperry instances
        concurrently, each with different parameter values. Progress is tracked
        with a progress bar, and failed runs are handled gracefully.

        Args:
            params (pd.DataFrame): Base parameter DataFrame containing all model parameters.
            config_file (str): Path to the model configuration file.
            population (int): Population index to use from the params DataFrame.
            model_dir (str): Path to directory containing the Sperry executable.
            workers (int, optional): Number of concurrent worker threads. Defaults to 4.
            X (list[dict[str, float]], optional): List of parameter dictionaries to override
                base parameters. Each dict contains parameter names as keys and values as floats.
            **kwargs: Additional keyword arguments passed to individual run() calls.

        Returns:
            list[pd.DataFrame | None]: List of model outputs, one per parameter set.
                Failed runs return None in the corresponding list position.

        Example:
            ```python
            import pandas as pd

            # Load base parameters
            params = pd.read_csv("parameters.csv")

            # Define parameter variations
            param_sets = [
                {'i_fieldCapFrac': 0.05, 'i_fieldCapPercInit': 0.8},
                {'i_fieldCapFrac': 0.06, 'i_fieldCapPercInit': 0.7},
                {'i_fieldCapFrac': 0.04, 'i_fieldCapPercInit': 0.9}
            ]

            # Run in parallel
            results = SperryModel.run_parallel(
                params=params,
                config_file='config.csv',
                population=1,
                model_dir='/path/to/Sperry',
                workers=8,
                X=param_sets,
                verbose=True
            )

            # Process results
            successful_runs = [r for r in results if r is not None]
            print(f"Successful runs: {len(successful_runs)}/{len(param_sets)}")
            ```

        Note:
            - Results are returned in the same order as input parameter sets
            - Failed runs are logged with their index and error message
            - Progress is displayed using tqdm progress bar
            - Each worker uses a temporary directory for file I/O
        """

        N = len(X) if X else 0
        res: list[pd.DataFrame | None] = [None for _ in range(N)]  # Ensure that we have an accessible index

        pbar = tqdm(total=N)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    cls.run,
                    params,
                    config_file,
                    population,
                    model_dir,
                    X=X[i] if X is not None else None,
                    **kwargs
                ):
                i for i in range(N)  # Store corresponding sample number
            }

            for future in as_completed(futures):
                pbar.update(1)
                idx = futures[future]
                try:
                    out = future.result()
                    res[idx] = out
                except Exception as e:
                    print(f"Subprocess for index {idx} failed: {e}")

        pbar.close()

        return res

    @classmethod
    def run(
        cls,
        params: pd.DataFrame,
        config_file: str,
        population: int,
        model_dir: str,
        X: dict[str, float] | None = None,
        **kwargs
    ) -> pd.DataFrame | None:
        """
        Execute a single Sperry model run with specified parameters.

        This method creates a temporary directory, modifies the parameter file
        with custom values (if provided), runs the Sperry model, and returns
        the output data.

        Args:
            params (pd.DataFrame): Base parameter DataFrame containing all model parameters.
                Must have columns matching Sperry parameter names.
            config_file (str): Path to the model configuration file that specifies
                model settings, input/output options, and simulation period.
            population (int): Population index (1-based) to use from the params DataFrame.
                This determines which row of parameters to use as the base.
            model_dir (str): Path to directory containing the Sperry executable (./run).
            X (dict[str, float], optional): Dictionary of parameter overrides.
                Keys must match column names in the params DataFrame.
            **kwargs: Additional keyword arguments passed to launch_model().
                Common options include:
                - verbose (bool): Enable detailed output
                - return_on_fail (bool): Attempts to get output files on failure

        Returns:
            pd.DataFrame | None: Model output containing timestep data with columns
                for state variables, fluxes, and environmental conditions.
                Returns None if the model run fails.

        Raises:
            FileNotFoundError: If expected output file is not created by the model.

        Example:
            ```python
            import pandas as pd

            # Load base parameters
            params = pd.read_csv("parameters.csv")

            # Run with custom parameters
            result = SperryModel.run(
                params=params,
                config_file='model_config.csv',
                population=1,
                model_dir='/path/to/Sperry',
                X={'i_fieldCapPercInit': 0.05, 'i_fieldCapFrac': 0.8},
                verbose=True
            )

            if result is not None:
                print(f"Model completed with {len(result)} timesteps")
                print(f"Output columns: {result.columns.tolist()}")
            else:
                print("Model run failed")
            ```

        Note:
            - Uses temporary directories to avoid file conflicts in parallel runs
            - Automatically determines output filename based on species/region/site IDs
            - Preserves original parameters DataFrame (modifications are local)
        """

        # Make a deep copy to preserve the original DataFrame
        params = params.copy(deep=True)

        with TemporaryDirectory() as tmp:
            TMP_PARAM_FILE = f"{tmp}/params.csv"

            # Overwrite parameters with sample params if X is provided
            if X is not None:
                for name in X.keys():
                    params.at[population - 1, name] = X[name]

            params.to_csv(TMP_PARAM_FILE, index=False)

            output = cls.launch_model(
                model_dir=model_dir,
                param_file=TMP_PARAM_FILE,
                config_file=config_file,
                population=population,
                save_location=tmp,
                **kwargs
            )

        return output

    @classmethod
    def launch_model(
        cls,
        model_dir: str,
        param_file: str,
        config_file: str,
        population: int,
        save_location: str,
        out: int = subprocess.DEVNULL,
        err: int = subprocess.DEVNULL,
        return_on_fail: bool = False,
        verbose: bool = False
    ) -> pd.DataFrame | None:
        """
        Launch the Sperry model executable and process its output.

        This method handles the low-level execution of the Sperry model as a
        subprocess, manages file I/O, and parses the resulting output files.

        Args:
            model_dir (str): Path to directory containing the Sperry executable (./run).
            param_file (str): Path to CSV file containing model parameters.
            config_file (str): Path to model configuration file.
            population (int): Population index (1-based) for parameter selection.
            save_location (str): Directory where model outputs will be saved.
            out (int, optional): File descriptor for stdout redirection.
                Defaults to subprocess.DEVNULL to suppress output.
            err (int, optional): File descriptor for stderr redirection.
                Defaults to subprocess.DEVNULL to suppress errors.
            return_on_fail (bool, optional): If True, return None on model failure
                instead of raising an exception. Defaults to False.
            verbose (bool, optional): If True, print status messages during execution.
                Defaults to False.

        Returns:
            pd.DataFrame | None: Model output data with timestep results.
                Returns None if model fails and return_on_fail=True.

        Raises:
            FileNotFoundError: If expected output file is not created by the model.
            subprocess.CalledProcessError: If model executable returns non-zero exit code
                and return_on_fail=False.

        Example:
            ```python
            import subprocess
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmpdir:
                # Create parameter file
                params.to_csv(f"{tmpdir}/params.csv", index=False)

                # Launch model
                result = SperryModel.launch_model(
                    model_dir='/path/to/Sperry',
                    param_file=f"{tmpdir}/params.csv",
                    config_file='config.csv',
                    population=1,
                    save_location=tmpdir,
                    out=subprocess.PIPE,  # Capture output
                    verbose=True
                )
            ```

        Note:
            - The Sperry executable must be named './run' in model_dir
            - Output filename is determined by species, region, and site IDs from parameters
            - Uses subprocess.run() for robust process management
            - Automatically detects and loads the correct output file
        """

        params = pd.read_csv(param_file)

        p = subprocess.run(
            [
                "./run",
                param_file,
                config_file,
                str(population),
                save_location
            ],
            cwd=model_dir,
            stdout=out if not verbose else None,
            stderr=err if not verbose else None
        )

        if p.returncode != 0:
            if verbose:
                print("Model failed with returncode: ", p.returncode)
            if not return_on_fail:
                return None

        # Get species, region, and site to determine output file
        species = params.at[population - 1, 'i_sp']
        region = params.at[population - 1, 'i_region']
        site = params.at[population - 1, 'i_site']

        output_file = os.path.join(
            save_location, f"timesteps_output_{species}_{region}_{site}.csv"
        )
        if not os.path.exists(output_file):
            raise FileNotFoundError(
                f"Expected output file not found: {output_file}"
            )

        out_file = pd.read_csv(output_file)

        return out_file

    @classmethod
    def evaluate_model(
        cls,
        output,
        ground,
        metric_config: MetricConfig,
        start_date: datetime,
        end_date: datetime
    ) -> EvalResults:
        """
        Evaluate model predictions against ground truth observations.

        This method computes multiple evaluation metrics by comparing model output
        to observed data over a specified time period. It handles data alignment,
        missing values, and multiple output variables.

        Args:
            output (pd.DataFrame | None): Model output DataFrame with timestep data.
                Must contain columns matching metric_config output names.
                If None, returns penalty values for all metrics.
            ground (pd.DataFrame): Ground truth observations DataFrame.
                Must contain 'julian-day' column and columns matching output variables.
            metric_config (MetricConfig): Configuration specifying which metrics to compute.
                Contains lists of metrics, their target output variables, and optimization modes.
            start_day (int): Julian day to start evaluation period (inclusive).
            end_day (int): Julian day to end evaluation period (inclusive).

        Returns:
            EvalResults: Dictionary mapping metric names to computed values.
                For minimization metrics, higher values indicate worse performance.
                For maximization metrics, higher values indicate better performance.

        Example:
            ```python
            from garisom_tools.config import MetricConfig

            # Setup metric configuration
            metric_config = MetricConfig.from_dict({
                'metrics': ['rmse', 'r2', 'nse'],
                'modes': ['min', 'max', 'max'],
                'params': ['leaftemp', 'leaftemp', 'E-MD']
            })

            # Load ground truth data
            ground_truth = pd.read_csv("observations.csv")  # Must have 'julian-day' column

            # Evaluate model output
            model_output = SperryModel.run(...)
            evaluation = SperryModel.evaluate_model(
                output=model_output,
                ground=ground_truth,
                metric_config=metric_config,
                start_date=datetime(2023, 07, 23)
                end_date=243
            )

            print(f"RMSE: {evaluation['rmse']:.3f}")
            print(f"R²: {evaluation['r2']:.3f}")
            print(f"NSE: {evaluation['nse']:.3f}")
            ```

        Note:
            - Data is filtered by date before metric calculation
            - Missing values (NaN) are automatically excluded from ground truth
            - Model predictions are aligned with ground truth observations by index
            - Failed model runs (output=None) receive penalty values (1e20 for min, -1e20 for max)
            - Supports multiple metrics on the same output variable using suffix notation (e.g., 'var.A', 'var.B')
        """
        out_names = [metric.output_name for metric in metric_config.metrics]
        pred = output[out_names].to_numpy(dtype=float) if output is not None else None

        metrics = metric_config.metrics
        modes = metric_config.modes

        errors: dict[str, np.typing.ArrayLike] = {}
        for idx, (metric, mode) in enumerate(zip(metrics, modes)):

            output_name = metric.output_name
            optim_name = metric.name
            eval_func = metric.func

            if pred is None or eval_func is None:
                err = 1e20 if mode == 'min' else -1e20 if mode == "max" else 0
            else:
                # Filter ground data based on year and julian-day, then drop NaN values
                start_year, start_day = start_date.year, start_date.timetuple().tm_yday
                end_year, end_day = end_date.year, end_date.timetuple().tm_yday

                if start_year == end_year:
                    mask = (
                        (ground['year'] == start_year) &
                        (ground['julian-day'] >= start_day) &
                        (ground['julian-day'] <= end_day)
                    )
                else:
                    mask = (
                        ((ground['year'] == start_year) & (ground['julian-day'] >= start_day)) |
                        ((ground['year'] > start_year) & (ground['year'] < end_year)) |
                        ((ground['year'] == end_year) & (ground['julian-day'] <= end_day))
                    )
                col_ground = ground[mask][output_name].dropna()

                ground_values = np.array([col_ground.to_numpy()]).squeeze(axis=0)

                # Align predictions with the filtered ground data
                col_pred = pred[:, idx]  # (T)
                col_pred = pd.DataFrame(col_pred)
                pred_values = col_pred.loc[col_ground.index].T.to_numpy().squeeze(axis=0)

                err = eval_func(ground_values, pred_values)

            errors[optim_name] = err

        return errors
