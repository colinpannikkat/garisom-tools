# Basic data utils
import pandas as pd
from typing import Any

# For model evaluation
import os
import subprocess
from tempfile import TemporaryDirectory

# Parallel runs
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import Model


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
        from datetime import datetime
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
                'start_date': datetime(2023, 7, 20),
                'end_day': datetime(2023, 8, 24)
            }
        )

        # Run with custom parameters
        result = model.run(X={'i_kmaxTree': 230, 'i_rootBeta': 0.8})

        # Evaluate against ground truth
        metrics = model.evaluate_model(
            result,
            ground_truth_data,
            metric_config,
            start_date=datetime(2023, 7, 20),
            end_date=datetime(2023, 8, 24)
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
                - 'start_date': int julian day to start evaluation period
                - 'end_date': int julian day to end evaluation period
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
