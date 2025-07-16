# Raytune
from ray import tune

# Basic data utils
import pandas as pd
import numpy as np
from typing import Callable, Any
from numpy.typing import ArrayLike
from abc import abstractmethod, ABC

# For model evaluation
import os
import subprocess
from tempfile import TemporaryDirectory
from functools import partial

# Optimizer stuff
from garisom_tools.config import MetricConfig
from garisom_tools.utils.results import EvalResults

# Parallel runs
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class Model(ABC):
    def __init__(
            self,
            run_kwargs: dict = None,
            eval_kwargs: dict = None,
    ):
        self.run_kwargs = run_kwargs
        self.eval_kwargs = eval_kwargs

    @staticmethod
    @abstractmethod
    def run_parallel(X: list[dict[str, Any]], *args, **kwargs) -> list[ArrayLike]:
        pass

    @staticmethod
    @abstractmethod
    def run(X: dict[str, Any], *args, **kwargs) -> ArrayLike | None:
        pass

    @staticmethod
    @abstractmethod
    def launch_model(*args, **kwargs) -> ArrayLike | None:
        pass

    @staticmethod
    @abstractmethod
    def evaluate_model(*args, **kwargs) -> EvalResults:
        pass

    def get_objective(
        self
    ) -> Callable:
        return partial(
            self.run,
            **self.run_kwargs
        )

    def setup_model_and_return_callable(self, metric: MetricConfig) -> Callable:
        """
        This function is used to supply RayTune with a callable that it can
        use to run the model, passing in the config with parameters.

        Inputs:
            - metric (MetricConfig): Metric config which is passed into
            evaluate_model, which is overidden in any child classes.

        Returns:
            Callable
        """

        objective = self.get_objective()

        def wrapped_model(config: dict) -> None:
            out = objective(config)
            errs = self.evaluate_model(
                out,
                metric_config=metric,
                **self.eval_kwargs,
            )
            tune.report(errs)

        return wrapped_model


class GarisomModel(Model):
    def __init__(
            self,
            run_kwargs: dict = None,
            eval_kwargs: dict = None
    ):
        super().__init__(run_kwargs=run_kwargs, eval_kwargs=eval_kwargs)

    @staticmethod
    def run_parallel(
        X: list[dict[str, float]],
        params: pd.DataFrame,
        config_file: str,
        population: int,
        model_dir: str,
        workers: int = 4,
        **kwargs
    ) -> list[pd.DataFrame | None]:

        N = len(X)
        res = [None for _ in range(N)]  # Ensure that we have an accessible index

        pbar = tqdm(total=N)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    GarisomModel.run,
                    X[i],
                    params,
                    config_file,
                    population,
                    model_dir,
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

    @staticmethod
    def run(
        X: dict[str, float],
        params: pd.DataFrame,
        config_file: str,
        population: int,
        model_dir: str,
        **kwargs
    ) -> pd.DataFrame | None:
        with TemporaryDirectory() as tmp:
            # Get unique TMP_DIR and make directory for specific process
            TMP_PARAM_FILE = f"{tmp}/params.csv"

            # Overwrite parameters with sample params
            for name in X.keys():
                params.at[population - 1, name] = X[name]

            # Setup parameter, configuration, and output files
            params.to_csv(TMP_PARAM_FILE, index=False)

            output = GarisomModel.launch_model(
                model_dir=model_dir,
                param_file=TMP_PARAM_FILE,
                config_file=config_file,
                population=population,
                save_location=tmp,
                **kwargs
            )

        return output

    @staticmethod
    def launch_model(
        model_dir: str,
        param_file: str,
        config_file: str,
        population: int,
        save_location: str,
        out: int = subprocess.DEVNULL,
        return_on_fail: bool = False
    ) -> pd.DataFrame | None:

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
            stdout=out,
            stderr=out
        )

        if p.returncode != 0 and not return_on_fail:
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

        out = pd.read_csv(output_file)

        return out

    @staticmethod
    def evaluate_model(
        output,
        ground,
        metric_config: MetricConfig,
        start_day: int,
        end_day: int
    ) -> EvalResults:

        out_names = [metric.output_name for metric in metric_config.metrics]
        pred = output[out_names].to_numpy(dtype=float) if output is not None else None

        metrics = metric_config.metrics
        modes = metric_config.modes

        errors = {}
        for idx, (metric, mode) in enumerate(zip(metrics, modes)):

            output_name = metric.output_name
            optim_name = metric.name
            eval_func = metric.func

            if pred is None:
                err = 1e20 if mode == 'min' else -1e20
            else:
                # Filter ground data based on julian-day and drop NaN values
                col_ground = ground[
                    ground['julian-day'].between(start_day, end_day)
                ][output_name].dropna()

                ground_values = np.array([col_ground.to_numpy()]).squeeze(axis=0)

                # Align predictions with the filtered ground data
                col_pred = pred[:, idx]  # (T)
                col_pred = pd.DataFrame(col_pred)
                pred_values = col_pred.loc[col_ground.index].T.to_numpy().squeeze(axis=0)

                err = eval_func(ground_values, pred_values)

            errors[optim_name] = err

        return errors
