"""Sensitivity analysis implementation using Sobol indices.

This module provides a comprehensive implementation of global sensitivity
analysis using Sobol indices via the SALib library. It supports first-order
and total-order sensitivity index computation for model parameters across
time series outputs and error metrics.

The implementation focuses on Sobol sequence sampling for efficient parameter
space exploration and variance-based sensitivity analysis. It includes
automated plotting capabilities for visualization of sensitivity indices
and parameter-output relationships.

Features:
    - Sobol sequence sampling for quasi-random parameter exploration
    - First-order and total-order sensitivity index computation
    - Time series sensitivity analysis with temporal dynamics
    - Error metric sensitivity analysis
    - Automated visualization and result saving
    - Parallel model execution support

Limitations:
    - Second-order indices are not currently supported
    - Analysis is limited to variance-based methods (Sobol indices)

References:
    - Saltelli, A., et al. (2008). Global Sensitivity Analysis: The Primer
    - Sobol, I.M. (2001). Global sensitivity indices for nonlinear mathematical models

Typical usage example:

    from garisom_tools.sa import SensitivityAnalysis, SensitivityAnalysisConfig
    from garisom_tools import Model

    model = Model()
    config = SensitivityAnalysisConfig.from_json("sa_config.json")
    sa = SensitivityAnalysis(model, config)
    sa.run("output_directory")
"""

# Model and config
from ..model import Model
from .config import SensitivityAnalysisConfig, SensitivityAnalysisProblem

# SALib
from SALib.sample import sobol as ssobol
from SALib.analyze import sobol as asobol

# Plotting
import matplotlib.pyplot as plt

# Logging
import logging

# Data and saving
import numpy as np
import os
import json


class SensitivityAnalysis:
    """Global sensitivity analysis using Sobol indices.

    This class implements comprehensive sensitivity analysis using Sobol
    sequence sampling and variance-based sensitivity indices. It supports
    analysis of model parameters across time series outputs and provides
    automated visualization and result saving capabilities.

    The analysis computes first-order and total-order Sobol indices for
    each parameter-output combination, revealing both individual parameter
    importance and interaction effects. Results include temporal sensitivity
    dynamics and error metric sensitivities.

    Attributes:
        model (Model): The model instance to analyze. Must implement run_parallel
            and evaluate_model methods for parameter sampling and evaluation.
        config (SensitivityAnalysisConfig): Configuration object containing
            problem definition, metrics, and execution parameters.

    Example:
        ```python
        from garisom_tools import Model
        from garisom_tools.sa import SensitivityAnalysis, SensitivityAnalysisConfig

        model = Model()
        config = SensitivityAnalysisConfig.from_json("config.json")
        sa = SensitivityAnalysis(model, config)
        sa.run("results/")
        ```
    """

    def __init__(
        self,
        model: Model,
        config: SensitivityAnalysisConfig,
    ):
        """Initialize the SensitivityAnalysis with model and configuration.

        Args:
            model (Model): The model instance to perform sensitivity analysis on.
                Must implement run_parallel() for batch execution and
                evaluate_model() for metric computation.
            config (SensitivityAnalysisConfig): Configuration containing the
                problem definition, metrics, and execution parameters for
                the sensitivity analysis.
        """
        self.model = model
        self.config = config

    def _get_samples(
        self,
        problem: SensitivityAnalysisProblem,
        n_samples: int,
        calc_second_order: bool = False,
        res_dir: str = "."
    ):
        """Generate Sobol sequence samples for sensitivity analysis.

        Creates parameter samples using Sobol sequences for efficient coverage
        of the parameter space. The samples are designed for variance-based
        sensitivity analysis and are saved to disk for reproducibility.

        Args:
            problem (SensitivityAnalysisProblem): Problem definition containing
                parameter names, bounds, and dimensionality information.
            n_samples (int): Number of base samples to generate. The actual
                number of samples will be larger due to Sobol sequence requirements.
            calc_second_order (bool, optional): Whether to generate samples for
                second-order indices. Currently not implemented. Defaults to False.
            res_dir (str, optional): Directory to save the generated samples.
                Defaults to current directory.

        Returns:
            np.ndarray: Array of parameter value dictionaries with shape (N,)
                where each element is a dict mapping parameter names to values.

        Raises:
            NotImplementedError: If calc_second_order is True, as second-order
                indices are not currently supported.

        Note:
            The method automatically saves the generated samples to
            "{res_dir}/sample.npy" for reproducibility and debugging purposes.
        """
        logging.info("Retrieving Sobol samples.")
        if calc_second_order:
            raise NotImplementedError("calc_second_order is not implemented yet.")

        # Generate Sobol sequence samples for sensitivity analysis
        samples = ssobol.sample(
            problem.to_dict(),
            N=n_samples,
            calc_second_order=calc_second_order
        )   # shape (N, D)

        # Convert samples to parameter dictionaries for model execution
        param_values = [
            {
                param: val
                for param, val in zip(problem.names, sample)
            }
            for sample in samples
        ]

        # Save samples for reproducibility and debugging
        np.save(f"{res_dir}/sample.npy", param_values)

        return param_values

    def _get_errors(
        self,
        output
    ):
        """Compute error metrics for each model output sample.

        Evaluates the model outputs against configured metrics to compute
        error values for sensitivity analysis. The errors are organized
        by metric name for subsequent sensitivity index computation.

        Args:
            output: List or array of model outputs from parameter samples.
                Each element should be compatible with the model's evaluate_model
                method.

        Returns:
            dict: Dictionary mapping metric names to lists of error values.
                Each list contains error values for all samples, maintaining
                the same order as the input outputs.

        Note:
            This method uses the model's evaluate_model method with the configured
            metric settings to compute errors for each sample consistently.
        """

        # Initialize error storage for each configured metric
        errors = {
            param.name: []
            for param in self.config.metric.metrics
        }

        # Compute error metrics for each model output sample
        for out in output:
            err = self.model.evaluate_model(
                out,
                metric_config=self.config.metric,
                **self.model.eval_kwargs
            )

            # Collect error values maintaining sample order
            for k in err.keys():
                errors[k].append(err[k])

        return errors

    def _analyze(
        self,
        output: np.ndarray,  # shape: (N, T, Y_D)
        out_names: list[str],
        errors: dict[list],
        problem: SensitivityAnalysisProblem,
        res_dir: str = "."
    ):
        """Compute Sobol sensitivity indices for outputs and error metrics.

        Performs comprehensive sensitivity analysis by computing first-order
        and total-order Sobol indices for both model outputs and error metrics.
        Results are computed across time steps and saved to disk.

        Args:
            output (np.ndarray): Model outputs with shape (N, T, Y_D) where
                N is the number of samples, T is time steps, and Y_D is the
                number of output variables.
            out_names (list[str]): Names of output variables corresponding
                to the last dimension of the output array.
            errors (dict[list]): Dictionary mapping metric names to lists
                of error values for each sample.
            problem (SensitivityAnalysisProblem): Problem definition containing
                parameter information for sensitivity analysis.
            res_dir (str, optional): Directory to save analysis results.
                Defaults to current directory.

        Returns:
            tuple: A tuple containing:
                - first_order_indices (np.ndarray): First-order indices with
                  shape (Y_D, D, T)
                - total_order_indices (np.ndarray): Total-order indices with
                  shape (Y_D, D, T)
                - error_first_order_indices (np.ndarray): Error first-order
                  indices with shape (Y_D, D)
                - error_total_order_indices (np.ndarray): Error total-order
                  indices with shape (Y_D, D)

        Note:
            Sensitivity indices are clipped to [0, 1] range and saved as
            .npy files for further analysis. The method processes both
            time-varying outputs and aggregated error metrics.
        """

        T = output.shape[1]
        D = problem.num_vars
        Y_D = len(out_names)

        first_order_indices = np.empty((Y_D, D, T))
        first_order_conf = np.empty((Y_D, D, T))
        total_order_indices = np.empty((Y_D, D, T))
        total_order_conf = np.empty((Y_D, D, T))

        error_first_order_indices = np.empty((Y_D, D))
        error_first_order_conf = np.empty((Y_D, D))
        error_total_order_indices = np.empty((Y_D, D))
        error_total_order_conf = np.empty((Y_D, D))

        logging.info("Analyzing output dimension indices.")
        for t in range(T):
            for i, _ in enumerate(out_names):
                si = asobol.analyze(
                    problem.to_dict(),
                    output[:, t, i],
                    print_to_console=False,
                    calc_second_order=False,
                )
                first_order_indices[i, :, t] = np.clip(si['S1'], 0, 1)
                first_order_conf[i, :, t] = si['S1_conf']
                total_order_indices[i, :, t] = np.clip(si['ST'], 0, 1)
                total_order_conf[i, :, t] = si['ST_conf']

        np.save(f"{res_dir}/first_order_indices.npy", first_order_indices)
        np.save(f"{res_dir}/first_order_conf.npy", first_order_conf)
        np.save(f"{res_dir}/total_order_indices.npy", total_order_indices)
        np.save(f"{res_dir}/total_order_conf.npy", total_order_conf)

        logging.info("Analyzing error indices.")
        np_errors = np.array([errors[out.name] for out in self.config.metric.metrics])

        for i, _ in enumerate(out_names):
            si = asobol.analyze(
                problem.to_dict(),
                np_errors[i, :],
                print_to_console=False,
                calc_second_order=False,
            )
            error_first_order_indices[i, :] = np.clip(si['S1'], 0, 1)
            error_first_order_conf[i, :] = si['S1_conf']
            error_total_order_indices[i, :] = np.clip(si['ST'], 0, 1)
            error_total_order_conf[i, :] = si['ST_conf']

        np.save(f"{res_dir}/error_first_order_indices.npy",
                error_first_order_indices)
        np.save(f"{res_dir}/error_first_order_conf.npy",
                error_first_order_conf)
        np.save(f"{res_dir}/error_total_order_indices.npy",
                error_total_order_indices)
        np.save(f"{res_dir}/error_total_order_conf.npy",
                error_total_order_conf)

        return first_order_indices, total_order_indices, error_first_order_indices, error_total_order_indices

    def run(self, out_dir: str):
        """Execute the complete sensitivity analysis workflow.

        Runs the entire sensitivity analysis process including sample generation,
        model execution, error computation, sensitivity index calculation,
        and result visualization. Results are saved to organized subdirectories.

        Args:
            out_dir (str): Output directory where all results will be saved.
                The method creates 'plots' and 'sa_results' subdirectories
                for organizing visualization and numerical results.

        Returns:
            None

        Note:
            This method orchestrates the complete workflow:
            1. Generate Sobol sequence parameter samples
            2. Execute model runs in parallel for all samples
            3. Compute error metrics for each sample
            4. Calculate sensitivity indices for outputs and errors
            5. Create comprehensive visualization plots
            6. Save all results and intermediate data to disk

        The method creates the following directory structure:
        - {out_dir}/plots/: Visualization files (PNG format)
        - {out_dir}/sa_results/: Numerical results (NPY and JSON files)

        Example:
            ```python
            sa = SensitivityAnalysis(model, config)
            sa.run("sensitivity_analysis_results")
            ```
        """

        plt_dir = os.path.join(out_dir, "plots")
        res_dir = os.path.join(out_dir, "sa_results")

        logging.info(f"Plots will be saved in: {plt_dir}")
        logging.info(f"Results will be saved in: {res_dir}")

        os.makedirs(plt_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        param_samples = self._get_samples(  # shape : (N, D)
            self.config.problem,
            self.config.samples,
            res_dir=res_dir
        )

        logging.info("Running model with samples.")
        outputs = self.model.run_parallel(
            X=param_samples,
            workers=self.config.workers,
            **self.model.run_kwargs
        )  # (N, T, Y_D)
        out_names = list({metric.output_name for metric in self.config.metric.metrics})  # unique
        outputs = [output[out_names] for output in outputs]

        # Get errors for every sample
        errors = self._get_errors(outputs)

        # Save errors
        with open(os.path.join(res_dir, "errors.json"), "w") as f:
            json.dump(errors, f, indent=4)

        # Convert to numpy for easy slicing
        outputs = np.array(outputs)  # (N, T, Y_D)
        outputs = np.nan_to_num(outputs)

        np.save(f"{res_dir}/model_output.npy", outputs)

        first, total, _, _ = self._analyze(outputs, out_names, errors, self.config.problem, res_dir)

        self.plot(
            self.config.problem,
            out_names,
            first,
            total,
            outputs,
            errors,
            param_samples,
            plt_dir
        )

    def plot(
        self,
        problem: SensitivityAnalysisProblem,
        out_names,
        first_order_indices,  # shape: (Y_D, D, T)
        total_order_indices,  # shape: (Y_D, D, T)
        outputs,              # shape: (N, T, Y_D)
        errors,
        param_values,
        plt_dir
    ):
        """Create comprehensive visualization plots for sensitivity analysis results.

        Generates multiple types of plots to visualize sensitivity analysis
        results including temporal sensitivity indices, parameter-output
        relationships, and error sensitivities. All plots are saved as PNG files.

        Args:
            problem (SensitivityAnalysisProblem): Problem definition containing
                parameter names and bounds information.
            out_names (list): List of output variable names for labeling plots.
            first_order_indices (np.ndarray): First-order Sobol indices with
                shape (Y_D, D, T) where Y_D is outputs, D is parameters, T is time.
            total_order_indices (np.ndarray): Total-order Sobol indices with
                shape (Y_D, D, T).
            outputs (np.ndarray): Model outputs with shape (N, T, Y_D) where
                N is the number of samples.
            errors (dict): Dictionary mapping metric names to error value lists.
            param_values (np.ndarray): Parameter sample values for scatter plots.
            plt_dir (str): Directory path where plot files will be saved.

        Returns:
            None

        Note:
            Creates the following plot types:
            - First-order sensitivity indices over time (grouped by pairs)
            - Total-order sensitivity indices over time (grouped by pairs)
            - Parameter vs mean output scatter plots
            - Parameter vs error metric scatter plots

            All plots include proper legends, grid lines, and descriptive titles.
            Files are saved with descriptive names indicating content and variables.
        """

        logging.info("Creating plots.")

        T = outputs.shape[1]
        time = np.arange(T)

        # Plot First-Order Sobol Indices
        for idx in range(0, len(out_names), 2):
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            for i, name in enumerate(problem.names):
                axes[0].scatter(time,
                                first_order_indices[idx, i, :],
                                label=f'{name}')
                if idx + 1 < len(out_names):
                    axes[1].scatter(time,
                                    first_order_indices[idx + 1, i, :],
                                    label=f'{name}')

            axes[0].set_title(
                f'First-Order Sobol Indices for {out_names[idx]}'
            )
            if idx + 1 < len(out_names):
                axes[1].set_title(
                    f'First-Order Sobol Indices for {out_names[idx + 1]}'
                )

            for ax in axes:
                ax.set_ylabel('Sobol Index')
                ax.legend()
                ax.grid(True)

            axes[1].set_xlabel('Time')
            plt.tight_layout()

            if idx + 1 < len(out_names):
                plt.savefig(
                    f"{plt_dir}/first-order_{out_names[idx]}_"
                    f"{out_names[idx + 1]}.png"
                )
            else:
                plt.savefig(f"{plt_dir}/first-order_{out_names[idx]}.png")
            plt.close(fig)

        # Plot Total-Order Sobol Indices
        for idx in range(0, len(out_names), 2):
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            for i, name in enumerate(problem.names):
                axes[0].scatter(time,
                                total_order_indices[idx, i, :],
                                label=f'{name}')
                if idx + 1 < len(out_names):
                    axes[1].scatter(time,
                                    total_order_indices[idx + 1, i, :],
                                    label=f'{name}')

            axes[0].set_title(
                f'Total-Order Sobol Indices for {out_names[idx]}'
            )
            if idx + 1 < len(out_names):
                axes[1].set_title(
                    f'Total-Order Sobol Indices for {out_names[idx + 1]}'
                )

            for ax in axes:
                ax.set_ylabel('Sobol Index')
                ax.legend()
                ax.grid(True)

            axes[1].set_xlabel('Time')
            plt.tight_layout()

            if idx + 1 < len(out_names):
                plt.savefig(
                    f"{plt_dir}/total-order_{out_names[idx]}_"
                    f"{out_names[idx + 1]}.png"
                )
            else:
                plt.savefig(f"{plt_dir}/total-order_{out_names[idx]}.png")
            plt.close(fig)

        # Convert param_values (list of dicts) to ndarray with shape (N, D)
        param_values = np.array([[sample[name] for name in problem.names] for sample in param_values])

        for idx, output_name in enumerate(out_names):
            fig, axes = plt.subplots(
                len(problem.names),
                1,
                figsize=(10, 8),
                sharex=True
            )

            for i, name in enumerate(problem.names):
                plt.figure(figsize=(10, 6))
                plt.scatter(param_values[:, i],
                            outputs[:, :, idx].mean(axis=1),
                            label=f'{output_name}')
                plt.title(f'Mean Output ({output_name}) vs {name}')
                plt.xlabel(name)
                plt.ylabel('Mean Output')
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(f"{plt_dir}/mean_output_{output_name}_vs_{name}.png")
                plt.close(fig)

        # Plot and save the errors for each output in errors
        for output_name, error_values in errors.items():
            for i, param in enumerate(problem.names):
                plt.figure(figsize=(10, 6))
                plt.scatter(param_values[:, i],
                            error_values,
                            marker='o',
                            label=f'{output_name}')
                plt.title(f'Error Metrics Across Samples for {output_name}')
                plt.xlabel(f'{param}')
                plt.ylabel('Error')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{plt_dir}/error_{param}_{output_name}.png")
                plt.close()
