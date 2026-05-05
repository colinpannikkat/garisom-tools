# Basic data utils
import pandas as pd
import numpy as np
from typing import Any, Callable

# For model evaluation
from datetime import datetime

# Optimizer stuff
from garisom_tools.config import MetricConfig
from garisom_tools.utils.results import EvalResults

# Parallel runs
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import Model


class PotkayModel(Model):
    """
    Concrete implementation of the Model interface for the Potkay model.

    The PotkayModel class provides functionality to run the Potkay leaf-level physiology model,
    which integrates hydraulic, temperature, and carbon assimilation calculations. The model
    computes transpiration response and net assimilation across a range of leaf water potentials.

    This implementation:
    - Runs Potkay as a Python-based model (not a subprocess)
    - Supports parallel execution of multiple parameter sets
    - Provides comprehensive error handling and logging
    - Evaluates model outputs against ground truth observations

    Attributes:
    - run_kwargs (dict): Arguments for model execution
    - eval_kwargs (dict): Arguments for model evaluation

    Example:
        ```python
        import pandas as pd
        from garisom_tools import PotkayModel
        from garisom_tools.config import MetricConfig
        from datetime import datetime

        # Create model instance
        model = PotkayModel(
            run_kwargs={
                'E_range': (0, 0.01, 1e-5),
            },
            eval_kwargs={
                'ground': ground_truth_data,
                'start_date': datetime(2023, 7, 20),
                'end_date': datetime(2023, 8, 24)
            }
        )

        # Run with custom parameters
        result = model.run(X={'V_cmax_25': 60e-6, 'J_max_25': 110e-6})

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
        Initialize the PotkayModel instance.

        Args:
            run_kwargs (dict, optional): Keyword arguments for model execution.
                Expected keys include environmental and leaf parameters.
            eval_kwargs (dict, optional): Keyword arguments for model evaluation.
        """
        super().__init__(run_kwargs=run_kwargs, eval_kwargs=eval_kwargs)

    @classmethod
    def run_parallel(
        cls,
        params: dict | None = None,
        X: list[dict[str, float]] | None = None,
        workers: int = 4,
        **kwargs
    ) -> list[pd.DataFrame | None]:
        """
        Execute Potkay model runs in parallel for multiple parameter sets.

        Args:
            params (dict): Base parameter dict containing all model parameters.
            X (list[dict[str, float]], optional): List of parameter dictionaries to override
                base parameters. Each dict contains parameter names as keys and values as floats.
            workers (int, optional): Number of concurrent worker threads. Defaults to 4.
            **kwargs: Additional keyword arguments passed to individual run() calls.

        Returns:
            list[pd.DataFrame | None]: List of model outputs, one per parameter set.
                Failed runs return None in the corresponding list position.
        """
        N = len(X) if X else 0
        res: list[pd.DataFrame | None] = [None for _ in range(N)]

        pbar = tqdm(total=N)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    cls.run,
                    params=params,
                    X=X[i] if X is not None else None,
                    **kwargs
                ):
                i for i in range(N)
            }

            for future in as_completed(futures):
                pbar.update(1)
                idx = futures[future]
                try:
                    out = future.result()
                    res[idx] = out
                except Exception as e:
                    print(f"Model run for index {idx} failed: {e}")

        pbar.close()

        return res

    @classmethod
    def run(
        cls,
        params: dict | None = None,
        X: dict[str, float] | None = None,
        **kwargs
    ) -> pd.DataFrame | None:
        """
        Execute a single Potkay model run with specified parameters.

        Args:
            params (dict): Base parameter dict containing all model parameters.
            X (dict[str, float], optional): Dictionary of parameter overrides.
                Keys should match parameter names in the Potkay model.
            **kwargs: Additional keyword arguments passed to launch_model().

        Returns:
            pd.DataFrame | None: Model output containing transpiration response curves
                with columns for E, A_n, lambda, T_l, P_x_l, and other outputs.
                Returns None if the model run fails.
        """
        try:
            # Overwrite parameters with sample params if X is provided
            if params is not None and X is not None:
                params.update(X)
            output = cls.launch_model(params=params, **kwargs)
            return output
        except Exception as e:
            print(f"Model run failed: {e}")
            return None

    @classmethod
    def launch_model(
        cls,
        params: dict[str, float] | None = None,
        E_range: tuple[float, float, float] = (0, 0.01, 1e-5),
        env_data: pd.DataFrame | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame | None:
        """
        Launch the Potkay model and compute transpiration response curves.

        This method computes leaf physiological responses across a range of transpiration
        rates, including hydraulic, thermal, and photosynthetic components.

        Args:
            params (dict[str, float], optional): Dictionary of parameter overrides.
                Parameters follow INPUTS_0_Constants.m naming convention.
            E_range (tuple, optional): Tuple (min, max, step) for transpiration vector in mol/m^2/s.
                Defaults to (0, 0.01, 1e-5).
            env_data (pd.DataFrame, optional): Time series environmental data with columns:
                - year: Year
                - julian_day (or day): Julian day (1-366)
                - hour: Hour of day (0-23)
                - T_a: Air temperature [C]
                - RH_a: Relative humidity [unitless, 0-1]
                - R_abs: Absorbed radiation [W/m^2]
                - PPFD: Photosynthetic photon flux density [mol/m^2/s]
                If provided, the model runs for each row; otherwise uses single values from X.
            verbose (bool, optional): If True, print status messages during execution.
                Defaults to False.
            **kwargs: Additional parameters for model (unused, for interface compatibility).

        Returns:
            pd.DataFrame | None: Model output with columns:
                - year, julian_day, hour: Time info (if env_data provided)
                - E_vect: Transpiration [mol/m^2/s]
                - A_n_vect: Net assimilation [mol/m^2/s]
                - R_d_vect: Dark respiration [mol/m^2/s]
                - g_w_vect: Stomatal conductance to water vapor [mol/m^2/s]
                - g_c_vect: Total conductance to CO2 [mol/m^2/s]
                - lambda_vect: Marginal C profit of water [unitless]
                - P_x_l_vect: Leaf xylem water potential [MPa]
                - T_l_vect: Leaf temperature [C]
                - VPD_vect: Vapor pressure deficit [kPa]
        """
        try:
            # Load default constants
            constants = cls._get_default_constants()

            # Apply parameter overrides
            if params is not None:
                constants.update(params)

            # Determine if we have multiple timesteps
            if env_data is not None and len(env_data) > 0:
                # Multi-timestep mode
                if verbose:
                    print(f"Running Potkay model for {len(env_data)} timesteps")
                results = []
                for _, row in env_data.iterrows():
                    # Extract environmental data for this timestep
                    env_params = {
                        'T_a': row.get('T_a', constants['T_a']),
                        'RH_a': row.get('RH_a', constants['RH_a']),
                        'R_abs': row.get('R_abs', constants['R_abs']),
                        'PPFD': row.get('PPFD', constants['PPFD']),
                    }
                    constants_ts = constants.copy()
                    constants_ts.update(env_params)

                    # Run model for this timestep
                    result_ts = cls._compute_timestep(constants_ts, E_range, verbose)

                    # Add time info
                    result_ts['year'] = row.get('year', np.nan)
                    result_ts['julian-day'] = row.get('julian-day', row.get('day', np.nan))
                    result_ts['standard-time'] = row.get('hour', np.nan)

                    results.append(result_ts)

                # Concatenate results
                output = pd.concat(results, ignore_index=True)
            else:
                # Single timestep mode
                if verbose:
                    print("Running Potkay model for single timestep")
                output = cls._compute_timestep(constants, E_range, verbose)

            return output

        except Exception as e:
            print(f"Model launch failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    def _get_default_constants(cls) -> dict:
        """Load default constants from INPUTS_0_Constants.m specifications."""

        # Physical Constants
        R_gas = 8.314  # universal gas constant in [J/mol/K]
        sigma = 5.67e-8  # Stefan-Boltzman constant in [W/m^2/K^4]
        g_const = 9.81  # acceleration due to gravity in [m/s^2]

        # Properties of water and air
        rho = 998  # density of water in [kg/m^3]
        m_water = 18e-3  # molar mass of water in [kg/mol]
        V_w = 18e-6  # partial molar volume of water [m3 mol-1]
        C_p = 29.2  # molar heat capacity of air in [J/mol/K]
        emiss = 0.97  # emissivity of leaf [unitless]
        P_atm = 101.325  # atmospheric pressure in [kPa]
        c_w = 4184  # thermal capacitance (specific heat capacity) of water in [J/kg/K]

        # TODO: change to variable parameter based on leaf characteristic dimension and wind
        # with default value if leaf characteristic dim or wind isn't available
        g_b = 2.4  # boundary layer conductance to vapor in [mol/m^2/s]

        # Atmospheric conditions
        c_a = 410e-6  # atmospheric CO2 in [mol/mol]
        o_a = 21 / 101.325  # atmospheric O2 in [mol/mol]
        T_a = 25  # air temperature in [C]
        RH_a = 0.4  # relative humidity [unitless]
        # TODO: R_abs should be calculated
        R_abs = 600  # absorbed shortwave radiation in [W/m^2]
        PPFD = 1200e-6  # photon flux density in [mol/m^2/s]

        # Soil Hydraulic Properties
        h_soil = -0  # soil hydraulic head in [m]

        # Leaf apoplastic resistance
        r_min = 2.5e3  # minimum hydraulic resistance [MPa m2 s mol-1]

        # Leaf area
        LA = 2e-2  # leaf area in [m2]

        # Soil hydraulic Brooks and Corey parameters
        tau_soil = 2.6  # [unitless]
        h_soil_star = -25e-2  # in [m]
        K_soil_max_25 = 1e-6  # soil saturated conductivity at 25C in [m/s]

        # Root parameters
        r_0 = 5e-4  # root radius in [m]
        r_b = 1e-2  # rhizosphere radius in [m]
        ERL = 10  # effective root length in [m]
        k_r_25 = 1e-9  # root volumetric conductance at 25C in [m^3/s/MPa]

        # Xylem hydraulic Brooks and Corey parameters
        tau_x = 5  # [unitless]
        P_x_star = -1.65  # in [MPa]
        k_x_max_25 = 1e-9  # xylem volumetric conductance at 25C in [m^3/s/MPa]

        # Farquhar photosynthesis parameters
        alpha = 0.86  # leaf absorptance [unitless]
        V_cmax_25 = 60e-6  # maximum carboxylation capacity at 25C [mol/m^2/s]
        J_max_25 = 110e-6  # maximum electron transport rate at 25C [mol/m^2/s]

        # Smoothing parameters
        theta_c = 0.98  # smoothing parameter for A_n hyperbolic minimum
        theta_J = 0.90  # smoothing parameter for J hyperbolic minimum

        # Non-stomatal limitation parameters
        a_c = 2.92  # in [MPa^-1]
        P_x_l_c_50 = -2.45  # in [MPa]
        a_j = a_c  # in [MPa^-1]
        P_x_l_j_50 = P_x_l_c_50  # in [MPa]

        # Respiration
        Q10_r = 2

        # Hydraulic cost parameters
        omega = 0.1  # unitless
        theta = 0.5  # in [m^2 * s / mol]
        xi = 0.5  # in [MPa^-1]

        return {
            'R': R_gas,
            'sigma': sigma,
            'g': g_const,
            'rho': rho,
            'm': m_water,
            'V_w': V_w,
            'C_p': C_p,
            'emiss': emiss,
            'P_atm': P_atm,
            'g_b': g_b,
            'c_a': c_a,
            'o_a': o_a,
            'T_a': T_a,
            'RH_a': RH_a,
            'R_abs': R_abs,
            'PPFD': PPFD,
            'h_soil': h_soil,
            'r_min': r_min,
            'LA': LA,
            'tau_soil': tau_soil,
            'h_soil_star': h_soil_star,
            'K_soil_max_25': K_soil_max_25,
            'r_0': r_0,
            'r_b': r_b,
            'ERL': ERL,
            'k_r_25': k_r_25,
            'tau_x': tau_x,
            'P_x_star': P_x_star,
            'k_x_max_25': k_x_max_25,
            'alpha': alpha,
            'V_cmax_25': V_cmax_25,
            'J_max_25': J_max_25,
            'theta_c': theta_c,
            'theta_J': theta_J,
            'a_c': a_c,
            'P_x_l_c_50': P_x_l_c_50,
            'a_j': a_j,
            'P_x_l_j_50': P_x_l_j_50,
            'Q10_r': Q10_r,
            "omega": omega,
            "theta": theta,
            "xi": xi,
            "c_w": c_w
        }

    @classmethod
    def _compute_timestep(cls, constants: dict, E_range: tuple, verbose: bool = False) -> pd.DataFrame:
        """Compute model outputs for a single timestep."""

        # Create transpiration vector
        E_min, E_max, E_step = E_range
        E_vect = np.arange(E_min, E_max, E_step)  # [mol/m^2/s]

        # Extract constants
        m = constants['m']
        rho = constants['rho']
        g = constants['g']
        h_soil = constants['h_soil']
        T_a = constants['T_a']
        LA = constants['LA']
        r_0 = constants['r_0']
        r_b = constants['r_b']
        ERL = constants['ERL']
        k_r_25 = constants['k_r_25']
        tau_soil = constants['tau_soil']
        h_soil_star = constants['h_soil_star']
        K_soil_max_25 = constants['K_soil_max_25']
        tau_x = constants['tau_x']
        P_x_star = constants['P_x_star']
        k_x_max_25 = constants['k_x_max_25']
        R_abs = constants['R_abs']
        RH_a = constants['RH_a']
        g_b = constants['g_b']
        P_atm = constants['P_atm']
        c_a = constants['c_a']
        o_a = constants['o_a']
        alpha = constants['alpha']
        PPFD = constants['PPFD']
        theta_J = constants['theta_J']
        theta_c = constants['theta_c']
        V_cmax_25 = constants['V_cmax_25']
        J_max_25 = constants['J_max_25']
        Q10_r = constants['Q10_r']
        a_c = constants['a_c']
        P_x_l_c_50 = constants['P_x_l_c_50']
        a_j = constants['a_j']
        P_x_l_j_50 = constants['P_x_l_j_50']
        r_min = constants['r_min']
        C_p = constants['C_p']
        emiss = constants['emiss']
        sigma = constants['sigma']
        V_w = constants['V_w']
        R_gas = constants['R']
        omega = constants['omega']
        theta = constants['theta']
        xi = constants['xi']
        c_w = constants['c_w']

        # Define temperature-dependent functions
        def rel_k_soil_func(T): return 1.25 ** ((T - 25) / 10)
        def rel_k_r_func(T): return 1.60 ** ((T - 25) / 10)
        def rel_k_x_func(T): return 1.25 ** ((T - 25) / 10)
        def r_func(P_x_l): return r_min * np.exp(-xi * P_x_l)  # xi = 0.5
        def Phi_PSII_func(T_l): return np.ones_like(T_l) * 0.25  # 1 - 0.75 (F_ss) / 1.0 (F_max)
        def V_cmax_func(T_l): return V_cmax_25 * np.exp(8e4 * (T_l + 273.15 - 290) / 290 / R_gas / (T_l + 273.15))
        def J_max_func(T_l): return J_max_25 * np.exp(8e4 * (T_l + 273.15 - 290) / 290 / R_gas / (T_l + 273.15))
        def Gamma_star_func(T_l): return np.ones_like(T_l) * 36e-6
        def K_c_func(T_l): return np.ones_like(T_l) * 275e-6
        def K_o_func(T_l): return np.ones_like(T_l) * 420000e-6
        def R_d_func(T_l): return (0.01 * V_cmax_func(25)) * Q10_r ** ((T_l - 25) / 10)
        def NSL_c_func(P_x_l): return 1.0 / (1.0 + np.exp(a_c * (P_x_l_c_50 - P_x_l)))
        def NSL_j_func(P_x_l): return 1.0 / (1.0 + np.exp(a_j * (P_x_l_j_50 - P_x_l)))

        with np.errstate(divide='ignore', invalid='ignore'):

            # Call hydraulics function
            P_x_l_vect, P_x_r_vect, P_0_vect, dP_x_ldE_vect = cls._hydraulics(
                E_vect, m, rho, g, h_soil, T_a,
                LA, r_0, r_b, ERL, k_r_25,
                rel_k_soil_func, rel_k_r_func, rel_k_x_func,
                tau_soil, h_soil_star, K_soil_max_25,
                tau_x, P_x_star, k_x_max_25
            )

            # Call conductances and temperature function
            T_l_vect, g_w_vect, g_c_vect, g_tot_vect, VPD_vect, dEdg_w_vect, dg_wdg_c_vect, L, RH_l_vect = \
                cls._conductances_and_temperature(
                    E_vect, P_x_l_vect, dP_x_ldE_vect, P_atm,
                    R_abs, T_a, RH_a, g_b, r_func,
                    m, C_p, emiss, sigma, V_w, R_gas
                )

            # Call carbon assimilation function
            A_n_vect, R_d_vect, lambda_vect = cls._carbon_assimilation(
                g_c_vect, T_l_vect,
                alpha, PPFD, Phi_PSII_func,
                theta_J, theta_c,
                V_cmax_func, J_max_func, Gamma_star_func,
                K_c_func, K_o_func, R_d_func, c_a, o_a,
                C_p, emiss, sigma, L, g_b,
                dEdg_w_vect, dg_wdg_c_vect,
                NSL_c_func, NSL_j_func,
                P_x_l_vect, dP_x_ldE_vect
            )

            risk_vect = cls._calculate_hydraulic_cost(
                E_vect,
                T_l_vect + 273.15,  # convert back to Kelvin
                P_x_l_vect,
                omega,
                theta,
                xi,
                c_w,
                L / m  # latent heat of vaporization in [J/kg]
            )

        profit = np.abs(lambda_vect - risk_vect)
        profit_max_idx = np.nanargmin(profit)

        # Package results into DataFrame
        output = pd.DataFrame(
            {
                'E-MD':  E_vect[profit_max_idx] * 1e3,  # mol/m^2/s -> mmol/m^2/s
                'A_n': A_n_vect[profit_max_idx],
                'R_d': R_d_vect[profit_max_idx],
                'GW': g_w_vect[profit_max_idx] * 1e3,  # mol/m^2/s -> mmol/m^2/s
                'g_c': g_c_vect[profit_max_idx] * 1e3,  # mol/m^2/s -> mmol/m^2/s
                'g_tot': g_tot_vect[profit_max_idx] * 1e3,  # mol/m^2/s -> mmol/m^2/s
                'lambda': lambda_vect[profit_max_idx],
                'risk': risk_vect[profit_max_idx],
                'profit': profit[profit_max_idx],
                'profit_idx': profit_max_idx,
                'P_x_l': P_x_l_vect[profit_max_idx],
                'P_x_r': P_x_r_vect[profit_max_idx],
                'P_0': P_0_vect[profit_max_idx],
                'leaftemp': T_l_vect[profit_max_idx],
                'leaf-air-temp-diff': T_l_vect[profit_max_idx] - T_a,
                'VPD': VPD_vect[profit_max_idx],
                'RH_l': RH_l_vect[profit_max_idx],
            },
            index=[0]
        )

        return output

    @staticmethod
    def _hydraulics(
        E_vect: np.ndarray,
        m,
        rho,
        g,
        h_soil,
        T_a,
        LA,
        r_0,
        r_b,
        ERL,
        k_r_25,
        rel_k_soil_func: Callable,
        rel_k_r_func: Callable,
        rel_k_x_func: Callable,
        tau_soil,
        h_soil_star,
        K_soil_max_25,
        tau_x,
        P_x_star,
        k_x_max_25
    ):
        """Port of FUNCTION_Hydraulics from MATLAB.

        Assuming immediate refilling.

        Outputs
            P_x_l - Leaf xylem water potential in [MPa]
            P_x_r - Root xylem water potential in [MPa]
            P_0 - Water potential at soil-root surface in [MPa]
            dP_x_ldE - derivative of leaf xylem water potential with respect to transpiration [MPa*m^2*s/mol]

        Inputs
            E_vect - transpiration vector [mol/m^2/s]
            m - molar mass of water in [kg/mol]
            rho - density of water in [kg/m^3]
            h_soil - soil hydraulic head in [m]
            LA - leaf area in [m2]
            r_0 - root radius in [m]
            r_b - rhizosphere radius in [m]
            ERL - effective root length in [m]
            k_r_25 - root (volumetric) conductance at 25C in [m^3/s/MPa]
            rel_k_soil_func - temperature dependence for saturated soil conductivity
            rel_k_r_func - temperature dependence for maximum root conductance
            rel_k_x_func - temperature dependence for maximum xylem conductance
            tau_soil - Brooks and Corey exponent for soil conductivity curve
            h_soil_star - hydraulic head at which soil conductivity begins to decrease in [m]
            K_soil_max_25 - soil saturated conductivity at 25C in [m/s]
            tau_x - Brooks and Corey exponent for xylem conductance (vulnerability) curve
            P_x_star - water potential at which xylem conductance begins to decrease in [MPa]
            k_x_max_25 - xylem (volumetric) conductance at 25C in [m^3/s/MPa]
        """

        E_volum_vect = m / rho * LA * E_vect  # volumetric flux in [m^3/s]

        K_soil_max = K_soil_max_25 * rel_k_soil_func(T_a)  # soil saturated conductivity in [m/s]
        k_r = k_r_25 * rel_k_r_func(T_a)  # root volumetric conductance in [m^3/s/MPa]
        k_x_max = k_x_max_25 * rel_k_x_func(T_a)  # xylem maximum volumetric conductance in [m^3/s/MPa]

        # Compute h_0_vect (soil-root surface hydraulic head)
        coeff = E_volum_vect * (1 - tau_soil) / 2 / np.pi / ERL / K_soil_max / np.abs(h_soil_star) ** tau_soil
        coeff *= (r_b ** 2 * np.log(r_b / r_0) / (r_b ** 2 - r_0 ** 2) - 0.5)
        h_0_vect = -(np.abs(h_soil) ** (1 - tau_soil) + coeff) ** (1 / (1 - tau_soil))
        h_0_vect[np.iscomplex(h_0_vect)] = np.nan  # ignore imaginary numbers

        P_0_vect = 1e-6 * rho * g * h_0_vect  # in [MPa]
        P_x_r_vect = P_0_vect - E_volum_vect / k_r

        # Compute P_x_l_vect (leaf xylem water potential)
        P_x_l_vect = -(np.abs(P_x_r_vect) ** (1 - tau_x) +
                       E_volum_vect * (1 - tau_x) / k_x_max / np.abs(P_x_star) ** tau_x) ** (1 / (1 - tau_x))
        P_x_l_vect[np.iscomplex(P_x_l_vect)] = np.nan  # ignore imaginary numbers

        # Compute derivatives
        dh_0dE_volum_vect = -(r_b ** 2 * np.log(r_b / r_0) / (r_b ** 2 - r_0 ** 2) - 0.5) / 2 / np.pi / ERL / K_soil_max
        dh_0dE_volum_vect *= (h_0_vect / h_soil_star) ** tau_soil

        dP_0dE_volum_vect = 1e-6 * rho * g * dh_0dE_volum_vect
        dP_x_rdE_volum_vect = dP_0dE_volum_vect - 1 / k_r
        dP_x_ldE_volum_vect = (-1 / k_x_max * (P_x_l_vect / P_x_star) ** tau_x +
                               (P_x_l_vect / P_x_r_vect) ** tau_x * dP_x_rdE_volum_vect)
        dP_x_ldE_vect = m / rho * LA * dP_x_ldE_volum_vect

        return P_x_l_vect, P_x_r_vect, P_0_vect, dP_x_ldE_vect

    @staticmethod
    def _conductances_and_temperature(
        E_vect: np.ndarray,
        P_x_l_vect: np.ndarray,
        dP_x_ldE_vect: np.ndarray,
        P_atm,
        R_abs,
        T_a,
        RH_a,
        g_b,
        r_func: Callable,
        m,
        C_p,
        emiss,
        sigma,
        V_w,
        R_gas
    ):
        """Port of FUNCTION_Conductances_and_Temperature from MATLAB."""

        # Convert to Kelvin
        T_a_K = T_a + 273.15

        # Latent heat of vaporization in [J/kg]
        L_kg = 1.91846e6 * (T_a_K / (T_a_K - 33.91)) ** 2  # from Henderson-Sellers (1984)
        L = m * L_kg  # convert to [J/mol]

        # Apoplast water potential
        r_vect = r_func(P_x_l_vect)
        psi_A_l_vect = P_x_l_vect - r_vect * E_vect

        # Derivative of r with respect to P_x_l
        dP_x_l = 0.01
        drdP_x_l_vect = (r_func(P_x_l_vect + dP_x_l) - r_func(P_x_l_vect)) / dP_x_l
        dpsi_A_ldE_vect = (1 - drdP_x_l_vect * E_vect) * dP_x_ldE_vect - r_vect

        # Solve for leaf temperature (leaf energy balance / F = 0) using Newton's method
        n = len(E_vect)
        T_l_K_vect = np.zeros(n)

        for i in range(n):
            T_l_K = T_a_K
            E_i = E_vect[i]

            for _ in range(100):
                F = emiss * sigma * T_l_K ** 4 + L * E_i + C_p * g_b * (T_l_K - T_a_K) - R_abs - sigma * T_a_K ** 4
                dFdT = 4 * emiss * sigma * T_l_K ** 3 + C_p * g_b

                T_l_K_new = T_l_K - 0.3 * F / dFdT

                # Check convergence
                residual_scale = L * E_i - C_p * g_b * T_a_K - R_abs - sigma * T_a_K ** 4
                if abs(residual_scale) < 1e-6:
                    if abs(F) < abs(C_p * g_b * T_a_K / 1e6):
                        break
                else:
                    if abs(F) < abs(residual_scale / 1e4):
                        break

                T_l_K = T_l_K_new

            T_l_K_vect[i] = T_l_K

        # Relative humidity of leaf
        RH_l_vect = np.exp(V_w * psi_A_l_vect / 1e-6 / R_gas / T_l_K_vect)

        # Convert temperatures back to Celsius
        T_a_C = T_a
        T_l_vect = T_l_K_vect - 273.15

        # Vapor pressures of air and leaf (Teten's equation)
        e_l_sat_vect = 0.61078 * np.exp(17.27 * T_l_vect / (T_l_vect + 237.3))  # in [kPa]
        e_l_vect = RH_l_vect * e_l_sat_vect  # in [kPa]
        e_a = RH_a * 0.61078 * np.exp(17.27 * T_a_C / (T_a_C + 237.3))  # in [kPa]
        VPD_vect = e_l_vect - e_a  # vapor pressure deficit in [kPa]

        # Stomatal conductances
        g_tot_vect = E_vect / (VPD_vect / P_atm)  # total conductance to vapor in [mol/m^2/s]
        g_w_vect = 1.0 / (1.0 / g_tot_vect - 1.0 / g_b)  # stomatal conductance to vapor
        g_w_vect[E_vect == 0] = 0
        g_c_vect = 1.0 / (1.6 / g_w_vect + 1.37 / g_b)  # total conductance to CO2
        g_c_vect[E_vect == 0] = 0

        # Slope of saturation vapor pressure curve
        s_vect = 17.27 * 237.3 * e_l_sat_vect / (T_l_vect + 237.3) ** 2

        # Partial derivatives
        denom_1 = 4 * emiss * sigma * T_l_K_vect ** 3 + C_p * g_b
        term1 = (RH_l_vect / P_atm * (s_vect - np.log(RH_l_vect) * e_l_sat_vect /
                 T_l_K_vect) * L / denom_1)
        term2 = (RH_l_vect / P_atm * np.log(RH_l_vect) * e_l_sat_vect /
                 psi_A_l_vect * dpsi_A_ldE_vect)
        dEdg_w_vect = E_vect / g_w_vect ** 2 / (
            1.0 / g_w_vect + 1.0 / g_b + term1 - term2
        )
        dEdg_w_vect[E_vect == 0] = VPD_vect[E_vect == 0] / P_atm

        dg_wdg_c_vect = (1.6 + 1.37 * g_w_vect / g_b) ** 2 / 1.6
        mask_neg = g_w_vect < 0
        dg_wdg_c_vect[mask_neg] = (-1.6 + 1.37 * g_w_vect[mask_neg] / g_b) ** 2 / -1.6

        return T_l_vect, g_w_vect, g_c_vect, g_tot_vect, VPD_vect, dEdg_w_vect, dg_wdg_c_vect, L, RH_l_vect

    @staticmethod
    def _carbon_assimilation(
        g_c_vect: np.ndarray,
        T_l_vect: np.ndarray,
        alpha,
        PPFD,
        Phi_PSII_func: Callable,
        theta_J,
        theta_c,
        V_cmax_func: Callable,
        J_max_func: Callable,
        Gamma_star_func: Callable,
        K_c_func: Callable,
        K_o_func: Callable,
        R_d_func: Callable,
        c_a,
        o_a,
        C_p,
        emiss,
        sigma,
        L,
        g_b,
        dEdg_w_vect: np.ndarray,
        dg_wdg_c_vect: np.ndarray,
        NSL_c_func: Callable,
        NSL_j_func: Callable,
        P_x_l_vect: np.ndarray,
        dP_x_ldE_vect: np.ndarray
    ):
        """Port of FUNCTION_Carbon_Assimilation from MATLAB.

        Inputs - Constants
            alpha - leaf absorptance [unitless]
            theta_J - smoothing parameter for A_n in hyperbolic minimum of A_j and A_c [unitless]
            theta_c - smoothing parameter for J in hyperbolic minimum of J_phi and J_max [unitless]
            C_p - molar heat capacity of air in [J/mol/K]
            emiss - emissivity of leaf [unitless]
            sigma - Stefan-Boltzman constant in [W/m^2/K^4]

        Inputs - Environmental Conditions
            PPFD - photon flux density in [mol/m^2/s]
            c_a - atmospheric CO2 in [mol/mol]
            o_a - atmospheric O2 in [mol/mol]
            g_b - boundary layer conductance to vapor in [mol/m^2/s]

        Inputs - Temperature-dependent local functions
            V_cmax_func - maximum carboxylation capacity under hydrated conditions in [mol/m^2/s]
            J_max_func - maximum electron transport rate under hydrated conditions in [mol/m^2/s]
            Gamma_star_func - CO2 compensation point [mol/mol]
            K_c_func - Michaelis-Menten coefficient for carboxylation [mol/mol]
            K_o_func - Michaelis-Menten coefficient for oxygenation [mol/mol]
            R_d_func - day respiration in [mol/m^2/s]
            Phi_PSII_func - photochemical efficiency of PSII [unitless]

        Inputs - Leaf water potential-dependent local functions
            NSL_c_func - "Non-Stomatal Limitation" function for carboxylation that modifies V_cmax [unitless]
            NSL_j_func - "Non-Stomatal Limitation" function for carboxylation that modifies J_max [unitless]

        Inputs - previously calculated variables ['vect' indicates a vector of values]
            L - latent heat of vaporization in [J/kg]
            g_c_vect - total conductance to CO2 in [mol/m^2/s]
            T_l_vect - leaf temperature in [C]
            dEdg_w_vect - derivative of transpiration with respect to stomatal conductance to vapor [unitless]
            dg_wdg_c_vect - derivative of stomatal conductance to vapor with respect to total conductance to
                CO2 [unitless]
            P_x_l_vect - leaf water potential in [MPa]
            dP_x_ldE_vect - derivative of leaf water potential with respect to transpiration in [MPa*m^2*s/mol]
        """

        # Gross Assimilation variables
        V_cmax_vect = V_cmax_func(T_l_vect) * NSL_c_func(P_x_l_vect)
        J_max_vect = J_max_func(T_l_vect) * NSL_j_func(P_x_l_vect)
        Phi_PSII_vect = Phi_PSII_func(T_l_vect)
        J_phi_vect = alpha / 2 * Phi_PSII_vect * PPFD

        # Hyperbolic minimum of J_max and J_phi
        J_vect = (J_max_vect + J_phi_vect) / (2 * theta_J) - \
            np.sqrt(((J_max_vect + J_phi_vect) / (2 * theta_J)) ** 2 - J_max_vect * J_phi_vect / theta_J)

        Gamma_star_vect = Gamma_star_func(T_l_vect)
        K_c_vect = K_c_func(T_l_vect)
        K_o_vect = K_o_func(T_l_vect)

        # Respiration
        R_d_vect = R_d_func(T_l_vect)

        # Local functions for A_c and A_j
        def A_c_func(g_c, A_n, V_cmax, Gamma_star, R_d, K_c, K_o):
            with np.errstate(divide='ignore', invalid='ignore'):
                result = V_cmax * (c_a - A_n / g_c - Gamma_star) / (c_a - A_n / g_c + K_c * (1 + o_a / K_o)) - R_d
            return np.asarray(result)

        def A_j_func(g_c, A_n, J, Gamma_star, R_d):
            with np.errstate(divide='ignore', invalid='ignore'):
                result = J / 4 * (c_a - A_n / g_c - Gamma_star) / (c_a - A_n / g_c + 2 * Gamma_star) - R_d
            return np.asarray(result)

        def hyperbolic_min(A_c, A_j):
            """Compute hyperbolic minimum of A_c and A_j."""
            return (A_c + A_j) / (2 * theta_c) - np.sqrt(
                ((A_c + A_j) / (2 * theta_c)) ** 2 - A_c * A_j / theta_c
            )

        n = len(g_c_vect)
        if np.any(~np.isnan(P_x_l_vect)):
            idx_max = np.nanargmax(~np.isnan(P_x_l_vect))
            n_stop = idx_max + np.sum(~np.isnan(P_x_l_vect[idx_max:]))
        else:
            n_stop = n
        n_stop = min(n_stop, n)

        A_n_vect = np.full(n, np.nan)
        lambda_vect = np.full(n, np.nan)

        for i in range(n_stop):

            g_c_i = g_c_vect[i]
            V_cmax_i = V_cmax_vect[i]
            J_i = J_vect[i]
            Gamma_star_i = Gamma_star_vect[i]
            R_d_i = R_d_vect[i]
            K_c_i = K_c_vect[i]
            K_o_i = K_o_vect[i]

            # Handle zero conductance or no light
            if g_c_i == 0:
                # if no light (i.e., A_n = 0)
                if J_i < 1e-16:
                    A_n_vect[i] = -R_d_i
                else:
                    # if there is light (i.e., A_n > 0)
                    A_n_vect[i] = 0
            elif g_c_i < 0:
                # if negative conductance -- unrealistic
                A_n_vect[i] = np.nan
            elif J_i < 1e-16:
                # if no light (i.e., A_n = 0)
                A_n_vect[i] = -R_d_i
            else:
                # Bisection with 3-point iteration
                A_n_max = min(
                    max(V_cmax_i, J_i / 4),
                    g_c_i * c_a
                )

                A_n_LB = -R_d_i
                A_n_UB = A_n_max
                A_n_M = (A_n_LB + A_n_UB) / 2
                A_n_i = np.hstack([A_n_LB, A_n_M, A_n_UB])

                for _ in range(100):
                    A_c_i = A_c_func(g_c_i, A_n_i, V_cmax_i, Gamma_star_i, R_d_i, K_c_i, K_o_i)
                    A_j_i = A_j_func(g_c_i, A_n_i, J_i, Gamma_star_i, R_d_i)
                    A_n_hypmin = hyperbolic_min(A_c_i, A_j_i)
                    F = (A_n_i - A_n_hypmin) / A_n_max

                    # Check convergence
                    if np.any(np.abs(F) < 1e-4):
                        break

                    # Update bounds
                    F_positive = F[F > 0]
                    F_negative = F[F < 0]

                    if len(F_positive) == 0:
                        F_UB = F[0]
                    else:
                        F_UB = np.min(F_positive)

                    if len(F_negative) == 0:
                        F_LB = np.nan
                    else:
                        F_LB = np.max(F_negative)

                    A_n_LB = min(A_n_i[F == F_LB]) if not np.isnan(F_LB) else -R_d_i
                    A_n_UB = max(A_n_i[F == F_UB])

                    if A_n_LB == A_n_UB:
                        if (A_n_UB == -R_d_i) and (A_n_LB == -R_d_i):
                            break
                        else:
                            raise Exception('ERROR: Photosynthesis module cannot converge on A_n!!!')

                    A_n_M = (A_n_LB + A_n_UB) / 2
                    A_n_i = np.hstack([A_n_LB, A_n_M, A_n_UB])

                # Select best solution
                A_n_vect[i] = A_n_i[np.argmin(np.abs(F))]  # type: ignore
                if np.isnan(A_n_vect[i]):
                    raise Exception("ERROR: NaN in A_n!!")

        c_i_vect = c_a - A_n_vect / g_c_vect

        if np.any(c_i_vect[2:] < 0):
            raise Exception("ERROR: Internal C02 cannot be negative.")

        # Assuming limited by carboxylation at low c_i
        c_i_min_vect = (R_d_vect * K_c_vect * (1 + o_a / K_o_vect) + V_cmax_vect * Gamma_star_vect) / (
            V_cmax_vect - R_d_vect
        )
        A_c_gross_0 = V_cmax_vect * (c_i_min_vect - Gamma_star_vect) / (
            c_i_min_vect + K_c_vect * (1 + o_a / K_o_vect)
        )
        mask_zero_gc = g_c_vect == 0
        if np.any((J_vect[mask_zero_gc] / 4) < A_c_gross_0[mask_zero_gc]):
            c_i_min_vect[mask_zero_gc] = np.inf
        c_i_vect[mask_zero_gc] = c_i_min_vect[mask_zero_gc]

        # Calculate A_c_vect and A_j_vect
        A_c_vect = A_c_func(g_c_vect, A_n_vect, V_cmax_vect, Gamma_star_vect, R_d_vect, K_c_vect, K_o_vect)
        A_c_vect[mask_zero_gc] = 0
        A_j_vect = A_j_func(g_c_vect, A_n_vect, J_vect, Gamma_star_vect, R_d_vect)
        A_j_vect[mask_zero_gc] = 0

        # Partial derivatives of hyperbolic minimum
        denom = (A_c_vect + A_j_vect) ** 2 - 4 * theta_c * A_c_vect * A_j_vect
        denom[denom < 0] = 1e-16  # avoid sqrt of negative
        sqrt_denom = np.sqrt(denom)

        dA_ndA_c_vect = (1 - (A_c_vect + (1 - 2 * theta_c) * A_j_vect) / sqrt_denom) / (2 * theta_c)
        dA_ndA_j_vect = (1 - (A_j_vect + (1 - 2 * theta_c) * A_c_vect) / sqrt_denom) / (2 * theta_c)

        # At g_c = 0, A_n = A_c and A_j = 0, assuming limited by carboxylation at low c_i
        dA_ndA_c_vect[mask_zero_gc] = 0
        dA_ndA_j_vect[mask_zero_gc] = 1

        # Slope of biochemical supply curve with respect to c_i
        dA_cdc_i_vect = V_cmax_vect * (Gamma_star_vect + K_c_vect * (1 + o_a / K_o_vect)) / (
            c_i_vect + K_c_vect * (1 + o_a / K_o_vect)
        ) ** 2
        dA_jdc_i_vect = 0.75 * J_vect * Gamma_star_vect / (c_i_vect + 2 * Gamma_star_vect) ** 2

        # Canopy-level slope of biochemical supply curve with respect to c_i
        k_vect = dA_ndA_c_vect * dA_cdc_i_vect + dA_ndA_j_vect * dA_jdc_i_vect

        # Canopy-level slope with respect to T_l
        denom_J = (J_max_vect + J_phi_vect) ** 2 - 4 * theta_J * J_max_vect * J_phi_vect
        denom_J[denom_J < 0] = 1e-16
        sqrt_denom_J = np.sqrt(denom_J)

        dJdJ_max_vect = (1 - (J_max_vect + (1 - 2 * theta_J) * J_phi_vect) / sqrt_denom_J) / (2 * theta_J)
        dJdJ_phi_vect = (1 - (J_phi_vect + (1 - 2 * theta_J) * J_max_vect) / sqrt_denom_J) / (2 * theta_J)

        dT_l = 0.01
        dV_cmaxdT_l_vect = (V_cmax_func(T_l_vect + dT_l) * NSL_c_func(P_x_l_vect) - V_cmax_vect) / dT_l
        dJ_maxdT_l_vect = (J_max_func(T_l_vect + dT_l) * NSL_j_func(P_x_l_vect) - J_max_vect) / dT_l
        dJ_phidT_l_vect = alpha / 2 * PPFD * (Phi_PSII_func(T_l_vect + dT_l) - Phi_PSII_vect) / dT_l
        dR_ddT_l_vect = (R_d_func(T_l_vect + dT_l) - R_d_vect) / dT_l

        dA_cdV_cmax_vect = (A_c_vect + R_d_vect) / V_cmax_vect
        dA_jdJ_vect = 0.25 * (c_i_vect - Gamma_star_vect) / (c_i_vect + 2 * Gamma_star_vect)
        dA_jdJ_vect[c_i_vect == np.inf] = 0.25

        xi_vect = (
            dA_ndA_c_vect * (dA_cdV_cmax_vect * dV_cmaxdT_l_vect - dR_ddT_l_vect)
            + dA_ndA_j_vect
            * (
                dA_jdJ_vect * dJdJ_max_vect * dJ_maxdT_l_vect
                - dR_ddT_l_vect
                + dJdJ_phi_vect * dJdJ_phi_vect * dJ_phidT_l_vect
            )
        )

        # Canopy-level slope with respect to P_x_l (non-stomatal limitations)
        dP_x_l = 0.01
        dlnNSL_cdP_x_l_vect = (NSL_c_func(P_x_l_vect + dP_x_l) - NSL_c_func(P_x_l_vect)) / dP_x_l
        dlnNSL_jdP_x_l_vect = (NSL_j_func(P_x_l_vect + dP_x_l) - NSL_j_func(P_x_l_vect)) / dP_x_l

        dA_ndP_x_l_vect = (
            dA_ndA_c_vect * (A_c_vect + R_d_vect) * dlnNSL_cdP_x_l_vect
            + dA_ndA_j_vect * (A_j_vect + R_d_vect) * dJdJ_max_vect * J_max_vect / J_vect * dlnNSL_jdP_x_l_vect
        )
        dA_ndP_x_l_vect[A_n_vect == -R_d_vect] = 0

        # Correction where multiplying 0 by infinity
        dA_ndP_x_l_time_dP_x_ldE_vect = dA_ndP_x_l_vect * dP_x_ldE_vect
        dA_ndP_x_l_time_dP_x_ldE_vect[dA_ndP_x_l_vect == 0] = 0

        # Marginal water-use efficiency: lambda = dA_n/dE
        T_l_K_vect = T_l_vect + 273.15  # Convert back to Kelvin
        lambda_vect = (
            k_vect
            / (k_vect + g_c_vect)
            * (c_a - c_i_vect)
            / dg_wdg_c_vect
            / dEdg_w_vect
            - g_c_vect / (k_vect + g_c_vect) * xi_vect * L / (4 * emiss * sigma * T_l_K_vect ** 3 + C_p * g_b)
            + g_c_vect / (k_vect + g_c_vect) * dA_ndP_x_l_time_dP_x_ldE_vect
        )

        # Correction at g_c = 0
        lambda_vect[mask_zero_gc] = (
            (c_a - c_i_vect[mask_zero_gc]) / dg_wdg_c_vect[mask_zero_gc] / dEdg_w_vect[mask_zero_gc]
        )

        # Nighttime correction
        mask_night = J_vect < 1e-16
        lambda_vect[mask_night] = dR_ddT_l_vect[mask_night] * L / (
            4 * emiss * sigma * T_l_K_vect[mask_night] ** 3 + C_p * g_b
        )

        # Check for NaNs
        # if np.any(np.isnan(lambda_vect[g_c_vect[:n_stop] > 0])):
        #     raise Exception("ERROR: NaN in 'lambda_vect'!!!")

        return A_n_vect, R_d_vect, lambda_vect

    @staticmethod
    def _calculate_hydraulic_cost(
        E_vect: np.ndarray,
        T_l_k_vect: np.ndarray,
        P_x_l_vect: np.ndarray,
        omega,
        theta,
        xi,
        c_w,
        L,
    ):
        return L / c_w / T_l_k_vect * (omega + theta * np.exp(-xi * P_x_l_vect) * E_vect)

    @classmethod
    def evaluate_model(
        cls,
        output: pd.DataFrame,
        ground: pd.DataFrame,
        metric_config: MetricConfig,
        start_date: datetime,
        end_date: datetime
    ) -> EvalResults:
        """
        Evaluate Potkay model output against ground truth data.

        Args:
            output (pd.DataFrame | None): Model output from launch_model().
            ground (pd.DataFrame): Ground truth observations with 'year' and 'julian-day' columns.
            metric_config (MetricConfig): Configuration for metrics to compute.
            start_date (datetime): Start date for evaluation period.
            end_date (datetime): End date for evaluation period.

        Returns:
            EvalResults: Dictionary mapping metric names to computed values.
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

                col_pred = pred[:, idx]
                col_pred = pd.DataFrame(col_pred)
                pred_values = col_pred.loc[col_ground.index].T.to_numpy().squeeze(axis=0)

                err = eval_func(ground_values, pred_values)

            errors[optim_name] = err

        return errors
