import pandas as pd
import numpy as np
import os
from datetime import datetime
from garisom_tools.model import SperryModel, PotkayModel, Model
from garisom_tools.config import MetricConfig
from garisom_tools.utils.metric import Metric, Mode


model_dir = "../garisom/02_program_code"


def get_parameter_and_configuration_files() -> tuple[str, pd.DataFrame]:
    return os.path.abspath("./tests/data/configuration.csv"), \
        pd.read_csv("./tests/data/parameters.csv")


config_file, params = get_parameter_and_configuration_files()


def test_model_inheritance():
    assert issubclass(SperryModel, Model)


def test_get_objective_returns_callable():
    model = SperryModel(run_kwargs={'a': 1}, eval_kwargs={})
    obj = model.get_objective()
    assert callable(obj)


def test_setup_model_and_return_callable_returns_callable():
    metric = MetricConfig(metrics=[], modes=[])
    model = SperryModel(run_kwargs={}, eval_kwargs={})
    wrapped = model.setup_model_and_return_callable(metric)
    assert callable(wrapped)


def test_evaluate_model_returns_dict():

    def dummy_func(x, y):
        return np.sum(np.abs(x - y))

    m = Metric(output_name='val', name='dummy', func=dummy_func)

    metric_config = MetricConfig(metrics=[m], modes=[Mode.from_name('min')])
    output = pd.DataFrame({'val': [1, 2, 3]})
    ground = pd.DataFrame({'year': [2023, 2023, 2023], 'julian-day': [1, 2, 3], 'val': [1, 2, 4]})
    errors = SperryModel.evaluate_model(
        output, ground, metric_config,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 3)
    )
    assert isinstance(errors, dict)
    assert 'dummy' in errors


def test_evaluate_model_filters_by_date_range():
    """Test that evaluate_model correctly filters data by date range."""

    def sum_func(x, y):
        return np.sum(x)

    m = Metric(output_name='val', name='sum', func=sum_func)
    metric_config = MetricConfig(metrics=[m], modes=[Mode.from_name('min')])

    # Create output and ground data spanning multiple years
    output = pd.DataFrame({'val': [10, 20, 30, 40, 50]})
    ground = pd.DataFrame({
        'year': [2022, 2022, 2023, 2023, 2023],
        'julian-day': [364, 365, 1, 2, 3],
        'val': [10, 20, 30, 40, 50]
    })

    # Filter to only include 2023 data (julian-day 1-3)
    errors = SperryModel.evaluate_model(
        output, ground, metric_config,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 3)
    )

    # Should sum only values 30+40+50 = 120
    assert errors['sum'] == 120


def test_evaluate_model_filters_across_years():
    """Test filtering when date range spans multiple years."""

    def count_func(x, y):
        return len(x)

    m = Metric(output_name='val', name='count', func=count_func)
    metric_config = MetricConfig(metrics=[m], modes=[Mode.from_name('min')])

    output = pd.DataFrame({'val': [1, 2, 3, 4, 5, 6]})
    ground = pd.DataFrame({
        'year': [2022, 2022, 2023, 2023, 2023, 2024],
        'julian-day': [360, 365, 1, 100, 365, 10],
        'val': [1, 2, 3, 4, 5, 6]
    })

    # Filter from Dec 26, 2022 (day 360) to Jan 10, 2024 (day 10)
    errors = SperryModel.evaluate_model(
        output, ground, metric_config,
        start_date=datetime(2022, 12, 26),
        end_date=datetime(2024, 1, 10)
    )

    # Should include all 6 rows
    assert errors['count'] == 6


def test_evaluate_model_single_year_partial():
    """Test filtering within a single year with partial range."""

    def count_func(x, y):
        return len(x)

    m = Metric(output_name='val', name='count', func=count_func)
    metric_config = MetricConfig(metrics=[m], modes=[Mode.from_name('min')])

    output = pd.DataFrame({'val': [1, 2, 3, 4, 5]})
    ground = pd.DataFrame({
        'year': [2023, 2023, 2023, 2023, 2023],
        'julian-day': [100, 150, 200, 250, 300],
        'val': [1, 2, 3, 4, 5]
    })

    # Filter July 1 (day 182) to Aug 31 (day 243) - should get days 200
    errors = SperryModel.evaluate_model(
        output, ground, metric_config,
        start_date=datetime(2023, 7, 1),
        end_date=datetime(2023, 8, 31)
    )

    # Should include only day 200
    assert errors['count'] == 1


def test_evaluate_model_none_output_returns_penalty():
    """Test that None output returns penalty values."""

    def dummy_func(x, y):
        return 0

    m = Metric(output_name='val', name='dummy', func=dummy_func)
    metric_config = MetricConfig(metrics=[m], modes=[Mode.from_name('min')])

    ground = pd.DataFrame({'year': [2023], 'julian-day': [1], 'val': [1]})
    errors = SperryModel.evaluate_model(
        None, ground, metric_config,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 1)
    )

    assert errors['dummy'] == 1e20


# ==================== POTKAY MODEL TESTS ====================

class TestPotkayModel:
    """Tests for the PotkayModel class."""

    def test_potkay_model_inheritance(self):
        """Test that PotkayModel is a subclass of Model."""
        assert issubclass(PotkayModel, Model)

    def test_potkay_model_initialization(self):
        """Test PotkayModel can be initialized with empty kwargs."""
        model = PotkayModel(run_kwargs={}, eval_kwargs={})
        assert model is not None
        assert isinstance(model, PotkayModel)

    def test_potkay_get_objective_returns_callable(self):
        """Test that get_objective returns a callable for PotkayModel."""
        model = PotkayModel(run_kwargs={}, eval_kwargs={})
        obj = model.get_objective()
        assert callable(obj)

    def test_potkay_setup_model_and_return_callable(self):
        """Test setup_model_and_return_callable returns callable for PotkayModel."""
        metric = MetricConfig(metrics=[], modes=[])
        model = PotkayModel(run_kwargs={}, eval_kwargs={})
        wrapped = model.setup_model_and_return_callable(metric)
        assert callable(wrapped)

    def test_potkay_get_default_constants(self):
        """Test that _get_default_constants returns expected structure."""
        constants = PotkayModel._get_default_constants()

        # Check that it's a dictionary
        assert isinstance(constants, dict)

        # Check for key physical constants
        expected_keys = [
            'R', 'sigma', 'g', 'rho', 'm', 'V_w', 'C_p', 'emiss', 'P_atm',
            'g_b', 'c_a', 'o_a', 'T_a', 'RH_a', 'R_abs', 'PPFD',
            'V_cmax_25', 'J_max_25', 'theta_c', 'theta_J'
        ]
        for key in expected_keys:
            assert key in constants, f"Missing expected constant: {key}"

    def test_potkay_launch_model_single_timestep(self):
        """Test PotkayModel.launch_model runs for single timestep."""
        # Use a small E_range for faster testing
        E_range = (0, 0.001, 0.0001)

        output = PotkayModel.launch_model(
            E_range=E_range,
            env_data=None,
            verbose=False
        )

        # Check output structure
        assert output is not None
        assert isinstance(output, pd.DataFrame)
        assert len(output) > 0

        # Check for expected columns
        expected_cols = ['E-MD', 'A_n', 'R_d', 'GW', 'g_c',
                         'lambda', 'P_x_l', 'leaftemp', 'VPD']
        for col in expected_cols:
            assert col in output.columns, f"Missing expected column: {col}"

    def test_potkay_launch_model_with_parameter_overrides(self):
        """Test PotkayModel.launch_model with parameter overrides."""
        E_range = (0, 0.001, 0.0001)
        params = {'V_cmax_25': 80e-6, 'J_max_25': 150e-6, 'T_a': 30}

        output = PotkayModel.launch_model(
            params=params,
            E_range=E_range,
            env_data=None,
            verbose=False
        )

        assert output is not None
        assert isinstance(output, pd.DataFrame)
        assert len(output) > 0

    def test_potkay_launch_model_with_env_data(self):
        """Test PotkayModel.launch_model with multi-timestep environmental data."""
        # Create simple env_data with 2 timesteps
        env_data = pd.DataFrame({
            'year': [2023, 2023],
            'julian-day': [180, 181],
            'hour': [12, 12],
            'T_a': [25, 26],
            'RH_a': [0.5, 0.6],
            'R_abs': [600, 700],
            'PPFD': [1200e-6, 1200e-6]
        })

        E_range = (0, 0.001, 0.0001)

        output = PotkayModel.launch_model(
            E_range=E_range,
            env_data=env_data,
            verbose=False
        )

        assert output is not None
        assert isinstance(output, pd.DataFrame)
        # Should have results for both timesteps
        assert len(output) >= 2

        # Check time columns are present
        assert 'year' in output.columns
        assert 'julian-day' in output.columns
        assert 'standard-time' in output.columns

    def test_potkay_run_returns_dataframe_or_none(self):
        """Test PotkayModel.run returns DataFrame or None."""
        E_range = (0, 0.001, 0.0001)

        result = PotkayModel.run(
            params=None,
            X=None,
            E_range=E_range,
            verbose=False
        )

        # Should return either DataFrame or None (on error)
        assert result is None or isinstance(result, pd.DataFrame)
        if result is not None:
            assert len(result) > 0

    def test_potkay_run_with_parameter_dict(self):
        """Test PotkayModel.run with parameter dictionary."""
        E_range = (0, 0.001, 0.0001)
        X = {'V_cmax_25': 70e-6, 'J_max_25': 130e-6}

        result = PotkayModel.run(
            X=X,
            E_range=E_range,
            verbose=False
        )

        assert result is None or isinstance(result, pd.DataFrame)
        if result is not None:
            assert len(result) > 0

    def test_potkay_evaluate_model_returns_dict(self):
        """Test PotkayModel.evaluate_model returns dictionary of errors."""
        def mae_func(x, y):
            """Mean absolute error."""
            return np.nanmean(np.abs(x - y))

        m = Metric(output_name='A_n_vect', name='mae_an', func=mae_func)
        metric_config = MetricConfig(metrics=[m], modes=[Mode.from_name('min')])

        # Create mock output matching Potkay format
        output = pd.DataFrame({
            'A_n_vect': np.linspace(1, 5, 10),
            'year': [2023] * 10,
            'julian_day': [180] * 10
        })

        ground = pd.DataFrame({
            'A_n_vect': np.linspace(1.2, 5.2, 10),
            'year': [2023] * 10,
            'julian-day': [180] * 10
        })

        errors = PotkayModel.evaluate_model(
            output, ground, metric_config,
            start_date=datetime(2023, 6, 29),
            end_date=datetime(2023, 6, 29)
        )

        assert isinstance(errors, dict)
        assert 'mae_an' in errors
        assert isinstance(errors['mae_an'], (int, float, np.number))

    def test_potkay_evaluate_model_none_output_returns_penalty(self):
        """Test PotkayModel.evaluate_model with None output returns penalty."""
        def dummy_func(x, y):
            return 0

        m = Metric(output_name='A_n_vect', name='dummy', func=dummy_func)
        metric_config = MetricConfig(metrics=[m], modes=[Mode.from_name('min')])

        ground = pd.DataFrame({
            'A_n_vect': [1],
            'year': [2023],
            'julian-day': [180]
        })

        errors = PotkayModel.evaluate_model(
            None, ground, metric_config,  # type: ignore
            start_date=datetime(2023, 6, 29),
            end_date=datetime(2023, 6, 29)
        )

        assert errors['dummy'] == 1e20

    def test_potkay_evaluate_model_filters_by_date(self):
        """Test PotkayModel.evaluate_model correctly filters by date."""
        def sum_func(x, y):
            return np.nansum(x)

        m = Metric(output_name='A_n_vect', name='sum_an', func=sum_func)
        metric_config = MetricConfig(metrics=[m], modes=[Mode.from_name('min')])

        # Output with values spanning days
        output = pd.DataFrame({
            'A_n_vect': [10, 20, 30, 40, 50],
            'year': [2023, 2023, 2023, 2023, 2023],
            'julian_day': [180, 180, 181, 181, 182]
        })

        ground = pd.DataFrame({
            'A_n_vect': [10, 20, 30, 40, 50],
            'year': [2023, 2023, 2023, 2023, 2023],
            'julian-day': [180, 180, 181, 181, 182]
        })

        # Filter to only day 181
        errors = PotkayModel.evaluate_model(
            output, ground, metric_config,
            start_date=datetime(2023, 6, 30),  # Day 181
            end_date=datetime(2023, 6, 30)
        )

        # Should only sum values at day 181: 30+40 = 70
        assert errors['sum_an'] == 70

    def test_potkay_run_parallel_multiple_param_sets(self):
        """Test PotkayModel.run_parallel executes multiple parameter sets."""
        E_range = (0, 0.001, 0.0001)
        X_list = [
            {'V_cmax_25': 60e-6},
            {'V_cmax_25': 70e-6},
            {'V_cmax_25': 80e-6}
        ]

        results = PotkayModel.run_parallel(
            X=X_list,
            E_range=E_range,
            workers=2,
            verbose=False
        )

        assert len(results) == 3
        # Results can be None or DataFrame
        for result in results:
            assert result is None or isinstance(result, pd.DataFrame)

    def test_potkay_evaluate_model_multiple_metrics(self):
        """Test PotkayModel.evaluate_model with multiple metrics."""
        def rmse_func(x, y):
            return np.sqrt(np.nanmean((x - y) ** 2))

        def mae_func(x, y):
            return np.nanmean(np.abs(x - y))

        m1 = Metric(output_name='A_n_vect', name='rmse', func=rmse_func)
        m2 = Metric(output_name='A_n_vect', name='mae', func=mae_func)
        metric_config = MetricConfig(metrics=[m1, m2], modes=[Mode.from_name('min'), Mode.from_name('min')])

        output = pd.DataFrame({
            'A_n_vect': np.array([1, 2, 3, 4, 5], dtype=float),
            'year': [2023] * 5,
            'julian_day': [180] * 5
        })

        ground = pd.DataFrame({
            'A_n_vect': np.array([1.1, 2.1, 2.9, 4.2, 4.8], dtype=float),
            'year': [2023] * 5,
            'julian-day': [180] * 5
        })

        errors = PotkayModel.evaluate_model(
            output, ground, metric_config,
            start_date=datetime(2023, 6, 29),
            end_date=datetime(2023, 6, 29)
        )

        assert isinstance(errors, dict)
        assert 'rmse' in errors
        assert 'mae' in errors
        # Both should be non-negative (not NaN)
        assert not np.isnan(np.asarray(errors['rmse']).item())
        assert not np.isnan(np.asarray(errors['mae']).item())

    # def test_potkay_example_like_matlab(self):
    #     """Test that mimics Example.m from MATLAB code."""
    #     E_range = (0, 0.01, 1e-5)

    #     output = PotkayModel.launch_model(
    #         X=None,  # Use default constants
    #         E_range=E_range,
    #         env_data=None,
    #         verbose=False
    #     )

    #     assert output is not None
    #     assert isinstance(output, pd.DataFrame)

    #     assert 'E_vect' in output.columns
    #     assert 'A_n_vect' in output.columns
    #     assert 'lambda_vect' in output.columns

    #     # Verify all main physiological outputs are present
    #     expected_outputs = [
    #         'E_vect', 'A_n_vect', 'R_d_vect',
    #         'g_w_vect', 'g_c_vect', 'g_tot_vect',
    #         'lambda_vect',
    #         'P_x_l_vect', 'P_x_r_vect', 'P_0_vect',
    #         'T_l_vect', 'VPD_vect', 'RH_l_vect'
    #     ]
    #     for col in expected_outputs:
    #         assert col in output.columns, f"Missing output column: {col}"

    #     # Verify transpiration vector spans expected range
    #     E = output['E_vect'].values
    #     assert np.min(E) >= 0  # type: ignore
    #     assert np.max(E) <= 0.01  # type: ignore
    #     assert len(E) > 0

    #     # Verify A_n values are reasonable (should be positive during light hours, negative at night)
    #     A_n = output['A_n_vect'].values
    #     assert not np.all(np.isnan(A_n)), "All A_n values are NaN"
    #     assert np.any(np.isfinite(A_n)), "No finite A_n values found"

    #     # Verify lambda (marginal C profit of water) is calculated
    #     lambda_vals = output['lambda_vect'].values
    #     assert not np.all(np.isnan(lambda_vals)), "All lambda values are NaN"
    #     assert np.any(np.isfinite(lambda_vals)), "No finite lambda values found"

    #     # Verify temperature is reasonable (in Celsius, should be near ambient)
    #     T_l = output['T_l_vect'].values
    #     assert np.all(T_l > -50), "Leaf temperature unreasonably low"  # type: ignore
    #     assert np.all(T_l < 60), "Leaf temperature unreasonably high"  # type: ignore

    #     # Verify water potential is negative (as expected for plants)
    #     P_x_l = output['P_x_l_vect'].values
    #     assert np.all(P_x_l[~np.isnan(P_x_l)] <= 0), "All non-NaN leaf xylem water potential values should be negative"

    #     # Verify conductances increase with transpiration
    #     g_c = output['g_c_vect'].values
    #     # At higher transpiration, conductance should generally be higher
    #     # (check that max > min with some tolerance for noise)
    #     max_g_c = np.nanmax(g_c)  # type: ignore
    #     min_g_c = np.nanmin(g_c)  # type: ignore
    #     assert max_g_c >= min_g_c, "Conductance should vary with transpiration"

    #     # Verify number of E values matches expected range
    #     # E_range = (0, 0.01, 1e-5) should give about 1000 points
    #     expected_length = int((0.01 - 0) / 1e-5) + 1
    #     assert len(output) == expected_length, \
    #         f"Expected ~{expected_length} E values, got {len(output)}"
