import pandas as pd
import numpy as np
import os
from datetime import datetime
from garisom_tools.model import SperryModel, Model
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


# def test_run_with_X():

#     run_kwargs = {
#         "params": params,
#         "config_file": config_file,
#         "population": 1,
#         "model_dir": os.path.abspath(model_dir),
#         "verbose": True,
#         "out": None,
#         "err": None
#     }

#     model = SperryModel()
#     out = model.run(X={'i_leafAreaIndex': 2.5}, **run_kwargs)

#     assert out is not None


# def test_run_no_X():

#     run_kwargs = {
#         "params": params,
#         "config_file": config_file,
#         "population": 1,
#         "model_dir": os.path.abspath(model_dir),
#         "verbose": True,
#         "out": None,
#         "err": None
#     }

#     model = SperryModel()
#     out = model.run(**run_kwargs)

#     assert out is not None
