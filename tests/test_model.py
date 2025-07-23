import pandas as pd
import numpy as np
import os
from garisom_tools.model import GarisomModel, Model
from garisom_tools.config import MetricConfig
from garisom_tools.utils.metric import Metric


model_dir = "../garisom/02_program_code"


def get_parameter_and_configuration_files() -> tuple[str, pd.DataFrame]:
    return os.path.abspath("./tests/data/configuration.csv"), \
        pd.read_csv("./tests/data/parameters.csv")


config_file, params = get_parameter_and_configuration_files()


def test_model_inheritance():
    assert issubclass(GarisomModel, Model)


def test_get_objective_returns_callable():
    model = GarisomModel(run_kwargs={'a': 1}, eval_kwargs={})
    obj = model.get_objective()
    assert callable(obj)


def test_setup_model_and_return_callable_returns_callable():
    metric = MetricConfig(metrics=[], modes=[])
    model = GarisomModel(run_kwargs={}, eval_kwargs={})
    wrapped = model.setup_model_and_return_callable(metric)
    assert callable(wrapped)


def test_evaluate_model_returns_dict():

    def dummy_func(x, y):
        return np.sum(np.abs(x - y))

    m = Metric(output_name='val', name='dummy', func=dummy_func)

    metric_config = MetricConfig(metrics=[m], modes=['min'])
    output = pd.DataFrame({'val': [1, 2, 3]})
    ground = pd.DataFrame({'julian-day': [1, 2, 3], 'val': [1, 2, 4]})
    errors = GarisomModel.evaluate_model(output, ground, metric_config, 1, 3)
    assert isinstance(errors, dict)
    assert 'dummy' in errors


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

#     model = GarisomModel()
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

#     model = GarisomModel()
#     out = model.run(**run_kwargs)

#     assert out is not None
