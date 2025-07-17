import pandas as pd
import numpy as np
from garisom_tools.model import GarisomModel, Model
from garisom_tools.config import MetricConfig
from garisom_tools.utils.metric import Metric


def test_dummy():
    assert True


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


def test_run_parallel_returns_list(monkeypatch):
    def dummy_run(X, *args, **kwargs):
        return pd.DataFrame({'a': [1]})

    monkeypatch.setattr(GarisomModel, "run", staticmethod(dummy_run))
    X = [{'x': 1}, {'x': 2}]
    params = pd.DataFrame({'x': [1, 2]})
    res = GarisomModel.run_parallel(X, params, 'config.cfg', 1, 'model_dir', workers=2)
    assert isinstance(res, list)
    assert all(isinstance(r, pd.DataFrame) for r in res if r is not None)
