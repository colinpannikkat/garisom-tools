import json
import os
import tempfile
from datetime import datetime

from garisom_tools.sa.config import (
    SensitivityAnalysisConfig,
    SensitivityAnalysisProblem
)


class TestSensitivityAnalysisConfigFromJson:
    """Tests for SensitivityAnalysisConfig.from_json date parsing."""

    def test_from_json_parses_date_strings(self):
        """Test that from_json parses YYYY-MM-DD date strings to datetime."""
        config_data = {
            "problem": {
                "num_vars": 2,
                "names": ["param1", "param2"],
                "bounds": [[0, 1], [10, 100]],
                "dists": ["unif", "unif"]
            },
            "metric": {
                "metrics": ["rmse"],
                "modes": ["min"],
                "params": ["output"]
            },
            "workers": 4,
            "samples": 64,
            "pop": 1,
            "start_date": "2023-07-20",
            "end_date": "2023-08-24"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = SensitivityAnalysisConfig.from_json(temp_path)

            assert isinstance(config.start_date, datetime)
            assert isinstance(config.end_date, datetime)
            assert config.start_date == datetime(2023, 7, 20)
            assert config.end_date == datetime(2023, 8, 24)
        finally:
            os.unlink(temp_path)

    def test_from_json_parses_different_dates(self):
        """Test parsing various date values."""
        config_data = {
            "problem": {
                "num_vars": 3,
                "names": ["a", "b", "c"],
                "bounds": [[0, 1], [0, 1], [0, 1]],
                "dists": None
            },
            "metric": {
                "metrics": ["nse"],
                "modes": ["max"],
                "params": ["flow"]
            },
            "workers": 8,
            "samples": 128,
            "pop": 2,
            "start_date": "2017-01-01",
            "end_date": "2018-12-31"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = SensitivityAnalysisConfig.from_json(temp_path)

            assert config.start_date == datetime(2017, 1, 1)
            assert config.end_date == datetime(2018, 12, 31)
            assert config.pop == 2
            assert config.workers == 8
            assert config.samples == 128
        finally:
            os.unlink(temp_path)

    def test_from_json_parses_problem_correctly(self):
        """Test that problem definition is parsed correctly."""
        config_data = {
            "problem": {
                "num_vars": 4,
                "names": ["temp", "humidity", "pressure", "wind"],
                "bounds": [[15, 35], [0.3, 0.9], [900, 1100], [0, 50]],
                "dists": ["unif", "norm", "truncnorm", "unif"]
            },
            "metric": {
                "metrics": ["rmse"],
                "modes": ["min"],
                "params": ["prediction"]
            },
            "workers": 4,
            "samples": 256,
            "pop": 1,
            "start_date": "2023-06-01",
            "end_date": "2023-09-30"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = SensitivityAnalysisConfig.from_json(temp_path)

            assert isinstance(config.problem, SensitivityAnalysisProblem)
            assert config.problem.num_vars == 4
            assert config.problem.names == ["temp", "humidity", "pressure", "wind"]
            assert len(config.problem.bounds) == 4
            assert config.problem.dists == ["unif", "norm", "truncnorm", "unif"]
        finally:
            os.unlink(temp_path)

    def test_from_json_parses_metric_correctly(self):
        """Test that metric configuration is parsed correctly."""
        config_data = {
            "problem": {
                "num_vars": 1,
                "names": ["x"],
                "bounds": [[0, 1]],
                "dists": None
            },
            "metric": {
                "metrics": ["rmse", "mse", "nse"],
                "modes": ["min", "min", "max"],
                "params": ["out1", "out2", "out3"]
            },
            "workers": 2,
            "samples": 32,
            "pop": 1,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = SensitivityAnalysisConfig.from_json(temp_path)

            assert config.metric is not None
            assert len(config.metric.metrics) == 3
            assert config.metric.modes == ["min", "min", "max"]
        finally:
            os.unlink(temp_path)


class TestSensitivityAnalysisProblem:
    """Tests for SensitivityAnalysisProblem class."""

    def test_to_dict_returns_correct_format(self):
        """Test that to_dict returns SALib-compatible format."""
        problem = SensitivityAnalysisProblem(
            num_vars=3,
            names=["x1", "x2", "x3"],
            bounds=[[0, 1], [0, 10], [-5, 5]],
            dists=["unif", "unif", "norm"]
        )

        result = problem.to_dict()

        assert result["num_vars"] == 3
        assert result["names"] == ["x1", "x2", "x3"]
        assert result["bounds"] == [[0, 1], [0, 10], [-5, 5]]
        assert result["dists"] == ["unif", "unif", "norm"]

    def test_to_dict_with_none_dists(self):
        """Test to_dict when dists is None."""
        problem = SensitivityAnalysisProblem(
            num_vars=2,
            names=["a", "b"],
            bounds=[[0, 1], [0, 1]],
            dists=None
        )

        result = problem.to_dict()

        assert result["dists"] is None
