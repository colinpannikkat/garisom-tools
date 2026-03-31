import json
import os
import tempfile
from datetime import datetime

from garisom_tools.optimization.config import (
    OptimizationConfig,
    GarisomOptimizationConfig
)


class TestGarisomOptimizationConfigFromJson:
    """Tests for GarisomOptimizationConfig.from_json date parsing."""

    def test_from_json_parses_date_strings(self):
        """Test that from_json parses YYYY-MM-DD date strings to datetime."""
        config_data = {
            "space": {
                "param1": ["uniform", [0, 1]]
            },
            "metric": {
                "metrics": ["rmse"],
                "modes": ["min"],
                "params": ["output"]
            },
            "num_worker": 4,
            "num_samples": 100,
            "population": 1,
            "start_date": "2023-07-20",
            "end_date": "2023-08-24"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = GarisomOptimizationConfig.from_json(temp_path)

            assert isinstance(config.start_date, datetime)
            assert isinstance(config.end_date, datetime)
            assert config.start_date == datetime(2023, 7, 20)
            assert config.end_date == datetime(2023, 8, 24)
        finally:
            os.unlink(temp_path)

    def test_from_json_parses_different_dates(self):
        """Test parsing various date formats."""
        config_data = {
            "space": {
                "param1": ["uniform", [0, 1]]
            },
            "metric": {
                "metrics": ["rmse"],
                "modes": ["min"],
                "params": ["output"]
            },
            "num_worker": 4,
            "num_samples": 100,
            "population": 2,
            "start_date": "2017-01-15",
            "end_date": "2018-12-31"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = GarisomOptimizationConfig.from_json(temp_path)

            assert config.start_date == datetime(2017, 1, 15)
            assert config.end_date == datetime(2018, 12, 31)
            assert config.population == 2
        finally:
            os.unlink(temp_path)

    def test_from_json_inherits_parent_parsing(self):
        """Test that parent class parsing (space, metric) still works."""
        config_data = {
            "space": {
                "learning_rate": ["uniform", [0.001, 0.1]],
                "batch_size": ["uniform", [16, 128]]
            },
            "metric": {
                "metrics": ["rmse", "mse"],
                "modes": ["min", "min"],
                "params": ["pred", "pred"]
            },
            "num_worker": 8,
            "num_samples": 200,
            "population": 1,
            "start_date": "2023-06-01",
            "end_date": "2023-09-30"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = GarisomOptimizationConfig.from_json(temp_path)

            assert config.num_worker == 8
            assert config.num_samples == 200
            assert config.space is not None
            assert config.metric is not None
            assert len(config.metric.metrics) == 2
        finally:
            os.unlink(temp_path)


class TestOptimizationConfigFromJson:
    """Tests for base OptimizationConfig.from_json."""

    def test_from_json_loads_basic_config(self):
        """Test that base config loads correctly."""
        config_data = {
            "space": {
                "param1": ["uniform", [0, 10]]
            },
            "metric": {
                "metrics": ["nse"],
                "modes": ["max"],
                "params": ["flow"]
            },
            "num_worker": 2,
            "num_samples": 50
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = OptimizationConfig.from_json(temp_path)

            assert config.num_worker == 2
            assert config.num_samples == 50
            assert config.space is not None
            assert config.metric is not None
        finally:
            os.unlink(temp_path)
