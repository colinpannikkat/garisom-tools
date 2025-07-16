from garisom_tools.utils.metric import Metric, Mode
from dataclasses import dataclass


@dataclass
class MetricConfig:
    metrics: list[Metric]
    modes: list[Mode]

    @classmethod
    def from_dict(cls, data: dict):
        params = data.get("params", [])
        metrics = data.get("metrics", [])
        metrics = [Metric.from_name(m, p) for m, p in zip(metrics, params)]
        modes = [Mode.from_name(m) for m in data.get("modes", [])]
        return cls(metrics=metrics, modes=modes)
