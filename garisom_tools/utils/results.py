from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import json
import os


EvalResults = dict[np.ndarray]


@dataclass
class MetricResult:
    scores: dict[str, float]
    parameters: dict[str, float]

    def to_dict(self): return asdict(self)


class ParamResults(dict[str, MetricResult]):
    def to_dict(self):
        return {k: v.to_dict() for k, v in self.items()}

    def to_json(self, outfile: str):
        with open(outfile, "+x") as f:
            json.dump(self.to_dict(), f)


class StatsResults(dict[str, pd.DataFrame]):

    def save(self, directory: str):
        [
            data.to_csv(
                os.path.join(directory, f"{stat}.csv"),
                index=False
            ) for stat, data in self.items()
        ]