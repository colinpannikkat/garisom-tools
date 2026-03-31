# garisom-tools

Assorted tools for the GARISOM model.

## Features
- Model wrapper for running in Python along with easy evaluation
  - Parallel execution with multiprocessing
- Bayesian parameter optimization using RayTune
- Easy configuration management
- Monte Carlo simulation
- Sensitivity analysis

## Installation

You can install this package using pip:

```bash
pip install .
```

Or using the new build system:

```bash
pip install --upgrade build
python -m build
```

## Usage

Import the package in your Python code:

```python
from garisom_tools import SperryModel
```

To run the Sperry model (Sperry et al. 2017), you must provide a model directory path. To download and build the model please see the [GARISOM repository](https://github.com/colinpannikkat/garisom).

### Example: Running the Model

Here's an example of how to run the Sperry model using the `run` method:

```python
import pandas as pd
from garisom_tools import SperryModel

# Example input parameters
X = {'i_leafAreaIndex': 4.61, 'i_kmaxTree': 345.}
params = pd.read_csv("./parameters.csv")  # Rest of parameters pulled from here
config_file = "./configuration.csv"
population = 1  # corresponds to row in configuration.csv
model_dir = "./garisom/02_program_code"

# Run the model
output = SperryModel.run(
    X=X,    # Optional argument, can run without passing in specific inputs
    params=params,
    config_file=config_file,
    population=population,
    model_dir=model_dir
)

print(output)
```

You can then evaluate against ground-truth data:
```python
from garisom_tools.utils.metric import Metric
from garisom_tools.config.metric import MetricConfig
from datetime import datetime

# Get ground data
ground = pd.read_csv("./ground.csv")

# Set up metric configuration
metric_config = MetricConfig([Metric.from_name('mse', 'P-PD')])

# Evaluate the model
errors = SperryModel.evaluate_model(
    output,
    ground,
    metric_config,
    start_date=datetime(2023, 7, 1),
    end_date=datetime(2023, 7, 31)
)

print(errors)
```

## Requirements
- numpy
- pandas
- scipy
- matplotlib
- scikit-learn
- ray[tune]
- tqdm

## License
MIT
