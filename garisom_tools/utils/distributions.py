"""
# Custom Probability Distributions

This module provides custom probability distributions compatible with Optuna
for use in optimization and Monte Carlo simulations. It includes normal and
truncated normal distributions as well as utility functions for SciPy distributions.

## Classes

- `NormalDistribution`: Normal distribution for Optuna optimization
- `TruncatedNormalDistribution`: Truncated normal distribution for Optuna
- `FloatDistribution`: Re-exported from Optuna for convenience

## Functions

- `get_scipy_truncated_normal`: Create SciPy truncated normal distribution
- `get_scipy_normal`: Create SciPy normal distribution
- `get_scipy_uniform`: Create SciPy uniform distribution

## Example Usage

```python
from garisom_tools.utils.distributions import (
    NormalDistribution, TruncatedNormalDistribution,
    get_scipy_normal
)
import numpy as np

# Create custom distributions for optimization
normal_dist = NormalDistribution(mu=0.5, sigma=0.1)
truncated_dist = TruncatedNormalDistribution(mu=0.5, sigma=0.1, a=0.0, b=1.0)

# Use with optimization
search_space = {
    'growth_rate': normal_dist,
    'water_efficiency': truncated_dist
}

# Create SciPy distributions for Monte Carlo
scipy_normal = get_scipy_normal(loc=0.5, scale=0.1)
samples = scipy_normal.rvs(size=1000)
```
"""

from scipy.stats import (
    truncnorm,
    norm,
    uniform
)
from optuna.distributions import FloatDistribution
from scipy.special import ndtri


class NormalDistribution(FloatDistribution):
    """
    Normal distribution compatible with Optuna optimization.

    This class implements a normal (Gaussian) distribution that can be used
    in Optuna optimization studies. It internally uses inverse transform
    sampling to convert uniform random variables to normal samples.

    Attributes:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        low (float): Lower bound for internal uniform sampling (1e-8).
        high (float): Upper bound for internal uniform sampling (1-1e-8).

    Example:
        ```python
        # Create normal distribution
        normal_dist = NormalDistribution(mu=0.5, sigma=0.1)

        # Use in optimization search space
        search_space = {
            'parameter1': normal_dist,
            'parameter2': NormalDistribution(mu=1.0, sigma=0.2)
        }

        # Sample values
        rng = np.random.default_rng(42)
        sample = normal_dist._sample(rng)
        ```

    Note:
        The distribution uses inverse transform sampling where uniform
        samples in [0,1] are transformed to normal samples using the
        inverse cumulative distribution function (quantile function).
    """
    def __init__(self, mu=0.0, sigma=1.0):
        """
        Initialize the normal distribution.

        Args:
            mu (float, optional): Mean of the distribution. Defaults to 0.0.
            sigma (float, optional): Standard deviation. Defaults to 1.0.
        """
        self.mu = mu
        self.sigma = sigma
        # For internal uniform sampling, map [low, high] to [0, 1]
        self.low = 1e-8
        self.high = 1 - 1e-8
        super().__init__(self.low, self.high)

    def single(self) -> bool:
        """Check if distribution represents a single value."""
        return False

    def _contains(self, param):
        """Check if parameter is within valid range."""
        return isinstance(param, float) and self.low <= param <= self.high

    def _sample(self, rng):
        """Sample from the normal distribution using inverse transform."""
        # Sample p ~ Uniform(0,1)
        p = rng.uniform(self.low, self.high)
        # Transform to normal
        return self.mu + self.sigma * ndtri(p)

    def to_internal_repr(self, param):
        """Transform normal value to internal uniform representation."""
        # Transform normal value back to uniform p for internal sampler state
        p = norm.cdf(param, loc=self.mu, scale=self.sigma)
        return p

    def to_external_repr(self, internal_param):
        """Transform internal uniform value to normal value."""
        # Transform uniform p to normal value
        return self.mu + self.sigma * ndtri(internal_param)


class TruncatedNormalDistribution(FloatDistribution):
    """
    Truncated normal distribution compatible with Optuna optimization.

    This class implements a normal distribution truncated to a finite interval
    [a, b]. It's useful when parameters have physical constraints or when
    you want to limit the search space while maintaining a smooth distribution.

    Attributes:
        mu (float): Mean of the underlying normal distribution.
        sigma (float): Standard deviation of the underlying normal distribution.
        a (float): Lower truncation bound.
        b (float): Upper truncation bound.

    Example:
        ```python
        # Create truncated normal for efficiency parameter (0 to 1)
        efficiency_dist = TruncatedNormalDistribution(
            mu=0.8,        # Prefer higher efficiency
            sigma=0.1,     # Some uncertainty
            a=0.0,         # Minimum efficiency
            b=1.0          # Maximum efficiency
        )

        # Use in optimization
        search_space = {
            'water_efficiency': efficiency_dist,
            'light_efficiency': TruncatedNormalDistribution(0.5, 0.15, 0.0, 1.0)
        }
        ```

    Note:
        - Values outside [a, b] have zero probability
        - The distribution is properly normalized within the truncation bounds
        - Mean and variance differ from the underlying normal distribution
    """
    def __init__(self, mu=0.0, sigma=1.0, a=1e-12, b=1e12):
        """
        Initialize the truncated normal distribution.

        Args:
            mu (float, optional): Mean of underlying normal. Defaults to 0.0.
            sigma (float, optional): Standard deviation of underlying normal. Defaults to 1.0.
            a (float, optional): Lower truncation bound. Defaults to 1e-12.
            b (float, optional): Upper truncation bound. Defaults to 1e12.
        """
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b

        # Calculate standardized bounds for truncnorm
        self._a_std = (a - mu) / sigma
        self._b_std = (b - mu) / sigma

        # Internal uniform sampling bounds in (0,1)
        self.low = 1e-8
        self.high = 1 - 1e-8

        super().__init__(self.low, self.high)

    def single(self) -> bool:
        """Check if distribution represents a single value."""
        return False

    def _contains(self, param):
        """Check if parameter is within truncation bounds."""
        return isinstance(param, float) and self.a <= param <= self.b

    def _sample(self, rng):
        """Sample from truncated normal using inverse CDF."""
        # Uniform p in [low, high]
        p = rng.uniform(self.low, self.high)
        # Sample from truncated normal using inverse CDF
        return truncnorm.ppf(p, self._a_std, self._b_std, loc=self.mu, scale=self.sigma)

    def to_internal_repr(self, param):
        """Transform truncated normal value to internal uniform representation."""
        # Map external value to internal uniform p
        p = truncnorm.cdf(param, self._a_std, self._b_std, loc=self.mu, scale=self.sigma)
        return min(max(p, self.low), self.high)

    def to_external_repr(self, internal_param):
        """Transform internal uniform value to truncated normal value."""
        # Map internal uniform p to truncated normal value
        return truncnorm.ppf(internal_param, self._a_std, self._b_std, loc=self.mu, scale=self.sigma)


def get_scipy_truncated_normal(loc=0.0, scale=1.0, a=1e-12, b=1e12):
    """
    Create a SciPy truncated normal distribution.

    Convenience function for creating truncated normal distributions
    for use in Monte Carlo simulations or other applications requiring
    SciPy distribution objects.

    Args:
        loc (float, optional): Mean of the underlying normal. Defaults to 0.0.
        scale (float, optional): Standard deviation. Defaults to 1.0.
        a (float, optional): Lower truncation bound. Defaults to 1e-12.
        b (float, optional): Upper truncation bound. Defaults to 1e12.

    Returns:
        scipy.stats.truncnorm: Configured truncated normal distribution.

    Example:
        ```python
        # Create truncated normal for Monte Carlo sampling
        dist = get_scipy_truncated_normal(loc=0.5, scale=0.1, a=0.0, b=1.0)
        
        # Generate samples
        samples = dist.rvs(size=1000)
        
        # Compute statistics
        mean = dist.mean()
        var = dist.var()
        ```
    """
    a_scaled = (a - loc) / scale
    b_scaled = (b - loc) / scale
    return truncnorm(a=a_scaled, b=b_scaled, loc=loc, scale=scale)


def get_scipy_normal(loc=0.0, scale=1.0):
    """
    Create a SciPy normal distribution.

    Convenience function for creating normal distributions for use in
    Monte Carlo simulations or other statistical applications.

    Args:
        loc (float, optional): Mean of the distribution. Defaults to 0.0.
        scale (float, optional): Standard deviation. Defaults to 1.0.

    Returns:
        scipy.stats.norm: Configured normal distribution.

    Example:
        ```python
        # Create normal distribution
        dist = get_scipy_normal(loc=0.5, scale=0.1)
        
        # Generate samples
        samples = dist.rvs(size=1000)
        
        # Compute PDF at specific points
        pdf_values = dist.pdf([0.4, 0.5, 0.6])
        ```
    """
    return norm(loc=loc, scale=scale)


def get_scipy_uniform(a=0.0, b=1.0):
    """
    Create a SciPy uniform distribution.

    Convenience function for creating uniform distributions over the
    interval [a, b] for use in Monte Carlo simulations.

    Args:
        a (float, optional): Lower bound of the interval. Defaults to 0.0.
        b (float, optional): Upper bound of the interval. Defaults to 1.0.

    Returns:
        scipy.stats.uniform: Configured uniform distribution.

    Example:
        ```python
        # Create uniform distribution over [0.1, 0.9]
        dist = get_scipy_uniform(a=0.1, b=0.9)
        
        # Generate samples
        samples = dist.rvs(size=1000)
        
        # All samples will be in [0.1, 0.9]
        assert all(0.1 <= x <= 0.9 for x in samples)
        ```

    Note:
        SciPy's uniform distribution is parameterized as uniform(loc, scale)
        where scale = b - a, so we transform the [a, b] interface accordingly.
    """
    return uniform(loc=a, scale=b - a)
