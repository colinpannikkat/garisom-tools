from scipy.stats import (
    truncnorm,
    norm,
    uniform
)
from optuna.distributions import FloatDistribution
from scipy.special import ndtri


class NormalDistribution(FloatDistribution):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        # For internal uniform sampling, map [low, high] to [0, 1]
        self.low = 1e-8
        self.high = 1 - 1e-8
        super().__init__(self.low, self.high)

    def single(self) -> bool:
        return False

    def _contains(self, param):
        return isinstance(param, float) and self.low <= param <= self.high

    def _sample(self, rng):
        # Sample p ~ Uniform(0,1)
        p = rng.uniform(self.low, self.high)
        # Transform to normal
        return self.mu + self.sigma * ndtri(p)

    def to_internal_repr(self, param):
        # Transform normal value back to uniform p for internal sampler state
        p = norm.cdf(param, loc=self.mu, scale=self.sigma)
        return p

    def to_external_repr(self, internal_param):
        # Transform uniform p to normal value
        return self.mu + self.sigma * ndtri(internal_param)


class TruncatedNormalDistribution(FloatDistribution):
    def __init__(self, mu=0.0, sigma=1.0, a=1e-12, b=1e12):
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
        return False

    def _contains(self, param):
        return isinstance(param, float) and self.a <= param <= self.b

    def _sample(self, rng):
        # Uniform p in [low, high]
        p = rng.uniform(self.low, self.high)
        # Sample from truncated normal using inverse CDF
        return truncnorm.ppf(p, self._a_std, self._b_std, loc=self.mu, scale=self.sigma)

    def to_internal_repr(self, param):
        # Map external value to internal uniform p
        p = truncnorm.cdf(param, self._a_std, self._b_std, loc=self.mu, scale=self.sigma)
        return min(max(p, self.low), self.high)

    def to_external_repr(self, internal_param):
        # Map internal uniform p to truncated normal value
        return truncnorm.ppf(internal_param, self._a_std, self._b_std, loc=self.mu, scale=self.sigma)


def get_scipy_truncated_normal(loc=0.0, scale=1.0, a=1e-12, b=1e12):
    a_scaled = (a - loc) / scale
    b_scaled = (b - loc) / scale
    return truncnorm(a=a_scaled, b=b_scaled, loc=loc, scale=scale)


def get_scipy_normal(loc=0.0, scale=1.0):
    return norm(loc=loc, scale=scale)


def get_scipy_uniform(a=0.0, b=1.0):
    return uniform(loc=a, scale=b - a)
