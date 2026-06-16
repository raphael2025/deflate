"""Shared fixtures: synthetic return series with known properties."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def noise_returns(rng):
    """Pure random walk: ~3 years of zero-edge daily returns."""
    return rng.normal(0.0, 0.01, 756)


@pytest.fixture
def signal_returns(rng):
    """A series with a genuine, persistent positive edge.

    Daily mean ~0.0008 with 1% vol -> annual Sharpe ~ 1.27.
    """
    return rng.normal(0.0008, 0.01, 756)


@pytest.fixture
def overfit_grid(rng):
    """A grid of strategies on pure noise: best in-sample is luck.

    Returns an (T, N) matrix where every column is independent noise -- so
    in-sample ranking should carry no out-of-sample information (high PBO).
    """
    return rng.normal(0.0, 0.01, size=(756, 40))


@pytest.fixture
def real_signal_grid(rng):
    """A grid with a genuine quality gradient across configs (low PBO).

    Each column has its own *true* expected return spanning a wide range, so the
    in-sample best column is genuinely the best and keeps winning out of sample
    -- the situation CSCV is designed to reward with a low PBO.
    """
    n_configs = 40
    true_mu = np.linspace(-0.0010, 0.0010, n_configs)  # real, persistent edge
    idio = rng.normal(0.0, 0.008, size=(756, n_configs))
    return idio + true_mu
