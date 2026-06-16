import numpy as np
import pytest

from deflate import BootResult, block_bootstrap, bootstrap_metrics, stationary_bootstrap


def test_bootstrap_noise_high_p_sr_le_0(noise_returns):
    res = bootstrap_metrics(noise_returns, n_boot=1000, rng=0)
    assert isinstance(res, BootResult)
    # No edge -> Sharpe straddles 0 -> substantial mass at/below 0.
    assert res.p_sharpe_le_0 > 0.2
    assert res.sharpe_ci[0] < 0 < res.sharpe_ci[1]


def test_bootstrap_signal_low_p_sr_le_0(signal_returns):
    res = bootstrap_metrics(signal_returns, n_boot=1000, rng=0)
    # Real edge -> Sharpe reliably positive.
    assert res.p_sharpe_le_0 < 0.1
    assert res.sharpe_ci[0] > -0.2


def test_bootstrap_worst_dd_worse_than_observed(signal_returns):
    res = bootstrap_metrics(signal_returns, n_boot=1000, rng=0)
    # The 1% worst-case drawdown should be at least as bad as observed.
    assert res.max_drawdown_p01 <= res.max_drawdown + 1e-9


def test_resampler_shapes(noise_returns):
    s = stationary_bootstrap(noise_returns, n_boot=50, mean_block=5, rng=1)
    b = block_bootstrap(noise_returns, n_boot=50, block=5, rng=1)
    assert s.shape == (50, len(noise_returns))
    assert b.shape == (50, len(noise_returns))
    # Resamples are drawn from the original values.
    assert set(np.unique(s)).issubset(set(np.unique(noise_returns)))


def test_reproducible(signal_returns):
    a = bootstrap_metrics(signal_returns, n_boot=500, rng=42)
    b = bootstrap_metrics(signal_returns, n_boot=500, rng=42)
    assert a.sharpe_mean == b.sharpe_mean
    assert a.p_sharpe_le_0 == b.p_sharpe_le_0


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        bootstrap_metrics([0.1, 0.2], n_boot=10)  # too few obs
    with pytest.raises(ValueError):
        bootstrap_metrics(np.random.randn(100), method="nope")
