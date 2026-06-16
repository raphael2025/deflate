import numpy as np
import pytest

from deflate import (
    DSRResult,
    PSRResult,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)


def test_dsr_low_on_noise_with_many_trials(noise_returns):
    res = deflated_sharpe(noise_returns, n_trials=200)
    assert isinstance(res, DSRResult)
    # Pure noise searched over 200 trials must NOT look significant.
    assert res.dsr < 0.95
    assert not res.is_significant
    assert res.expected_max_sharpe > 0  # selection bias raises the bar


def test_dsr_high_on_real_signal(signal_returns):
    res = deflated_sharpe(signal_returns, n_trials=1)
    # A real ~1.3 Sharpe edge with one trial should clear the bar comfortably.
    assert res.dsr > 0.95
    assert res.is_significant


def test_dsr_monotonic_in_trials(signal_returns):
    few = deflated_sharpe(signal_returns, n_trials=1).dsr
    many = deflated_sharpe(signal_returns, n_trials=5000).dsr
    # More trials -> higher bar -> lower (or equal) DSR.
    assert many <= few


def test_dsr_n_trials_one_equals_psr0(signal_returns):
    dsr = deflated_sharpe(signal_returns, n_trials=1).dsr
    psr = probabilistic_sharpe(signal_returns, benchmark_sharpe=0.0).psr
    assert dsr == pytest.approx(psr, abs=1e-9)


def test_psr_in_unit_interval(signal_returns):
    res = probabilistic_sharpe(signal_returns)
    assert isinstance(res, PSRResult)
    assert 0.0 <= res.psr <= 1.0


def test_expected_max_sharpe_increases_with_trials():
    v = 0.01
    assert expected_max_sharpe(1, v) == 0.0
    assert expected_max_sharpe(1000, v) > expected_max_sharpe(10, v) > 0


def test_input_validation():
    with pytest.raises(ValueError):
        deflated_sharpe([0.01], n_trials=10)  # too few obs
    with pytest.raises(ValueError):
        deflated_sharpe(np.zeros(100), n_trials=10)  # zero variance
    with pytest.raises(ValueError):
        deflated_sharpe(np.random.randn(100), n_trials=0)  # bad n_trials
