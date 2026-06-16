import numpy as np

from deflate import Verdict, verdict


def test_verdict_flags_overfit_noise(noise_returns, overfit_grid):
    v = verdict(noise_returns, n_trials=200, returns_matrix=overfit_grid,
                n_boot=800, n_splits=8, rng=0)
    assert isinstance(v, Verdict)
    assert v.is_overfit
    assert v.confidence > 0.0
    assert v.reasons
    assert "dsr" in v.metrics and "bootstrap" in v.metrics
    assert "pbo" in v.metrics
    # __str__ produces a readable ruling.
    s = str(v)
    assert "LIKELY OVERFIT" in s
    assert "Why:" in s


def test_verdict_passes_real_signal(signal_returns):
    v = verdict(signal_returns, n_trials=1, n_boot=800, rng=0)
    assert not v.is_overfit
    assert "PLAUSIBLY ROBUST" in str(v)


def test_verdict_with_placebo(rng):
    returns = rng.normal(0.0008, 0.01, 600)
    sig = rng.normal(0.004, 0.01, 300)
    pla = rng.normal(0.0, 0.01, 1500)
    v = verdict(returns, n_trials=1, signal_returns=sig, placebo_returns=pla,
                n_boot=800, rng=0)
    assert v.placebo is not None
    assert v.placebo.beats_placebo
