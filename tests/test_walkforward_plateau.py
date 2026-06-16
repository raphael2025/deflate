import numpy as np
import pytest

from deflate import (
    PlateauResult,
    WFResult,
    parameter_plateau,
    walk_forward,
)


def test_walk_forward_stable_signal(signal_returns):
    res = walk_forward(signal_returns, n_folds=5)
    assert isinstance(res, WFResult)
    assert res.n_folds == 5
    assert res.frac_positive >= 0.6  # a real edge shows up in most folds


def test_walk_forward_unstable_noise(noise_returns):
    res = walk_forward(noise_returns, n_folds=5)
    # Pure noise: folds flip sign, low consistency.
    assert 0.0 <= res.frac_positive <= 1.0
    assert abs(res.consistency) < 5.0


def test_walk_forward_validation():
    with pytest.raises(ValueError):
        walk_forward(np.random.randn(100), n_folds=1)
    with pytest.raises(ValueError):
        walk_forward(np.random.randn(3), n_folds=5)


def test_plateau_detects_broad_peak():
    # A smooth quadratic surface: the peak's neighbours are nearly as good.
    x = np.linspace(-3, 3, 21)
    scores = 2.0 - 0.05 * x**2  # broad plateau around 0
    res = parameter_plateau(scores, min_ratio=0.5)
    assert isinstance(res, PlateauResult)
    assert res.is_plateau
    assert res.neighbor_ratio > 0.9


def test_plateau_flags_lonely_spike():
    scores = np.full(21, 0.1)
    scores[10] = 3.0  # one giant spike surrounded by mediocrity
    res = parameter_plateau(scores, min_ratio=0.5)
    assert not res.is_plateau
    assert res.neighbor_ratio < 0.5


def test_plateau_2d():
    xx, yy = np.meshgrid(np.linspace(-2, 2, 11), np.linspace(-2, 2, 11))
    scores = 1.0 - 0.1 * (xx**2 + yy**2)
    res = parameter_plateau(scores)
    assert res.is_plateau
    assert len(res.best_index) == 2
