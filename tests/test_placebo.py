import numpy as np
import pytest

from deflate import PlaceboResult, placebo_test


def test_placebo_ci_straddles_zero_when_no_edge(rng):
    sig = rng.normal(0.0, 0.01, 400)
    pla = rng.normal(0.0, 0.01, 2000)
    res = placebo_test(sig, placebo_returns=pla, n_boot=1000, rng=0)
    assert isinstance(res, PlaceboResult)
    assert res.diff_ci[0] < 0 < res.diff_ci[1]  # CI straddles 0
    assert not res.beats_placebo
    assert res.p_value > 0.05


def test_placebo_detects_real_edge(rng):
    sig = rng.normal(0.004, 0.01, 400)   # clearly positive forward returns
    pla = rng.normal(0.0, 0.01, 2000)    # random entries: no edge
    res = placebo_test(sig, placebo_returns=pla, n_boot=1000, rng=0)
    assert res.diff > 0
    assert res.beats_placebo            # CI excludes 0 on the positive side
    assert res.p_value < 0.05


def test_placebo_mask_mode(rng):
    fwd = rng.normal(0.0, 0.01, 1000)
    mask = np.zeros(1000, dtype=bool)
    mask[:300] = True
    fwd[:300] += 0.005  # the signalled entries have a real edge
    res = placebo_test(mask, forward_returns=fwd, n_boot=1000, rng=0)
    assert res.n_signal == 300
    assert res.n_placebo == 700
    assert res.diff > 0


def test_placebo_validation(rng):
    with pytest.raises(ValueError):
        placebo_test(rng.normal(size=100))  # no placebo or forward
    with pytest.raises(ValueError):
        placebo_test([0.1], placebo_returns=[0.1])  # too few obs
