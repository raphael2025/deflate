import numpy as np
import pandas as pd
import pytest

from deflate import PBOResult, pbo


def test_pbo_high_on_noise_grid(overfit_grid):
    res = pbo(overfit_grid, n_splits=10)
    assert isinstance(res, PBOResult)
    # Independent noise columns: in-sample best is luck -> PBO near/above 0.5.
    assert res.pbo > 0.35
    assert res.n_configs == 40


def test_pbo_low_on_shared_real_edge(real_signal_grid):
    res = pbo(real_signal_grid, n_splits=10)
    # When all configs share a real edge, the in-sample winner holds up OOS.
    assert res.pbo < 0.4


def test_pbo_accepts_dataframe(overfit_grid):
    df = pd.DataFrame(overfit_grid, columns=[f"cfg{i}" for i in range(40)])
    res = pbo(df, n_splits=8)
    assert 0.0 <= res.pbo <= 1.0
    assert len(res.logits) == res.n_splits


def test_pbo_validation(overfit_grid):
    with pytest.raises(ValueError):
        pbo(overfit_grid, n_splits=7)  # odd
    with pytest.raises(ValueError):
        pbo(overfit_grid, n_splits=2)  # < 4
    with pytest.raises(ValueError):
        pbo(overfit_grid[:, :1])  # single config
