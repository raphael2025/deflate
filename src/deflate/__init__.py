"""deflate -- the backtest lie detector.

Your backtest is probably overfit. ``deflate`` tells you how badly.

A small, dependency-light toolkit of statistically rigorous overfitting checks
for trading-strategy backtests:

* :func:`deflated_sharpe` / :func:`probabilistic_sharpe` -- correct the Sharpe
  ratio for multiple testing, track-record length and fat tails
  (Bailey & López de Prado, 2012, 2014).
* :func:`pbo` -- Probability of Backtest Overfitting via CSCV
  (Bailey, Borwein, López de Prado & Zhu, 2017).
* :func:`bootstrap_metrics` -- block / stationary bootstrap of the Sharpe and
  the realistic worst-case drawdown distribution.
* :func:`placebo_test` -- does the signal actually beat random entries?
* :func:`walk_forward` -- out-of-sample stability across folds.
* :func:`parameter_plateau` -- is the chosen parameter a plateau or a spike?
* :func:`verdict` -- run the whole battery and get a one-line ruling.

Quick start
-----------
>>> import numpy as np, deflate
>>> rng = np.random.default_rng(0)
>>> returns = rng.normal(0.0003, 0.01, 750)   # ~3 years of daily noise
>>> v = deflate.verdict(returns, n_trials=200) # we searched 200 configs
>>> print(v)                                   # doctest: +SKIP
"""

from __future__ import annotations

from .bootstrap import (
    BootResult,
    block_bootstrap,
    bootstrap_metrics,
    stationary_bootstrap,
)
from .pbo import PBOResult, pbo
from .placebo import PlaceboResult, placebo_test
from .plateau import PlateauResult, parameter_plateau
from .sharpe import (
    DSRResult,
    PSRResult,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)
from .verdict import Verdict, verdict
from .walkforward import WFResult, walk_forward

__version__ = "0.1.0"

__all__ = [
    # sharpe
    "deflated_sharpe",
    "probabilistic_sharpe",
    "expected_max_sharpe",
    "DSRResult",
    "PSRResult",
    # pbo
    "pbo",
    "PBOResult",
    # bootstrap
    "bootstrap_metrics",
    "stationary_bootstrap",
    "block_bootstrap",
    "BootResult",
    # placebo
    "placebo_test",
    "PlaceboResult",
    # walk-forward
    "walk_forward",
    "WFResult",
    # plateau
    "parameter_plateau",
    "PlateauResult",
    # verdict
    "verdict",
    "Verdict",
    "__version__",
]
