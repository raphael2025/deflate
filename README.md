# deflate

**Your backtest is probably overfit. `deflate` tells you how badly.**

[![CI](https://github.com/raphael2025/deflate/actions/workflows/ci.yml/badge.svg)](https://github.com/raphael2025/deflate/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`deflate` is a small, dependency-light **backtest lie detector**. You hand it
your strategy's return series (and, honestly, how many configurations you
tried), and it runs the statistically rigorous overfitting checks that almost
no backtesting framework runs for you — then hands back a one-line verdict.

It implements the published methods of Bailey & López de Prado (Deflated Sharpe
Ratio, Probability of Backtest Overfitting) plus block-bootstrap drawdown
distributions, a placebo/permutation test, walk-forward stability and parameter
plateau detection.

---

## Install

```bash
pip install deflate
# optional plots:
pip install "deflate[plots]"
```

## Quickstart

Grid-search a "strategy" out of **pure noise**, keep the best-looking one, and
watch `deflate` catch it red-handed:

```python
import numpy as np
import deflate

rng = np.random.default_rng(0)

# 200 configs of pure noise; keep the one with the best in-sample Sharpe.
grid = rng.normal(0.0, 0.01, size=(750, 200))
best = grid[:, grid[:375].mean(0).argmax()]   # in-sample winner = luck

# One line tells you how badly you fooled yourself:
print(deflate.verdict(best, n_trials=200, returns_matrix=grid))
```

```
============================================================
  deflate verdict:  LIKELY OVERFIT
  confidence:       100%
============================================================
  Deflated Sharpe : DSR=0.50  (annual SR=1.10, n_trials=200)
  Bootstrap       : P(SR<=0)=0.36  SR 90% CI=[-1.16, 1.27]
                    realistic worst DD (1%): -29.0%  (observed -10.1%)
  Walk-forward    : 40% of folds positive  (mean fold SR=0.21)
  PBO (CSCV)      : 0.55
============================================================
  Why:
    - Deflated Sharpe is 0.50 (< 0.95): after correcting for 200 trial(s)...
    - PBO is 0.55 (> 0.50): the in-sample best configuration tends to
      underperform out of sample.
    - ...
============================================================
```

(Run `python examples/demo_catch_overfit.py` for the full, contrasted demo.)

---

## What each check defends against

| Check | Function | What it catches |
|-------|----------|-----------------|
| **Deflated Sharpe Ratio** | `deflated_sharpe(returns, n_trials)` | **Multiple testing.** Corrects the Sharpe for how many configs you tried, your track-record length, and fat tails. DSR < 0.95 → the Sharpe is plausibly luck. |
| **Probabilistic Sharpe** | `probabilistic_sharpe(returns, benchmark_sharpe)` | Whether the true Sharpe clears a benchmark, given non-normal returns. |
| **PBO (CSCV)** | `pbo(returns_matrix)` | **In-sample optimisation that breaks out of sample.** Across all symmetric train/test splits, does the in-sample winner stay above the median out of sample? PBO ≈ 0.5+ → overfit. |
| **Bootstrap** | `bootstrap_metrics(returns)` | **Fragile Sharpe & understated drawdowns.** Block-resamples the series for a Sharpe CI, `P(SR ≤ 0)`, and the *realistic* worst-case drawdown (usually far worse than the single historical max). |
| **Placebo test** | `placebo_test(signal, ...)` | **Signals that don't beat chance.** Compares signalled entries to matched random entries; if the difference CI straddles 0, your "edge" is noise. |
| **Walk-forward** | `walk_forward(returns)` | **Time-concentrated performance / regime dependence.** Sharpe stability across consecutive folds. |
| **Parameter plateau** | `parameter_plateau(grid_scores)` | **Lonely spikes.** A trustworthy optimum sits on a plateau; a spike surrounded by mediocre neighbours is a fit to noise. |
| **Verdict** | `verdict(returns, n_trials, ...)` | Runs the whole battery and returns a single `is_overfit` ruling with reasons. |

---

## Why this exists

The large majority of published and home-cooked backtests are overfit, and most
backtesting tools won't tell you — they happily report a Sharpe of 2.5 from a
grid search over thousands of parameter sets without ever deflating it for the
selection bias that produced it. The single most under-reported number in
quant is **how many things you tried before you found this one**.

`deflate` was extracted from a real crypto research pipeline where these exact
checks were used to **falsify an entire suite of "profitable" crypto and equity
strategies** — predictive models, copy-trading and wallet-alpha signals that all
collapsed once the Deflated Sharpe and PBO were applied honestly. The statistics
here are the ones that did the falsifying.

> The goal of `deflate` is not to make you feel good. It's to stop you trading
> a backtest that was never real.

---

## API at a glance

```python
import deflate

deflate.deflated_sharpe(returns, n_trials, periods_per_year=252)  -> DSRResult
deflate.probabilistic_sharpe(returns, benchmark_sharpe=0.0)       -> PSRResult
deflate.pbo(returns_matrix, n_splits=16)                          -> PBOResult
deflate.bootstrap_metrics(returns, n_boot=2000, block=5)          -> BootResult
deflate.placebo_test(signal_returns, placebo_returns=...)         -> PlaceboResult
deflate.walk_forward(returns, n_folds=5)                          -> WFResult
deflate.parameter_plateau(grid_scores)                            -> PlateauResult
deflate.verdict(returns, n_trials=1, returns_matrix=None, ...)    -> Verdict
```

Every result is a typed, frozen `dataclass`; `Verdict.__str__` prints the
human-readable ruling shown above. Optional plotting lives in `deflate.plots`
(`plot_bootstrap_sharpe`, `plot_equity_curve`, `plot_pbo`).

All Sharpe-based functions take a `periods_per_year` argument — use `252` for
daily equities, `365` for daily crypto, `12` for monthly.

---

## References

- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.*
  Journal of Portfolio Management, 40(5).
- Bailey, D. H., & López de Prado, M. (2012). *The Sharpe Ratio Efficient
  Frontier.* Journal of Risk, 15(2).
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017).
  *The Probability of Backtest Overfitting.* Journal of Computational Finance,
  20(4).
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap.* JASA,
  89(428).

## License

MIT — see [LICENSE](LICENSE).
