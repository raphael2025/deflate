#!/usr/bin/env python3
"""Catch an overfit strategy red-handed.

We generate PURE NOISE (no edge whatsoever), then "discover" a strategy by
grid-searching 300 parameter combinations and keeping the best-looking one.
Its in-sample Sharpe looks great -- and `deflate` exposes it as a fluke.

For contrast we then run a strategy with a genuine edge through the same gauntlet
and watch it pass.

Run:  python examples/demo_catch_overfit.py
"""

from __future__ import annotations

import numpy as np

import deflate


def grid_search_on_noise(rng, n_days=1000, n_configs=300):
    """Grid-search a 'strategy' out of pure noise.

    Returns ``(reported_returns, full_grid, oos_grid)`` where ``reported_returns``
    is the *in-sample* track record of the winning config -- exactly the curve a
    naive quant would proudly show. The full grid drives the PBO check.
    """
    grid = rng.normal(0.0, 0.01, size=(n_days, n_configs))
    split = n_days // 2
    in_sample = grid[:split]
    # Pick the config with the best IN-SAMPLE Sharpe -- the cardinal sin.
    sr = in_sample.mean(axis=0) / in_sample.std(axis=0)
    best = int(np.argmax(sr))
    # The quant reports the in-sample curve (this is what gets you excited).
    return grid[:split, best], grid, grid[split:]


def main() -> None:
    rng = np.random.default_rng(7)

    print("\n" + "#" * 60)
    print("#  CASE 1: a strategy grid-searched out of PURE NOISE")
    print("#" * 60)
    reported, grid, _ = grid_search_on_noise(rng)
    naive_sharpe = reported.mean() / reported.std() * np.sqrt(252)
    print(f"\nReported (in-sample) annualised Sharpe of the 'winner': "
          f"{naive_sharpe:.2f}  (looks tradable, right?)\n")

    # We searched 300 configs -> n_trials=300; the grid drives PBO.
    v = deflate.verdict(reported, n_trials=grid.shape[1],
                        returns_matrix=grid, n_splits=10, rng=0)
    print(v)
    assert v.is_overfit, "demo failed: should have flagged the noise strategy!"
    print("\n>>> deflate caught the overfit strategy "
          "(DSR collapses once you admit 300 trials; PBO confirms it).\n")

    print("\n" + "#" * 60)
    print("#  CASE 2: a strategy with a GENUINE edge (one honest trial)")
    print("#" * 60)
    real = rng.normal(0.0008, 0.01, 756)  # ~1.27 annual Sharpe, real drift
    real_sharpe = real.mean() / real.std() * np.sqrt(252)
    print(f"\nAnnualised Sharpe: {real_sharpe:.2f}\n")
    v2 = deflate.verdict(real, n_trials=1, rng=0)
    print(v2)
    print("\n>>> deflate clears the genuine strategy.\n")


if __name__ == "__main__":
    main()
