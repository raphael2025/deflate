#!/usr/bin/env python3
"""10-line quickstart: catch an overfit backtest with deflate.verdict()."""

import numpy as np
import deflate

rng = np.random.default_rng(0)

# A grid search over 200 configs on pure noise; we keep the best-looking one.
grid = rng.normal(0.0, 0.01, size=(750, 200))
best = grid[:, grid[:375].mean(0).argmax()]   # best in-sample Sharpe = luck

# One line tells you how badly you fooled yourself:
print(deflate.verdict(best, n_trials=200, returns_matrix=grid))
