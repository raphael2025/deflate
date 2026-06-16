"""Probability of Backtest Overfitting (PBO) via CSCV.

Implements the Combinatorially-Symmetric Cross-Validation (CSCV) estimate of
the Probability of Backtest Overfitting from Bailey, Borwein, López de Prado &
Zhu (2017). Given a matrix of returns for *N* competing strategy configurations,
PBO measures how often the configuration that looks best **in sample** fails to
beat the median **out of sample**.

A PBO near 0.5 (or above) means in-sample ranking carries no out-of-sample
information -- the selection process is overfitting. A low PBO (say < 0.2) means
the in-sample winner tends to keep winning out of sample.

References
----------
Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017).
    "The Probability of Backtest Overfitting." Journal of Computational Finance,
    20(4), 39-69.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

__all__ = ["PBOResult", "pbo"]


@dataclass(frozen=True)
class PBOResult:
    """Result of a CSCV Probability of Backtest Overfitting computation.

    Attributes
    ----------
    pbo : float
        Probability of backtest overfitting in ``[0, 1]``: the fraction of
        train/test splits in which the in-sample best configuration ranks below
        the out-of-sample median. ``>= 0.5`` indicates overfitting.
    n_splits : int
        Number of symmetric combinatorial splits evaluated.
    n_configs : int
        Number of competing configurations (columns).
    median_logit : float
        Median of the logit-transformed out-of-sample ranks. Negative values
        correspond to overfitting (rank below median).
    logits : tuple[float, ...]
        The per-split logit values, for plotting the distribution.
    """

    pbo: float
    n_splits: int
    n_configs: int
    median_logit: float
    logits: tuple[float, ...]


def _as_matrix(returns_matrix) -> pd.DataFrame:
    if isinstance(returns_matrix, pd.DataFrame):
        M = returns_matrix.copy()
    else:
        M = pd.DataFrame(np.asarray(returns_matrix, dtype=float))
    M = M.dropna(how="any")
    if M.shape[1] < 2:
        raise ValueError("`returns_matrix` needs at least 2 configurations (columns).")
    if M.shape[0] < 4:
        raise ValueError("`returns_matrix` needs at least 4 time observations (rows).")
    return M


def _sharpe_cols(block: pd.DataFrame) -> pd.Series:
    """Per-column (per-period) Sharpe over a block of rows."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=0)
    return (mu / sd.replace(0.0, np.nan)).fillna(0.0)


def pbo(returns_matrix, n_splits: int = 16) -> PBOResult:
    r"""Probability of Backtest Overfitting via CSCV.

    The time axis is partitioned into ``S`` consecutive blocks. For every way of
    splitting the ``S`` blocks into equal in-sample / out-of-sample halves
    (there are ``C(S, S/2)`` of them):

    1. Rank configurations by in-sample Sharpe; take the best, ``n*``.
    2. Find the out-of-sample percentile rank ``\bar\omega`` of ``n*``.
    3. Map it to a logit ``\lambda = \ln(\bar\omega / (1 - \bar\omega))``.

    PBO is the fraction of splits with ``\lambda <= 0`` -- i.e. the in-sample
    winner landed in the bottom half out of sample.

    Parameters
    ----------
    returns_matrix : pandas.DataFrame or 2-D array-like
        Shape ``(T, N)``: periodic returns for ``N`` competing configurations
        over the same ``T`` periods (e.g. one column per parameter setting from
        a grid search). Rows with any NaN are dropped.
    n_splits : int, default 16
        Number of consecutive time blocks ``S``. Must be even and ``>= 4``.
        Larger ``S`` gives more combinatorial splits (``C(S, S/2)``) and a
        smoother estimate, at higher cost. ``S=16`` yields 12870 splits.

    Returns
    -------
    PBOResult

    Notes
    -----
    Per López de Prado, ``S`` should be even so the in/out split is balanced.
    The number of splits grows combinatorially; keep ``S <= 16`` unless you
    really need it.
    """
    if n_splits < 4 or n_splits % 2 != 0:
        raise ValueError("`n_splits` must be an even integer >= 4.")
    M = _as_matrix(returns_matrix)
    T, N = M.shape
    if n_splits > T:
        raise ValueError(
            f"`n_splits`={n_splits} exceeds the {T} available time observations."
        )

    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    logits: list[float] = []

    for train_blocks in combinations(range(n_splits), half):
        test_blocks = [j for j in range(n_splits) if j not in train_blocks]
        tr_idx = np.concatenate([blocks[j] for j in train_blocks])
        te_idx = np.concatenate([blocks[j] for j in test_blocks])

        sr_in = _sharpe_cols(M.iloc[tr_idx])
        sr_out = _sharpe_cols(M.iloc[te_idx])

        best = sr_in.idxmax()
        # Out-of-sample percentile rank of the in-sample best (1 = top).
        rank = float(sr_out.rank(pct=True)[best])
        rank = min(max(rank, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(rank / (1.0 - rank))))

    logits_arr = np.asarray(logits)
    pbo_value = float(np.mean(logits_arr <= 0.0))
    return PBOResult(
        pbo=pbo_value,
        n_splits=len(logits),
        n_configs=N,
        median_logit=float(np.median(logits_arr)),
        logits=tuple(logits),
    )
