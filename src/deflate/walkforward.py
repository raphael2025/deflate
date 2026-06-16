"""Walk-forward (out-of-sample) evaluation.

Splits a return series into consecutive folds and reports the per-fold Sharpe.
A robust strategy keeps a positive, stable Sharpe across folds; one that decays
fold-over-fold (or only worked early) is a red flag for overfitting or regime
dependence.

This is an *anchored* split of an already-realised return series -- it does not
re-fit parameters. For full parameter re-optimisation use it as a scaffold and
plug your own fit/predict per fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WFResult", "walk_forward"]


@dataclass(frozen=True)
class WFResult:
    """Result of a walk-forward evaluation.

    Attributes
    ----------
    fold_sharpes : tuple[float, ...]
        Annualised Sharpe ratio of each consecutive fold.
    fold_returns : tuple[float, ...]
        Total (summed) return of each fold.
    n_folds : int
        Number of folds.
    frac_positive : float
        Fraction of folds with a positive Sharpe.
    mean_sharpe : float
        Mean of the fold Sharpes.
    std_sharpe : float
        Standard deviation of the fold Sharpes (lower = more stable).
    consistency : float
        ``mean_sharpe / std_sharpe`` style stability score (information-ratio of
        the fold Sharpes); higher is more consistent. ``inf`` if all folds equal.
    """

    fold_sharpes: tuple[float, ...]
    fold_returns: tuple[float, ...]
    n_folds: int
    frac_positive: float
    mean_sharpe: float
    std_sharpe: float
    consistency: float


def walk_forward(
    returns,
    n_folds: int = 5,
    periods_per_year: int = 252,
) -> WFResult:
    """Evaluate a return series across consecutive out-of-sample folds.

    Parameters
    ----------
    returns : array-like
        1-D periodic return series.
    n_folds : int, default 5
        Number of consecutive (non-overlapping) folds. Must be ``>= 2`` and no
        larger than the number of observations.
    periods_per_year : int, default 252
        Annualisation factor for the per-fold Sharpe ratios.

    Returns
    -------
    WFResult
    """
    if n_folds < 2:
        raise ValueError("`n_folds` must be >= 2.")
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < n_folds * 2:
        raise ValueError(
            f"Need at least {n_folds * 2} observations for {n_folds} folds; "
            f"got {r.size}."
        )

    folds = np.array_split(r, n_folds)
    sharpes: list[float] = []
    totals: list[float] = []
    for f in folds:
        sd = f.std(ddof=0)
        sharpes.append(
            float(f.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
        )
        totals.append(float(f.sum()))

    arr = np.asarray(sharpes)
    std = float(arr.std(ddof=0))
    mean = float(arr.mean())
    consistency = float(mean / std) if std > 0 else float("inf")
    return WFResult(
        fold_sharpes=tuple(sharpes),
        fold_returns=tuple(totals),
        n_folds=n_folds,
        frac_positive=float(np.mean(arr > 0)),
        mean_sharpe=mean,
        std_sharpe=std,
        consistency=consistency,
    )
