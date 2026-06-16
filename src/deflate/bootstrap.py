"""Block and stationary bootstrap of Sharpe and drawdown distributions.

Resampling the return series in *blocks* preserves short-horizon
autocorrelation, so the resulting Sharpe and maximum-drawdown distributions are
realistic. From them we read:

  * a confidence interval on the Sharpe ratio,
  * ``P(SR <= 0)`` -- the probability the edge is actually noise,
  * the **realistic worst-case drawdown** (5th / 1st percentile of the maxDD
    distribution), which is almost always worse than the single historical maxDD.

References
----------
Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
    Journal of the American Statistical Association, 89(428), 1303-1313.
Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary
    Observations." Annals of Statistics, 17(3), 1217-1241.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BootResult", "bootstrap_metrics", "block_bootstrap", "stationary_bootstrap"]


@dataclass(frozen=True)
class BootResult:
    """Result of a bootstrap of Sharpe and drawdown distributions.

    All Sharpe figures are annualised by ``sqrt(periods_per_year)``. Drawdowns
    are expressed as fractions (e.g. ``-0.25`` = a 25% peak-to-trough decline)
    on a cumulative-sum (additive) equity curve.

    Attributes
    ----------
    sharpe : float
        Observed annualised Sharpe ratio of the input series.
    sharpe_mean : float
        Mean of the bootstrap Sharpe distribution.
    sharpe_ci : tuple[float, float]
        ``(5th, 95th)`` percentile confidence interval on the Sharpe.
    p_sharpe_le_0 : float
        Probability that the Sharpe is ``<= 0`` across bootstrap resamples.
    max_drawdown : float
        Observed maximum drawdown of the input series (fraction, negative).
    max_drawdown_median : float
        Median bootstrap maximum drawdown.
    max_drawdown_p05 : float
        5th-percentile (i.e. a bad-but-plausible) bootstrap maximum drawdown.
    max_drawdown_p01 : float
        1st-percentile (near worst-case) bootstrap maximum drawdown.
    n_boot : int
        Number of bootstrap resamples.
    method : str
        ``"stationary"`` or ``"block"``.
    """

    sharpe: float
    sharpe_mean: float
    sharpe_ci: tuple[float, float]
    p_sharpe_le_0: float
    max_drawdown: float
    max_drawdown_median: float
    max_drawdown_p05: float
    max_drawdown_p01: float
    n_boot: int
    method: str


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 4:
        raise ValueError("`returns` must contain at least 4 finite observations.")
    return r


def _sharpe(r: np.ndarray, ann: float) -> float:
    sd = r.std(ddof=0)
    return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0


def _max_drawdown(r: np.ndarray) -> float:
    """Maximum drawdown of an additive (cumsum) equity curve, as a fraction."""
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


def stationary_bootstrap(
    returns,
    n_boot: int = 2000,
    mean_block: float = 5.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    r"""Stationary bootstrap resampling (Politis & Romano, 1994).

    Builds ``n_boot`` resampled index sequences of the same length as the input.
    Block lengths are geometric with mean ``mean_block`` (continuation
    probability ``1 - 1/mean_block``), wrapping around the series end. This is
    the recommended bootstrap for serially dependent return series.

    Parameters
    ----------
    returns : array-like
        1-D periodic return series.
    n_boot : int, default 2000
        Number of resamples.
    mean_block : float, default 5.0
        Mean block length (in periods). Must be ``>= 1``.
    rng : numpy.random.Generator | int | None
        Random source / seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_boot, T)`` array of resampled return series.
    """
    if mean_block < 1:
        raise ValueError("`mean_block` must be >= 1.")
    r = _clean(returns)
    T = r.size
    gen = np.random.default_rng(rng)
    p = 1.0 / mean_block
    out = np.empty((n_boot, T), dtype=float)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        i = gen.integers(T)
        cont = gen.random(T) >= p  # True -> continue current block
        starts = gen.integers(0, T, size=T)  # new start when a block breaks
        for k in range(T):
            idx[k] = i
            i = (i + 1) % T if cont[k] else int(starts[k])
        out[b] = r[idx]
    return out


def block_bootstrap(
    returns,
    n_boot: int = 2000,
    block: int = 5,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    r"""Moving-block bootstrap resampling (Künsch, 1989).

    Resamples fixed-length contiguous blocks of size ``block`` with replacement
    and concatenates them to the original length. Simpler than the stationary
    bootstrap but with a fixed block size.

    Parameters
    ----------
    returns : array-like
        1-D periodic return series.
    n_boot : int, default 2000
        Number of resamples.
    block : int, default 5
        Block length in periods (``>= 1``).
    rng : numpy.random.Generator | int | None
        Random source / seed.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_boot, T)`` array of resampled return series.
    """
    if block < 1:
        raise ValueError("`block` must be >= 1.")
    r = _clean(returns)
    T = r.size
    gen = np.random.default_rng(rng)
    n_blocks = int(np.ceil(T / block))
    max_start = max(1, T - block + 1)
    out = np.empty((n_boot, T), dtype=float)
    for b in range(n_boot):
        starts = gen.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        idx = np.minimum(idx, T - 1)
        out[b] = r[idx]
    return out


def bootstrap_metrics(
    returns,
    n_boot: int = 2000,
    block: int = 5,
    method: str = "stationary",
    periods_per_year: int = 252,
    rng: np.random.Generator | int | None = None,
) -> BootResult:
    """Bootstrap the Sharpe and maximum-drawdown distributions of a strategy.

    Parameters
    ----------
    returns : array-like
        1-D periodic return series (e.g. daily strategy returns).
    n_boot : int, default 2000
        Number of bootstrap resamples.
    block : int, default 5
        Block length (block bootstrap) or mean block length (stationary).
    method : {"stationary", "block"}, default "stationary"
        Resampling scheme.
    periods_per_year : int, default 252
        Annualisation factor for the Sharpe ratio (use 365 for crypto daily).
    rng : numpy.random.Generator | int | None
        Random source / seed for reproducibility.

    Returns
    -------
    BootResult
    """
    r = _clean(returns)
    if method == "stationary":
        samples = stationary_bootstrap(r, n_boot=n_boot, mean_block=block, rng=rng)
    elif method == "block":
        samples = block_bootstrap(r, n_boot=n_boot, block=block, rng=rng)
    else:
        raise ValueError("`method` must be 'stationary' or 'block'.")

    sds = samples.std(axis=1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(
            sds > 0, samples.mean(axis=1) / sds * np.sqrt(periods_per_year), 0.0
        )
    eq = np.cumsum(samples, axis=1)
    running_max = np.maximum.accumulate(eq, axis=1)
    mdds = (eq - running_max).min(axis=1)

    return BootResult(
        sharpe=_sharpe(r, periods_per_year),
        sharpe_mean=float(sharpes.mean()),
        sharpe_ci=(float(np.percentile(sharpes, 5)), float(np.percentile(sharpes, 95))),
        p_sharpe_le_0=float(np.mean(sharpes <= 0.0)),
        max_drawdown=_max_drawdown(r),
        max_drawdown_median=float(np.median(mdds)),
        max_drawdown_p05=float(np.percentile(mdds, 5)),
        max_drawdown_p01=float(np.percentile(mdds, 1)),
        n_boot=n_boot,
        method=method,
    )
