"""Deflated and Probabilistic Sharpe Ratio.

Implements the Probabilistic Sharpe Ratio (PSR) and the Deflated Sharpe Ratio
(DSR) of Bailey & López de Prado. The DSR corrects an observed Sharpe ratio for

  * non-normal returns (skewness and excess kurtosis),
  * track-record length,
  * **selection bias from multiple trials** -- the single biggest source of
    backtest overfitting.

References
----------
Bailey, D. H., & López de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting, and Non-Normality." Journal of Portfolio Management, 40(5).
Bailey, D. H., & López de Prado, M. (2012).
    "The Sharpe Ratio Efficient Frontier." Journal of Risk, 15(2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, skew

__all__ = ["DSRResult", "PSRResult", "deflated_sharpe", "probabilistic_sharpe"]

_EULER_MASCHERONI = 0.5772156649015329


def _clean(returns) -> np.ndarray:
    """Coerce input to a 1-D float array of finite returns, with validation."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("`returns` must contain at least 2 finite observations.")
    if r.std(ddof=0) == 0:
        raise ValueError("`returns` has zero standard deviation; Sharpe is undefined.")
    return r


def _sharpe_per_period(r: np.ndarray) -> float:
    """Non-annualised Sharpe ratio (population std, matching López de Prado)."""
    return float(r.mean() / r.std(ddof=0))


@dataclass(frozen=True)
class PSRResult:
    """Result of a Probabilistic Sharpe Ratio computation.

    Attributes
    ----------
    sharpe : float
        Observed (per-period) Sharpe ratio.
    sharpe_annual : float
        Observed Sharpe ratio annualised by ``sqrt(periods_per_year)``.
    benchmark_sharpe : float
        Per-period benchmark Sharpe ``sr*`` against which significance is tested.
    psr : float
        Probability that the true Sharpe exceeds the benchmark, in ``[0, 1]``.
    n_obs : int
        Number of return observations used.
    skew : float
        Sample skewness of returns.
    kurtosis : float
        Sample (non-excess) kurtosis of returns.
    """

    sharpe: float
    sharpe_annual: float
    benchmark_sharpe: float
    psr: float
    n_obs: int
    skew: float
    kurtosis: float


@dataclass(frozen=True)
class DSRResult:
    """Result of a Deflated Sharpe Ratio computation.

    Attributes
    ----------
    sharpe : float
        Observed (per-period) Sharpe ratio.
    sharpe_annual : float
        Observed Sharpe ratio annualised by ``sqrt(periods_per_year)``.
    dsr : float
        Deflated Sharpe Ratio in ``[0, 1]`` -- the probability the strategy's
        true Sharpe is positive after deflating for ``n_trials`` and
        non-normality. Conventionally significant when ``dsr > 0.95``.
    expected_max_sharpe : float
        The benchmark ``SR_0`` -- the per-period Sharpe one expects as the
        maximum across ``n_trials`` purely random strategies.
    n_trials : int
        Number of independent trials / configurations searched.
    n_obs : int
        Number of return observations used.
    skew : float
        Sample skewness of returns.
    kurtosis : float
        Sample (non-excess) kurtosis of returns.
    """

    sharpe: float
    sharpe_annual: float
    dsr: float
    expected_max_sharpe: float
    n_trials: int
    n_obs: int
    skew: float
    kurtosis: float

    @property
    def is_significant(self) -> bool:
        """True if ``dsr`` clears the conventional 0.95 threshold."""
        return self.dsr > 0.95


def probabilistic_sharpe(
    returns,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 252,
) -> PSRResult:
    r"""Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    Estimates the probability that the *true* Sharpe ratio exceeds a benchmark
    ``sr*``, accounting for track-record length and the non-normality of
    returns (skewness ``\gamma_3`` and kurtosis ``\gamma_4``):

    .. math::

        \widehat{PSR}(sr^*) = \Phi\!\left(
            \frac{(\widehat{SR} - sr^*)\sqrt{T-1}}
                 {\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}}
        \right)

    where ``SR`` and ``sr*`` are expressed in the **same (per-period)**
    frequency as ``returns``.

    Parameters
    ----------
    returns : array-like
        1-D series of periodic returns (e.g. daily). NaNs/infs are dropped.
    benchmark_sharpe : float, default 0.0
        Benchmark Sharpe ``sr*``. Pass an **annualised** value here; it is
        converted internally to per-period using ``periods_per_year``.
    periods_per_year : int, default 252
        Periods per year used for annualisation (252 trading days, 365 for
        crypto daily, 12 for monthly, etc.).

    Returns
    -------
    PSRResult

    Notes
    -----
    Uses population standard deviation (``ddof=0``) for the Sharpe estimate, as
    in López de Prado's reference implementation.
    """
    r = _clean(returns)
    T = r.size
    sr = _sharpe_per_period(r)
    sk = float(skew(r))
    ku = float(kurtosis(r, fisher=False))  # non-excess kurtosis
    sr_star = float(benchmark_sharpe) / np.sqrt(periods_per_year)

    denom = np.sqrt(max(1.0 - sk * sr + (ku - 1.0) / 4.0 * sr**2, 1e-12))
    psr = float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / denom))
    return PSRResult(
        sharpe=sr,
        sharpe_annual=sr * np.sqrt(periods_per_year),
        benchmark_sharpe=sr_star,
        psr=psr,
        n_obs=T,
        skew=sk,
        kurtosis=ku,
    )


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    r"""Expected maximum Sharpe across ``n_trials`` random strategies.

    This is the benchmark ``SR_0`` used to deflate the Sharpe ratio. Under the
    null that all trials have a true Sharpe of zero, the expected maximum of the
    ``N`` estimated Sharpes is approximated (Bailey & López de Prado, 2014) by

    .. math::

        E[\max_N] \approx \sqrt{V} \left[
            (1-\gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
            + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)
        \right]

    where ``V`` is the variance of the Sharpe estimates across trials and
    ``\gamma`` is the Euler-Mascheroni constant.

    Parameters
    ----------
    n_trials : int
        Number of independent trials. Must be >= 2 (with one trial there is no
        selection bias and ``SR_0 = 0``).
    sr_variance : float
        Variance of the Sharpe estimates across the trials.

    Returns
    -------
    float
        The per-period benchmark Sharpe ``SR_0``.
    """
    if n_trials < 1:
        raise ValueError("`n_trials` must be >= 1.")
    if n_trials == 1:
        return 0.0
    if sr_variance < 0:
        raise ValueError("`sr_variance` must be non-negative.")
    g = _EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe(
    returns,
    n_trials: int,
    sr_variance: float | None = None,
    periods_per_year: int = 252,
) -> DSRResult:
    r"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    The DSR is the Probabilistic Sharpe Ratio evaluated against a benchmark
    ``SR_0`` equal to the *expected maximum* Sharpe one would obtain from
    ``n_trials`` random strategies. It answers: **"after accounting for how many
    configurations I tried, for the length of my track record, and for fat
    tails, what is the probability that the true Sharpe is positive?"**

    A value below 0.95 means the observed Sharpe is plausibly an artefact of
    multiple testing -- i.e. the backtest is likely overfit.

    Parameters
    ----------
    returns : array-like
        1-D series of periodic returns (e.g. daily). NaNs/infs are dropped.
    n_trials : int
        Number of independent strategy configurations searched/tested to arrive
        at this one (grid points, parameter combos, variants). Be honest: this
        is the lever that exposes overfitting. With ``n_trials=1`` the DSR
        reduces to ``PSR(0)``.
    sr_variance : float, optional
        Variance of the Sharpe estimates across the trials, used to compute
        ``SR_0``. If ``None`` (default), it is estimated from the asymptotic
        variance of a single Sharpe under normality, ``(1 + 0.5 * SR^2) / T``,
        which is the standard fallback when the cross-trial dispersion is
        unknown. Provide the empirical variance of your trial Sharpes when you
        have it for a tighter estimate.
    periods_per_year : int, default 252
        Periods per year used only for reporting the annualised Sharpe.

    Returns
    -------
    DSRResult

    Notes
    -----
    Uses population standard deviation (``ddof=0``) for the Sharpe estimate.
    The per-period Sharpe and ``SR_0`` are compared in the same frequency, so
    annualisation does not affect the DSR -- only the reported numbers.
    """
    if n_trials < 1:
        raise ValueError("`n_trials` must be >= 1.")
    r = _clean(returns)
    T = r.size
    sr = _sharpe_per_period(r)
    sk = float(skew(r))
    ku = float(kurtosis(r, fisher=False))

    if sr_variance is None:
        sr_variance = (1.0 + 0.5 * sr**2) / T
    sr0 = expected_max_sharpe(n_trials, sr_variance)

    denom = np.sqrt(max(1.0 - sk * sr + (ku - 1.0) / 4.0 * sr**2, 1e-12))
    dsr = float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))
    return DSRResult(
        sharpe=sr,
        sharpe_annual=sr * np.sqrt(periods_per_year),
        dsr=dsr,
        expected_max_sharpe=sr0,
        n_trials=n_trials,
        n_obs=T,
        skew=sk,
        kurtosis=ku,
    )
