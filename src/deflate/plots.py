"""Optional plotting helpers (require matplotlib).

These degrade gracefully: if matplotlib is not installed, importing
``deflate.plots`` still works, but calling a plot function raises a clear
``ImportError`` telling you to ``pip install deflate[plots]``.
"""

from __future__ import annotations

import numpy as np

from .bootstrap import bootstrap_metrics, stationary_bootstrap

__all__ = ["plot_bootstrap_sharpe", "plot_equity_curve", "plot_pbo"]


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "matplotlib is required for plotting. Install it with "
            "`pip install deflate[plots]` or `pip install matplotlib`."
        ) from exc
    return plt


def plot_bootstrap_sharpe(
    returns,
    n_boot: int = 2000,
    block: int = 5,
    periods_per_year: int = 252,
    ax=None,
    rng=None,
):
    """Histogram of the bootstrap Sharpe distribution with the observed Sharpe.

    Returns the matplotlib ``Axes``.
    """
    plt = _require_mpl()
    samples = stationary_bootstrap(returns, n_boot=n_boot, mean_block=block, rng=rng)
    sds = samples.std(axis=1, ddof=0)
    sharpes = np.where(
        sds > 0, samples.mean(axis=1) / sds * np.sqrt(periods_per_year), 0.0
    )
    res = bootstrap_metrics(
        returns, n_boot=n_boot, block=block, periods_per_year=periods_per_year, rng=rng
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sharpes, bins=60, color="#4c72b0", alpha=0.8)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.axvline(res.sharpe, color="#c44e52", lw=2, label=f"observed SR={res.sharpe:.2f}")
    ax.axvline(res.sharpe_ci[0], color="black", ls=":", lw=1, label="90% CI")
    ax.axvline(res.sharpe_ci[1], color="black", ls=":", lw=1)
    ax.set_title(f"Bootstrap Sharpe distribution (P(SR<=0)={res.p_sharpe_le_0:.2f})")
    ax.set_xlabel("annualised Sharpe")
    ax.set_ylabel("count")
    ax.legend()
    return ax


def plot_equity_curve(returns, ax=None):
    """Cumulative (additive) equity curve with the running peak shaded.

    Returns the matplotlib ``Axes``.
    """
    plt = _require_mpl()
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eq, color="#4c72b0", lw=1.5, label="equity")
    ax.fill_between(np.arange(len(eq)), eq, peak, color="#c44e52", alpha=0.25,
                    label="drawdown")
    ax.set_title("Equity curve (cumulative return)")
    ax.set_xlabel("period")
    ax.set_ylabel("cumulative return")
    ax.legend()
    return ax


def plot_pbo(pbo_result, ax=None):
    """Histogram of the CSCV logit distribution from a :class:`PBOResult`.

    Returns the matplotlib ``Axes``.
    """
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    logits = np.asarray(pbo_result.logits)
    ax.hist(logits, bins=40, color="#55a868", alpha=0.8)
    ax.axvline(0, color="#c44e52", lw=2, label="overfit boundary (logit=0)")
    ax.set_title(f"CSCV logit distribution (PBO={pbo_result.pbo:.2f})")
    ax.set_xlabel("out-of-sample logit of in-sample best")
    ax.set_ylabel("count")
    ax.legend()
    return ax
