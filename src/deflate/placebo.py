"""Placebo / permutation test: does the signal beat random entries?

A strategy that fires on a "signal" only has an edge if its trades do better
than trades taken at random under the *same conditions*. The placebo test
compares the mean forward return of signalled entries against a pool of matched
random entries (the "placebo"), and bootstraps the difference to get a
confidence interval and a permutation p-value.

If the event-minus-placebo difference's CI straddles zero, the signal is
indistinguishable from chance -- a classic overfit tell.

This mirrors the event-study methodology of de-meaning against a matched
control sample, with a block-bootstrap CI on the difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PlaceboResult", "placebo_test"]


@dataclass(frozen=True)
class PlaceboResult:
    """Result of a placebo / permutation test.

    Attributes
    ----------
    signal_mean : float
        Mean forward return of signalled entries.
    placebo_mean : float
        Mean forward return of the random (placebo) pool.
    diff : float
        ``signal_mean - placebo_mean`` -- the excess return attributable to the
        signal.
    diff_ci : tuple[float, float]
        ``(5th, 95th)`` percentile bootstrap CI on ``diff``. If it straddles 0,
        the signal is not distinguishable from random.
    p_value : float
        Permutation p-value: probability of a difference at least as large as
        observed under the null of no signal (two-sided).
    n_signal : int
        Number of signalled entries.
    n_placebo : int
        Number of placebo entries.
    win_rate : float
        Fraction of signalled entries with a positive forward return.

    """

    signal_mean: float
    placebo_mean: float
    diff: float
    diff_ci: tuple[float, float]
    p_value: float
    n_signal: int
    n_placebo: int
    win_rate: float

    @property
    def beats_placebo(self) -> bool:
        """True if the difference CI excludes zero on the positive side."""
        return self.diff_ci[0] > 0.0


def _block_bootstrap_mean(
    x: np.ndarray, n_boot: int, block: int, gen: np.random.Generator
) -> np.ndarray:
    """Bootstrap distribution of the mean of ``x`` using moving blocks."""
    n = x.size
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = gen.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = np.minimum(idx, n - 1)
        means[b] = x[idx].mean()
    return means


def placebo_test(
    signal_returns,
    placebo_returns=None,
    forward_returns=None,
    n_boot: int = 2000,
    block: int = 5,
    rng: np.random.Generator | int | None = None,
) -> PlaceboResult:
    """Test whether signalled entries beat matched random entries.

    There are two ways to call this:

    1. **Pre-computed forward returns** (recommended). Pass ``signal_returns``
       (forward returns of the signalled trades) and ``placebo_returns``
       (forward returns of matched random trades). The placebo pool should be
       drawn under the same conditions (time-of-day, regime, side, ...) as the
       real entries so the comparison is fair.

    2. **Boolean signal + forward returns**. Pass ``signal_returns`` as a
       boolean mask aligned with ``forward_returns``; entries where the mask is
       True become the signal sample and the rest become the placebo. (Use this
       only when you have no better matched control.)

    Parameters
    ----------
    signal_returns : array-like
        Forward returns of signalled entries, OR a boolean mask (see above).
    placebo_returns : array-like, optional
        Forward returns of the matched random pool. Required in mode 1.
    forward_returns : array-like, optional
        Forward returns aligned with the boolean mask. Required in mode 2.
    n_boot : int, default 2000
        Bootstrap / permutation resamples.
    block : int, default 5
        Block length for the bootstrap CI on the difference.
    rng : numpy.random.Generator | int | None
        Random source / seed.

    Returns
    -------
    PlaceboResult
    """
    gen = np.random.default_rng(rng)

    if forward_returns is not None and placebo_returns is None:
        mask = np.asarray(signal_returns).ravel().astype(bool)
        fwd = np.asarray(forward_returns, dtype=float).ravel()
        if mask.shape != fwd.shape:
            raise ValueError("`signal_returns` mask and `forward_returns` must align.")
        finite = np.isfinite(fwd)
        sig = fwd[mask & finite]
        pla = fwd[(~mask) & finite]
    elif placebo_returns is not None:
        sig = np.asarray(signal_returns, dtype=float).ravel()
        pla = np.asarray(placebo_returns, dtype=float).ravel()
        sig = sig[np.isfinite(sig)]
        pla = pla[np.isfinite(pla)]
    else:
        raise ValueError(
            "Provide either `placebo_returns` (mode 1) or `forward_returns` "
            "with a boolean `signal_returns` mask (mode 2)."
        )

    if sig.size < 2 or pla.size < 2:
        raise ValueError("Both signal and placebo samples need >= 2 observations.")

    signal_mean = float(sig.mean())
    placebo_mean = float(pla.mean())
    diff = signal_mean - placebo_mean

    # Block-bootstrap CI on the difference: resample the signal sample's mean
    # (autocorrelation-aware) against the fixed placebo mean.
    boot_signal = _block_bootstrap_mean(sig, n_boot, block, gen)
    diffs = boot_signal - placebo_mean
    diff_ci = (float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95)))

    # Permutation test: pool both samples, repeatedly relabel, compare diffs.
    pooled = np.concatenate([sig, pla])
    n_sig = sig.size
    perm_diffs = np.empty(n_boot)
    for b in range(n_boot):
        perm = gen.permutation(pooled)
        perm_diffs[b] = perm[:n_sig].mean() - perm[n_sig:].mean()
    p_value = float((np.sum(np.abs(perm_diffs) >= abs(diff)) + 1) / (n_boot + 1))

    return PlaceboResult(
        signal_mean=signal_mean,
        placebo_mean=placebo_mean,
        diff=diff,
        diff_ci=diff_ci,
        p_value=p_value,
        n_signal=int(n_sig),
        n_placebo=int(pla.size),
        win_rate=float((sig > 0).mean()),
    )
