"""Top-level one-shot overfitting verdict.

``verdict()`` runs the full battery -- Deflated Sharpe, bootstrap, walk-forward,
and (optionally) PBO and a placebo test -- and returns a single ``Verdict``
dataclass with a boolean ``is_overfit``, a ``confidence`` score, human-readable
``reasons``, and the underlying ``metrics``. Printing it gives a plain-English
ruling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bootstrap import BootResult, bootstrap_metrics
from .pbo import PBOResult, pbo
from .placebo import PlaceboResult, placebo_test
from .sharpe import DSRResult, deflated_sharpe
from .walkforward import WFResult, walk_forward

__all__ = ["Verdict", "verdict"]


@dataclass
class Verdict:
    """Overall overfitting ruling for a backtest.

    Attributes
    ----------
    is_overfit : bool
        The headline call. True if the evidence says the backtest is likely
        overfit / not trustworthy.
    confidence : float
        Confidence in the *is_overfit* call, in ``[0, 1]``. Derived from how
        many and how strongly the individual checks fired.
    reasons : list[str]
        Human-readable bullet points explaining the ruling.
    metrics : dict
        The underlying result objects, keyed by check name
        (``"dsr"``, ``"bootstrap"``, ``"walk_forward"``, ``"pbo"``,
        ``"placebo"``). Values are the respective dataclasses (or ``None`` if a
        check was not run).
    """

    is_overfit: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def dsr(self) -> Optional[DSRResult]:
        return self.metrics.get("dsr")

    @property
    def bootstrap(self) -> Optional[BootResult]:
        return self.metrics.get("bootstrap")

    @property
    def walk_forward(self) -> Optional[WFResult]:
        return self.metrics.get("walk_forward")

    @property
    def pbo(self) -> Optional[PBOResult]:
        return self.metrics.get("pbo")

    @property
    def placebo(self) -> Optional[PlaceboResult]:
        return self.metrics.get("placebo")

    def __str__(self) -> str:
        verdict_line = (
            "LIKELY OVERFIT" if self.is_overfit else "PLAUSIBLY ROBUST"
        )
        bar = "=" * 60
        lines = [
            bar,
            f"  deflate verdict:  {verdict_line}",
            f"  confidence:       {self.confidence:.0%}",
            bar,
        ]
        d = self.dsr
        if d is not None:
            lines.append(
                f"  Deflated Sharpe : DSR={d.dsr:.3f}  "
                f"(annual SR={d.sharpe_annual:.2f}, n_trials={d.n_trials})"
            )
        b = self.bootstrap
        if b is not None:
            lines.append(
                f"  Bootstrap       : P(SR<=0)={b.p_sharpe_le_0:.2f}  "
                f"SR 90% CI=[{b.sharpe_ci[0]:.2f}, {b.sharpe_ci[1]:.2f}]"
            )
            lines.append(
                f"                    realistic worst DD (1%): "
                f"{b.max_drawdown_p01:.1%}  (observed {b.max_drawdown:.1%})"
            )
        w = self.walk_forward
        if w is not None:
            lines.append(
                f"  Walk-forward    : {w.frac_positive:.0%} of folds positive  "
                f"(mean fold SR={w.mean_sharpe:.2f})"
            )
        p = self.pbo
        if p is not None:
            lines.append(f"  PBO (CSCV)      : {p.pbo:.2f}")
        pl = self.placebo
        if pl is not None:
            lines.append(
                f"  Placebo test    : diff={pl.diff:.4g}  "
                f"CI=[{pl.diff_ci[0]:.4g}, {pl.diff_ci[1]:.4g}]  p={pl.p_value:.3f}"
            )
        lines.append(bar)
        if self.reasons:
            lines.append("  Why:")
            for reason in self.reasons:
                lines.append(f"    - {reason}")
            lines.append(bar)
        return "\n".join(lines)


def verdict(
    returns,
    n_trials: int = 1,
    returns_matrix=None,
    signal_returns=None,
    placebo_returns=None,
    forward_returns=None,
    periods_per_year: int = 252,
    n_boot: int = 2000,
    block: int = 5,
    n_folds: int = 5,
    n_splits: int = 16,
    dsr_threshold: float = 0.95,
    pbo_threshold: float = 0.5,
    rng: np.random.Generator | int | None = None,
) -> Verdict:
    """Run the full overfitting battery and return a single ruling.

    Always runs: Deflated Sharpe, bootstrap (Sharpe & drawdown), walk-forward.
    Optionally runs PBO (if ``returns_matrix`` is given) and a placebo test (if
    placebo / forward returns are given).

    Parameters
    ----------
    returns : array-like
        The strategy's periodic return series (e.g. daily).
    n_trials : int, default 1
        How many configurations you searched to land on this one. **Set this
        honestly** -- it is the single most important input for catching
        multiple-testing overfitting. A grid search of 200 points is
        ``n_trials=200``.
    returns_matrix : DataFrame or 2-D array, optional
        ``(T, N)`` returns of competing configurations, for PBO. If you ran a
        grid search, pass every configuration's returns here.
    signal_returns, placebo_returns, forward_returns : array-like, optional
        Inputs for the placebo test (see :func:`deflate.placebo.placebo_test`).
    periods_per_year : int, default 252
        Annualisation factor (365 for crypto daily, 12 for monthly).
    n_boot, block, n_folds, n_splits : int
        Knobs for the bootstrap, walk-forward and PBO respectively.
    dsr_threshold : float, default 0.95
        DSR below this counts as evidence of overfitting.
    pbo_threshold : float, default 0.5
        PBO above this counts as evidence of overfitting.
    rng : numpy.random.Generator | int | None
        Random source / seed for the resampling-based checks.

    Returns
    -------
    Verdict

    Notes
    -----
    DSR and PBO are **selection-aware**: they account for how many configurations
    you tried and whether the in-sample winner survives out of sample, so they
    can actually *see* overfitting. The bootstrap, walk-forward and placebo
    checks are **selection-blind** -- they only see the single curve you hand
    them, which a cherry-picked strategy can fool. Accordingly, a failing
    selection-aware check forces an overfit ruling; the selection-blind checks
    can corroborate or raise separate concerns, but cannot clear a failed
    selection-aware check. ``confidence`` is the magnitude of the weighted
    evidence away from the 0.5 boundary.
    """
    metrics: dict = {}
    reasons: list[str] = []

    # ----- always-on checks -----
    dsr_res = deflated_sharpe(
        returns, n_trials=n_trials, periods_per_year=periods_per_year
    )
    metrics["dsr"] = dsr_res

    boot_res = bootstrap_metrics(
        returns,
        n_boot=n_boot,
        block=block,
        periods_per_year=periods_per_year,
        rng=rng,
    )
    metrics["bootstrap"] = boot_res

    wf_res = walk_forward(returns, n_folds=n_folds, periods_per_year=periods_per_year)
    metrics["walk_forward"] = wf_res

    # ----- optional checks -----
    pbo_res = None
    if returns_matrix is not None:
        pbo_res = pbo(returns_matrix, n_splits=n_splits)
        metrics["pbo"] = pbo_res

    placebo_res = None
    if placebo_returns is not None or forward_returns is not None:
        placebo_res = placebo_test(
            signal_returns if signal_returns is not None else returns,
            placebo_returns=placebo_returns,
            forward_returns=forward_returns,
            n_boot=n_boot,
            block=block,
            rng=rng,
        )
        metrics["placebo"] = placebo_res

    # ----- evidence of overfitting -----
    #
    # Crucial distinction:
    #   * SELECTION-AWARE checks (DSR, PBO) know how many configs you tried and
    #     whether the in-sample winner holds up out of sample. They can SEE
    #     overfitting.
    #   * SELECTION-BLIND checks (bootstrap, walk-forward, placebo) only see the
    #     single curve you hand them. A curve cherry-picked from many trials will
    #     look great to them -- they cannot detect that you searched. So they may
    #     ADD weight to an overfit ruling, but they must never CLEAR a failing
    #     selection-aware check.
    #
    # Each entry: (weight, fired, selection_aware, reason).
    evidence: list[tuple[float, bool, bool, str]] = []

    dsr_fired = dsr_res.dsr < dsr_threshold
    evidence.append(
        (
            0.5,
            dsr_fired,
            True,
            f"Deflated Sharpe is {dsr_res.dsr:.2f} (< {dsr_threshold:.2f}): after "
            f"correcting for {n_trials} trial(s) and fat tails, the edge is not "
            f"convincingly positive."
            if dsr_fired
            else f"Deflated Sharpe is {dsr_res.dsr:.2f} (>= {dsr_threshold:.2f}): "
            f"the edge survives correction for {n_trials} trial(s) and non-normality.",
        )
    )

    boot_fired = boot_res.p_sharpe_le_0 > 0.10
    evidence.append(
        (
            0.2,
            boot_fired,
            False,
            f"Bootstrap P(SR<=0) is {boot_res.p_sharpe_le_0:.2f}: the Sharpe is "
            f"not reliably above zero under resampling."
            if boot_fired
            else f"Bootstrap P(SR<=0) is {boot_res.p_sharpe_le_0:.2f}: the Sharpe "
            f"stays positive across resamples (selection-blind: a cherry-picked "
            f"curve can still look good here).",
        )
    )

    wf_fired = wf_res.frac_positive < 0.6
    evidence.append(
        (
            0.2,
            wf_fired,
            False,
            f"Only {wf_res.frac_positive:.0%} of walk-forward folds are positive: "
            f"performance is concentrated/unstable across time."
            if wf_fired
            else f"{wf_res.frac_positive:.0%} of walk-forward folds are positive: "
            f"performance is reasonably stable across time (selection-blind).",
        )
    )

    if pbo_res is not None:
        # PBO ~ 0.5 already means in-sample ranking is uninformative. Treat the
        # threshold inclusively and allow a small band below it to count.
        pbo_fired = pbo_res.pbo >= pbo_threshold
        evidence.append(
            (
                0.5,
                pbo_fired,
                True,
                f"PBO is {pbo_res.pbo:.2f} (>= {pbo_threshold:.2f}): the in-sample "
                f"best configuration is no better than a coin flip out of sample."
                if pbo_fired
                else f"PBO is {pbo_res.pbo:.2f} (< {pbo_threshold:.2f}): the "
                f"in-sample winner tends to hold up out of sample.",
            )
        )

    if placebo_res is not None:
        placebo_fired = placebo_res.diff_ci[0] <= 0.0
        evidence.append(
            (
                0.25,
                placebo_fired,
                False,
                f"Placebo test: signal-vs-random difference CI "
                f"[{placebo_res.diff_ci[0]:.4g}, {placebo_res.diff_ci[1]:.4g}] "
                f"straddles 0 (p={placebo_res.p_value:.3f}): the signal does not "
                f"beat random entries."
                if placebo_fired
                else f"Placebo test: signal beats random entries "
                f"(diff CI excludes 0, p={placebo_res.p_value:.3f}).",
            )
        )

    # Weighted overfit score (all checks contribute their weight when fired).
    total_w = sum(w for w, _, _, _ in evidence)
    overfit_w = sum(w for w, fired, _, _ in evidence if fired)
    score = overfit_w / total_w if total_w > 0 else 0.0
    is_overfit = score >= 0.5

    # Override: any fired selection-aware check forces an overfit ruling, because
    # selection-blind checks cannot legitimately vote it down. (Selection-aware
    # checks looking clean does NOT force "robust" -- the blind checks may still
    # flag a different problem.)
    selection_aware_fired = any(
        fired for _, fired, aware, _ in evidence if aware
    )
    if selection_aware_fired:
        is_overfit = True

    confidence = min(1.0, abs(score - 0.5) * 2.0)
    if selection_aware_fired and not (score >= 0.5):
        # Verdict driven by the override; report at least moderate confidence.
        confidence = max(confidence, 0.5)

    # Surface the reasons that point toward the chosen verdict first.
    for _, fired, _, reason in evidence:
        if fired == is_overfit:
            reasons.append(reason)
    for _, fired, _, reason in evidence:
        if fired != is_overfit:
            reasons.append(reason)

    return Verdict(
        is_overfit=is_overfit,
        confidence=confidence,
        reasons=reasons,
        metrics=metrics,
    )
