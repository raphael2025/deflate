"""Parameter-plateau detection.

A trustworthy parameter setting sits on a *plateau*: its neighbours in the
parameter grid perform almost as well. A setting that sits on a lonely *spike*
-- great by itself, surrounded by mediocre neighbours -- is almost certainly a
fit to noise.

``parameter_plateau`` takes the score of every grid point and quantifies how
plateau-like the best point is, by comparing it to its local neighbourhood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PlateauResult", "parameter_plateau"]


@dataclass(frozen=True)
class PlateauResult:
    """Result of a parameter-plateau analysis.

    Attributes
    ----------
    best_index : tuple[int, ...]
        Grid index of the best-scoring parameter point.
    best_score : float
        Score at the best point.
    neighbor_mean : float
        Mean score of the best point's immediate grid neighbours.
    neighbor_ratio : float
        ``neighbor_mean / best_score`` (clipped sensibly). Near 1.0 means a
        broad plateau; near 0 (or negative) means a lonely spike.
    is_plateau : bool
        True if the best point sits on a plateau (``neighbor_ratio`` above
        ``min_ratio``).
    grid_cv : float
        Coefficient of variation of all grid scores -- a smoother grid (lower
        CV) is generally more trustworthy.
    """

    best_index: tuple[int, ...]
    best_score: float
    neighbor_mean: float
    neighbor_ratio: float
    is_plateau: bool
    grid_cv: float


def _neighbor_offsets(ndim: int) -> list[tuple[int, ...]]:
    """All offsets in {-1,0,1}^ndim except the all-zero (self) offset."""
    from itertools import product

    return [o for o in product((-1, 0, 1), repeat=ndim) if any(o)]


def parameter_plateau(
    grid_scores,
    min_ratio: float = 0.5,
) -> PlateauResult:
    """Assess whether the best grid point sits on a plateau or a spike.

    Parameters
    ----------
    grid_scores : array-like
        Scores (e.g. Sharpe ratios) over a parameter grid. Can be 1-D (a single
        swept parameter) or N-D (a multi-parameter grid). NaNs are treated as
        ``-inf`` so they never win and pull down neighbourhoods.
    min_ratio : float, default 0.5
        Minimum ``neighbor_mean / best_score`` for the best point to count as a
        plateau. With the default, neighbours must average at least half the
        peak score.

    Returns
    -------
    PlateauResult

    Notes
    -----
    Neighbours are the immediate grid cells (the Moore neighbourhood) of the
    best point. ``neighbor_ratio`` is computed against ``best_score`` only when
    the best score is positive; if the best score is non-positive the ratio is
    set to 0 (a non-positive optimum is not a credible plateau).
    """
    scores = np.asarray(grid_scores, dtype=float)
    if scores.size < 2:
        raise ValueError("`grid_scores` must contain at least 2 points.")
    filled = np.where(np.isfinite(scores), scores, -np.inf)

    best_flat = int(np.argmax(filled))
    best_index = np.unravel_index(best_flat, scores.shape)
    best_score = float(filled[best_index])

    neighbor_vals: list[float] = []
    for off in _neighbor_offsets(scores.ndim):
        nb = tuple(int(b) + d for b, d in zip(best_index, off))
        if all(0 <= nb[k] < scores.shape[k] for k in range(scores.ndim)):
            v = filled[nb]
            if np.isfinite(v):
                neighbor_vals.append(float(v))

    neighbor_mean = float(np.mean(neighbor_vals)) if neighbor_vals else best_score
    if best_score > 0:
        neighbor_ratio = neighbor_mean / best_score
    else:
        neighbor_ratio = 0.0

    finite_scores = filled[np.isfinite(filled)]
    mu = float(np.mean(finite_scores))
    grid_cv = float(np.std(finite_scores) / abs(mu)) if mu != 0 else float("inf")

    return PlateauResult(
        best_index=tuple(int(i) for i in best_index),
        best_score=best_score,
        neighbor_mean=neighbor_mean,
        neighbor_ratio=float(neighbor_ratio),
        is_plateau=bool(neighbor_ratio >= min_ratio),
        grid_cv=grid_cv,
    )
