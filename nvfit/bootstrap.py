"""Bootstrap parameter uncertainties for 1D fits.

Two resampling schemes:

* ``iterations`` (physically correct): an iteration matrix (points x iterations, NaN for
  invalid/missing) is resampled over its columns with replacement; the mean trace (and its
  SEM) of each resample is refit.  This propagates the real shot-to-shot noise.
* ``residuals`` (fallback): residuals of the baseline fit are resampled with replacement
  and added back to the baseline curve (assumes i.i.d. residuals, ignores correlation).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _mean_and_sem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(matrix)
    n = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.nansum(np.where(valid, matrix, 0.0), axis=1) / np.maximum(n, 1), np.nan)
        dev = np.where(valid, matrix - mean[:, None], 0.0)
        var = np.where(n > 1, np.sum(dev**2, axis=1) / np.maximum(n - 1, 1), np.nan)
        sem = np.sqrt(var / np.maximum(n, 1))
    return mean, sem, n


def bootstrap_fit(
    fit_callable: Callable[..., object],
    x,
    y,
    n_boot: int = 200,
    seed: int = 0,
    iteration_matrix=None,
) -> dict:
    """Bootstrap ``fit_callable(x, y, sem)``.

    ``fit_callable`` returns an object with ``param_names``, ``params`` and ``y_fit``
    (a ``FitResult``).  ``sem`` is the per-point SEM of the resampled mean trace for
    iteration resampling and ``None`` for residual resampling.  ``iteration_matrix`` has
    shape (len(x), n_iterations).  Fewer optimizer starts per refit (e.g. multistart 4)
    should be configured inside ``fit_callable``.

    Returns {"method", "n_boot", "n_success", "baseline": {name: value},
    "params": {name: {median, p16, p84, std}}} (units are those of the fitted parameters).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    base = fit_callable(x, y, None)
    names = list(base.param_names)
    out = {"baseline": {n: float(v) for n, v in zip(names, base.params)}, "n_boot": int(n_boot)}

    matrix = None
    if iteration_matrix is not None:
        matrix = np.asarray(iteration_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(x) or matrix.shape[1] < 2:
            matrix = None
    method = "iterations" if matrix is not None else "residuals"
    resid = y - np.asarray(base.y_fit, dtype=float)
    resid = resid[np.isfinite(resid)]

    draws = []
    for _ in range(int(n_boot)):
        try:
            if matrix is not None:
                cols = rng.integers(0, matrix.shape[1], matrix.shape[1])
                mean, sem, n = _mean_and_sem(matrix[:, cols])
                ok = np.isfinite(mean)
                if ok.sum() < 5:
                    continue
                res = fit_callable(x[ok], mean[ok], sem[ok])
            else:
                yb = np.asarray(base.y_fit, dtype=float) + rng.choice(resid, size=len(x), replace=True)
                res = fit_callable(x, yb, None)
            if list(res.param_names) != names:
                continue  # model selection flipped; not the same parameter set
            p = np.asarray(res.params, dtype=float)
            if np.all(np.isfinite(p)):
                draws.append(p)
        except Exception:
            continue

    out.update({"method": method, "n_success": len(draws)})
    stats = {}
    if draws:
        d = np.vstack(draws)
        p16, med, p84 = np.percentile(d, [16, 50, 84], axis=0)
        std = np.std(d, axis=0, ddof=1) if len(d) > 1 else np.full(d.shape[1], np.nan)
        for i, n in enumerate(names):
            stats[n] = {"median": float(med[i]), "p16": float(p16[i]), "p84": float(p84[i]), "std": float(std[i])}
    else:
        for n in names:
            stats[n] = {"median": float("nan"), "p16": float("nan"), "p84": float("nan"), "std": float("nan")}
    out["params"] = stats
    return out
