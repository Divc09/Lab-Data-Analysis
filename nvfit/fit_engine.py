from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import least_squares

from .diagnostics import aic, bic, bound_hit_flags, durbin_watson, r_squared, rmse


@dataclass
class FitResult:
    model_name: str
    param_names: list[str]
    params: np.ndarray
    errors: np.ndarray
    y_fit: np.ndarray
    r2: float
    rmse: float
    aic: float
    bic: float
    durbin_watson: float
    bound_hits: list[bool]
    success: bool
    message: str
    selection_note: str = ""
    extras: dict[str, object] = field(default_factory=dict)


def _estimate_covariance(jac: np.ndarray, cost: float, dof: int) -> np.ndarray:
    """Parameter covariance from the Jacobian.

    Columns are equilibrated to unit norm first so parameters with wildly
    different units (e.g. T in ns vs. chirp2 in ns^-2) do not make J^T J
    numerically singular, which previously made ``pinv`` silently discard
    real directions and return fake, tiny variances.  Parameters that remain
    unidentifiable (numerically null Jacobian direction) get NaN variance.
    """
    s_sq = (2.0 * cost) / max(1, dof)
    jac = np.asarray(jac, dtype=float)
    n = jac.shape[1]
    norms = np.linalg.norm(jac, axis=0)
    ok = np.isfinite(norms) & (norms > 0.0)
    cov = np.full((n, n), np.nan)
    if not np.any(ok):
        return cov
    idx = np.flatnonzero(ok)
    js = jac[:, idx] / norms[idx]
    _, sv, vt = np.linalg.svd(js, full_matrices=False)
    keep = sv > sv[0] * 1e-8
    inv_s2 = np.where(keep, 1.0 / np.where(keep, sv, 1.0) ** 2, 0.0)
    cov_s = (vt.T * inv_s2) @ vt
    # a parameter is unidentifiable if it loads on a dropped (null) direction
    null_load = np.sum(vt[~keep, :] ** 2, axis=0) if not np.all(keep) else np.zeros(len(idx))
    cov_s[np.ix_(null_load > 1e-2, np.ones(len(idx), bool))] = np.nan
    cov_s[np.ix_(np.ones(len(idx), bool), null_load > 1e-2)] = np.nan
    cov[np.ix_(idx, idx)] = s_sq * cov_s / np.outer(norms[idx], norms[idx])
    return cov


def sem_to_weights(sem) -> np.ndarray | None:
    """Relative inverse-variance weights (1/sem^2, median-normalised) from per-point SEM.

    Non-finite or non-positive SEM entries are replaced by the median valid SEM (and
    SEM is floored at 1e-3 of that median) so one bad point cannot dominate.  Returns
    None when fewer than half of the entries are usable.
    """
    if sem is None:
        return None
    sem = np.asarray(sem, dtype=float).ravel()
    good = np.isfinite(sem) & (sem > 0)
    if sem.size == 0 or np.count_nonzero(good) < 0.5 * sem.size:
        return None
    med = float(np.median(sem[good]))
    sem = np.where(good, np.maximum(sem, 1e-3 * med), med)
    w = 1.0 / np.square(sem)
    return w / float(np.median(w))


def reduced_chi2(y, y_fit, sem, n_free: int) -> float:
    """sum((y - y_fit)^2 / sem^2) / dof with absolute weights; NaN if SEM unusable."""
    y, y_fit, sem = (np.asarray(v, dtype=float).ravel() for v in (y, y_fit, sem))
    ok = np.isfinite(y) & np.isfinite(y_fit) & np.isfinite(sem) & (sem > 0)
    dof = int(np.count_nonzero(ok)) - int(n_free)
    if dof < 1:
        return float("nan")
    return float(np.sum(((y - y_fit)[ok] / sem[ok]) ** 2) / dof)


def fit_model_multistart(
    *,
    model_name: str,
    model_func: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    param_names: list[str],
    lower: Iterable[float],
    upper: Iterable[float],
    seeds: list[np.ndarray],
    weights: np.ndarray | None = None,
    loss: str = "soft_l1",
    f_scale: float | None = None,
    max_nfev: int = 20000,
    x_scale: str | np.ndarray | float = 1.0,
    selection_note: str = "",
) -> FitResult:
    lower_arr = np.asarray(list(lower), dtype=float)
    upper_arr = np.asarray(list(upper), dtype=float)
    if weights is None:
        weights = np.ones_like(y, dtype=float)
    w_sqrt = np.sqrt(weights)
    if f_scale is None:
        y_span = float(np.percentile(y, 95) - np.percentile(y, 5))
        f_scale = max(1e-6, 0.08 * y_span)

    best = None
    best_cost = np.inf
    failed_attempts: list[str] = []
    attempted = 0
    for seed in seeds:
        attempted += 1
        seed = np.asarray(seed, dtype=float)
        seed = np.clip(seed, lower_arr, upper_arr)
        try:
            res = least_squares(
                lambda p: w_sqrt * (model_func(x, *p) - y),
                seed,
                bounds=(lower_arr, upper_arr),
                loss=loss,
                f_scale=f_scale,
                max_nfev=max_nfev,
                x_scale=x_scale,
            )
        except Exception as exc:
            failed_attempts.append(str(exc))
            continue
        if res.cost < best_cost:
            best_cost = res.cost
            best = res

    if best is None:
        detail = failed_attempts[-1] if failed_attempts else "no optimizer attempts were available"
        raise RuntimeError(f"All {attempted} fit attempts failed: {detail}")

    p_opt = best.x
    y_fit = model_func(x, *p_opt)
    dof = len(y) - len(p_opt)
    # best.cost is the robust-loss sum for soft_l1 etc.; use the actual (weighted) residuals for s^2
    raw_cost = 0.5 * float(np.sum(np.square(best.fun)))
    cov = _estimate_covariance(best.jac, raw_cost, dof=max(1, dof))
    p_err = np.sqrt(np.maximum(np.diag(cov), 0.0))  # NaN propagates for unidentifiable parameters

    message = str(best.message)
    if failed_attempts:
        message += f"; retained best result after {len(failed_attempts)} of {attempted} starts failed"

    return FitResult(
        model_name=model_name,
        param_names=param_names,
        params=p_opt,
        errors=p_err,
        y_fit=y_fit,
        r2=r_squared(y, y_fit),
        rmse=rmse(y, y_fit),
        aic=aic(y, y_fit, len(p_opt)),
        bic=bic(y, y_fit, len(p_opt)),
        durbin_watson=durbin_watson(y - y_fit),
        bound_hits=bound_hit_flags(p_opt, lower_arr, upper_arr),
        success=bool(best.success),
        message=message,
        selection_note=selection_note,
    )

