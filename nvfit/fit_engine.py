from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
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
    covariance: np.ndarray | None = None
    free_mask: list[bool] | None = None
    uncertainty: dict[str, object] = field(default_factory=dict)
    model_func: Callable | None = field(default=None, repr=False, compare=False)


class UnderdeterminedFitError(ValueError):
    """The data cannot identify the model's free parameters (raised before optimizing)."""


def check_fit_readiness(model_name: str, x, n_free: int) -> None:
    """Fail early, with a fixable explanation, when a fit cannot be identified by the data.

    Fewer finite points (or distinct x values) than free parameters leaves the model unidentifiable.
    Exactly as many is a saturated fit: it runs, but parameter errors are reported as unavailable.
    """
    x = np.asarray(x, dtype=float)
    n_points = int(np.count_nonzero(np.isfinite(x)))
    n_distinct = int(np.unique(x[np.isfinite(x)]).size)
    hint = "widen the fit range, reduce binning, lock parameters, or choose a simpler model"
    if n_points < n_free:
        raise UnderdeterminedFitError(f"{n_points} points can't identify {n_free} free parameters of {model_name}; {hint}.")
    if n_distinct < n_free:
        raise UnderdeterminedFitError(
            f"{n_distinct} distinct x values can't identify {n_free} free parameters of {model_name}; {hint}."
        )


_controls = ContextVar("fit_parameter_controls", default=None)


def current_parameter_controls(model_name):
    scope = _controls.get()
    if scope is None or scope[0] != model_name:
        return None
    return scope[1]


@contextmanager
def parameter_controls(controls, *, model_name):
    """Apply controls only to the named model, never to nested/fallback models."""
    token = _controls.set((model_name, controls))
    try:
        yield
    finally:
        _controls.reset(token)


def apply_parameter_controls(names, lower, upper, seed, controls):
    for i, name in enumerate(names):
        if not controls or name not in controls:
            continue
        value, locked, lo, hi = controls[name]
        if lo is not None:
            lower[i] = lo
        if hi is not None:
            upper[i] = hi
        if value is not None and np.isfinite(value):
            seed[i] = value
            if locked:
                lower[i] = upper[i] = value


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
    # full_matrices so an underdetermined Jacobian (fewer rows than parameters) keeps its
    # null directions; their singular values are zero and mark those parameters NaN below.
    _, sv, vt = np.linalg.svd(js, full_matrices=js.shape[0] < js.shape[1])
    sv = np.concatenate([sv, np.zeros(vt.shape[0] - len(sv))])
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
    jac_func: Callable[..., np.ndarray] | None = None,
) -> FitResult:
    """``jac_func(x, *params)``, when given, returns d model / d params (n_points x n_params) and replaces the
    finite-difference Jacobian; it must match ``model_func`` exactly."""
    lower_arr = np.asarray(list(lower), dtype=float)
    upper_arr = np.asarray(list(upper), dtype=float)
    seeds = [np.asarray(seed, dtype=float).copy() for seed in seeds]
    if not seeds:
        raise ValueError("At least one optimizer start is required.")
    apply_parameter_controls(param_names, lower_arr, upper_arr, seeds[0], current_parameter_controls(model_name))
    if np.any(upper_arr < lower_arr):
        raise ValueError("Parameter bounds are reversed; lower must not exceed upper.")
    # A locked parameter arrives as lower == upper.  least_squares needs a strictly positive
    # interval, so optimize only the free parameters and re-insert the fixed values.
    fixed = upper_arr == lower_arr
    free = ~fixed
    if not np.any(free):
        raise ValueError("All fit parameters are locked; unlock at least one parameter to fit.")
    check_fit_readiness(model_name, x, int(np.count_nonzero(free)))
    fixed_vals = lower_arr.copy()

    def full(p_free: np.ndarray) -> np.ndarray:
        p = fixed_vals.copy()
        p[free] = p_free
        return p

    weighted = weights is not None
    if weights is None:
        weights = np.ones_like(y, dtype=float)
    w_sqrt = np.sqrt(weights)
    if f_scale is None:
        y_span = float(np.percentile(y, 95) - np.percentile(y, 5))
        f_scale = max(1e-6, 0.08 * y_span)
    if not isinstance(x_scale, str):
        xs = np.asarray(x_scale, dtype=float)
        x_scale = xs[free] if xs.ndim == 1 and xs.size == len(free) else x_scale

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
                lambda p: w_sqrt * (model_func(x, *full(p)) - y),
                seed[free],
                jac=(lambda p: w_sqrt[:, None] * jac_func(x, *full(p))[:, free]) if jac_func is not None else "2-point",
                bounds=(lower_arr[free], upper_arr[free]),
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

    p_opt = full(best.x)
    y_fit = model_func(x, *p_opt)
    n_free = int(np.count_nonzero(free))
    dof = len(y) - n_free
    # best.cost is the robust-loss sum for soft_l1 etc.; use the actual (weighted) residuals for s^2
    raw_cost = 0.5 * float(np.sum(np.square(best.fun)))
    p_err = np.zeros(len(p_opt))  # locked parameters are exact, not estimated
    covariance = np.zeros((len(p_opt), len(p_opt)))
    if dof >= 1:
        cov = _estimate_covariance(best.jac, raw_cost, dof=dof)
        covariance[np.ix_(free, free)] = cov
        p_err[free] = np.sqrt(np.maximum(np.diag(cov), 0.0))  # NaN propagates for unidentifiable parameters
    else:
        # no residual degrees of freedom: the data cannot estimate parameter uncertainty
        p_err[free] = np.nan
        covariance[np.ix_(free, free)] = np.nan
    hits = bound_hit_flags(p_opt, lower_arr, upper_arr)
    hits = [bool(h and not f) for h, f in zip(hits, fixed)]

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
        aic=aic(y, y_fit, n_free),
        bic=bic(y, y_fit, n_free),
        durbin_watson=durbin_watson(y - y_fit),
        bound_hits=hits,
        success=bool(best.success),
        message=message,
        selection_note=selection_note,
        extras={"fixed_params": [n for n, f in zip(param_names, fixed) if f]} if np.any(fixed) else {},
        covariance=covariance,
        free_mask=free.tolist(),
        model_func=model_func,
        uncertainty={
            "method": "optimizer Jacobian; relative residual scale",
            "loss": loss,
            "weighted": weighted,
            "dof": dof,
            "n_observations": len(y),
            "band": "1-sigma model-mean confidence; excludes prediction noise",
        },
    )
