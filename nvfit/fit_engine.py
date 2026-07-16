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
    s_sq = (2.0 * cost) / max(1, dof)
    jtj = jac.T @ jac
    return s_sq * np.linalg.pinv(jtj)


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
    for seed in seeds:
        seed = np.asarray(seed, dtype=float)
        seed = np.clip(seed, lower_arr, upper_arr)
        res = least_squares(
            lambda p: w_sqrt * (model_func(x, *p) - y),
            seed,
            bounds=(lower_arr, upper_arr),
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            x_scale=x_scale,
        )
        if res.cost < best_cost:
            best_cost = res.cost
            best = res

    if best is None:
        raise RuntimeError("No fit attempt executed.")

    p_opt = best.x
    y_fit = model_func(x, *p_opt)
    dof = len(y) - len(p_opt)
    cov = _estimate_covariance(best.jac, best.cost, dof=max(1, dof))
    p_err = np.sqrt(np.maximum(np.diag(cov), 0.0))

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
        message=str(best.message),
        selection_note=selection_note,
    )

