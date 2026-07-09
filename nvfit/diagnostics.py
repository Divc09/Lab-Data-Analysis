from __future__ import annotations

import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def aic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    rss = max(float(rss), 1e-18)
    return float(n * np.log(rss / n) + 2 * n_params)


def bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    rss = max(float(rss), 1e-18)
    return float(n * np.log(rss / n) + n_params * np.log(n))


def durbin_watson(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    den = np.sum(residuals**2)
    if den <= 0:
        return float("nan")
    return float(np.sum(diff**2) / den)


def bound_hit_flags(params: np.ndarray, lower: np.ndarray, upper: np.ndarray, atol: float = 1e-9) -> list[bool]:
    hits = []
    for p, lo, hi in zip(params, lower, upper):
        lo_hit = np.isfinite(lo) and abs(p - lo) <= atol * max(1.0, abs(lo))
        hi_hit = np.isfinite(hi) and abs(p - hi) <= atol * max(1.0, abs(hi))
        hits.append(bool(lo_hit or hi_hit))
    return hits

