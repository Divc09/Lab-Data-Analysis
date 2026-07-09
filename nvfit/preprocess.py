from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def apply_roi(x: np.ndarray, y: np.ndarray, min_x: float | None, max_x: float | None) -> tuple[np.ndarray, np.ndarray]:
    if min_x is None:
        min_x = float(np.min(x))
    if max_x is None:
        max_x = float(np.max(x))
    mask = (x >= min_x) & (x <= max_x)
    return x[mask], y[mask]


def bin_trace(x: np.ndarray, y: np.ndarray, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    b = max(1, int(bin_size))
    if b == 1:
        return x.copy(), y.copy()
    n = (len(x) // b) * b
    if n <= 0:
        return x.copy(), y.copy()
    x_b = np.mean(x[:n].reshape(-1, b), axis=1)
    y_b = np.mean(y[:n].reshape(-1, b), axis=1)
    return x_b, y_b


def smooth_trace(y: np.ndarray, window: int = 1, polyorder: int = 2) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1:
        return y.copy()
    if w % 2 == 0:
        w += 1
    if w >= len(y):
        w = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if w < 3:
        return y.copy()
    p = min(polyorder, w - 1)
    return savgol_filter(y, window_length=w, polyorder=p, mode="interp")

