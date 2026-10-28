"""Time-trace analysis helpers operating on ``(t_s, y[, sem])`` arrays.

Conventions
-----------
* ``t`` is **measured elapsed time in seconds** (never reconstructed from a cadence).
* ``y`` is in whatever unit the caller has (e.g. counts/s); the tools are unit-agnostic.
* Noise sigma is a *per-point* standard deviation, estimated robustly from the MAD of
  first differences: sigma = 1.4826 * MAD(diff(y)) / sqrt(2). This assumes white
  noise of roughly constant amplitude; slow drift barely affects it, but
  intensity-dependent (Poisson-like) noise makes it an average value.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

MAD_TO_SIGMA = 1.4826


def _clean_xy(t, y, sem=None):
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValueError("t and y must have the same length.")
    s = None if sem is None else np.asarray(sem, dtype=float).ravel()
    if s is not None and s.shape != t.shape:
        raise ValueError("sem must match t and y in length.")
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if s is not None:
        s = s[ok]
    order = np.argsort(t, kind="stable")
    return t[order], y[order], (None if s is None else s[order])


def robust_sigma(y) -> float:
    """Per-point noise sigma from the MAD of first differences (see module docstring)."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        raise ValueError("Need at least 3 points to estimate noise.")
    d = np.diff(y)
    sig = MAD_TO_SIGMA * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2.0)
    if sig <= 0:  # quantised / flat data: fall back to the plain std of differences
        sig = float(np.std(d)) / math.sqrt(2.0)
    if sig <= 0:
        sig = 1e-12 * max(1.0, float(np.max(np.abs(y))))
    return sig


def rolling_sigma(y, window: int = 11) -> np.ndarray:
    """Local per-point sigma: rolling MAD of first differences (/sqrt2), window in points."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 4:
        raise ValueError("Need at least 4 points for a rolling noise estimate.")
    d = np.diff(y)
    w = int(min(window, n - 1))
    if w % 2 == 0:
        w -= 1
    w = max(w, 3)
    half = w // 2
    mode = "reflect" if half < d.size else "edge"
    dp = np.pad(d, (half, half), mode=mode)
    win = np.lib.stride_tricks.sliding_window_view(dp, w)[: d.size]
    med = np.median(win, axis=1, keepdims=True)
    sig_d = MAD_TO_SIGMA * np.median(np.abs(win - med), axis=1) / math.sqrt(2.0)
    sig_d = np.where(sig_d > 0, sig_d, robust_sigma(y))
    return np.interp(np.arange(n), np.arange(n - 1) + 0.5, sig_d)


# ---------------------------------------------------------------------------
# Step detection
# ---------------------------------------------------------------------------

def detect_steps(t, y, min_size_sigma: float = 4.0, min_segment: int = 5, max_steps: int = 100) -> dict:
    """Binary-segmentation change-point detection on a piecewise-constant mean.

    A split of a segment into left/right parts (each >= ``min_segment`` points) is accepted
    when its significance  z = |mean_R - mean_L| / (sigma * sqrt(1/nL + 1/nR))  is
    >= ``min_size_sigma``, where sigma is the robust per-point noise (MAD of first
    differences). The most significant pending split is taken first, then sides are searched recursively.

    Returns a dict with ``steps`` (each: ``t_s`` = midpoint between the last sample before and the
    first after the step, ``t_half_gap_s`` = half that gap, ``index`` = first index after the step,
    ``level_before``/``level_after`` = segment means (y units), ``size`` = after-before (signed, y units),
    ``size_sigma`` = |size|/sigma, ``significance`` = z above), ``segments``, ``sigma`` and ``n``.
    """
    t, y, _ = _clean_xy(t, y)
    n = t.size
    min_segment = max(int(min_segment), 1)
    if n < 2 * min_segment or n < 4:
        return {"steps": [], "segments": [], "sigma": None, "n": int(n),
                "note": f"Too few points ({n}) for min_segment={min_segment}."}
    sigma = robust_sigma(y)
    cs = np.concatenate([[0.0], np.cumsum(y)])

    def best_split(lo: int, hi: int):
        if hi - lo < 2 * min_segment:
            return None
        k = np.arange(lo + min_segment, hi - min_segment + 1)
        n1 = (k - lo).astype(float)
        n2 = (hi - k).astype(float)
        m1 = (cs[k] - cs[lo]) / n1
        m2 = (cs[hi] - cs[k]) / n2
        z = np.abs(m2 - m1) / (sigma * np.sqrt(1.0 / n1 + 1.0 / n2))
        j = int(np.argmax(z))
        return int(k[j]), float(z[j])

    bounds: list[int] = []
    queue = [(0, n)]
    while queue and len(bounds) < max_steps:
        cands = [(best_split(lo, hi), lo, hi) for lo, hi in queue]
        cands = [c for c in cands if c[0] is not None and c[0][1] >= min_size_sigma]
        if not cands:
            break
        (k, _z), lo, hi = max(cands, key=lambda c: c[0][1])
        queue = [seg for seg in queue if seg != (lo, hi)]
        queue += [(lo, k), (k, hi)]
        bounds.append(k)
    bounds.sort()
    edges = [0, *bounds, n]
    means = [float((cs[b] - cs[a]) / (b - a)) for a, b in zip(edges[:-1], edges[1:])]
    segments = [{"i0": a, "i1": b, "t0_s": float(t[a]), "t1_s": float(t[b - 1]), "level": m, "n": b - a}
                for (a, b), m in zip(zip(edges[:-1], edges[1:]), means)]
    steps = []
    for j, k in enumerate(bounds):
        n1 = edges[j + 1] - edges[j]
        n2 = edges[j + 2] - edges[j + 1]
        size = means[j + 1] - means[j]
        steps.append({
            "index": int(k),
            "t_s": float(0.5 * (t[k - 1] + t[k])),
            "t_half_gap_s": float(0.5 * (t[k] - t[k - 1])),
            "level_before": means[j],
            "level_after": means[j + 1],
            "size": float(size),
            "size_sigma": float(abs(size) / sigma),
            "significance": float(abs(size) / (sigma * math.sqrt(1.0 / n1 + 1.0 / n2))),
        })
    return {"steps": steps, "segments": segments, "sigma": float(sigma), "n": int(n),
            "min_size_sigma": float(min_size_sigma), "min_segment": int(min_segment),
            "note": "sigma = 1.4826*MAD(diff y)/sqrt2 (white-noise assumption); slow drift can appear as a staircase of small steps."}


# ---------------------------------------------------------------------------
# Ratio of two detectors
# ---------------------------------------------------------------------------

def _match_times(t1: np.ndarray, t2: np.ndarray, tol: float):
    """Index pairs (i1, i2): exact timestamp matches first, then nearest unused within ``tol``."""
    used2 = np.zeros(t2.size, dtype=bool)
    pairs: dict[int, int] = {}
    pos: dict[float, int] = {}
    for j, v in enumerate(t2):
        pos.setdefault(float(v), j)
    for i, v in enumerate(t1):
        j = pos.get(float(v))
        if j is not None and not used2[j]:
            pairs[i] = j
            used2[j] = True
    n_exact = len(pairs)
    for i in range(t1.size):
        if i in pairs:
            continue
        j0 = int(np.searchsorted(t2, t1[i]))
        best, bd = None, tol
        for j in (j0 - 1, j0):
            if 0 <= j < t2.size and not used2[j]:
                d = abs(t2[j] - t1[i])
                if d <= bd:
                    best, bd = j, d
        if best is not None:
            pairs[i] = best
            used2[best] = True
    idx = sorted(pairs)
    return np.array(idx, dtype=int), np.array([pairs[i] for i in idx], dtype=int), n_exact


def ratio(t1, y1, t2, y2, sem1=None, sem2=None, tol: float | None = None) -> dict:
    """Detector-2 / detector-1 ratio r = y2/y1 on matched timestamps.

    Matching: exact equal timestamps first; remaining points are paired with the nearest unused
    sample within ``tol`` (default 0.5 * median dt of detector 1). Unmatched points are dropped and counted.
    The returned time axis is detector 1's timestamp (seconds).
    Error (first order): sigma_r = sqrt((s2/y1)^2 + (r*s1/y1)^2), identical to
    |r|*sqrt((s1/y1)^2+(s2/y2)^2) but finite when y2 = 0. Points with y1 == 0 are dropped.
    If a sem is not supplied, per-point sigma is estimated with ``rolling_sigma`` (rolling MAD of first
    differences, 11 points) and ``sigma_source`` says so. Noise correlation between detectors is ignored.
    """
    t1, y1, s1 = _clean_xy(t1, y1, sem1)
    t2, y2, s2 = _clean_xy(t2, y2, sem2)
    if t1.size < 2 or t2.size < 2:
        raise ValueError("Each detector needs at least 2 points.")
    if tol is None:
        tol = 0.5 * float(np.median(np.diff(t1)))
    i1, i2, n_exact = _match_times(t1, t2, tol)
    src = []

    def estimated(y, name):
        # rolling_sigma needs 4 points; below that the per-point error is reported as unknown, not guessed.
        if y.size >= 4:
            src.append(f"{name}: rolling-MAD estimate")
            return rolling_sigma(y)
        src.append(f"{name}: not estimated (fewer than 4 points)")
        return np.full(y.shape, np.nan)

    if s1 is None:
        s1 = estimated(y1, "detector 1")
    if s2 is None:
        s2 = estimated(y2, "detector 2")
    a, b, sa, sb, tt = y1[i1], y2[i2], s1[i1], s2[i2], t1[i1]
    good = a != 0
    a, b, sa, sb, tt = a[good], b[good], sa[good], sb[good], tt[good]
    r = b / a
    sr = np.sqrt((sb / a) ** 2 + (r * sa / a) ** 2)
    return {
        "t_s": tt, "ratio": r, "sigma": sr,
        "n_matched": int(i1.size), "n_exact": int(n_exact), "n_nearest": int(i1.size - n_exact),
        "n_unmatched_1": int(t1.size - i1.size), "n_unmatched_2": int(t2.size - i2.size),
        "n_dropped_zero_denominator": int((~good).sum()),
        "tolerance_s": float(tol),
        "sigma_source": "; ".join(src) if src else "supplied sem",
    }


# ---------------------------------------------------------------------------
# Normalisation and drift
# ---------------------------------------------------------------------------

def _window_mask(t, window):
    if window is None:
        return np.ones(t.size, dtype=bool)
    lo, hi = float(window[0]), float(window[1])
    if hi < lo:
        lo, hi = hi, lo
    return (t >= lo) & (t <= hi)


def normalize_to_window(t, y, a: Sequence[float], sem=None) -> dict:
    """Express y as percent of its mean over the window ``a = (t_start_s, t_end_s)``.

    Mean error: sqrt(sum sem_i^2)/N if sem given, else std(ddof=1)/sqrt(N). Per-point error:
    100*sqrt(s_i^2/m^2 + y_i^2 sigma_m^2/m^4), with s_i the supplied sem or else the robust per-point
    sigma. The correlation between y_i and m for points inside the window is ignored.
    """
    t, y, s = _clean_xy(t, y, sem)
    mask = _window_mask(t, a)
    n = int(mask.sum())
    if n < 2:
        raise ValueError("Normalisation window contains fewer than 2 points.")
    m = float(np.mean(y[mask]))
    if m == 0:
        raise ValueError("Window mean is zero; cannot normalise.")
    if s is not None:
        sm = float(np.sqrt(np.sum(s[mask] ** 2)) / n)
        si = s
        src = "supplied sem"
    else:
        sm = float(np.std(y[mask], ddof=1) / math.sqrt(n))
        si = np.full(t.size, robust_sigma(y))
        src = "window std/sqrt(N) for the mean; robust per-point sigma for each point"
    pct = 100.0 * y / m
    spct = 100.0 * np.sqrt((si / m) ** 2 + (y * sm / m**2) ** 2)
    return {"t_s": t, "percent": pct, "sigma_percent": spct, "window_mean": m, "window_mean_sigma": sm,
            "n_window": n, "sigma_source": src}


def drift_rate(t, y, window=None) -> dict:
    """OLS linear fit y = y0 + slope*t over ``window`` (t_start_s, t_end_s) or the whole trace.

    ``slope_per_s`` and ``slope_se_per_s`` are in y-units per second (SE from residual variance, N-2 dof);
    ``percent_per_min`` = 100*60*slope/mean(y over window), SE scaled the same way (uncertainty of the
    mean is neglected).
    """
    t, y, _ = _clean_xy(t, y)
    mask = _window_mask(t, window)
    tw, yw = t[mask], y[mask]
    n = tw.size
    if n < 3:
        raise ValueError("Drift window needs at least 3 points.")
    tc = tw - tw.mean()
    sxx = float(np.sum(tc**2))
    if sxx == 0:
        raise ValueError("Degenerate time window.")
    slope = float(np.sum(tc * (yw - yw.mean())) / sxx)
    resid = yw - (yw.mean() + slope * tc)
    se = float(math.sqrt(np.sum(resid**2) / (n - 2) / sxx))
    mean = float(yw.mean())
    scale = 100.0 * 60.0 / mean if mean != 0 else float("nan")
    return {"slope_per_s": slope, "slope_se_per_s": se, "window_mean": mean,
            "percent_per_min": slope * scale,
            "percent_per_min_se": se * abs(scale) if math.isfinite(scale) else float("nan"),
            "n": int(n), "t_start_s": float(tw[0]), "t_end_s": float(tw[-1]),
            "z": slope / se if se > 0 else float("inf")}
