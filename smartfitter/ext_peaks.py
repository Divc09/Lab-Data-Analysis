"""ODMR multi-peak "peak-pick only" (v1 parity: gui_app.py run_fit peak-pick branch). No model is fitted."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from . import core


def pick_peaks(x, y, polarity="Auto", ranges=None, prominence_frac=0.08):
    """v1 algorithm: find_peaks on y (negated for dips) with prominence = 0.08 * peak-to-peak; with ranges, the best
    detected peak inside each range (or the extreme sample there if none was detected)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 3:
        raise ValueError("Too few finite points to pick peaks.")
    pol = str(polarity or "Auto").lower()
    if pol == "auto":  # dips when the extremes below the median are larger than those above it
        med = float(np.median(y))
        pol = "dip" if med - float(np.min(y)) > float(np.max(y)) - med else "peak"
    yp = -y if pol == "dip" else y
    peaks, _ = find_peaks(yp, prominence=max(1e-9, prominence_frac * float(np.ptp(yp))))
    chosen = list(peaks)
    usable = [(lo, hi) for lo, hi in (ranges or []) if lo is not None or hi is not None]
    if usable:
        chosen = []
        for lo, hi in usable:
            lo = -np.inf if lo is None else float(lo)
            hi = np.inf if hi is None else float(hi)
            local = peaks[(x[peaks] >= lo) & (x[peaks] <= hi)]
            if not len(local):
                local = np.flatnonzero((x >= lo) & (x <= hi))
            if len(local):
                chosen.append(int(local[np.argmax(yp[local])]))
        chosen = sorted(set(chosen))
    return {
        "polarity": pol,
        "peaks": [{"x": float(x[i]), "y": float(y[i])} for i in chosen],
        "prominence": prominence_frac * float(np.ptp(yp)),
    }


@core.analysis_result
def odmr_pick_peaks(ws, path, detector, mode, proc=None, polarity="Auto", ranges=None, smoothed=False, prominence=0.08):
    payload = core.trace_payload(ws, path, detector, mode, proc or {})
    p = payload["proc"]
    y = p.get("smooth") if smoothed and p.get("smooth") is not None else p["y"]
    nan = lambda a: [np.nan if v is None else v for v in a]
    out = pick_peaks(nan(p["x"]), nan(y), polarity, ranges, float(prominence))
    out.update(
        {
            "detector": payload.get("detector", detector),
            "mode": mode,
            "source": "smoothed" if smoothed else "processed mean",
            "note": "Peak-pick only: detected sample positions (resolution = point spacing); no model, no uncertainties.",
        }
    )
    return out


ENDPOINTS = {"odmr_pick_peaks": odmr_pick_peaks}
