"""SmartFitter endpoints: time-trace tools (steps, detector ratio, drift) and DEER helpers.

Registered through ENDPOINTS = {name: fn(ws, **kwargs)} (see server.make_handler).
Time traces store measured elapsed seconds in ``ExperimentTrace.x_ns`` (metadata ``x_unit`` == "s").
"""

from __future__ import annotations

from . import core

import numpy as np

from nvfit import deer_tools, trace_tools
from .core import arr, clean


def _load(ws, path, detector, mode, need_time=True):
    tr = ws.trace(path, detector, mode)
    meta = tr.metadata or {}
    unit = str(meta.get("x_unit") or "")
    if need_time:
        if tr.scan_dim != "time":
            raise ValueError(f"{tr.file_name} is a {tr.experiment_type} scan, not a time trace.")
        if unit not in ("s", ""):
            raise ValueError(f"Unexpected time-trace x unit {unit!r} (expected seconds).")
    x = np.asarray(tr.x_ns, dtype=float)
    y = np.asarray(tr.y, dtype=float)
    sem = None if tr.y_sem is None else np.asarray(tr.y_sem, dtype=float)
    info = {
        "detector": tr.detector_id,
        "value_unit": meta.get("value_unit"),
        "x_unit": unit or None,
        "x_label": tr.x_label,
        "y_label": tr.y_label,
        "reference_only": bool(meta.get("reference_only")),
        "file": tr.file_name,
    }
    if info["reference_only"]:
        info["caveat"] = (
            "Reference-only data: y is the reference fluorescence "
            f"({info['value_unit'] or 'native units'}), not a signal or contrast."
        )
    return x, y, sem, info


@core.analysis_result
def tt_steps(ws, path, detector=None, mode="signal", opts=None):
    t, y, _, info = _load(ws, path, detector, mode)
    o = opts or {}
    res = trace_tools.detect_steps(
        t,
        y,
        min_size_sigma=float(o.get("min_size_sigma", 4.0)),
        min_segment=int(o.get("min_segment", 5)),
        max_steps=int(o.get("max_steps", 100)),
    )
    return clean({**res, "info": info})


@core.analysis_result(detector_argument="den_detector")
def tt_ratio(ws, path, mode="signal", num_detector="detector2", den_detector="detector1"):
    t1, y1, s1, i1 = _load(ws, path, den_detector, mode)
    t2, y2, s2, i2 = _load(ws, path, num_detector, mode)
    if i1["detector"] == i2["detector"]:
        raise ValueError("Numerator and denominator detectors resolve to the same detector.")
    res = trace_tools.ratio(t1, y1, t2, y2, s1, s2)
    for k in ("t_s", "ratio", "sigma"):
        res[k] = arr(res[k])
    return clean({**res, "numerator": i2, "denominator": i1, "definition": f"{i2['detector']} / {i1['detector']}"})


@core.analysis_result
def tt_drift(ws, path, detector=None, mode="signal", window=None):
    t, y, _, info = _load(ws, path, detector, mode)
    return clean({**trace_tools.drift_rate(t, y, window), "info": info})


@core.analysis_result(reference_paths=("ref_path",))
def deer_ratio(ws, path, detector=None, mode="signal", ref_path=None, ref_detector=None):
    if not ref_path:
        raise ValueError("ref_path is required.")
    x, y, s, info = _load(ws, path, detector, mode, need_time=False)
    if info["reference_only"]:
        raise ValueError("The DEER file is reference-only (no signal); a DEER ratio would be fabricated.")
    xr, yr, sr, rinfo = _load(ws, ref_path, ref_detector or detector, mode, need_time=False)
    from nvfit.units import compatibility, trace_units

    xs, ys = compatibility(
        trace_units(ws.trace(path, detector, mode)),
        trace_units(ws.trace(ref_path, ref_detector or detector, mode)),
        label="DEER reference ratio",
    )
    xr, yr = xr * xs, yr * ys
    if sr is not None:
        sr = sr * abs(ys)
    res = deer_tools.deer_ratio(x, y, xr, yr, s, sr)
    for k in ("x", "ratio", "sem"):
        res[k] = arr(res[k])
    return clean({**res, "info": info, "ref_info": rinfo})


@core.analysis_result
def deer_resonance(ws, B_T=None, f_GHz=None, g=deer_tools.G_FREE):
    if B_T is None and f_GHz is None:
        raise ValueError("Give B_T or f_GHz.")
    out = {"g": float(g), "mu_B_over_h_GHz_per_T": deer_tools.MU_B_OVER_H_GHZ_PER_T}
    if B_T is not None:
        out.update({"B_T": float(B_T), "f_GHz": deer_tools.expected_target_resonance(float(B_T), g)})
    if f_GHz is not None:
        out.update({"f_in_GHz": float(f_GHz), "B_T_from_f": deer_tools.field_from_resonance(float(f_GHz), g)})
    return clean(out)


ENDPOINTS = {
    "tt_steps": tt_steps,
    "tt_ratio": tt_ratio,
    "tt_drift": tt_drift,
    "deer_ratio": deer_ratio,
    "deer_resonance": deer_resonance,
}
