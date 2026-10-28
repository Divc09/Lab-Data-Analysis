"""Measurement planning: convergence with iterations and the most informative next scan points."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from nvfit.iterations import iteration_matrix, selection_indices
from nvfit import planning
from . import core
from .ext_iterations import lock_model_choice

PREFERRED = ("T1", "T2", "T2_star", "tau", "f", "f1", "x0", "x0_1", "center_1")


def _summary(fit, n):
    return {"iterations": n, "model": fit.get("model"), "status": fit.get("status"), "r2": fit.get("r2"),
            "success": fit.get("success"),
            "params": [{k: p.get(k) for k in ("name", "unit", "value", "error", "locked")} for p in fit.get("params", [])]}


@core.analysis_result
def fit_convergence(ws, path, detector, mode, proc=None, spec=None, target_percent=5.0):
    """Refit cumulative iteration means (first 2, 3, ... N in acquisition order) and project the iterations needed."""
    proc, requested_spec = deepcopy(proc or {}), deepcopy(spec or {})
    if requested_spec.get("robust"):
        raise ValueError("Convergence uses ordinary fits; turn off robust fitting.")
    tr = ws.trace(path, detector, mode)
    matrix = iteration_matrix(tr, mode)
    if matrix is None or tr.scan_dim == "scan2d" or tr.experiment_type == "TimeTrace":
        raise ValueError("Convergence needs a 1D scan with per-iteration data.")
    indices = selection_indices(matrix.shape[1], proc.get("iterations"))
    if len(indices) < 3:
        raise ValueError("Convergence needs at least 3 selected iterations.")
    target = float(target_percent) / 100.0
    if not target > 0:
        raise ValueError("The target precision must be a positive percentage.")
    base_proc = {**proc, "iterations": indices}
    base_proc.pop("iteration_review", None)
    baseline = core.run_fit(ws, path, detector, mode, base_proc, requested_spec)
    if baseline.get("success") is False:
        raise ValueError("The fit with every selected iteration did not converge. Review the model and range first.")
    fit_spec, locked = lock_model_choice(requested_spec, baseline)
    names = {p["name"] for p in baseline["params"]}
    rows = []
    for n in planning.prefix_counts(len(indices)):
        if n == len(indices):
            fit = baseline
        else:
            try:
                fit = core.run_fit(ws, path, detector, mode, {**base_proc, "iterations": indices[:n]}, fit_spec)
            except Exception as exc:
                rows.append({"iterations": n, "params": [], "status": "ERROR", "reason": str(exc), "comparable": False})
                continue
        row = _summary(fit, n)
        row["comparable"] = (fit.get("model") == baseline.get("model") and fit.get("success") is not False
                             and {p["name"] for p in fit.get("params", [])} == names)
        rows.append(row)
    meta = tr.metadata or {}
    timing = planning.acquisition_timing(tr.source_path, meta.get("completed_iterations") or matrix.shape[1], len(tr.x_ns))
    return {
        "rows": rows, "summary": planning.convergence_summary(rows, target), "target_percent": float(target_percent),
        "selected": indices, "detector": tr.detector_id, "mode": mode, "model": baseline.get("model"),
        "processing": proc, "requested_spec": requested_spec, "fit_spec": fit_spec, "locked": locked,
        "timing": timing, "acquisition": {"completed": meta.get("completed_iterations"), "target": meta.get("target_iterations"),
                                          "status": meta.get("run_status")},
        "note": "Each row fits the mean of the first N selected iterations (acquisition order) with the same settings; "
                "Auto choices are fixed from the full fit. Projections assume errors fall as 1/sqrt(N) and that "
                "the fitted value stays put. The measured exponent checks that assumption on these data.",
    }


@core.analysis_result
def suggest_next_points(ws, path, detector, mode, proc=None, spec=None, parameter=None, count=10, x_min=None, x_max=None):
    """Scan positions that would most reduce one parameter's error, from the current fit."""
    proc, spec = deepcopy(proc or {}), deepcopy(spec or {})
    if spec.get("robust"):
        raise ValueError("Point suggestions use ordinary fits; turn off robust fitting.")
    count = int(count)
    if not 1 <= count <= 500:
        raise ValueError("Suggest between 1 and 500 points.")
    tr, x, y, exp_type, cfg, locks = core.fit_inputs(ws, path, detector, mode, proc, spec)
    result, _ = core.fit_dispatch(tr, x, y, exp_type, cfg, locks, spec)
    if not result.success:
        raise ValueError("The fit did not converge; suggestions need a converged fit.")
    free = [n for i, n in enumerate(result.param_names) if result.free_mask is None or result.free_mask[i]]
    if parameter not in free:
        parameter = next((n for n in PREFERRED if n in free), free[0] if free else None)
    if parameter is None:
        raise ValueError("Every parameter is locked.")
    plot_y = (result.extras or {}).get("plot_y")
    target_y = np.asarray(plot_y, float) if plot_y is not None else y
    value = float(result.params[list(result.param_names).index(parameter)])
    lo = float(np.min(x)) if x_min is None else float(x_min)
    if x_max is None:
        hi = float(np.max(x))
        if parameter in core.DECAY_PARAMS and math.isfinite(value) and value > 0:
            # Decays are pinned down near 1-2 lifetimes; let the search reach past a short scan, within reason.
            hi = min(max(hi, 5.0 * value), 10.0 * hi)
    else:
        hi = float(x_max)
    out = planning.suggest_points(result, x, target_y, parameter, count, lo, hi)
    meta = tr.metadata or {}
    return {**out, "free": free, "detector": tr.detector_id, "mode": mode, "processing": proc, "spec": spec,
            "x_unit": meta.get("x_unit") or "", "fit_target_space": "plot_y" if plot_y is not None else "data"}


ENDPOINTS = {"fit_convergence": fit_convergence, "suggest_next_points": suggest_next_points}
