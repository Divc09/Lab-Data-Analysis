"""Refit iteration means after omissions without changing the source or selection."""

from __future__ import annotations

from copy import deepcopy
import math

from nvfit.iterations import iteration_matrix, iteration_sources, selection_indices
from . import core
from .ext_iterations import lock_model_choice


def _summary(fit):
    # Keep sessions small: these diagnostics need parameters, not a curve for every omission.
    return {k: fit.get(k) for k in ("model", "params", "status", "reason", "success", "r2", "extras", "provenance")}


@core.analysis_result
def iteration_sensitivity(ws, path, detector, mode, proc=None, spec=None, omit_groups=None):
    proc, requested_spec = deepcopy(proc or {}), deepcopy(spec or {})
    if requested_spec.get("robust"):
        raise ValueError("Iteration sensitivity requires ordinary fits; turn off robust fitting.")
    tr = ws.trace(path, detector, mode)
    matrix = iteration_matrix(tr, mode)
    if matrix is None or tr.scan_dim == "scan2d" or tr.experiment_type == "TimeTrace":
        raise ValueError("Iteration sensitivity requires a 1D per-iteration matrix.")
    indices = selection_indices(matrix.shape[1], proc.get("iterations"))
    if len(indices) < 2:
        raise ValueError("Select at least two iterations for an omission comparison.")
    groups = [[i] for i in indices] if omit_groups is None else omit_groups
    if not isinstance(groups, list) or not groups or len(groups) > 200:
        raise ValueError("Test between 1 and 200 omission groups; select a smaller baseline if needed.")
    groups = [selection_indices(matrix.shape[1], group) if isinstance(group, list) else [] for group in groups]
    for group in groups:
        if not group or not set(group).issubset(indices) or len(group) >= len(indices):
            raise ValueError("Each omission group must remove selected iterations and retain at least one.")
    # Explicitly recompute even the full mean from the same complete-iteration matrix.
    # Loader defaults can include partial acquisitions in signal/reference averages.
    baseline_proc = {**proc, "iterations": indices}
    baseline_proc.pop("iteration_review", None)
    baseline = core.run_fit(ws, path, detector, mode, baseline_proc, requested_spec)
    if baseline.get("success") is False or not any(
        isinstance(p.get("value"), (int, float)) and math.isfinite(p["value"]) for p in baseline.get("params", [])
    ):
        raise ValueError("The full-selection fit did not converge to usable parameters. Review the model/range first.")
    spec, locked = lock_model_choice(requested_spec, baseline)
    sources = iteration_sources(tr)
    rows = []
    names = {p["name"] for p in baseline["params"]}
    for group in groups:
        retained = [i for i in indices if i not in group]
        row = {"omitted": group, "retained": retained, "sources": [sources[i] for i in group]}
        try:
            fit = core.run_fit(ws, path, detector, mode, {**baseline_proc, "iterations": retained}, spec)
            comparable = (
                fit.get("model") == baseline.get("model") and {p["name"] for p in fit.get("params", [])} == names
            )
            comparable = comparable and fit.get("success") is not False
            reason = fit.get("reason", "")
            if not comparable:
                reason += "; model/parameter set changed or optimizer failed; parameter changes are not comparable"
            row.update(fit=_summary(fit), comparable=comparable, status=fit.get("status"), reason=reason)
        except Exception as exc:
            row.update(fit=None, comparable=False, status="ERROR", reason=str(exc))
        rows.append(row)
    return {
        "baseline": _summary(baseline),
        "rows": rows,
        "selected": indices,
        "sources": [sources[i] for i in indices],
        "detector": tr.detector_id,
        "mode": mode,
        "processing": proc,
        "requested_spec": requested_spec,
        "fit_spec": spec,
        "locked": locked,
        "note": "Each row refits the retained iteration mean with the same processing, range, seeds/bounds and weighting settings. "
        "Available Auto choices are resolved on the full selection. SEM weights, when enabled, are recomputed per subset. "
        "Sensitivity is not an outlier test or a confidence interval. Large changes can reflect drift, noise or an unstable fit. "
        "Errors are ordinary fit errors; fits share observations. With two baseline iterations each omission fits one trace without across-iteration SEM.",
    }


ENDPOINTS = {"iteration_sensitivity": iteration_sensitivity}
