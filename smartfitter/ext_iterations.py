"""Iteration fit and pooled-run endpoints, using the ordinary load/fit paths."""

from __future__ import annotations

from copy import deepcopy
import uuid

from nvfit.iterations import iteration_matrix, iteration_sources, selection_indices, combine_datasets
from . import core


@core.plain_endpoint
def combine_iterations(ws, runs=None, grid="identical", name="", path=None, mode="contrast"):
    path = path or "combined:" + uuid.uuid4().hex
    if not path.startswith("combined:"):
        raise ValueError("Combined dataset IDs must start with combined:.")
    recipe = {
        "path": path,
        "name": (name or "Combined iterations").strip()[:160],
        "grid": grid,
        "runs": deepcopy(runs or []),
    }
    # Validate before publishing; no source data or MAT files are modified.
    ds = combine_datasets(ws.dataset, recipe, mode)
    with ws._lock:
        ws.combined[path] = recipe
    tr = next(iter(ds.detectors.values()))
    return {"path": path, "recipe": recipe, "info": core.describe(ds, tr)}


# "Auto" model choices are resolved once, on the selected mean, and that model is used for every iteration.
# Otherwise Auto picks a different model per noisy iteration (seen on 317RabiFullRun: phase-ramp for #1, adaptive for
# #2-3), and the parameter-vs-iteration trend compares unrelated parameters.
RABI_MODES = {
    "phase_ramp": "Phase-ramp",
    "chirped": "Chirped cosine",
    "adaptive_pulse_area": "Adaptive pulse-area (recommended)",
    "long_damped": "Long damped Rabi",
    "constant": "Constant-frequency damped cosine",
}
RAMSEY_MODELS = {
    "RamseySimple": "Simple damped cosine",
    "RamseyHyperfine": "N14 hyperfine",
    "RamseyExtended": "Extended",
}
T1_PREPS = {"direct": "Direct T1", "direct_decay": "Direct decay", "normalized_1_minus_data": "Normalize then 1-data"}


def lock_model_choice(spec: dict, reference: dict) -> tuple[dict, dict]:
    locked, changes = deepcopy(spec), {}
    extras = reference.get("extras") or {}
    if locked.get("params") and not locked.get("params_model"):
        control_model = (extras.get("parameter_controls") or {}).get("model")
        if control_model:
            locked["params_model"] = control_model
    family = extras.get("rabi_model_family")
    if family in RABI_MODES and str(spec.get("rabi_mode", "Auto compare")).lower().startswith("auto"):
        locked["rabi_mode"] = changes["rabi_mode"] = RABI_MODES[family]
    prep = extras.get("t1_prep_mode")
    if prep in T1_PREPS and str(spec.get("t1_prep", "Auto")).lower() == "auto":
        locked["t1_prep"] = changes["t1_prep"] = T1_PREPS[prep]
    ramsey = RAMSEY_MODELS.get(str(reference.get("model", "")))
    if ramsey and str(spec.get("ramsey_model", "Auto")).lower() == "auto":
        locked["ramsey_model"] = changes["ramsey_model"] = ramsey
    polarity = str(extras.get("odmr_polarity", "")).lower()
    if polarity in ("peak", "dip") and str(spec.get("odmr_polarity", "Auto")).lower() == "auto":
        locked["odmr_polarity"] = changes["odmr_polarity"] = polarity.capitalize()
    return locked, changes


@core.analysis_result
def fit_iterations(ws, path, detector, mode, proc=None, spec=None):
    proc, spec = deepcopy(proc or {}), deepcopy(spec or {})
    tr = ws.trace(path, detector, mode)
    matrix = iteration_matrix(tr, mode)
    if matrix is None:
        raise ValueError("No complete per-iteration data are available for this observable/detector.")
    indices = selection_indices(matrix.shape[1], proc.get("iterations"))
    if not indices:
        raise ValueError("Select at least one iteration before fitting individually.")
    sources = iteration_sources(tr)
    try:
        reference = core.run_fit(ws, path, detector, mode, proc, spec)
    except Exception:  # the mean fit failing must not block the individual fits; Auto then stays per-iteration
        reference = {}
    spec, locked = lock_model_choice(spec, reference)
    rows = []
    for i in indices:
        single = {**proc, "iterations": [i]}
        row = {"index": i, "iteration": i + 1, "source": sources[i]}
        try:
            fit = core.run_fit(ws, path, detector, mode, single, spec)
            fit.pop("raw", None)
            row.update({"fit": fit, "status": fit["status"], "reason": fit["reason"]})
        except Exception as exc:
            row.update({"fit": None, "status": "ERROR", "reason": str(exc)})
        rows.append(row)
    note = "Independent fits; single iterations have no across-iteration SEM. Reported errors are ordinary model-fit errors, not between-iteration uncertainty."
    if locked:
        note += (
            f" Auto choice resolved on the selected mean ({reference.get('model')}) and used for every iteration: "
            + ", ".join(f"{k} = {v}" for k, v in locked.items())
            + "."
        )
    return {
        "rows": rows,
        "detector": tr.detector_id,
        "mode": mode,
        "processing": proc,
        "fit_spec": spec,
        "locked": locked,
        "reference_model": reference.get("model"),
        "note": note,
    }


ENDPOINTS = {"combine_iterations": combine_iterations, "fit_iterations": fit_iterations}
