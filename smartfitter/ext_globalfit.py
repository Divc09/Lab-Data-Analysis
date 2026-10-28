"""SmartFitter endpoint: global fit across several processed traces with shared parameters."""

from __future__ import annotations

import numpy as np

from nvfit.global_fit import MODEL_PARAMS, global_fit, model_curve

from . import core
from nvfit.units import compatibility, trace_units, unit_scale


def _arr(v):
    return None if v is None else np.asarray(v, dtype=float)


@core.owned_result
def global_fit_endpoint(ws, items, model, shared, spec=None):
    """items = [{path, detector, mode, proc}]; x/y are proc.fit_x / proc.fit_y (x unit = file x unit, ns for time axes)."""
    spec = spec or {}
    datasets, labels, analyses = [], [], []
    first_units = None
    calculation_unit = None
    for it in items:
        payload = core.trace_payload(ws, it["path"], it.get("detector"), it.get("mode", "contrast"), it.get("proc"))
        proc = payload.get("proc")
        if not proc or proc.get("fit_x") is None:
            raise ValueError(f"No 1D fit data for {it['path']}")
        current_units = trace_units(ws.trace(it["path"], payload["detector"], it.get("mode", "contrast")))
        if first_units is None:
            first_units = current_units
            canonical_units = {"TimeTraceDecay": "s", "ODMR": "GHz", "GaussianPeaks": "um"}
            candidate = canonical_units.get(model, "ns")
            try:
                unit_scale(current_units["x_unit"], candidate)
                calculation_unit = candidate
            except ValueError:
                if model != "GaussianPeaks":
                    raise ValueError(f"{model} requires {candidate}-compatible x units; got {current_units['x_unit']}")
                calculation_unit = current_units["x_unit"]
        _, yscale = compatibility(first_units, current_units, label="Global fit")
        xscale = unit_scale(current_units["x_unit"], calculation_unit)
        x, y = _arr(proc["fit_x"]) * xscale, _arr(proc["fit_y"]) * yscale
        sem = _arr(proc.get("sem"))
        if sem is not None:
            sem = sem * abs(yscale)
        if sem is not None and sem.shape != x.shape:
            sem = None
        datasets.append((x, y, sem) if sem is not None else (x, y))
        analyses.append(payload["analysis"])
        labels.append(
            {
                "path": it["path"],
                "detector": payload.get("detector"),
                "label": (payload.get("info") or {}).get("file_name") or str(it["path"]),
            }
        )
    bounds = spec.get("bounds")
    if isinstance(bounds, dict):
        bounds = {k: tuple(v) for k, v in bounds.items()}
    res = global_fit(
        model,
        datasets,
        list(shared or []),
        seeds=spec.get("seeds"),
        bounds=bounds,
        multistart=int(spec.get("multistart", 12)),
    )
    for d, lab in zip(res["datasets"], labels):
        x = d["x"]
        xd = np.linspace(float(np.min(x)), float(np.max(x)), 800)
        d["curve"] = {"x": xd, "y": model_curve(model, d["params"], xd)}
        d["x"], d["y"], d["y_fit"] = x, d["y"], d["y_fit"]
        d.update(lab)
    res["available_params"] = MODEL_PARAMS[model]
    res["calculation_units"] = {"x": calculation_unit, "y": first_units["y_unit"]}
    res["analyses"] = analyses
    res["request"] = {"items": items, "model": model, "shared": shared, "spec": spec}
    from nvfit.analysis_identity import analysis_descriptor, result_envelope

    revisions = {a["descriptor"]["source"]: a["source_revision"] for a in analyses}
    dependencies = {**revisions, **{p: r for a in analyses for p, r in a["dependencies"].items()}}
    descriptor = analysis_descriptor(
        "aggregate",
        [a["descriptor"]["detector"] for a in analyses],
        first_units["observable"],
        {"items": [a["descriptor"] for a in analyses]},
    )
    res["analysis"] = result_envelope(descriptor, revisions, res["request"], dependencies, "global_fit")
    return core.clean(res)


ENDPOINTS = {"global_fit": global_fit_endpoint}
