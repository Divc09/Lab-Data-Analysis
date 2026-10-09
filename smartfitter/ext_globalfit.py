"""SmartFitter endpoint: global fit across several processed traces with shared parameters."""
from __future__ import annotations

import numpy as np

from nvfit.global_fit import MODEL_PARAMS, global_fit, model_curve

from . import core


def _arr(v):
    return None if v is None else np.asarray(v, dtype=float)


def global_fit_endpoint(ws, items, model, shared, spec=None):
    """items = [{path, detector, mode, proc}]; x/y are proc.fit_x / proc.fit_y (x unit = file x unit, ns for time axes)."""
    spec = spec or {}
    datasets, labels = [], []
    for it in items:
        payload = core.trace_payload(ws, it["path"], it.get("detector"), it.get("mode", "contrast"), it.get("proc"))
        proc = payload.get("proc")
        if not proc or proc.get("fit_x") is None:
            raise ValueError(f"No 1D fit data for {it['path']}")
        x, y = _arr(proc["fit_x"]), _arr(proc["fit_y"])
        sem = _arr(proc.get("sem"))
        if sem is not None and sem.shape != x.shape:
            sem = None
        datasets.append((x, y, sem) if sem is not None else (x, y))
        labels.append({"path": it["path"], "detector": payload.get("detector"),
                       "label": (payload.get("info") or {}).get("file_name") or str(it["path"])})
    bounds = spec.get("bounds")
    if isinstance(bounds, dict):
        bounds = {k: tuple(v) for k, v in bounds.items()}
    res = global_fit(model, datasets, list(shared or []), seeds=spec.get("seeds"), bounds=bounds,
                     multistart=int(spec.get("multistart", 12)))
    for d, lab in zip(res["datasets"], labels):
        x = d["x"]
        xd = np.linspace(float(np.min(x)), float(np.max(x)), 800)
        d["curve"] = {"x": xd, "y": model_curve(model, d["params"], xd)}
        d["x"], d["y"], d["y_fit"] = x, d["y"], d["y_fit"]
        d.update(lab)
    res["available_params"] = MODEL_PARAMS[model]
    return core.clean(res)


ENDPOINTS = {"global_fit": global_fit_endpoint}
