"""Bootstrap uncertainty endpoint for SmartFitter (auto-registered by server.make_handler)."""
from __future__ import annotations

import dataclasses

import numpy as np

from nvfit.bootstrap import bootstrap_fit
from nvfit.fit_workflows import aligned_indices, aligned_sem

from . import core

_RAMSEY_SPEC_NAME = {"RamseySimple": "Simple damped cosine", "RamseyHyperfine": "N14 hyperfine", "RamseyExtended": "Extended"}


def bootstrap(ws, path, detector, mode, proc, spec, n_boot=200):
    """Bootstrap errors for the fit selected by ``spec`` (same data selection as ``fit``).

    Iteration resampling is used when the per-iteration matrix reproduces the fitted trace
    (unprocessed mean of complete iterations); otherwise residual resampling.
    """
    proc, spec = proc or {}, dict(spec or {})
    tr, x, y, exp_type, cfg, locks = core.fit_inputs(ws, path, detector, mode, proc, spec)
    cfg = dataclasses.replace(cfg, multistart=min(int(cfg.multistart), 4))
    spec["multistart"] = cfg.multistart
    base, _ = core.fit_dispatch(tr, x, y, exp_type, cfg, locks, spec)
    if base.model_name in _RAMSEY_SPEC_NAME:
        spec["ramsey_model"] = _RAMSEY_SPEC_NAME[base.model_name]  # keep the same model across resamples

    matrix = None
    full = core._iteration_matrix(tr, mode)
    idx = aligned_indices(tr, x)
    if full is not None and idx is not None:
        sub = full[idx]
        with np.errstate(all="ignore"):
            mean = np.nanmean(sub, axis=1)
        if np.all(np.isfinite(mean)) and np.allclose(mean, y, rtol=1e-6, atol=1e-9 * max(1.0, float(np.max(np.abs(y))))):
            matrix = sub

    def fit_callable(xx, yy, sem):
        xx = np.asarray(xx, float)
        sem = aligned_sem(tr, xx) if sem is None else np.asarray(sem, float)  # residual mode: keep the data SEM
        t = dataclasses.replace(tr, x_ns=xx, y=np.asarray(yy, float), y_sem=sem)
        return core.fit_dispatch(t, xx, yy, exp_type, cfg, locks, spec)[0]

    res = bootstrap_fit(fit_callable, x, y, n_boot=int(n_boot), seed=0, iteration_matrix=matrix)
    analytic = {n: (float(e) if np.isfinite(e) else None) for n, e in zip(base.param_names, np.asarray(base.errors, float))}
    out = {"params": {}, "n_success": res["n_success"], "n_boot": res["n_boot"], "method": res["method"],
           "model": base.model_name, "note": "bootstrap refits use multistart<=4; std is the 1-sigma bootstrap error"}
    for n, st in res["params"].items():
        out["params"][n] = {**st, "value": res["baseline"].get(n), "analytic_error": analytic.get(n)}
    out.update({k: v for k, v in out["params"].items()})  # {param: {...}} at top level as requested
    return core.clean(out)


ENDPOINTS = {"bootstrap": bootstrap}
