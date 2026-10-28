"""SmartFitter endpoints: magnetic field from NV ODMR resonance frequencies (see nvfit/nv_field.py).

Frequencies are GHz at the API.  ODMR files saved by the MATLAB acquisition store the SRS RF axis in
GHz (verified on TestData/NewCodebaseTests/ODMR_*.mat: 2.114-2.122 GHz, metadata scan_parameter 'frequency'),
so the default x unit is GHz; pass spec["x_unit"] in {"GHz","MHz","Hz"} to override.  If unset, the unit is
inferred from the magnitude of the fitted centres and the inference is reported.
"""

from __future__ import annotations

import numpy as np

from nvfit.nv_field import fit_field
from . import core

_TO_GHZ = {"ghz": 1.0, "mhz": 1e-3, "khz": 1e-6, "hz": 1e-9}


@core.analysis_result
def nv_field(ws, freqs_ghz, errs=None, fit_D=False, fit_E=False, mode="auto", D=2.870, E=0.0):
    """Field from explicit resonance frequencies [GHz]; errs = 1-sigma per line [GHz]."""
    res = fit_field(freqs_ghz, errs, mode=mode, D=float(D), E=float(E), fit_D=bool(fit_D), fit_E=bool(fit_E))
    return core.clean(res.to_dict())


def _centres(fit: dict) -> tuple[list[float], list[float | None]]:
    by = {p["name"]: p for p in fit["params"]}
    names = [n for n in by if n == "x0" or n.startswith("x0_")]
    names.sort(key=lambda n: (0 if n == "x0" else int(n.split("_")[1])))
    return [by[n]["value"] for n in names], [by[n].get("error") for n in names]


@core.analysis_result
def nv_field_from_fit(ws, path, detector=None, mode="contrast", proc=None, spec=None):
    """Run the ODMR multi-peak fit, then convert fitted centres (x0, x0_1, ...) to a field."""
    spec = dict(spec or {})
    unit = spec.pop("x_unit", None)
    nv_mode = spec.pop("field_mode", "auto")
    fit_D, fit_E = bool(spec.pop("fit_D", False)), bool(spec.pop("fit_E", False))
    spec.setdefault("odmr_multi", True)
    spec.setdefault("odmr_peaks", 8)
    spec.setdefault("odmr_auto_peaks", True)
    fit = core.run_fit(ws, path, detector, mode, proc or {}, spec)
    centres, errs = _centres(fit)
    if not centres:
        raise ValueError("The fit returned no resonance centres (x0); is this an ODMR trace?")
    c = np.asarray(centres, float)
    if unit is None:
        med = float(np.median(np.abs(c)))
        unit, how = ("GHz" if 0.05 < med < 20 else "MHz" if 50 < med < 2e4 else "Hz"), "inferred from magnitude"
    else:
        how = "specified"
    k = _TO_GHZ[str(unit).lower()]
    f_ghz = c * k
    s_ghz = None
    if all(e is not None and e > 0 for e in errs):
        s_ghz = np.asarray(errs, float) * k
    res = fit_field(f_ghz, s_ghz, mode=nv_mode, fit_D=fit_D, fit_E=fit_E).to_dict()
    res["notes"] = list(res["notes"]) + [f"x unit {unit} ({how}); centres converted to GHz."]
    return core.clean(
        {
            "field": res,
            "centres_ghz": f_ghz.tolist(),
            "centre_errors_ghz": None if s_ghz is None else s_ghz.tolist(),
            "fit": {k2: fit[k2] for k2 in ("model", "r2", "status", "reason", "params", "x", "y_fit") if k2 in fit},
        }
    )


ENDPOINTS = {"nv_field": nv_field, "nv_field_from_fit": nv_field_from_fit}
