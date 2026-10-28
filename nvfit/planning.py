"""Measurement planning from a finished fit: convergence projection, next scan points, ODMR sensitivity.

Everything here is a linearized estimate around the current fit. It assumes the model is right, that new data have
the same noise as the residuals of the existing data, and (for convergence) that parameter errors shrink like
1/sqrt(iterations). The functions report those assumptions next to the numbers; they are planning aids, not
validated predictions.
"""

from __future__ import annotations

from dataclasses import replace
import math
import re

import numpy as np

from .prepared_fit import evaluate_result

GAMMA_E_HZ_PER_T = 28.024e9  # NV electron gyromagnetic ratio gamma_e / 2pi (same constant as nv_field)
LORENTZIAN_SHOT_FACTOR = 4.0 / (3.0 * math.sqrt(3.0))  # CW-ODMR shot-noise prefactor for a Lorentzian line
HZ_PER_UNIT = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


def acquisition_timing(source_path, iterations, points) -> dict:
    """Wall-clock duration from the start stamp in the file name to the file's last save.

    The files record no per-iteration timestamps, so this is the only timing available. It includes every overhead
    (sweeps, RF switching, saves); a value derived from it is conservative. Returns nulls when no start stamp exists
    or the span is implausible (e.g. a copied file whose mtime is the copy time).
    """
    import os
    from .scan_stack import parse_file_time

    out = {"duration_s": None, "per_iteration_s": None, "per_point_s": None,
           "source": "no start time in the file name; enter the time yourself"}
    try:
        start, source = parse_file_time(source_path)
        saved = os.path.getmtime(source_path)
    except OSError:
        return out
    if not source.startswith("filename") or not math.isfinite(start):
        return out
    duration = saved - start
    if not 0 < duration < 7 * 86400:
        out["source"] = "file-name start and last save disagree (copied or edited file?); enter the time yourself"
        return out
    out.update(duration_s=duration, source="estimated: last save minus the start time in the file name, includes overhead")
    if iterations and iterations > 0:
        out["per_iteration_s"] = duration / iterations
    if points and points > 0:
        out["per_point_s"] = duration / points
    return out


# ---------------------------------------------------------------------------
# 1. Convergence: how many more iterations reach a target precision
# ---------------------------------------------------------------------------

# Case-sensitive: lowercase c is the Rabi offset, uppercase C the ODMR contrast amplitude.
_LOCATION = re.compile(r"^(x0(_\d+)?|center(_\d+)?|y0|c|phi\d*|t0|delay|background|offset|baseline)$")


def is_location(name: str) -> bool:
    """Positions and offsets: their error relative to the value depends on where zero is, so only an absolute
    target is meaningful (planning.js mirrors this list)."""
    return bool(_LOCATION.match(name))


def prefix_counts(total: int, max_fits: int = 10) -> list[int]:
    """Cumulative iteration counts to refit, roughly log-spaced from 2 to ``total`` (always ending at ``total``)."""
    if total < 3:
        return [total]
    counts = np.unique(np.round(np.geomspace(2, total, min(max_fits, total - 1))).astype(int))
    return sorted(set(int(c) for c in counts if 2 <= c <= total) | {total})


def scaling_exponent(counts, errors) -> float | None:
    """Slope of log(error) against log(iterations); about -0.5 when averaging still helps."""
    pts = [(math.log(n), math.log(e)) for n, e in zip(counts, errors)
           if n and e is not None and math.isfinite(e) and e > 0]
    if len(pts) < 3 or len({p[0] for p in pts}) < 3:
        return None
    lx, ly = np.array(pts).T
    return float(np.polyfit(lx, ly, 1)[0])


def project_iterations(value: float, error: float, iterations: int, target_relative: float) -> dict:
    """Iterations needed for error/|value| <= target, assuming error ~ 1/sqrt(iterations)."""
    out = {"relative": None, "needed": None, "more": None}
    if not (math.isfinite(value) and math.isfinite(error) and error > 0 and value != 0 and iterations > 0):
        return out
    relative = error / abs(value)
    out["relative"] = relative
    if target_relative > 0:
        needed = math.ceil(iterations * (relative / target_relative) ** 2)
        out["needed"] = max(needed, iterations)
        out["more"] = max(needed - iterations, 0)
    return out


def convergence_summary(rows: list[dict], target_relative: float) -> list[dict]:
    """Per-parameter projection from cumulative fits.

    ``rows``: [{"iterations": n, "params": [{"name", "value", "error", "locked"}], "comparable": bool}], last row = all.
    """
    final = rows[-1]
    out = []
    for p in final["params"]:
        if p.get("locked"):
            continue
        series = [(r["iterations"], next((q for q in r["params"] if q["name"] == p["name"]), None))
                  for r in rows if r.get("comparable")]
        counts = [n for n, q in series if q]
        errors = [q.get("error") for _, q in series if q]
        exponent = scaling_exponent(counts, errors)
        value, error = p.get("value"), p.get("error")
        location = is_location(p["name"])
        projection = project_iterations(value if value is not None else math.nan,
                                        error if error is not None else math.nan, final["iterations"],
                                        0.0 if location else target_relative)
        note = ""
        if exponent is None:
            note = "too few comparable fits to check the 1/sqrt(N) assumption"
        elif exponent > -0.25:
            note = (f"error is not shrinking with averaging (exponent {exponent:+.2f}, expected -0.5): drift, model "
                    "mismatch or a poorly constrained parameter; more iterations may not help")
        elif exponent < -0.8:
            note = f"error fell faster than 1/sqrt(N) (exponent {exponent:+.2f}); early fits were poorly constrained"
        out.append({"name": p["name"], "unit": p.get("unit"), "value": value, "error": error, "location": location,
                    "exponent": exponent, **projection, "note": note})
    return out


# ---------------------------------------------------------------------------
# 2. Next scan points: where one more point most reduces a parameter's error
# ---------------------------------------------------------------------------

def free_jacobian(result, x: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """d model / d free parameter at ``x`` (central differences), in the space the model was fitted."""
    params = np.asarray(result.params, float)
    free = list(range(len(params))) if result.free_mask is None else [i for i, f in enumerate(result.free_mask) if f]
    cols = []
    for i in free:
        step = 1e-6 * abs(params[i]) if params[i] != 0 else 1e-6
        hi, lo = params.copy(), params.copy()
        hi[i] += step
        lo[i] -= step
        cols.append((evaluate_result(replace(result, params=hi), x) - evaluate_result(replace(result, params=lo), x)) / (2 * step))
    return np.column_stack(cols) if cols else np.zeros((len(x), 0)), free


def candidate_grid(x: np.ndarray, x_min: float, x_max: float, limit: int = 400) -> np.ndarray:
    """Candidate positions on the existing scan step when it is regular, else evenly spaced."""
    xs = np.unique(np.asarray(x, float)[np.isfinite(x)])
    steps = np.diff(xs)
    if steps.size and np.allclose(steps, steps[0], rtol=1e-3) and steps[0] > 0:
        start = xs[0] + math.ceil((x_min - xs[0]) / steps[0] - 1e-9) * steps[0]
        count = int(math.floor((x_max - start) / steps[0] + 1e-9)) + 1
        if 2 <= count <= limit:
            return start + steps[0] * np.arange(count)
    return np.linspace(x_min, x_max, limit)


def suggest_points(result, x, target_y, parameter: str, count: int = 10, x_min: float | None = None,
                   x_max: float | None = None) -> dict:
    """Greedy choice of ``count`` new scan points that most reduce the linearized variance of ``parameter``.

    Each added point is one more scan position measured with the same averaging (and so the same noise) as the
    existing points. Points may repeat. The comparison adds the same number of points spread like the current scan.
    """
    x = np.asarray(x, float)
    resid = np.asarray(target_y, float) - np.asarray(result.y_fit, float)
    ok = np.isfinite(x) & np.isfinite(resid)
    x, resid = x[ok], resid[ok]
    names = list(result.param_names)
    if parameter not in names:
        raise ValueError(f"{parameter} is not a parameter of {result.model_name}.")
    jac, free = free_jacobian(result, x)
    if names.index(parameter) not in free:
        raise ValueError(f"{parameter} is locked; choose a free parameter.")
    n, p = jac.shape
    if n <= p:
        raise ValueError("The fit has no spare degrees of freedom to estimate the noise; add points first.")
    sigma2 = float(np.sum(resid**2) / (n - p))
    info = jac.T @ jac / sigma2
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        raise ValueError("The parameters are not identifiable from the current points (singular information matrix).")
    k = free.index(names.index(parameter))
    current = math.sqrt(max(cov[k, k], 0.0))
    lo = float(np.min(x)) if x_min is None else float(x_min)
    hi = float(np.max(x)) if x_max is None else float(x_max)
    if not hi > lo:
        raise ValueError("The candidate range is empty; the upper limit must exceed the lower one.")
    grid = candidate_grid(x, lo, hi)
    gj, _ = free_jacobian(result, grid)
    finite = np.all(np.isfinite(gj), axis=1)
    grid, gj = grid[finite], gj[finite]
    if not len(grid):
        raise ValueError("The model cannot be evaluated on the candidate range.")
    chosen = []
    v = cov.copy()
    for _ in range(int(count)):
        vj = gj @ v  # rows: V j_c
        gain = vj[:, k] ** 2 / (sigma2 + np.einsum("ij,ij->i", vj, gj))
        best = int(np.argmax(gain))
        chosen.append(float(grid[best]))
        u = vj[best]
        v = v - np.outer(u, u) / (sigma2 + float(u @ gj[best]))
    after = math.sqrt(max(v[k, k], 0.0))
    uniform = current * math.sqrt(n / (n + count))
    counts: dict[float, int] = {}
    for c in chosen:
        counts[c] = counts.get(c, 0) + 1
    value = float(result.params[names.index(parameter)])
    return {
        "parameter": parameter, "value": value, "model": result.model_name,
        "points": [{"x": c, "repeats": r} for c, r in sorted(counts.items())],
        "count": int(count), "range": [lo, hi], "existing_points": int(n), "noise_rms": math.sqrt(sigma2),
        "error_now": current, "error_after": after, "error_uniform": uniform,
        "equivalent_uniform_points": int(round(n * ((current / after) ** 2 - 1))) if after > 0 else None,
        "fit_error": float(result.errors[names.index(parameter)]) if np.isfinite(result.errors[names.index(parameter)]) else None,
        "note": "Linearized at the current fit. Assumes the model is right and each new point has the residual noise of "
                "the existing points (same averaging per point). Repeats mean measuring that position more than once. "
                "'Spread like the current scan' adds the same number of points proportionally to the existing grid. "
                "σ here is the ordinary least-squares error. The fit's ± uses the robust soft-L1 loss, whose Jacobian "
                "down-weights large residuals while the noise level comes from the raw residuals, so it is usually "
                "larger (1.3-1.8x on the test files); compare the percentage improvements rather than the absolute σ.",
    }


# ---------------------------------------------------------------------------
# D. ODMR magnetic sensitivity
# ---------------------------------------------------------------------------

def odmr_sensitivity(result, x, residuals, x_unit: str, mode: str, reference_rate: float | None = None) -> dict | None:
    """Empirical slope-and-noise sensitivity of a fitted ODMR line, plus the CW shot-noise limit when possible.

    Empirical: delta_B per point = sigma_y / (gamma_e * max|dy/df|), with sigma_y the residual RMS (dof-corrected)
    of the averaged points. Multiplying by sqrt(integration time per point) gives eta in T/sqrt(Hz).
    Shot-noise limit (Dreau et al., PRB 84, 195204, 2011): eta = 4/(3 sqrt 3) * FWHM / (gamma_e * C * sqrt(R)),
    with C the fractional contrast and R the detected count rate.
    """
    if not result.model_name.startswith("ODMR"):
        return None
    hz = HZ_PER_UNIT.get(str(x_unit or "").strip())
    if hz is None:
        return {"error": f"Sensitivity needs a frequency axis in Hz, kHz, MHz or GHz (this file uses {x_unit or 'no unit'})."}
    xs = np.asarray(x, float)
    res = np.asarray(residuals, float)
    ok = np.isfinite(xs) & np.isfinite(res)
    n_free = int(sum(result.free_mask)) if result.free_mask is not None else len(result.params)
    if ok.sum() <= n_free:
        return {"error": "Too few points to estimate the noise."}
    noise = float(math.sqrt(np.sum(res[ok] ** 2) / (ok.sum() - n_free)))
    dense = np.linspace(float(np.min(xs[ok])), float(np.max(xs[ok])), 4001)
    y = evaluate_result(result, dense)
    slope = np.gradient(y, dense) / hz  # y units per Hz
    best = int(np.nanargmax(np.abs(slope)))
    max_slope = float(abs(slope[best]))
    out = {
        "noise_rms": noise, "max_slope_per_hz": max_slope, "bias_frequency": float(dense[best]), "x_unit": x_unit,
        "delta_b_per_point_t": noise / (GAMMA_E_HZ_PER_T * max_slope) if max_slope > 0 else None,
        "mode": mode, "shot_noise": None,
        "note": "Empirical: residual noise of the averaged points divided by the steepest fitted slope, converted with "
                "gamma_e = 28.024 GHz/T. Multiply by sqrt(total integration time per point) for eta. Uses the fit's "
                "observable; contrast = (reference - signal) / reference.",
    }
    p = dict(zip(result.param_names, np.asarray(result.params, float)))
    peaks = [(p["C"], p["w"])] if "C" in p else [(p[f"C{i}"], p[f"w{i}"]) for i in range(1, 9) if f"C{i}" in p]
    if not peaks:
        return out
    # The steepest line sets the bias point; use its contrast and width for the shot-noise limit.
    amp, hwhm = max(peaks, key=lambda cw: abs(cw[0]) / max(abs(cw[1]), 1e-300))
    if mode == "contrast":
        contrast = abs(amp)
    elif mode == "signal" and p.get("y0"):
        contrast = abs(amp) / abs(p["y0"])
    else:
        contrast = None
    if contrast and reference_rate and reference_rate > 0 and hwhm:
        fwhm_hz = 2.0 * abs(hwhm) * hz
        eta = LORENTZIAN_SHOT_FACTOR * fwhm_hz / (GAMMA_E_HZ_PER_T * contrast * math.sqrt(reference_rate))
        out["shot_noise"] = {"eta_t_per_rthz": eta, "contrast": contrast, "fwhm_hz": fwhm_hz, "rate_per_s": reference_rate,
                             "note": "Photon shot-noise limit for CW ODMR on a Lorentzian line (Dreau et al. 2011), "
                                     "with R the mean detected reference rate. Ignores laser/MW noise and overhead."}
    return out
