"""Global (multi-dataset) least-squares fits with selectable shared parameters.

Conventions
-----------
* Model callables come from ``nvfit.models.MODEL_REGISTRY`` with signature ``f(x, *params)``.
  Parameters carry the units of the x axis / y axis of the data (e.g. T2 in the x unit,
  frequency in 1/x-unit).  Nothing is rescaled.
* Residuals are ``sqrt(w) * (model - y)`` concatenated over datasets.  ``w`` is
  ``1/sem**2`` when *every* dataset has a finite positive SEM, otherwise 1; user
  ``weights`` multiply this.  Total chi2 is therefore a true chi2 only when SEM is given.
* Parameter errors: ``nvfit.fit_engine._estimate_covariance`` with cost = 0.5 * sum(res**2)
  (errors are scaled by the reduced chi2, i.e. they do not assume SEM is absolute).
* AIC/BIC use the pooled Gaussian form ``N ln(RSS/N) + 2k`` / ``N ln(RSS/N) + k ln N`` for
  both the global fit and the sum of independent fits, so they are directly comparable.
  Delta = global - independent; negative favours sharing.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from .fit_engine import _estimate_covariance
from .models import MODEL_REGISTRY

# Parameter names of the models supported for global fitting (signature f(x, *params)).
MODEL_PARAMS: dict[str, list[str]] = {
    "SpinEcho": ["y0", "A", "T2", "n"],
    "T1Exp": ["y0", "A", "T1"],
    "T1Stretch": ["y0", "A", "T1", "n"],
    "T1NormExp": ["y0", "A", "T1"],
    "T1NormStretch": ["y0", "A", "T1", "n"],
    "T1DecayExp": ["y0", "A", "T1"],
    "T1DecayStretch": ["y0", "A", "T1", "n"],
    "RamseySimple": ["y0", "A", "f", "phi", "tau", "n"],
    "RabiCosine": ["c", "a", "T", "beta", "f", "phi"],
    "ODMR": ["y0", "C", "w", "x0"],
    "GaussianPeaks": ["background", "amplitude", "center", "sigma"],  # one peak
    "TimeTraceDecay": ["background", "amplitude", "tau_s"],
}


def _dominant_frequency(x: np.ndarray, y: np.ndarray) -> float:
    """Dominant oscillation frequency (1/x-unit) from an FFT on a uniform resampling."""
    order = np.argsort(x)
    xs, ys = x[order], y[order] - np.mean(y)
    n = len(xs)
    if n < 6 or xs[-1] <= xs[0]:
        return 1.0 / max(np.ptp(x), 1e-12)
    xu = np.linspace(xs[0], xs[-1], n)
    yu = np.interp(xu, xs, ys)
    spec = np.abs(np.fft.rfft(yu * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=xu[1] - xu[0])
    spec[0] = 0.0
    return float(freqs[int(np.argmax(spec))]) or 1.0 / max(np.ptp(x), 1e-12)


def _defaults(model: str, x: np.ndarray, y: np.ndarray):
    """Return (seed, lower, upper) arrays for one dataset."""
    span = max(float(np.ptp(x)), 1e-12)
    ymed = float(np.median(y))
    yamp = float(np.percentile(y, 95) - np.percentile(y, 5)) or 1.0
    ylo, yhi = float(np.min(y)) - 2 * yamp, float(np.max(y)) + 2 * yamp
    amp = 4 * yamp
    tlo, thi = span * 1e-3, span * 1e3
    first, last = float(y[np.argmin(x)]), float(y[np.argmax(x)])
    if model in ("SpinEcho", "T1NormExp", "T1NormStretch", "T1DecayExp", "T1DecayStretch"):
        s = [last, first - last, span / 3]
        lo, hi = [ylo, -amp, tlo], [yhi, amp, thi]
        if model not in ("T1NormExp", "T1DecayExp"):
            s.append(1.0); lo.append(0.2); hi.append(6.0)
        if model == "SpinEcho":
            s = [s[0], s[1], s[2], 1.0]
    elif model in ("T1Exp", "T1Stretch"):
        s = [first, last - first, span / 3]
        lo, hi = [ylo, -amp, tlo], [yhi, amp, thi]
        if model == "T1Stretch":
            s.append(1.0); lo.append(0.2); hi.append(6.0)
    elif model == "RamseySimple":
        s = [ymed, yamp / 2, _dominant_frequency(x, y), 0.0, span / 2, 1.0]
        lo = [ylo, -amp, 0.0, -2 * np.pi, tlo, 0.2]
        hi = [yhi, amp, 50.0 * s[2] + 10.0 / span, 2 * np.pi, thi, 6.0]
    elif model == "RabiCosine":
        f0 = _dominant_frequency(x, y)
        s = [ymed, yamp / 2, span, 1.0, f0, 0.0]
        lo = [ylo, -amp, tlo, 0.2, 0.0, -2 * np.pi]
        hi = [yhi, amp, thi, 4.0, 5.0 * f0 + 10.0 / span, 2 * np.pi]
    elif model == "ODMR":
        pol = 1.0 if (np.max(y) - ymed) >= (ymed - np.min(y)) else -1.0
        i = int(np.argmax(pol * (y - ymed)))
        s = [ymed, pol * yamp, span / 20, float(x[i])]
        lo = [ylo, -amp, span * 1e-4, float(np.min(x))]
        hi = [yhi, amp, span, float(np.max(x))]
    elif model == "GaussianPeaks":
        pol = 1.0 if (np.max(y) - ymed) >= (ymed - np.min(y)) else -1.0
        i = int(np.argmax(pol * (y - ymed)))
        s = [ymed, pol * yamp, float(x[i]), span / 20]
        lo = [ylo, -amp, float(np.min(x)), span * 1e-4]
        hi = [yhi, amp, float(np.max(x)), span]
    elif model == "TimeTraceDecay":
        s = [last, first - last, span / 3]
        lo, hi = [ylo, -amp, tlo], [yhi, amp, thi]
    else:
        raise ValueError(f"Unsupported model for global fit: {model}")
    return np.array(s, float), np.array(lo, float), np.array(hi, float)


def _call(model_name: str, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.asarray(MODEL_REGISTRY[model_name](x, *p), dtype=float)


def _prepare(datasets):
    out = []
    for d in datasets:
        if len(d) < 2:
            raise ValueError("Each dataset must be (x, y[, sem]).")
        x = np.asarray(d[0], float)
        y = np.asarray(d[1], float)
        sem = np.asarray(d[2], float) if len(d) > 2 and d[2] is not None else None
        if x.shape != y.shape or (sem is not None and sem.shape != x.shape):
            raise ValueError("x, y and sem must have identical shapes.")
        ok = np.isfinite(x) & np.isfinite(y)
        if sem is not None:
            ok &= np.isfinite(sem) & (sem > 0)
        if int(ok.sum()) < 4:
            raise ValueError("Each dataset needs at least 4 finite points.")
        out.append((x[ok], y[ok], None if sem is None else sem[ok]))
    return out


def _per_param(spec, names, k, kind):
    """Normalise dict-or-list-of-dicts user input to a list of per-dataset dicts."""
    if spec is None:
        return [{} for _ in range(k)]
    if isinstance(spec, dict):
        return [dict(spec) for _ in range(k)]
    if len(spec) != k:
        raise ValueError(f"{kind} must be a dict or a list with one dict per dataset.")
    return [dict(s or {}) for s in spec]


def global_fit(
    model_name: str,
    datasets: Sequence[tuple],
    shared: Sequence[str],
    seeds=None,
    bounds=None,
    weights=None,
    multistart: int = 12,
) -> dict:
    """Fit ``model_name`` to several datasets with the parameters in ``shared`` common to all.

    datasets : list of (x, y[, sem]);  shared : parameter names (see MODEL_PARAMS)
    seeds    : {name: value} or list of such dicts (per dataset) overriding starting values
    bounds   : {name: (lo, hi)} (all datasets) or list of such dicts (per dataset)
    weights  : None or list of per-dataset arrays (multiplies 1/sem^2 or 1)
    """
    if model_name not in MODEL_PARAMS or model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name!r} not supported; choose from {sorted(MODEL_PARAMS)}")
    names = MODEL_PARAMS[model_name]
    shared = list(dict.fromkeys(shared or []))
    for s in shared:
        if s not in names:
            raise ValueError(f"Shared parameter {s!r} is not a parameter of {model_name}: {names}")
    data = _prepare(datasets)
    nd = len(data)
    if nd < 1:
        raise ValueError("At least one dataset is required.")
    P = len(names)
    sh_idx = [names.index(s) for s in shared]
    free_idx = [i for i in range(P) if i not in sh_idx]
    seed_over = _per_param(seeds, names, nd, "seeds")
    bound_over = _per_param(bounds, names, nd, "bounds")

    # --- weights -------------------------------------------------------
    use_sem = all(d[2] is not None for d in data)
    w = []
    for j, (x, y, sem) in enumerate(data):
        wj = 1.0 / sem**2 if use_sem else np.ones_like(y)
        if weights is not None and weights[j] is not None:
            uw = np.asarray(weights[j], float)
            if uw.size == len(datasets[j][0]):  # given on the original (unfiltered) points
                ok0 = np.isfinite(np.asarray(datasets[j][0], float)) & np.isfinite(np.asarray(datasets[j][1], float))
                if len(datasets[j]) > 2 and datasets[j][2] is not None:
                    s0 = np.asarray(datasets[j][2], float)
                    ok0 &= np.isfinite(s0) & (s0 > 0)
                uw = uw[ok0]
            wj = wj * uw
        w.append(np.sqrt(wj))

    # --- per-dataset defaults and bounds ------------------------------------
    base = []
    for j, (x, y, _) in enumerate(data):
        s, lo, hi = _defaults(model_name, x, y)
        for k, n in enumerate(names):
            if n in seed_over[j]:
                s[k] = float(seed_over[j][n])
            if n in bound_over[j]:
                lo[k], hi[k] = (float(b) for b in bound_over[j][n])
        s = np.clip(s, lo, hi)
        base.append((s, lo, hi))

    # --- layout of the global parameter vector: [shared..., ds0 free..., ds1 free...]
    nsh, nfr = len(sh_idx), len(free_idx)
    lo_sh = np.array([max(b[1][i] for b in base) for i in sh_idx])
    hi_sh = np.array([min(b[2][i] for b in base) for i in sh_idx])
    bad = lo_sh >= hi_sh  # incompatible per-dataset bounds -> use the union
    if np.any(bad):
        lo_sh[bad] = [min(b[1][i] for b in base) for i in np.array(sh_idx)[bad]]
        hi_sh[bad] = [max(b[2][i] for b in base) for i in np.array(sh_idx)[bad]]
    lower = np.concatenate([lo_sh] + [b[1][free_idx] for b in base])
    upper = np.concatenate([hi_sh] + [b[2][free_idx] for b in base])

    def unpack(theta):
        out = []
        for j in range(nd):
            p = np.empty(P)
            p[sh_idx] = theta[:nsh]
            p[free_idx] = theta[nsh + j * nfr: nsh + (j + 1) * nfr]
            out.append(p)
        return out

    def resid(theta):
        ps = unpack(theta)
        return np.concatenate([w[j] * (_call(model_name, data[j][0], ps[j]) - data[j][1]) for j in range(nd)])

    rng = np.random.default_rng(12345)
    n_starts = max(1, int(multistart))

    def run(fun, x0, lo, hi):
        try:
            return least_squares(fun, np.clip(x0, lo, hi), bounds=(lo, hi), x_scale="jac", max_nfev=4000)
        except Exception:
            return None

    def perturbed(s, lo, hi, scale=0.15):
        span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
        trial = s + rng.normal(0.0, scale, size=s.shape) * np.minimum(np.abs(s) + 1e-12 * span, 0.2 * span + np.abs(s))
        return np.clip(trial, lo, hi)

    # --- independent fits (also supply seeds) ---------------------------------
    indep = []
    for j, (x, y, _) in enumerate(data):
        s, lo, hi = base[j]
        starts = [s] + [perturbed(s, lo, hi, 0.3) for _ in range(max(0, n_starts - 1))]
        if "phi" in names:
            for ph in (-np.pi / 2, np.pi / 2, np.pi):
                t = s.copy(); t[names.index("phi")] = ph; starts.append(t)
        best = None
        for st in starts:
            r = run(lambda p, j=j: w[j] * (_call(model_name, data[j][0], p) - data[j][1]), st, lo, hi)
            if r is not None and np.all(np.isfinite(r.fun)) and (best is None or r.cost < best.cost):
                best = r
        if best is None:
            raise RuntimeError(f"Independent fit failed for dataset {j}.")
        indep.append(best)

    # --- global multistart -------------------------------------------------------
    ind_params = [r.x for r in indep]
    starts = []
    mean_sh = np.mean([p[sh_idx] for p in ind_params], axis=0) if nsh else np.zeros(0)
    def theta_from(sh_vals, perds):
        return np.concatenate([sh_vals] + [p[free_idx] for p in perds])
    starts.append(theta_from(mean_sh, ind_params))
    for p in ind_params:
        starts.append(theta_from(p[sh_idx], ind_params))
    if "phi" in names:
        for ph in (-np.pi / 2, np.pi / 2, np.pi):
            perds = [p.copy() for p in ind_params]
            for q in perds:
                q[names.index("phi")] = ph
            starts.append(theta_from(mean_sh, perds))
    while len(starts) < n_starts:
        starts.append(perturbed(starts[0], lower, upper, 0.3))
    starts = starts[: max(n_starts, 1 + nd)]
    best = None
    for st in starts:
        r = run(resid, st, lower, upper)
        if r is not None and np.all(np.isfinite(r.fun)) and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        raise RuntimeError("Global fit failed for all starts.")

    # --- assemble ---------------------------------------------------------------------
    N = int(sum(len(d[1]) for d in data))
    k_glob = len(best.x)
    k_ind = P * nd
    rss_glob = float(np.sum(best.fun**2))
    rss_ind = float(sum(np.sum(r.fun**2) for r in indep))

    def crit(rss, k):
        rss = max(rss, 1e-300)
        return N * np.log(rss / N) + 2 * k, N * np.log(rss / N) + k * np.log(N)

    aic_g, bic_g = crit(rss_glob, k_glob)
    aic_i, bic_i = crit(rss_ind, k_ind)

    cov = _estimate_covariance(best.jac, 0.5 * rss_glob, max(1, N - k_glob))
    err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    err = np.where(np.isnan(np.diag(cov)), np.nan, err)

    ps = unpack(best.x)
    res_list = np.split(best.fun, np.cumsum([len(d[1]) for d in data])[:-1])
    ds_out = []
    for j, (x, y, _) in enumerate(data):
        yfit = _call(model_name, x, ps[j])
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - float(np.sum((y - yfit) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        e = np.empty(P)
        e[sh_idx] = err[:nsh]
        e[free_idx] = err[nsh + j * nfr: nsh + (j + 1) * nfr]
        # independent-fit R2 for reference
        yi = _call(model_name, x, ind_params[j])
        r2i = 1.0 - float(np.sum((y - yi) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        ds_out.append({
            "index": j,
            "n": int(len(y)),
            "params": {n: float(ps[j][i]) for i, n in enumerate(names)},
            "errors": {n: float(e[i]) for i, n in enumerate(names)},
            "r2": r2,
            "r2_independent": r2i,
            "chi2": float(np.sum(res_list[j] ** 2)),
            "y_fit": yfit,
            "x": x,
            "y": y,
        })

    ind_out = []
    for j, r in enumerate(indep):
        cj = _estimate_covariance(r.jac, 0.5 * float(np.sum(r.fun**2)), max(1, len(data[j][1]) - P))
        ind_out.append({
            "params": {n: float(r.x[i]) for i, n in enumerate(names)},
            "errors": {n: float(np.sqrt(max(cj[i, i], 0.0))) if np.isfinite(cj[i, i]) else float("nan") for i, n in enumerate(names)},
            "chi2": float(np.sum(r.fun**2)),
        })

    d_aic, d_bic = aic_g - aic_i, bic_g - bic_i
    if not shared:
        verdict = "No parameters shared: global fit equals independent fits."
    elif d_aic < 0 and d_bic < 0:
        verdict = "Sharing is supported (AIC and BIC both lower than independent fits)."
    elif d_aic < 0 or d_bic < 0:
        verdict = "Ambiguous: only one of AIC/BIC favours sharing."
    else:
        verdict = "Sharing is not supported (AIC and BIC both higher than independent fits)."

    return {
        "model": model_name,
        "param_names": list(names),
        "shared_names": shared,
        "shared": {n: {"value": float(ps[0][names.index(n)]), "error": float(err[i])} for i, n in enumerate(shared)},
        "datasets": ds_out,
        "n_points": N,
        "n_params": k_glob,
        "weighted_by_sem": bool(use_sem),
        "total_chi2": rss_glob,
        "reduced_chi2": rss_glob / max(1, N - k_glob),
        "aic": float(aic_g),
        "bic": float(bic_g),
        "independent": {"datasets": ind_out, "n_params": k_ind, "total_chi2": rss_ind,
                        "aic": float(aic_i), "bic": float(bic_i)},
        "delta_aic": float(d_aic),
        "delta_bic": float(d_bic),
        "sharing_justified": bool(shared) and d_aic < 0 and d_bic < 0,
        "verdict": verdict,
        "success": bool(best.success),
    }


def model_curve(model_name: str, params: dict, x: np.ndarray) -> np.ndarray:
    """Evaluate a registry model on ``x`` from a {name: value} dict."""
    return _call(model_name, np.asarray(x, float), np.array([params[n] for n in MODEL_PARAMS[model_name]], float))
