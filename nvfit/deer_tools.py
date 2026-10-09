"""DEER helper functions: reference division, spin-resonance expectations, spectrum summary.

Conventions: frequencies in GHz, magnetic field in tesla, g dimensionless.
Free-electron-like resonance  f = g * mu_B * B / h  with mu_B/h = 13.996245 GHz/T (CODATA 2018).
"""
from __future__ import annotations

import numpy as np

from .models import odmr_lorentzian

MU_B_OVER_H_GHZ_PER_T = 9.2740100783e-24 / 6.62607015e-34 / 1e9  # 13.99624493...
G_FREE = 2.0023


def expected_target_resonance(B_T: float, g: float = G_FREE) -> float:
    """Spin resonance frequency in GHz for field B_T (tesla): f = g*(mu_B/h)*B."""
    return float(g * MU_B_OVER_H_GHZ_PER_T * B_T)


def field_from_resonance(f_GHz: float, g: float = G_FREE) -> float:
    """Inverse of ``expected_target_resonance``: B in tesla from f in GHz."""
    return float(f_GHz / (g * MU_B_OVER_H_GHZ_PER_T))


def deer_ratio(x, y_deer, x_ref, y_ref, sem_deer=None, sem_ref=None) -> dict:
    """Normalised DEER signal = y_deer / y_ref on the DEER x grid.

    If the grids are identical (same length, equal within 1e-9 relative) they are matched exactly.
    Otherwise the reference (and its sem, linearly) is interpolated onto ``x`` provided the grids
    overlap; DEER points outside the reference range are dropped; no overlap raises ValueError.
    Error: sigma = sqrt((s_d/ref)^2 + (ratio*s_r/ref)^2) (first order, independent errors).
    """
    x = np.asarray(x, float).ravel()
    yd = np.asarray(y_deer, float).ravel()
    xr = np.asarray(x_ref, float).ravel()
    yr = np.asarray(y_ref, float).ravel()
    sd = None if sem_deer is None else np.asarray(sem_deer, float).ravel()
    sr = None if sem_ref is None else np.asarray(sem_ref, float).ravel()
    if x.shape != yd.shape or xr.shape != yr.shape:
        raise ValueError("x/y length mismatch.")
    notes = []
    if x.shape == xr.shape and np.allclose(x, xr, rtol=1e-9, atol=0):
        ref, ref_s, keep, interpolated = yr, sr, np.ones(x.size, bool), False
        notes.append("Reference measured on the identical x grid (matched exactly).")
    else:
        order = np.argsort(xr)
        xr, yr = xr[order], yr[order]
        sr = None if sr is None else sr[order]
        lo, hi = xr[0], xr[-1]
        if x.max() < lo or x.min() > hi:
            raise ValueError("DEER and reference x grids do not overlap; cannot interpolate.")
        keep = (x >= lo) & (x <= hi)
        ref = np.interp(x, xr, yr)
        ref_s = None if sr is None else np.interp(x, xr, sr)
        interpolated = True
        notes.append(f"Reference INTERPOLATED (linear) onto the DEER x grid; {int((~keep).sum())} DEER point(s) outside "
                     f"the reference range [{lo:g}, {hi:g}] dropped. Reference sem interpolated linearly (approximate).")
    good = keep & (ref != 0) & np.isfinite(ref) & np.isfinite(yd)
    r = np.full(x.size, np.nan)
    r[good] = yd[good] / ref[good]
    sem = None
    if sd is not None or ref_s is not None:
        a = np.zeros(x.size) if sd is None else sd
        b = np.zeros(x.size) if ref_s is None else ref_s
        sem = np.full(x.size, np.nan)
        sem[good] = np.sqrt((a[good] / ref[good]) ** 2 + (r[good] * b[good] / ref[good]) ** 2)
        if sd is None or ref_s is None:
            notes.append("Only one of the two sems was supplied; the error covers that term only.")
    return {"x": x[good], "ratio": r[good], "sem": None if sem is None else sem[good],
            "interpolated": interpolated, "n_dropped": int((~good).sum()), "note": " ".join(notes)}


def deer_spectrum_summary(freqs_ghz, signal, B_T: float | None = None, g: float = G_FREE) -> dict:
    """Lorentzian peak/dip summary of a DEER spectrum (frequency in GHz).

    Fits ``nvfit.models.odmr_lorentzian`` y0 + C*w^2/((f-f0)^2+w^2) (w = HWHM; C<0 is a dip),
    reports f0 and FWHM = 2|w| (GHz) with 1-sigma from the covariance, and, if B_T is given, the offset
    f0 - g*mu_B*B/h (GHz and MHz) plus the g implied by f0.
    """
    from scipy.optimize import curve_fit

    f = np.asarray(freqs_ghz, float).ravel()
    y = np.asarray(signal, float).ravel()
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok], y[ok]
    if f.size < 5:
        raise ValueError("Need at least 5 finite points.")
    o = np.argsort(f)
    f, y = f[o], y[o]
    y0 = float(np.median(y))
    dev = y - y0
    i = int(np.argmax(np.abs(dev)))
    C0, x0 = float(dev[i]), float(f[i])
    w0 = max((f[-1] - f[0]) / 20.0, float(np.median(np.diff(f))))
    out: dict = {"n": int(f.size), "kind": "dip" if C0 < 0 else "peak"}
    try:
        p, cov = curve_fit(odmr_lorentzian, f, y, p0=[y0, C0, w0, x0], maxfev=20000)
        err = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(4, np.nan)
        out.update({"fit_ok": True, "y0": float(p[0]), "contrast": float(p[1]), "f0_GHz": float(p[3]),
                    "f0_sigma_GHz": float(err[3]), "fwhm_GHz": float(2 * abs(p[2])),
                    "fwhm_sigma_GHz": float(2 * err[2]), "kind": "dip" if p[1] < 0 else "peak",
                    "note": "Lorentzian fit (odmr_lorentzian)."})
    except Exception as exc:
        out.update({"fit_ok": False, "f0_GHz": x0, "f0_sigma_GHz": None, "fwhm_GHz": None,
                    "note": f"Lorentzian fit failed ({exc}); f0 is the extremum sample, no width."})
    if B_T is not None:
        fe = expected_target_resonance(B_T, g)
        out.update({"B_T": float(B_T), "g": float(g), "expected_GHz": fe,
                    "offset_GHz": out["f0_GHz"] - fe, "offset_MHz": 1e3 * (out["f0_GHz"] - fe),
                    "g_implied": out["f0_GHz"] / (MU_B_OVER_H_GHZ_PER_T * B_T) if B_T else None})
    return out
