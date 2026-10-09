"""Magnetic field from NV ground-state ODMR resonance frequencies.

Model and conventions (all stated explicitly, nothing perturbative)
-------------------------------------------------------------------
* Ground-state spin-1 Hamiltonian in an NV's own frame (h = 1, so energies are
  frequencies in GHz; z = NV symmetry axis)::

      H = D Sz^2 + E (Sx^2 - Sy^2) + gamma_e (Bx Sx + By Sy + Bz Sz)

  ``D`` = 2.870 GHz (room temperature zero-field splitting; a parameter and
  optionally fitted), ``E`` = strain/electric splitting (default 0),
  ``gamma_e`` = 28.024 GHz/T = 2.8024 MHz/G (electron gyromagnetic ratio,
  g ~ 2.0028; NV hyperfine/14N/13C structure and nuclear Zeeman are ignored).
  Basis of the matrices: |ms=+1>, |ms=0>, |ms=-1> (Sx, Sy, Sz = spin-1).
* Energy unit GHz, field unit tesla (1 mT = 10 G) at the API; fitted fields are
  also reported in mT and gauss.
* Four NV orientations along the diamond <111> axes in the CRYSTAL frame
  (cubic axes x, y, z = [100], [010], [001]), each divided by sqrt(3)::

      NV0 (1,1,1)  NV1 (1,-1,-1)  NV2 (-1,1,-1)  NV3 (-1,-1,1)

  The sign of each axis is irrelevant for E = 0 (lines are a set symmetric in
  +/-Bz).  For E != 0 the in-plane x axis of NV i is the projection of the crystal
  z axis on the plane perpendicular to the NV axis (y = z_nv x x_nv); the same E
  is applied in every NV's own frame.  This is an assumption, not a measurement
  (real strain differs per NV), so E-fits of ensembles are indicative only.
* B in the crystal frame is parameterised as magnitude |B|, polar angle theta
  (from crystal z) and azimuth phi (from crystal x toward y), in degrees in the
  results.
* Transition frequencies: the eigenstate with the largest ms=0 character is
  the "ms=0-like" state; the two lines are |E_k - E_0| for the other two states,
  returned sorted (lower, upper).  This stays valid through the level
  anticrossing (B_par ~ D/gamma ~ 102.4 mT).

Degeneracies
------------
* B and -B give identical line sets (time reversal).
* For E = 0 the line set is also invariant under the cubic symmetry of the four
  <111> axes (permutations of the x, y, z components and sign flips of each
  component).  Together with B -> -B that is 48 crystal-frame directions giving
  the same spectrum.  Reported angles are therefore reduced to the fundamental
  domain |Bx| >= |By| >= |Bz| >= 0 (theta in [54.74, 90] deg, phi in [0, 45] deg).
  With E fitted/non-zero this reduction is not exact and is not applied.
* Lines cannot be tied to physical NV orientations without extra information
  (e.g. a known field direction or line intensity pattern); the orientation
  labels in the results are labels of the canonical solution.
* A single NV orientation (2 lines) fixes only |B| and the angle between B and
  that NV axis, i.e. B lies on a cone around the axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from .fit_engine import _estimate_covariance

# --- constants -------------------------------------------------------------
D_ROOM_GHZ = 2.870          # zero-field splitting at room temperature [GHz]
GAMMA_E_GHZ_PER_T = 28.024  # gamma_e / h = 28.024 GHz/T = 2.8024 MHz/G
NOMINAL_SIGMA_GHZ = 1e-3    # 1 MHz, used only to scale residuals when no sigma is given

_S2 = 1.0 / math.sqrt(2.0)
SX = _S2 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
SY = _S2 * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)
_SX2_SY2 = SX @ SX - SY @ SY
_SZ2 = SZ @ SZ

NV_AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / math.sqrt(3.0)


def _nv_frames() -> np.ndarray:
    """(4, 3, 3) array; row k of frame i = unit vector of NV i's x, y, z axis in the crystal frame."""
    ref = np.array([0.0, 0.0, 1.0])
    out = np.zeros((4, 3, 3))
    for i, n in enumerate(NV_AXES):
        x = ref - np.dot(ref, n) * n
        x /= np.linalg.norm(x)
        y = np.cross(n, x)
        out[i] = np.vstack([x, y, n])
    return out


NV_FRAMES = _nv_frames()


# --- forward model ---------------------------------------------------------
def b_vector(B_T: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    return B_T * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def _lines_from_nv_frame_fields(b_nv: np.ndarray, D: float, E: float, gamma: float) -> np.ndarray:
    """b_nv (N, 3) field [T] in each NV's own frame -> (N, 2) sorted lines [GHz] (exact diagonalisation)."""
    b_nv = np.atleast_2d(np.asarray(b_nv, float))
    H = (D * _SZ2 + E * _SX2_SY2)[None, :, :] + gamma * (
        b_nv[:, 0, None, None] * SX + b_nv[:, 1, None, None] * SY + b_nv[:, 2, None, None] * SZ
    )
    e, v = np.linalg.eigh(H)
    k0 = np.argmax(np.abs(v[:, 1, :]) ** 2, axis=1)          # ms=0-like state
    rows = np.arange(len(e))
    d = np.abs(e - e[rows, k0][:, None])
    d[rows, k0] = np.inf
    d.sort(axis=1)
    return d[:, :2]


def single_nv_lines(B_par_T: float, B_perp_T: float = 0.0, D: float = D_ROOM_GHZ, E: float = 0.0,
                    gamma: float = GAMMA_E_GHZ_PER_T) -> np.ndarray:
    """(f_lower, f_upper) [GHz] of one NV with field component B_par along its axis and B_perp along its x axis."""
    return _lines_from_nv_frame_fields(np.array([[B_perp_T, 0.0, B_par_T]]), D, E, gamma)[0]


def nv_transition_frequencies(B_T: float, theta_deg: float, phi_deg: float, D: float = D_ROOM_GHZ,
                              E: float = 0.0, gamma: float = GAMMA_E_GHZ_PER_T) -> np.ndarray:
    """(4, 2) array of ms=0 -> ms=+/-1 lines [GHz], sorted (lower, upper), one row per NV orientation."""
    return transition_frequencies_vec(b_vector(B_T, theta_deg, phi_deg), D, E, gamma)


def transition_frequencies_vec(B_crystal_T, D: float = D_ROOM_GHZ, E: float = 0.0,
                               gamma: float = GAMMA_E_GHZ_PER_T) -> np.ndarray:
    b_nv = np.einsum("ikj,j->ik", NV_FRAMES, np.asarray(B_crystal_T, float))
    return _lines_from_nv_frame_fields(b_nv, D, E, gamma)


def canonical_direction(B_T: float, theta_deg: float, phi_deg: float) -> tuple[float, float]:
    """Reduce (theta, phi) [deg] to the E=0 symmetry fundamental domain |x|>=|y|>=|z|."""
    if B_T <= 0:
        return 0.0, 0.0
    a = np.sort(np.abs(b_vector(1.0, theta_deg, phi_deg)))[::-1]
    return (math.degrees(math.acos(min(1.0, a[2]))), math.degrees(math.atan2(a[1], a[0])))


def angles_to_axes_deg(B_T: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    """Angle (0..90 deg) between B and each NV axis (axis sign ignored)."""
    v = b_vector(1.0, theta_deg, phi_deg)
    return np.degrees(np.arccos(np.clip(np.abs(NV_AXES @ v), 0.0, 1.0)))


# --- quick (linear, small-field) estimate ------------------------------------
def quick_estimate(freqs_ghz, gamma: float = GAMMA_E_GHZ_PER_T) -> dict[str, Any]:
    """Small-field estimate |B_par| ~ (f_upper - f_lower) / (2 gamma_e).

    Lines are paired outermost-with-innermost (lowest with highest, ...), which is
    the right pairing to first order since each orientation gives D -/+ gamma|B_par,i|.
    With 4 complete pairs, sum_i B_par,i^2 = (4/3)|B|^2 gives |B| ~ sqrt(3/4 sum B_par,i^2).
    Neglects B_perp (second-order shifts), strain E and any pairing error.
    """
    f = np.sort(np.asarray(freqs_ghz, float).ravel())
    n = len(f)
    npair = n // 2
    pairs = []
    for i in range(npair):
        lo, hi = f[i], f[n - 1 - i]
        pairs.append({"f_lower_ghz": float(lo), "f_upper_ghz": float(hi), "center_ghz": float(0.5 * (lo + hi)),
                      "B_par_T": float((hi - lo) / (2.0 * gamma))})
    out: dict[str, Any] = {"pairs": pairs, "gamma_ghz_per_t": gamma, "B_total_T": None, "kind": "lower_bound"}
    if n == 1:
        # single line: assume on-axis, D = room-temperature value
        out["pairs"] = [{"f_lower_ghz": float(f[0]), "f_upper_ghz": None, "center_ghz": None,
                         "B_par_T": float(abs(D_ROOM_GHZ - f[0]) / gamma)}]
        out["kind"] = "single_line_assuming_D"
        out["B_total_T"] = out["pairs"][0]["B_par_T"]
    elif pairs:
        bp = np.array([p["B_par_T"] for p in pairs])
        out["B_total_T"] = float(np.sqrt(0.75 * np.sum(bp ** 2))) if npair == 4 else float(bp.max())
        out["kind"] = "ensemble_4_pairs" if npair == 4 else "lower_bound"
    return out


# --- results container -------------------------------------------------------
@dataclass
class FieldFit:
    mode: str
    success: bool
    param_names: list[str]
    params: list[float]
    errors: list[float | None]
    units: dict[str, str]
    B_T: float
    B_T_err: float | None
    theta_deg: float | None = None
    theta_deg_err: float | None = None
    phi_deg: float | None = None
    phi_deg_err: float | None = None
    B_par_T: float | None = None
    B_perp_T: float | None = None
    angle_to_axis_deg: float | None = None
    angle_to_axis_deg_err: float | None = None
    angles_to_axes_deg: list[float] | None = None
    D_ghz: float = D_ROOM_GHZ
    E_ghz: float = 0.0
    gamma_ghz_per_t: float = GAMMA_E_GHZ_PER_T
    lines_measured_ghz: list[float] = field(default_factory=list)
    sigma_ghz: list[float | None] = field(default_factory=list)
    lines_predicted_ghz: list[float] = field(default_factory=list)
    residuals_ghz: list[float] = field(default_factory=list)
    orientation_lines_ghz: list[list[float]] = field(default_factory=list)
    assignment: list[dict[str, Any]] = field(default_factory=list)
    chi2: float = float("nan")
    dof: int = 0
    redchi2: float | None = None
    rms_mhz: float = float("nan")
    covariance: list[list[float | None]] | None = None
    errors_basis: str = ""
    B_range_T: list[float] | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    on_axis_estimate: dict[str, Any] | None = None
    quick: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def B_mT(self) -> float:
        return self.B_T * 1e3

    @property
    def B_gauss(self) -> float:
        return self.B_T * 1e4

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["B_mT"] = self.B_mT
        d["B_gauss"] = self.B_gauss
        d["B_mT_err"] = None if self.B_T_err is None else self.B_T_err * 1e3
        return d


def _finite_or_none(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _propagate(func, p: np.ndarray, cov: np.ndarray) -> float | None:
    """1-sigma of scalar func(p) by first-order propagation; NaN-variance params with zero gradient are ignored."""
    p = np.asarray(p, float)
    g = np.zeros(len(p))
    for i in range(len(p)):
        h = 1e-6 * max(1.0, abs(p[i]))
        pp, pm = p.copy(), p.copy()
        pp[i] += h
        pm[i] -= h
        g[i] = (func(pp) - func(pm)) / (2 * h)
    bad = ~np.isfinite(np.diag(cov))
    if np.any(bad & (np.abs(g) > 1e-9)):
        return None
    ok = ~bad
    var = float(g[ok] @ cov[np.ix_(ok, ok)] @ g[ok])
    return math.sqrt(var) if var >= 0 and math.isfinite(var) else None


def _central_jac(fun, x, lower, upper) -> np.ndarray:
    """Central-difference Jacobian (one-sided at a bound) of fun at x."""
    x = np.asarray(x, float)
    cols = []
    for i in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] = min(x[i] + h, upper[i])
        xm[i] = max(x[i] - h, lower[i])
        cols.append((fun(xp) - fun(xm)) / (xp[i] - xm[i]))
    return np.array(cols).T


def _covariance(jac: np.ndarray, resid: np.ndarray, sigma_given: bool, n: int) -> tuple[np.ndarray, str, int]:
    npar = jac.shape[1]
    dof = n - npar
    cost = 0.5 * float(resid @ resid)
    if sigma_given:
        cov = _estimate_covariance(jac, 0.5 * max(1, dof), max(1, dof))       # s^2 = 1: sigma taken as true
        basis = "supplied uncertainties taken as true 1-sigma (chi2 reported separately)"
    elif dof > 0:
        cov = _estimate_covariance(jac, cost, dof)                             # s^2 = residual variance
        basis = "no uncertainties supplied: scaled by residual variance (dof > 0)"
    else:
        cov = _estimate_covariance(jac, 0.5, 1)                                # s^2 = 1 with nominal sigma
        basis = (f"no uncertainties and dof = 0: assumed sigma = {NOMINAL_SIGMA_GHZ * 1e3:g} MHz on every line "
                 "(errors are NOT data-derived)")
    return cov, basis, dof


# --- input handling ----------------------------------------------------------
def _prep(freqs_ghz, uncertainties):
    f = np.asarray(freqs_ghz, float).ravel()
    if f.size == 0 or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("freqs_ghz must be non-empty, finite and positive (GHz).")
    if f.size > 8:
        raise ValueError("At most 8 resonance lines (4 NV orientations x 2) are supported.")
    if uncertainties is None:
        s = None
    else:
        s = np.asarray(uncertainties, float).ravel()
        if s.shape != f.shape or not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("uncertainties must match freqs_ghz and be finite and > 0 (GHz).")
    order = np.argsort(f)
    return f[order], (None if s is None else s[order])


# --- single orientation ------------------------------------------------------
def _fit_single(f, s, D, E, gamma, notes) -> FieldFit:
    if len(f) != 2:
        raise ValueError("single-orientation mode needs exactly 2 lines.")
    sig = np.full(2, NOMINAL_SIGMA_GHZ) if s is None else s

    def res(x):
        return (single_nv_lines(x[0] * 1e-3, x[1] * 1e-3, D, E, gamma) - f) / sig

    half = 0.5 * (f[1] - f[0])
    arg = half ** 2 - E ** 2
    on_axis = {"B_par_T": math.sqrt(arg) / gamma if arg > 0 else 0.0, "D_ghz": float(0.5 * (f[0] + f[1])),
               "formula": "B_par = sqrt(((f+ - f-)/2)^2 - E^2)/gamma_e,  D = (f+ + f-)/2  (B_perp = 0)"}
    b0 = on_axis["B_par_T"] * 1e3
    sols: list[tuple[float, np.ndarray, Any]] = []
    for pf in (1.0, 0.5, 1.5, 3.0, 0.0):
        for pp in (0.0, 0.5, 2.0, 10.0, 40.0):
            try:
                r = least_squares(res, [b0 * pf + 1e-3, pp], bounds=([0, 0], [3000, 3000]), x_scale=[max(b0, 1.0), 5.0],
                                  xtol=1e-14, ftol=1e-14, gtol=1e-14)
            except Exception:
                continue
            sols.append((r.cost, r.x, r))
    if not sols:
        raise RuntimeError("single-orientation solve failed")
    sols.sort(key=lambda t: t[0])
    tol = 0.5 * (1e-6 / np.max(sig)) ** 2          # rms residual < ~1 kHz -> "exact" solution
    exact = [t for t in sols if t[0] <= max(tol, sols[0][0] * (1 + 1e-6) + tol)]
    uniq: list[tuple[float, np.ndarray, Any]] = []
    for t in exact:
        if not any(np.allclose(t[1], u[1], atol=1e-3, rtol=1e-4) for u in uniq):
            uniq.append(t)
    uniq.sort(key=lambda t: float(np.hypot(*t[1])))
    best = uniq[0] if sols[0][0] <= tol else sols[0]
    if sols[0][0] > tol:
        notes.append("No exact (sub-kHz) solution: lines are inconsistent with this model for the given D/E "
                     "(e.g. mean of the two lines above D, or splitting beyond range).")
    x, r = best[1], best[2]
    bpar, bperp = x[0] * 1e-3, x[1] * 1e-3
    cov, basis, dof = _covariance(r.jac, r.fun, s is not None, 2)
    btot = lambda p: math.hypot(p[0], p[1]) * 1e-3                                  # noqa: E731  [T]
    ang = lambda p: math.degrees(math.atan2(p[1], p[0]))                             # noqa: E731
    chi2 = float(r.fun @ r.fun)
    ff = FieldFit(
        mode="single", success=bool(sols[0][0] <= tol), param_names=["B_par_mT", "B_perp_mT"], params=[x[0], x[1]],
        errors=[_finite_or_none(math.sqrt(max(cov[i, i], 0))) if np.isfinite(cov[i, i]) else None for i in range(2)],
        units={"B_par_mT": "mT", "B_perp_mT": "mT"},
        B_T=btot(x), B_T_err=None, B_par_T=bpar, B_perp_T=bperp,
        angle_to_axis_deg=ang(x), D_ghz=D, E_ghz=E, gamma_ghz_per_t=gamma,
        lines_measured_ghz=f.tolist(), sigma_ghz=[None] * 2 if s is None else s.tolist(),
        lines_predicted_ghz=(f + r.fun * sig).tolist(), residuals_ghz=(r.fun * sig).tolist(),
        orientation_lines_ghz=[(f + r.fun * sig).tolist()],
        chi2=chi2, dof=dof, redchi2=(chi2 / dof if dof > 0 else None), rms_mhz=float(np.sqrt(np.mean((r.fun * sig) ** 2)) * 1e3),
        covariance=[[_finite_or_none(v) for v in row] for row in cov], errors_basis=basis,
        on_axis_estimate=on_axis,
    )
    ff.B_T_err = _propagate(btot, x, cov)
    ff.angle_to_axis_deg_err = _propagate(ang, x, cov)
    ff.B_range_T = [float(min(np.hypot(*u[1]) for u in uniq) * 1e-3), float(max(np.hypot(*u[1]) for u in uniq) * 1e-3)]
    ff.alternatives = [{"B_par_mT": float(u[1][0]), "B_perp_mT": float(u[1][1]), "B_mT": float(np.hypot(*u[1])),
                        "angle_to_axis_deg": ang(u[1])} for u in uniq[1:]]
    if ff.alternatives:
        notes.append("Additional exact solutions exist (listed in 'alternatives'), typically on the branch beyond "
                     "the level anticrossing; B_range_T spans them.")
    notes.append("One NV orientation fixes |B| and the angle to that NV axis only: B lies on a cone around the axis. "
                 "B_par >= 0 by convention (B -> -B swaps the two lines).")
    notes.append("B_perp is taken along the NV x axis; this matters only if E != 0. If B_perp ~ 0 its error is "
                 "unidentifiable (quadratic effect) and reported as None.")
    ff.notes = notes
    return ff


# --- ensemble -----------------------------------------------------------------
def _fit_ensemble(f, s, D, E, gamma, fit_D, fit_E, n_starts, seed, notes) -> FieldFit:
    n = len(f)
    names = ["B_mT", "theta_rad", "phi_rad"] + (["D_ghz"] if fit_D else []) + (["E_ghz"] if fit_E else [])
    npar = len(names)
    if n < npar:
        raise ValueError(f"{n} line(s) cannot determine {npar} parameters ({', '.join(names)}); "
                         "supply more lines or fix D/E.")
    sig = np.full(n, NOMINAL_SIGMA_GHZ) if s is None else s

    def unpack(x):
        i = 3
        d, e = D, E
        if fit_D:
            d = x[i]; i += 1
        if fit_E:
            e = x[i]
        return d, e

    def predict(x):
        d, e = unpack(x)
        return transition_frequencies_vec(b_vector(x[0] * 1e-3, math.degrees(x[1]), math.degrees(x[2])), d, e, gamma).ravel()

    def assign(p):
        c = ((f[:, None] - p[None, :]) / sig[:, None]) ** 2
        rows, cols = linear_sum_assignment(c)
        a = np.empty(n, int)
        a[rows] = cols
        return a

    def res_free(x):
        p = predict(x)
        return (f - p[assign(p)]) / sig

    lower = [0.0, -np.inf, -np.inf] + ([2.0] if fit_D else []) + ([-0.3] if fit_E else [])
    upper = [3000.0, np.inf, np.inf] + ([3.5] if fit_D else []) + ([0.3] if fit_E else [])
    q = quick_estimate(f, gamma)
    bq = (q["B_total_T"] or 0.0) * 1e3
    bq = max(bq, 0.05)
    rng = np.random.default_rng(seed)
    results = []
    for k in range(int(n_starts)):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        b0 = bq * (1.0, 0.7, 1.4, 2.0, 0.4)[k % 5] * (1.0 if k < 5 else rng.uniform(0.6, 1.6))
        x0 = [b0, math.acos(v[2]), math.atan2(v[1], v[0])] + ([D] if fit_D else []) + ([E] if fit_E else [])
        try:
            r = least_squares(res_free, x0, bounds=(lower, upper), x_scale=[max(b0, 1.0), 1.0, 1.0] + [1e-2] * (npar - 3),
                              xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
        except Exception:
            continue
        results.append(r)
    if not results:
        raise RuntimeError("ensemble fit failed for every start")
    results.sort(key=lambda r: r.cost)
    best = results[0]

    def canon(x):
        x = np.array(x, float)
        if x[0] < 0:
            x[0] = 0
        if not fit_E and E == 0.0:
            th, ph = canonical_direction(x[0], math.degrees(x[1]), math.degrees(x[2]))
            x[1], x[2] = math.radians(th), math.radians(ph)
        return x

    xb = canon(best.x)
    pb = predict(xb)
    a = assign(pb)

    def res_fixed(x):
        return (f - predict(x)[a]) / sig

    r2 = least_squares(res_fixed, xb, bounds=(lower, upper), x_scale=[max(xb[0], 1.0), 1.0, 1.0] + [1e-2] * (npar - 3),
                       xtol=1e-14, ftol=1e-14, gtol=1e-14, jac="3-point", max_nfev=200)
    if r2.cost <= best.cost * (1 + 1e-9) + 1e-30:
        xb = canon(r2.x)
    pb = predict(xb)
    r_fix = res_fixed(xb)
    jac = _central_jac(res_fixed, xb, lower, upper)
    cov, basis, dof = _covariance(jac, r_fix, s is not None, n)
    errs = [float(math.sqrt(max(cov[i, i], 0))) if np.isfinite(cov[i, i]) else None for i in range(npar)]
    B_T = xb[0] * 1e-3
    th_deg, ph_deg = math.degrees(xb[1]), math.degrees(xb[2])

    def axes_f(p, j):
        return float(angles_to_axes_deg(1.0, math.degrees(p[1]), math.degrees(p[2]))[j])

    ff = FieldFit(
        mode="ensemble", success=bool(best.success), param_names=names,
        params=[float(v) for v in xb], errors=errs,
        units={"B_mT": "mT", "theta_rad": "rad", "phi_rad": "rad", "D_ghz": "GHz", "E_ghz": "GHz"},
        B_T=B_T, B_T_err=None if errs[0] is None else errs[0] * 1e-3,
        theta_deg=th_deg, theta_deg_err=None if errs[1] is None else math.degrees(errs[1]),
        phi_deg=ph_deg, phi_deg_err=None if errs[2] is None else math.degrees(errs[2]),
        angles_to_axes_deg=angles_to_axes_deg(B_T, th_deg, ph_deg).tolist(),
        gamma_ghz_per_t=gamma, lines_measured_ghz=f.tolist(), sigma_ghz=[None] * n if s is None else s.tolist(),
        lines_predicted_ghz=pb[a].tolist(), residuals_ghz=(f - pb[a]).tolist(),
        orientation_lines_ghz=pb.reshape(4, 2).tolist(),
        chi2=float(r_fix @ r_fix), dof=dof, redchi2=(float(r_fix @ r_fix) / dof if dof > 0 else None),
        rms_mhz=float(np.sqrt(np.mean((f - pb[a]) ** 2)) * 1e3),
        covariance=[[_finite_or_none(v) for v in row] for row in cov], errors_basis=basis, quick=q,
    )
    d_fit, e_fit = unpack(xb)
    ff.D_ghz, ff.E_ghz = float(d_fit), float(e_fit)
    ff.assignment = [{"measured_ghz": float(f[i]), "predicted_ghz": float(pb[a[i]]), "orientation": int(a[i] // 2),
                      "branch": "lower" if a[i] % 2 == 0 else "upper"} for i in range(n)]
    if np.all(np.isfinite(np.diag(cov))):
        errs_ax = [_propagate(lambda p, j=j: axes_f(p, j), xb, cov) for j in range(4)]
        ff.notes.append("1-sigma of angles to NV axes (deg): " + ", ".join("n/a" if e is None else f"{e:.2g}" for e in errs_ax))
    # alternatives: distinct minima with comparable cost
    seen = [(round(xb[0], 3), round(math.degrees(xb[1]), 1), round(math.degrees(xb[2]), 1))]
    for r in results[1:]:
        xc = canon(r.x)
        key = (round(xc[0], 3), round(math.degrees(xc[1]), 1), round(math.degrees(xc[2]), 1))
        if key in seen or r.cost > max(10 * best.cost, best.cost + 0.5 * (3.0 / np.max(sig)) ** 2 * 1e-6):
            continue
        seen.append(key)
        ff.alternatives.append({"B_mT": float(xc[0]), "theta_deg": key[1], "phi_deg": key[2],
                                "chi2": float(2 * r.cost)})
    # notes
    lines = pb.reshape(4, 2)
    distinct = 1
    for i in range(1, 4):
        if all(np.max(np.abs(lines[i] - lines[j])) > 1e-6 for j in range(i)):
            distinct += 1
    ff.notes.append(f"{distinct} distinct NV line pair(s) predicted"
                    + ("; field is along/near a symmetry direction so some orientations are degenerate." if distinct < 4 else "."))
    if n < 2 * distinct:
        ff.notes.append("Fewer measured lines than predicted distinct lines: unobserved lines are unconstrained, "
                        "so the solution may not be unique (see alternatives).")
    if not fit_E and E == 0.0:
        ff.notes.append("Angles are reduced to the cubic fundamental domain (|Bx|>=|By|>=|Bz|; theta 54.7-90 deg, "
                        "phi 0-45 deg): 48 crystal-frame directions (24 Td operations x B -> -B) give the same "
                        "spectrum. Sign of B is undetermined.")
    else:
        ff.notes.append("E != 0: cubic reduction of angles not applied; E is assumed identical in each NV frame "
                        "(model assumption).")
    if xb[0] < 1e-3 * max(1.0, bq):
        ff.notes.append("|B| ~ 0: direction undefined, angle errors meaningless.")
    return ff


# --- public API -----------------------------------------------------------------
def fit_field(freqs_ghz, uncertainties=None, mode: str = "auto", *, D: float = D_ROOM_GHZ, E: float = 0.0,
              fit_D: bool = False, fit_E: bool = False, gamma: float = GAMMA_E_GHZ_PER_T,
              n_starts: int = 40, seed: int = 0) -> FieldFit:
    """Magnetic field from ODMR line positions.

    freqs_ghz     : 1-8 resonance frequencies [GHz] (any order).
    uncertainties : optional 1-sigma per line [GHz].
    mode          : 'auto' (2 lines -> 'single', 1 -> 'quick', >2 -> 'ensemble'), 'single', 'ensemble', 'quick'.
    D, E          : zero-field splitting and strain [GHz] when not fitted (E in each NV's own frame).
    fit_D, fit_E  : also fit D / E (ensemble mode only; needs lines >= number of parameters).
    Returns a FieldFit (``.to_dict()`` for JSON).  See module docstring for conventions.
    """
    f, s = _prep(freqs_ghz, uncertainties)
    notes: list[str] = []
    if mode == "auto":
        mode = "quick" if len(f) == 1 else ("single" if len(f) == 2 else "ensemble")
    if mode == "quick":
        q = quick_estimate(f, gamma)
        B = float(q["B_total_T"] or 0.0)
        notes.append(f"Quick small-field estimate only ({q['kind']}): |B_par| ~ (f+ - f-)/(2 gamma_e); "
                     "ignores B_perp, E and line-pairing errors.")
        return FieldFit(mode="quick", success=True, param_names=["B_T"], params=[B], errors=[None], units={"B_T": "T"},
                        B_T=B, B_T_err=None, D_ghz=D, E_ghz=E, gamma_ghz_per_t=gamma, lines_measured_ghz=f.tolist(),
                        sigma_ghz=[None] * len(f) if s is None else s.tolist(), quick=q,
                        errors_basis="not computed", notes=notes)
    if mode == "single":
        if fit_D or fit_E:
            raise ValueError("2 lines cannot determine (B_par, B_perp) and D/E; fix D/E or use an ensemble with more lines.")
        return _fit_single(f, s, D, E, gamma, notes)
    if mode == "ensemble":
        return _fit_ensemble(f, s, D, E, gamma, fit_D, fit_E, n_starts, seed, notes)
    raise ValueError("mode must be 'auto', 'single', 'ensemble' or 'quick'")
