from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit_engine import FitResult, fit_model_multistart, reduced_chi2, sem_to_weights
from .models import ramsey_extended, ramsey_hyperfine_n14, ramsey_simple


@dataclass
class RamseySelectionResult:
    best: FitResult
    candidates: list[FitResult]
    selected_by: str


def estimate_ramsey_frequency(x_ns: np.ndarray, y: np.ndarray, head_ns: float = 1200.0) -> float:
    # Adapt head to data span so short traces aren't degraded
    data_span = float(np.max(x_ns) - np.min(x_ns)) if len(x_ns) > 1 else head_ns
    effective_head = min(head_ns, 0.4 * data_span) if data_span > 0 else head_ns
    mask = x_ns <= (np.min(x_ns) + effective_head)
    xh = x_ns[mask]
    yh = y[mask]
    if len(xh) < 16:
        xh, yh = x_ns, y
    dt = float(np.median(np.diff(xh))) if len(xh) > 1 else 1.0
    if dt <= 0:
        dt = 1.0
    y0 = yh - np.mean(yh)
    freqs = np.fft.rfftfreq(len(y0), dt)
    mag = np.abs(np.fft.rfft(y0))
    if len(freqs) <= 1:
        return 0.002
    idx = int(np.argmax(mag[1:]) + 1)
    return max(float(freqs[idx]), 2e-4)


def _seed_ramsey_simple(x_ns: np.ndarray, y: np.ndarray, n_starts: int = 18) -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    amp = 0.5 * (np.percentile(y, 95) - np.percentile(y, 5))
    y0 = float(np.median(y))
    f0 = estimate_ramsey_frequency(x_ns, y)
    tspan = float(np.max(x_ns) - np.min(x_ns))
    seeds: list[np.ndarray] = []
    for _ in range(n_starts):
        seeds.append(
            np.array(
                [
                    y0 + (rng.random() - 0.5) * 0.01,
                    max(1e-4, amp * (0.4 + 1.6 * rng.random())),
                    f0 * (0.6 + 1.8 * rng.random()),
                    (2 * rng.random() - 1) * np.pi,
                    200.0 + min(10000.0, tspan * (0.2 + 1.2 * rng.random())),
                    0.8 + 3.0 * rng.random(),
                ]
            )
        )
    return seeds


def _seed_ramsey_hyperfine(x_ns: np.ndarray, y: np.ndarray, n_starts: int = 18) -> list[np.ndarray]:
    # same parameterization as simple; model embeds fixed +/-2.16 MHz splitting
    return _seed_ramsey_simple(x_ns, y, n_starts=n_starts)


def _seed_ramsey_extended(x_ns: np.ndarray, y: np.ndarray, n_starts: int = 24) -> list[np.ndarray]:
    rng = np.random.default_rng(17)
    amp = 0.5 * (np.percentile(y, 95) - np.percentile(y, 5))
    y0 = float(np.median(y))
    f0 = estimate_ramsey_frequency(x_ns, y)
    tspan = float(np.max(x_ns) - np.min(x_ns))
    seeds: list[np.ndarray] = []
    for _ in range(n_starts):
        seeds.append(
            np.array(
                [
                    max(1e-4, amp * (0.2 + 1.0 * rng.random())),  # A1
                    max(1e-4, amp * (0.1 + 1.2 * rng.random())),  # A2
                    120.0 + min(12000.0, tspan * (0.2 + 1.6 * rng.random())),  # T2*
                    0.8 + 3.0 * rng.random(),  # n
                    f0 * (0.7 + 1.8 * rng.random()),  # f1
                    f0 * (0.8 + 2.2 * rng.random()),  # f2
                    (2 * rng.random() - 1) * np.pi,  # phi1
                    (2 * rng.random() - 1) * np.pi,  # phi2
                    y0 + (rng.random() - 0.5) * 0.01,  # B
                    (rng.random() - 0.5) * 0.04,  # D
                    100.0 + min(9000.0, tspan * (0.2 + 2.0 * rng.random())),  # Td
                ]
            )
        )
    return seeds


def ramsey_weights(x, y, sem=None):
    weights = np.ones_like(y, dtype=float)
    weights[x > 2200] = .7
    weights[x > 3200] = .5
    w_sem = sem_to_weights(sem) if sem is not None and len(sem) == len(y) else None
    return weights if w_sem is None else weights * w_sem


def fit_ramsey_physics_first(x_ns, y, sem=None, *, multistart=None, locks=None, model="Auto") -> RamseySelectionResult:
    """Fit the requested Ramsey candidate(s), retaining general parameter controls."""
    from .fit_engine import parameter_controls
    x_ns, y = np.asarray(x_ns, float), np.asarray(y, float)
    weights = ramsey_weights(x_ns, y, sem)
    lo, hi = float(np.min(y)), float(np.max(y))
    span = max(hi - lo, 1e-9)
    tspan = max(float(np.ptp(x_ns)), 1.)
    dx = float(np.min(np.diff(np.unique(x_ns)))) if len(np.unique(x_ns)) > 1 else 1.
    maxfreq = max(.5 / dx, 2.1e-4)
    short = ["y0", "A", "f", "phi", "tau", "n"]
    bounds = ([lo - 2 * span, 0., min(2e-4, maxfreq/100), -2*np.pi, max(dx/10, 1e-6), .5],
              [hi + 2 * span, 4 * span, maxfreq, 2*np.pi, max(12000., 20*tspan), 5.])
    definitions = [
        ("RamseySimple", ramsey_simple, short, *bounds, _seed_ramsey_simple, 18),
        ("RamseyHyperfine", ramsey_hyperfine_n14, short, *bounds, _seed_ramsey_hyperfine, 18),
        ("RamseyExtended", ramsey_extended,
         ["A1", "A2", "T2_star", "n", "f1", "f2", "phi1", "phi2", "B", "D", "Td"],
         [0., 0., max(dx/10,1e-6), .5, bounds[0][2], bounds[0][2], -2*np.pi, -2*np.pi, lo-2*span, -4*span, max(dx/10,1e-6)],
         [4*span, 4*span, max(12000.,20*tspan), 5., maxfreq, maxfreq, 2*np.pi, 2*np.pi, hi+2*span, 4*span, max(9000.,20*tspan)],
         _seed_ramsey_extended, 24),
    ]
    aliases = {"Simple damped cosine":"RamseySimple", "N14 hyperfine":"RamseyHyperfine", "Extended":"RamseyExtended"}
    wanted = aliases.get(model, model)
    if wanted not in ('Auto', '', None) and wanted not in {d[0] for d in definitions}:
        raise ValueError(f"Unknown Ramsey model: {model}")
    candidates = []
    for name, func, names, lower, upper, seed_fn, default_starts in definitions:
        if wanted not in ('Auto', '', None) and wanted != name:
            continue
        count = default_starts if multistart is None else int(multistart)
        if count < 1: raise ValueError('Multistart must be positive.')
        with parameter_controls(locks):
            candidate = fit_model_multistart(model_name=name, model_func=func, x=x_ns, y=y,
                param_names=names, lower=lower, upper=upper, seeds=seed_fn(x_ns,y,n_starts=count),
                weights=weights, x_scale='jac')
        if sem is not None:
            nfree = sum(candidate.free_mask) if candidate.free_mask is not None else len(candidate.params)
            chi2 = reduced_chi2(y, candidate.y_fit, sem, nfree)
            if np.isfinite(chi2): candidate.extras.update(reduced_chi2=chi2, sem_weights_applied=True)
        candidates.append(candidate)
    high_quality = [c for c in candidates if c.success and np.isfinite(c.r2) and c.r2 >= .85]
    if high_quality:
        best = min(high_quality, key=lambda r:(sum(r.bound_hits), r.bic, -r.r2))
        selected_by = 'quality_gate_r2>=0.85_then_bound_hits_then_bic'
    else:
        best = max(candidates, key=lambda r:(r.success, r.r2 if np.isfinite(r.r2) else -np.inf, -sum(r.bound_hits)))
        selected_by = 'fallback_max_r2_then_bound_hits_then_bic'
    return RamseySelectionResult(best=best, candidates=candidates, selected_by=selected_by)
