from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit_engine import FitResult, fit_model_multistart
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


def fit_ramsey_physics_first(x_ns: np.ndarray, y: np.ndarray) -> RamseySelectionResult:
    # Use full trace, with slight tail down-weighting for robustness to late-time noise.
    weights = np.ones_like(y)
    weights[x_ns > 2200] = 0.7
    weights[x_ns > 3200] = 0.5

    candidates: list[FitResult] = []

    simple = fit_model_multistart(
        model_name="RamseySimple",
        model_func=ramsey_simple,
        x=x_ns,
        y=y,
        param_names=["y0", "A", "f", "phi", "tau", "n"],
        lower=[-0.2, 0.0, 2e-4, -2 * np.pi, 50.0, 0.5],
        upper=[0.1, 0.2, 0.03, 2 * np.pi, 12000.0, 5.0],
        seeds=_seed_ramsey_simple(x_ns, y),
        weights=weights,
    )
    candidates.append(simple)

    hyper = fit_model_multistart(
        model_name="RamseyHyperfine",
        model_func=ramsey_hyperfine_n14,
        x=x_ns,
        y=y,
        param_names=["y0", "A", "f", "phi", "tau", "n"],
        lower=[-0.2, 0.0, 2e-4, -2 * np.pi, 50.0, 0.5],
        upper=[0.1, 0.2, 0.03, 2 * np.pi, 12000.0, 5.0],
        seeds=_seed_ramsey_hyperfine(x_ns, y),
        weights=weights,
    )
    candidates.append(hyper)

    extended = fit_model_multistart(
        model_name="RamseyExtended",
        model_func=ramsey_extended,
        x=x_ns,
        y=y,
        param_names=["A1", "A2", "T2_star", "n", "f1", "f2", "phi1", "phi2", "B", "D", "Td"],
        lower=[0.0, 0.0, 50.0, 0.5, 2e-4, 2e-4, -2 * np.pi, -2 * np.pi, -0.2, -0.2, 10.0],
        upper=[0.2, 0.2, 12000.0, 5.0, 0.03, 0.03, 2 * np.pi, 2 * np.pi, 0.1, 0.2, 9000.0],
        seeds=_seed_ramsey_extended(x_ns, y),
        weights=weights,
    )
    candidates.append(extended)

    # Physics-first selection with fit-quality guard:
    # 1) prefer candidates that clear a minimum quality target (R^2 >= 0.85)
    # 2) among those, prefer fewer bound hits, then lower BIC
    # 3) if none clear target, fall back to highest R^2
    high_quality = [c for c in candidates if np.isfinite(c.r2) and c.r2 >= 0.85]
    if high_quality:
        ranked = sorted(high_quality, key=lambda r: (sum(r.bound_hits), r.bic, -r.r2))
        best = ranked[0]
        selected_by = "quality_gate_r2>=0.85_then_bound_hits_then_bic"
    else:
        ranked = sorted(candidates, key=lambda r: (-r.r2, sum(r.bound_hits), r.bic))
        best = ranked[0]
        selected_by = "fallback_max_r2_then_bound_hits_then_bic"
    return RamseySelectionResult(best=best, candidates=candidates, selected_by=selected_by)

