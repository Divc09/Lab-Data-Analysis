from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .fit_engine import FitResult, fit_model_multistart
from .models import (
    build_custom_expression_model,
    odmr_lorentzian,
    odmr_multi_lorentzian,
    rabi_chirp_model,
    rabi_cosine_decay_model,
    rabi_dual_model,
    rabi_model,
    rabi_phase_ramp_model,
    rabi_shifted_model,
    spin_echo_model,
    t1_decay_exponential_model,
    t1_decay_stretched_model,
    t1_rise_exponential_model,
    t1_rise_stretched_model,
)
from .ramsey import estimate_ramsey_frequency
from .rabi_utils import fit_rabi_envelope

if TYPE_CHECKING:
    from .io_mat import ExperimentTrace


LockMap = Mapping[str, tuple[float, bool, float | None, float | None]]


@dataclass(frozen=True)
class FitWorkflowConfig:
    multistart: int = 12
    rabi_use_sem: bool = True
    rabi_pi_lock: bool = False
    rabi_pi_ns: float = 200.0
    rabi_pi2_ns: float = 100.0
    rabi_dead_time_ns: float = 0.0
    rabi_mode: str = "Phase-ramp (recommended physical fit)"
    odmr_multipeak: bool = False
    odmr_peak_count: int = 2
    odmr_auto_peak: bool = True
    odmr_polarity: str = "Auto"
    odmr_peak_ranges: tuple[tuple[float, float], ...] = ()
    t1_prep_mode: str = "Auto"
    t1_model_family: str = "Stretched exponential"
    custom_spec: dict | None = None
    custom_expr: str = ""


def _scaled_perturbation(lower: list[float], upper: list[float], frac: float = 0.05) -> list[float]:
    return [frac * (u - l) for l, u in zip(lower, upper)]


def _apply_locks(
    param_names: list[str],
    lower: list[float],
    upper: list[float],
    seed: np.ndarray,
    locks: LockMap | None,
) -> None:
    if not locks:
        return
    for i, name in enumerate(param_names):
        if name not in locks:
            continue
        value, is_locked, lo, hi = locks[name]
        if lo is not None:
            lower[i] = lo
        if hi is not None:
            upper[i] = hi
        if is_locked:
            lower[i] = value
            upper[i] = value
            seed[i] = value


def _spin_echo_seed_and_bounds(exp_type: str, tspan: float) -> tuple[list[float], list[float], np.ndarray]:
    lower = [-0.2, -0.2, 1.0, 0.5]
    if exp_type == "DynamicDecoupling":
        upper = [0.2, 0.2, max(200000.0, 8.0 * tspan), 5.0]
        seed0 = np.array([0.0, 0.0, max(1000.0, tspan / 3.0), 1.5])
    else:
        upper = [0.2, 0.2, 20000.0, 5.0]
        seed0 = np.array([0.0, 0.0, max(100.0, tspan / 3.0), 1.5])
    return lower, upper, seed0


def _with_selection(result: FitResult, note: str, extras: dict[str, object] | None = None) -> FitResult:
    result.selection_note = note
    if extras:
        result.extras.update(extras)
    return result


def _append_message(result: FitResult, text: str) -> FitResult:
    if text:
        result.message = f"{result.message}\n{text}".strip()
    return result


def _detect_odmr_polarity(x: np.ndarray, y: np.ndarray, mode: str) -> str:
    requested = mode.lower().strip()
    if requested in {"peak", "dip"}:
        return requested
    peak_center = float(x[int(np.argmax(y))]) if len(x) else 0.0
    dip_center = float(x[int(np.argmin(y))]) if len(x) else 0.0
    if abs(peak_center - float(np.median(x))) <= abs(dip_center - float(np.median(x))):
        return "peak"
    return "dip"


def _odmr_detection_signal(y: np.ndarray, polarity: str) -> np.ndarray:
    if polarity == "peak":
        return y - np.min(y)
    return np.max(y) - y


def _detect_odmr_peaks(x: np.ndarray, y: np.ndarray, polarity: str) -> np.ndarray:
    if len(x) < 7:
        return np.array([], dtype=int)
    dx = float(np.median(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        dx = 1.0
    span = float(np.max(x) - np.min(x))
    inv = _odmr_detection_signal(y, polarity)
    win = min(len(y) if len(y) % 2 == 1 else len(y) - 1, 7)
    if win >= 5:
        inv = savgol_filter(inv, win, 2)
    prominence = max(1e-12, 0.08 * float(np.ptp(inv)))
    min_distance = max(1, int(round(0.08 * span / dx)))
    peaks, _ = find_peaks(inv, prominence=prominence, distance=min_distance)
    return peaks.astype(int)


def _odmr_peaks_in_ranges(
    x: np.ndarray,
    y: np.ndarray,
    polarity: str,
    peak_indices: np.ndarray,
    ranges: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Choose at most one resonance per requested range.

    A local extremum is used as a seed when the generic peak detector does not
    find a sufficiently prominent peak inside a requested range. This makes a
    user-entered range an actual fit constraint rather than a display filter.
    """
    if not ranges:
        return np.asarray(peak_indices, dtype=int), []

    chosen: list[int] = []
    chosen_ranges: list[tuple[float, float]] = []
    detection_signal = _odmr_detection_signal(y, polarity)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    for raw_lo, raw_hi in ranges:
        lo, hi = sorted((float(raw_lo), float(raw_hi)))
        lo = max(lo, x_min)
        hi = min(hi, x_max)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
            continue
        in_range = np.flatnonzero((x >= lo) & (x <= hi))
        if not len(in_range):
            continue
        detected = np.asarray(peak_indices, dtype=int)
        detected = detected[(x[detected] >= lo) & (x[detected] <= hi)]
        candidates = detected if len(detected) else in_range
        best = int(candidates[np.argmax(detection_signal[candidates])])
        if best not in chosen:
            chosen.append(best)
            chosen_ranges.append((lo, hi))
    return np.asarray(chosen, dtype=int), chosen_ranges


def _prepare_t1_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prep_mode: str,
    trace: "ExperimentTrace | None" = None,
) -> tuple[np.ndarray, str]:
    mode = prep_mode.lower().strip()
    auto = mode == "auto"
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
    use_normalized = mode.startswith("normalize") or (auto and corr > 0.25)
    if not use_normalized:
        return y, "direct"
    base = y.copy()
    if trace is not None and len(trace.y) == len(y):
        base = trace.y.astype(float, copy=True)
    lo = float(np.min(base))
    span = max(float(np.max(base) - lo), 1e-9)
    normalized = (base - lo) / span
    return 1.0 - normalized, "normalized_1_minus_data"


def _rabi_fit_axis(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) - float(np.min(x))


def _first_peak_from_series(x: np.ndarray, y: np.ndarray) -> float:
    ys = np.asarray(y, dtype=float).copy()
    if len(ys) >= 5:
        win = min(len(ys) if len(ys) % 2 == 1 else len(ys) - 1, 7)
        if win >= 5:
            ys = savgol_filter(ys, win, 2)
    prominence = max(1e-6, 0.25 * float(np.ptp(ys)))
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    distance = max(1, int(round(14.0 / max(dx, 1e-9))))
    peaks, _ = find_peaks(ys, prominence=prominence, distance=distance)
    if len(peaks):
        return float(x[int(peaks[0])])
    return float(x[int(np.argmax(ys))])


def _rabi_frequency_seed(x: np.ndarray, y: np.ndarray) -> float:
    ys = np.asarray(y, dtype=float).copy()
    if len(ys) >= 5:
        win = min(len(ys) if len(ys) % 2 == 1 else len(ys) - 1, 7)
        if win >= 5:
            ys = savgol_filter(ys, win, 2)
    prominence = max(1e-6, 0.25 * float(np.ptp(ys)))
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    distance = max(1, int(round(14.0 / max(dx, 1e-9))))
    peaks, _ = find_peaks(ys, prominence=prominence, distance=distance)
    if len(peaks) >= 2:
        period = float(x[int(peaks[1])] - x[int(peaks[0])])
        if period > 0:
            return 1.0 / period
    return max(estimate_ramsey_frequency(x, y), 5e-5)


def _rabi_peak_period_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    ys = np.asarray(y, dtype=float).copy()
    if len(ys) >= 5:
        win = min(len(ys) if len(ys) % 2 == 1 else len(ys) - 1, 11)
        if win >= 5:
            ys = savgol_filter(ys, win, 2)
    span_y = float(np.ptp(ys))
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    distance = max(1, int(round(12.0 / max(dx, 1e-9))))
    prominence = max(1e-6, 0.18 * span_y)
    peaks, _ = find_peaks(ys, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-ys, prominence=prominence, distance=distance)
    periods: list[float] = []
    if len(peaks) >= 2:
        periods.extend(float(v) for v in np.diff(x[peaks[: min(len(peaks), 16)]]))
    if len(troughs) >= 2:
        periods.extend(float(v) for v in np.diff(x[troughs[: min(len(troughs), 16)]]))
    valid = [p for p in periods if np.isfinite(p) and p > 0.0]
    if valid:
        return float(np.median(valid)), int(len(peaks)), int(len(troughs))
    seed = _rabi_frequency_seed(x, y)
    return float(1.0 / max(seed, 1e-9)), int(len(peaks)), int(len(troughs))


def _rabi_long_scan_features(x: np.ndarray, y: np.ndarray) -> dict[str, float | bool]:
    span = float(np.max(x) - np.min(x)) if len(x) else 0.0
    period, peak_count, trough_count = _rabi_peak_period_stats(x, y)
    freq = 1.0 / max(period, 1e-9)
    cycles = span * freq
    n = len(y)
    head_n = max(8, min(n, int(round(0.16 * n))))
    tail_n = max(8, min(n, int(round(0.25 * n))))
    early_ptp = float(np.percentile(y[:head_n], 95) - np.percentile(y[:head_n], 5)) if n else 0.0
    tail_ptp = float(np.percentile(y[-tail_n:], 95) - np.percentile(y[-tail_n:], 5)) if n else 0.0
    decay_ratio = early_ptp / max(tail_ptp, 1e-9)
    is_long = bool(
        span >= 1200.0
        and cycles >= 12.0
        and (peak_count + trough_count) >= 8
        and early_ptp >= max(0.012, 1.8 * tail_ptp)
    )
    return {
        "span_ns": float(span),
        "period_ns": float(period),
        "frequency_ghz": float(freq),
        "cycle_count": float(cycles),
        "peak_count": float(peak_count),
        "trough_count": float(trough_count),
        "early_ptp": float(early_ptp),
        "tail_ptp": float(tail_ptp),
        "decay_ratio": float(decay_ratio),
        "is_long_decayed": is_long,
    }


def _rabi_fit_weights(trace: "ExperimentTrace | None", x: np.ndarray, *, enabled: bool) -> np.ndarray | None:
    if not enabled or trace is None or trace.y_sem is None:
        return None
    if len(trace.y_sem) != len(x):
        return None
    if not np.allclose(trace.x_ns, x):
        return None
    sem = np.asarray(trace.y_sem, dtype=float)
    if not np.all(np.isfinite(sem)) or not np.all(sem > 0):
        return None
    weights = 1.0 / np.square(sem)
    weights = weights / float(np.median(weights))
    return weights


def _rabi_calibration_weights(x: np.ndarray, y: np.ndarray, base_weights: np.ndarray | None = None) -> np.ndarray:
    x_rel = _rabi_fit_axis(x)
    span = max(float(np.max(x_rel) - np.min(x_rel)), 1.0)
    early = np.exp(-x_rel / max(0.20 * span, 1.0))
    local = np.abs(np.asarray(y, dtype=float) - float(np.median(y)))
    if float(np.max(local)) > 0.0:
        local = local / float(np.max(local))
    weights = 1.0 + 3.5 * early + 1.5 * local
    if base_weights is not None and len(base_weights) == len(weights):
        weights = weights * np.asarray(base_weights, dtype=float)
    return weights / max(float(np.median(weights)), 1e-12)


def _weighted_fit_quality(y: np.ndarray, y_fit: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    w = np.asarray(weights, dtype=float)
    w = w / max(float(np.mean(w)), 1e-12)
    resid = np.asarray(y, dtype=float) - np.asarray(y_fit, dtype=float)
    wrmse = float(np.sqrt(np.mean(w * resid * resid)))
    center = float(np.average(y, weights=w))
    ss_res = float(np.sum(w * resid * resid))
    ss_tot = float(np.sum(w * (np.asarray(y, dtype=float) - center) ** 2))
    wr2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return wr2, wrmse


def _fit_rabi_constant_reference(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None, n_starts: int) -> FitResult:
    rng = np.random.default_rng(71)
    x_fit = _rabi_fit_axis(x)
    c0 = float(np.median(y))
    a0 = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    f0 = _rabi_frequency_seed(x, y)
    names = ["y0", "A", "T", "beta", "f", "phi"]
    lower = [-1.0, 0.0, 1.0, 0.1, 1.0 / 100.0, -10.0 * np.pi]
    upper = [1.0, 1.0, 1e6, 4.0, 1.0 / 5.0, 10.0 * np.pi]
    seed0 = np.array([c0, max(a0, 1e-4), max(600.0, float(np.ptp(x)) * 3.0), 1.2, f0, 0.0])
    scale = _scaled_perturbation(lower, upper)
    seeds = [seed0 + rng.normal(0, scale) for _ in range(max(8, n_starts // 2))]
    result = fit_model_multistart(
        model_name="RabiCosine",
        model_func=rabi_cosine_decay_model,
        x=x_fit,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        weights=weights,
        selection_note="reference constant-frequency damped cosine Rabi fit",
    )
    result.extras.update({"x_offset_ns": float(np.min(x)), "rabi_model_family": "constant"})
    return result


def _fit_rabi_long_damped(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None, n_starts: int) -> FitResult:
    rng = np.random.default_rng(89)
    x_fit = _rabi_fit_axis(x)
    features = _rabi_long_scan_features(x, y)
    period = float(features["period_ns"])
    f0 = float(features["frequency_ghz"])
    span = max(float(features["span_ns"]), 1.0)
    tail_n = max(8, min(len(y), int(round(0.25 * len(y)))))
    c0 = float(np.median(y[-tail_n:]))
    head_n = max(12, min(len(y), int(round(0.14 * len(y)))))
    a0 = float(0.5 * (np.percentile(y[:head_n], 95) - np.percentile(y[:head_n], 5)))
    amp_bound = max(0.12, 5.0 * max(a0, float(np.percentile(y, 95) - np.percentile(y, 5)), 1e-4))
    f_half_width = max(0.004, 0.18 * max(f0, 1e-5))
    names = ["y0", "A", "T", "beta", "f", "phi"]
    lower = [-1.0, -amp_bound, 20.0, 0.3, max(5e-5, f0 - f_half_width), -8.0 * np.pi]
    upper = [1.0, amp_bound, max(1e6, 20.0 * span), 4.0, min(0.2, f0 + f_half_width), 8.0 * np.pi]
    if lower[4] >= upper[4]:
        lower[4], upper[4] = max(5e-5, 0.5 * f0), min(0.2, 1.5 * f0)
    decay_guesses = [max(80.0, span / 10.0), max(150.0, span / 6.0), max(250.0, span / 3.0), max(400.0, span / 2.0)]
    beta_guesses = [1.0, 1.6, 2.4, 3.4]
    freq_guesses = [f0, max(lower[4], f0 * 0.97), min(upper[4], f0 * 1.03)]
    seeds: list[np.ndarray] = []
    for i in range(max(10, min(28, int(n_starts) + 8))):
        T = decay_guesses[i % len(decay_guesses)]
        beta = beta_guesses[(i // len(decay_guesses)) % len(beta_guesses)]
        f = freq_guesses[i % len(freq_guesses)]
        sign = -1.0 if i % 2 else 1.0
        phi = 0.0 if i % 4 < 2 else np.pi / 2.0
        seed = np.array([c0, sign * max(a0, 1e-4), T, beta, f, phi], dtype=float)
        if i >= 8:
            seed += rng.normal(0.0, [0.01, 0.15 * max(a0, 1e-4), 0.15 * T, 0.15, 0.01 * max(f0, 1e-5), 0.4])
        seeds.append(seed)
    cal_weights = _rabi_calibration_weights(x, y, weights)
    result = fit_model_multistart(
        model_name="RabiLongDamped",
        model_func=rabi_cosine_decay_model,
        x=x_fit,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        weights=cal_weights,
        loss="soft_l1",
        f_scale=max(1e-6, 0.05 * float(np.percentile(y[:head_n], 95) - np.percentile(y[:head_n], 5))),
        max_nfev=8000,
        selection_note="selected long damped Rabi model",
    )
    weighted_r2, weighted_rmse = _weighted_fit_quality(y, result.y_fit, cal_weights)
    early_rmse = _rabi_early_rmse(y, result.y_fit, x)
    result.extras.update(
        {
            "x_offset_ns": float(np.min(x)),
            "rabi_model_family": "long_damped",
            "rabi_long_scan": True,
            "rabi_long_features": features,
            "rabi_calibration_weighted_r2": float(weighted_r2),
            "rabi_calibration_weighted_rmse": float(weighted_rmse),
            "rabi_early_rmse": float(early_rmse),
            "rabi_period_seed_ns": float(period),
        }
    )
    return result


def _fit_rabi_phase_ramp(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None, n_starts: int) -> FitResult:
    rng = np.random.default_rng(73)
    x_fit = _rabi_fit_axis(x)
    c0 = float(np.median(y))
    a0 = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    f0 = _rabi_frequency_seed(x, y)
    names = ["y0", "A", "f", "tau_ramp", "phi"]
    lower = [-1.0, 0.0, 1.0 / 100.0, 0.01, -10.0 * np.pi]
    upper = [1.0, 1.0, 1.0 / 5.0, 100.0, 10.0 * np.pi]
    seed0 = np.array([c0, max(a0, 1e-4), f0, 15.0, 0.0])
    scale = _scaled_perturbation(lower, upper)
    seeds = [seed0 + rng.normal(0, scale) for _ in range(n_starts)]
    result = fit_model_multistart(
        model_name="RabiPhaseRamp",
        model_func=rabi_phase_ramp_model,
        x=x_fit,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        weights=weights,
        selection_note="selected phase-ramp Rabi model",
    )
    result.extras.update({"x_offset_ns": float(np.min(x)), "rabi_model_family": "phase_ramp"})
    return result


def _fit_rabi_chirp(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None, n_starts: int) -> FitResult:
    rng = np.random.default_rng(79)
    x_fit = _rabi_fit_axis(x)
    c0 = float(np.median(y))
    a0 = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    f0 = _rabi_frequency_seed(x, y)
    names = ["y0", "A", "T", "beta", "f", "alpha", "phi"]
    lower = [-1.0, 0.0, 1.0, 0.1, 1.0 / 100.0, -1e-3, -10.0 * np.pi]
    upper = [1.0, 1.0, 1e6, 4.0, 1.0 / 5.0, 1e-3, 10.0 * np.pi]
    seed0 = np.array([c0, max(a0, 1e-4), max(800.0, float(np.ptp(x)) * 3.0), 1.2, f0, 0.0, 0.0])
    scale = _scaled_perturbation(lower, upper)
    seeds = [seed0 + rng.normal(0, scale) for _ in range(max(8, n_starts // 2))]
    result = fit_model_multistart(
        model_name="RabiChirp",
        model_func=rabi_chirp_model,
        x=x_fit,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        weights=weights,
        selection_note="selected chirped Rabi model",
    )
    result.extras.update({"x_offset_ns": float(np.min(x)), "rabi_model_family": "chirped"})
    return result


def _rabi_early_rmse(y: np.ndarray, y_fit: np.ndarray, x: np.ndarray) -> float:
    cutoff = float(np.min(x) + 0.2 * (np.max(x) - np.min(x)))
    mask = x <= cutoff
    if not np.any(mask):
        return float(np.sqrt(np.mean((y - y_fit) ** 2)))
    return float(np.sqrt(np.mean((y[mask] - y_fit[mask]) ** 2)))


def _choose_rabi_model(
    *,
    x: np.ndarray,
    y: np.ndarray,
    phase_ramp: FitResult,
    constant: FitResult,
    chirped: FitResult | None,
    long_damped: FitResult | None = None,
) -> FitResult:
    if long_damped is not None and bool(long_damped.extras.get("rabi_long_scan", False)):
        long_features = dict(long_damped.extras.get("rabi_long_features", {}))
        long_wr2 = float(long_damped.extras.get("rabi_calibration_weighted_r2", long_damped.r2))
        phase_wr2 = _weighted_fit_quality(y, phase_ramp.y_fit, _rabi_calibration_weights(x, y))[0]
        constant_wr2 = _weighted_fit_quality(y, constant.y_fit, _rabi_calibration_weights(x, y))[0]
        best_existing_wr2 = max(float(phase_wr2), float(constant_wr2), float(chirped.r2 if chirped is not None else -np.inf))
        if long_wr2 >= best_existing_wr2 + 0.05 or long_damped.rmse <= min(phase_ramp.rmse, constant.rmse) * 1.15:
            note = (
                "auto-selected long damped Rabi for decayed long scan"
                f" ({long_features.get('cycle_count', 0.0):.1f} cycles, decay ratio {long_features.get('decay_ratio', 0.0):.2f})"
            )
            long_damped.extras["rabi_long_phase_reference_r2"] = float(phase_ramp.r2)
            long_damped.extras["rabi_long_phase_reference_weighted_r2"] = float(phase_wr2)
            long_damped.extras["rabi_long_constant_reference_r2"] = float(constant.r2)
            long_damped.extras["rabi_long_constant_reference_weighted_r2"] = float(constant_wr2)
            return _with_selection(long_damped, note)

    data_first_peak = _first_peak_from_series(x, y)
    ramp_first_peak = _first_peak_from_series(x, phase_ramp.y_fit)
    ramp_peak_error = abs(ramp_first_peak - data_first_peak)
    ramp_rmse_improvement = phase_ramp.rmse / max(constant.rmse, 1e-12)

    if ramp_peak_error <= 2.5 and ramp_rmse_improvement <= 0.85:
        return _with_selection(phase_ramp, "selected phase-ramp Rabi model")

    if chirped is not None:
        chirped_peak_error = abs(_first_peak_from_series(x, chirped.y_fit) - data_first_peak)
        if chirped.rmse <= phase_ramp.rmse * 0.75 or (chirped_peak_error <= 2.5 and chirped.r2 >= max(0.95, phase_ramp.r2 + 0.03)):
            chirped.extras["rabi_physical_reference_params"] = {
                name: float(value) for name, value in zip(phase_ramp.param_names, phase_ramp.params)
            }
            chirped.extras["rabi_physical_reference_model"] = "RabiPhaseRamp"
            return _with_selection(chirped, "selected chirped Rabi fallback because phase-ramp still missed startup alignment")

    if phase_ramp.rmse < constant.rmse:
        return _with_selection(phase_ramp, f"selected phase-ramp Rabi model; first-peak mismatch remains {ramp_peak_error:.2f} ns")
    return _with_selection(constant, "fell back to constant-frequency Rabi reference fit")


def _fit_rabi_shifted(
    x: np.ndarray,
    y: np.ndarray,
    config: FitWorkflowConfig,
    locks: LockMap | None = None,
) -> FitResult:
    rng = np.random.default_rng(22)
    tspan = float(np.max(x) - np.min(x))
    yamp = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    y0 = float(np.median(y))
    f0 = estimate_ramsey_frequency(x, y)
    n_starts = int(config.multistart)

    names = ["y0", "m", "A", "f", "phi", "tau", "t0"]
    lower = [-0.2, -1e-3, -0.2, 1e-5, -2 * np.pi, 1.0, float(np.min(x) - 0.1 * tspan)]
    upper = [0.2, 1e-3, 0.2, 0.2, 2 * np.pi, max(12000.0, 8.0 * tspan), float(np.min(x) + 0.2 * tspan)]
    f_candidates = [max(f0, 5e-5), max(f0 * 0.5, 5e-5), min(max(f0, 5e-5) * 1.7, 0.2)]
    seed0 = np.array([y0, 0.0, yamp, f_candidates[0], 0.0, max(200.0, tspan * 0.8), float(np.min(x))])
    _apply_locks(names, lower, upper, seed0, locks)
    scale = _scaled_perturbation(lower, upper)
    seeds: list[np.ndarray] = []
    for i in range(n_starts):
        seed = seed0.copy()
        seed[3] = f_candidates[i % len(f_candidates)]
        seed[5] = max(100.0, tspan * (0.45 + 1.7 * rng.random()))
        seed[6] = float(np.min(x) + tspan * (-0.02 + 0.08 * rng.random()))
        seeds.append(seed + rng.normal(0, scale))

    if config.rabi_pi_lock:
        dead_time_ns = float(config.rabi_dead_time_ns)
        f_lock = np.nan
        if config.rabi_pi_ns > 0 and (config.rabi_pi_ns + dead_time_ns) > 0:
            f_lock = 1.0 / (2.0 * (config.rabi_pi_ns + dead_time_ns))
        elif config.rabi_pi2_ns > 0 and (config.rabi_pi2_ns + dead_time_ns) > 0:
            f_lock = 1.0 / (4.0 * (config.rabi_pi2_ns + dead_time_ns))
        if np.isfinite(f_lock) and f_lock > 0:
            half_width = max(5e-6, 0.08 * f_lock)
            lower[3] = max(lower[3], f_lock - half_width)
            upper[3] = min(upper[3], f_lock + half_width)
            for seed in seeds:
                seed[3] = np.clip(f_lock + rng.normal(0, 0.25 * half_width), lower[3], upper[3])

    return fit_model_multistart(
        model_name="RabiShifted",
        model_func=rabi_shifted_model,
        x=x,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
    )


def _fit_rabi_classic(
    x: np.ndarray,
    y: np.ndarray,
    config: FitWorkflowConfig,
    locks: LockMap | None = None,
) -> FitResult:
    rng = np.random.default_rng(21)
    tspan = float(np.max(x) - np.min(x))
    y_min = float(np.percentile(y, 5))
    y_max = float(np.percentile(y, 95))
    yamp = max(y_max - y_min, 1e-6)
    y0 = y_min

    ys = y.copy()
    win = min(len(y) if len(y) % 2 == 1 else len(y) - 1, 7)
    if win >= 5:
        ys = savgol_filter(y, win, 2)
    prominence = max(1e-6, 0.25 * float(np.ptp(ys)))
    peak_distance = max(4, int(round(0.03 * max(len(x), 1))))
    peaks, _ = find_peaks(ys, prominence=prominence, distance=peak_distance)
    first_peak = float(x[peaks[0]]) if len(peaks) else float(x[np.argmax(ys)])
    if len(peaks) >= 2:
        period = float(x[peaks[1]] - x[peaks[0]])
        f0 = 1.0 / max(period, 1e-9)
    else:
        f0 = estimate_ramsey_frequency(x, y)
    n_starts = int(config.multistart)

    names = ["y0", "m", "A", "f", "t0", "tau"]
    lower = [-0.02, -1e-3, 0.0, 1e-5, float(np.min(x) - 10.0), 1.0]
    upper = [0.08, 1e-3, 0.2, 0.2, float(np.min(x) + 40.0), max(12000.0, 8.0 * tspan)]
    f_candidates = [max(f0, 5e-5), max(f0 * 0.9, 5e-5), min(max(f0, 5e-5) * 1.1, 0.2)]
    t0_seed = max(float(np.min(x) - 2.0), first_peak - 1.0 / (2.0 * max(f_candidates[0], 1e-6)))
    seed0 = np.array([y0, 0.0, yamp, f_candidates[0], t0_seed, max(200.0, tspan * 0.8)])
    _apply_locks(names, lower, upper, seed0, locks)
    scale = _scaled_perturbation(lower, upper)
    seeds: list[np.ndarray] = []
    for i in range(n_starts):
        seed = seed0.copy()
        seed[3] = f_candidates[i % len(f_candidates)]
        seed[4] = np.clip(t0_seed + rng.normal(0, 3.0), lower[4], upper[4])
        seed[5] = max(100.0, tspan * (0.45 + 1.7 * rng.random()))
        seeds.append(seed + rng.normal(0, scale))

    if config.rabi_pi_lock:
        dead_time_ns = float(config.rabi_dead_time_ns)
        f_lock = np.nan
        if config.rabi_pi_ns > 0 and (config.rabi_pi_ns + dead_time_ns) > 0:
            f_lock = 1.0 / (2.0 * (config.rabi_pi_ns + dead_time_ns))
        elif config.rabi_pi2_ns > 0 and (config.rabi_pi2_ns + dead_time_ns) > 0:
            f_lock = 1.0 / (4.0 * (config.rabi_pi2_ns + dead_time_ns))
        if np.isfinite(f_lock) and f_lock > 0:
            half_width = max(5e-6, 0.08 * f_lock)
            lower[3] = max(lower[3], f_lock - half_width)
            upper[3] = min(upper[3], f_lock + half_width)
            for seed in seeds:
                seed[3] = np.clip(f_lock + rng.normal(0, 0.25 * half_width), lower[3], upper[3])

    return fit_model_multistart(
        model_name="Rabi",
        model_func=rabi_model,
        x=x,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
    )


def _fit_rabi_dual(x: np.ndarray, y: np.ndarray, config: FitWorkflowConfig) -> FitResult:
    rng = np.random.default_rng(23)
    tspan = float(np.max(x) - np.min(x))
    yamp = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    y0 = float(np.median(y))
    f0 = max(estimate_ramsey_frequency(x, y), 5e-5)
    n_starts = int(config.multistart)

    names = ["y0", "m", "A1", "A2", "f1", "f2", "phi1", "phi2", "tau"]
    lower = [-0.2, -1e-3, -0.2, -0.2, 1e-5, 1e-5, -2 * np.pi, -2 * np.pi, 1.0]
    upper = [0.2, 1e-3, 0.2, 0.2, 0.2, 0.2, 2 * np.pi, 2 * np.pi, max(12000.0, 8.0 * tspan)]
    freq_pairs = [
        (f0, min(0.2, f0 * 1.18)),
        (max(5e-5, f0 * 0.88), min(0.2, f0 * 1.15)),
        (max(5e-5, f0 * 0.75), min(0.2, f0 * 1.3)),
    ]
    seed0 = np.array([y0, 0.0, 0.55 * yamp, 0.35 * yamp, freq_pairs[0][0], freq_pairs[0][1], 0.0, 0.5, max(200.0, tspan)])
    scale = _scaled_perturbation(lower, upper)
    seeds: list[np.ndarray] = []
    for i in range(n_starts):
        seed = seed0.copy()
        pair = freq_pairs[i % len(freq_pairs)]
        seed[4] = pair[0]
        seed[5] = pair[1]
        seed[8] = max(120.0, tspan * (0.45 + 1.7 * rng.random()))
        seeds.append(seed + rng.normal(0, scale))

    return fit_model_multistart(
        model_name="RabiDual",
        model_func=rabi_dual_model,
        x=x,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
    )


def _fit_t1(
    x: np.ndarray,
    y: np.ndarray,
    config: FitWorkflowConfig,
    *,
    trace: "ExperimentTrace | None" = None,
    locks: LockMap | None = None,
) -> FitResult:
    y_fit, prep_used = _prepare_t1_series(x, y, prep_mode=config.t1_prep_mode, trace=trace)
    rng = np.random.default_rng(41)
    tspan = float(np.max(x) - np.min(x))
    y0 = float(np.min(y_fit)) if prep_used == "direct" else float(np.percentile(y_fit, 10))
    y1 = float(np.max(y_fit))
    amp = max(abs(y1 - y0), 1e-6)

    family = config.t1_model_family.lower().strip()
    use_stretch = "stretched" in family
    is_norm = prep_used == "normalized_1_minus_data"
    if is_norm:
        base_name = "T1NormStretch" if use_stretch else "T1NormExp"
        model = t1_decay_stretched_model if use_stretch else t1_decay_exponential_model
        seed0 = np.array([max(0.0, y0), amp, max(100.0, tspan / 4.0)] + ([1.2] if use_stretch else []), dtype=float)
    else:
        base_name = "T1Stretch" if use_stretch else "T1Exp"
        model = t1_rise_stretched_model if use_stretch else t1_rise_exponential_model
        seed0 = np.array([y0, amp, max(100.0, tspan / 4.0)] + ([1.2] if use_stretch else []), dtype=float)

    names = ["y0", "A", "T1"] + (["n"] if use_stretch else [])
    lower = [min(y0, y1) - amp, 0.0, 1.0] + ([0.4] if use_stretch else [])
    upper = [max(y0, y1) + amp, max(4.0 * amp, 1.0), max(200000.0, 8.0 * tspan)] + ([4.5] if use_stretch else [])
    _apply_locks(names, lower, upper, seed0, locks)
    scale = _scaled_perturbation(lower, upper)
    seeds = [seed0 + rng.normal(0, scale) for _ in range(max(4, int(config.multistart)))]
    result = fit_model_multistart(
        model_name=base_name,
        model_func=model,
        x=x,
        y=y_fit,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        selection_note=f"selected {prep_used} T1 with {'stretched' if use_stretch else 'regular'} exponential model",
    )
    plot_label = "Normalized 1 - data [a.u.]" if prep_used == "normalized_1_minus_data" else "T1 signal [a.u.]"
    result.extras.update(
        {
            "t1_prep_mode": prep_used,
            "t1_model_family": "stretched" if use_stretch else "regular",
            "plot_y": y_fit.copy(),
            "plot_y_label": plot_label,
        }
    )
    return result


def _rabi_dual_pathology(dual: FitResult) -> str:
    p = {name: float(value) for name, value in zip(dual.param_names, dual.params)}
    if {"f1", "f2"} <= set(p):
        delta = abs(p["f1"] - p["f2"])
        scale = max(abs(p["f1"]), abs(p["f2"]), 1e-6)
        if delta < max(5e-4, 0.03 * scale):
            return "dual candidate rejected: frequencies are nearly identical"
    if {"A1", "A2"} <= set(p):
        if abs(p["A1"]) >= 0.19 or abs(p["A2"]) >= 0.19:
            return "dual candidate rejected: amplitude saturated at search bound"
    if sum(bool(v) for v in dual.bound_hits) >= 2:
        return "dual candidate rejected: multiple parameters hit bounds"
    return ""


def _select_rabi_candidate(classic: FitResult, dual: FitResult, config: FitWorkflowConfig) -> FitResult:
    mode = config.rabi_mode
    if mode in {"Single reliable Rabi", "Classic single (recommended)", "Single (recommended)"}:
        return _with_selection(classic, "selected classic single-frequency Rabi model")

    dual_reason = _rabi_dual_pathology(dual)
    if mode == "Dual only":
        return _with_selection(dual, dual_reason or "selected dual-frequency Rabi model")

    if dual_reason:
        return _with_selection(classic, f"auto-selected classic single Rabi; {dual_reason}")

    bic_gap = dual.bic - classic.bic
    r2_gain = dual.r2 - classic.r2
    if bic_gap < -6.0 and r2_gain > 0.003:
        return _with_selection(dual, "auto-selected dual-frequency Rabi by lower BIC and better residual fit")
    return _with_selection(classic, "auto-selected classic single Rabi for simpler, more stable fit")


def _attach_rabi_envelope(result: FitResult, x: np.ndarray, y: np.ndarray) -> FitResult:
    try:
        result.extras["rabi_envelope"] = fit_rabi_envelope(
            x,
            y,
            model_name=result.model_name,
            param_names=list(result.param_names),
            params=np.asarray(result.params, dtype=float),
            y_fit=np.asarray(result.y_fit, dtype=float),
        )
    except Exception as exc:
        result.extras["rabi_envelope"] = {"success": False, "message": f"Rabi envelope fit failed: {exc}"}
    return result


def _single_odmr_bounds(
    x: np.ndarray,
    y: np.ndarray,
    polarity: str,
    center_range: tuple[float, float] | None = None,
) -> tuple[list[str], list[float], list[float], np.ndarray]:
    lo = float(np.percentile(y, 5))
    hi = float(np.percentile(y, 95))
    amp = max(abs(hi - lo), 1e-9)
    span = max(float(np.max(x) - np.min(x)), 1e-9)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else span
    x0_lo, x0_hi = float(np.min(x)), float(np.max(x))
    if center_range is not None:
        x0_lo = max(x0_lo, float(min(center_range)))
        x0_hi = min(x0_hi, float(max(center_range)))
    center_mask = (x >= x0_lo) & (x <= x0_hi)
    center_indices = np.flatnonzero(center_mask)
    if not len(center_indices):
        center_indices = np.arange(len(x))
        x0_lo, x0_hi = float(np.min(x)), float(np.max(x))
    local_y = y[center_indices]
    local_idx = int(np.argmax(local_y)) if polarity == "peak" else int(np.argmin(local_y))
    x0 = float(x[center_indices[local_idx]])
    names = ["y0", "C", "w", "x0"]
    if polarity == "peak":
        lower = [lo - 2.0 * amp, 0.0, max(dx * 0.25, 1e-9), x0_lo]
        upper = [hi + 2.0 * amp, 4.0 * amp, span, x0_hi]
        seed0 = np.array([lo, max(0.6 * amp, 1e-9), max(3.0 * dx, 0.03 * span), x0])
    else:
        lower = [lo - 2.0 * amp, -4.0 * amp, max(dx * 0.25, 1e-9), x0_lo]
        upper = [hi + 2.0 * amp, 0.0, span, x0_hi]
        seed0 = np.array([hi, -max(0.6 * amp, 1e-9), max(3.0 * dx, 0.03 * span), x0])
    return names, lower, upper, seed0


def _fit_odmr_single(
    x: np.ndarray,
    y: np.ndarray,
    n_starts: int,
    polarity: str,
    locks: LockMap | None = None,
    center_range: tuple[float, float] | None = None,
) -> FitResult:
    rng = np.random.default_rng(29)
    y_scale = max(float(np.percentile(y, 95) - np.percentile(y, 5)), 1e-9)
    y_fit = y / y_scale
    names, lower, upper, seed0 = _single_odmr_bounds(x, y_fit, polarity, center_range=center_range)
    _apply_locks(names, lower, upper, seed0, locks)
    scale = _scaled_perturbation(lower, upper)
    seeds = [seed0 + rng.normal(0, scale) for _ in range(n_starts)]
    result = fit_model_multistart(
        model_name="ODMR",
        model_func=odmr_lorentzian,
        x=x,
        y=y_fit,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
    )
    return _rescale_odmr_result(result, y_scale, extras={"odmr_polarity": polarity})


def _fit_odmr_multi(
    x: np.ndarray,
    y: np.ndarray,
    n_starts: int,
    peak_indices: np.ndarray,
    max_peaks: int,
    polarity: str,
    center_ranges: list[tuple[float, float]] | None = None,
) -> FitResult:
    rng = np.random.default_rng(31)
    y_scale = max(float(np.percentile(y, 95) - np.percentile(y, 5)), 1e-9)
    y_fit = y / y_scale
    lo = float(np.percentile(y_fit, 5))
    hi = float(np.percentile(y_fit, 95))
    amp = max(abs(hi - lo), 1e-9)
    span = max(float(np.max(x) - np.min(x)), 1e-9)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else span
    peaks = np.asarray(peak_indices[:max_peaks], dtype=int)
    names = ["y0"]
    lower = [lo - 2.0 * amp]
    upper = [hi + 2.0 * amp]
    seed = [lo if polarity == "peak" else hi]
    for idx, pk in enumerate(peaks, start=1):
        names += [f"C{idx}", f"w{idx}", f"x0_{idx}"]
        peak_amp = max(0.45 * amp, 1e-9)
        signed_amp = peak_amp if polarity == "peak" else -peak_amp
        seed += [signed_amp, max(2.5 * dx, 0.025 * span), float(x[pk])]
        if center_ranges is not None and idx <= len(center_ranges):
            center_lo, center_hi = center_ranges[idx - 1]
        else:
            center_lo, center_hi = float(np.min(x)), float(np.max(x))
        lower += [0.0 if polarity == "peak" else -4.0 * amp, max(dx * 0.25, 1e-9), center_lo]
        upper += [4.0 * amp if polarity == "peak" else 0.0, 0.18 * span, center_hi]
    scale = _scaled_perturbation(lower, upper)
    seeds = [np.asarray(seed, dtype=float) + rng.normal(0, scale) for _ in range(n_starts)]
    result = fit_model_multistart(
        model_name="ODMRMulti",
        model_func=odmr_multi_lorentzian,
        x=x,
        y=y_fit,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
    )
    return _rescale_odmr_result(result, y_scale, extras={"odmr_polarity": polarity})


def _rescale_odmr_result(result: FitResult, y_scale: float, extras: dict[str, object] | None = None) -> FitResult:
    params = result.params.copy()
    errors = result.errors.copy()
    for idx, name in enumerate(result.param_names):
        if name == "y0" or name.startswith("C"):
            params[idx] *= y_scale
            errors[idx] *= y_scale
    return FitResult(
        model_name=result.model_name,
        param_names=result.param_names,
        params=params,
        errors=errors,
        y_fit=result.y_fit * y_scale,
        r2=result.r2,
        rmse=result.rmse * y_scale,
        aic=result.aic,
        bic=result.bic,
        durbin_watson=result.durbin_watson,
        bound_hits=result.bound_hits,
        success=result.success,
        message=result.message,
        selection_note=result.selection_note,
        extras={**dict(result.extras), **(extras or {})},
    )


def _select_odmr_candidate(single: FitResult, multi: FitResult | None, peak_indices: np.ndarray, x: np.ndarray) -> FitResult:
    detected = [float(v) for v in np.asarray(x[peak_indices], dtype=float)[:4]] if len(peak_indices) else []
    if multi is None:
        return _with_selection(single, "selected single-peak ODMR candidate", {"detected_peaks": detected})
    if multi.bic <= single.bic - 4.0 and multi.r2 >= single.r2 + 0.05:
        return _with_selection(multi, "selected multi-peak ODMR candidate by lower BIC and stronger explained variance", {"detected_peaks": detected})
    if sum(multi.bound_hits) > sum(single.bound_hits):
        return _with_selection(single, "selected single-peak ODMR candidate because multi-peak fit hit more bounds", {"detected_peaks": detected})
    if multi.bic <= single.bic - 4.0 and multi.r2 >= single.r2 - 0.02:
        return _with_selection(multi, "selected multi-peak ODMR candidate by lower BIC", {"detected_peaks": detected})
    return _with_selection(single, "selected single-peak ODMR candidate by parsimony", {"detected_peaks": detected})


def fit_non_ramsey(
    exp_type: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    config: FitWorkflowConfig | None = None,
    locks: LockMap | None = None,
    custom_spec: dict | None = None,
    trace_experiment_type: str | None = None,
    trace: "ExperimentTrace | None" = None,
) -> FitResult:
    cfg = config or FitWorkflowConfig()
    requested_exp_type = exp_type
    if exp_type == "DEERFrequency":
        exp_type = "ODMR"
    elif exp_type == "DEERDuration":
        exp_type = "Rabi"
    elif exp_type == "DEERDecay":
        exp_type = "SpinEcho"

    def _semantic_result(result: FitResult) -> FitResult:
        if requested_exp_type != exp_type:
            note = result.selection_note or "selected model"
            result.selection_note = f"{note}; semantic family {requested_exp_type}"
            result.extras["semantic_experiment_type"] = requested_exp_type
            result.extras["fit_engine_experiment_type"] = exp_type
        return result

    n_starts = max(4, int(cfg.multistart))
    tspan = float(np.max(x) - np.min(x))
    yamp = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    y0 = float(np.median(y))

    if exp_type == "Rabi":
        weights = _rabi_fit_weights(trace, x, enabled=bool(cfg.rabi_use_sem))
        mode = cfg.rabi_mode.lower().strip()
        if mode.startswith("long"):
            return _semantic_result(_attach_rabi_envelope(_with_selection(_fit_rabi_long_damped(x, y, weights, n_starts), "selected long damped Rabi model"), x, y))
        phase_ramp = _fit_rabi_phase_ramp(x, y, weights, n_starts)
        constant = _fit_rabi_constant_reference(x, y, weights, n_starts)
        chirped = _fit_rabi_chirp(x, y, weights, n_starts)
        features = _rabi_long_scan_features(x, y)
        long_damped = _fit_rabi_long_damped(x, y, weights, n_starts) if bool(features["is_long_decayed"]) else None
        if mode.startswith("phase-ramp"):
            if long_damped is not None and bool(features["is_long_decayed"]):
                chosen = _choose_rabi_model(x=x, y=y, phase_ramp=phase_ramp, constant=constant, chirped=chirped, long_damped=long_damped)
                if chosen is long_damped:
                    return _semantic_result(_attach_rabi_envelope(chosen, x, y))
            return _semantic_result(_attach_rabi_envelope(_with_selection(phase_ramp, "selected phase-ramp Rabi model"), x, y))
        if mode.startswith("chirped"):
            return _semantic_result(_attach_rabi_envelope(_with_selection(chirped, "selected chirped Rabi model"), x, y))
        if mode.startswith("constant-frequency"):
            return _semantic_result(_attach_rabi_envelope(_with_selection(constant, "selected constant-frequency damped cosine Rabi model"), x, y))
        return _semantic_result(_attach_rabi_envelope(_choose_rabi_model(x=x, y=y, phase_ramp=phase_ramp, constant=constant, chirped=chirped, long_damped=long_damped), x, y))

    if exp_type == "T1":
        return _semantic_result(_fit_t1(x, y, cfg, trace=trace, locks=locks))

    if exp_type in ("SpinEcho", "DynamicDecoupling"):
        names = ["y0", "A", "T2", "n"]
        lower, upper, seed0 = _spin_echo_seed_and_bounds(trace_experiment_type or exp_type, tspan)
        seed0[0] = y0
        seed0[1] = yamp
        _apply_locks(names, lower, upper, seed0, locks)
        rng = np.random.default_rng(33)
        scale = _scaled_perturbation(lower, upper)
        seeds = [seed0 + rng.normal(0, scale) for _ in range(n_starts)]
        return _semantic_result(fit_model_multistart(
            model_name="SpinEcho",
            model_func=spin_echo_model,
            x=x,
            y=y,
            param_names=names,
            lower=lower,
            upper=upper,
            seeds=seeds,
            selection_note="selected stretched-exponential decay model",
        ))

    if exp_type == "ODMR":
        polarities = ["peak", "dip"] if cfg.odmr_polarity.lower() == "auto" else [_detect_odmr_polarity(x, y, cfg.odmr_polarity)]
        candidates: list[FitResult] = []
        for polarity in polarities:
            peak_indices = _detect_odmr_peaks(x, y, polarity)
            peak_indices, constrained_ranges = _odmr_peaks_in_ranges(
                x, y, polarity, peak_indices, cfg.odmr_peak_ranges
            )
            single_range: tuple[float, float] | None = None
            if constrained_ranges:
                strengths = _odmr_detection_signal(y, polarity)[peak_indices]
                single_range = constrained_ranges[int(np.argmax(strengths))]
            single = _fit_odmr_single(
                x, y, n_starts, polarity, locks=locks, center_range=single_range
            )
            multi: FitResult | None = None
            if cfg.odmr_multipeak or len(peak_indices) > 1:
                max_peaks = int(min(max(2, cfg.odmr_peak_count), 4))
                if cfg.odmr_auto_peak:
                    max_peaks = int(min(max(2, len(peak_indices)), 4))
                if len(peak_indices) > 1:
                    multi = _fit_odmr_multi(
                        x,
                        y,
                        n_starts,
                        peak_indices,
                        max_peaks=max_peaks,
                        polarity=polarity,
                        center_ranges=constrained_ranges or None,
                    )
            candidates.append(_select_odmr_candidate(single, multi, peak_indices, x))
        best = min(candidates, key=lambda item: (sum(item.bound_hits), item.bic, -item.r2))
        polarity = str(best.extras.get("odmr_polarity", "dip"))
        if cfg.odmr_polarity.lower() == "auto":
            best.selection_note = f"{best.selection_note}; auto-selected {polarity} polarity"
        return _semantic_result(best)

    if exp_type == "CustomModel" and (custom_spec or cfg.custom_spec):
        rng = np.random.default_rng(35)
        spec = dict(custom_spec or cfg.custom_spec or {})
        expr_ui = cfg.custom_expr.strip()
        if expr_ui:
            spec["expression"] = expr_ui
        names = list(spec["param_names"])
        lower = list(spec["lower"])
        upper = list(spec["upper"])
        seed0 = np.asarray(spec["seed"], dtype=float)
        _apply_locks(names, lower, upper, seed0, locks)
        model = build_custom_expression_model(str(spec["expression"]), names)
        scale = [max(1e-6, 0.02 * max(abs(v), 1.0)) for v in seed0]
        seeds = [seed0 + rng.normal(0, scale, size=len(seed0)) for _ in range(n_starts)]
        return _semantic_result(fit_model_multistart(
            model_name=str(spec.get("name", "CustomModel")),
            model_func=model,
            x=x,
            y=y,
            param_names=names,
            lower=lower,
            upper=upper,
            seeds=seeds,
            selection_note="selected custom model",
        ))

    names = ["y0", "A", "T2", "n"]
    lower, upper, seed0 = _spin_echo_seed_and_bounds("SpinEcho", tspan)
    seed0[0] = y0
    seed0[1] = yamp
    rng = np.random.default_rng(37)
    scale = _scaled_perturbation(lower, upper)
    seeds = [seed0 + rng.normal(0, scale) for _ in range(n_starts)]
    result = fit_model_multistart(
        model_name="SpinEcho",
        model_func=spin_echo_model,
        x=x,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        selection_note="fallback to stretched-exponential decay model",
    )
    return _semantic_result(result)
