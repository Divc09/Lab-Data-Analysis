from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Mapping

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .fit_engine import (
    FitResult,
    UnderdeterminedFitError,
    fit_model_multistart,
    reduced_chi2,
    sem_to_weights,
    parameter_controls,
    current_parameter_controls,
)
from .models import (
    gaussian_peaks,
    time_trace_decay,
    build_custom_expression_model,
    odmr_lorentzian,
    odmr_multi_lorentzian,
    odmr_multi_lorentzian_jac,
    rabi_adaptive_pulse_area_model,
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
from .rabi_utils import extract_rabi_dead_time_ns, fit_rabi_envelope

if TYPE_CHECKING:
    from .io_mat import ExperimentTrace


LockMap = Mapping[str, tuple[float, bool, float | None, float | None]]


@dataclass(frozen=True)
class FitWorkflowConfig:
    multistart: int = 12
    gaussian_peak_count: int = 1
    rabi_use_sem: bool = True
    sem_weights: bool = False  # SEM weighting for non-Rabi models (Rabi uses rabi_use_sem)
    rabi_pi_lock: bool = False
    rabi_pi_ns: float = 200.0
    rabi_pi2_ns: float = 100.0
    rabi_dead_time_ns: float = 0.0
    rabi_mode: str = "Adaptive pulse-area (recommended)"
    odmr_multipeak: bool = False
    odmr_peak_count: int = 2
    odmr_auto_peak: bool = True
    odmr_polarity: str = "Auto"
    odmr_peak_ranges: tuple[tuple[float, float], ...] = ()
    t1_prep_mode: str = "Auto"
    t1_model_family: str = "Stretched exponential"
    custom_spec: dict | None = None
    custom_expr: str = ""
    t1_normalization: tuple[float, float] | None = None
    parameter_model: str | None = None


def _fit_time_trace_decay(x, y, n_starts, locks=None, weights=None) -> FitResult:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) <= 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Exponential time-trace fitting needs at least four finite samples.")
    spacing = np.diff(np.unique(x))
    span, variation = float(np.ptp(x)), float(np.ptp(y))
    if not spacing.size or variation <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))):
        raise ValueError("Time-trace fitting needs distinct timestamps and a varying observable.")
    names = ["background", "amplitude", "tau_s"]
    lower = [float(np.min(y) - 5 * variation), -10 * variation, float(np.min(spacing)) / 10]
    upper = [float(np.max(y) + 5 * variation), 10 * variation, 100 * span]
    seeds = []
    for tau in np.geomspace(max(lower[2] * 2, span / 20), 5 * span, max(1, n_starts)):
        design = np.column_stack([np.ones_like(x), np.exp(-x / tau)])
        background, amplitude = np.linalg.lstsq(design, y, rcond=None)[0]
        seed = np.array([background, amplitude, tau])
        _apply_locks(names, lower, upper, seed, locks)
        seeds.append(seed)
    result = fit_model_multistart(
        model_name="TimeTraceDecay",
        model_func=time_trace_decay,
        x=x,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        x_scale="jac",
        weights=weights,
        selection_note="Descriptive exponential drift/bleaching; tau_s is in seconds, not a spin lifetime",
    )
    result.extras["parameter_units"] = {"tau_s": "s"}
    return result


def _fit_gaussian_peaks(x, y, count, n_starts, locks=None, weights=None) -> FitResult:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    count = int(count)
    if not 1 <= count <= 8:
        raise ValueError("Choose between 1 and 8 Gaussian peaks.")
    if len(x) <= 1 + 3 * count or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("The fit needs more finite samples than fitted parameters (1 + 3 per peak).")
    order = np.argsort(x)
    sx, sy = x[order], y[order]
    spacing = np.diff(np.unique(sx))
    if not spacing.size:
        raise ValueError("Peak fitting needs distinct scan positions.")
    dx, span = float(np.median(spacing)), float(np.ptp(sx))
    amplitude = float(np.ptp(sy))
    if amplitude <= np.finfo(float).eps:
        raise ValueError("The scan has no variation to fit.")
    baseline = float(np.percentile(sy, 10))
    peaks, properties = find_peaks(sy, prominence=0.03 * amplitude, distance=max(1, len(x) // 40))
    ranked = peaks[np.argsort(properties["prominences"])[::-1]][:count].tolist()
    # Additional starts can separate overlapping peaks with only one local maximum.
    for index in np.argsort(sy)[::-1]:
        if len(ranked) >= count:
            break
        if all(abs(sx[index] - sx[other]) >= span / (3 * count) for other in ranked):
            ranked.append(int(index))
    if len(ranked) < count:
        raise ValueError("The selected range cannot resolve the requested number of peaks.")
    names, seed = ["background"], [baseline]
    lower, upper = [float(np.min(sy) - amplitude)], [float(np.max(sy) + amplitude)]
    min_sigma = max(float(np.min(spacing)) / 4, span * 1e-8)
    for i, index in enumerate(sorted(ranked, key=lambda k: sx[k]), 1):
        sigma = span / (6 * count)
        if index in peaks:
            _, _, left, right = peak_widths(sy, [index], rel_height=0.5)
            sigma = float(np.diff(np.interp([left[0], right[0]], np.arange(len(sx)), sx))[0]) / 2.354820045
        names += [f"amplitude_{i}", f"center_{i}", f"sigma_{i}"]
        seed += [max(float(sy[index] - baseline), amplitude * 0.1), float(sx[index]), max(sigma, dx)]
        lower += [0.0, float(sx[0]), min_sigma]
        upper += [5 * amplitude, float(sx[-1]), span]
    seed = np.asarray(seed)
    _apply_locks(names, lower, upper, seed, locks)
    rng = np.random.default_rng(42)
    seeds = [seed.copy()]
    scales = np.array([amplitude * 0.05] + [amplitude * 0.1, span / (12 * count), span / (20 * count)] * count)
    seeds += [seed + rng.normal(size=len(seed)) * scales for _ in range(max(1, n_starts) - 1)]
    result = fit_model_multistart(
        model_name="GaussianPeaks",
        model_func=gaussian_peaks,
        x=x,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        x_scale="jac",
        weights=weights,
        selection_note=f"{count} Gaussian peak(s) on a constant background; centers and widths in scan axis units",
    )
    result.extras["peak_fwhm"] = {f"peak_{i}": float(2.354820045 * result.params[3 * i]) for i in range(1, count + 1)}
    return result


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
        if value is None or not np.isfinite(value):
            continue
        # a supplied value is the starting seed whether or not it is locked
        seed[i] = value
        if is_locked:
            lower[i] = value
            upper[i] = value


def _spin_echo_seed_and_bounds(exp_type: str, tspan: float) -> tuple[list[float], list[float], np.ndarray]:
    lower = [-0.2, -0.2, 1.0, 0.5]
    if exp_type == "DynamicDecoupling":
        upper = [0.2, 0.2, max(200000.0, 8.0 * tspan), 5.0]
        seed0 = np.array([0.0, 0.0, max(1000.0, tspan / 3.0), 1.5])
    else:
        upper = [0.2, 0.2, 20000.0, 5.0]
        seed0 = np.array([0.0, 0.0, max(100.0, tspan / 3.0), 1.5])
    return lower, upper, seed0


def model_candidate_summary(result: FitResult, role: str = "", chosen: bool = False) -> dict[str, object]:
    """One row of an Auto model comparison: the scores the selection rules look at."""
    nfree = int(sum(result.free_mask)) if result.free_mask is not None else len(result.params)
    return {
        "model": result.model_name,
        "role": role,
        "r2": float(result.r2),
        "bic": float(result.bic),
        "rmse": float(result.rmse),
        "free_params": nfree,
        "bound_hits": int(sum(result.bound_hits)),
        "converged": bool(result.success),
        "chosen": bool(chosen),
    }


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
    norm_extras: dict | None = None,
    normalization: tuple[float, float] | None = None,
) -> tuple[np.ndarray, str]:
    mode = prep_mode.lower().strip()
    if mode.startswith("direct decay"):
        # Explicit form of Auto's falling-trace branch (used to keep per-iteration fits on one model).
        return y, "direct_decay"
    auto = mode == "auto"
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
    use_normalized = mode.startswith("normalize") or (auto and corr > 0.25)
    if not use_normalized:
        # A falling T1 trace (e.g. MW pi-referenced T1 contrast) is fitted directly with the decay model.
        if auto and corr < -0.25:
            return y, "direct_decay"
        return y, "direct"
    base = y.copy()
    source = "y"
    # y is the selected target: processing/baseline/smoothing must never be bypassed.
    lo, span = normalization or (float(np.min(base)), max(float(np.ptp(base)), 1e-9))
    if norm_extras is not None:
        norm_extras.update(t1_norm_lo=lo, t1_norm_span=span, t1_norm_source=source)
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


def aligned_indices(trace: "ExperimentTrace | None", x: np.ndarray) -> np.ndarray | None:
    """Indices into ``trace.x_ns`` matching each ``x`` by exact abscissa value (handles range
    cuts and exclusions).  None if x was re-binned/transformed or a point is not matched uniquely."""
    if trace is None:
        return None
    tx = np.asarray(trace.x_ns, dtype=float)
    x = np.asarray(x, dtype=float)
    if tx.size == 0 or x.size == 0:
        return None
    order = np.argsort(tx)
    if tx.size > 1:
        pos = np.clip(np.searchsorted(tx[order], x), 1, tx.size - 1)
        cand = np.stack([order[pos - 1], order[pos]])
        best = cand[np.argmin(np.abs(tx[cand] - x), axis=0), np.arange(x.size)]
    else:
        best = np.zeros(x.size, dtype=int)
    if not np.allclose(tx[best], x, rtol=1e-9, atol=1e-9 * max(1.0, float(np.max(np.abs(x))))):
        return None
    if len(np.unique(best)) != len(best):
        return None
    return best


def aligned_sem(trace: "ExperimentTrace | None", x: np.ndarray) -> np.ndarray | None:
    """Per-point SEM of ``trace`` matched to ``x`` (see ``aligned_indices``); None if unavailable."""
    if trace is None or trace.y_sem is None:
        return None
    sem = np.asarray(trace.y_sem, dtype=float)
    if sem.shape != np.asarray(trace.x_ns).shape:
        return None
    idx = aligned_indices(trace, x)
    return None if idx is None else sem[idx]


def _attach_sem_chi2(
    result: FitResult, y: np.ndarray, sem: np.ndarray, n_locked: int, t1_span: float, *, weighted: bool
) -> FitResult:
    y_ref = np.asarray(result.extras.get("plot_y", y), dtype=float)
    sem_eff = np.asarray(sem, dtype=float)
    if result.extras.get("t1_prep_mode") == "normalized_1_minus_data":
        sem_eff = sem_eff / max(t1_span, 1e-9)  # T1 series is 1 - (base-lo)/span
    if len(y_ref) != len(result.y_fit) or len(sem_eff) != len(y_ref):
        return result
    chi2 = reduced_chi2(y_ref, result.y_fit, sem_eff, len(result.params) - n_locked)
    if np.isfinite(chi2):
        result.extras["reduced_chi2"] = chi2
        tag = "SEM-weighted fit" if weighted else "SEM available"
        result.selection_note = f"{result.selection_note}; {tag}, reduced chi2={chi2:.2f} (absolute 1/sem^2)".lstrip(
            "; "
        )
    return result


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
    # Pointwise SEM from only a few iterations is itself noisy.  Unlimited
    # inverse-variance weights let one accidentally small SEM dominate the
    # waveform.  Winsorize and shrink the precision toward equal weighting;
    # the shrinkage vanishes gradually as complete iterations accumulate.
    weights = np.clip(weights, 0.25, 4.0)
    n_iter = float((trace.metadata or {}).get("complete_iterations", 0.0) or 0.0)
    shrink = n_iter / (n_iter + 8.0) if n_iter > 0.0 else 0.5
    weights = 1.0 + shrink * (weights - 1.0)
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
    # The transient begins at the physical start of every programmed pulse,
    # not at the minimum value chosen for a scan.  Using x-min(x) biases the
    # fitted turn-on time whenever a Rabi scan starts above zero.
    x_fit = np.asarray(x, dtype=float)
    c0 = float(np.median(y))
    a0 = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    f0 = _rabi_frequency_seed(x, y)
    span = max(float(np.ptp(x)), 1.0)
    names = ["y0", "A", "T", "beta", "f", "tau_ramp", "phi"]
    lower = [-1.0, 0.0, 1.0, 0.1, 1.0 / 1000.0, 0.01, -10.0 * np.pi]
    upper = [1.0, 1.0, 1e6, 4.0, 1.0 / 3.0, max(200.0, span), 10.0 * np.pi]
    seed0 = np.array([c0, max(a0, 1e-4), max(span, 100.0), 1.0, f0, min(15.0, 0.1 * span), 0.0])
    scale = np.array([0.01, 0.25 * max(a0, 1e-4), 0.35 * span, 0.25, 0.08 * max(f0, 1e-4), 0.08 * span, 0.7])
    seeds = []
    for i in range(max(8, n_starts)):
        seed = seed0.copy()
        seed[2] = [max(40.0, 0.5 * span), max(80.0, span), max(120.0, 2.0 * span)][i % 3]
        seed[3] = [0.7, 1.0, 1.6][(i // 3) % 3]
        seed[5] = [max(0.5, 0.03 * span), max(1.0, 0.08 * span), max(2.0, 0.15 * span)][i % 3]
        seeds.append(seed + rng.normal(0, scale))
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
        x_scale="jac",
        selection_note="selected phase-ramp Rabi model",
    )
    result.extras.update({"rabi_model_family": "phase_ramp"})
    return result


def _fit_rabi_adaptive(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    n_starts: int,
    *,
    metadata: Mapping[str, object] | None = None,
    chirped_reference: FitResult | None = None,
) -> FitResult:
    """Fit a delay-anchored smooth pulse-area model.

    This is the calibration model for scans whose apparent Rabi rate changes
    smoothly with programmed duration.  The polynomial is applied to pulse
    area (rotation phase), so its coefficients must not be interpreted as a
    microwave-carrier chirp.
    """
    rng = np.random.default_rng(97)
    x_fit = np.asarray(x, dtype=float)
    span = max(float(np.ptp(x_fit)), 1.0)
    dx = max(float(np.median(np.diff(np.sort(x_fit)))) if len(x_fit) > 1 else 1.0, 1e-6)
    c0 = float(np.median(y))
    a0 = max(float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5))), 1e-5)
    f_seed = _rabi_frequency_seed(x_fit, y)
    alpha_seed = 0.0
    if chirped_reference is not None:
        cp = {name: float(value) for name, value in zip(chirped_reference.param_names, chirped_reference.params)}
        f_seed = float(cp.get("f", f_seed))
        alpha_seed = float(cp.get("alpha", 0.0))
    first_peak = _first_peak_from_series(x_fit, y)
    saved_delay = extract_rabi_dead_time_ns(metadata)
    derived_delay = first_peak - 0.5 / max(f_seed, 1e-9)
    delay_seed = saved_delay if saved_delay > 0.0 else derived_delay
    delay_upper = max(first_peak, float(np.min(x_fit)) + 1.0 / max(f_seed, 1e-9), 30.0)
    delay_seed = float(np.clip(delay_seed, 0.0, delay_upper))

    f_lo = max(5e-4, 0.55 * f_seed)
    f_hi = min(0.333, max(1.65 * f_seed, f_lo + 5e-4))
    chirp1_bound = max(5e-6, 1.5 * max(f_seed, 1e-4) / span)
    chirp2_bound = max(5e-9, 2.0 * max(f_seed, 1e-4) / (span * span))
    amp_bound = max(0.2, 6.0 * a0)
    names = ["y0", "A", "T", "beta", "f0", "chirp1", "chirp2", "delay"]
    lower = [-1.0, 0.0, dx, 0.1, f_lo, -chirp1_bound, -chirp2_bound, 0.0]
    upper = [1.0, amp_bound, max(1e6, 20.0 * span), 4.0, f_hi, chirp1_bound, chirp2_bound, delay_upper]

    base = np.array([c0, a0, max(span, 100.0), 1.0, f_seed, alpha_seed, 0.0, delay_seed], dtype=float)
    structured: list[np.ndarray] = []
    count = max(14, int(n_starts))
    for i in range(count):
        seed = base.copy()
        seed[2] = [max(40.0, 0.6 * span), max(80.0, span), max(120.0, 2.0 * span)][i % 3]
        seed[3] = [0.65, 1.0, 1.5][(i // 3) % 3]
        seed[4] = np.clip(f_seed * [0.92, 1.0, 1.08][i % 3], f_lo, f_hi)
        seed[5] = np.clip(
            [0.0, alpha_seed, 0.5 * f_seed / span, -0.25 * f_seed / span][i % 4], -chirp1_bound, chirp1_bound
        )
        seed[6] = np.clip(
            [0.0, -0.22 * f_seed / (span * span), 0.15 * f_seed / (span * span)][i % 3], -chirp2_bound, chirp2_bound
        )
        seed[7] = np.clip([delay_seed, max(0.0, derived_delay), max(0.0, saved_delay)][i % 3], 0.0, delay_upper)
        jitter = np.array(
            [
                0.004,
                0.15 * a0,
                0.18 * span,
                0.15,
                0.025 * f_seed,
                0.08 * chirp1_bound,
                0.08 * chirp2_bound,
                max(0.5, 0.04 * delay_upper),
            ]
        )
        structured.append(seed + rng.normal(0.0, jitter))

    result = fit_model_multistart(
        model_name="RabiAdaptive",
        model_func=rabi_adaptive_pulse_area_model,
        x=x_fit,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=structured,
        weights=weights,
        loss="linear",
        max_nfev=10000,
        x_scale="jac",
        selection_note="selected adaptive pulse-area Rabi model",
    )
    result.extras.update(
        {
            "rabi_model_family": "adaptive_pulse_area",
            "rabi_delay_seed_ns": float(delay_seed),
            "rabi_saved_ramp_time_ns": float(saved_delay),
            "pulse_area_interpretation": "chirp coefficients describe effective rotation rate versus programmed duration, not microwave carrier chirp",
        }
    )
    return result


def _rabi_adaptive_pathology(result: FitResult, x: np.ndarray) -> str:
    p = {name: float(value) for name, value in zip(result.param_names, result.params)}
    required = {"f0", "chirp1", "chirp2", "delay"}
    if not required <= set(p):
        return "adaptive candidate rejected: missing pulse-area parameters"
    u_max = max(float(np.max(x) - p["delay"]), 0.0)
    u = np.linspace(0.0, u_max, 256)
    freq = p["f0"] + p["chirp1"] * u + p["chirp2"] * u * u
    if not np.all(np.isfinite(freq)) or float(np.min(freq)) <= 0.0:
        return "adaptive candidate rejected: non-positive effective Rabi rate"
    ratio = float(np.max(freq) / max(np.min(freq), 1e-12))
    if ratio > 3.0:
        return "adaptive candidate rejected: implausibly large pulse-area rate variation"
    if sum(bool(v) for v in result.bound_hits) >= 3:
        return "adaptive candidate rejected: multiple parameters hit bounds"
    return ""


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


def _optional_candidate(fit):
    try:
        return fit()
    except UnderdeterminedFitError:
        return None


def _choose_rabi_model(
    *,
    x: np.ndarray,
    y: np.ndarray,
    phase_ramp: FitResult,
    constant: FitResult,
    chirped: FitResult | None,
    adaptive: FitResult | None = None,
    long_damped: FitResult | None = None,
) -> FitResult:
    if adaptive is not None:
        adaptive_reason = _rabi_adaptive_pathology(adaptive, x)
        alternatives = (
            [phase_ramp, constant]
            + ([chirped] if chirped is not None else [])
            + ([long_damped] if long_damped is not None else [])
        )
        best_alt = min(alternatives, key=lambda item: item.bic)
        if not adaptive_reason and (adaptive.bic <= best_alt.bic - 4.0 or adaptive.r2 >= best_alt.r2 + 0.015):
            adaptive.extras["rabi_reference_model"] = best_alt.model_name
            adaptive.extras["rabi_reference_bic"] = float(best_alt.bic)
            return _with_selection(adaptive, "selected adaptive pulse-area Rabi model by BIC and residual improvement")
        if adaptive_reason:
            adaptive.extras["rabi_rejection_reason"] = adaptive_reason

    if long_damped is not None and bool(long_damped.extras.get("rabi_long_scan", False)):
        long_features = dict(long_damped.extras.get("rabi_long_features", {}))
        alternatives = [phase_ramp, constant] + ([chirped] if chirped is not None else [])
        best_alt = min(alternatives, key=lambda item: item.bic)
        if long_damped.bic <= best_alt.bic - 4.0 and long_damped.rmse < best_alt.rmse:
            note = (
                "auto-selected long damped Rabi for decayed long scan"
                f" ({long_features.get('cycle_count', 0.0):.1f} cycles, decay ratio {long_features.get('decay_ratio', 0.0):.2f})"
            )
            long_damped.extras["rabi_long_phase_reference_r2"] = float(phase_ramp.r2)
            long_damped.extras["rabi_long_constant_reference_r2"] = float(constant.r2)
            return _with_selection(long_damped, note)

    data_first_peak = _first_peak_from_series(x, y)
    ramp_first_peak = _first_peak_from_series(x, phase_ramp.y_fit)
    ramp_peak_error = abs(ramp_first_peak - data_first_peak)
    ramp_rmse_improvement = phase_ramp.rmse / max(constant.rmse, 1e-12)

    if ramp_peak_error <= 2.5 and ramp_rmse_improvement <= 0.85:
        return _with_selection(phase_ramp, "selected phase-ramp Rabi model")

    if chirped is not None:
        chirped_peak_error = abs(_first_peak_from_series(x, chirped.y_fit) - data_first_peak)
        if chirped.rmse <= phase_ramp.rmse * 0.75 or (
            chirped_peak_error <= 2.5 and chirped.r2 >= max(0.95, phase_ramp.r2 + 0.03)
        ):
            chirped.extras["rabi_physical_reference_params"] = {
                name: float(value) for name, value in zip(phase_ramp.param_names, phase_ramp.params)
            }
            chirped.extras["rabi_physical_reference_model"] = "RabiPhaseRamp"
            return _with_selection(
                chirped, "selected chirped Rabi fallback because phase-ramp still missed startup alignment"
            )

    if phase_ramp.rmse < constant.rmse:
        return _with_selection(
            phase_ramp, f"selected phase-ramp Rabi model; first-peak mismatch remains {ramp_peak_error:.2f} ns"
        )
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
    seed0 = np.array(
        [y0, 0.0, 0.55 * yamp, 0.35 * yamp, freq_pairs[0][0], freq_pairs[0][1], 0.0, 0.5, max(200.0, tspan)]
    )
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
    weights: np.ndarray | None = None,
) -> FitResult:
    norm_extras = {}
    y_fit, prep_used = _prepare_t1_series(
        x, y, prep_mode=config.t1_prep_mode, trace=trace, norm_extras=norm_extras, normalization=config.t1_normalization
    )
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
    elif prep_used == "direct_decay":
        base_name = "T1DecayStretch" if use_stretch else "T1DecayExp"
        model = t1_decay_stretched_model if use_stretch else t1_decay_exponential_model
        seed0 = np.array([y0, amp, max(100.0, tspan / 4.0)] + ([1.2] if use_stretch else []), dtype=float)
    else:
        base_name = "T1Stretch" if use_stretch else "T1Exp"
        model = t1_rise_stretched_model if use_stretch else t1_rise_exponential_model
        seed0 = np.array([y0, amp, max(100.0, tspan / 4.0)] + ([1.2] if use_stretch else []), dtype=float)

    names = ["y0", "A", "T1"] + (["n"] if use_stretch else [])
    lower = [min(y0, y1) - amp, 0.0, 1.0] + ([0.4] if use_stretch else [])
    upper = [max(y0, y1) + amp, max(4.0 * amp, 1.0), max(200000.0, 8.0 * tspan)] + ([4.5] if use_stretch else [])
    _apply_locks(names, lower, upper, seed0, locks)
    scale = _scaled_perturbation(lower, upper)
    seeds = [seed0 + rng.normal(0, scale) for _ in range(max(1, int(config.multistart)))]
    result = fit_model_multistart(
        model_name=base_name,
        model_func=model,
        x=x,
        y=y_fit,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
        weights=weights,
        selection_note=f"selected {prep_used} T1 with {'stretched' if use_stretch else 'regular'} exponential model",
    )
    plot_label = "Normalized 1 - data [a.u.]" if prep_used == "normalized_1_minus_data" else "T1 signal [a.u.]"
    result.extras.update(
        {
            **norm_extras,
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


def _odmr_controls(controls, y_scale):
    return {
        name: tuple(
            v / y_scale if i != 1 and v is not None and (name == "y0" or name.startswith("C")) else v
            for i, v in enumerate(values)
        )
        for name, values in (controls or {}).items()
    }


def _fit_odmr_single(
    x: np.ndarray,
    y: np.ndarray,
    n_starts: int,
    polarity: str,
    locks: LockMap | None = None,
    center_range: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
) -> FitResult:
    rng = np.random.default_rng(29)
    y_scale = max(float(np.percentile(y, 95) - np.percentile(y, 5)), 1e-9)
    y_fit = y / y_scale
    names, lower, upper, seed0 = _single_odmr_bounds(x, y_fit, polarity, center_range=center_range)
    controls = _odmr_controls(locks or current_parameter_controls("ODMR"), y_scale)
    _apply_locks(names, lower, upper, seed0, controls)
    scale = _scaled_perturbation(lower, upper)
    seeds = [seed0 + rng.normal(0, scale) for _ in range(n_starts)]
    with parameter_controls(controls, model_name="ODMR"):
        result = fit_model_multistart(
            model_name="ODMR",
            model_func=odmr_lorentzian,
            x=x,
            y=y_fit,
            param_names=names,
            lower=lower,
            upper=upper,
            seeds=seeds,
            weights=weights,
        )
    return _rescale_odmr_result(result, y_scale, extras={"odmr_polarity": polarity})


# Residual evaluations per multi-peak start (the analytic Jacobian is not counted). On ODMR_20260324_023911 every
# start that reached the best cost needed <= ~2000; starts at the cap had drifted to worse solutions.
ODMR_MULTI_MAX_NFEV = 2000


def _fit_odmr_multi(
    x: np.ndarray,
    y: np.ndarray,
    n_starts: int,
    peak_indices: np.ndarray,
    max_peaks: int,
    polarity: str,
    center_ranges: list[tuple[float, float]] | None = None,
    weights: np.ndarray | None = None,
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
    with parameter_controls(_odmr_controls(current_parameter_controls("ODMRMulti"), y_scale), model_name="ODMRMulti"):
        result = fit_model_multistart(
            model_name="ODMRMulti",
            model_func=odmr_multi_lorentzian,
            jac_func=odmr_multi_lorentzian_jac,
            x=x,
            y=y_fit,
            param_names=names,
            lower=lower,
            upper=upper,
            seeds=seeds,
            weights=weights,
            # Converging starts need a few hundred evaluations; with too many peaks for the data, others drift
            # between near-equivalent solutions up to the default 20000 (about 10 s each) without a better cost.
            max_nfev=ODMR_MULTI_MAX_NFEV,
        )
    return _rescale_odmr_result(result, y_scale, extras={"odmr_polarity": polarity})


def _rescale_odmr_result(result: FitResult, y_scale: float, extras: dict[str, object] | None = None) -> FitResult:
    params = result.params.copy()
    errors = result.errors.copy()
    factors = np.ones(len(params))
    for idx, name in enumerate(result.param_names):
        if name == "y0" or name.startswith("C"):
            params[idx] *= y_scale
            errors[idx] *= y_scale
            factors[idx] = y_scale
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
        covariance=result.covariance * np.outer(factors, factors) if result.covariance is not None else None,
        free_mask=result.free_mask,
        uncertainty=result.uncertainty,
        model_func=result.model_func,
    )


def _top_up_odmr_peaks(x: np.ndarray, y: np.ndarray, polarity: str, peak_indices: np.ndarray, count: int) -> np.ndarray:
    """The detected peaks, plus the next-strongest separated extrema until there are ``count`` seeds."""
    chosen = [int(i) for i in np.asarray(peak_indices, dtype=int)[:count]]
    if len(chosen) >= count:
        return np.asarray(chosen, dtype=int)
    signal = _odmr_detection_signal(y, polarity)
    span = float(np.ptp(x)) or 1.0
    gap = span / (4.0 * count)  # keep seeds apart so two Lorentzians do not start on one resonance
    for i in np.argsort(signal)[::-1]:
        if len(chosen) >= count:
            break
        if all(abs(float(x[i]) - float(x[j])) >= gap for j in chosen):
            chosen.append(int(i))
    return np.asarray(chosen, dtype=int)


def _fit_odmr_requested(x, y, n_starts, polarity, peak_indices, ranges, count, locks, single_range, weights) -> FitResult:
    count = int(min(max(1, count), 4))
    detected = [float(v) for v in np.asarray(x[np.asarray(peak_indices, dtype=int)], dtype=float)[:4]] if len(peak_indices) else []
    if count == 1:
        result = _fit_odmr_single(x, y, n_starts, polarity, locks=locks, center_range=single_range, weights=weights)
        note = "fitted 1 peak as requested"
        if len(peak_indices) > 1:
            note += f" ({len(peak_indices)} candidate resonances were detected)"
        return _with_selection(result, note, {"detected_peaks": detected, "requested_peaks": 1})
    seeds = _top_up_odmr_peaks(x, y, polarity, peak_indices, count)
    if len(seeds) < count:
        raise ValueError(f"Cannot place {count} separated peaks in this scan; choose fewer peaks or Auto.")
    result = _fit_odmr_multi(x, y, n_starts, seeds, max_peaks=count, polarity=polarity,
                             center_ranges=ranges or None, weights=weights)
    note = f"fitted {count} peaks as requested"
    if len(peak_indices) < count:
        found, added = len(peak_indices), count - len(peak_indices)
        note += (f"; {found} {'was' if found == 1 else 'were'} detected, so {added} started at the next-strongest "
                 f"{polarity}{'' if added == 1 else 's'}")
    return _with_selection(result, note, {"detected_peaks": detected, "requested_peaks": count})


def _select_odmr_candidate(
    single: FitResult, multi: FitResult | None, peak_indices: np.ndarray, x: np.ndarray
) -> FitResult:
    detected = [float(v) for v in np.asarray(x[peak_indices], dtype=float)[:4]] if len(peak_indices) else []
    if multi is None:
        return _with_selection(single, "selected single-peak ODMR candidate", {"detected_peaks": detected})
    if multi.bic <= single.bic - 4.0 and multi.r2 >= single.r2 + 0.05:
        return _with_selection(
            multi,
            "selected multi-peak ODMR candidate by lower BIC and stronger explained variance",
            {"detected_peaks": detected},
        )
    if sum(multi.bound_hits) > sum(single.bound_hits):
        return _with_selection(
            single,
            "selected single-peak ODMR candidate because multi-peak fit hit more bounds",
            {"detected_peaks": detected},
        )
    if multi.bic <= single.bic - 4.0 and multi.r2 >= single.r2 - 0.02:
        return _with_selection(multi, "selected multi-peak ODMR candidate by lower BIC", {"detected_peaks": detected})
    return _with_selection(single, "selected single-peak ODMR candidate by parsimony", {"detected_peaks": detected})


def _fit_non_ramsey_impl(
    exp_type: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    config: FitWorkflowConfig | None = None,
    locks: LockMap | None = None,
    custom_spec: dict | None = None,
    trace_experiment_type: str | None = None,
    trace: "ExperimentTrace | None" = None,
    sem: np.ndarray | None = None,
) -> FitResult:
    cfg = config or FitWorkflowConfig()
    w_sem = sem_to_weights(sem) if cfg.sem_weights and sem is not None else None
    if (
        trace is not None
        and trace.experiment_type == "TimeTrace"
        and exp_type not in {"TimeTrace", "TimeTraceDecay", "CustomModel"}
    ):
        raise ValueError("Elapsed-time data require TimeTraceDecay or a custom model with x in seconds.")
    if exp_type in {"TimeTrace", "TimeTraceDecay"}:
        return _fit_time_trace_decay(x, y, cfg.multistart, locks, w_sem)
    if exp_type == "GaussianPeaks":
        return _fit_gaussian_peaks(x, y, cfg.gaussian_peak_count, cfg.multistart, locks, w_sem)
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

    n_starts = max(1, int(cfg.multistart))
    tspan = float(np.max(x) - np.min(x))
    yamp = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    y0 = float(np.median(y))

    if exp_type == "Rabi":
        weights = _rabi_fit_weights(trace, x, enabled=bool(cfg.rabi_use_sem))
        mode = cfg.rabi_mode.lower().strip()
        if mode.startswith("long"):
            return _semantic_result(
                _attach_rabi_envelope(
                    _with_selection(_fit_rabi_long_damped(x, y, weights, n_starts), "selected long damped Rabi model"),
                    x,
                    y,
                )
            )
        if mode.startswith("constant-frequency"):
            constant = _fit_rabi_constant_reference(x, y, weights, n_starts)
            return _semantic_result(
                _attach_rabi_envelope(
                    _with_selection(constant, "selected constant-frequency damped cosine Rabi model"), x, y
                )
            )
        if mode.startswith("chirped"):
            chirped = _fit_rabi_chirp(x, y, weights, n_starts)
            return _semantic_result(
                _attach_rabi_envelope(_with_selection(chirped, "selected chirped Rabi model"), x, y)
            )
        if mode.startswith("adaptive"):
            chirped = _fit_rabi_chirp(x, y, weights, n_starts)
            adaptive = _fit_rabi_adaptive(
                x,
                y,
                weights,
                n_starts,
                metadata=(trace.metadata if trace is not None else None),
                chirped_reference=chirped,
            )
            return _semantic_result(
                _attach_rabi_envelope(_with_selection(adaptive, "selected adaptive pulse-area Rabi model"), x, y)
            )
        phase_ramp = _fit_rabi_phase_ramp(x, y, weights, n_starts)
        constant = _fit_rabi_constant_reference(x, y, weights, n_starts)
        # Optional richer candidates are skipped when the data cannot identify them; the phase-ramp and
        # constant references above stay required.
        chirped = _optional_candidate(lambda: _fit_rabi_chirp(x, y, weights, n_starts))
        adaptive = _optional_candidate(lambda: _fit_rabi_adaptive(
            x,
            y,
            weights,
            n_starts,
            metadata=(trace.metadata if trace is not None else None),
            chirped_reference=chirped,
        ))
        features = _rabi_long_scan_features(x, y)
        long_damped = (
            _optional_candidate(lambda: _fit_rabi_long_damped(x, y, weights, n_starts))
            if bool(features["is_long_decayed"]) else None
        )
        if mode.startswith("phase-ramp"):
            return _semantic_result(
                _attach_rabi_envelope(_with_selection(phase_ramp, "selected phase-ramp Rabi model"), x, y)
            )
        chosen = _choose_rabi_model(
            x=x,
            y=y,
            phase_ramp=phase_ramp,
            constant=constant,
            chirped=chirped,
            adaptive=adaptive,
            long_damped=long_damped,
        )
        # Keep the whole comparison so the UI can show why Auto chose this model.
        considered = {"phase-ramp": phase_ramp, "constant": constant, "chirped": chirped, "adaptive": adaptive}
        if bool(features["is_long_decayed"]):
            considered["long damped"] = long_damped
        chosen.extras["model_candidates"] = [
            model_candidate_summary(fit, role, chosen=fit is chosen)
            if fit is not None
            else {"model": role, "skipped": "the data cannot identify this model's parameters"}
            for role, fit in considered.items()
        ]
        return _semantic_result(_attach_rabi_envelope(chosen, x, y))

    if exp_type == "T1":
        return _semantic_result(_fit_t1(x, y, cfg, trace=trace, locks=locks, weights=w_sem))

    if exp_type in ("SpinEcho", "DynamicDecoupling"):
        names = ["y0", "A", "T2", "n"]
        lower, upper, seed0 = _spin_echo_seed_and_bounds(trace_experiment_type or exp_type, tspan)
        seed0[0] = y0
        seed0[1] = yamp
        _apply_locks(names, lower, upper, seed0, locks)
        rng = np.random.default_rng(33)
        scale = _scaled_perturbation(lower, upper)
        seeds = [seed0 + rng.normal(0, scale) for _ in range(n_starts)]
        return _semantic_result(
            fit_model_multistart(
                model_name="SpinEcho",
                model_func=spin_echo_model,
                x=x,
                y=y,
                param_names=names,
                lower=lower,
                upper=upper,
                seeds=seeds,
                weights=w_sem,
                selection_note="selected stretched-exponential decay model",
            )
        )

    if exp_type == "ODMR":
        polarities = (
            ["peak", "dip"] if cfg.odmr_polarity.lower() == "auto" else [_detect_odmr_polarity(x, y, cfg.odmr_polarity)]
        )
        candidates: list[FitResult] = []
        for polarity in polarities:
            peak_indices = _detect_odmr_peaks(x, y, polarity)
            peak_indices, constrained_ranges = _odmr_peaks_in_ranges(x, y, polarity, peak_indices, cfg.odmr_peak_ranges)
            single_range: tuple[float, float] | None = None
            if constrained_ranges:
                strengths = _odmr_detection_signal(y, polarity)[peak_indices]
                single_range = constrained_ranges[int(np.argmax(strengths))]
            if not cfg.odmr_auto_peak:
                # An explicit peak count is honoured exactly; the note says when peaks had to be seeded.
                candidates.append(_fit_odmr_requested(x, y, n_starts, polarity, peak_indices, constrained_ranges,
                                                      int(cfg.odmr_peak_count), locks, single_range, w_sem))
                continue
            single = _fit_odmr_single(x, y, n_starts, polarity, locks=locks, center_range=single_range, weights=w_sem)
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
                        weights=w_sem,
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
        return _semantic_result(
            fit_model_multistart(
                model_name=str(spec.get("name", "CustomModel")),
                model_func=model,
                x=x,
                y=y,
                param_names=names,
                lower=lower,
                upper=upper,
                seeds=seeds,
                weights=w_sem,
                selection_note="selected custom model",
            )
        )

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
        weights=w_sem,
        selection_note="fallback to stretched-exponential decay model",
    )
    return _semantic_result(result)


def parameter_control_model(exp_type, config, x, y):
    """Resolve the model whose parameter editor supplied these controls.

    Auto selection can evaluate other models, but their parameter names are not a
    contract for transferring controls. Older requests without an editor model
    use the workflow's primary model; new editor requests name it explicitly.
    """
    if config.parameter_model:
        return config.parameter_model
    family = {
        "DEERDuration": "Rabi",
        "DEERDecay": "SpinEcho",
        "DEERFrequency": "ODMR",
        "DynamicDecoupling": "SpinEcho",
    }.get(exp_type, exp_type)
    if family == "Rabi":
        mode = config.rabi_mode.lower().strip()
        for prefix, model in (
            ("long", "RabiLongDamped"),
            ("constant-frequency", "RabiCosine"),
            ("chirped", "RabiChirp"),
            ("adaptive", "RabiAdaptive"),
        ):
            if mode.startswith(prefix):
                return model
        return "RabiPhaseRamp"
    if family == "T1":
        _, prep = _prepare_t1_series(x, y, prep_mode=config.t1_prep_mode, normalization=config.t1_normalization)
        prefix = {"direct": "T1", "direct_decay": "T1Decay", "normalized_1_minus_data": "T1Norm"}[prep]
        return prefix + ("Stretch" if "stretched" in config.t1_model_family.lower() else "Exp")
    if family == "ODMR":
        return "ODMRMulti" if config.odmr_multipeak else "ODMR"
    if family == "CustomModel":
        return str((config.custom_spec or {}).get("name", "CustomModel"))
    return family


def describe_parameter_controls(result, controls, model_name):
    if not controls:
        return result
    applied = result.model_name == model_name
    result.extras["parameter_controls"] = {"model": model_name, "applied": applied}
    if not applied:
        note = (
            f"Parameter controls belong to {model_name}; they were not applied to selected model {result.model_name}."
        )
        result.selection_note = "; ".join(filter(None, (result.selection_note, note)))
    return result


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
    """Fit a non-Ramsey experiment.  With ``config.sem_weights`` (or Rabi SEM weighting) the
    per-point SEM of the mean trace weights the fit and ``extras['reduced_chi2']`` is reported
    (sum((y-fit)/sem)^2/dof, absolute 1/sem^2 weights)."""
    cfg = config or FitWorkflowConfig()
    if custom_spec is not None:
        cfg = replace(cfg, custom_spec=custom_spec)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sem = aligned_sem(trace, x)
    control_model = parameter_control_model(exp_type, cfg, x, y)
    with parameter_controls(locks, model_name=control_model):
        result = _fit_non_ramsey_impl(
            exp_type,
            x,
            y,
            config=cfg,
            locks=None,
            custom_spec=custom_spec,
            trace_experiment_type=trace_experiment_type,
            trace=trace,
            sem=sem,
        )
    describe_parameter_controls(result, locks, control_model)
    if sem is None:
        if cfg.sem_weights and trace is not None:
            result.extras["sem_weights_applied"] = False
        return result
    rabi = result.model_name.startswith("Rabi") or exp_type in {"Rabi", "DEERDuration"}
    weighted = bool(cfg.sem_weights) or (
        rabi and _rabi_fit_weights(trace, x, enabled=bool(cfg.rabi_use_sem)) is not None
    )
    if not weighted:
        return result
    n_locked = (
        len(result.params) - sum(result.free_mask)
        if result.free_mask is not None
        else sum(1 for n in result.param_names if locks and n in locks and locks[n][1])
    )
    base = y
    result.extras["sem_weights_applied"] = True
    span = result.extras.get("t1_norm_span", float(np.ptp(base)))
    return _attach_sem_chi2(result, y, sem, n_locked, float(span), weighted=True)
