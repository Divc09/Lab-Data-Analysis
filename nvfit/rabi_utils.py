from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.signal import find_peaks, hilbert, savgol_filter


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def extract_rabi_dead_time_ns(metadata: Mapping[str, object] | None) -> float:
    if not metadata:
        return 0.0

    candidate_keys = (
        "rabi_dead_time_ns",
        "rf_ramp_time_ns",
        "dead_time_ns",
        "pulse_dead_time_ns",
        "pulse_length_dead_time_ns",
        "rabiDeadTimeNs",
        "rfRampTimeNs",
        "deadTimeNs",
        "pulseDeadTimeNs",
        "pulseLengthDeadTimeNs",
        "rabiDeadTime",
        "RFRampTime",
        "rfRampTime",
        "deadTime",
        "pulseDeadTime",
    )
    for key in candidate_keys:
        if key not in metadata:
            continue
        value = _safe_float(metadata.get(key))
        if value is not None:
            return value
    return 0.0


def rabi_timing_metrics_from_frequency(freq_ghz: float, dead_time_ns: float = 0.0) -> dict[str, float]:
    if not np.isfinite(freq_ghz) or freq_ghz <= 0.0:
        return {}

    dead_time = dead_time_ns if np.isfinite(dead_time_ns) else 0.0
    pi_nominal = 1.0 / (2.0 * freq_ghz)
    pi2_nominal = 1.0 / (4.0 * freq_ghz)
    return {
        "rabi_freq_MHz": float(freq_ghz * 1000.0),
        "rabi_dead_time_ns": float(dead_time),
        "pi_time_ns_nominal": float(pi_nominal),
        "pi_over_2_time_ns_nominal": float(pi2_nominal),
        "pi_time_ns": float(pi_nominal),
        "pi_over_2_time_ns": float(pi2_nominal),
        "programmed_pi_time_ns": float(dead_time + pi_nominal),
        "programmed_pi_over_2_time_ns": float(dead_time + pi2_nominal),
    }


def _rabi_envelope_model(t: np.ndarray, floor: float, amp: float, T2rho: float, beta: float) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    T_safe = max(float(T2rho), 1e-9)
    beta_safe = max(float(beta), 1e-9)
    return float(floor) + float(amp) * np.exp(-((np.abs(t_arr / T_safe)) ** beta_safe))


def _baseline_from_rabi_params(
    x: np.ndarray,
    model_name: str,
    param_names: list[str],
    params: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    p = {name: float(value) for name, value in zip(param_names, params)}
    if model_name in {"RabiAdaptive", "RabiPhaseRamp", "RabiCosine", "RabiChirp", "RabiLongDamped"}:
        intercept = float(p.get("y0", p.get("c", 0.0)))
        slope = 0.0
        mode = "constant"
    else:
        intercept = float(p.get("y0", p.get("c", 0.0)))
        slope = float(p.get("m", 0.0))
        mode = "linear" if abs(slope) > 0.0 else "constant"
    baseline = intercept + slope * np.asarray(x, dtype=float)
    return baseline, {
        "baseline_mode": mode,
        "baseline_intercept": intercept,
        "baseline_slope": slope,
    }


def _envelope_source_from_trace(
    x_rel: np.ndarray,
    centered: np.ndarray,
    params: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, str]:
    amp = np.abs(np.asarray(centered, dtype=float))
    if len(amp) >= 7:
        win = min(len(amp) if len(amp) % 2 == 1 else len(amp) - 1, 11)
        if win >= 7:
            amp = savgol_filter(amp, win, 2)
    dx = float(np.median(np.diff(x_rel))) if len(x_rel) > 1 else 1.0
    freq_candidates = [float(params[k]) for k in ("f0", "f", "f1", "f2") if k in params and float(params[k]) > 0.0]
    if freq_candidates and dx > 0.0:
        period = 1.0 / max(freq_candidates)
        distance = max(1, int(round(0.2 * period / dx)))
    else:
        distance = max(1, int(round(0.03 * max(len(x_rel), 1))))
    prominence = max(1e-10, 0.08 * float(np.nanmax(amp) - np.nanmin(amp)))
    peaks, _ = find_peaks(amp, prominence=prominence, distance=distance)
    if len(peaks) >= 4:
        return x_rel[peaks], amp[peaks], "abs_peak_trace"

    try:
        env = np.abs(hilbert(centered - float(np.nanmedian(centered))))
        if np.all(np.isfinite(env)) and len(env) >= 4:
            return x_rel, env, "hilbert_trace"
    except Exception:
        pass
    return x_rel, amp, "abs_trace"


def fit_rabi_envelope(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str,
    param_names: list[str],
    params: np.ndarray,
    y_fit: np.ndarray | None = None,
) -> dict[str, object]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if len(x_arr) < 6 or int(np.sum(finite)) < 6:
        return {"success": False, "message": "not enough finite points for Rabi envelope fit"}
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    x_rel = x_arr - float(np.min(x_arr))

    p = {name: float(value) for name, value in zip(param_names, params)}
    baseline, baseline_payload = _baseline_from_rabi_params(x_arr, model_name, param_names, params)
    centered = y_arr - baseline
    if y_fit is not None and len(y_fit) == len(finite):
        y_fit_arr = np.asarray(y_fit, dtype=float)[finite][order]
        clean_centered = y_fit_arr - baseline
        if np.nanmax(np.abs(clean_centered)) > 0.0:
            centered = 0.7 * centered + 0.3 * clean_centered

    fit_t, fit_amp, source = _envelope_source_from_trace(x_rel, centered, p)
    valid = np.isfinite(fit_t) & np.isfinite(fit_amp) & (fit_amp >= 0.0)
    fit_t = fit_t[valid]
    fit_amp = fit_amp[valid]
    if len(fit_t) < 4 or float(np.nanmax(fit_amp) - np.nanmin(fit_amp)) <= 0.0:
        return {
            "success": False,
            "message": "could not extract a usable Rabi envelope",
            **baseline_payload,
            "source": source,
            "source_points": int(len(fit_t)),
        }

    tspan = max(float(np.max(x_rel) - np.min(x_rel)), 1.0)
    amp_lo = max(0.0, float(np.nanpercentile(fit_amp, 10)))
    amp_hi = max(amp_lo + 1e-9, float(np.nanpercentile(fit_amp, 90)))
    decay_seed = _safe_float(p.get("T")) or _safe_float(p.get("tau")) or max(tspan / 2.0, 1.0)
    beta_seed = _safe_float(p.get("beta")) or 1.0
    floor_seed = min(amp_lo, 0.25 * amp_hi)
    amp_seed = max(amp_hi - floor_seed, 1e-8)
    lower = np.array([0.0, 0.0, 1.0, 0.2], dtype=float)
    upper = np.array([max(amp_hi * 1.5, 1e-6), max(amp_hi * 4.0, 1e-6), max(1e6, 10.0 * tspan), 4.0], dtype=float)
    seed0 = np.array([floor_seed, amp_seed, max(float(decay_seed), 1.0), float(np.clip(beta_seed, 0.2, 4.0))], dtype=float)
    rng = np.random.default_rng(83)
    scale = np.array([0.10 * max(upper[0], 1e-6), 0.12 * max(upper[1], 1e-6), 0.4 * seed0[2], 0.3], dtype=float)
    seeds = [seed0]
    for factor in (0.35, 0.75, 1.5, 3.0):
        seeds.append(np.array([floor_seed, amp_seed, np.clip(tspan * factor, lower[2], upper[2]), seed0[3]], dtype=float))
    seeds.extend([seed0 + rng.normal(0.0, scale) for _ in range(6)])

    best = None
    best_cost = np.inf
    y_span = max(float(np.nanpercentile(fit_amp, 95) - np.nanpercentile(fit_amp, 5)), 1e-9)
    for seed in seeds:
        seed = np.clip(np.asarray(seed, dtype=float), lower, upper)
        try:
            res = least_squares(
                lambda q: _rabi_envelope_model(fit_t, *q) - fit_amp,
                seed,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=max(1e-9, 0.08 * y_span),
                max_nfev=10000,
            )
        except Exception:
            continue
        if res.cost < best_cost:
            best = res
            best_cost = float(res.cost)
    if best is None:
        return {"success": False, "message": "Rabi envelope optimizer did not run", **baseline_payload, "source": source}

    opt = np.asarray(best.x, dtype=float)
    fit_curve = _rabi_envelope_model(fit_t, *opt)
    resid = fit_amp - fit_curve
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((fit_amp - float(np.mean(fit_amp))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    rmse = float(np.sqrt(np.mean(resid**2)))
    dof = max(1, len(fit_amp) - len(opt))
    try:
        cov = (2.0 * float(best.cost) / dof) * np.linalg.pinv(best.jac.T @ best.jac)
        errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        errors = np.full_like(opt, np.nan, dtype=float)

    names = ["floor", "amp", "T2rho", "beta"]
    return {
        "success": bool(best.success),
        "message": str(best.message),
        "source": source,
        "source_points": int(len(fit_amp)),
        "params": {name: float(value) for name, value in zip(names, opt)},
        "errors": {name: float(value) for name, value in zip(names, errors)},
        "r2": float(r2),
        "rmse": rmse,
        **baseline_payload,
    }


def rabi_envelope_metrics(payload: Mapping[str, object] | None) -> dict[str, float]:
    if not payload or not bool(payload.get("success", False)):
        return {}
    params = payload.get("params")
    if not isinstance(params, Mapping):
        return {}
    T2rho = _safe_float(params.get("T2rho"))
    if T2rho is None or T2rho <= 0.0:
        return {}
    out = {
        "T2rho_ns": float(T2rho),
        "rabi_envelope_decay_ns": float(T2rho),
    }
    beta = _safe_float(params.get("beta"))
    amp = _safe_float(params.get("amp"))
    floor = _safe_float(params.get("floor"))
    r2 = _safe_float(payload.get("r2"))
    rmse = _safe_float(payload.get("rmse"))
    if beta is not None:
        out["rabi_envelope_beta"] = float(beta)
    if amp is not None:
        out["rabi_envelope_amp"] = float(amp)
    if floor is not None:
        out["rabi_envelope_floor"] = float(floor)
    if r2 is not None:
        out["rabi_envelope_r2"] = float(r2)
    if rmse is not None:
        out["rabi_envelope_rmse"] = float(rmse)
    return out


def rabi_envelope_bounds(
    x: np.ndarray,
    payload: Mapping[str, object] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not payload or not bool(payload.get("success", False)):
        return None
    params = payload.get("params")
    if not isinstance(params, Mapping):
        return None
    floor = _safe_float(params.get("floor"))
    amp = _safe_float(params.get("amp"))
    T2rho = _safe_float(params.get("T2rho"))
    beta = _safe_float(params.get("beta"))
    intercept = _safe_float(payload.get("baseline_intercept"))
    slope = _safe_float(payload.get("baseline_slope"))
    if None in {floor, amp, T2rho, beta, intercept, slope}:
        return None
    x_arr = np.asarray(x, dtype=float)
    x_rel = x_arr - float(np.nanmin(x_arr))
    baseline = float(intercept) + float(slope) * x_arr
    env = _rabi_envelope_model(x_rel, float(floor), float(amp), float(T2rho), float(beta))
    return baseline + env, baseline - env


def _first_rabi_peak_ns(params: Mapping[str, float], metadata: Mapping[str, object] | None = None) -> float | None:
    if {"f0", "chirp1", "chirp2", "delay"} <= set(params):
        adaptive = _adaptive_pulse_times(params)
        if adaptive is None:
            return None
        _pi2, pi = adaptive
        delay = _safe_float(params.get("delay")) or 0.0
        return float(delay + pi)

    freq = _safe_float(params.get("f"))
    if freq is None or freq <= 0.0:
        return None
    x_min = _safe_float((metadata or {}).get("x_min_ns")) if metadata is not None else None
    x_max = _safe_float((metadata or {}).get("x_max_ns")) if metadata is not None else None
    if x_min is None:
        x_min = 0.0
    if x_max is None:
        x_max = x_min + max(6.0 / freq, 100.0)
    x_rel = np.linspace(0.0, max(x_max - x_min, 1.0 / freq), 4096)
    period = 1.0 / freq

    if "tau_ramp" in params and "phi" in params and "alpha" not in params and "t0" not in params:
        tau_ramp = _safe_float(params.get("tau_ramp"))
        phi = _safe_float(params.get("phi"))
        amplitude = _safe_float(params.get("A")) or 0.0
        if tau_ramp is None or phi is None:
            return None
        tau_safe = max(tau_ramp, 1e-9)
        x_abs = x_min + x_rel
        theta = 2.0 * np.pi * freq * (x_abs - tau_safe * (1.0 - np.exp(-x_abs / tau_safe))) + phi
        y = amplitude * np.cos(theta)
        peaks, _ = find_peaks(y if amplitude >= 0.0 else -y, prominence=max(1e-9, 0.15 * float(np.ptp(y))))
        if len(peaks):
            return float(x_abs[int(peaks[0])])
        return float(x_abs[int(np.argmax(y if amplitude >= 0.0 else -y))])

    if "alpha" in params and "phi" in params and "t0" not in params:
        alpha = _safe_float(params.get("alpha"))
        phi = _safe_float(params.get("phi"))
        amplitude = _safe_float(params.get("A")) or 0.0
        if alpha is None or phi is None:
            return None
        theta = 2.0 * np.pi * freq * x_rel + np.pi * alpha * x_rel * x_rel + phi
        y = amplitude * np.cos(theta)
        peaks, _ = find_peaks(y if amplitude >= 0.0 else -y, prominence=max(1e-9, 0.15 * float(np.ptp(y))))
        if len(peaks):
            return float(x_min + x_rel[int(peaks[0])])
        return float(x_min + x_rel[int(np.argmax(y if amplitude >= 0.0 else -y))])

    t0 = _safe_float(params.get("t0")) or 0.0
    if "phi" not in params:
        peak = float(t0 + 1.0 / (2.0 * freq))
        if x_min is not None:
            while peak < x_min:
                peak += period
        if x_max is not None and peak > x_max + period:
            return None
        return peak
    phase = _safe_float(params.get("phi"))
    if phase is None:
        return None
    amplitude = _safe_float(params.get("A")) or 0.0
    phase_target = np.pi if amplitude < 0.0 else 0.0
    base = phase_target - phase
    k = float(np.ceil(-base / (2.0 * np.pi)))
    te = (base + 2.0 * np.pi * k) / (2.0 * np.pi * freq)
    te = te if te >= 0.0 else te + (1.0 / freq)
    peak = float(t0 + te)
    if x_min is not None:
        while peak < x_min:
            peak += period
    if x_max is not None and peak > x_max + period:
        return None
    return peak


def _adaptive_pulse_times(params: Mapping[str, float]) -> tuple[float, float] | None:
    """Return effective pi/2 and pi durations for an adaptive pulse-area fit."""
    f0 = _safe_float(params.get("f0"))
    chirp1 = _safe_float(params.get("chirp1"))
    chirp2 = _safe_float(params.get("chirp2"))
    if f0 is None or chirp1 is None or chirp2 is None or f0 <= 0.0:
        return None

    def cycles(u: float) -> float:
        return f0 * u + 0.5 * chirp1 * u * u + (chirp2 * u * u * u) / 3.0

    upper = max(4.0 / f0, 20.0)
    for _ in range(12):
        grid = np.linspace(0.0, upper, 512)
        rate = f0 + chirp1 * grid + chirp2 * grid * grid
        if np.all(rate > 0.0) and cycles(upper) >= 0.5:
            break
        upper *= 2.0
    else:
        return None
    try:
        pi2 = brentq(lambda u: cycles(u) - 0.25, 0.0, upper)
        pi = brentq(lambda u: cycles(u) - 0.5, 0.0, upper)
    except ValueError:
        return None
    return float(pi2), float(pi)


def build_rabi_nv_metrics(
    params: Mapping[str, float],
    *,
    metadata: Mapping[str, object] | None = None,
    errors: Mapping[str, float] | None = None,
) -> dict[str, float]:
    freq_candidates = [(name, float(params[name])) for name in ("f0", "f", "f1", "f2") if name in params and float(params[name]) > 0.0]
    metrics: dict[str, float] = {}
    if not freq_candidates:
        return metrics

    freq_name, freq_value = max(freq_candidates, key=lambda item: item[1])
    saved_dead_time = extract_rabi_dead_time_ns(metadata)
    dead_time = saved_dead_time
    derived_first_peak = _first_rabi_peak_ns(params, metadata=metadata)
    explicit_delay = _safe_float(params.get("delay"))
    explicit_t0 = _safe_float(params.get("t0"))
    if explicit_delay is not None:
        dead_time = max(0.0, explicit_delay)
    elif explicit_t0 is not None:
        dead_time = max(0.0, explicit_t0)
    elif saved_dead_time <= 0.0 and derived_first_peak is not None:
        dead_time = max(0.0, derived_first_peak - (1.0 / (2.0 * freq_value)))
    metrics.update(rabi_timing_metrics_from_frequency(freq_value, dead_time))
    metrics["rabi_period_ns"] = float(1.0 / freq_value)
    if saved_dead_time > 0.0:
        metrics["saved_rf_ramp_time_ns"] = float(saved_dead_time)
    adaptive_times = _adaptive_pulse_times(params)
    if adaptive_times is not None:
        pi2_time, pi_time = adaptive_times
        metrics["pi_time_ns"] = float(pi_time)
        metrics["pi_over_2_time_ns"] = float(pi2_time)
        metrics["programmed_pi_time_ns"] = float(dead_time + pi_time)
        metrics["programmed_pi_over_2_time_ns"] = float(dead_time + pi2_time)
        metrics["first_lobe_rabi_freq_MHz"] = float(500.0 / pi_time)
    if derived_first_peak is not None:
        metrics["first_peak_ns"] = float(derived_first_peak)
        metrics["delay_ns"] = float(dead_time)
    if "tau_ramp" in params:
        tau_ramp = _safe_float(params.get("tau_ramp"))
        if tau_ramp is not None:
            metrics["tau_ramp_ns"] = float(tau_ramp)
    if "alpha" in params:
        alpha = _safe_float(params.get("alpha"))
        if alpha is not None:
            metrics["chirp_alpha"] = float(alpha)
    if "chirp1" in params:
        chirp1 = _safe_float(params.get("chirp1"))
        chirp2 = _safe_float(params.get("chirp2"))
        if chirp1 is not None:
            metrics["pulse_area_chirp1_per_ns2"] = float(chirp1)
        if chirp2 is not None:
            metrics["pulse_area_chirp2_per_ns3"] = float(chirp2)

    if errors is not None and freq_name in errors:
        freq_err = _safe_float(errors.get(freq_name))
        if freq_err is not None:
            pi_err = abs(freq_err / (2.0 * (freq_value ** 2)))
            metrics["pi_time_ns_nominal_err"] = float(pi_err)
            metrics["pi_over_2_time_ns_nominal_err"] = float(0.5 * pi_err)
            # For fixed-frequency models the nominal and calibrated times are
            # identical.  Adaptive pulse-area timing also depends on chirp1
            # and chirp2, so f0 uncertainty alone must not be presented as the
            # full calibrated pulse-time uncertainty.
            if adaptive_times is None:
                metrics["pi_time_ns_err"] = float(pi_err)
                metrics["pi_over_2_time_ns_err"] = float(0.5 * pi_err)
    if errors is not None and "delay" in errors:
        delay_err = _safe_float(errors.get("delay"))
        if delay_err is not None:
            metrics["delay_ns_err"] = float(abs(delay_err))

    return metrics
