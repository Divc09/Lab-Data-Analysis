from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import numpy as np


def time_trace_decay(t, background, amplitude, tau_s):
    """Descriptive exponential drift against measured elapsed seconds."""
    return background + amplitude * np.exp(-np.asarray(t, dtype=float) / tau_s)


def ramsey_simple(t, y0, A, f, phi, tau, n):
    x = np.abs(t / tau)
    return y0 + A * np.exp(-(x**n)) * np.cos(2 * np.pi * f * t + phi)


def ramsey_hyperfine_n14(t, y0, A, f, phi, tau, n, df=0.00216):
    env = np.exp(-((np.abs(t / tau)) ** n))
    carrier = (
        np.cos(2 * np.pi * (f - df) * t + phi)
        + np.cos(2 * np.pi * f * t + phi)
        + np.cos(2 * np.pi * (f + df) * t + phi)
    ) / 3.0
    return y0 + A * env * carrier


def ramsey_extended(t, A1, A2, T2_star, n, f1, f2, phi1, phi2, B, D, Td):
    env = np.exp(-((np.abs(t / T2_star)) ** n))
    osc = A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)
    return B + D * np.exp(-t / Td) + env * osc


def rabi_model(t, y0, m, A, f, t0, tau):
    te = np.maximum(t - t0, 0.0)
    return y0 + m * t + A * (np.sin(np.pi * f * te) ** 2) * np.exp(-t / tau)


def rabi_cosine_decay_model(t, c, a, T, beta, f, phi):
    x = np.asarray(t, dtype=float)
    env = np.exp(-((np.abs(x / T)) ** beta))
    return c + a * env * np.cos(2 * np.pi * f * x + phi)


def rabi_phase_ramp_model(t, c, a, T, beta, f, tau_ramp, phi):
    x = np.asarray(t, dtype=float)
    T_safe = np.maximum(T, 1e-9)
    beta_safe = np.maximum(beta, 1e-9)
    env = np.exp(-((np.abs(x / T_safe)) ** beta_safe))
    tau_safe = np.maximum(tau_ramp, 1e-9)
    theta = 2 * np.pi * f * (x - tau_safe * (1.0 - np.exp(-x / tau_safe))) + phi
    return c + a * env * np.cos(theta)


def rabi_phase_ramp_jac(t, c, a, T, beta, f, tau_ramp, phi):
    """Analytic d model / d (c, a, T, beta, f, tau_ramp, phi) for rabi_phase_ramp_model, for T, beta and tau_ramp
    above the model's 1e-9 safety floor (always true within the fit bounds)."""
    x = np.asarray(t, dtype=float)
    u = np.abs(x / T)
    ub = u**beta
    env = np.exp(-ub)
    decay = np.exp(-x / tau_ramp)
    lag = x - tau_ramp * (1.0 - decay)
    theta = 2 * np.pi * f * lag + phi
    cos, sin = np.cos(theta), np.sin(theta)
    log_u = np.log(u, out=np.zeros_like(u), where=u > 0)
    osc = -a * env * sin
    return np.column_stack([np.ones_like(x), env * cos, a * cos * env * beta * ub / T, -a * cos * env * ub * log_u,
                            osc * 2 * np.pi * lag, osc * 2 * np.pi * f * (decay * x / tau_ramp - (1.0 - decay)), osc])


def rabi_adaptive_pulse_area_model(t, c, a, T, beta, f0, chirp1, chirp2, delay):
    """Rabi signal with a smooth, monotonic-capable pulse-area calibration.

    ``f0 + chirp1*u + chirp2*u**2`` is the instantaneous effective Rabi
    frequency after the fitted timing delay.  The polynomial describes pulse
    area versus programmed duration; it does not imply that the microwave
    carrier frequency is chirped.
    """
    x = np.asarray(t, dtype=float)
    u = np.maximum(x - delay, 0.0)
    T_safe = np.maximum(T, 1e-9)
    beta_safe = np.maximum(beta, 1e-9)
    env = np.exp(-((u / T_safe) ** beta_safe))
    cycles = f0 * u + 0.5 * chirp1 * u * u + (chirp2 * u * u * u) / 3.0
    return c - a * env * np.cos(2.0 * np.pi * cycles)


def rabi_chirp_model(t, c, a, T, beta, f, alpha, phi):
    x = np.asarray(t, dtype=float)
    env = np.exp(-((np.abs(x / T)) ** beta))
    theta = 2 * np.pi * f * x + np.pi * alpha * x * x + phi
    return c + a * env * np.cos(theta)


def rabi_chirp_jac(t, c, a, T, beta, f, alpha, phi):
    """Analytic d model / d (c, a, T, beta, f, alpha, phi) for rabi_chirp_model."""
    x = np.asarray(t, dtype=float)
    u = np.abs(x / T)
    ub = u**beta
    env = np.exp(-ub)
    theta = 2 * np.pi * f * x + np.pi * alpha * x * x + phi
    cos, sin = np.cos(theta), np.sin(theta)
    log_u = np.log(u, out=np.zeros_like(u), where=u > 0)  # u^beta * ln u -> 0 as u -> 0
    osc = -a * env * sin
    return np.column_stack([np.ones_like(x), env * cos, a * cos * env * beta * ub / T, -a * cos * env * ub * log_u,
                            osc * 2 * np.pi * x, osc * np.pi * x * x, osc])


def rabi_shifted_model(t, y0, m, A, f, phi, tau, t0):
    te = np.maximum(t - t0, 0.0)
    return y0 + m * t + A * np.cos(2 * np.pi * f * te + phi) * np.exp(-te / tau)


def rabi_dual_model(t, y0, m, A1, A2, f1, f2, phi1, phi2, tau):
    env = np.exp(-t / tau)
    osc = A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)
    return y0 + m * t + env * osc


def spin_echo_model(t, y0, A, T2, n):
    return y0 + A * np.exp(-((np.abs(t / T2)) ** n))


def t1_rise_exponential_model(t, y0, A, T1):
    x = np.abs(t / T1)
    return y0 + A * (1.0 - np.exp(-x))


def t1_rise_stretched_model(t, y0, A, T1, n):
    x = np.abs(t / T1)
    return y0 + A * (1.0 - np.exp(-(x**n)))


def t1_decay_exponential_model(t, y0, A, T1):
    x = np.abs(t / T1)
    return y0 + A * np.exp(-x)


def t1_decay_stretched_model(t, y0, A, T1, n):
    x = np.abs(t / T1)
    return y0 + A * np.exp(-(x**n))


def odmr_lorentzian(t, y0, C, w, x0):
    return y0 + C * (w**2 / ((t - x0) ** 2 + w**2))


def odmr_multi_lorentzian(t, y0, *peak_params):
    """
    Multi-peak ODMR Lorentzian model.
    peak_params grouped as (C1, w1, x01, C2, w2, x02, ...).
    """
    y = np.full_like(t, float(y0), dtype=float)
    ntrip = len(peak_params) // 3
    for i in range(ntrip):
        C = float(peak_params[3 * i + 0])
        w = float(peak_params[3 * i + 1])
        x0 = float(peak_params[3 * i + 2])
        y = y + C * (w**2 / ((t - x0) ** 2 + w**2))
    return y


def odmr_multi_lorentzian_jac(t, y0, *peak_params):
    """Analytic d model / d (y0, C1, w1, x01, ...) for odmr_multi_lorentzian, one column per parameter."""
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t)]
    for i in range(len(peak_params) // 3):
        C, w, x0 = (float(v) for v in peak_params[3 * i: 3 * i + 3])
        d = t - x0
        denom = d**2 + w**2
        shape = w**2 / denom
        cols += [shape, C * 2.0 * w * d**2 / denom**2, C * 2.0 * w**2 * d / denom**2]
    return np.column_stack(cols)


@dataclass(frozen=True)
class CustomModelSpec:
    name: str
    expression: str
    param_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    seed: tuple[float, ...]
    x_label: str = "X"
    y_label: str = "Signal"


def build_custom_expression_model(expression: str, param_names: list[str]) -> Callable[..., np.ndarray]:
    """
    Build a custom model callable from a NumPy-safe expression.
    Expression can reference the independent variable as t or x (aliases), np, and
    parameters by name.
    """
    allowed_functions = {
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "clip": np.clip,
        "pi": np.pi,
    }
    reserved = {"t", "x"} & set(param_names)
    if reserved:
        raise ValueError(f"'{sorted(reserved)[0]}' is the independent variable and cannot be a parameter name.")
    normalized = expression.replace("np.", "")
    tree = ast.parse(normalized, mode="eval")
    allowed_names = {"t", "x", *param_names, *allowed_functions}
    allowed_calls = set(allowed_functions) - {"pi"}
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
        ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Custom expressions may use only numbers, operators, parameters, t/x, and approved functions.")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unsupported name in custom expression: {node.id}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls):
            raise ValueError("Custom expressions may call only approved mathematical functions.")
    code = compile(tree, "<custom model>", "eval")

    def _model(t, *params):
        local = {"t": t, "x": t}
        for n, v in zip(param_names, params):
            local[n] = v
        return np.asarray(eval(code, {"__builtins__": {}}, {**allowed_functions, **local}), dtype=float)

    return _model


def gaussian_peaks(x, background, *peak_params):
    """Constant background plus (amplitude, center, sigma) Gaussian peaks."""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, background)
    for amplitude, center, sigma in np.asarray(peak_params).reshape(-1, 3):
        result += amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return result


MODEL_REGISTRY = {
    "TimeTraceDecay": time_trace_decay,
    "GaussianPeaks": gaussian_peaks,
    "RamseySimple": ramsey_simple,
    "RamseyHyperfine": ramsey_hyperfine_n14,
    "RamseyExtended": ramsey_extended,
    "Rabi": rabi_model,
    "RabiCosine": rabi_cosine_decay_model,
    "RabiLongDamped": rabi_cosine_decay_model,
    "RabiPhaseRamp": rabi_phase_ramp_model,
    "RabiAdaptive": rabi_adaptive_pulse_area_model,
    "RabiChirp": rabi_chirp_model,
    "RabiShifted": rabi_shifted_model,
    "RabiDual": rabi_dual_model,
    "SpinEcho": spin_echo_model,
    "T1Exp": t1_rise_exponential_model,
    "T1Stretch": t1_rise_stretched_model,
    "T1NormExp": t1_decay_exponential_model,
    "T1NormStretch": t1_decay_stretched_model,
    "T1DecayExp": t1_decay_exponential_model,
    "T1DecayStretch": t1_decay_stretched_model,
    "ODMR": odmr_lorentzian,
    "ODMRMulti": odmr_multi_lorentzian,
}

