from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import numpy as np


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


def rabi_phase_ramp_model(t, c, a, f, tau_ramp, phi):
    x = np.asarray(t, dtype=float)
    tau_safe = np.maximum(tau_ramp, 1e-9)
    theta = 2 * np.pi * f * (x - tau_safe * (1.0 - np.exp(-x / tau_safe))) + phi
    return c + a * np.cos(theta)


def rabi_chirp_model(t, c, a, T, beta, f, alpha, phi):
    x = np.asarray(t, dtype=float)
    env = np.exp(-((np.abs(x / T)) ** beta))
    theta = 2 * np.pi * f * x + np.pi * alpha * x * x + phi
    return c + a * env * np.cos(theta)


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
    Expression can reference: t, np, and parameters by name.
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
    normalized = expression.replace("np.", "")
    tree = ast.parse(normalized, mode="eval")
    allowed_names = {"t", *param_names, *allowed_functions}
    allowed_calls = set(allowed_functions) - {"pi"}
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
        ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Custom expressions may use only numbers, operators, parameters, t, and approved functions.")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unsupported name in custom expression: {node.id}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls):
            raise ValueError("Custom expressions may call only approved mathematical functions.")
    code = compile(tree, "<custom model>", "eval")

    def _model(t, *params):
        local = {"t": t}
        for n, v in zip(param_names, params):
            local[n] = v
        return np.asarray(eval(code, {"__builtins__": {}}, {**allowed_functions, **local}), dtype=float)

    return _model


MODEL_REGISTRY = {
    "RamseySimple": ramsey_simple,
    "RamseyHyperfine": ramsey_hyperfine_n14,
    "RamseyExtended": ramsey_extended,
    "Rabi": rabi_model,
    "RabiCosine": rabi_cosine_decay_model,
    "RabiLongDamped": rabi_cosine_decay_model,
    "RabiPhaseRamp": rabi_phase_ramp_model,
    "RabiChirp": rabi_chirp_model,
    "RabiShifted": rabi_shifted_model,
    "RabiDual": rabi_dual_model,
    "SpinEcho": spin_echo_model,
    "T1Exp": t1_rise_exponential_model,
    "T1Stretch": t1_rise_stretched_model,
    "T1NormExp": t1_decay_exponential_model,
    "T1NormStretch": t1_decay_stretched_model,
    "ODMR": odmr_lorentzian,
    "ODMRMulti": odmr_multi_lorentzian,
}

