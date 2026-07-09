"""Reusable, source-preserving analysis transforms for SmartFitter."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnalysisStep:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": self.params, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalysisStep":
        return cls(str(value.get("kind", "")), dict(value.get("params") or {}), bool(value.get("enabled", True)))


@dataclass
class ProcessedSeries:
    """Fit-safe data plus its displayed/derived analysis representation."""

    fit_x: np.ndarray
    fit_y: np.ndarray
    analysis_x: np.ndarray
    analysis_y: np.ndarray
    source_groups: list[np.ndarray]
    excluded_raw_indices: set[int]
    exclusion_ranges: list[tuple[float, float]]
    provenance: list[dict[str, Any]] = field(default_factory=list)


_ALLOWED_EXPR_FUNCTIONS = {
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "clip": np.clip,
}


def _finite(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _safe_expression(expression: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    tree = ast.parse(expression, mode="eval")
    allowed_names = {"x", "y", *(_ALLOWED_EXPR_FUNCTIONS.keys())}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unsupported name in expression: {node.id}")
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.List, ast.Tuple, ast.comprehension)):
            raise ValueError("Expressions may use only x, y, numbers, operators, and approved functions.")
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
                                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)):
            raise ValueError("Unsupported expression syntax.")
    result = eval(compile(tree, "<analysis expression>", "eval"), {"__builtins__": {}}, {"x": x, "y": y, **_ALLOWED_EXPR_FUNCTIONS})
    result = np.asarray(result, dtype=float)
    if result.shape == ():
        result = np.full_like(y, float(result), dtype=float)
    if result.shape != y.shape:
        raise ValueError("Expression must return one value per displayed point.")
    return result


def apply_steps(x: np.ndarray, y: np.ndarray, steps: list[AnalysisStep]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Apply display/measurement transforms without mutating caller arrays."""
    out_x, out_y = np.asarray(x, dtype=float).copy(), np.asarray(y, dtype=float).copy()
    provenance: list[dict[str, Any]] = []
    for step in steps:
        if not step.enabled:
            continue
        kind, p = step.kind, step.params
        if len(out_x) == 0:
            break
        finite_x, finite_y = _finite(out_x, out_y)
        if len(finite_y) == 0:
            raise ValueError(f"{kind} requires at least one finite point.")
        if kind == "baseline":
            method = str(p.get("method", "median"))
            if method == "linear":
                if len(finite_x) < 2:
                    raise ValueError("Linear baseline requires at least two finite points.")
                baseline = np.polyval(np.polyfit(finite_x, finite_y, 1), out_x)
            elif method == "mean":
                baseline = float(np.mean(finite_y))
            elif method == "first":
                baseline = float(finite_y[0])
            else:
                baseline = float(np.median(finite_y))
            out_y = out_y - baseline
        elif kind == "detrend":
            if len(finite_x) < 2:
                raise ValueError("Detrend requires at least two finite points.")
            out_y = out_y - np.polyval(np.polyfit(finite_x, finite_y, 1), out_x)
        elif kind == "normalize":
            method = str(p.get("method", "minmax"))
            if method == "zscore":
                scale = float(np.std(finite_y))
                out_y = (out_y - float(np.mean(finite_y))) / (scale if scale > 0 else 1.0)
            elif method == "area":
                area = float(np.trapezoid(np.abs(finite_y), finite_x)) if len(finite_x) > 1 else abs(float(finite_y[0]))
                out_y = out_y / (area if area > 0 else 1.0)
            else:
                lo, hi = float(np.min(finite_y)), float(np.max(finite_y))
                out_y = (out_y - lo) / ((hi - lo) if hi > lo else 1.0)
        elif kind == "derivative":
            if len(out_x) < 2:
                raise ValueError("Derivative requires at least two points.")
            out_y = np.gradient(out_y, out_x)
        elif kind == "integral":
            if len(out_x) < 2:
                out_y = np.zeros_like(out_y)
            else:
                dx = np.diff(out_x)
                out_y = np.concatenate([[0.0], np.cumsum(0.5 * (out_y[1:] + out_y[:-1]) * dx)])
        elif kind == "resample":
            count = max(2, int(p.get("count", len(out_x))))
            out_x = np.linspace(float(np.min(finite_x)), float(np.max(finite_x)), count)
            out_y = np.interp(out_x, finite_x, finite_y)
        elif kind == "expression":
            out_y = _safe_expression(str(p.get("expression", "y")), out_x, out_y)
        else:
            raise ValueError(f"Unknown analysis step: {kind}")
        provenance.append({"kind": kind, "params": dict(p)})
    return out_x, out_y, provenance


def process_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_size: int = 1,
    roi: tuple[float | None, float | None] = (None, None),
    excluded_raw_indices: set[int] | None = None,
    exclusion_ranges: list[tuple[float, float]] | None = None,
    steps: list[AnalysisStep] | None = None,
) -> ProcessedSeries:
    """Build a fit-safe binned/ROI series and a separate derived view.

    Bins retain their raw source-index membership.  Masking therefore remains
    stable when the user changes ROI, smoothing, or display transforms.
    """
    raw_x, raw_y = np.asarray(x, dtype=float).copy(), np.asarray(y, dtype=float).copy()
    excluded = set(excluded_raw_indices or set())
    ranges = [(min(float(a), float(b)), max(float(a), float(b))) for a, b in (exclusion_ranges or [])]
    b = max(1, int(bin_size))
    bx: list[float] = []
    by: list[float] = []
    groups: list[np.ndarray] = []
    for start in range(0, len(raw_x), b):
        raw_indices = np.arange(start, min(start + b, len(raw_x)), dtype=int)
        keep = np.isfinite(raw_x[raw_indices]) & np.isfinite(raw_y[raw_indices])
        keep &= ~np.isin(raw_indices, list(excluded))
        for lo, hi in ranges:
            keep &= ~((raw_x[raw_indices] >= lo) & (raw_x[raw_indices] <= hi))
        source = raw_indices[keep]
        if len(source):
            bx.append(float(np.mean(raw_x[source])))
            by.append(float(np.mean(raw_y[source])))
            groups.append(source)
    fit_x, fit_y = np.asarray(bx, dtype=float), np.asarray(by, dtype=float)
    lo, hi = roi
    if lo is not None:
        keep = fit_x >= float(lo)
        fit_x, fit_y = fit_x[keep], fit_y[keep]
        groups = [g for g, ok in zip(groups, keep) if ok]
    if hi is not None:
        keep = fit_x <= float(hi)
        fit_x, fit_y = fit_x[keep], fit_y[keep]
        groups = [g for g, ok in zip(groups, keep) if ok]
    analysis_x, analysis_y, provenance = apply_steps(fit_x, fit_y, list(steps or []))
    if len(analysis_x) != len(groups):
        # Resampling maps each displayed sample to its nearest fit-safe bin.
        groups = [groups[int(np.argmin(np.abs(fit_x - value)))] for value in analysis_x] if len(fit_x) else []
    return ProcessedSeries(fit_x, fit_y, analysis_x, analysis_y, groups, excluded, ranges, provenance)
