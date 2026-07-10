"""Shared, serializable application state for SmartFitter."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .analysis_tools import AnalysisStep, ProcessedSeries
from .fit_engine import FitResult
from .io_mat import ExperimentTrace
from .profiles import FitProfile


def json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def state_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_result_to_dict(result: FitResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return json_safe({
        "model_name": result.model_name,
        "param_names": result.param_names,
        "params": result.params,
        "errors": result.errors,
        "y_fit": result.y_fit,
        "r2": result.r2,
        "rmse": result.rmse,
        "aic": result.aic,
        "bic": result.bic,
        "durbin_watson": result.durbin_watson,
        "bound_hits": result.bound_hits,
        "success": result.success,
        "message": result.message,
        "selection_note": result.selection_note,
        "extras": result.extras,
    })


def fit_result_from_dict(value: dict[str, Any] | None) -> FitResult | None:
    if not value:
        return None
    return FitResult(
        model_name=str(value["model_name"]),
        param_names=[str(item) for item in value.get("param_names", [])],
        params=np.asarray(value.get("params", []), dtype=float),
        errors=np.asarray(value.get("errors", []), dtype=float),
        y_fit=np.asarray(value.get("y_fit", []), dtype=float),
        r2=float(value.get("r2", np.nan)),
        rmse=float(value.get("rmse", np.nan)),
        aic=float(value.get("aic", np.nan)),
        bic=float(value.get("bic", np.nan)),
        durbin_watson=float(value.get("durbin_watson", np.nan)),
        bound_hits=[bool(item) for item in value.get("bound_hits", [])],
        success=bool(value.get("success", False)),
        message=str(value.get("message", "")),
        selection_note=str(value.get("selection_note", "")),
        extras=dict(value.get("extras") or {}),
    )


@dataclass
class AnalysisDocument:
    trace: ExperimentTrace | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    y_smooth: np.ndarray | None = None
    fit_result: FitResult | None = None
    fit_target: np.ndarray | None = None
    fit_signature: str | None = None
    pending_locks: dict[str, tuple[float, bool, float | None, float | None]] = field(default_factory=dict)
    status: str = "N/A"
    status_reason: str = ""
    profile: FitProfile | None = None
    odmr_peaks: np.ndarray | None = None
    excluded_points: set[int] = field(default_factory=set)
    exclusion_ranges: list[tuple[float, float]] = field(default_factory=list)
    loaded_traces: dict[str, ExperimentTrace] = field(default_factory=dict)
    sample_metadata: dict[str, str] = field(default_factory=dict)
    active_preset_name: str = "default"
    custom_model_spec: dict[str, Any] | None = None
    analysis_steps: list[AnalysisStep] = field(default_factory=list)
    processed: ProcessedSeries | None = None
    analysis_x: np.ndarray | None = None
    analysis_y: np.ndarray | None = None
    dirty: bool = False


@dataclass(frozen=True)
class FitRequest:
    request_id: int
    signature: str


@dataclass(frozen=True)
class FitOutcome:
    request: FitRequest
    result: Any


@dataclass(frozen=True)
class HistoryEntry:
    label: str
    snapshot: dict[str, Any]


class HistoryStack:
    """Chronological snapshot history shared by parameters and analysis edits."""

    def __init__(self, limit: int = 100):
        self.limit = max(1, int(limit))
        self.undo_entries: list[HistoryEntry] = []
        self.redo_entries: list[HistoryEntry] = []

    def push(self, label: str, snapshot: dict[str, Any]) -> None:
        entry = HistoryEntry(label, snapshot)
        if self.undo_entries and self.undo_entries[-1].snapshot == snapshot:
            return
        self.undo_entries.append(entry)
        self.undo_entries = self.undo_entries[-self.limit :]
        self.redo_entries.clear()

    def undo(self, current: dict[str, Any]) -> HistoryEntry | None:
        if not self.undo_entries:
            return None
        entry = self.undo_entries.pop()
        self.redo_entries.append(HistoryEntry(entry.label, current))
        return entry

    def redo(self, current: dict[str, Any]) -> HistoryEntry | None:
        if not self.redo_entries:
            return None
        entry = self.redo_entries.pop()
        self.undo_entries.append(HistoryEntry(entry.label, current))
        return entry

    def clear(self) -> None:
        self.undo_entries.clear()
        self.redo_entries.clear()
