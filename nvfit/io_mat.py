from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from scipy.io import loadmat


DataMode = Literal["contrast", "signal", "reference", "difference", "raw_signal"]
DetectorId = Literal["detector1", "detector2"]

DETECTOR_LABELS: dict[str, str] = {
    "detector1": "Detector 1",
    "detector2": "Detector 2",
}


@dataclass
class ExperimentTrace:
    source_path: str
    file_name: str
    experiment_type: str
    scan_axes: tuple[str, ...]
    scan_dim: Literal["time", "scan1d", "scan2d"]
    fit_allowed: bool
    x_ns: np.ndarray
    signal: np.ndarray
    reference: np.ndarray
    y: np.ndarray
    y_label: str
    x_label: str
    mode: DataMode
    raw_x: np.ndarray | None = None
    raw_x_label: str | None = None
    metadata: dict[str, object] | None = None
    x2d: np.ndarray | None = None
    y2d: np.ndarray | None = None
    z2d: np.ndarray | None = None
    y_sem: np.ndarray | None = None
    iteration_contrast: np.ndarray | None = None
    iteration_signal: np.ndarray | None = None
    iteration_reference: np.ndarray | None = None
    iteration_valid: np.ndarray | None = None
    signal2d: np.ndarray | None = None
    reference2d: np.ndarray | None = None
    detector_id: str = "detector1"
    detector_label: str = "Detector 1"
    available_detectors: tuple[str, ...] = ("detector1",)


@dataclass(frozen=True)
class DetectorLoadFailure:
    detector_id: str
    detector_label: str
    message: str
    run_status: str = ""
    error_identifier: str = ""
    error_message: str = ""


@dataclass
class ExperimentDataset:
    source_path: str
    file_name: str
    detector_mode: str
    available_detectors: tuple[str, ...]
    detectors: dict[str, ExperimentTrace]
    detector_errors: dict[str, DetectorLoadFailure]
    metadata: dict[str, object]


class MatDataError(ValueError):
    """Structured MAT loading failure with acquisition and detector context."""

    def __init__(
        self,
        message: str,
        *,
        detector_id: str = "",
        run_status: str = "",
        error_identifier: str = "",
        error_message: str = "",
    ) -> None:
        super().__init__(message)
        self.detector_id = detector_id
        self.run_status = run_status
        self.error_identifier = error_identifier
        self.error_message = error_message


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(np.asarray(value).astype(str).ravel()).strip()
        if value.dtype.kind in {"i", "u"} and value.ndim == 1:
            try:
                return "".join(chr(int(v)) for v in value.ravel() if int(v) != 0).strip()
            except Exception:
                return str(np.asarray(value).ravel().tolist()).strip()
    return str(value).strip()


def _keyword_match(text: str, keywords: tuple[str, ...]) -> bool:
    tl = text.lower()
    return any(k in tl for k in keywords)


def _canonical_mode(mode: DataMode | str) -> DataMode:
    text = str(mode).strip().lower()
    if text == "raw_signal":
        return "signal"
    if text in {"contrast", "signal", "reference", "difference"}:
        return text  # type: ignore[return-value]
    raise ValueError(f"Unsupported observable mode: {mode}")


def infer_experiment_type(
    file_name: str,
    scan_types: tuple[str, ...] = (),
    *,
    notes: str = "",
    parameter: str = "",
    identifier: str = "",
) -> str:
    fn = file_name.lower()
    scan_types_l = [s.lower() for s in scan_types]
    notes_l = notes.lower()
    parameter_l = parameter.lower()
    identifier_l = identifier.lower()
    combined = " ".join([fn, notes_l, parameter_l, identifier_l, *scan_types_l])
    dd_keywords = ("xy8", "cpmg", "dynamic decoupling", "dynamical decoupling")
    stage_keywords = ("stage x", "stage y", "stage z", "stage_x", "stage_y", "stage_z")

    # Prefer explicit sequence metadata and physical scan axes over generic
    # words such as "duration" or "frequency".  Those generic axis names are
    # useful fallbacks, but on their own they are not as strong as a sequence
    # name in the filename/notes or a stage-axis declaration.

    if "deer" in combined:
        if "position" in combined:
            return "DEERPosition"
        if "decay" in combined or "scan axis: tau" in combined:
            return "DEERDecay"
        if "duration" in combined:
            return "DEERDuration"
        if "frequency" in combined or "freq" in combined:
            return "DEERFrequency"
        return "DEERFrequency"

    if any(_keyword_match(value, stage_keywords) for value in (*scan_types_l, parameter_l, identifier_l)):
        return "LineScan"

    if _keyword_match(combined, dd_keywords):
        return "DynamicDecoupling"

    if "ramsey" in combined:
        return "Ramsey"
    if re.search(r"(?:^|[^a-z0-9])t\s*1(?:[^a-z0-9]|$)", combined) or "longitudinal relaxation" in combined:
        return "T1"
    if _keyword_match(combined, ("spin echo", "spinecho", "hahn echo", "t2 decay")):
        return "SpinEcho"
    if "rabi" in combined:
        return "Rabi"
    if "odmr" in combined:
        return "ODMR"

    if any("rf frequency" in s or "frequency" in s for s in scan_types_l) or "frequency" in parameter_l:
        return "ODMR"
    if any("pulse duration" in s or "duration" in s for s in scan_types_l) or "duration" in parameter_l:
        return "Rabi"
    if "echo" in combined or re.search(r"(?:^|[^a-z0-9])t\s*2(?:[^a-z0-9]|$)", combined):
        return "SpinEcho"
    return "LineScan"


def _as_scan_list(loaded_scan) -> list:
    if isinstance(loaded_scan, np.ndarray):
        flat = loaded_scan.ravel()
        return [s for s in flat if hasattr(s, "nsteps")]
    return [loaded_scan]


def _pick_scan_axis(loaded_scan) -> tuple[float, float, int]:
    scans = _as_scan_list(loaded_scan)
    if not scans:
        raise ValueError("loadedScan array does not contain scan structs")
    nvals = [int(np.asarray(s.nsteps, dtype=float).ravel()[0]) if np.asarray(s.nsteps).size else 0 for s in scans]
    loaded_scan = scans[int(np.argmax(nvals))]

    starts = np.asarray(loaded_scan.starts, dtype=float).ravel()
    ends = np.asarray(loaded_scan.ends, dtype=float).ravel()
    nsteps = np.asarray(loaded_scan.nsteps, dtype=float).ravel()

    if nsteps.size == 0:
        raise ValueError("loadedScan.nsteps is empty")
    if nsteps.size == 1:
        idx = 0
    else:
        idx = int(np.argmax(nsteps))
        if nsteps[idx] <= 1:
            idx = int(np.argmax(np.abs(ends - starts)))

    n = int(nsteps[idx]) if idx < len(nsteps) else int(nsteps[0])
    start = float(starts[idx] if idx < len(starts) else starts[0])
    end = float(ends[idx] if idx < len(ends) else ends[0])
    return start, end, n


def _scan_axis_info(one_scan) -> tuple[str, np.ndarray]:
    axis_type = str(np.asarray(getattr(one_scan, "type", ["Axis"])).ravel()[0])
    starts = np.asarray(one_scan.starts, dtype=float).ravel()
    ends = np.asarray(one_scan.ends, dtype=float).ravel()
    nsteps = np.asarray(one_scan.nsteps, dtype=float).ravel()
    if nsteps.size == 0:
        raise ValueError("nsteps missing in loadedScan entry")
    n = int(nsteps[0])
    start = float(starts[0]) if len(starts) else 0.0
    end = float(ends[0]) if len(ends) else start
    axis = np.linspace(start, end, n)
    return axis_type, axis


def _extract_vectors(saved_data) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(saved_data, "signal") and hasattr(saved_data, "reference"):
        signal = np.asarray(saved_data.signal, dtype=float).ravel()
        reference = np.asarray(saved_data.reference, dtype=float).ravel()
    elif hasattr(saved_data, "signalVector") and hasattr(saved_data, "referenceVector"):
        signal = np.asarray(saved_data.signalVector, dtype=float).ravel()
        reference = np.asarray(saved_data.referenceVector, dtype=float).ravel()
    else:
        raise ValueError("Could not find signal/reference vectors in savedData.")
    return signal, reference


def _compose_observable(signal: np.ndarray, reference: np.ndarray, mode: DataMode) -> tuple[np.ndarray, str]:
    resolved_mode = _canonical_mode(mode)
    if resolved_mode == "signal":
        return signal.copy(), "Signal [counts]"
    if resolved_mode == "reference":
        return reference.copy(), "Reference [counts]"
    if resolved_mode == "difference":
        return (reference - signal), "Difference [reference - signal]"
    ref = reference.copy()
    ref[ref == 0] = np.nan
    return (reference - signal) / ref, "Contrast"


def _stage_axis_count(scan_axes: tuple[str, ...]) -> int:
    return sum(1 for a in scan_axes if _keyword_match(a, ("stage x", "stage y", "stage z")))


def _plot_only_scan_type(experiment_type: str) -> bool:
    return experiment_type in {"LineScan", "Scan2D", "DEERPosition"}


def _finalize_trace(
    *,
    mat_path: Path,
    experiment_type: str,
    scan_axes: tuple[str, ...],
    scan_vectors: list[np.ndarray],
    x_ns: np.ndarray,
    signal: np.ndarray,
    reference: np.ndarray,
    y: np.ndarray,
    y_label: str,
    x_label: str,
    mode: DataMode,
    raw_x: np.ndarray | None = None,
    raw_x_label: str | None = None,
    metadata: dict[str, object] | None = None,
    y_sem: np.ndarray | None = None,
    iteration_contrast: np.ndarray | None = None,
    iteration_signal: np.ndarray | None = None,
    iteration_reference: np.ndarray | None = None,
    iteration_valid: np.ndarray | None = None,
) -> ExperimentTrace:
    signal_full = np.asarray(signal, dtype=float).copy()
    reference_full = np.asarray(reference, dtype=float).copy()
    y_full = np.asarray(y, dtype=float).copy()

    lengths = [len(x_ns), len(y), len(signal), len(reference)]
    if raw_x is not None:
        lengths.append(len(raw_x))
    L = min(lengths)
    x_ns = np.asarray(x_ns, dtype=float).ravel()[:L]
    signal = signal_full[:L]
    reference = reference_full[:L]
    y = y_full[:L]
    raw_x_use = None if raw_x is None else np.asarray(raw_x, dtype=float).ravel()[:L]
    y_sem_use = None if y_sem is None else np.asarray(y_sem, dtype=float).ravel()[:L]
    contrast_iters_use = None if iteration_contrast is None else np.asarray(iteration_contrast, dtype=float)[:L]
    signal_iters_use = None if iteration_signal is None else np.asarray(iteration_signal, dtype=float)[:L]
    reference_iters_use = None if iteration_reference is None else np.asarray(iteration_reference, dtype=float)[:L]
    valid_iters_use = None if iteration_valid is None else np.asarray(iteration_valid, dtype=bool)[:L]

    valid = np.isfinite(x_ns) & np.isfinite(y)
    if y_sem_use is not None:
        valid &= np.isfinite(y_sem_use)
    x_ns = x_ns[valid]
    signal = signal[valid]
    reference = reference[valid]
    y = y[valid]
    if raw_x_use is not None:
        raw_x_use = raw_x_use[valid]
    if y_sem_use is not None:
        y_sem_use = y_sem_use[valid]
    if contrast_iters_use is not None:
        contrast_iters_use = contrast_iters_use[valid, :]
    if signal_iters_use is not None:
        signal_iters_use = signal_iters_use[valid, :]
    if reference_iters_use is not None:
        reference_iters_use = reference_iters_use[valid, :]
    if valid_iters_use is not None:
        valid_iters_use = valid_iters_use[valid, :]

    order = np.argsort(x_ns)
    x_ns = x_ns[order]
    signal = signal[order]
    reference = reference[order]
    y = y[order]
    if raw_x_use is not None:
        raw_x_use = raw_x_use[order]
    if y_sem_use is not None:
        y_sem_use = y_sem_use[order]
    if contrast_iters_use is not None:
        contrast_iters_use = contrast_iters_use[order, :]
    if signal_iters_use is not None:
        signal_iters_use = signal_iters_use[order, :]
    if reference_iters_use is not None:
        reference_iters_use = reference_iters_use[order, :]
    if valid_iters_use is not None:
        valid_iters_use = valid_iters_use[order, :]

    if len(x_ns) == 0 or len(y) == 0:
        raise ValueError(f"No valid data points remain after loading {mat_path.name} in {mode} mode.")

    scan_dim: Literal["time", "scan1d", "scan2d"] = "time"
    fit_allowed = True
    x2d = y2d = z2d = None
    signal2d = reference2d = None
    has_stage_axes = _stage_axis_count(scan_axes)

    if (has_stage_axes >= 2 or (_plot_only_scan_type(experiment_type) and len(scan_vectors) >= 2)) and len(scan_vectors) >= 2:
        ax0 = scan_vectors[0]
        ax1 = scan_vectors[1]
        nx, ny = len(ax0), len(ax1)
        L2 = nx * ny
        if len(y_full) >= L2:
            z = y_full[:L2].reshape(ny, nx)
            signal_map = signal_full[:L2].reshape(ny, nx)
            reference_map = reference_full[:L2].reshape(ny, nx)
            xv, yv = np.meshgrid(ax0, ax1)
            x2d, y2d, z2d = xv, yv, z
            signal2d, reference2d = signal_map, reference_map
            scan_dim = "scan2d"
            fit_allowed = False
            x_ns = ax0.copy()
            y = np.nanmean(z2d, axis=0)
            signal = np.nanmean(signal2d, axis=0)
            reference = np.nanmean(reference2d, axis=0)
            raw_x_use = None
            y_sem_use = None
            contrast_iters_use = None
            signal_iters_use = None
            reference_iters_use = None
            valid_iters_use = None
    elif has_stage_axes == 1 or (_plot_only_scan_type(experiment_type) and len(scan_vectors) >= 1):
        scan_dim = "scan1d"
        fit_allowed = False
        if scan_vectors:
            ax = scan_vectors[0]
            L1 = min(len(ax), len(y), len(signal), len(reference))
            x_ns = ax[:L1]
            y = y[:L1]
            signal = signal[:L1]
            reference = reference[:L1]
            raw_x_use = None if raw_x_use is None else raw_x_use[:L1]
            y_sem_use = None if y_sem_use is None else y_sem_use[:L1]

    if scan_dim == "scan2d" and experiment_type == "LineScan":
        experiment_type = "Scan2D"

    metadata_out = dict(metadata or {})
    metadata_out.setdefault("x_min_ns", float(np.min(x_ns)))
    metadata_out.setdefault("x_max_ns", float(np.max(x_ns)))
    return ExperimentTrace(
        source_path=str(mat_path.resolve()),
        file_name=mat_path.name,
        experiment_type=experiment_type,
        scan_axes=scan_axes,
        scan_dim=scan_dim,
        fit_allowed=fit_allowed,
        x_ns=x_ns,
        signal=signal,
        reference=reference,
        y=y,
        y_label=y_label,
        x_label=x_label,
        mode=mode,
        raw_x=raw_x_use,
        raw_x_label=raw_x_label,
        metadata=metadata_out,
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        y_sem=y_sem_use,
        iteration_contrast=contrast_iters_use,
        iteration_signal=signal_iters_use,
        iteration_reference=reference_iters_use,
        iteration_valid=valid_iters_use,
        signal2d=signal2d,
        reference2d=reference2d,
    )


def _label_from_parameter(parameter: str) -> str:
    p = parameter.strip().lower()
    if "duration" in p:
        return "Pulse duration [ns]"
    if "frequency" in p:
        return "RF frequency [GHz]"
    if "stage x" in p:
        return "Stage X"
    if "stage y" in p:
        return "Stage Y"
    if "stage z" in p:
        return "Stage Z"
    if not p:
        return "Scan parameter"
    return p.replace("_", " ").title()


def _candidate_axis_from_bounds(start: float, end: float, step: float, nsteps: int) -> np.ndarray:
    return start + step * np.arange(nsteps, dtype=float)


def _collect_axis_candidates(bounds_obj, step_obj, nsteps: int) -> list[tuple[float, float, float, np.ndarray]]:
    candidates: list[tuple[float, float, float, np.ndarray]] = []
    seen: set[tuple[float, float, float]] = set()
    step_arr = np.asarray(step_obj, dtype=float).ravel()

    def add_candidate(start: float, end: float, step: float):
        if not np.isfinite(step) or step <= 0:
            return
        implied = int(round((end - start) / step)) + 1
        if implied != nsteps:
            return
        key = (round(start, 9), round(end, 9), round(step, 9))
        if key in seen:
            return
        seen.add(key)
        candidates.append((start, end, step, _candidate_axis_from_bounds(start, end, step, nsteps)))

    try:
        numeric_bounds = np.asarray(bounds_obj, dtype=float).ravel()
    except Exception:
        numeric_bounds = np.asarray([], dtype=float)
    if numeric_bounds.size == 2 and step_arr.size >= 1:
        add_candidate(float(numeric_bounds[0]), float(numeric_bounds[1]), float(step_arr[0]))

    bounds_arr = np.asarray(bounds_obj, dtype=object).ravel()
    for idx, bound in enumerate(bounds_arr):
        pair = np.asarray(bound, dtype=float).ravel()
        if pair.size < 2:
            continue
        if step_arr.size == 0:
            step = np.nan
        elif idx < len(step_arr):
            step = float(step_arr[idx])
        else:
            step = float(step_arr[0])
        add_candidate(float(pair[0]), float(pair[1]), step)

    return candidates


def _extract_new_format_axes(scan_info) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, float]]]:
    nsteps = int(np.asarray(getattr(scan_info, "nSteps", 0), dtype=float).ravel()[0])
    if nsteps < 2:
        raise ValueError("scanInfo.nSteps must be at least 2")
    candidates = _collect_axis_candidates(getattr(scan_info, "bounds", []), getattr(scan_info, "stepSize", []), nsteps)

    if not candidates:
        raise ValueError("Could not derive a valid scan axis from scanInfo.bounds/stepSize")

    candidates.sort(key=lambda item: (item[2], abs(item[1] - item[0])))
    primary = candidates[0][3]
    alternate = candidates[1][3] if len(candidates) > 1 else None
    meta = [{"start": c[0], "end": c[1], "step": c[2]} for c in candidates]
    return primary, alternate, meta


def _parse_sequence_metadata(file_name: str, notes: str) -> dict[str, object]:
    combined = " ".join(part for part in [Path(file_name).stem, notes] if part)
    metadata: dict[str, object] = {}

    seq_match = re.search(r"(?<![A-Za-z0-9])((?:XY\d+|CPMG|DD)(?:-\d+)?)(?![A-Za-z0-9])", combined, flags=re.IGNORECASE)
    if seq_match:
        sequence_name = seq_match.group(1).upper()
        metadata["sequence_name"] = sequence_name
        count_match = re.search(r"-(\d+)$", sequence_name)
        if count_match:
            metadata["dd_pulse_count"] = int(count_match.group(1))

    pi_match = re.search(r"(?:π|pi)\s*:\s*([0-9.]+)\s*ns", combined, flags=re.IGNORECASE)
    if pi_match:
        metadata["pi_pulse_ns"] = float(pi_match.group(1))

    rf_match = re.search(r"rf\s*:\s*([0-9.]+)\s*ghz", combined, flags=re.IGNORECASE)
    if rf_match:
        metadata["rf_frequency_ghz"] = float(rf_match.group(1))

    return metadata


def _parse_extended_sequence_metadata(file_name: str, notes: str, params: object | None = None) -> dict[str, object]:
    combined = " ".join(part for part in [Path(file_name).stem, notes] if part)
    metadata = _parse_sequence_metadata(file_name, notes)

    seq_match = re.search(r"(?<![A-Za-z0-9])((?:XY\d+|CPMG|DD)(?:[-_]\d+)?)(?![A-Za-z0-9])", combined, flags=re.IGNORECASE)
    if seq_match:
        sequence_name = seq_match.group(1).replace("_", "-").upper()
        metadata["sequence_name"] = sequence_name
        count_match = re.search(r"-(\d+)$", sequence_name)
        if count_match:
            metadata["dd_pulse_count"] = int(count_match.group(1))

    pi_match = re.search(r"(?:π|Ï€|pi)\s*[:=]\s*([0-9.]+)\s*ns", combined, flags=re.IGNORECASE)
    if pi_match:
        metadata["pi_pulse_ns"] = float(pi_match.group(1))

    tau_match = re.search(r"(?:τ|tau)\s*=?\s*([0-9.]+)\s*ns", combined, flags=re.IGNORECASE)
    if tau_match:
        metadata["tau_ns"] = float(tau_match.group(1))

    rf2_matches = re.findall(r"rf2\s*[:=]\s*([0-9.]+)\s*(ghz|mhz|ns)?", combined, flags=re.IGNORECASE)
    if rf2_matches:
        rf2_value, rf2_unit = rf2_matches[-1]
        rf2_unit_l = rf2_unit.lower()
        if rf2_unit_l == "ns":
            metadata["rf2_duration_ns"] = float(rf2_value)
        else:
            value = float(rf2_value)
            metadata["rf2_frequency_ghz"] = value / 1000.0 if rf2_unit_l == "mhz" else value

    rf_match = re.search(r"(?<!rf2\s)rf\s*[:=]\s*([0-9.]+)\s*(ghz|mhz)?", combined, flags=re.IGNORECASE)
    if rf_match:
        value = float(rf_match.group(1))
        metadata["rf_frequency_ghz"] = value / 1000.0 if (rf_match.group(2) or "").lower() == "mhz" else value

    if params is not None:
        for attr_name, key in (
            ("piTime", "pi_pulse_ns"),
            ("tauTime", "tau_ns"),
            ("RF1ResonanceFrequency", "rf_frequency_ghz"),
            ("RFResonanceFrequency", "rf_frequency_ghz"),
            ("RFFrequency", "rf_frequency_ghz"),
            ("RF2Frequency", "rf2_frequency_ghz"),
            ("RF2Duration", "rf2_duration_ns"),
            ("nXY", "dd_pulse_count"),
        ):
            value = _safe_float_attr(params, attr_name)
            if value is not None and key not in metadata:
                metadata[key] = int(round(value)) if key == "dd_pulse_count" else value

    return metadata


def _safe_float_attr(obj, name: str) -> float | None:
    if not hasattr(obj, name):
        return None
    arr = np.asarray(getattr(obj, name), dtype=float).ravel()
    if arr.size == 0 or not np.isfinite(arr[0]):
        return None
    return float(arr[0])


def _safe_int_attr(obj, name: str) -> int | None:
    value = _safe_float_attr(obj, name)
    if value is None:
        return None
    return int(round(value))


def _safe_text_attr(obj, name: str) -> str:
    if obj is None or not hasattr(obj, name):
        return ""
    return _clean_text(getattr(obj, name))


def _reduce_new_format_values(data, nsteps: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values = np.asarray(getattr(data, "values", None), dtype=object)
    if values.ndim not in (1, 2):
        raise ValueError("data.values must be a 1D or 2D object array")
    if values.shape[0] < nsteps:
        raise ValueError("data.values has fewer rows than scanInfo.nSteps")
    values = values[:nsteps] if values.ndim == 1 else values[:nsteps, :]

    iterations = np.asarray(getattr(data, "iteration", []), dtype=float).ravel()
    iter_capacity = int(values.shape[1]) if values.ndim == 2 else 1
    failed_points = _normalize_failed_points(getattr(data, "failedPoints", np.zeros((nsteps, iter_capacity), dtype=int)), (nsteps, iter_capacity))

    signal = np.full(nsteps, np.nan, dtype=float)
    reference = np.full(nsteps, np.nan, dtype=float)
    points_with_data = 0

    for i in range(nsteps):
        if values.ndim == 1 and iterations.size > i and np.isfinite(iterations[i]) and iterations[i] <= 0:
            continue
        max_iter = iter_capacity
        if values.ndim == 2 and iterations.size > i and np.isfinite(iterations[i]) and iterations[i] > 0:
            max_iter = min(max_iter, int(iterations[i]))
        sig_vals: list[float] = []
        ref_vals: list[float] = []
        current_cells = [values[i]] if values.ndim == 1 else [values[i, j] for j in range(max_iter)]
        for j, cell in enumerate(current_cells):
            if i < failed_points.shape[0] and j < failed_points.shape[1] and int(failed_points[i, j]) != 0:
                continue
            pair = np.asarray(cell, dtype=float).ravel()
            if pair.size < 2 or not np.isfinite(pair[0]) or not np.isfinite(pair[1]):
                continue
            ref_vals.append(float(pair[0]))
            sig_vals.append(float(pair[1]))
        if sig_vals and ref_vals:
            signal[i] = float(np.mean(sig_vals))
            reference[i] = float(np.mean(ref_vals))
            points_with_data += 1

    meta = {
        "reduction": "mean_valid_iterations",
        "points_with_data": int(points_with_data),
        "iterations_per_point": int(iter_capacity),
    }
    return signal, reference, meta


def _normalize_failed_points(failed_points_obj: object, shape: tuple[int, int]) -> np.ndarray:
    failed_points = np.asarray(failed_points_obj)
    if failed_points.ndim == 2 and failed_points.shape[0] >= shape[0] and failed_points.shape[1] >= shape[1]:
        return np.asarray(failed_points[: shape[0], : shape[1]], dtype=int)
    return np.zeros(shape, dtype=int)


def _extract_complete_iteration_contrast_stats(
    data,
    nsteps: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, object]]:
    values = np.asarray(getattr(data, "values", None), dtype=object)
    if values.ndim != 2 or values.shape[0] < nsteps:
        return None, None, None, None, None, None, {"complete_iterations": 0}
    values = values[:nsteps, :]

    failed_points = _normalize_failed_points(getattr(data, "failedPoints", np.zeros(values.shape, dtype=int)), values.shape)

    contrast = np.full(values.shape, np.nan, dtype=float)
    signal = np.full(values.shape, np.nan, dtype=float)
    reference = np.full(values.shape, np.nan, dtype=float)
    valid = np.zeros(values.shape, dtype=bool)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if int(failed_points[i, j]) != 0:
                continue
            pair = np.asarray(values[i, j], dtype=float).ravel()
            if pair.size < 2 or not np.isfinite(pair[0]) or not np.isfinite(pair[1]):
                continue
            ref, sig = float(pair[0]), float(pair[1])
            reference[i, j] = ref
            signal[i, j] = sig
            valid[i, j] = True
            if ref != 0:
                contrast[i, j] = (ref - sig) / ref

    good_cols = [j for j in range(contrast.shape[1]) if np.all(np.isfinite(contrast[:, j]))]
    if not good_cols:
        meta = {
            "complete_iterations": 0,
            "valid_iteration_points": int(np.sum(valid)),
            "iteration_capacity": int(values.shape[1]),
        }
        return None, None, contrast, signal, reference, valid, meta

    contrast = contrast[:, good_cols]
    signal = signal[:, good_cols]
    reference = reference[:, good_cols]
    valid = valid[:, good_cols]
    mean = np.mean(contrast, axis=1)
    if contrast.shape[1] > 1:
        sem = np.std(contrast, axis=1, ddof=1) / np.sqrt(contrast.shape[1])
        positive = sem[np.isfinite(sem) & (sem > 0)]
        fill = float(np.median(positive)) if positive.size else 1.0
        sem = np.where(np.isfinite(sem) & (sem > 0), sem, fill)
    else:
        sem = np.full(contrast.shape[0], 1.0, dtype=float)

    meta = {
        "complete_iterations": int(contrast.shape[1]),
        "valid_iteration_points": int(np.sum(valid)),
        "iteration_capacity": int(values.shape[1]),
        "contrast_reduction": "mean_complete_iterations",
    }
    return mean, sem, contrast, signal, reference, valid, meta


def _complete_iteration_payload(
    n_points: int,
    contrast_mean: np.ndarray | None,
    contrast_iters: np.ndarray | None,
    signal_iters: np.ndarray | None,
    reference_iters: np.ndarray | None,
    valid_iters: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if contrast_mean is None or len(contrast_mean) != n_points:
        return None, None, None, None
    matrices = (contrast_iters, signal_iters, reference_iters, valid_iters)
    if any(matrix is None for matrix in matrices):
        return None, None, None, None
    normalized = tuple(np.asarray(matrix) for matrix in matrices)
    if any(matrix.ndim != 2 or matrix.shape[0] != n_points or matrix.shape[1] == 0 for matrix in normalized):
        return None, None, None, None
    return contrast_iters, signal_iters, reference_iters, valid_iters


def _as_snapshot_scan_list(scan_info) -> list:
    if isinstance(scan_info, np.ndarray):
        return [item for item in scan_info.ravel() if hasattr(item, "nSteps")]
    return [scan_info] if hasattr(scan_info, "nSteps") else []


def _snapshot_axis_descriptors(scan_info) -> tuple[tuple[str, ...], list[np.ndarray], list[dict[str, float]], str, str, str]:
    scan_list = _as_snapshot_scan_list(scan_info)
    scan_axes: list[str] = []
    scan_vectors: list[np.ndarray] = []
    axis_meta: list[dict[str, float]] = []
    parameters: list[str] = []
    identifiers: list[str] = []
    notes: list[str] = []

    for one_scan in scan_list:
        parameter = _safe_text_attr(one_scan, "parameter")
        identifier = _safe_text_attr(one_scan, "identifier")
        note = _safe_text_attr(one_scan, "notes")
        if parameter:
            parameters.append(parameter)
        if identifier:
            identifiers.append(identifier)
        if note:
            notes.append(note)
        try:
            axis, _, meta = _extract_new_format_axes(one_scan)
            scan_vectors.append(axis)
            scan_axes.append(_label_from_parameter(parameter))
            axis_meta.extend(meta)
        except Exception:
            continue

    return (
        tuple(scan_axes),
        scan_vectors,
        axis_meta,
        " | ".join(parameters),
        " | ".join(identifiers),
        " | ".join(notes),
    )


def _physical_dd_axis_if_needed(
    exp_type: str,
    axis: np.ndarray,
    x_label: str,
    metadata: dict[str, object],
) -> tuple[np.ndarray, str]:
    if exp_type != "DynamicDecoupling":
        return axis, x_label
    dd_pulse_count = metadata.get("dd_pulse_count")
    reduced_tau_half = metadata.get("reduced_tau_by_two_time_ns")
    if dd_pulse_count is None or reduced_tau_half is None:
        return axis, x_label
    raw_axis = np.asarray(axis, dtype=float)
    tau_full_actual = 2.0 * (raw_axis + float(reduced_tau_half))
    physical_axis = float(dd_pulse_count) * tau_full_actual
    metadata.setdefault("raw_x_label", x_label)
    metadata.setdefault("physical_x_label", "Total evolution time (ns)")
    metadata.setdefault("dd_axis_transform", "dd_pulse_count * 2 * (raw_x + reduced_tau_by_two_time_ns)")
    return physical_axis, "Total evolution time (ns)"


def _snapshot_series(metric_obj, expect_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str, str]:
    x = np.asarray(getattr(metric_obj, "x", []), dtype=float).ravel()
    y = np.asarray(getattr(metric_obj, "y", []), dtype=float).ravel()
    z = None
    if expect_dim == 2 and hasattr(metric_obj, "z"):
        z = np.asarray(getattr(metric_obj, "z"), dtype=float)
    return (
        x,
        y,
        z,
        _safe_text_attr(metric_obj, "xLabel"),
        _safe_text_attr(metric_obj, "yLabel"),
        _safe_text_attr(metric_obj, "zLabel") or _safe_text_attr(metric_obj, "yLabel"),
    )


def _extract_snapshot_metadata(raw: dict[str, Any]) -> dict[str, object]:
    metadata: dict[str, object] = {"schema": "snapshot_run"}
    progress = raw.get("progress")
    meta = raw.get("metadata")
    params = raw.get("params")

    if meta is not None:
        save_type = _safe_text_attr(meta, "saveType")
        run_status = _safe_text_attr(meta, "runStatus")
        scan_notes = _safe_text_attr(meta, "scanNotes")
        base_save_path = _safe_text_attr(meta, "baseSavePath")
        avg_png = _safe_text_attr(meta, "averageContrastPNG")
        detector2_avg_png = _safe_text_attr(meta, "detector2AverageContrastPNG")
        if save_type:
            metadata["save_type"] = save_type
        if run_status:
            metadata["run_status"] = run_status
        if scan_notes:
            metadata["scan_notes"] = scan_notes
        if base_save_path:
            metadata["base_save_path"] = base_save_path
        if avg_png:
            metadata["average_contrast_png"] = avg_png
        if detector2_avg_png:
            metadata["detector2_average_contrast_png"] = detector2_avg_png
        error_info = getattr(meta, "errorInfo", None)
        error_identifier = _safe_text_attr(error_info, "identifier")
        error_message = _safe_text_attr(error_info, "message")
        if error_identifier:
            metadata["acquisition_error_identifier"] = error_identifier
        if error_message:
            metadata["acquisition_error_message"] = error_message

    if progress is not None:
        progress_status = _safe_text_attr(progress, "status")
        if progress_status:
            metadata["progress_status"] = progress_status
        for attr_name, key in (
            ("completedIterations", "completed_iterations"),
            ("currentIteration", "current_iteration"),
            ("targetIterations", "target_iterations"),
            ("checkpointCount", "checkpoint_count"),
            ("odometer", "odometer"),
        ):
            value = _safe_int_attr(progress, attr_name)
            if value is not None:
                metadata[key] = value
        partial = _safe_int_attr(progress, "hasPartialIteration")
        if partial is not None:
            metadata["has_partial_iteration"] = bool(partial)

    if params is not None:
        detector_mode = _safe_text_attr(params, "detectorMode")
        if detector_mode:
            metadata["detector_mode"] = detector_mode.lower()
        for attr_name, key in (
            ("xOffset", "x_offset_ns"),
            ("dataOnBuffer", "data_on_buffer_ns"),
            ("extraBuffer", "extra_buffer_ns"),
            ("RFRampTime", "rf_ramp_time_ns"),
        ):
            value = _safe_float_attr(params, attr_name)
            if value is not None:
                metadata[key] = value
        for attr_name in (
            "rabiDeadTimeNs",
            "deadTimeNs",
            "pulseDeadTimeNs",
            "rabiDeadTime",
            "deadTime",
            "pulseDeadTime",
        ):
            value = _safe_float_attr(params, attr_name)
            if value is not None:
                metadata["rabi_dead_time_ns"] = value
                metadata["rabi_dead_time_source"] = f"params.{attr_name}"
                break

    data = raw.get("Data")
    data_detector_mode = _safe_text_attr(data, "detectorMode")
    if data_detector_mode:
        metadata["detector_mode"] = data_detector_mode.lower()

    return metadata


def _load_param_grid_stage_scan(
    mat_path: Path,
    raw: dict[str, Any],
    mode: DataMode,
    detector_id: str = "detector1",
) -> ExperimentTrace:
    mode = _canonical_mode(mode)
    data = raw["Data"]
    params = raw.get("params")
    values_attr = "detector2Values" if detector_id == "detector2" else "values"
    values = np.asarray(getattr(data, values_attr, None), dtype=object)
    if values.ndim != 2:
        raise ValueError(f"Parameter-grid fallback expects Data.{values_attr} to be a 2D object array.")
    ny, nx = values.shape

    scan_axes_raw = np.asarray(getattr(params, "scanAxes", ["x", "y"]), dtype=object).ravel().tolist() if params is not None else ["x", "y"]
    axis_names = tuple(f"Stage {str(v).strip().upper()}" for v in scan_axes_raw[:2]) if len(scan_axes_raw) >= 2 else ("Stage X", "Stage Y")
    raw_bounds = np.asarray(getattr(params, "scanBounds", []), dtype=object).ravel()
    if raw_bounds.size >= 2:
        start_pair = np.asarray(raw_bounds[0], dtype=float).ravel()
        end_pair = np.asarray(raw_bounds[1], dtype=float).ravel()
    else:
        start_pair = np.asarray([], dtype=float)
        end_pair = np.asarray([], dtype=float)
    if start_pair.size >= 2 and end_pair.size >= 2:
        x_axis = np.linspace(float(start_pair[0]), float(end_pair[0]), nx)
        y_axis = np.linspace(float(start_pair[1]), float(end_pair[1]), ny)
    else:
        x_axis = np.arange(nx, dtype=float)
        y_axis = np.arange(ny, dtype=float)

    signal = np.full((ny, nx), np.nan, dtype=float)
    reference = np.full((ny, nx), np.nan, dtype=float)
    failed_points = np.asarray(getattr(data, "failedPoints", np.zeros((ny, nx), dtype=int)))
    if failed_points.shape != (ny, nx):
        failed_points = np.zeros((ny, nx), dtype=int)
    iterations = np.asarray(getattr(data, "iteration", []), dtype=float)
    has_grid_iterations = iterations.shape == (ny, nx)
    for iy in range(ny):
        for ix in range(nx):
            if int(failed_points[iy, ix]) != 0:
                continue
            if has_grid_iterations and (not np.isfinite(iterations[iy, ix]) or iterations[iy, ix] <= 0):
                continue
            pair = np.asarray(values[iy, ix], dtype=float).ravel()
            if pair.size < 2 or not np.isfinite(pair[0]) or not np.isfinite(pair[1]):
                continue
            reference[iy, ix] = float(pair[0])
            signal[iy, ix] = float(pair[1])

    signal_flat = signal.reshape(-1)
    reference_flat = reference.reshape(-1)
    y_flat, y_label = _compose_observable(signal_flat, reference_flat, mode)
    metadata = {
        **_extract_snapshot_metadata(raw),
        "schema": "param_grid_stage_scan",
        "scan_notes": _safe_text_attr(params, "scanNotes"),
        "scan_axes": list(axis_names),
        "rf_frequency_ghz": _safe_float_attr(params, "RFFrequency"),
        "detector_id": detector_id,
    }
    return _finalize_trace(
        mat_path=mat_path,
        experiment_type="LineScan",
        scan_axes=axis_names,
        scan_vectors=[x_axis, y_axis],
        x_ns=x_axis,
        signal=signal_flat,
        reference=reference_flat,
        y=y_flat,
        y_label=y_label,
        x_label=axis_names[0],
        mode=mode,
        metadata=metadata,
    )


def _snapshot_detector_data(data: object, detector_id: str) -> SimpleNamespace:
    values_attr = "detector2Values" if detector_id == "detector2" else "values"
    if data is None or not hasattr(data, values_attr):
        raise MatDataError(
            f"Saved data does not contain {DETECTOR_LABELS.get(detector_id, detector_id)} values.",
            detector_id=detector_id,
        )
    return SimpleNamespace(
        values=getattr(data, values_attr),
        iteration=getattr(data, "iteration", []),
        failedPoints=getattr(data, "failedPoints", []),
    )


def _snapshot_acquired_cell_count(data: object) -> int:
    """Count finite raw cells that acquisition progress marks as acquired."""
    values = np.asarray(getattr(data, "values", None), dtype=object)
    if values.ndim not in (1, 2):
        return 0
    iterations = np.asarray(getattr(data, "iteration", []), dtype=float)
    failed = np.asarray(getattr(data, "failedPoints", np.zeros(values.shape, dtype=int)))
    count = 0
    for index in np.ndindex(values.shape):
        if failed.shape == values.shape and int(failed[index]) != 0:
            continue
        acquired = True
        if iterations.shape == values.shape:
            acquired = bool(np.isfinite(iterations[index]) and iterations[index] > 0)
        elif iterations.size == values.shape[0]:
            row_iteration = float(iterations.ravel()[index[0]])
            acquired = bool(np.isfinite(row_iteration) and row_iteration > 0)
            if values.ndim == 2:
                acquired = acquired and index[1] < int(row_iteration)
        elif iterations.size == 1:
            acquired = bool(np.isfinite(iterations.ravel()[0]) and iterations.ravel()[0] > 0)
        if not acquired:
            continue
        try:
            pair = np.asarray(values[index], dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if pair.size >= 2 and np.isfinite(pair[0]) and np.isfinite(pair[1]):
            count += 1
    return count


def _load_snapshot_run(
    mat_path: Path,
    raw: dict[str, Any],
    mode: DataMode,
    detector_id: str = "detector1",
) -> ExperimentTrace:
    mode = _canonical_mode(mode)
    stored_data = raw["Data"]
    data = _snapshot_detector_data(stored_data, detector_id)
    scan_info = raw["scanInfo"]
    root_analysis = raw.get("analysis")
    analysis = (
        getattr(root_analysis, "detector2", None)
        if detector_id == "detector2" and root_analysis is not None
        else root_analysis
    )
    params = raw.get("params")

    scan_axes, scan_vectors, axis_meta, parameter, identifier, notes = _snapshot_axis_descriptors(scan_info)
    metadata = _extract_snapshot_metadata(raw)
    metadata["detector_id"] = detector_id
    metadata["detector_label"] = DETECTOR_LABELS.get(detector_id, detector_id)
    if parameter:
        metadata["scan_parameter"] = parameter
    if identifier:
        metadata["scan_identifier"] = identifier
    if notes:
        metadata["notes"] = notes
    if axis_meta:
        metadata["axis_candidates"] = axis_meta
    rf_frequency = _safe_float_attr(scan_info, "RFFrequency")
    if rf_frequency is not None:
        metadata["rf_frequency_ghz"] = rf_frequency
    rf1_frequency = _safe_float_attr(scan_info, "RF1Frequency")
    if rf1_frequency is not None:
        metadata.setdefault("rf_frequency_ghz", rf1_frequency)
    rf2_frequency = _safe_float_attr(scan_info, "RF2Frequency")
    if rf2_frequency is not None:
        metadata["rf2_frequency_ghz"] = rf2_frequency
    reduced_tau = _safe_float_attr(scan_info, "reducedTauTime")
    reduced_tau_half = _safe_float_attr(scan_info, "reducedTauByTwoTime")
    if reduced_tau is not None:
        metadata["reduced_tau_time_ns"] = reduced_tau
    if reduced_tau_half is not None:
        metadata["reduced_tau_by_two_time_ns"] = reduced_tau_half
    metadata.update(_parse_extended_sequence_metadata(mat_path.name, notes or str(metadata.get("scan_notes", "")), params))

    analysis_dim = _safe_int_attr(analysis, "dimension") if analysis is not None else None
    if analysis_dim is not None:
        metadata["analysis_dimension"] = analysis_dim
    acquired_raw_points = _snapshot_acquired_cell_count(data)
    metadata["acquired_raw_points"] = acquired_raw_points
    if analysis_dim == 2 and acquired_raw_points == 0:
        run_status = str(metadata.get("run_status") or metadata.get("progress_status") or "")
        error_identifier = str(metadata.get("acquisition_error_identifier") or "")
        error_message = str(metadata.get("acquisition_error_message") or "")
        raise MatDataError(
            "No 2D cells were acquired; saved analysis arrays are not treated as measurements.",
            detector_id=detector_id,
            run_status=run_status,
            error_identifier=error_identifier,
            error_message=error_message,
        )
    (
        contrast_mean,
        contrast_sem,
        contrast_iters,
        signal_iters,
        reference_iters,
        valid_iters,
        contrast_meta,
    ) = _extract_complete_iteration_contrast_stats(data, len(scan_vectors[0]) if scan_vectors else 0)
    metadata.update(contrast_meta)

    exp_type = infer_experiment_type(
        mat_path.name,
        scan_types=scan_axes,
        notes=str(metadata.get("scan_notes", notes)),
        parameter=parameter,
        identifier=identifier,
    )

    if analysis is not None and analysis_dim in (1, 2):
        contrast_obj = getattr(analysis, "averageContrast", None)
        signal_obj = getattr(analysis, "averageSignal", None)
        reference_obj = getattr(analysis, "averageReference", None)
        if contrast_obj is not None and signal_obj is not None and reference_obj is not None:
            signal_x, signal_y, signal_z, signal_x_label, _, _ = _snapshot_series(signal_obj, analysis_dim)
            _ref_x, ref_y, ref_z, _, _, _ = _snapshot_series(reference_obj, analysis_dim)
            if mode == "contrast":
                mode_obj = contrast_obj
            elif mode == "reference":
                mode_obj = reference_obj
            else:
                mode_obj = signal_obj
            x_axis, y_axis, selected_z, x_label, axis_y_label, value_label = _snapshot_series(mode_obj, analysis_dim)
            if mode == "difference":
                if analysis_dim == 2 and signal_z is not None and ref_z is not None:
                    selected_z = ref_z - signal_z
                else:
                    selected_z = None
                y_axis = ref_y - signal_y
                series_label = "Difference [reference - signal]"
            elif mode == "reference":
                series_label = value_label or "Reference [counts]"
            elif mode == "signal":
                series_label = value_label or "Signal [counts]"
            else:
                series_label = value_label or "Contrast"

            if analysis_dim == 2 and selected_z is not None and signal_z is not None and ref_z is not None:
                metadata["analysis_source"] = "average_exports"
                return _finalize_trace(
                    mat_path=mat_path,
                    experiment_type=exp_type,
                    scan_axes=(x_label or (scan_axes[0] if len(scan_axes) > 0 else "X"), axis_y_label or (scan_axes[1] if len(scan_axes) > 1 else "Y")),
                    scan_vectors=[x_axis, y_axis],
                    x_ns=x_axis,
                    signal=signal_z.reshape(-1),
                    reference=ref_z.reshape(-1),
                    y=selected_z.reshape(-1),
                    y_label=series_label,
                    x_label=x_label or "Scan X",
                    mode=mode,
                    metadata=metadata,
                )
            if analysis_dim == 1 and len(x_axis) and len(signal_y) and len(ref_y):
                metadata["analysis_source"] = "average_exports"
                x_label_use = x_label or signal_x_label or (scan_axes[0] if scan_axes else "Scan parameter")
                raw_x_axis = x_axis.copy()
                raw_x_label_use = x_label_use
                x_axis, x_label_use = _physical_dd_axis_if_needed(exp_type, x_axis, x_label_use, metadata)
                y_series = y_axis if mode != "difference" else (ref_y - signal_y)
                y_sem_use = None
                contrast_iters_use, signal_iters_use, reference_iters_use, valid_iters_use = _complete_iteration_payload(
                    len(x_axis),
                    contrast_mean,
                    contrast_iters,
                    signal_iters,
                    reference_iters,
                    valid_iters,
                )
                if mode == "contrast" and contrast_mean is not None and len(contrast_mean) == len(x_axis):
                    y_series = contrast_mean
                    y_sem_use = contrast_sem
                return _finalize_trace(
                    mat_path=mat_path,
                    experiment_type=exp_type,
                    scan_axes=(x_label_use,),
                    scan_vectors=[x_axis],
                    x_ns=x_axis,
                    signal=signal_y,
                    reference=ref_y,
                    y=y_series,
                    y_label=series_label,
                    x_label=x_label_use,
                    mode=mode,
                    raw_x=raw_x_axis,
                    raw_x_label=raw_x_label_use,
                    metadata=metadata,
                    y_sem=y_sem_use,
                    iteration_contrast=contrast_iters_use,
                    iteration_signal=signal_iters_use,
                    iteration_reference=reference_iters_use,
                    iteration_valid=valid_iters_use,
                )

    if not scan_vectors:
        raise ValueError("Snapshot MAT file did not contain usable analysis exports or scan axes.")

    nsteps = int(np.prod([len(v) for v in scan_vectors], dtype=int)) if len(scan_vectors) > 1 else len(scan_vectors[0])
    values = np.asarray(getattr(data, "values", None), dtype=object)
    if len(scan_vectors) >= 2 and values.ndim == 2:
        return _load_param_grid_stage_scan(mat_path, raw, mode, detector_id=detector_id)
    signal, reference, reduction_meta = _reduce_new_format_values(data, nsteps)
    metadata.update(reduction_meta)
    y, y_label = _compose_observable(signal, reference, mode)
    y_sem_use = contrast_sem if mode == "contrast" and contrast_mean is not None and len(contrast_mean) == len(y) else None
    contrast_iters_use, signal_iters_use, reference_iters_use, valid_iters_use = _complete_iteration_payload(
        len(y),
        contrast_mean,
        contrast_iters,
        signal_iters,
        reference_iters,
        valid_iters,
    )
    if mode == "contrast" and contrast_mean is not None and len(contrast_mean) == len(y):
        y = contrast_mean
    raw_x = scan_vectors[0]
    x_ns = raw_x if len(scan_vectors) == 1 else np.arange(nsteps, dtype=float)
    x_label = scan_axes[0] if scan_axes else _label_from_parameter(parameter)
    raw_x_for_trace = raw_x if len(scan_vectors) == 1 else None
    raw_x_label_for_trace = x_label if len(scan_vectors) == 1 else None
    if len(scan_vectors) == 1:
        x_ns, x_label = _physical_dd_axis_if_needed(exp_type, x_ns, x_label, metadata)
    return _finalize_trace(
        mat_path=mat_path,
        experiment_type=exp_type,
        scan_axes=scan_axes if scan_axes else (x_label,),
        scan_vectors=scan_vectors,
        x_ns=x_ns,
        signal=signal,
        reference=reference,
        y=y,
        y_label=y_label,
        x_label=x_label,
        mode=mode,
        raw_x=raw_x_for_trace,
        raw_x_label=raw_x_label_for_trace,
        metadata=metadata,
        y_sem=y_sem_use,
        iteration_contrast=contrast_iters_use,
        iteration_signal=signal_iters_use,
        iteration_reference=reference_iters_use,
        iteration_valid=valid_iters_use,
    )


def _load_legacy_saved_data(mat_path: Path, raw: dict[str, Any], mode: DataMode) -> ExperimentTrace:
    mode = _canonical_mode(mode)
    sd = raw["savedData"]
    if not hasattr(sd, "loadedScan"):
        raise ValueError("savedData.loadedScan not found.")

    scans = _as_scan_list(sd.loadedScan)
    scan_axes_list: list[str] = []
    scan_vectors: list[np.ndarray] = []
    for s in scans:
        try:
            axis_name, axis_vec = _scan_axis_info(s)
            scan_axes_list.append(axis_name)
            scan_vectors.append(axis_vec)
        except Exception:
            continue
    scan_axes = tuple(scan_axes_list)

    start_ns, end_ns, nsteps = _pick_scan_axis(sd.loadedScan)
    if nsteps < 2:
        raise ValueError("Need at least 2 points to build x-axis.")
    x_ns = np.linspace(start_ns, end_ns, nsteps)

    signal, reference = _extract_vectors(sd)
    y, y_label = _compose_observable(signal, reference, mode)
    exp_type = infer_experiment_type(mat_path.name, scan_types=scan_axes)
    return _finalize_trace(
        mat_path=mat_path,
        experiment_type=exp_type,
        scan_axes=scan_axes,
        scan_vectors=scan_vectors,
        x_ns=x_ns,
        signal=signal,
        reference=reference,
        y=y,
        y_label=y_label,
        x_label=(scan_axes[0] if scan_axes else "Pulse duration (ns)"),
        mode=mode,
        metadata={"schema": "savedData"},
    )


def _load_new_data_scaninfo(mat_path: Path, raw: dict[str, Any], mode: DataMode) -> ExperimentTrace:
    mode = _canonical_mode(mode)
    data = raw["data"]
    scan_info = raw["scanInfo"]
    parameter = _clean_text(getattr(scan_info, "parameter", ""))
    identifier = _clean_text(getattr(scan_info, "identifier", ""))
    notes = _clean_text(getattr(scan_info, "notes", ""))
    raw_x, alternate_x, axis_meta = _extract_new_format_axes(scan_info)
    signal, reference, reduction_meta = _reduce_new_format_values(data, len(raw_x))
    (
        contrast_mean,
        contrast_sem,
        contrast_iters,
        signal_iters,
        reference_iters,
        valid_iters,
        contrast_meta,
    ) = _extract_complete_iteration_contrast_stats(data, len(raw_x))
    y, y_label = _compose_observable(signal, reference, mode)

    raw_x_label = _label_from_parameter(parameter)
    scan_axes = (raw_x_label,)
    scan_vectors = [raw_x]
    metadata: dict[str, object] = {
        "schema": "data_scanInfo",
        "scan_parameter": parameter,
        "scan_identifier": identifier,
        "notes": notes,
        "axis_candidates": axis_meta,
        **reduction_meta,
        **contrast_meta,
    }
    rf_frequency = _safe_float_attr(scan_info, "RFFrequency")
    if rf_frequency is not None:
        metadata["rf_frequency_ghz"] = rf_frequency
    reduced_tau = _safe_float_attr(scan_info, "reducedTauTime")
    reduced_tau_half = _safe_float_attr(scan_info, "reducedTauByTwoTime")
    if reduced_tau is not None:
        metadata["reduced_tau_time_ns"] = reduced_tau
    if reduced_tau_half is not None:
        metadata["reduced_tau_by_two_time_ns"] = reduced_tau_half
    metadata.update(_parse_extended_sequence_metadata(mat_path.name, notes, raw.get("params")))

    exp_type = infer_experiment_type(
        mat_path.name,
        scan_types=scan_axes,
        notes=notes,
        parameter=parameter,
        identifier=identifier,
    )
    x_ns = raw_x.copy()
    x_label = raw_x_label

    if exp_type == "DynamicDecoupling":
        dd_pulse_count = metadata.get("dd_pulse_count")
        if dd_pulse_count is None:
            raise ValueError("Dynamic-decoupling trace detected but DD pulse count could not be inferred.")
        if reduced_tau_half is None:
            raise ValueError("Dynamic-decoupling trace detected but reducedTauByTwoTime is missing.")
        tau_half_actual = raw_x + reduced_tau_half
        tau_full_actual = 2.0 * tau_half_actual
        x_ns = float(dd_pulse_count) * tau_full_actual
        x_label = "Total evolution time (ns)"
        metadata["raw_x_label"] = raw_x_label
        metadata["physical_x_label"] = x_label
        if alternate_x is not None and reduced_tau is not None:
            alt_full = alternate_x + reduced_tau
            if not np.allclose(alt_full, tau_full_actual, rtol=0.0, atol=1e-6):
                warnings = list(metadata.get("warnings", [])) if isinstance(metadata.get("warnings"), list) else []
                warnings.append("Alternate full-delay axis disagreed with reducedTauByTwoTime-derived axis; using reducedTauByTwoTime path.")
                metadata["warnings"] = warnings

    if exp_type == "LineScan":
        if _keyword_match(parameter, ("stage x", "stage y", "stage z")) or _keyword_match(identifier, ("stage",)):
            scan_vectors = [raw_x]
        else:
            metadata["warnings"] = list(metadata.get("warnings", [])) + ["Ambiguous new-format scan classified as LineScan and kept plot-only."]

    contrast_iters_use, signal_iters_use, reference_iters_use, valid_iters_use = _complete_iteration_payload(
        len(y),
        contrast_mean,
        contrast_iters,
        signal_iters,
        reference_iters,
        valid_iters,
    )
    return _finalize_trace(
        mat_path=mat_path,
        experiment_type=exp_type,
        scan_axes=scan_axes,
        scan_vectors=scan_vectors,
        x_ns=x_ns,
        signal=signal,
        reference=reference,
        y=(contrast_mean if mode == "contrast" and contrast_mean is not None and len(contrast_mean) == len(y) else y),
        y_label=y_label,
        x_label=x_label,
        mode=mode,
        raw_x=raw_x,
        raw_x_label=raw_x_label,
        metadata=metadata,
        y_sem=(contrast_sem if mode == "contrast" and contrast_mean is not None and len(contrast_mean) == len(y) else None),
        iteration_contrast=contrast_iters_use,
        iteration_signal=signal_iters_use,
        iteration_reference=reference_iters_use,
        iteration_valid=valid_iters_use,
    )


def _canonical_detector_id(detector_id: str | int) -> str:
    text = str(detector_id).strip().lower().replace("_", "").replace(" ", "")
    aliases = {
        "1": "detector1",
        "detector1": "detector1",
        "primary": "detector1",
        "2": "detector2",
        "detector2": "detector2",
        "secondary": "detector2",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported detector id: {detector_id}")
    return aliases[text]


def _snapshot_detector_ids(raw: dict[str, Any]) -> tuple[str, ...]:
    data = raw.get("Data")
    params = raw.get("params")
    analysis = raw.get("analysis")
    detector_mode = _safe_text_attr(data, "detectorMode") or _safe_text_attr(params, "detectorMode")
    has_detector2 = bool(
        (data is not None and hasattr(data, "detector2Values"))
        or (analysis is not None and hasattr(analysis, "detector2"))
        or detector_mode.lower() == "dual"
    )
    return ("detector1", "detector2") if has_detector2 else ("detector1",)


def _attach_detector_context(
    trace: ExperimentTrace,
    detector_id: str,
    available_detectors: tuple[str, ...],
) -> ExperimentTrace:
    trace.detector_id = detector_id
    trace.detector_label = DETECTOR_LABELS.get(detector_id, detector_id)
    trace.available_detectors = available_detectors
    metadata = dict(trace.metadata or {})
    metadata.update(
        {
            "detector_id": detector_id,
            "detector_label": trace.detector_label,
            "available_detectors": list(available_detectors),
            "detector_count": len(available_detectors),
            "detector_mode": "dual" if len(available_detectors) > 1 else "single",
        }
    )
    trace.metadata = metadata
    return trace


def _detector_failure(
    detector_id: str,
    exc: Exception,
    metadata: dict[str, object],
) -> DetectorLoadFailure:
    run_status = str(getattr(exc, "run_status", "") or metadata.get("run_status") or metadata.get("progress_status") or "")
    error_identifier = str(getattr(exc, "error_identifier", "") or metadata.get("acquisition_error_identifier") or "")
    error_message = str(getattr(exc, "error_message", "") or metadata.get("acquisition_error_message") or "")
    label = DETECTOR_LABELS.get(detector_id, detector_id)
    detail = str(exc).strip() or type(exc).__name__
    context: list[str] = []
    if run_status:
        context.append(f"run_status={run_status}")
    if error_identifier:
        context.append(error_identifier)
    if error_message:
        context.append(error_message)
    message = f"{label}: {detail}"
    if context:
        message += " Acquisition details: " + " | ".join(context)
    return DetectorLoadFailure(
        detector_id=detector_id,
        detector_label=label,
        message=message,
        run_status=run_status,
        error_identifier=error_identifier,
        error_message=error_message,
    )


def load_experiment_dataset(path: str | Path, mode: DataMode = "contrast") -> ExperimentDataset:
    mat_path = Path(path)
    mode = _canonical_mode(mode)
    raw = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    metadata: dict[str, object] = {}
    available_detectors: tuple[str, ...] = ("detector1",)
    detector_mode = "single"
    loaders: dict[str, Any] = {}

    if "savedData" in raw:
        loaders["detector1"] = lambda: _load_legacy_saved_data(mat_path, raw, mode)
    elif "data" in raw and "scanInfo" in raw:
        loaders["detector1"] = lambda: _load_new_data_scaninfo(mat_path, raw, mode)
    elif "Data" in raw and "scanInfo" in raw:
        metadata = _extract_snapshot_metadata(raw)
        available_detectors = _snapshot_detector_ids(raw)
        detector_mode = "dual" if len(available_detectors) > 1 else "single"
        for detector_id in available_detectors:
            loaders[detector_id] = lambda detector_id=detector_id: _load_snapshot_run(
                mat_path,
                raw,
                mode,
                detector_id=detector_id,
            )
    else:
        raise ValueError("No supported MATLAB root structure found. Expected 'savedData', 'data'+'scanInfo', or snapshot 'Data'+'scanInfo'.")

    detectors: dict[str, ExperimentTrace] = {}
    detector_errors: dict[str, DetectorLoadFailure] = {}
    for detector_id in available_detectors:
        try:
            detectors[detector_id] = _attach_detector_context(
                loaders[detector_id](),
                detector_id,
                available_detectors,
            )
        except Exception as exc:
            detector_errors[detector_id] = _detector_failure(detector_id, exc, metadata)

    metadata = {
        **metadata,
        "detector_mode": detector_mode,
        "available_detectors": list(available_detectors),
        "detector_count": len(available_detectors),
    }
    return ExperimentDataset(
        source_path=str(mat_path.resolve()),
        file_name=mat_path.name,
        detector_mode=detector_mode,
        available_detectors=available_detectors,
        detectors=detectors,
        detector_errors=detector_errors,
        metadata=metadata,
    )


def load_saved_data_mat(
    path: str | Path,
    mode: DataMode = "contrast",
    detector_id: str | int = "detector1",
) -> ExperimentTrace:
    requested = _canonical_detector_id(detector_id)
    dataset = load_experiment_dataset(path, mode=mode)
    if requested not in dataset.available_detectors:
        raise MatDataError(
            f"{DETECTOR_LABELS.get(requested, requested)} is not available in {dataset.file_name}.",
            detector_id=requested,
        )
    if requested in dataset.detectors:
        return dataset.detectors[requested]
    failure = dataset.detector_errors.get(requested)
    if failure is None:
        raise MatDataError(
            f"{DETECTOR_LABELS.get(requested, requested)} could not be loaded from {dataset.file_name}.",
            detector_id=requested,
        )
    raise MatDataError(
        failure.message,
        detector_id=failure.detector_id,
        run_status=failure.run_status,
        error_identifier=failure.error_identifier,
        error_message=failure.error_message,
    )
