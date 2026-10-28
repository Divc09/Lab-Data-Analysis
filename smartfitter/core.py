"""Analysis backend for SmartFitter v2.

Everything scientific is delegated to the validated ``nvfit`` package (loaders,
processing, fit workflows, NV metrics, spot detection, map overlays, batch).
This module only adapts those results into JSON payloads for the interface.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import math
import hashlib
import uuid
from copy import deepcopy
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from nvfit.analysis_tools import AnalysisStep, process_series
from nvfit.app_state import fit_result_to_dict, json_safe
from nvfit.fit_engine import FitResult
from nvfit.prepared_fit import PreparedFit, evaluate_result
from nvfit.analysis_identity import analysis_descriptor, result_envelope, source_path
from nvfit.units import compatibility, trace_units, unit_scale
from nvfit.fit_workflows import FitWorkflowConfig, aligned_sem, fit_non_ramsey
from nvfit.io_mat import DETECTOR_LABELS, ExperimentDataset, ExperimentTrace, load_experiment_dataset
from nvfit.iterations import iteration_matrix, selected_trace, selection_indices, iteration_sources, matrix_stats, combine_datasets
from nvfit.map_comparison import axis_edges, axis_key, build_map_overlay, compatible_axes
from nvfit.models import MODEL_REGISTRY
from nvfit.preprocess import smooth_trace
from nvfit.profiles import PROFILES, classify_status
from nvfit.rabi_utils import rabi_envelope_bounds
from nvfit.ramsey import fit_ramsey_physics_first
from nvfit.scan_spots import detect_fluorescent_spots
from nvfit import pro_batch

from .params import param_meta

APP_VERSION = "2.0"
SETTINGS_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "SmartFitter"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"
PEEK_CACHE_NAME = "peek_cache.json"
PEEK_CACHE_MAX = 5000
MODES = ("contrast", "signal", "reference", "difference")
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def clean(value: Any) -> Any:
    """JSON-safe conversion that maps NaN/inf to None (JSON has no NaN)."""
    value = json_safe(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def arr(values: Any) -> list | None:
    if values is None:
        return None
    a = np.asarray(values, dtype=float)
    out = a.tolist()
    if a.ndim == 1:
        return [v if math.isfinite(v) else None for v in out]
    return clean(out)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS: dict[str, Any] = {
    "recent_folders": [],
    "recent_files": [],
    "watch_folders": [],
    "last_dir": "",
    "plot_style": {},          # user default presentation (Plot editor "Save as my default")
    "fit_presets": {},
    "swap_2d_axes": True,       # matches the original app's default
    "pixel_geometry": "physical",
    "colormap": "viridis",
    "detector_labels": {"detector1": "Detector 1", "detector2": "Detector 2"},
}


def load_settings() -> dict[str, Any]:
    data = dict(DEFAULT_SETTINGS)
    try:
        data.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return data


_SETTINGS_LOCK = threading.RLock()


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with _SETTINGS_LOCK:
        data = load_settings()
        data.update(patch or {})
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(clean(data), indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS_PATH)
        return data


def remember(kind: str, path: str, limit: int = 15) -> None:
    with _SETTINGS_LOCK:
        settings = load_settings()
        items = [p for p in settings.get(kind, []) if os.path.normcase(p) != os.path.normcase(path)]
        save_settings({kind: [path, *items][:limit]})


# ---------------------------------------------------------------------------
# Loading / classification
# ---------------------------------------------------------------------------

TYPE_TAGS = {
    "Rabi": "RABI", "Ramsey": "RAMSEY", "SpinEcho": "ECHO", "DynamicDecoupling": "DD", "T1": "T1",
    "ODMR": "ODMR", "DEERFrequency": "DEER", "DEERDuration": "DEER", "DEERDecay": "DEER",
    "DEERPosition": "DEER", "LineScan": "Z", "Scan2D": "XY", "TimeTrace": "TT",
}


class Workspace:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], tuple[int, ExperimentDataset]] = {}
        self._peek: dict[str, tuple[int, dict]] = {}
        self._disk: dict[str, dict] | None = None  # persistent peek cache, loaded lazily
        self._disk_dirty = False
        self._disk_saved = -1e9
        self._pool = ThreadPoolExecutor(max_workers=4)
        self.jobs: dict[str, dict] = {}
        self.combined: dict[str, dict] = {}
        self._revisions: dict[str, tuple[tuple, dict]] = {}

    def source_revision(self, path: str) -> dict:
        if path.startswith('combined:'):
            recipe = self.combined[path]
            return {'recipe':recipe,'sources':{r['path']:self.source_revision(r['path']) for r in recipe['runs']}}
        p = Path(path)
        if not p.exists() and type(self).dataset is not Workspace.dataset:
            # Explicit in-memory dataset providers have no filesystem timestamp.
            dataset=self.dataset(path)
            digest=hashlib.sha256();size=0
            for detector,tr in sorted(dataset.detectors.items()):
                digest.update(detector.encode())
                for name in ('x_ns','y','iteration_contrast','iteration_signal','iteration_reference','z2d'):
                    value=getattr(tr,name,None)
                    if value is None:continue
                    value=np.asarray(value,dtype=float);digest.update(str(value.shape).encode());digest.update(value.tobytes());size+=value.nbytes
            return {'sha256':digest.hexdigest(),'size':size,'virtual':True}
        stat = p.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        key = source_path(str(p.resolve()))
        with self._lock:
            hit = self._revisions.get(key)
            if hit and hit[0] == stamp: return hit[1]
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        after = p.stat()
        if (after.st_mtime_ns,after.st_size) != stamp:
            raise ValueError(f'Source changed while reading: {p.name}; retry after acquisition finishes saving.')
        revision = {'mtime_ns':str(stat.st_mtime_ns),'size':stat.st_size,'sha256':digest.hexdigest()}
        with self._lock: self._revisions[key] = (stamp,revision)
        return revision

    # -- datasets ---------------------------------------------------------
    def dataset(self, path: str, mode: str = "contrast") -> ExperimentDataset:
        if path.startswith("combined:"):
            recipe = self.combined.get(path)
            if recipe is None:
                raise ValueError("Combined dataset is not registered. Reopen its saved analysis session.")
            return combine_datasets(self.dataset, recipe, mode)
        p = str(Path(path))
        mtime = Path(p).stat().st_mtime_ns
        key = (os.path.normcase(p), mode)
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] == mtime:
                return hit[1]
        ds = load_experiment_dataset(p, mode)
        if Path(p).stat().st_mtime_ns != mtime:
            raise ValueError(f'Source changed while loading: {Path(p).name}; retry after saving.')
        with self._lock:
            self._cache[key] = (mtime, ds)
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return ds

    def trace(self, path: str, detector: str | None, mode: str) -> ExperimentTrace:
        ds = self.dataset(path, mode)
        if detector:
            # An explicit detector must never be answered with another channel's data.
            if detector in ds.detectors:
                return ds.detectors[detector]
            failure = ds.detector_errors.get(detector)
            label = DETECTOR_LABELS.get(detector, detector)
            raise ValueError(failure.message if failure else f"{label} is not available in {ds.file_name}.")
        det = next((d for d in ds.available_detectors if d in ds.detectors), None)
        if det is None:
            failure = next(iter(ds.detector_errors.values()), None)
            raise ValueError(failure.message if failure else "No detector data could be loaded.")
        return ds.detectors[det]

    def peek(self, path: str) -> dict:
        """Lightweight description used by the library list (cached by mtime)."""
        if path.startswith("combined:"):
            ds = self.dataset(path)
            return describe(ds, next(iter(ds.detectors.values())))
        p = Path(path)
        try:
            mtime = p.stat().st_mtime_ns
        except OSError as exc:
            return {"path": str(p), "error": str(exc)}
        key = os.path.normcase(str(p))
        hit = self._peek.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
        size = p.stat().st_size
        with self._lock:
            rec = self._disk_cache().get(key)
        if rec and rec.get("mtime_ns") == mtime and rec.get("size") == size and isinstance(rec.get("info"), dict):
            self._peek[key] = (mtime, rec["info"])
            return rec["info"]
        info: dict[str, Any] = {"path": str(p), "name": p.name, "mtime": mtime / 1e9, "size": p.stat().st_size}
        try:
            ds = self.dataset(str(p), "contrast")
            tr = next(iter(ds.detectors.values()))
            meta = tr.metadata or {}
            info.update({
                "type": tr.experiment_type,
                "tag": TYPE_TAGS.get(tr.experiment_type, tr.experiment_type[:4].upper()),
                "dim": tr.scan_dim,
                "detectors": len(ds.available_detectors),
                "points": int(tr.z2d.size if tr.scan_dim == "scan2d" and tr.z2d is not None else len(tr.x_ns)),
                "grid": list(tr.z2d.shape) if tr.z2d is not None and tr.scan_dim == "scan2d" else None,
                "axes": list(tr.scan_axes),
                "run_status": str(meta.get("run_status") or meta.get("save_type") or ""),
                "iterations": f"{meta.get('completed_iterations')}/{meta.get('target_iterations')}" if meta.get("target_iterations") else "",
                "reference_only": bool(meta.get("reference_only")),
            })
        except Exception as exc:  # unreadable/unsupported file: keep it visible with the reason
            info.update({"type": "Unreadable", "tag": "?", "dim": "", "error": str(exc)[:300]})
        self._peek[key] = (mtime, info)
        with self._lock:
            cache = self._disk_cache()
            cache.pop(key, None)
            cache[key] = {"mtime_ns": mtime, "size": size, "info": json.loads(json.dumps(clean(info)))}
            while len(cache) > PEEK_CACHE_MAX:
                cache.pop(next(iter(cache)))
            self._disk_dirty = True
        return info

    def peek_many(self, paths: list[str]) -> list[dict]:
        out = list(self._pool.map(self.peek, paths))
        self._flush_peek_cache()
        return out

    def _disk_cache(self) -> dict[str, dict]:
        if self._disk is None:  # caller holds self._lock
            try:
                data = json.loads((SETTINGS_DIR / PEEK_CACHE_NAME).read_text(encoding="utf-8"))
                self._disk = data if isinstance(data, dict) else {}
            except Exception:
                self._disk = {}
        return self._disk

    def _flush_peek_cache(self, min_interval: float = 2.0) -> None:
        with self._lock:
            if not self._disk_dirty or self._disk is None or time.monotonic() - self._disk_saved < min_interval:
                return
            try:
                SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
                path = SETTINGS_DIR / PEEK_CACHE_NAME
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._disk), encoding="utf-8")
                os.replace(tmp, path)
                self._disk_dirty = False
            except Exception:
                pass
            self._disk_saved = time.monotonic()


def list_dir(path: str) -> dict:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Not a folder: {path}")
    folders, files = [], []
    for entry in sorted(os.scandir(root), key=lambda e: e.name.lower()):
        try:
            if entry.is_dir() and not entry.name.startswith((".", "__")):
                folders.append({"name": entry.name, "path": entry.path})
            elif entry.is_file() and entry.name.lower().endswith(".mat"):
                st = entry.stat()
                files.append({"name": entry.name, "path": entry.path, "mtime": st.st_mtime, "size": st.st_size})
        except OSError:
            continue
    parent = str(root.parent) if root.parent != root else None
    return {"path": str(root), "name": root.name or str(root), "parent": parent, "folders": folders, "files": files}


# ---------------------------------------------------------------------------
# Trace payloads
# ---------------------------------------------------------------------------

def _iteration_matrix(tr: ExperimentTrace, mode: str) -> np.ndarray | None:
    return iteration_matrix(tr, mode)


def reference_only(tr: ExperimentTrace) -> bool:
    meta = tr.metadata or {}
    if meta.get("reference_only"):
        return True
    sig, ref = np.asarray(tr.signal, float), np.asarray(tr.reference, float)
    if sig.size and np.all(~np.isfinite(sig) | (sig == 0)) and np.any(np.isfinite(ref) & (ref != 0)):
        return True
    return False


def describe(ds: ExperimentDataset, tr: ExperimentTrace) -> dict:
    meta = tr.metadata or {}
    ref_only = reference_only(tr)
    observables = list(meta.get("available_observables") or (["reference"] if ref_only else list(MODES)))
    timing = meta.get("time_trace") if isinstance(meta.get("time_trace"), dict) else {}
    return clean({
        "path": ds.source_path,
        "name": ds.file_name,
        "type": tr.experiment_type,
        "tag": TYPE_TAGS.get(tr.experiment_type, tr.experiment_type[:4].upper()),
        "dim": tr.scan_dim,
        "fit_allowed": bool(tr.fit_allowed) or tr.experiment_type == "TimeTrace",
        "axes": list(tr.scan_axes),
        "x_label": tr.x_label,
        "raw_x_label": tr.raw_x_label,
        "x_unit": meta.get("x_unit"),
        "axis_units": meta.get("axis_units") if isinstance(meta.get("axis_units"), dict) else None,
        "y_label": tr.y_label,
        "detectors": list(ds.available_detectors),
        "detector_labels": {k: v.detector_label for k, v in ds.detectors.items()},
        "detector_errors": {k: v.message for k, v in ds.detector_errors.items()},
        "detector_mode": ds.detector_mode,
        "reference_only": ref_only,
        "observables": observables,
        "unit": meta.get("value_unit", ""),
        "combined": meta.get("combined"),
        "combined_recipe": ds.metadata.get("combined_recipe"),
        "is_time_trace": bool(meta.get("is_time_trace")),
        "irregular": bool(meta.get("time_sampling_irregular")),
        "timing": {k: timing.get(k) for k in ("timing", "durationSeconds", "cadenceSeconds", "checkpointSeconds", "sourceExperiment", "fixedValue", "events", "failedPoints", "scheduledSeconds", "latenessSeconds", "nPoints")} if timing else None,
        "metadata": {k: (v if isinstance(v, (int, float, bool)) or v is None else str(v)[:400]) for k, v in meta.items() if k not in ("time_trace", "source_point_indices")},
    })


def smoothing_window(n: int, requested: int) -> int:
    """Effective window used by nvfit.preprocess.smooth_trace."""
    w = max(1, int(requested))
    if w <= 1:
        return 1
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if n % 2 == 0 else n
    return w if w >= 3 else 1


def analysis_for(ws, path, detector, mode, proc=None, settings=None, kind='fit'):
    revision = ws.source_revision(path) if hasattr(ws,'source_revision') else None
    dependencies = {}
    baseline = (proc or {}).get('baseline') or {}
    if baseline.get('path') and hasattr(ws,'source_revision'):
        dependencies[source_path(baseline['path'])] = ws.source_revision(baseline['path'])
    settings = {k:v for k,v in (settings or {}).items() if k != 'apply_both'}
    return result_envelope(analysis_descriptor(path,detector,mode,proc),revision,settings,dependencies,kind)


def trace_payload(ws: Workspace, path: str, detector: str | None, mode: str, proc: dict | None = None) -> dict:
    ds = ws.dataset(path, mode)
    tr = ws.trace(path, detector, mode)
    proc = proc or {}
    info = describe(ds, tr)
    analysis = analysis_for(ws,path,tr.detector_id,mode,proc,kind='data')
    info['source_revision'] = analysis['source_revision']
    if info['source_revision'] and 'mtime_ns' in info['source_revision']:
        info['mtime'] = int(info['source_revision']['mtime_ns']) / 1e9
    out: dict[str, Any] = {"info": info, "detector": tr.detector_id, "mode": mode}
    out['analysis'] = analysis
    if tr.scan_dim == "scan2d" and tr.z2d is not None:
        x2d, y2d = np.asarray(tr.x2d, float), np.asarray(tr.y2d, float)
        maps = {"value": tr.z2d}
        if tr.signal2d is not None:
            maps["signal"] = tr.signal2d
        if tr.reference2d is not None:
            maps["reference"] = tr.reference2d
        out["map"] = {
            "x": arr(x2d[0, :]), "y": arr(y2d[:, 0]),
            "x_label": tr.scan_axes[0] if len(tr.scan_axes) > 0 else "X",
            "y_label": tr.scan_axes[1] if len(tr.scan_axes) > 1 else "Y",
            "z": {k: arr(v) for k, v in maps.items()},
            "label": tr.y_label,
        }
        return clean(out)

    original = tr
    tr = selected_trace(tr, mode, proc.get("iterations"))
    x = np.asarray(tr.x_ns, float)
    y = np.asarray(tr.y, float)
    its = _iteration_matrix(original, mode)
    indices = selection_indices(its.shape[1], proc.get("iterations")) if its is not None else []
    plot_its = its
    base = proc.get("baseline") or {}
    if base.get("path") and os.path.normcase(base["path"]) != os.path.normcase(path):
        # Subtract another loaded file (same observable), interpolated onto this file's axis.
        btr = ws.trace(base["path"], base.get("detector") or detector, mode)
        xscale, yscale = compatibility(trace_units(tr),trace_units(btr),label='Baseline subtraction')
        bx, by = np.asarray(btr.x_ns, float), np.asarray(btr.y, float)
        bx, by = bx*xscale, by*yscale
        ok = np.isfinite(bx) & np.isfinite(by)
        order = np.argsort(bx[ok])
        bvals = np.interp(x, bx[ok][order], by[ok][order], left=np.nan, right=np.nan)
        y = y - bvals
        if its is not None:
            plot_its = its - bvals[:, None]
        out["baseline_note"] = f"Subtracted {btr.file_name} ({btr.detector_label}); points outside its range are dropped."
    steps = [AnalysisStep.from_dict(s) for s in proc.get("steps", []) if s.get("enabled", True)]
    ps = process_series(
        x, y,
        bin_size=int(proc.get("bin", 1) or 1),
        roi=(proc.get("roi_min"), proc.get("roi_max")),
        excluded_raw_indices=set(int(i) for i in proc.get("excluded", [])),
        exclusion_ranges=[tuple(r) for r in proc.get("exclusion_ranges", [])],
        steps=steps,
    )
    smooth_w = int(proc.get("smooth", 1) or 1)
    sem = np.asarray(tr.y_sem, float) if tr.y_sem is not None else None
    proc_sem, proc_sd, processed_its = None, None, None
    if its is not None and not steps:
        # Average within each bin *per iteration*, then estimate variation across iterations.
        # This retains within-iteration point covariance instead of assuming independent x points.
        processed_its = np.array([np.mean(plot_its[g], axis=0) for g in ps.source_groups]).reshape(-1, its.shape[1])
        _, proc_sd, estimated_sem, _ = matrix_stats(processed_its[:, indices])
        proc_sem = estimated_sem if len(indices) > 1 else None
    elif sem is not None and len(ps.source_groups) == len(ps.analysis_y) and not steps:
        proc_sem = np.array([math.sqrt(float(np.nansum(sem[g] ** 2))) / max(1, len(g)) for g in ps.source_groups])
    effective_window = smoothing_window(len(ps.analysis_y), smooth_w)
    provenance = {"selected_iterations": [i + 1 for i in indices], "iteration_count": len(indices),
                  "iteration_sources": iteration_sources(tr) if its is not None else [],
                  "combined": info.get("combined"), "sem": "sample SD (ddof=1) / sqrt(n); undefined for n<2",
                  "smoothing": {"method": "Savitzky-Golay", "requested_window": smooth_w, "window": effective_window, "polyorder": min(2, effective_window - 1)}}
    out["provenance"] = provenance
    out.update({
        "raw": {"x": arr(x), "y": arr(y), "signal": arr(tr.signal), "reference": arr(tr.reference), "sem": arr(sem),
                "raw_x": arr(tr.raw_x) if tr.raw_x is not None else None,
                "trace_y": arr(tr.y) if out.get("baseline_note") else None},
        "proc": {"x": arr(ps.analysis_x), "y": arr(ps.analysis_y), "fit_x": arr(ps.fit_x), "fit_y": arr(ps.fit_y),
                 "groups": [g.tolist() for g in ps.source_groups], "sem": arr(proc_sem), "sd": arr(proc_sd),
                 "smooth": arr(smooth_trace(ps.analysis_y, smooth_w)) if effective_window > 1 else None,
                 "smoothing_window": effective_window, "provenance": ps.provenance},
        "iterations": {"count": int(its.shape[1]), "selected": indices, "values": arr(its.T),
                       "x": arr(ps.fit_x), "processed_values": arr(processed_its.T) if processed_its is not None else None,
                       "sources": iteration_sources(original)} if its is not None else None,
    })
    if tr.experiment_type == "TimeTrace":
        out["time"] = _time_trace_extras(tr)
    return clean(out)


def _time_trace_extras(tr: ExperimentTrace) -> dict:
    meta = tr.metadata or {}
    timing = meta.get("time_trace") if isinstance(meta.get("time_trace"), dict) else {}
    x = np.asarray(tr.x_ns, float)
    dt = np.diff(x) if len(x) > 1 else np.array([])
    failed = timing.get("failedPoints") or []
    idx = meta.get("source_point_indices") or list(range(len(x)))
    retries = []
    try:
        fp = np.asarray(failed, float)
        for i, src in enumerate(idx):
            if 0 <= int(src) < len(fp) and fp[int(src)] > 0 and i < len(x):
                retries.append({"index": i, "t": float(x[i]), "count": float(fp[int(src)])})
    except Exception:
        pass
    events = timing.get("events") or []
    if not isinstance(events, list):
        events = []
    return clean({
        "dt": {"median": float(np.median(dt)) if dt.size else None, "mean": float(np.mean(dt)) if dt.size else None,
               "sd": float(np.std(dt)) if dt.size else None, "p5": float(np.percentile(dt, 5)) if dt.size else None,
               "p95": float(np.percentile(dt, 95)) if dt.size else None, "max": float(np.max(dt)) if dt.size else None},
        "retries": retries, "events": events,
        "scheduled": timing.get("scheduledSeconds"), "lateness": timing.get("latenessSeconds"),
    })


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

FIT_FAMILY = {"DEERFrequency": "ODMR", "DEERDuration": "Rabi", "DEERDecay": "SpinEcho"}


def _locks(spec: dict) -> dict | None:
    locks = {}
    for name, item in (spec.get("params") or {}).items():
        value = item.get("value")
        lock = bool(item.get("lock"))
        lo, hi = item.get("min"), item.get("max")
        if value is None and lo is None and hi is None and not lock:
            continue
        locks[name] = (float(value) if value is not None else float("nan"), lock,
                       float(lo) if lo is not None else None, float(hi) if hi is not None else None)
    return locks or None


def _evaluate(result: FitResult, x: np.ndarray, x_data: np.ndarray) -> np.ndarray | None:
    """Evaluate the fitted model on ``x``; verified against the stored y_fit first."""
    try:
        check = evaluate_result(result, x_data)
        ref = np.asarray(result.y_fit, float)
        if check.shape != ref.shape or not np.allclose(check, ref, rtol=1e-6, atol=1e-9 * max(1.,float(np.nanmax(np.abs(ref))))):
            return None
        return evaluate_result(result, x)
    except Exception:
        return None


def fit_inputs(ws: Workspace, path: str, detector: str | None, mode: str, proc: dict, spec: dict):
    """Select the fit data exactly as run_fit does: (trace, x, y, exp_type, cfg, locks)."""
    tr = ws.trace(path, detector, mode)
    if tr.scan_dim == "scan2d":
        raise ValueError("2D maps are characterized with spot detection, not the 1D fitter.")
    tr = selected_trace(tr, mode, proc.get("iterations"))
    data = trace_payload(ws, path, detector, mode, proc)
    transformed = bool(spec.get("fit_transformed") and any(s.get("enabled", True) for s in proc.get("steps", [])))
    key_x, key_y = ("x", "y") if transformed else ("fit_x", "fit_y")
    x = np.asarray(data["proc"][key_x], float)
    y = np.asarray(data["proc"][key_y], float)
    window = smoothing_window(len(y), int(proc.get("smooth", 1) or 1)) if spec.get("fit_smoothed") else 1
    if spec.get("fit_smoothed"):
        if window <= 1:
            raise ValueError("Fit smoothed requires an effective smoothing window of at least 3 points. Set it in Process.")
        y = smooth_trace(y, int(proc.get("smooth", 1) or 1))
    provenance = {**data["provenance"], "fit_target": "smoothed" if spec.get("fit_smoothed") else "raw_mean",
                  "fit_transformed": transformed, "smoothing_window": window,
                  "uncertainty_note": "Smoothing correlates samples and underestimates ordinary fit uncertainties; no corrected estimate is supplied. SEM weights are disabled." if window > 1 else ""}
    review = proc.get("iteration_review")
    if review and review.get("retained") == proc.get("iterations"):
        provenance["iteration_review"] = review
    tr = replace(tr, metadata={**(tr.metadata or {}), "fit_provenance": provenance})
    if not transformed and not proc.get("steps") and int(proc.get("bin", 1) or 1) > 1:
        # Feed the existing weighting routines the SEM of binned iteration means on the actual fit grid.
        tr = replace(tr, x_ns=x.copy(), y_sem=np.asarray(data["proc"]["sem"], float) if data["proc"]["sem"] is not None else None,
                     iteration_contrast=None, iteration_signal=None, iteration_reference=None, iteration_valid=None)
    lo, hi = spec.get("range_min"), spec.get("range_max")
    keep = np.isfinite(x) & np.isfinite(y)
    if lo is not None:
        keep &= x >= float(lo)
    if hi is not None:
        keep &= x <= float(hi)
    x, y = x[keep], y[keep]
    sem = data['proc'].get('sem')
    sem = np.asarray(sem,float) if sem is not None else None
    usable_sem = sem is not None and sem.shape == keep.shape and not proc.get('steps') and not spec.get('fit_smoothed')
    tr = replace(tr, x_ns=x.copy(), y=y.copy(), y_sem=sem[keep] if usable_sem else None)
    if len(x) < 5:
        raise ValueError("Fewer than 5 points in the fit range.")
    exp_type = spec.get("experiment") or tr.experiment_type
    model = spec.get("model") or "Auto"
    if model == "Auto" and tr.experiment_type == "LineScan" and not spec.get("experiment"):
        # 1D stage/fluorescence scans: Auto means a Gaussian peak fit (the LineScan profile's model).
        model = "GaussianPeaks"
        spec = {**spec, "peaks": spec.get("peaks") or 1}
    elif model == "Auto" and not tr.fit_allowed and tr.experiment_type != "TimeTrace" and not spec.get("experiment"):
        raise ValueError("This scan is plot-only under Auto. Choose Gaussian peaks or a custom model to fit it.")
    if tr.experiment_type == "TimeTrace":
        exp_type = "TimeTraceDecay" if model in ("Auto", "TimeTraceDecay") else model
    if model == "GaussianPeaks":
        exp_type = "GaussianPeaks"
    if model == "CustomModel":
        exp_type = "CustomModel"
    cfg = FitWorkflowConfig(
        multistart=int(spec.get("multistart", 20)),
        gaussian_peak_count=int(spec.get("peaks", 1)),
        rabi_use_sem=bool(spec.get("sem_weights", True)) and not spec.get("fit_smoothed"),
        sem_weights=bool(spec.get("sem_weights_all", False)) and not spec.get("fit_smoothed"),  # non-Rabi SEM weighting is opt-in (UI default off)
        rabi_pi_lock=bool(spec.get("pi_lock", False)),
        rabi_pi_ns=float(spec.get("pi_ns", 200.0)),
        rabi_pi2_ns=float(spec.get("pi2_ns", 100.0)),
        rabi_dead_time_ns=float(spec.get("dead_time_ns", 0.0)),
        rabi_mode=str(spec.get("rabi_mode", "Auto compare")),
        odmr_multipeak=bool(spec.get("odmr_multi", False)),
        odmr_peak_count=int(spec.get("odmr_peaks", 2)),
        odmr_auto_peak=bool(spec.get("odmr_auto_peaks", True)),
        odmr_polarity=str(spec.get("odmr_polarity", "Auto")),
        odmr_peak_ranges=tuple(tuple(r) for r in spec.get("odmr_ranges", []) if r and None not in r),
        t1_prep_mode=str(spec.get("t1_prep", "Auto")),
        t1_model_family=str(spec.get("t1_family", "Regular exponential")),
        custom_expr=str(spec.get("custom_expr", "")),
        custom_spec=spec.get("custom_spec"),
    )
    return tr, x, y, exp_type, cfg, _locks(spec)


def fit_dispatch(tr: ExperimentTrace, x: np.ndarray, y: np.ndarray, exp_type: str, cfg: FitWorkflowConfig, locks, spec: dict):
    """Run the model-family fit; returns (FitResult, ramsey candidate summaries | None)."""
    family = FIT_FAMILY.get(exp_type, exp_type)
    request = PreparedFit.create(x, y, family, cfg, locks, tr, spec.get('ramsey_model', 'Auto'))
    return request.run(robust=bool(spec.get('robust')))


def _conf_band(result: FitResult, dense_x: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """1-sigma model-mean band from the fit's covariance and free parameters."""
    if result.covariance is None:
        return None
    try:
        covariance = np.asarray(result.covariance, float)
        free = np.asarray(result.free_mask if result.free_mask is not None else [True]*len(result.params), bool)
        if not np.all(np.isfinite(covariance[np.ix_(free, free)])):
            return None
        columns = []
        for i, value in enumerate(result.params):
            if not free[i]:
                columns.append(np.zeros_like(dense_x, dtype=float)); continue
            step = 1e-6 * max(abs(value), 1e-3)
            hi, lo = result.params.copy(), result.params.copy()
            hi[i] += step; lo[i] -= step
            columns.append((evaluate_result(replace(result,params=hi), dense_x)-evaluate_result(replace(result,params=lo), dense_x))/(2*step))
        jac = np.column_stack(columns)
        variance = np.einsum('ij,jk,ik->i',jac,covariance,jac)
        return np.sqrt(np.maximum(variance,0.))
    except Exception:
        return None


def run_fit(ws: Workspace, path: str, detector: str | None, mode: str, proc: dict, spec: dict) -> dict:
    tr, x, y, exp_type, cfg, locks = fit_inputs(ws, path, detector, mode, proc, spec)
    analysis = analysis_for(ws,path,tr.detector_id,mode,proc,spec)
    t0 = time.perf_counter()
    result, candidates = fit_dispatch(tr, x, y, exp_type, cfg, locks, spec)
    elapsed = time.perf_counter() - t0

    return _fit_payload(tr,x,y,exp_type,result,candidates,elapsed,analysis)


def _fit_payload(tr,x,y,exp_type,result,candidates=None,elapsed=0.,analysis=None):
    mode=tr.mode
    profile_name = tr.experiment_type if tr.experiment_type in PROFILES else FIT_FAMILY.get(tr.experiment_type, "Rabi")
    profile = PROFILES.get(profile_name) or PROFILES.get("Rabi")
    status, reason = classify_status(r2=float(result.r2) if np.isfinite(result.r2) else -1.0,
                                     bound_hits=int(sum(result.bound_hits)), profile=profile)
    if not result.success:
        status, reason = "WARN" if status == "PASS" else status, f"{reason}; optimizer: {result.message.splitlines()[0]}"
    try:
        metrics = pro_batch._extract_nv_metrics(exp_type, result, tr.metadata)
    except Exception as exc:
        metrics = {"metrics_error": str(exc)}

    dense_x = np.linspace(float(np.min(x)), float(np.max(x)), 800)
    dense_y = _evaluate(result, dense_x, x)
    extras = dict(result.extras or {})
    envelope = None
    if isinstance(extras.get("rabi_envelope"), dict):
        bounds = rabi_envelope_bounds(dense_x, extras["rabi_envelope"])
        if bounds is not None:
            envelope = {"x": arr(dense_x), "upper": arr(bounds[0]), "lower": arr(bounds[1])}
    plot_y = extras.pop("plot_y", None)
    band=_conf_band(result,dense_x,x,np.asarray(plot_y,float) if plot_y is not None else y) if dense_y is not None else None
    uncertainty={**(result.uncertainty or {}),'band_available':band is not None,'band_note':'1-sigma uncertainty of model mean from retained fit covariance' if band is not None else 'Band unavailable: fit covariance or model evaluator missing/invalid'}
    params = []
    errs = np.asarray(result.errors, float)
    for i, name in enumerate(result.param_names):
        meta = param_meta(name)
        err = float(errs[i]) if i < len(errs) else float("nan")
        val = float(result.params[i])
        # A standard error many orders below the value usually means a degenerate covariance.
        reliable = math.isfinite(err) and err > 0 and (val == 0 or err / max(abs(val), 1e-300) > 1e-7)
        fixed=result.free_mask is not None and not bool(result.free_mask[i])
        unit, desc = meta.units, meta.description
        native_time=(tr.metadata or {}).get('x_unit') or ('s' if tr.experiment_type=='TimeTrace' else 'ns')
        if tr.scan_dim=='time' and tr.experiment_type!='TimeTrace':
            if unit=='ns':unit=native_time
            if unit=='GHz':unit={'ps':'THz','ns':'GHz','us':'MHz','µs':'MHz','μs':'MHz','ms':'kHz','s':'Hz'}.get(native_time,'1/'+native_time)
        if result.model_name.startswith("T1"):
            if name == "T1":
                unit = tr.metadata.get("x_unit") or "ns"
                desc = "T1 timescale in the file's x units (not display-calibrated units)."
            elif name in ("A", "y0"):
                normalized = extras.get("t1_prep_mode") == "normalized_1_minus_data"
                unit = "normalized a.u." if normalized else ("dimensionless contrast" if mode == "contrast" else tr.metadata.get("value_unit") or "a.u.")
                desc = f"{name} in normalized 1 - data space." if normalized else f"{name} in the measured {mode} space."
        params.append({"name": name, "symbol": meta.symbol, "unit": unit, "desc": desc,
                       "value": val, "error": 0. if fixed else err if reliable else None,"locked":fixed,
                       "bound_hit": bool(result.bound_hits[i]) if i < len(result.bound_hits) else False})
    residuals = y - np.asarray(result.y_fit, float) if plot_y is None else np.asarray(plot_y, float) - np.asarray(result.y_fit, float)
    return clean({
        "analysis": analysis,
        "model": result.model_name,
        "experiment": exp_type,
        "selection": result.selection_note,
        "provenance": (tr.metadata or {}).get("fit_provenance"),
        "message": result.message,
        "success": result.success,
        "status": status, "reason": reason,
        "r2": result.r2, "rmse": result.rmse, "aic": result.aic, "bic": result.bic, "dw": result.durbin_watson,
        "bound_hits": int(sum(result.bound_hits)),
        "params": params,
        "metrics": metrics,
        "x": arr(x), "y_fit": arr(result.y_fit), "residuals": arr(residuals),
        "plot_y": arr(plot_y) if plot_y is not None else None,
        "plot_y_label": extras.pop("plot_y_label", None),
        "dense": {"x": arr(dense_x), "y": arr(dense_y), "band": arr(band)} if dense_y is not None else None,
        "envelope": envelope,
        "candidates": candidates,
        "uncertainty": uncertainty,
        "extras": {k: v for k, v in extras.items() if not isinstance(v, (np.ndarray,)) or np.size(v) < 64},
        "elapsed_s": elapsed,
        "raw": fit_result_to_dict(result),
    })


# ---------------------------------------------------------------------------
# 2D maps: spots and overlays
# ---------------------------------------------------------------------------

def _map_arrays(ws: Workspace, path: str, detector: str, mode: str, source: str = "value"):
    tr = ws.trace(path, detector, mode)
    if tr.scan_dim != "scan2d" or tr.z2d is None:
        raise ValueError("Not a 2D map.")
    x = np.asarray(tr.x2d, float)[0, :]
    y = np.asarray(tr.y2d, float)[:, 0]
    if source == "signal" and tr.signal2d is not None:
        z = tr.signal2d
    elif source == "reference" and tr.reference2d is not None:
        z = tr.reference2d
    elif source == "difference" and tr.signal2d is not None and tr.reference2d is not None:
        z = np.asarray(tr.reference2d, float) - np.asarray(tr.signal2d, float)
    elif source == "auto":
        # "Auto fluorescence": prefer structured raw counts so a flat derived contrast does not hide spots.
        candidates = [m for m in (tr.signal2d, tr.reference2d) if m is not None]
        z = tr.z2d
        best = -1.0
        for m in candidates + [tr.z2d]:
            m = np.asarray(m, float)
            finite = m[np.isfinite(m)]
            if finite.size and np.nanmean(np.abs(finite)) > 0:
                score = float(np.nanstd(finite) / (np.nanmean(np.abs(finite)) + 1e-300))
                if score > best:
                    best, z = score, m
    else:
        z = tr.z2d
    return tr, x, y, np.asarray(z, float)


def spots(ws: Workspace, path: str, detector: str, mode: str, opts: dict) -> dict:
    tr, x, y, z = _map_arrays(ws, path, detector, mode, opts.get("source", "auto"))
    polarity = {"Auto": "auto", "Bright": "bright", "Dark": "dark", "Bright and dark": "both"}.get(opts.get("polarity", "Auto"), "auto")
    res = detect_fluorescent_spots(x, y, z, polarity=polarity,
                                   threshold_sigma=float(opts.get("threshold", 3.0)),
                                   min_pixels=int(opts.get("min_pixels", 4)),
                                   max_spots=int(opts.get("max_spots", 12)),
                                   background_window_pixels=int(opts.get("bg_window", 0) or 0))
    rows = []
    for s in res.spots:
        d = asdict(s)
        d.update({
            "fwhm_major": s.sigma_major * FWHM_PER_SIGMA, "fwhm_minor": s.sigma_minor * FWHM_PER_SIGMA,
            "hwhm_major": s.sigma_major * FWHM_PER_SIGMA / 2, "hwhm_minor": s.sigma_minor * FWHM_PER_SIGMA / 2,
            "axis_ratio": s.sigma_major / s.sigma_minor if s.sigma_minor else None,
            "detector": tr.detector_id,
        })
        rows.append(d)
    return clean({"spots": rows, "noise_sigma": res.noise_sigma, "bg_window": res.background_window_pixels,
                  "polarities": list(res.tested_polarities), "source_map": arr(z)})


def compare_spots(ws: Workspace, a: dict, b: dict, mode: str) -> dict:
    dx, dy = b["center_x"] - a["center_x"], b["center_y"] - a["center_y"]
    out = {"dx": dx, "dy": dy, "distance": math.hypot(dx, dy),
           "fwhm_major_ratio": b["fwhm_major"] / a["fwhm_major"] if a.get("fwhm_major") else None,
           "fwhm_minor_ratio": b["fwhm_minor"] / a["fwhm_minor"] if a.get("fwhm_minor") else None,
           "angle_diff": b["angle_deg"] - a["angle_deg"],
           "amplitude_ratio": b["amplitude"] / a["amplitude"] if a.get("amplitude") else None}
    try:
        _, x1, y1, z1 = _map_arrays(ws, a["path"], a["detector"], mode, a.get("source", "auto"))
        _, x2, y2, z2 = _map_arrays(ws, b["path"], b["detector"], mode, b.get("source", "auto"))
        if z1.shape == z2.shape:
            cx, cy = (a["center_x"] + b["center_x"]) / 2, (a["center_y"] + b["center_y"]) / 2
            r = 2.0 * max(a.get("fwhm_major", 0), b.get("fwhm_major", 0))
            mx = np.abs(x1 - cx) <= r
            my = np.abs(y1 - cy) <= r
            p, q = z1[np.ix_(my, mx)].ravel(), z2[np.ix_(my, mx)].ravel()
            ok = np.isfinite(p) & np.isfinite(q)
            if ok.sum() >= 4:
                out["profile_correlation"] = float(np.corrcoef(p[ok], q[ok])[0, 1])
    except Exception:
        pass
    pixel = None
    try:
        pixel = abs(float(np.median(np.diff(x1))))
    except Exception:
        pass
    out["distance_px"] = out["distance"] / pixel if pixel else None
    return clean(out)


def map_overlay(ws: Workspace, first: dict, second: dict, opts: dict) -> dict:
    first_analysis=analysis_for(ws,first['path'],first['detector'],first.get('mode','contrast'),{'map_source':first.get('source','value')},kind='map')
    second_analysis=analysis_for(ws,second['path'],second['detector'],second.get('mode','contrast'),{'map_source':second.get('source','value')},kind='map')
    t1, x1, y1, z1 = _map_arrays(ws, first["path"], first["detector"], first.get("mode", "contrast"), first.get("source", "value"))
    t2, x2, y2, z2 = _map_arrays(ws, second["path"], second["detector"], second.get("mode", "contrast"), second.get("source", "value"))
    if not compatible_axes(tuple(t1.scan_axes), tuple(t2.scan_axes)):
        raise ValueError("The maps are not in the same plane.")
    if axis_key(t1.scan_axes[0]) != axis_key(t2.scan_axes[0]):  # opposite stored order: transpose the second map
        x2, y2, z2 = y2, x2, z2.T
    def map_unit(tr,axis):
        units=(tr.metadata or {}).get('axis_units') or {}
        if not isinstance(units,dict): units={}
        return units.get(axis_key(axis),units.get(axis,(tr.metadata or {}).get('x_unit') or ''))
    x2=x2*unit_scale(map_unit(t2,t1.scan_axes[0]),map_unit(t1,t1.scan_axes[0]))
    y2=y2*unit_scale(map_unit(t2,t1.scan_axes[1]),map_unit(t1,t1.scan_axes[1]))
    obs1=first.get('source','value');obs2=second.get('source','value')
    obs1=first.get('mode','contrast') if obs1=='value' else obs1
    obs2=second.get('mode','contrast') if obs2=='value' else obs2
    if obs1!=obs2:raise ValueError('Map comparison requires matching source observables.')
    value_unit=lambda tr:'dimensionless' if obs1=='contrast' else (tr.metadata or {}).get('value_unit') or 'a.u.'
    z2=z2*unit_scale(value_unit(t2),value_unit(t1))
    ov = build_map_overlay((x1, y1, z1), (x2, y2, z2), shared=bool(opts.get("shared")),
                           background_percentile=float(opts.get("background", 10.0)),
                           palette=str(opts.get("palette", "Pink / teal")))
    png = _overlay_png(ov.x_edges, ov.y_edges, ov.rgba, swap=bool(opts.get("swap")))
    request={'first':first,'second':second,'opts':opts}
    analysis=result_envelope(first_analysis['descriptor'],first_analysis['source_revision'],request,
        {source_path(second['path']):second_analysis['source_revision']},'map_overlay')
    return clean({"x_edges": arr(ov.x_edges), "y_edges": arr(ov.y_edges), "png": png, "ranges": ov.ranges,
                  "request":request,"analysis":analysis,"analyses":[first_analysis,second_analysis],
                  "units":{"x":map_unit(t1,t1.scan_axes[0]),"y":map_unit(t1,t1.scan_axes[1]),"value":value_unit(t1)},
                  "x_label": t1.scan_axes[0], "y_label": t1.scan_axes[1],
                  "cells": {"primary": arr(ov.primary), "secondary": arr(ov.secondary), "rgba": arr(ov.rgba)}})


def _overlay_png(xe: np.ndarray, ye: np.ndarray, rgba: np.ndarray, swap: bool = False, max_px: int = 1200) -> str:
    """Rasterize native cell rectangles by copying each cell's colour (no interpolation)."""
    from PIL import Image

    if swap:
        xe, ye, rgba = ye, xe, np.transpose(rgba, (1, 0, 2))
    w_span, h_span = float(xe[-1] - xe[0]), float(ye[-1] - ye[0])
    min_cell = min(np.min(np.diff(xe)) / w_span, np.min(np.diff(ye)) / h_span)
    scale = int(min(max_px, max(rgba.shape[1], rgba.shape[0], math.ceil(4 / max(min_cell, 1e-6)))))
    nx = max(rgba.shape[1], min(max_px, scale))
    ny = max(rgba.shape[0], min(max_px, int(round(nx * h_span / w_span)) if w_span else scale))
    px = xe[0] + (np.arange(nx) + 0.5) * w_span / nx
    py = ye[0] + (np.arange(ny) + 0.5) * h_span / ny
    ix = np.clip(np.searchsorted(xe, px, side="right") - 1, 0, rgba.shape[1] - 1)
    iy = np.clip(np.searchsorted(ye, py, side="right") - 1, 0, rgba.shape[0] - 1)
    img = rgba[np.ix_(iy, ix)]
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)[::-1]  # image rows top-down
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Time traces
# ---------------------------------------------------------------------------

def time_stats(ws: Workspace, path: str, mode: str, windows: dict) -> dict:
    ds = ws.dataset(path, mode)
    out = {}
    for det, tr in ds.detectors.items():
        x, y = np.asarray(tr.x_ns, float), np.asarray(tr.y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        row = {"mean": float(np.mean(y)), "sd": float(np.std(y, ddof=1)) if len(y) > 1 else None, "n": int(len(y))}
        row["cv"] = row["sd"] / row["mean"] if row["sd"] is not None and row["mean"] else None
        for key in ("a", "b"):
            w = windows.get(key)
            if w:
                sel = (x >= w[0]) & (x <= w[1])
                row[f"{key}_mean"] = float(np.mean(y[sel])) if sel.any() else None
                row[f"{key}_n"] = int(sel.sum())
        if row.get("a_mean") and row.get("b_mean") is not None:
            row["change"] = row["b_mean"] / row["a_mean"] - 1.0
        out[det] = row
    return clean(out)


def periodogram(ws: Workspace, path: str, detector: str, mode: str) -> dict:
    """Lomb–Scargle power for unevenly sampled time traces (no resampling)."""
    from scipy.signal import lombscargle

    tr = ws.trace(path, detector, mode)
    x, y = np.asarray(tr.x_ns, float), np.asarray(tr.y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok] - np.mean(y[ok])
    span = float(x.max() - x.min())
    dt = float(np.median(np.diff(x)))
    f = np.linspace(1.0 / span, 0.5 / dt, 2000)
    p = lombscargle(x, y, 2 * np.pi * f, normalize=True)
    return clean({"freq_hz": arr(f), "power": arr(p), "nyquist_hz": 0.5 / dt})


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _row_key(r: dict) -> tuple:
    return (r.get("relative_path") or r.get("file") or "", r.get("detector_id") or "")


def merge_summary_rows(old: list[dict], new: list[dict]) -> list[dict]:
    """Replace old rows matching a new row's (path, detector) key, keep the rest in order, append unmatched new rows."""
    new_by_key = {_row_key(r): r for r in new}
    used, out = set(), []
    for r in old:
        k = _row_key(r)
        if k in new_by_key:
            if k not in used:
                out.append(new_by_key[k])
                used.add(k)
        else:
            out.append(r)
    out.extend(r for r in new if _row_key(r) not in used)
    return out


def _merge_csv(path: Path, old: list[dict]) -> list[dict]:
    """Merge `old` rows into the freshly written CSV at `path`, rewriting it with the union of columns."""
    if not path.exists():
        return old
    with path.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        new, cols = list(rd), list(rd.fieldnames or [])
    if not old:
        return new
    merged = merge_summary_rows(old, new)
    names = list(old[0].keys())
    for r in merged:
        names.extend(k for k in r if k not in names)
    for c in cols:
        if c not in names:
            names.append(c)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=names, restval="")
        wr.writeheader()
        wr.writerows(merged)
    return merged


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def batch_run_record(output_dir):
    from nvfit.batch_manifest import load_manifest
    path=Path(output_dir)/'batch_manifest.json'
    return load_manifest(path,validate_sources=False) if path.exists() else None


def batch_start(ws: Workspace, opts: dict) -> str:
    opts=deepcopy(opts)
    from nvfit.batch_manifest import load_manifest
    replay=load_manifest(opts['manifest_path']) if opts.get('manifest_path') else None
    output_dir=replay['options']['output_dir'] if replay else opts['output_dir']
    job_id='batch'+uuid.uuid4().hex
    job={'id':job_id,'state':'running','done':0,'total':0,'log':[],'rows':[],'error':None,'output_dir':output_dir,'cancel':False,'manifest_path':None}
    with ws._lock:
        if any(j['state']=='running' and os.path.normcase(j['output_dir'])==os.path.normcase(output_dir) for j in ws.jobs.values()):
            raise ValueError('A batch is already writing this output folder.')
        ws.jobs[job_id]=job
    def progress(done,total,name,message):
        job['done'],job['total']=done,total
        job['log'].append(f'[{done}/{total}] {name}: {message}')
        record=batch_run_record(output_dir)
        if record:
            job['manifest_path']=record['manifest_path']
            job['unfinished_count']=sum(r['state']=='pending' for r in record['rows'])
            job['discovery_complete']=record.get('discovery_complete',True)
            job['rows']=[r['summary'] for r in record['rows'] if r['state'] in ('completed','preserved')]
    def work():
        try:
            runtime={'progress_callback':progress,'cancel_requested':lambda:job['cancel']}
            if replay:
                result=pro_batch.replay_batch(opts['manifest_path'],selected_rows=opts.get('rows'),resume_unfinished=bool(opts.get('resume_unfinished')),**runtime)
            else:
                result=pro_batch.run_batch(input_dir=Path(opts['input_dir']),output_dir=Path(output_dir),
                    mode=opts.get('mode','contrast'),bin_size=int(opts.get('bin',1)),smooth_window=int(opts.get('smooth',1)),
                    roi_min=opts.get('roi_min'),roi_max=opts.get('roi_max'),
                    single_file=Path(opts['single_file']) if opts.get('single_file') else None,
                    model_override=opts.get('model') or None,multistart=int(opts.get('multistart',12)),robust=bool(opts.get('robust')),
                    odmr_multipeak=bool(opts.get('odmr_multi')),dpi=int(opts.get('dpi',180)),write_origin_bundles=bool(opts.get('origin',True)),
                    recursive=bool(opts.get('recursive')),skip_checkpoints=bool(opts.get('skip_checkpoints')),detector=opts.get('detector','all'),
                    detector_labels=opts.get('detector_labels'),ui_options=opts,**runtime)
            if result:job.update(result)
            else:job.update(state='finished',rows=read_batch_summary(output_dir))
        except KeyboardInterrupt:
            job['state']='cancelled'
        except Exception as exc:
            job['state'],job['error']='failed',str(exc)
        finally:
            if job.get('manifest_path'):
                record=load_manifest(job['manifest_path'],validate_sources=False)
                job.update(done=record['completed_count'],total=len(record['requested_rows']),
                    unfinished_count=sum(r['state']=='pending' for r in record['rows']),
                    discovery_complete=record.get('discovery_complete',True),
                    rows=[r['summary'] for r in record['rows'] if r['state'] in ('completed','preserved')])
    threading.Thread(target=work,daemon=True).start()
    return job_id


def batch_import(ws,manifest_path,row_key):
    from nvfit.batch_manifest import load_manifest, fingerprint
    manifest=load_manifest(manifest_path)
    row=next((r for r in manifest['rows'] if r['row_id']==row_key),None)
    if not row or not row.get('summary') or row['summary']['status'] not in ('PASS','WARN'):
        raise ValueError('This row has no ordinary converged fit to import.')
    output=next((o for o in row.get('outputs',[]) if o['path'].endswith('_result.json')),None)
    if not output or fingerprint(output['path'])['sha256']!=output['revision']['sha256']:
        raise ValueError('The stored result is missing or changed.')
    saved=json.loads(Path(output['path']).read_text(encoding='utf-8'))
    series=saved.get('series')
    if not series:raise ValueError('This older run lacks stored curves; start a new batch before importing.')
    options=manifest['options'];mode=saved['provenance']['mode'];path=row['source_path'];det=row['detector_id']
    proc={'bin':options['bin_size'],'smooth':options['smooth_window'],'roi_min':options['roi_min'],'roi_max':options['roi_max'],
          'iterations':None,'excluded':[],'exclusion_ranges':[],'steps':[]}
    extras=saved.get('fit_extras') or {}
    custom=manifest.get('auxiliary',{}).get('custom_model')
    if custom:extras['custom_spec']=custom
    spec={'model':'CustomModel' if custom else 'Auto','experiment':saved['experiment_type'],'multistart':options['multistart'],
          'robust':options['robust'],'odmr_multi':options['odmr_multipeak'],'odmr_peaks':4,'odmr_auto_peaks':True,'odmr_polarity':'Auto',
          'rabi_mode':'Single reliable Rabi','sem_weights':True,'sem_weights_all':False,
          't1_prep':{'normalized_1_minus_data':'Normalize then 1-data','direct_decay':'Direct decay','direct':'Direct T1'}.get(extras.get('t1_prep_mode'),'Auto'),
          't1_family':'Stretched exponential' if extras.get('t1_model_family')=='stretched' else 'Regular exponential',
          'range_min':None,'range_max':None,'fit_smoothed':options['smooth_window']>1,'params':{},'custom_spec':custom,
          'batch_options':options}
    names=list(saved['params']);metrics=saved['metrics']
    result=FitResult(model_name=saved['selected_model'],param_names=names,params=np.array([saved['params'][n] for n in names],float),
        errors=np.array([saved['errors'].get(n) for n in names],float),y_fit=np.asarray(series['y_fit'],float),
        r2=metrics['r2'],rmse=metrics['rmse'],aic=metrics['aic'],bic=metrics['bic'],durbin_watson=metrics['durbin_watson'],
        bound_hits=[bool(saved['bound_hits'].get(n)) for n in names],success=metrics['success'],message=metrics['message'],
        extras=extras,selection_note=saved['selection_note'],covariance=np.asarray(saved['covariance'],float) if saved.get('covariance') is not None else None,
        free_mask=saved.get('free_mask'),uncertainty=saved.get('uncertainty') or {})
    analysis=analysis_for(ws,path,det,mode,proc,spec)
    analysis['imported_from']={'run_id':manifest['run_id'],'row_id':row_key,'manifest_path':manifest_path}
    tr=ws.trace(path,det,mode)
    fit=_fit_payload(tr,np.asarray(series['x'],float),np.asarray(series['input_y'],float),saved['experiment_type'],result,analysis=analysis)
    fit.update(status=saved['status'],reason=saved['status_reason'],metrics=saved['nv_metrics'])
    data=trace_payload(ws,path,det,mode,proc)
    data['proc'].update(x=series['x'],y=series['input_y'],fit_x=series['x'],fit_y=series['input_y'],smooth=None)
    return clean({'path':path,'detector':det,'mode':mode,'proc':proc,'spec':spec,'data':{det:data},'fit':fit})


def batch_rerun(ws,manifest_path,rows):
    if not rows:raise ValueError('Choose at least one row to rerun.')
    return batch_start(ws,{'manifest_path':manifest_path,'rows':list(rows)})


def batch_resume(ws,manifest_path):
    from nvfit.batch_manifest import load_manifest
    record=load_manifest(manifest_path)
    pending=[r['row_id'] for r in record['rows'] if r['state'] in ('pending','not_requested')]
    if not pending and record.get('discovery_complete',True):
        raise ValueError('This run has no unfinished rows.')
    return batch_start(ws,{'manifest_path':manifest_path,'rows':pending,'resume_unfinished':True})


def read_batch_summary(output_dir: str) -> list[dict]:
    record=batch_run_record(output_dir)
    if record:return [r['summary'] for r in record['rows'] if r['state'] in ('completed','preserved')]
    path = Path(output_dir) / "batch_summary.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def batch_cli(opts: dict) -> str:
    parts = ["python -m nvfit.pro_batch"]
    if opts.get("single_file"):
        parts.append(f'--single-file "{opts["single_file"]}"')
    else:
        parts.append(f'--input-dir "{opts.get("input_dir", "")}"')
    parts.append(f'--output-dir "{opts.get("output_dir", "")}"')
    if opts.get("mode", "contrast") != "contrast":
        parts.append(f"--mode {opts['mode']}")
    parts.append(f"--detector {opts.get('detector', 'all')}")
    for flag, key, default in (("--bin-size", "bin", 1), ("--smooth-window", "smooth", 1), ("--multistart", "multistart", 12)):
        if int(opts.get(key, default)) != default:
            parts.append(f"{flag} {int(opts[key])}")
    if opts.get("model"):
        parts.append(f"--model-override {opts['model']}")
    for flag, key in (("--robust", "robust"), ("--odmr-multipeak", "odmr_multi"), ("--recursive", "recursive"), ("--skip-checkpoints", "skip_checkpoints")):
        if opts.get(key):
            parts.append(flag)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _decode_data_url(data: str) -> bytes:
    return base64.b64decode(data.split(",", 1)[1] if "," in data else data)


def svg_to_pdf_bytes(svg: str) -> bytes:
    """Convert a Plotly-style SVG string to vector PDF bytes (svglib + reportlab)."""
    import re
    import xml.etree.ElementTree as ET
    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg

    # svglib mishandles percentage font sizes (Plotly superscripts use 70%): resolve them to px.
    def fix(node, parent_px=16.0):
        style = node.get("style") or ""
        m = re.search(r"font-size:\s*([\d.]+)(px|%)", style)
        px = parent_px
        if m:
            px = float(m.group(1)) * (parent_px / 100.0 if m.group(2) == "%" else 1.0)
            if m.group(2) == "%":
                node.set("style", style.replace(m.group(0), f"font-size: {px:.2f}px"))
        for child in node:
            fix(child, px)

    root = ET.fromstring(svg.encode("utf-8"))
    fix(root)
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    drawing = svg2rlg(io.BytesIO(ET.tostring(root, encoding="utf-8")))
    if drawing is None:
        raise ValueError("SVG could not be converted")
    return renderPDF.drawToString(drawing)


def validate_analysis(ws, envelope):
    from nvfit.analysis_identity import scientific_revision, canonical
    try:
        expected=result_envelope(envelope['descriptor'],envelope['source_revision'],envelope['settings'],envelope['dependencies'],envelope['kind'])
        if expected['signature']!=envelope['signature'] or envelope.get('unverified'):
            return {'fresh':False,'reason':'Unverified result signature'}
        sources=dict(envelope.get('dependencies') or {})
        path=envelope['descriptor']['source']
        if path=='aggregate':sources.update(envelope['source_revision'])
        else:sources[path]=envelope['source_revision']
        for path,revision in sources.items():
            if canonical(scientific_revision(ws.source_revision(path)))!=canonical(scientific_revision(revision)):
                return {'fresh':False,'reason':f'Source changed: {path}'}
        return {'fresh':True,'reason':''}
    except (OSError,KeyError,ValueError) as exc:
        return {'fresh':False,'reason':str(exc)}


def export_results(ws: Workspace, req: dict) -> dict:
    if req.get("combined_recipe"):
        recipe = req["combined_recipe"]
        # Validate virtual sources before writing any output.
        combine_datasets(ws.dataset, recipe, req.get("mode", "contrast"))
        with ws._lock:
            ws.combined[recipe["path"]] = recipe
    for envelope in (req.get('analysis'),(req.get('fit') or {}).get('analysis'),(req.get('map_comparison') or {}).get('analysis')):
        if envelope:
            validation=validate_analysis(ws,envelope)
            if not validation['fresh']:raise ValueError('Export source changed; reload before saving. '+validation['reason'])
    out_dir = Path(req["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = req["stem"]
    written = []
    images = req.get("images") or {}
    if images.get("png"):
        p = out_dir / f"{stem}_fit.png"
        p.write_bytes(_decode_data_url(images["png"]))
        written.append(str(p))
    if images.get("svg"):
        p = out_dir / f"{stem}_fit.svg"
        p.write_text(images["svg"], encoding="utf-8")
        written.append(str(p))
    pdf_done = False
    if images.get("pdf_svg"):
        try:
            p = out_dir / Path(req.get("pdf_filename") or f"{stem}_fit.pdf").name
            p.write_bytes(svg_to_pdf_bytes(images["pdf_svg"]))
            written.append(str(p))
            pdf_done = True
        except Exception:  # fall back to the raster PDF below
            pass
    if images.get("pdf_png") and not pdf_done:
        from PIL import Image

        p = out_dir / Path(req.get("pdf_filename") or f"{stem}_fit.pdf").name
        Image.open(io.BytesIO(_decode_data_url(images["pdf_png"]))).convert("RGB").save(p, "PDF", resolution=float(req.get("dpi", 300)))
        written.append(str(p))
    if images.get("report_png"):
        p = out_dir / f"{stem}_report.png"
        p.write_bytes(_decode_data_url(images["report_png"]))
        written.append(str(p))
    doc = {
        "app": f"SmartFitter {APP_VERSION}", "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis":req.get("analysis"), "historical_results":req.get("historical_results"),
        "source": req.get("source"), "detector": req.get("detector"), "mode": req.get("mode"),
        "processing": req.get("processing"), "fit_spec": req.get("fit_spec"), "fit": req.get("fit"),
        "spots": req.get("spots"), "map_comparison": req.get("map_comparison"), "plot_state": req.get("plot_state"),
        "provenance": req.get("provenance"), "iteration_fits": req.get("iteration_fits"), "iteration_sensitivity": req.get("iteration_sensitivity"), "combined_recipe": req.get("combined_recipe"),
    }
    if req.get("json", True):
        p = out_dir / f"{stem}_result.json"
        p.write_text(json.dumps(clean(doc), indent=2), encoding="utf-8")
        written.append(str(p))
    fit = req.get("fit") or {}
    src = req.get("source")
    if req.get("csv") and fit:
        p = out_dir / f"{stem}_result.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["parameter", "value", "error", "unit"])
            for prm in fit.get("params", []):
                w.writerow([prm["name"], prm["value"], prm.get("error"), prm.get("unit", "")])
            for k, v in (fit.get("metrics") or {}).items():
                w.writerow([k, v, "", ""])
            for k in ("r2", "rmse", "aic", "bic", "dw"):
                w.writerow([k, fit.get(k), "", ""])
            for key, value in (fit.get("provenance") or req.get("provenance") or {}).items():
                w.writerow(["provenance." + key, json.dumps(clean(value), ensure_ascii=False), "", ""])
        written.append(str(p))
    if req.get("origin") and src:
        data = trace_payload(ws, src, req.get("detector"), req.get("mode", "contrast"), req.get("processing"))
        if "proc" in data:
            p = out_dir / f"{stem}_origin_bundle.csv"
            fx = {round(v, 12): i for i, v in enumerate(fit.get("x") or [])}
            fp = fit.get("provenance") or {}
            with p.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["x", "y", "y_smooth", "y_fit", "residual", "fit_target_y", "x_unit", "y_unit", "fit_space", "fit_target", "smoothing_window", "selected_iterations", "iteration_sources", "uncertainty_note"])
                sm = data["proc"].get("smooth") or [None] * len(data["proc"]["x"])
                def at(key, i):
                    values = fit.get(key) or []
                    return values[i] if i is not None and i < len(values) else None
                for xv, yv, sv in zip(data["proc"]["x"], data["proc"]["y"], sm):
                    i = fx.get(round(xv, 12)) if xv is not None else None
                    w.writerow([xv, yv, sv, at("y_fit", i), at("residuals", i), at("plot_y", i),
                                data["info"].get("x_unit"), "dimensionless" if req.get("mode", "contrast") == "contrast" else data["info"].get("unit"),
                                fit.get("plot_y_label") or req.get("mode", "contrast"), fp.get("fit_target"), fp.get("smoothing_window"),
                                json.dumps(fp.get("selected_iterations", [])), json.dumps(fp.get("iteration_sources", [])), fp.get("uncertainty_note", "")])
            written.append(str(p))
    if req.get("detectors_raw") and src:
        ds = ws.dataset(src, req.get("mode", "contrast"))
        if len(ds.detectors) > 1:
            p = out_dir / f"{stem}__detectors_raw.csv"
            with p.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["detector_id", "detector_label", "x", "y_coord", "signal", "reference", "contrast", "difference"])
                for det, tr in ds.detectors.items():
                    if tr.scan_dim == "scan2d" and tr.signal2d is not None and tr.reference2d is not None:
                        xs, ys = np.asarray(tr.x2d, float), np.asarray(tr.y2d, float)
                        for r, c in np.ndindex(xs.shape):
                            s, ref = tr.signal2d[r, c], tr.reference2d[r, c]
                            w.writerow([det, tr.detector_label, xs[r, c], ys[r, c], s, ref, (ref - s) / ref if ref else "", ref - s])
                    else:
                        for i, xv in enumerate(tr.x_ns):
                            s, ref = tr.signal[i], tr.reference[i]
                            w.writerow([det, tr.detector_label, xv, "", s, ref, (ref - s) / ref if ref else "", ref - s])
            written.append(str(p))
    if req.get("time_trace") and src:
        ds = ws.dataset(src, req.get("mode", "reference"))
        p = out_dir / f"{stem}__time_trace.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["source_file", "detector_id", "source_point_index", "elapsed_s", "reference", "signal", "contrast", "difference", "scheduled_start_s", "lateness_s", "retries", "n_points", "recorded_units"])
            for tr in ds.detectors.values():
                meta = tr.metadata or {}
                timing = meta.get("time_trace") if isinstance(meta.get("time_trace"), dict) else {}
                ref_only = bool(meta.get("reference_only"))
                idx = meta.get("source_point_indices") or list(range(len(tr.x_ns)))

                def val(values, i):
                    try:
                        v = float(values[i])
                        return v if math.isfinite(v) else ""
                    except Exception:
                        return ""
                for i, t in enumerate(tr.x_ns):
                    ref, sig = tr.reference[i], tr.signal[i]
                    contrast = (ref - sig) / ref if np.isfinite(sig) and ref else float("nan")
                    src_i = int(idx[i]) if i < len(idx) else i
                    w.writerow([tr.source_path, tr.detector_id, src_i, t, val(tr.reference, i),
                                "" if ref_only else val(tr.signal, i), "" if ref_only or not np.isfinite(contrast) else contrast,
                                "" if ref_only else ref - sig, val(timing.get("scheduledSeconds") or [], src_i),
                                val(timing.get("latenessSeconds") or [], src_i), val(timing.get("failedPoints") or [], src_i),
                                val(timing.get("nPoints") or [], src_i), meta.get("value_unit", "counts")])
        written.append(str(p))
    if req.get("spots_csv") and req.get("spots"):
        p = out_dir / f"{stem}__spots.csv"
        rows = req["spots"]
        keys = [k for k in rows[0].keys()] if rows else []
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        written.append(str(p))
    if req.get("map_comparison_csv") and (req.get("map_comparison") or req.get("map_comparison_request")):
        mc = req.get('map_comparison') or map_overlay(ws, *req["map_comparison_request"])
        p = out_dir / f"{stem}__map_comparison.csv"
        xe, ye = mc["x_edges"], mc["y_edges"]
        prim, sec, rgba = mc["cells"]["primary"], mc["cells"]["secondary"], mc["cells"]["rgba"]
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["x_low", "x_high", "y_low", "y_high", "primary_recorded", "secondary_recorded", "red", "green", "blue", "coverage"])
            for r in range(len(prim)):
                for c in range(len(prim[r])):
                    w.writerow([xe[c], xe[c + 1], ye[r], ye[r + 1], prim[r][c], sec[r][c], *rgba[r][c]])
        written.append(str(p))
    if req.get("extra_csv"):
        for name, text in req["extra_csv"].items():
            p = out_dir / f"{stem}{name}"
            p.write_text(text, encoding="utf-8")
            written.append(str(p))
    return {"written": written, "folder": str(out_dir)}
