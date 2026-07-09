from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nvfit.io_mat import ExperimentTrace, load_saved_data_mat
from nvfit.preprocess import apply_roi, bin_trace, smooth_trace
from nvfit.pro_batch import (
    _extract_nv_metrics,
    _fit_non_ramsey,
    _plot_fit,
    _plot_scan1d,
    _plot_scan2d,
)
from nvfit.ramsey import fit_ramsey_physics_first


ROOT_DEFAULT = Path(r"D:\BacklundLabResearch\Data\Experiments\429_TDD")
OUT_DEFAULT = ROOT_DEFAULT / "analysis_outputs" / "429_TDD_full_characterization"

PRIORITY_TYPES = {
    "Rabi",
    "Ramsey",
    "SpinEcho",
    "T1",
    "DynamicDecoupling",
    "DEERFrequency",
    "DEERDuration",
    "DEERDecay",
    "DEERPosition",
}

LIGHT_CONTEXT_TYPES = {"ODMR", "LineScan"}


@dataclass
class AnalysisRecord:
    path: Path
    relative_path: str
    stem: str
    trace: ExperimentTrace
    fit: Any | None = None
    fit_error: str = ""
    status: str = "LOADED"
    priority: str = "context"
    quality_score: float = float("nan")
    figure: str = ""
    residual_figure: str = ""
    iteration_figure: str = ""
    fft_figure: str = ""
    notes: str = ""
    iteration_stats: dict[str, Any] | None = None


def _slug(text: str, max_len: int = 150) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text[:max_len] or "item"


def _safe_float(value: Any) -> float:
    try:
        f = float(value)
    except Exception:
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    f = _safe_float(value)
    if not math.isfinite(f):
        return ""
    if abs(f) >= 1000 or (abs(f) < 0.001 and f != 0):
        return f"{f:.{digits}e}"
    return f"{f:.{digits}g}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _rel(out_dir: Path, file_path: str | Path) -> str:
    path = Path(file_path)
    try:
        return path.relative_to(out_dir).as_posix()
    except Exception:
        return path.as_posix()


def _is_checkpoint(path: Path) -> bool:
    return "__checkpoint" in path.stem.lower()


def _is_alignment_like(record: AnalysisRecord) -> bool:
    text = f"{record.relative_path} {record.trace.metadata.get('scan_notes', '') if record.trace.metadata else ''}".lower()
    return any(k in text for k in ("align", "assignment", "magnetposition", "magnetalign", "rough", "stage_scan"))


def _record_priority(record: AnalysisRecord) -> str:
    if record.trace.experiment_type in PRIORITY_TYPES:
        return "sensing"
    if record.trace.experiment_type == "ODMR" and not _is_alignment_like(record):
        return "odmr_context"
    if record.trace.experiment_type in LIGHT_CONTEXT_TYPES:
        return "alignment_context" if _is_alignment_like(record) else "context"
    return "context"


def _finite_xy(trace: ExperimentTrace) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(trace.x_ns, dtype=float).ravel()
    y = np.asarray(trace.y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _normal_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    med = np.nanmedian(y)
    scale = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        scale = np.nanstd(y)
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        return y - med
    return (y - med) / scale


def _quality_score(record: AnalysisRecord) -> float:
    if record.fit is None:
        return -1.0
    r2 = _safe_float(record.fit.r2)
    rmse = _safe_float(record.fit.rmse)
    n = len(record.trace.x_ns)
    penalty = 0.0
    if _is_checkpoint(record):
        penalty += 0.10
    if _is_alignment_like(record):
        penalty += 0.20
    if math.isfinite(rmse):
        penalty += min(0.25, rmse)
    return (r2 if math.isfinite(r2) else 0.0) + min(n, 2000) / 200000.0 - penalty


def _fit_record(record: AnalysisRecord, multistart: int, odmr_multipeak: bool) -> None:
    trace = record.trace
    if not trace.fit_allowed or trace.scan_dim != "time":
        record.status = "PLOT_ONLY"
        return
    if record.priority == "alignment_context" and trace.experiment_type == "ODMR":
        record.status = "PLOT_ONLY"
        record.notes = "Alignment-like ODMR kept as context plot only."
        return
    x, y = _finite_xy(trace)
    if len(x) < 5:
        record.status = "SKIPPED"
        record.fit_error = "Too few finite points."
        return
    starts = multistart
    if record.priority == "odmr_context":
        starts = max(3, min(multistart, 5))
    elif record.priority == "context":
        starts = max(2, min(multistart, 4))
    try:
        if trace.experiment_type == "Ramsey":
            record.fit = fit_ramsey_physics_first(x, y).best
        else:
            record.fit = _fit_non_ramsey(
                trace.experiment_type,
                x,
                y,
                n_starts=starts,
                odmr_multipeak=(odmr_multipeak and record.priority != "alignment_context")
                or trace.experiment_type == "DEERFrequency",
                trace=trace,
            )
        record.status = "FIT_OK"
    except Exception as exc:
        record.status = "FIT_FAILED"
        record.fit_error = f"{type(exc).__name__}: {exc}"
    record.quality_score = _quality_score(record)


def _plot_record(record: AnalysisRecord, out_dir: Path, dpi: int) -> None:
    per_file = out_dir / "figures" / "per_file"
    per_file.mkdir(parents=True, exist_ok=True)
    file_base = _slug(record.relative_path.replace("\\", "__").replace("/", "__"))
    out_png = per_file / f"{file_base}.png"
    trace = record.trace
    x, y = _finite_xy(trace)
    title = f"{trace.experiment_type} | {record.relative_path}"
    try:
        if trace.scan_dim == "scan2d" and trace.x2d is not None and trace.y2d is not None and trace.z2d is not None:
            _plot_scan2d(out_png, trace.x2d, trace.y2d, trace.z2d, title, "Scan X", "Scan Y", dpi=dpi)
        elif record.fit is not None:
            _plot_fit(out_png, x, y, record.fit, title, trace.y_label, status=record.status, dpi=dpi)
        else:
            xlabel = trace.scan_axes[0] if trace.scan_axes else trace.x_label
            _plot_scan1d(out_png, x, y, title, xlabel, trace.y_label, dpi=dpi)
        record.figure = _rel(out_dir, out_png)
    except Exception as exc:
        record.fit_error = (record.fit_error + "; " if record.fit_error else "") + f"plot failed: {exc}"


def _iteration_stats(record: AnalysisRecord) -> dict[str, Any]:
    arr = record.trace.iteration_contrast
    if arr is None:
        return {}
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return {}
    valid_cols = np.isfinite(arr).sum(axis=0) >= max(3, int(0.2 * arr.shape[0]))
    arr = arr[:, valid_cols]
    if arr.size == 0:
        return {}
    iter_means = np.nanmean(arr, axis=0)
    iter_stds = np.nanstd(arr, axis=0)
    n_iter = int(arr.shape[1])
    drift = float(iter_means[-1] - iter_means[0]) if n_iter > 1 else float("nan")
    if n_iter > 2:
        slope = float(np.polyfit(np.arange(n_iter), iter_means, 1)[0])
    else:
        slope = float("nan")
    return {
        "relative_path": record.relative_path,
        "experiment_type": record.trace.experiment_type,
        "n_iterations": n_iter,
        "mean_contrast": float(np.nanmean(iter_means)),
        "std_iteration_mean": float(np.nanstd(iter_means)),
        "mean_within_iteration_std": float(np.nanmean(iter_stds)),
        "first_last_delta": drift,
        "drift_per_iteration": slope,
    }


def _plot_iteration_diagnostics(record: AnalysisRecord, out_dir: Path, dpi: int) -> dict[str, Any]:
    if record.iteration_stats is not None:
        return record.iteration_stats
    stats = _iteration_stats(record)
    if not stats:
        return {}
    arr = np.asarray(record.trace.iteration_contrast, dtype=float)
    valid_cols = np.isfinite(arr).sum(axis=0) >= max(3, int(0.2 * arr.shape[0]))
    arr = arr[:, valid_cols]
    x = np.asarray(record.trace.x_ns, dtype=float)
    fig_dir = out_dir / "figures" / "diagnostics" / "iterations"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"{_slug(record.relative_path.replace('\\', '__').replace('/', '__'))}_iterations.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    max_iter = min(arr.shape[1], 18)
    for idx in range(max_iter):
        axes[0].plot(x, arr[:, idx], lw=0.9, alpha=0.45)
    axes[0].plot(x, np.nanmean(arr, axis=1), color="black", lw=2.0, label="mean")
    axes[0].set_title("Per-iteration traces")
    axes[0].set_xlabel(record.trace.x_label)
    axes[0].set_ylabel("Contrast")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    means = np.nanmean(arr, axis=0)
    axes[1].plot(np.arange(1, len(means) + 1), means, "o-", ms=3)
    axes[1].axhline(np.nanmean(means), color="black", lw=0.8, ls="--")
    axes[1].set_title("Iteration mean contrast drift")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean contrast")
    axes[1].grid(alpha=0.25)
    fig.suptitle(record.relative_path, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    record.iteration_figure = _rel(out_dir, out_png)
    stats["figure"] = record.iteration_figure
    record.iteration_stats = stats
    return stats


def _plot_fft(record: AnalysisRecord, out_dir: Path, dpi: int) -> None:
    if record.trace.experiment_type not in {"Ramsey", "Rabi", "DEERDuration"}:
        return
    x, y = _finite_xy(record.trace)
    if len(x) < 16:
        return
    dx = np.diff(x)
    step = float(np.nanmedian(dx))
    if not np.isfinite(step) or step <= 0:
        return
    y0 = y - np.nanmean(y)
    win = np.hanning(len(y0))
    spec = np.abs(np.fft.rfft(y0 * win))
    freq = np.fft.rfftfreq(len(y0), d=step)
    if len(freq) <= 1:
        return
    fig_dir = out_dir / "figures" / "diagnostics" / "fft"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"{_slug(record.relative_path.replace('\\', '__').replace('/', '__'))}_fft.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(freq[1:] * 1000.0, spec[1:], lw=1.4)
    ax.set_title(f"FFT diagnostic | {record.relative_path}", fontsize=10)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Amplitude [a.u.]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    record.fft_figure = _rel(out_dir, out_png)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _inventory_row(record: AnalysisRecord) -> dict[str, Any]:
    md = record.trace.metadata or {}
    return {
        "relative_path": record.relative_path,
        "file": record.path.name,
        "experiment_type": record.trace.experiment_type,
        "priority": record.priority,
        "scan_dim": record.trace.scan_dim,
        "fit_allowed": record.trace.fit_allowed,
        "n_points": len(record.trace.x_ns),
        "x_label": record.trace.x_label,
        "y_label": record.trace.y_label,
        "is_checkpoint": _is_checkpoint(record.path),
        "is_alignment_like": _is_alignment_like(record),
        "scan_notes": md.get("scan_notes", ""),
        "rf_frequency_ghz": md.get("rf_frequency_ghz", ""),
        "rf2_frequency_ghz": md.get("rf2_frequency_ghz", ""),
        "rf2_duration_ns": md.get("rf2_duration_ns", ""),
        "pi_pulse_ns": md.get("pi_pulse_ns", ""),
        "tau_ns": md.get("tau_ns", ""),
        "dd_pulse_count": md.get("dd_pulse_count", ""),
        "completed_iterations": md.get("completed_iterations", ""),
        "target_iterations": md.get("target_iterations", ""),
        "figure": record.figure,
    }


def _fit_row(record: AnalysisRecord) -> dict[str, Any]:
    md = record.trace.metadata or {}
    row = {
        "relative_path": record.relative_path,
        "experiment_type": record.trace.experiment_type,
        "priority": record.priority,
        "status": record.status,
        "quality_score": record.quality_score,
        "fit_error": record.fit_error,
        "notes": record.notes,
        "figure": record.figure,
        "iteration_figure": record.iteration_figure,
        "fft_figure": record.fft_figure,
        "scan_notes": md.get("scan_notes", ""),
    }
    if record.fit is not None:
        row.update(
            {
                "model": record.fit.model_name,
                "r2": record.fit.r2,
                "rmse": record.fit.rmse,
                "aic": record.fit.aic,
                "bic": record.fit.bic,
                "durbin_watson": record.fit.durbin_watson,
                "params_json": json.dumps(
                    {n: float(v) for n, v in zip(record.fit.param_names, record.fit.params)},
                    sort_keys=True,
                ),
                "errors_json": json.dumps(
                    {n: float(v) for n, v in zip(record.fit.param_names, record.fit.errors)},
                    sort_keys=True,
                ),
                "nv_metrics_json": json.dumps(
                    _json_safe(_extract_nv_metrics(record.trace.experiment_type, record.fit, md)),
                    sort_keys=True,
                ),
            }
        )
    return row


def _top_records(records: list[AnalysisRecord], exp_type: str, n: int = 8) -> list[AnalysisRecord]:
    subset = [r for r in records if r.trace.experiment_type == exp_type and r.fit is not None]
    subset.sort(key=lambda r: r.quality_score, reverse=True)
    return subset[:n]


def _plot_overlay(records: list[AnalysisRecord], exp_type: str, out_dir: Path, dpi: int, n: int = 10) -> str:
    selected = _top_records(records, exp_type, n=n)
    if not selected:
        return ""
    fig_dir = out_dir / "figures" / "curated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"{_slug(exp_type)}_top_overlay.png"
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for r in selected:
        x, y = _finite_xy(r.trace)
        label = Path(r.relative_path).stem
        if len(label) > 42:
            label = label[:39] + "..."
        ax.plot(x, _normal_y(y), "o-", ms=2.4, lw=1.0, alpha=0.75, label=label)
    ax.set_title(f"{exp_type}: top normalized traces")
    ax.set_xlabel(selected[0].trace.x_label)
    ax.set_ylabel("Normalized contrast")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return _rel(out_dir, out_png)


def _plot_metric_summary(records: list[AnalysisRecord], out_dir: Path, dpi: int) -> str:
    metrics: dict[str, list[float]] = {}
    for r in records:
        if r.fit is None:
            continue
        nv = _extract_nv_metrics(r.trace.experiment_type, r.fit, r.trace.metadata or {})
        for key in ("pi_time_ns", "pi_over_2_time_ns", "T2_star_ns", "T2_ns", "T2_dd_ns", "T1_ns", "detuning_MHz"):
            if key in nv:
                val = _safe_float(nv[key])
                if math.isfinite(val):
                    metrics.setdefault(key, []).append(val)
    if not metrics:
        return ""
    fig_dir = out_dir / "figures" / "curated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / "nv_metric_distributions.png"
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, max(2.2 * n, 4.0)))
    if n == 1:
        axes = [axes]
    for ax, (key, vals) in zip(axes, sorted(metrics.items())):
        ax.hist(vals, bins=min(18, max(5, len(vals) // 2)), color="#1976d2", alpha=0.75)
        ax.axvline(np.nanmedian(vals), color="black", lw=1.2, ls="--", label=f"median {_fmt(np.nanmedian(vals))}")
        ax.set_title(key)
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return _rel(out_dir, out_png)


def _plot_odmr_timeline(records: list[AnalysisRecord], out_dir: Path, dpi: int) -> str:
    rows = []
    for r in records:
        if r.trace.experiment_type != "ODMR" or r.fit is None:
            continue
        nv = _extract_nv_metrics("ODMR", r.fit, r.trace.metadata or {})
        center = _safe_float(nv.get("center_freq_units"))
        if math.isfinite(center):
            rows.append((r.relative_path, center, _safe_float(r.fit.r2), _is_alignment_like(r)))
    if len(rows) < 2:
        return ""
    fig_dir = out_dir / "figures" / "curated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / "odmr_center_timeline.png"
    x = np.arange(len(rows))
    colors = ["#9e9e9e" if row[3] else "#d81b60" for row in rows]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.scatter(x, [row[1] for row in rows], c=colors, s=18)
    ax.set_title("ODMR fitted center progression")
    ax.set_xlabel("ODMR file order")
    ax.set_ylabel("Center frequency [scan units]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return _rel(out_dir, out_png)


def _noise_spectrum_rows(records: list[AnalysisRecord]) -> list[dict[str, Any]]:
    rows = []
    pattern = re.compile(r"NoiseSpectrum_tau(?P<tau>[0-9]+)ns_f(?P<freq>[0-9p.]+)MHz_XY8_(?P<n>[0-9]+)", re.IGNORECASE)
    for record in records:
        match = pattern.search(record.path.stem)
        if not match:
            continue
        tau_ns = float(match.group("tau"))
        freq_mhz = float(match.group("freq").replace("p", "."))
        xy8_n = int(match.group("n"))
        _, y = _finite_xy(record.trace)
        rows.append(
            {
                "relative_path": record.relative_path,
                "tau_ns": tau_ns,
                "filter_frequency_mhz": freq_mhz,
                "xy8_repetitions": xy8_n,
                "mean_contrast": float(np.nanmean(y)) if len(y) else float("nan"),
                "std_contrast": float(np.nanstd(y)) if len(y) else float("nan"),
                "n_points": int(len(y)),
                "figure": record.figure,
            }
        )
    rows.sort(key=lambda r: (r["filter_frequency_mhz"], r["xy8_repetitions"]))
    return rows


def _plot_noise_spectrum(records: list[AnalysisRecord], out_dir: Path, dpi: int) -> tuple[str, list[dict[str, Any]]]:
    rows = _noise_spectrum_rows(records)
    if not rows:
        return "", []
    fig_dir = out_dir / "figures" / "curated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / "noise_spectrum_xy8_summary.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    xy8_values = sorted({int(r["xy8_repetitions"]) for r in rows})
    for xy8 in xy8_values:
        group = [r for r in rows if int(r["xy8_repetitions"]) == xy8]
        group.sort(key=lambda r: float(r["filter_frequency_mhz"]))
        ax.plot(
            [float(r["filter_frequency_mhz"]) for r in group],
            [float(r["mean_contrast"]) for r in group],
            "o-",
            ms=4,
            lw=1.2,
            label=f"XY8-{xy8}",
        )
    ax.set_title("Single-point XY8 noise spectrum response")
    ax.set_xlabel("Filter frequency [MHz]")
    ax.set_ylabel("Mean contrast")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return _rel(out_dir, out_png), rows


def _summaries(records: list[AnalysisRecord]) -> dict[str, Any]:
    by_type: dict[str, list[AnalysisRecord]] = {}
    for r in records:
        by_type.setdefault(r.trace.experiment_type, []).append(r)
    summary: dict[str, Any] = {
        "total_files": len(records),
        "fit_ok": sum(1 for r in records if r.status == "FIT_OK"),
        "fit_failed": sum(1 for r in records if r.status == "FIT_FAILED"),
        "skipped": sum(1 for r in records if r.status == "SKIPPED"),
        "plot_only": sum(1 for r in records if r.status == "PLOT_ONLY"),
        "checkpoints": sum(1 for r in records if _is_checkpoint(r.path)),
        "alignment_like": sum(1 for r in records if _is_alignment_like(r)),
        "by_type": {},
    }
    for exp, items in sorted(by_type.items()):
        fits = [r for r in items if r.fit is not None]
        r2s = [_safe_float(r.fit.r2) for r in fits]
        r2s = [v for v in r2s if math.isfinite(v)]
        summary["by_type"][exp] = {
            "n": len(items),
            "fit_ok": len(fits),
            "median_r2": float(np.nanmedian(r2s)) if r2s else None,
            "best": _top_records(records, exp, 1)[0].relative_path if _top_records(records, exp, 1) else "",
        }
    return summary


def _metric_table(records: list[AnalysisRecord]) -> list[dict[str, Any]]:
    rows = []
    for exp in sorted({r.trace.experiment_type for r in records}):
        top = _top_records(records, exp, 5)
        for rank, r in enumerate(top, 1):
            nv = _extract_nv_metrics(r.trace.experiment_type, r.fit, r.trace.metadata or {}) if r.fit else {}
            row = {
                "rank": rank,
                "experiment_type": exp,
                "relative_path": r.relative_path,
                "model": r.fit.model_name if r.fit else "",
                "r2": r.fit.r2 if r.fit else "",
                "quality_score": r.quality_score,
            }
            for key in ("pi_time_ns", "pi_over_2_time_ns", "T2_star_ns", "T2_ns", "T2_dd_ns", "T1_ns", "detuning_MHz", "center_freq_units", "fwhm_units"):
                if key in nv:
                    row[key] = nv[key]
            rows.append(row)
    return rows


def _make_report(
    records: list[AnalysisRecord],
    out_dir: Path,
    overlay_figures: dict[str, str],
    metric_fig: str,
    odmr_fig: str,
    noise_fig: str,
    summary: dict[str, Any],
) -> None:
    top_rows = _metric_table(records)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines = [
        "# 429_TDD NV Sensing Diamond Characterization",
        "",
        f"Generated: {generated}",
        "",
        "## Executive Summary",
        "",
        f"- Loaded {summary['total_files']} `.mat` files; {summary['fit_ok']} fit successfully, {summary['plot_only']} were plot-only spatial/context scans, {summary['skipped']} were single-point/non-fit traces, and {summary['fit_failed']} fit attempts failed.",
        f"- The analysis intentionally prioritizes sensing sequences over repetitive alignment data. {summary['alignment_like']} files were flagged as alignment-like context and {summary['checkpoints']} files were checkpoints.",
        "- The strongest scientific sections are Rabi calibration, Ramsey detuning/dephasing, spin echo and XY8 coherence, T1, DEER frequency/duration/decay scans, and representative ODMR context.",
        "",
        "## Dataset Composition",
        "",
        "| Experiment type | Files | Fits | Median R2 | Best example |",
        "|---|---:|---:|---:|---|",
    ]
    for exp, data in summary["by_type"].items():
        md_lines.append(
            f"| {exp} | {data['n']} | {data['fit_ok']} | {_fmt(data['median_r2'])} | `{data['best']}` |"
        )
    md_lines += [
        "",
        "## Curated Comparison Figures",
        "",
    ]
    for exp in ("Rabi", "Ramsey", "SpinEcho", "T1", "DynamicDecoupling", "DEERFrequency", "DEERDuration", "DEERDecay", "ODMR"):
        fig = overlay_figures.get(exp)
        if fig:
            md_lines += [f"### {exp}", "", f"![{exp}]({fig})", ""]
    if metric_fig:
        md_lines += ["### NV Metric Distributions", "", f"![NV metrics]({metric_fig})", ""]
    if odmr_fig:
        md_lines += ["### ODMR Context", "", f"![ODMR center timeline]({odmr_fig})", ""]
    if noise_fig:
        md_lines += [
            "### XY8 Noise Spectrum Single-Point Summary",
            "",
            f"![XY8 noise spectrum]({noise_fig})",
            "",
            "These files are not decay curves, so they are intentionally summarized as contrast response versus filter frequency and XY8 repetition count rather than forced into a stretched-exponential fit.",
            "",
        ]

    md_lines += [
        "## Best Fits And Extracted Metrics",
        "",
        "| Type | Rank | File | Model | R2 | Key metrics |",
        "|---|---:|---|---|---:|---|",
    ]
    for row in top_rows:
        metric_bits = []
        for key, value in row.items():
            if key in {"rank", "experiment_type", "relative_path", "model", "r2", "quality_score"}:
                continue
            metric_bits.append(f"{key}={_fmt(value)}")
        md_lines.append(
            f"| {row['experiment_type']} | {row['rank']} | `{row['relative_path']}` | {row['model']} | {_fmt(row['r2'])} | {'; '.join(metric_bits)} |"
        )

    md_lines += [
        "",
        "## Scientific Reading Guide",
        "",
        "- Use the top overlays to compare reproducibility across runs. Large shape changes inside one sequence family are more scientifically meaningful than small fit-score differences.",
        "- Use the iteration diagnostic figures to separate sample physics from acquisition drift, refocus/optimization failures, or late-run instability.",
        "- Treat low-R2 Rabi and DEER-duration fits as qualitative when the trace has beating, phase drift, or weak oscillation contrast; those files are still useful for locating regimes and controls.",
        "- Treat repetitive magnet-alignment ODMR files as context for how the final resonance condition was reached, not as independent sample-characterization measurements.",
        "",
        "## Output Index",
        "",
        "- `tables/inventory.csv`: every loaded file with metadata, priority, and figure links.",
        "- `tables/fit_summary.csv`: fit model, quality metrics, parameters, and NV-derived metrics.",
        "- `tables/iteration_summary.csv`: per-iteration drift and repeatability diagnostics.",
        "- `tables/best_metrics.csv`: compact table of the highest-ranked files by sequence type.",
        "- `figures/per_file/`: one plot per file.",
        "- `figures/curated/`: report-level overlays and metric summaries.",
        "- `figures/diagnostics/`: FFT and per-iteration stability plots.",
    ]

    report_md = out_dir / "report_429_TDD.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    body = []
    for line in md_lines:
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line:
            alt = line[2 : line.find("]")]
            src = line[line.find("(") + 1 : line.rfind(")")]
            body.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("- "):
            body.append(f"<p>{html.escape(line)}</p>")
        elif line.startswith("|"):
            continue
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
        else:
            body.append("")
    css = """
    body { font-family: Arial, sans-serif; max-width: 1180px; margin: 32px auto; line-height: 1.45; color: #202124; }
    h1, h2, h3 { color: #111827; }
    img { max-width: 100%; border: 1px solid #ddd; }
    figure { margin: 18px 0 28px; }
    figcaption { color: #555; font-size: 13px; }
    p { margin: 8px 0; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
    """
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><title>429_TDD report</title><style>{css}</style></head><body>{''.join(body)}</body></html>"
    (out_dir / "report_429_TDD.html").write_text(html_doc, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("*.mat"))
    records: list[AnalysisRecord] = []
    print(f"Loading {len(files)} .mat files from {root}")
    for idx, path in enumerate(files, 1):
        rel = str(path.relative_to(root))
        try:
            trace = load_saved_data_mat(path, mode=args.mode)
            record = AnalysisRecord(path=path, relative_path=rel, stem=_slug(rel), trace=trace)
            record.priority = _record_priority(record)
            records.append(record)
        except Exception as exc:
            print(f"[load failed] {rel}: {exc}")
        if idx % 25 == 0:
            print(f"  loaded {idx}/{len(files)}")

    print("Fitting and plotting records")
    for idx, record in enumerate(records, 1):
        _fit_record(record, args.multistart, args.odmr_multipeak)
        _plot_record(record, out_dir, args.dpi)
        _plot_fft(record, out_dir, args.dpi)
        if record.priority == "sensing":
            _plot_iteration_diagnostics(record, out_dir, args.dpi)
        if idx % 20 == 0:
            print(f"  processed {idx}/{len(records)}")

    print("Writing tables and curated figures")
    inventory_rows = [_inventory_row(r) for r in records]
    fit_rows = [_fit_row(r) for r in records]
    iter_rows = [_plot_iteration_diagnostics(r, out_dir, args.dpi) for r in records if r.priority == "sensing"]
    iter_rows = [r for r in iter_rows if r]
    best_rows = _metric_table(records)

    _write_csv(out_dir / "tables" / "inventory.csv", inventory_rows)
    _write_csv(out_dir / "tables" / "fit_summary.csv", fit_rows)
    _write_csv(out_dir / "tables" / "iteration_summary.csv", iter_rows)
    _write_csv(out_dir / "tables" / "best_metrics.csv", best_rows)
    (out_dir / "tables" / "summary.json").write_text(json.dumps(_json_safe(_summaries(records)), indent=2), encoding="utf-8")

    overlay_figures = {}
    for exp in ("Rabi", "Ramsey", "SpinEcho", "T1", "DynamicDecoupling", "DEERFrequency", "DEERDuration", "DEERDecay", "ODMR"):
        fig = _plot_overlay(records, exp, out_dir, args.dpi, n=args.overlay_count)
        if fig:
            overlay_figures[exp] = fig
    metric_fig = _plot_metric_summary(records, out_dir, args.dpi)
    odmr_fig = _plot_odmr_timeline(records, out_dir, args.dpi)
    noise_fig, noise_rows = _plot_noise_spectrum(records, out_dir, args.dpi)
    _write_csv(out_dir / "tables" / "noise_spectrum_summary.csv", noise_rows)
    summary = _summaries(records)
    _make_report(records, out_dir, overlay_figures, metric_fig, odmr_fig, noise_fig, summary)

    print(f"Done. Report: {out_dir / 'report_429_TDD.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full 429_TDD scientific characterization report")
    parser.add_argument("--input-dir", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--mode", choices=["contrast", "signal", "reference", "difference"], default="contrast")
    parser.add_argument("--multistart", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--overlay-count", type=int, default=10)
    parser.add_argument("--odmr-multipeak", action="store_true", default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
