from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from .diagnostics import aic, bic, durbin_watson, r_squared, rmse
from .analysis_tools import process_series
from .fit_engine import FitResult, fit_model_multistart
from .fit_workflows import FitWorkflowConfig, fit_non_ramsey
from .io_mat import DETECTOR_LABELS, ExperimentTrace, load_experiment_dataset
from .models import (
    MODEL_REGISTRY,
    build_custom_expression_model,
    spin_echo_model,
)
from .preprocess import smooth_trace
from .profiles import PROFILES, classify_status
from .rabi_utils import build_rabi_nv_metrics, rabi_envelope_bounds, rabi_envelope_metrics
from .ramsey import estimate_ramsey_frequency, fit_ramsey_physics_first


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scaled_perturbation(lower, upper, frac: float = 0.05):
    """Return per-parameter perturbation scale = frac * (upper - lower)."""
    return [frac * (u - l) for l, u in zip(lower, upper)]


def _spin_echo_seed_and_bounds(exp_type: str, tspan: float) -> tuple[list[float], list[float], np.ndarray]:
    lower = [-0.2, -0.2, 1.0, 0.5]
    if exp_type == "DynamicDecoupling":
        upper = [0.2, 0.2, max(200000.0, 8.0 * tspan), 5.0]
        seed0 = np.array([0.0, 0.0, max(1000.0, tspan / 3.0), 1.5])
    else:
        upper = [0.2, 0.2, 20000.0, 5.0]
        seed0 = np.array([0.0, 0.0, max(100.0, tspan / 3.0), 1.5])
    return lower, upper, seed0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_non_ramsey(
    exp_type: str,
    x: np.ndarray,
    y: np.ndarray,
    n_starts: int = 12,
    custom_spec: dict | None = None,
    odmr_multipeak: bool = False,
    *,
    trace=None,
) -> FitResult:
    return fit_non_ramsey(
        exp_type,
        x,
        y,
        config=FitWorkflowConfig(
            multistart=n_starts,
            rabi_mode="Single reliable Rabi" if exp_type == "Rabi" else "Classic single (recommended)",
            odmr_multipeak=odmr_multipeak,
            odmr_peak_count=4,
            odmr_auto_peak=True,
            odmr_polarity="Auto",
            custom_spec=custom_spec,
        ),
        custom_spec=custom_spec,
        trace_experiment_type=exp_type,
        trace=trace,
    )


def _fit_odmr_multipeak(x: np.ndarray, y: np.ndarray, n_starts: int, rng) -> FitResult:
    """Fit ODMR with automatic multi-Lorentzian peak detection."""
    yamp = 0.5 * (np.percentile(y, 95) - np.percentile(y, 5))
    tspan = float(np.max(x) - np.min(x))
    y_peak = -y  # ODMR dips → flip for peak detection
    prominence = max(1e-9, 0.08 * float(np.ptp(y_peak)))
    pks, _ = find_peaks(y_peak, prominence=prominence)
    if len(pks) == 0:
        pks = np.array([np.argmin(y)], dtype=int)
    else:
        ord_idx = np.argsort(y_peak[pks])[::-1]
        n_peaks = int(max(1, min(8, len(pks))))
        pks = pks[ord_idx[:n_peaks]]
        pks = np.sort(pks)

    names = ["y0"]
    lower = [float(np.min(y))]
    upper = [float(np.max(y))]
    seed = [float(np.max(y))]
    span = max(1e-9, tspan)

    for pk in pks:
        c0 = -abs(0.5 * yamp)
        w0 = max(1e-6, 0.03 * span)
        x0i = float(x[pk])
        idx = len(names) // 3 + 1
        names += [f"C{idx}", f"w{idx}", f"x0_{idx}"]
        seed += [c0, w0, x0i]
        lower += [-1.5 * abs(yamp) - 1.0, 1e-8, float(np.min(x))]
        upper += [0.0, span, float(np.max(x))]

    scale = _scaled_perturbation(lower, upper)
    seeds = [np.asarray(seed) + rng.normal(0, scale) for _ in range(n_starts)]
    return fit_model_multistart(
        model_name="ODMRMulti", model_func=odmr_multi_lorentzian, x=x, y=y,
        param_names=names, lower=lower, upper=upper, seeds=seeds,
    )


def _run_robust_fit(
    exp_type: str,
    x: np.ndarray,
    y: np.ndarray,
    n_starts: int = 12,
    custom_spec: dict | None = None,
    odmr_multipeak: bool = False,
    progress_callback=None,
) -> FitResult:
    """Robust fit: run fits on multiple data subsets, compare all on full data."""

    # 1. Baseline fit on full data
    if exp_type == "Ramsey":
        sel = fit_ramsey_physics_first(x, y)
        base_res = sel.best
    else:
        base_res = _fit_non_ramsey(exp_type, x, y, n_starts=n_starts,
                                   custom_spec=custom_spec, odmr_multipeak=odmr_multipeak)

    # 2. Data subsets
    n = len(x)
    cuts = [
        ("full",         0,              0),
        ("trim_start4",  int(n * 0.04),  0),
        ("trim_start8",  int(n * 0.08),  0),
        ("trim_end5",    0,              int(n * 0.05)),
        ("trim_both4",   int(n * 0.04),  int(n * 0.04)),
        ("trim_start12", int(n * 0.12),  0),
        ("trim_end10",   0,              int(n * 0.10)),
    ]

    candidates = [("full", base_res, base_res)]
    sub_starts = max(1, n_starts // 2)

    for idx, (label, onset, offset) in enumerate(cuts[1:]):
        if n - onset - offset < 10:
            continue
        if progress_callback:
            progress_callback(f"Robust: fitting subset {idx+1}/{len(cuts)-1} ({label})...")
        xv = x[onset: n - offset if offset else n]
        yv = y[onset: n - offset if offset else n]
        try:
            if exp_type == "Ramsey":
                sub_sel = fit_ramsey_physics_first(xv, yv)
                sub_res = sub_sel.best
            else:
                sub_res = _fit_non_ramsey(exp_type, xv, yv, n_starts=sub_starts,
                                         custom_spec=custom_spec, odmr_multipeak=odmr_multipeak)
            # Re-evaluate on full data using the actual model from the result
            actual_model = MODEL_REGISTRY.get(sub_res.model_name)
            if actual_model is not None:
                x_eval = x
                if isinstance(sub_res.extras, dict) and sub_res.extras.get("x_offset_ns") is not None and sub_res.model_name in {"RabiPhaseRamp", "RabiCosine", "RabiChirp", "RabiLongDamped"}:
                    x_eval = x - float(sub_res.extras["x_offset_ns"])
                y_fit_full = actual_model(x_eval, *sub_res.params)
                full_eval = FitResult(
                    model_name=sub_res.model_name,
                    param_names=sub_res.param_names,
                    params=sub_res.params,
                    errors=sub_res.errors,
                    y_fit=y_fit_full,
                    r2=r_squared(y, y_fit_full),
                    rmse=rmse(y, y_fit_full),
                    aic=aic(y, y_fit_full, len(sub_res.params)),
                    bic=bic(y, y_fit_full, len(sub_res.params)),
                    durbin_watson=durbin_watson(y - y_fit_full),
                    bound_hits=sub_res.bound_hits,
                    success=sub_res.success,
                    message=f"{sub_res.message} (robust subset: {label})",
                    extras=dict(sub_res.extras),
                )
            else:
                full_eval = sub_res
            candidates.append((label, sub_res, full_eval))
        except Exception:
            continue

    # 3. Pick best by full-data R²
    best_label, _, best_full = max(candidates, key=lambda c: c[2].r2)

    # Build summary string
    summary_lines = ["Robust fit candidate comparison (all evaluated on full data):"]
    summary_lines.append(f"{'Subset':<16} {'R²(sub)':>10} {'R²(full)':>10} {'RMSE':>12} {'BIC':>10}")
    summary_lines.append("-" * 62)
    for label, sub_res, full_eval in candidates:
        marker = " ★" if label == best_label else ""
        summary_lines.append(
            f"{label:<16} {sub_res.r2:>10.4f} {full_eval.r2:>10.4f} "
            f"{full_eval.rmse:>12.6g} {full_eval.bic:>10.2f}{marker}"
        )
    summary_lines.append(f"\nBest subset: {best_label}")

    # Attach summary to result message
    best_full_copy = FitResult(
        model_name=best_full.model_name,
        param_names=best_full.param_names,
        params=best_full.params,
        errors=best_full.errors,
        y_fit=best_full.y_fit,
        r2=best_full.r2,
        rmse=best_full.rmse,
        aic=best_full.aic,
        bic=best_full.bic,
        durbin_watson=best_full.durbin_watson,
        bound_hits=best_full.bound_hits,
        success=best_full.success,
        message=best_full.message + "\n" + "\n".join(summary_lines),
        extras=dict(best_full.extras),
    )
    return best_full_copy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PARAM_UNITS = {
    "y0": "arb.",  "A": "arb.",  "m": "1/ns",  "C": "arb.",
    "f": "GHz",  "f1": "GHz",  "f2": "GHz",
    "phi": "rad",  "phi1": "rad",  "phi2": "rad",
    "tau": "ns",  "T2": "ns",  "T2_star": "ns",
    "n": "",  "w": "",  "x0": "",
}


def _format_param_line(name: str, val: float, err: float) -> str:
    unit = _PARAM_UNITS.get(name, "")
    unit_str = f" {unit}" if unit else ""
    return f"  {name} = {val:.4g}{unit_str} ± {err:.2g}"


def _plot_fit(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    result: FitResult,
    title: str,
    y_label: str,
    status: str = "",
    reason: str = "",
    dpi: int = 180,
):
    """Create publication-quality fit plot with stats annotation."""
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.plot(x, y, "o", ms=3.0, alpha=0.55, color="#607d8b", label="Data")
    ax.plot(x, result.y_fit, "-", lw=2.0, color="#e53935", label=f"Fit: {result.model_name}")
    if isinstance(result.extras, dict):
        bounds = rabi_envelope_bounds(x, result.extras.get("rabi_envelope"))
        if bounds is not None:
            upper, lower = bounds
            ax.plot(x, upper, "--", lw=1.4, color="#7b1fa2", alpha=0.9, label="Rabi envelope")
            ax.plot(x, lower, "--", lw=1.4, color="#7b1fa2", alpha=0.9, label="_nolegend_")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    # Stats annotation box
    stat_lines = [
        f"R² = {result.r2:.4f}",
        f"RMSE = {result.rmse:.4g}",
        f"BIC = {result.bic:.1f}",
    ]
    if status:
        stat_lines.append(f"Status: {status}")
    stat_lines.append("")
    for n, v, e in zip(result.param_names, result.params, result.errors):
        stat_lines.append(_format_param_line(n, v, e))

    stats_text = "\n".join(stat_lines)
    props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#bdbdbd")
    ax.text(
        0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=7.5, verticalalignment="top", horizontalalignment="right",
        fontfamily="monospace", bbox=props,
    )

    # Residual subplot
    residual = y - result.y_fit
    axr.plot(x, residual, ".", ms=2.5, color="#78909c")
    axr.axhline(0.0, ls="--", lw=0.8, color="k")
    axr.set_xlabel("Pulse duration (ns)")
    axr.set_ylabel("Residual")
    axr.grid(True, alpha=0.25)

    fig.subplots_adjust(hspace=0.08, left=0.10, right=0.95, top=0.94, bottom=0.08)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_scan1d(out_path: Path, x: np.ndarray, y: np.ndarray,
                  title: str, x_label: str, y_label: str, dpi: int = 180):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, "o-", ms=3.5, lw=1.2, color="#1976d2")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_scan2d(out_path: Path, x2d: np.ndarray, y2d: np.ndarray,
                  z2d: np.ndarray, title: str, x_label: str, y_label: str,
                  dpi: int = 180):
    fig, ax = plt.subplots(figsize=(9, 7))
    # Percentile normalization for better contrast
    vmin = np.nanpercentile(z2d, 2)
    vmax = np.nanpercentile(z2d, 98)
    mesh = ax.pcolormesh(x2d, y2d, z2d, shading="auto", cmap="inferno",
                         vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, label="Signal", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# NV metrics extraction
# ---------------------------------------------------------------------------

def _extract_nv_metrics(exp_type: str, result: FitResult, trace_metadata: dict | None = None) -> dict[str, object]:
    semantic_exp_type = exp_type
    if exp_type == "DEERFrequency":
        exp_type = "ODMR"
    elif exp_type == "DEERDuration":
        exp_type = "Rabi"
    elif exp_type == "DEERDecay":
        exp_type = "SpinEcho"
    p = {n: float(v) for n, v in zip(result.param_names, result.params)}
    e = {n: float(v) for n, v in zip(result.param_names, result.errors)}
    metrics: dict[str, object] = {}
    if exp_type == "Ramsey":
        if "tau" in p:
            metrics["T2_star_ns"] = p["tau"]
        if "T2_star" in p:
            metrics["T2_star_ns"] = p["T2_star"]
        if "n" in p:
            metrics["stretch_n"] = p["n"]
        if "f" in p:
            metrics["detuning_MHz"] = p["f"] * 1000.0
    elif exp_type == "Rabi":
        physical_params = p
        if isinstance(result.extras, dict) and result.extras.get("rabi_physical_reference_params"):
            physical_params = {k: float(v) for k, v in dict(result.extras.get("rabi_physical_reference_params", {})).items()}
        metrics.update(build_rabi_nv_metrics(physical_params, metadata=trace_metadata, errors=e))
        if isinstance(result.extras, dict):
            metrics.update(rabi_envelope_metrics(result.extras.get("rabi_envelope")))
        if "tau" in p:
            metrics["rabi_decay_ns"] = p["tau"]
        if "t0" in p:
            metrics["t0_ns"] = p["t0"]
    elif exp_type in ("SpinEcho", "T1", "DynamicDecoupling"):
        if "T2" in p:
            if exp_type == "SpinEcho":
                metrics["T2_ns"] = p["T2"]
            else:
                metrics["T2_dd_ns"] = p["T2"]
        if "T1" in p and exp_type == "T1":
            metrics["T1_ns"] = p["T1"]
        if "n" in p:
            metrics["stretch_n"] = p["n"]
        if exp_type == "T1":
            prep_mode = result.extras.get("t1_prep_mode") if isinstance(result.extras, dict) else None
            family = result.extras.get("t1_model_family") if isinstance(result.extras, dict) else None
            if prep_mode:
                metrics["t1_prep_mode"] = str(prep_mode)
            if family:
                metrics["t1_model_family"] = str(family)
        if exp_type == "DynamicDecoupling" and trace_metadata is not None:
            dd_count = trace_metadata.get("dd_pulse_count")
            rf_ghz = trace_metadata.get("rf_frequency_ghz")
            pi_ns = trace_metadata.get("pi_pulse_ns")
            if dd_count is not None:
                metrics["dd_pulse_count"] = float(dd_count)
            if rf_ghz is not None:
                metrics["rf_frequency_GHz"] = float(rf_ghz)
            if pi_ns is not None:
                metrics["pi_pulse_ns"] = float(pi_ns)
    elif exp_type == "ODMR":
        if "x0" in p:
            metrics["center_freq_units"] = p["x0"]
        if "w" in p:
            metrics["fwhm_units"] = 2.0 * p["w"]
        # Multi-peak metrics
        centers = [p[k] for k in sorted(p) if k.startswith("x0_")]
        if centers:
            metrics["num_peaks"] = float(len(centers))
            metrics["center_mean"] = float(np.mean(centers))
    if semantic_exp_type.startswith("DEER") and trace_metadata is not None:
        metrics["semantic_family"] = semantic_exp_type
        for key in ("rf_frequency_ghz", "rf2_frequency_ghz", "rf2_duration_ns", "pi_pulse_ns", "tau_ns"):
            if trace_metadata.get(key) is not None:
                metrics[key] = trace_metadata[key]
    return metrics


# ---------------------------------------------------------------------------
# Result-row builder (eliminates code duplication)
# ---------------------------------------------------------------------------

def _build_result_row(
    mat_file: Path,
    exp_type: str,
    profile,
    result: FitResult | None,
    status: str,
    reason: str,
    file_meta: dict,
    trace_metadata: dict,
    provenance: dict,
    model_note: str = "",
    relative_path: str = "",
    scan_dim: str = "",
    fit_allowed: bool = True,
) -> tuple[dict, dict, dict]:
    """Build (result_json, summary_row, validation_row) triple."""
    stem = mat_file.stem
    warnings_value = trace_metadata.get("warnings", "")
    if isinstance(warnings_value, list):
        warnings_text = "; ".join(str(v) for v in warnings_value)
    else:
        warnings_text = str(warnings_value) if warnings_value else ""
    completion = ""
    if trace_metadata.get("completed_iterations") is not None or trace_metadata.get("target_iterations") is not None:
        completion = f"{trace_metadata.get('completed_iterations', '')}/{trace_metadata.get('target_iterations', '')}"
    detector_id = str(trace_metadata.get("detector_id") or provenance.get("detector_id") or "detector1")
    detector_label = str(
        trace_metadata.get("detector_label")
        or provenance.get("detector_label")
        or DETECTOR_LABELS.get(detector_id, detector_id)
    )
    detector_count = int(trace_metadata.get("detector_count") or provenance.get("detector_count") or 1)

    if result is not None:
        nv = _extract_nv_metrics(exp_type, result, trace_metadata)
        result_json = {
            "file": mat_file.name,
            "relative_path": relative_path,
            "detector_id": detector_id,
            "detector_label": detector_label,
            "detector_count": detector_count,
            "experiment_type": exp_type,
            "scan_family": exp_type,
            "scan_dim": scan_dim,
            "fit_allowed": fit_allowed,
            "profile": profile.name,
            "selected_model": result.model_name,
            "selection_note": result.selection_note or model_note,
            "metrics": {
                "r2": result.r2,
                "rmse": result.rmse,
                "aic": result.aic,
                "bic": result.bic,
                "durbin_watson": result.durbin_watson,
                "success": result.success,
                "message": result.message,
            },
            "params": {n: float(v) for n, v in zip(result.param_names, result.params)},
            "errors": {n: float(e) for n, e in zip(result.param_names, result.errors)},
            "bound_hits": {n: bool(h) for n, h in zip(result.param_names, result.bound_hits)},
            "fit_extras": dict(result.extras) if isinstance(result.extras, dict) else {},
            "nv_metrics": nv,
            "sample_metadata": file_meta,
            "trace_metadata": trace_metadata,
            "provenance": provenance,
        }
        summary_row = {
            "file": mat_file.name,
            "relative_path": relative_path,
            "detector_id": detector_id,
            "detector_label": detector_label,
            "detector_count": detector_count,
            "experiment_type": exp_type,
            "scan_family": exp_type,
            "scan_dim": scan_dim,
            "fit_allowed": fit_allowed,
            "profile": profile.name,
            "model": result.model_name,
            "r2": result.r2,
            "rmse": result.rmse,
            "aic": result.aic,
            "bic": result.bic,
            "durbin_watson": result.durbin_watson,
            "bound_hits_count": int(sum(result.bound_hits)),
            "success": result.success,
            "status": status,
            "status_reason": reason,
            "save_type": trace_metadata.get("save_type", ""),
            "run_status": trace_metadata.get("run_status", ""),
            "iteration_completion": completion,
            "completed_iterations": trace_metadata.get("completed_iterations", ""),
            "target_iterations": trace_metadata.get("target_iterations", ""),
            "checkpoint_count": trace_metadata.get("checkpoint_count", ""),
            "warnings": warnings_text,
            "sample_id": file_meta.get("sample_id", ""),
            "condition": file_meta.get("condition", ""),
            "power": file_meta.get("power", ""),
        }
        validation_row = {
            "file": mat_file.name,
            "relative_path": relative_path,
            "detector_id": detector_id,
            "detector_label": detector_label,
            "detector_count": detector_count,
            "experiment_type": exp_type,
            "scan_family": exp_type,
            "scan_dim": scan_dim,
            "status": status,
            "reason": reason,
            "r2": result.r2,
            "r2_min": profile.r2_min,
            "bound_hits": int(sum(result.bound_hits)),
            "bound_hits_max": profile.max_bound_hits,
            "model": result.model_name,
        }
    else:
        result_json = {
            "file": mat_file.name,
            "relative_path": relative_path,
            "detector_id": detector_id,
            "detector_label": detector_label,
            "detector_count": detector_count,
            "experiment_type": exp_type,
            "scan_family": exp_type,
            "scan_dim": scan_dim,
            "fit_allowed": fit_allowed,
            "profile": profile.name,
            "selected_model": "none",
            "selection_note": model_note,
            "metrics": {},
            "params": {},
            "errors": {},
            "bound_hits": {},
            "nv_metrics": {},
            "status": status,
            "status_reason": reason,
            "sample_metadata": file_meta,
            "trace_metadata": trace_metadata,
            "provenance": provenance,
        }
        summary_row = {
            "file": mat_file.name,
            "relative_path": relative_path,
            "detector_id": detector_id,
            "detector_label": detector_label,
            "detector_count": detector_count,
            "experiment_type": exp_type,
            "scan_family": exp_type,
            "scan_dim": scan_dim,
            "fit_allowed": fit_allowed,
            "profile": profile.name,
            "model": "none",
            "r2": "",
            "rmse": "",
            "aic": "",
            "bic": "",
            "durbin_watson": "",
            "bound_hits_count": 0,
            "success": status not in {"FAIL"},
            "status": status,
            "status_reason": reason,
            "save_type": trace_metadata.get("save_type", ""),
            "run_status": trace_metadata.get("run_status", ""),
            "iteration_completion": completion,
            "completed_iterations": trace_metadata.get("completed_iterations", ""),
            "target_iterations": trace_metadata.get("target_iterations", ""),
            "checkpoint_count": trace_metadata.get("checkpoint_count", ""),
            "warnings": warnings_text,
            "sample_id": file_meta.get("sample_id", ""),
            "condition": file_meta.get("condition", ""),
            "power": file_meta.get("power", ""),
        }
        validation_row = {
            "file": mat_file.name,
            "relative_path": relative_path,
            "detector_id": detector_id,
            "detector_label": detector_label,
            "detector_count": detector_count,
            "experiment_type": exp_type,
            "scan_family": exp_type,
            "scan_dim": scan_dim,
            "status": status,
            "reason": reason,
            "r2": "",
            "r2_min": profile.r2_min,
            "bound_hits": 0,
            "bound_hits_max": profile.max_bound_hits,
            "model": "none",
        }

    return result_json, summary_row, validation_row


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def _load_mapping_file(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): (v if isinstance(v, dict) else {"value": v}) for k, v in data.items()}
        return {}
    if path.suffix.lower() == ".csv":
        out: dict[str, dict] = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row.get("file") or row.get("filename")
                if not fn:
                    continue
                out[str(fn)] = dict(row)
        return out
    return {}


def _write_origin_bundle(out_dir: Path, stem: str, x: np.ndarray, y: np.ndarray,
                          y_fit: np.ndarray | None, meta: dict):
    p = out_dir / f"{stem}_origin_bundle.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "y_fit", "residual"])
        y_fit_arr = y_fit if y_fit is not None else np.full_like(y, np.nan, dtype=float)
        residual = y - y_fit_arr if y_fit is not None else np.full_like(y, np.nan, dtype=float)
        for xv, yv, yfv, rv in zip(x, y, y_fit_arr, residual):
            writer.writerow([xv, yv, yfv, rv])
    with (out_dir / f"{stem}_origin_bundle_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _discover_mat_files(input_dir: Path, *, recursive: bool, skip_checkpoints: bool) -> list[Path]:
    pattern = "**/*.mat" if recursive else "*.mat"
    files = sorted(input_dir.glob(pattern))
    if skip_checkpoints:
        files = [p for p in files if "__checkpoint" not in p.stem.lower()]
    return files


def _relative_batch_path(input_dir: Path, mat_file: Path) -> str:
    try:
        return str(mat_file.relative_to(input_dir))
    except ValueError:
        return mat_file.name


def _output_stem(input_dir: Path, mat_file: Path, *, recursive: bool) -> str:
    if not recursive:
        return mat_file.stem
    rel = Path(_relative_batch_path(input_dir, mat_file)).with_suffix("")
    return "__".join(rel.parts)


def _canonical_batch_detector(value: str) -> str:
    normalized = str(value or "all").strip().lower().replace(" ", "")
    aliases = {
        "all": "all",
        "both": "all",
        "detector1": "detector1",
        "1": "detector1",
        "detector2": "detector2",
        "2": "detector2",
    }
    if normalized not in aliases:
        raise ValueError("detector must be 'all', 'detector1', or 'detector2'")
    return aliases[normalized]


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def run_batch(
    input_dir: Path,
    output_dir: Path,
    mode: str = "contrast",
    bin_size: int = 1,
    smooth_window: int = 1,
    roi_min: float | None = None,
    roi_max: float | None = None,
    single_file: Path | None = None,
    metadata_file: Path | None = None,
    exclude_file: Path | None = None,
    plugin_file: Path | None = None,
    model_override: str | None = None,
    multistart: int = 12,
    robust: bool = False,
    odmr_multipeak: bool = False,
    dpi: int = 180,
    write_origin_bundles: bool = True,
    recursive: bool = False,
    skip_checkpoints: bool = False,
    detector: str = "all",
    detector_labels: dict[str, str] | None = None,
    progress_callback=None,
):
    """
    Run batch fitting on all .mat files in *input_dir*.

    Parameters
    ----------
    model_override : str or None
        Force a specific model for ALL files (e.g. 'Rabi', 'SpinEcho', 'ODMR',
        'Ramsey').  If None, type is inferred from metadata + filename.
    multistart : int
        Number of random seed restarts per fit (default 12).
    robust : bool
        If True, run subset-based robust fitting for each file.
    odmr_multipeak : bool
        If True, use automatic multi-Lorentzian for ODMR data.
    dpi : int
        Output plot resolution in dots-per-inch.
    write_origin_bundles : bool
        If False, skip writing per-file origin bundle CSVs.
    progress_callback : callable or None
        Called with (file_index, total_files, filename, status_msg) tuples.
    """
    detector = _canonical_batch_detector(detector)
    labels = {**DETECTOR_LABELS, **(detector_labels or {})}
    output_dir.mkdir(parents=True, exist_ok=True)
    if single_file is not None:
        mat_files = [single_file]
    else:
        mat_files = _discover_mat_files(input_dir, recursive=recursive, skip_checkpoints=skip_checkpoints)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {input_dir}")

    summary_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    metadata_map = _load_mapping_file(metadata_file)
    exclude_map = _load_mapping_file(exclude_file)
    custom_spec: dict | None = None
    if plugin_file is not None and plugin_file.exists():
        custom_spec = json.loads(plugin_file.read_text(encoding="utf-8"))

    work_items: list[tuple[Path, ExperimentTrace | None, dict | None, int]] = []
    for mat_file in mat_files:
        try:
            dataset = load_experiment_dataset(mat_file, mode=mode)  # type: ignore[arg-type]
        except Exception as exc:
            work_items.append(
                (
                    mat_file,
                    None,
                    {
                        "detector_id": "detector1",
                        "detector_label": labels["detector1"],
                        "message": str(exc),
                        "run_status": "load_error",
                        "error_identifier": type(exc).__name__,
                        "error_message": str(exc),
                        "available_channels": [],
                    },
                    1,
                )
            )
            continue
        detector_count = len(dataset.available_detectors)
        requested_ids = list(dataset.available_detectors) if detector == "all" else [detector]
        for detector_id in requested_ids:
            trace = dataset.detectors.get(detector_id)
            if trace is not None:
                trace.detector_label = labels.get(detector_id, DETECTOR_LABELS.get(detector_id, detector_id))
                work_items.append((mat_file, trace, None, detector_count))
                continue
            failure = dataset.detector_errors.get(detector_id)
            work_items.append(
                (
                    mat_file,
                    None,
                    {
                        "detector_id": detector_id,
                        "detector_label": labels.get(detector_id, DETECTOR_LABELS.get(detector_id, detector_id)),
                        "message": failure.message if failure is not None else f"{detector_id} is unavailable",
                        "run_status": failure.run_status if failure is not None else str(dataset.metadata.get("run_status", "unavailable")),
                        "error_identifier": failure.error_identifier if failure is not None else "detector_unavailable",
                        "error_message": failure.error_message if failure is not None else "",
                        "available_channels": list(dataset.available_detectors),
                    },
                    detector_count,
                )
            )

    n_total = len(work_items)

    for file_idx, (mat_file, trace, load_failure, detector_count) in enumerate(work_items):
        relative_path = _relative_batch_path(input_dir, mat_file)
        file_meta = metadata_map.get(mat_file.name, {})
        detector_id = trace.detector_id if trace is not None else str((load_failure or {}).get("detector_id", "detector1"))
        detector_label = trace.detector_label if trace is not None else str((load_failure or {}).get("detector_label", labels.get(detector_id, detector_id)))
        base_stem = _output_stem(input_dir, mat_file, recursive=recursive)
        stem = f"{base_stem}__{detector_id}" if detector_count > 1 else base_stem

        if load_failure is not None or trace is None:
            trace_metadata = {
                "detector_id": detector_id,
                "detector_label": detector_label,
                "detector_count": detector_count,
                "run_status": str((load_failure or {}).get("run_status", "load_error")),
                "error_identifier": str((load_failure or {}).get("error_identifier", "load_error")),
                "error_message": str((load_failure or {}).get("error_message", "")),
            }
            reason = str((load_failure or {}).get("message", "Detector could not be loaded"))
            provenance = {
                "app_name": "SmartFitterPy",
                "app_version": "2026.08",
                "timestamp_utc": _now_utc_iso(),
                "mode": mode,
                "relative_path": relative_path,
                "detector_id": detector_id,
                "detector_label": detector_label,
                "detector_count": detector_count,
                "available_channels": list((load_failure or {}).get("available_channels", [])),
            }
            profile = PROFILES["LineScan"]
            rj, sr, vr = _build_result_row(
                mat_file,
                "Unknown",
                profile,
                None,
                "FAIL",
                reason,
                file_meta,
                trace_metadata,
                provenance,
                "detector_load_error",
                relative_path=relative_path,
                scan_dim="unavailable",
                fit_allowed=False,
            )
            with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                json.dump(rj, f, indent=2)
            summary_rows.append(sr)
            validation_rows.append(vr)
            if progress_callback:
                progress_callback(file_idx, n_total, f"{mat_file.name} [{detector_label}]", f"FAIL | {reason}")
            continue

        exp_type = trace.experiment_type
        ex = exclude_map.get(mat_file.name, {})
        excluded = str(ex.get("exclude", "false")).lower() in ("1", "true", "yes", "y")

        # Apply model override
        if model_override:
            exp_type = model_override
        elif exp_type == "LineScan" and "t1" in mat_file.name.lower():
            exp_type = "T1"

        if custom_spec is not None:
            exp_type = "CustomModel"

        profile = PROFILES.get(exp_type if exp_type in PROFILES else "LineScan")

        fit_png = output_dir / f"{stem}_fit.png"
        trace_metadata = dict(trace.metadata or {})
        trace_metadata.update(
            {
                "detector_id": detector_id,
                "detector_label": detector_label,
                "detector_count": detector_count,
            }
        )
        provenance = {
            "app_name": "SmartFitterPy",
            "app_version": "2026.08",
            "timestamp_utc": _now_utc_iso(),
            "mode": mode,
            "relative_path": relative_path,
            "detector_id": detector_id,
            "detector_label": detector_label,
            "detector_count": detector_count,
            "available_channels": list(trace.available_detectors),
            "excluded": excluded,
            "model_override": model_override or "",
            "robust": robust,
            "odmr_multipeak": odmr_multipeak,
            "multistart": multistart,
            "recursive": recursive,
            "skip_checkpoints": skip_checkpoints,
        }

        try:
            processed = process_series(trace.x_ns, trace.y, bin_size=bin_size, roi=(roi_min, roi_max))
            x, y = processed.fit_x, processed.fit_y
            if smooth_window > 1:
                y = smooth_trace(y, smooth_window)
        except Exception as exc:
            reason = f"Preprocessing failed: {type(exc).__name__}: {exc}"
            rj, sr, vr = _build_result_row(
                mat_file, exp_type, profile, None, "FAIL", reason,
                file_meta, trace_metadata, provenance, "preprocess_error",
                relative_path=relative_path, scan_dim=trace.scan_dim, fit_allowed=trace.fit_allowed,
            )
            with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                json.dump(rj, f, indent=2)
            summary_rows.append(sr)
            validation_rows.append(vr)
            if progress_callback:
                progress_callback(file_idx, n_total, f"{mat_file.name} [{detector_label}]", f"FAIL | {reason}")
            continue

        if progress_callback:
            progress_callback(file_idx, n_total, f"{mat_file.name} [{detector_label}]", "Processing...")

        # --- Spatial scans: plot only, no fit ---
        if trace.scan_dim == "scan2d" and trace.z2d is not None and trace.x2d is not None and trace.y2d is not None:
            _plot_scan2d(
                fit_png, trace.x2d, trace.y2d, trace.z2d,
                title=f"{mat_file.name} | {detector_label} | 2D scan",
                x_label=(trace.scan_axes[0] if len(trace.scan_axes) > 0 else "X"),
                y_label=(trace.scan_axes[1] if len(trace.scan_axes) > 1 else "Y"),
                dpi=dpi,
            )
            rj, sr, vr = _build_result_row(
                mat_file, exp_type, profile, None,
                "PASS", "2D scan plotted (no fit expected)", file_meta, trace_metadata, provenance, "scan2d_plot_only",
                relative_path=relative_path, scan_dim=trace.scan_dim, fit_allowed=trace.fit_allowed,
            )
            with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                json.dump(rj, f, indent=2)
            if write_origin_bundles:
                _write_origin_bundle(output_dir, stem, x, y, None,
                                     {"sample_metadata": file_meta, "trace_metadata": trace_metadata, "provenance": provenance})
            summary_rows.append(sr)
            validation_rows.append(vr)
            continue

        if trace.scan_dim == "scan1d" and not trace.fit_allowed:
            _plot_scan1d(
                fit_png, x, y,
                title=f"{mat_file.name} | {detector_label} | 1D scan",
                x_label=(trace.scan_axes[0] if trace.scan_axes else trace.x_label),
                y_label=trace.y_label, dpi=dpi,
            )
            rj, sr, vr = _build_result_row(
                mat_file, exp_type, profile, None,
                "PASS", "1D scan plotted (no fit expected)", file_meta, trace_metadata, provenance, "scan1d_plot_only",
                relative_path=relative_path, scan_dim=trace.scan_dim, fit_allowed=trace.fit_allowed,
            )
            with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                json.dump(rj, f, indent=2)
            if write_origin_bundles:
                _write_origin_bundle(output_dir, stem, x, y, None,
                                     {"sample_metadata": file_meta, "trace_metadata": trace_metadata, "provenance": provenance})
            summary_rows.append(sr)
            validation_rows.append(vr)
            continue

        if excluded:
            _plot_scan1d(
                fit_png, x, y,
                title=f"{mat_file.name} | {detector_label} | EXCLUDED",
                x_label=trace.x_label, y_label=trace.y_label, dpi=dpi,
            )
            rj, sr, vr = _build_result_row(
                mat_file, exp_type, profile, None,
                "EXCLUDED", "Excluded by exclude-file flag", file_meta, trace_metadata, provenance, "excluded_flag",
                relative_path=relative_path, scan_dim=trace.scan_dim, fit_allowed=trace.fit_allowed,
            )
            with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                json.dump(rj, f, indent=2)
            if write_origin_bundles:
                _write_origin_bundle(output_dir, stem, x, y, None,
                                     {"sample_metadata": file_meta, "trace_metadata": trace_metadata, "provenance": provenance})
            summary_rows.append(sr)
            validation_rows.append(vr)
            continue

        # --- Fitting ---
        try:
            if robust:
                result = _run_robust_fit(
                    exp_type, x, y, n_starts=multistart,
                    custom_spec=custom_spec, odmr_multipeak=odmr_multipeak,
                )
                model_note = "robust_fit"
            elif exp_type == "Ramsey":
                sel = fit_ramsey_physics_first(x, y)
                result = sel.best
                model_note = sel.selected_by
            else:
                result = _fit_non_ramsey(
                    exp_type, x, y, n_starts=multistart,
                    custom_spec=custom_spec, odmr_multipeak=odmr_multipeak,
                    trace=trace,
                )
                model_note = "single_model"
        except Exception as exc:
            try:
                result = _fit_non_ramsey("SpinEcho", x, y, n_starts=multistart)
                model_note = f"fallback_after_error:{type(exc).__name__}"
            except Exception as fallback_exc:
                reason = (
                    f"Fit failed: {type(exc).__name__}: {exc}; "
                    f"fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
                )
                rj, sr, vr = _build_result_row(
                    mat_file, exp_type, profile, None, "FAIL", reason,
                    file_meta, trace_metadata, provenance, "fit_error",
                    relative_path=relative_path, scan_dim=trace.scan_dim, fit_allowed=trace.fit_allowed,
                )
                with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                    json.dump(rj, f, indent=2)
                summary_rows.append(sr)
                validation_rows.append(vr)
                if progress_callback:
                    progress_callback(file_idx, n_total, f"{mat_file.name} [{detector_label}]", f"FAIL | {reason}")
                continue

        status, reason = classify_status(
            r2=result.r2, bound_hits=int(sum(result.bound_hits)), profile=profile,
        )

        _plot_fit(
            fit_png, x, y, result,
            title=f"{mat_file.name} | {detector_label} | {result.model_name} | R²={result.r2:.4f}",
            y_label=trace.y_label, status=status, reason=reason, dpi=dpi,
        )

        rj, sr, vr = _build_result_row(
            mat_file, exp_type, profile, result,
            status, reason, file_meta, trace_metadata, provenance, model_note,
            relative_path=relative_path, scan_dim=trace.scan_dim, fit_allowed=trace.fit_allowed,
        )
        with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
            json.dump(rj, f, indent=2)
        if write_origin_bundles:
            nv = _extract_nv_metrics(exp_type, result, trace_metadata)
            _write_origin_bundle(
                output_dir, stem, x, y, result.y_fit,
                {"sample_metadata": file_meta, "trace_metadata": trace_metadata, "provenance": provenance, "nv_metrics": nv},
            )
        summary_rows.append(sr)
        validation_rows.append(vr)

        if progress_callback:
            progress_callback(file_idx, n_total, f"{mat_file.name} [{detector_label}]", f"{status} | R²={result.r2:.4f}")

    # --- Summary CSVs ---
    _SUMMARY_FIELDS = [
        "file", "relative_path", "detector_id", "detector_label", "detector_count",
        "experiment_type", "scan_family", "scan_dim", "fit_allowed",
        "profile", "model", "r2", "rmse", "aic", "bic",
        "durbin_watson", "bound_hits_count", "success", "status", "status_reason",
        "save_type", "run_status", "iteration_completion", "completed_iterations",
        "target_iterations", "checkpoint_count", "warnings",
        "sample_id", "condition", "power",
    ]
    with (output_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    _VALIDATION_FIELDS = [
        "file", "relative_path", "detector_id", "detector_label", "detector_count",
        "experiment_type", "scan_family", "scan_dim", "status", "reason", "r2", "r2_min",
        "bound_hits", "bound_hits_max", "model",
    ]
    with (output_dir / "validation_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_VALIDATION_FIELDS)
        writer.writeheader()
        writer.writerows(validation_rows)

    # Aggregate statistics
    agg: dict[tuple[str, str], list[float]] = {}
    for row in summary_rows:
        if row.get("status") == "EXCLUDED":
            continue
        try:
            r2_val = float(row.get("r2"))
        except Exception:
            continue
        key = (str(row.get("experiment_type", "")), str(row.get("condition", "")))
        agg.setdefault(key, []).append(r2_val)
    with (output_dir / "aggregate_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_type", "condition", "n", "mean_r2", "std_r2"])
        for (exp, cond), vals in sorted(agg.items()):
            writer.writerow([exp, cond, len(vals), float(np.mean(vals)), float(np.std(vals))])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SmartFitter Pro batch pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--single-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./batch_out"))
    parser.add_argument("--mode", choices=["contrast", "signal", "raw_signal", "reference", "difference"], default="contrast")
    parser.add_argument(
        "--detector",
        choices=["all", "detector1", "detector2"],
        default="all",
        help="Detector stream(s) to process (default: all available streams).",
    )
    parser.add_argument("--detector1-label", default="Detector 1", help="Presentation label for detector1 outputs.")
    parser.add_argument("--detector2-label", default="Detector 2", help="Presentation label for detector2 outputs.")
    parser.add_argument("--bin-size", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--roi-min", type=float, default=None)
    parser.add_argument("--roi-max", type=float, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None,
                        help="CSV/JSON mapping filename to metadata fields.")
    parser.add_argument("--exclude-file", type=Path, default=None,
                        help="CSV/JSON with exclude flags per filename.")
    parser.add_argument("--plugin-file", type=Path, default=None,
                        help="Custom model JSON spec.")
    parser.add_argument("--model-override", type=str, default=None,
                        choices=["Rabi", "SpinEcho", "DynamicDecoupling", "ODMR", "Ramsey", "T1", "DEERFrequency", "DEERDuration", "DEERDecay", "DEERPosition"],
                        help="Force a specific model for ALL files.")
    parser.add_argument("--multistart", type=int, default=12,
                        help="Number of random-seed restarts per fit (default: 12).")
    parser.add_argument("--robust", action="store_true",
                        help="Use subset-based robust fitting.")
    parser.add_argument("--odmr-multipeak", action="store_true",
                        help="Use automatic multi-Lorentzian for ODMR data.")
    parser.add_argument("--dpi", type=int, default=180,
                        help="Output plot resolution (default: 180).")
    parser.add_argument("--no-origin-bundle", action="store_true",
                        help="Skip writing per-file origin bundle CSVs.")
    parser.add_argument("--recursive", action="store_true",
                        help="Find .mat files recursively under --input-dir and use collision-safe output names.")
    parser.add_argument("--skip-checkpoints", action="store_true",
                        help="Skip files whose stem contains __checkpoint.")
    args = parser.parse_args()

    def _cli_progress(idx, total, name, msg):
        print(f"  [{idx+1}/{total}] {name}: {msg}")

    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        detector=args.detector,
        detector_labels={"detector1": args.detector1_label, "detector2": args.detector2_label},
        bin_size=args.bin_size,
        smooth_window=args.smooth_window,
        roi_min=args.roi_min,
        roi_max=args.roi_max,
        single_file=args.single_file,
        metadata_file=args.metadata_file,
        exclude_file=args.exclude_file,
        plugin_file=args.plugin_file,
        model_override=args.model_override,
        multistart=args.multistart,
        robust=args.robust,
        odmr_multipeak=args.odmr_multipeak,
        dpi=args.dpi,
        write_origin_bundles=not args.no_origin_bundle,
        recursive=args.recursive,
        skip_checkpoints=args.skip_checkpoints,
        progress_callback=_cli_progress,
    )
    print(f"Batch complete. Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
