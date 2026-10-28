from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .diagnostics import aic, bic, durbin_watson, r_squared, rmse
from .analysis_tools import process_series
from .fit_engine import FitResult, fit_model_multistart
from .fit_workflows import FitWorkflowConfig, fit_non_ramsey
from .io_mat import DETECTOR_LABELS, ExperimentTrace, is_stage_scan, load_experiment_dataset
from .models import (
    MODEL_REGISTRY,
    build_custom_expression_model,
    spin_echo_model,
)
from .preprocess import smooth_trace
from .profiles import PROFILES, classify_status
from .prepared_fit import PreparedFit
from .batch_manifest import (
    RunRecord,
    fingerprint,
    row_id,
    load_manifest,
    output_lock,
    atomic_json,
    atomic_csv,
    json_value,
)
from .analysis_identity import analysis_descriptor, result_envelope
from .rabi_utils import build_rabi_nv_metrics, rabi_envelope_bounds, rabi_envelope_metrics
from .ramsey import estimate_ramsey_frequency, fit_ramsey_physics_first

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_default(value):
    """Serialize numpy values left in fit extras (e.g. the T1 ``plot_y`` series)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


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
    cfg = FitWorkflowConfig(
        multistart=n_starts,
        rabi_mode="Single reliable Rabi" if exp_type == "Rabi" else "Classic single (recommended)",
        odmr_multipeak=odmr_multipeak,
        odmr_peak_count=4,
        odmr_auto_peak=True,
        odmr_polarity="Auto",
        custom_spec=custom_spec,
    )
    return PreparedFit.create(x, y, exp_type, cfg, trace=trace).run()[0]


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
        model_name="ODMRMulti",
        model_func=odmr_multi_lorentzian,
        x=x,
        y=y,
        param_names=names,
        lower=lower,
        upper=upper,
        seeds=seeds,
    )


def _run_robust_fit(
    exp_type: str,
    x: np.ndarray,
    y: np.ndarray,
    n_starts: int = 12,
    custom_spec: dict | None = None,
    odmr_multipeak: bool = False,
    progress_callback=None,
    *,
    trace=None,
) -> FitResult:
    """Subset strategy over the same prepared request used by ordinary fitting."""
    cfg = FitWorkflowConfig(
        multistart=n_starts,
        rabi_mode="Single reliable Rabi" if exp_type == "Rabi" else "Classic single (recommended)",
        odmr_multipeak=odmr_multipeak,
        odmr_peak_count=4,
        custom_spec=custom_spec,
    )
    return PreparedFit.create(x, y, exp_type, cfg, trace=trace).run(robust=True, progress=progress_callback)[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PARAM_UNITS = {
    "y0": "arb.",
    "A": "arb.",
    "m": "1/ns",
    "C": "arb.",
    "f": "GHz",
    "f1": "GHz",
    "f2": "GHz",
    "phi": "rad",
    "phi1": "rad",
    "phi2": "rad",
    "tau": "ns",
    "T2": "ns",
    "T2_star": "ns",
    "n": "",
    "w": "",
    "x0": "",
}


def _format_param_line(name: str, val: float, err: float) -> str:
    unit = _PARAM_UNITS.get(name, "")
    unit_str = f" {unit}" if unit else ""
    return f"  {name} = {val:.4g}{unit_str} ± {err:.2g}"


def _pyplot():
    """matplotlib.pyplot with the non-interactive Agg backend, imported on first use: only batch figure
    writing needs it, and importing it at module load added ~0.1 s to every app start."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


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
    y = np.asarray(result.extras.get("plot_y", y), float)
    y_label = str(result.extras.get("plot_y_label") or y_label)
    plt = _pyplot()
    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
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
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        horizontalalignment="right",
        fontfamily="monospace",
        bbox=props,
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


def _plot_scan1d(out_path: Path, x: np.ndarray, y: np.ndarray, title: str, x_label: str, y_label: str, dpi: int = 180):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, "o-", ms=3.5, lw=1.2, color="#1976d2")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_scan2d(
    out_path: Path,
    x2d: np.ndarray,
    y2d: np.ndarray,
    z2d: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    dpi: int = 180,
):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 7))
    # Percentile normalization for better contrast
    vmin = np.nanpercentile(z2d, 2)
    vmax = np.nanpercentile(z2d, 98)
    mesh = ax.pcolormesh(x2d, y2d, z2d, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
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
            physical_params = {
                k: float(v) for k, v in dict(result.extras.get("rabi_physical_reference_params", {})).items()
            }
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
    if exp_type in ("Ramsey", "Rabi", "SpinEcho", "T1", "DynamicDecoupling"):
        from .units import unit_scale

        factor = unit_scale((trace_metadata or {}).get("x_unit") or "ns", "ns")
        for key in list(metrics):
            # Metadata pulse timings already carry explicit ns; only model-derived quantities scale.
            if (
                key.endswith("_ns")
                and key not in ("pi_pulse_ns", "tau_ns", "rf2_duration_ns")
                and isinstance(metrics[key], (int, float))
            ):
                metrics[key] *= factor
            elif key in ("detuning_MHz", "rabi_freq_MHz") and isinstance(metrics[key], (int, float)):
                metrics[key] /= factor
    return metrics


# ---------------------------------------------------------------------------
# Result-row builder (eliminates code duplication)
# ---------------------------------------------------------------------------


def _fit_status(result: FitResult, profile, model_note: str = "") -> tuple[str, str]:
    """Separate optimizer/model outcomes from the fit-quality gate."""
    quality, reason = classify_status(
        r2=result.r2,
        bound_hits=int(sum(result.bound_hits)),
        profile=profile,
    )
    if not np.isfinite(result.r2) or not np.isfinite(result.rmse):
        quality, reason = "FAIL", "Nonfinite fit metrics"
    fallback = model_note.startswith("fallback_after_error:")
    detail = f"Quality {quality}: {reason}"
    if fallback:
        detail = f"{model_note}; {detail}"
    if not result.success:
        return "UNCONVERGED", f"Optimizer did not converge: {result.message}; {detail}"
    if fallback:
        return "FALLBACK", f"Fallback model {result.model_name}; {detail}"
    return quality, reason


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
            "covariance": result.covariance,
            "free_mask": result.free_mask,
            "uncertainty": result.uncertainty,
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
            "success": False if status == "FAIL" else "",
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

    result_json.update(status=status, status_reason=reason, model_note=model_note)
    summary_row["model_note"] = model_note
    validation_row["model_note"] = model_note
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


def _write_origin_bundle(out_dir: Path, stem: str, x: np.ndarray, y: np.ndarray, y_fit: np.ndarray | None, meta: dict):
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


def _unique_stem(stem: str, used: set[str]) -> str:
    """Reserve a run-unique output stem; flattening ``a/b.mat`` and ``a__b.mat`` (or adding a
    detector suffix) can otherwise make two sources write the same files.  Case-insensitive
    because Windows paths are."""
    candidate, n = stem, 2
    while candidate.lower() in used:
        candidate, n = f"{stem}__{n}", n + 1
    used.add(candidate.lower())
    return candidate


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


def _run_batch_locked(
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
    cancel_requested=None,
    resume_manifest=None,
    selected_rows=None,
    ui_options=None,
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
        Called with (completed_rows, requested_rows, filename, status_msg); progress increments only after row commit.
    """
    detector = _canonical_batch_detector(detector)
    labels = {**DETECTOR_LABELS, **(detector_labels or {})}
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_manifest is not None:
        mat_files = [Path(source["path"]) for source in resume_manifest["sources"]]
    elif single_file is not None:
        mat_files = [single_file]
    else:
        mat_files = _discover_mat_files(input_dir, recursive=recursive, skip_checkpoints=skip_checkpoints)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {input_dir}")

    options = {
        key: value
        for key, value in locals().copy().items()
        if key
        in {
            "input_dir",
            "output_dir",
            "mode",
            "bin_size",
            "smooth_window",
            "roi_min",
            "roi_max",
            "single_file",
            "metadata_file",
            "exclude_file",
            "plugin_file",
            "model_override",
            "multistart",
            "robust",
            "odmr_multipeak",
            "dpi",
            "write_origin_bundles",
            "recursive",
            "skip_checkpoints",
            "detector",
            "detector_labels",
        }
    }
    options = json_value(options)
    sources = [
        {
            "path": str(path.resolve()),
            "relative_path": _relative_batch_path(input_dir, path),
            "revision": fingerprint(path),
        }
        for path in mat_files
    ]
    summary_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    auxiliary = (resume_manifest or {}).get("auxiliary")
    metadata_map = auxiliary["metadata"] if auxiliary is not None else _load_mapping_file(metadata_file)
    exclude_map = auxiliary["exclude"] if auxiliary is not None else _load_mapping_file(exclude_file)
    custom_spec: dict | None = None
    if auxiliary is not None:
        custom_spec = auxiliary.get("custom_model")
    elif plugin_file is not None and plugin_file.exists():
        custom_spec = json.loads(plugin_file.read_text(encoding="utf-8"))

    work_items: list[tuple[Path, ExperimentTrace | None, dict | None, int]] = []
    discovery_cancelled = False
    for mat_file in mat_files:
        if cancel_requested and cancel_requested():
            discovery_cancelled = True
            break
        try:
            expected = next(item["revision"] for item in sources if item["path"] == str(mat_file.resolve()))
            if fingerprint(mat_file)["sha256"] != expected["sha256"]:
                raise ValueError("Source changed before loading")
            dataset = load_experiment_dataset(mat_file, mode=mode)  # type: ignore[arg-type]
            if fingerprint(mat_file)["sha256"] != expected["sha256"]:
                raise ValueError("Source changed while loading")
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
                        "run_status": (
                            failure.run_status
                            if failure is not None
                            else str(dataset.metadata.get("run_status", "unavailable"))
                        ),
                        "error_identifier": failure.error_identifier if failure is not None else "detector_unavailable",
                        "error_message": failure.error_message if failure is not None else "",
                        "available_channels": list(dataset.available_detectors),
                    },
                    detector_count,
                )
            )

    used_stems = set()
    stem_by_row = {}
    row_specs = []
    previous = {r["row_id"]: r for r in (resume_manifest or {}).get("rows", [])}
    for mat_file, tr, failure, count in work_items:
        relative = _relative_batch_path(input_dir, mat_file)
        det = tr.detector_id if tr is not None else (failure or {}).get("detector_id", "detector1")
        key = row_id(relative, det)
        base = _output_stem(input_dir, mat_file, recursive=recursive)
        stem = previous.get(key, {}).get("output_stem") or _unique_stem(
            f"{base}__{det}" if count > 1 else base, used_stems
        )
        stem_by_row[(relative, det)] = stem
        row_specs.append(
            {
                "row_id": key,
                "source_path": str(mat_file.resolve()),
                "relative_path": relative,
                "detector_id": det,
                "output_stem": stem,
            }
        )
    requested = {row["row_id"] for row in row_specs} if selected_rows is None else set(selected_rows)
    if (resume_manifest or {}).get("_resume_unfinished"):
        requested -= {r["row_id"] for r in resume_manifest["rows"] if r["state"] in ("completed", "preserved")}
    unknown = requested - {r["row_id"] for r in row_specs}
    if unknown:
        raise ValueError(f"Unknown batch row(s): {sorted(unknown)}")
    record = RunRecord(
        output_dir,
        options,
        sources,
        row_specs,
        requested,
        parent=resume_manifest,
        auxiliary={"metadata": metadata_map, "exclude": exclude_map, "custom_model": custom_spec},
        ui_options=ui_options,
    )
    record.data["discovery_complete"] = not discovery_cancelled
    record.commit()
    preserved_summary = [r["summary"] for r in record.data["rows"] if r["state"] == "preserved" and r.get("summary")]
    preserved_validation = [
        r["validation"] for r in record.data["rows"] if r["state"] == "preserved" and r.get("validation")
    ]
    n_total = len(requested)

    def reports():
        committed = [r for r in record.data["rows"] if r["state"] in ("completed", "preserved")]
        _write_reports(output_dir, [r["summary"] for r in committed], [r["validation"] for r in committed])
        _write_reports(record.folder, [r["summary"] for r in committed], [r["validation"] for r in committed])

    reports()

    def finish(rj, sr, vr, stem, mat_file, x=None, y=None, result=None):
        key = row_id(sr["relative_path"], sr["detector_id"])
        source = next(v for v in sources if Path(v["path"]).resolve() == mat_file.resolve())
        fields = {
            "output_stem": stem,
            "run_id": record.id,
            "manifest_path": str(record.path),
            "row_id": key,
            "source_path": str(mat_file.resolve()),
            "source_sha256": source["revision"]["sha256"],
        }
        sr.update(fields)
        vr.update(fields)
        rj.update(fields)
        rj["analysis"] = result_envelope(
            analysis_descriptor(
                str(mat_file.resolve()),
                sr["detector_id"],
                mode,
                {"bin": bin_size, "smooth": smooth_window, "roi_min": roi_min, "roi_max": roi_max},
            ),
            source["revision"],
            options,
            kind="batch_fit",
        )
        if result is not None:
            rj["series"] = {
                "x": x.tolist(),
                "input_y": y.tolist(),
                "target_y": np.asarray(result.extras.get("plot_y", y)).tolist(),
                "y_fit": np.asarray(result.y_fit).tolist(),
            }
        atomic_json(output_dir / f"{stem}_result.json", rj)
        summary_rows.append(sr)
        validation_rows.append(vr)
        record.complete(key, rj, sr, vr, origin=write_origin_bundles)
        reports()
        if progress_callback:
            progress_callback(len(summary_rows), n_total, mat_file.name, f'{sr["status"]} | {sr["status_reason"]}')

    state = "cancelled" if discovery_cancelled else "finished"
    error = None
    try:
        if progress_callback:
            progress_callback(0, n_total, "", "Run record created")
        for file_idx, (mat_file, trace, load_failure, detector_count) in enumerate(work_items):
            relative_path = _relative_batch_path(input_dir, mat_file)
            file_meta = metadata_map.get(mat_file.name, {})
            detector_id = (
                trace.detector_id if trace is not None else str((load_failure or {}).get("detector_id", "detector1"))
            )
            detector_label = (
                trace.detector_label
                if trace is not None
                else str((load_failure or {}).get("detector_label", labels.get(detector_id, detector_id)))
            )
            key = row_id(relative_path, detector_id)
            if key not in requested:
                continue
            if cancel_requested and cancel_requested():
                state = "cancelled"
                break
            expected = next(item["revision"] for item in sources if item["path"] == str(mat_file.resolve()))
            if fingerprint(mat_file)["sha256"] != expected["sha256"]:
                raise ValueError(f"Source changed during batch: {mat_file}")
            stem = stem_by_row[(relative_path, detector_id)]

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
                    json.dump(rj, f, indent=2, default=_json_default)
                finish(rj, sr, vr, stem, mat_file)
                continue

            exp_type = trace.experiment_type
            ex = exclude_map.get(mat_file.name, {})
            excluded = str(ex.get("exclude", "false")).lower() in ("1", "true", "yes", "y")

            # Apply model override
            auto_line_gaussian = False
            if model_override:
                exp_type = model_override
            elif (
                exp_type == "LineScan"
                and "t1" in mat_file.name.lower()
                and trace.scan_dim == "scan1d"
                and not is_stage_scan(tuple(trace.scan_axes))
            ):
                # Legacy fallback for unnamed sequence scans only; a stage scan named e.g. "...AfterT1..." stays a scan.
                exp_type = "T1"
            elif exp_type == "LineScan" and trace.scan_dim == "scan1d" and custom_spec is None:
                # Auto for a 1D line scan fits one Gaussian peak on a background, as in Analyze and Compare.
                exp_type = "GaussianPeaks"
                auto_line_gaussian = True

            if custom_spec is not None:
                exp_type = "CustomModel"

            profile = (
                PROFILES["TimeTrace"]
                if trace.experiment_type == "TimeTrace"
                else PROFILES.get(exp_type if exp_type in PROFILES else "LineScan")
            )

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
                "mode": trace.mode,
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
                    mat_file,
                    exp_type,
                    profile,
                    None,
                    "FAIL",
                    reason,
                    file_meta,
                    trace_metadata,
                    provenance,
                    "preprocess_error",
                    relative_path=relative_path,
                    scan_dim=trace.scan_dim,
                    fit_allowed=trace.fit_allowed,
                )
                with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                    json.dump(rj, f, indent=2, default=_json_default)
                finish(rj, sr, vr, stem, mat_file)
                continue

            # --- Spatial scans: plot only, no fit ---
            if (
                not excluded
                and trace.scan_dim == "scan2d"
                and trace.z2d is not None
                and trace.x2d is not None
                and trace.y2d is not None
            ):
                _plot_scan2d(
                    fit_png,
                    trace.x2d,
                    trace.y2d,
                    trace.z2d,
                    title=f"{mat_file.name} | {detector_label} | 2D scan",
                    x_label=(trace.scan_axes[0] if len(trace.scan_axes) > 0 else "X"),
                    y_label=(trace.scan_axes[1] if len(trace.scan_axes) > 1 else "Y"),
                    dpi=dpi,
                )
                rj, sr, vr = _build_result_row(
                    mat_file,
                    exp_type,
                    profile,
                    None,
                    "PLOT_ONLY",
                    "2D scan plotted (no fit expected)",
                    file_meta,
                    trace_metadata,
                    provenance,
                    "scan2d_plot_only",
                    relative_path=relative_path,
                    scan_dim=trace.scan_dim,
                    fit_allowed=trace.fit_allowed,
                )
                with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                    json.dump(rj, f, indent=2, default=_json_default)
                if write_origin_bundles:
                    _write_origin_bundle(
                        output_dir,
                        stem,
                        x,
                        y,
                        None,
                        {"sample_metadata": file_meta, "trace_metadata": trace_metadata, "provenance": provenance},
                    )
                finish(rj, sr, vr, stem, mat_file)
                continue

            time_plot_only = trace.experiment_type == "TimeTrace" and (
                not trace.fit_allowed or (not model_override and custom_spec is None)
            )
            # Auto maps 1D line scans to GaussianPeaks above. A scan too short for that fit (background plus
            # amplitude, centre and width needs at least 5 points) is plotted instead of failing.
            line_scan_auto_plot = (
                trace.scan_dim == "scan1d"
                and trace.experiment_type == "LineScan"
                and exp_type == "LineScan"
                and not model_override
                and custom_spec is None
            ) or (auto_line_gaussian and len(x) < 5)
            if not excluded and (
                (trace.scan_dim == "scan1d" and not trace.fit_allowed) or line_scan_auto_plot or time_plot_only
            ):
                _plot_scan1d(
                    fit_png,
                    x,
                    y,
                    title=f"{mat_file.name} | {detector_label} | {'Time trace' if time_plot_only else '1D scan'}",
                    x_label=(trace.scan_axes[0] if trace.scan_axes else trace.x_label),
                    y_label=trace.y_label,
                    dpi=dpi,
                )
                rj, sr, vr = _build_result_row(
                    mat_file,
                    exp_type,
                    profile,
                    None,
                    "PLOT_ONLY",
                    (
                        "Time trace plotted; exponential fitting is optional"
                        if time_plot_only
                        else (
                            f"1D line scan plotted: {len(x)} points are too few for the Auto Gaussian fit (needs 5)"
                            if auto_line_gaussian
                            else "1D line scan plotted (LineScan override has no model; choose GaussianPeaks to fit)"
                            if line_scan_auto_plot
                            else "1D scan plotted (no fit expected)"
                        )
                    ),
                    file_meta,
                    trace_metadata,
                    provenance,
                    "time_trace_plot_only" if time_plot_only else "scan1d_plot_only",
                    relative_path=relative_path,
                    scan_dim=trace.scan_dim,
                    fit_allowed=trace.fit_allowed,
                )
                with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                    json.dump(rj, f, indent=2, default=_json_default)
                if write_origin_bundles:
                    _write_origin_bundle(
                        output_dir,
                        stem,
                        x,
                        y,
                        None,
                        {"sample_metadata": file_meta, "trace_metadata": trace_metadata, "provenance": provenance},
                    )
                finish(rj, sr, vr, stem, mat_file)
                continue

            if excluded:
                _plot_scan1d(
                    fit_png,
                    x,
                    y,
                    title=f"{mat_file.name} | {detector_label} | EXCLUDED",
                    x_label=trace.x_label,
                    y_label=trace.y_label,
                    dpi=dpi,
                )
                rj, sr, vr = _build_result_row(
                    mat_file,
                    exp_type,
                    profile,
                    None,
                    "EXCLUDED",
                    "Excluded by exclude-file flag",
                    file_meta,
                    trace_metadata,
                    provenance,
                    "excluded_flag",
                    relative_path=relative_path,
                    scan_dim=trace.scan_dim,
                    fit_allowed=trace.fit_allowed,
                )
                with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                    json.dump(rj, f, indent=2, default=_json_default)
                if write_origin_bundles:
                    _write_origin_bundle(
                        output_dir,
                        stem,
                        x,
                        y,
                        None,
                        {"sample_metadata": file_meta, "trace_metadata": trace_metadata, "provenance": provenance},
                    )
                finish(rj, sr, vr, stem, mat_file)
                continue

            # --- Fitting ---
            try:
                if trace.experiment_type == "TimeTrace" and exp_type not in {
                    "TimeTrace",
                    "TimeTraceDecay",
                    "CustomModel",
                }:
                    raise ValueError("Elapsed-time data require TimeTraceDecay or a custom model with x in seconds.")
                if robust:
                    result = _run_robust_fit(
                        exp_type,
                        x,
                        y,
                        n_starts=multistart,
                        custom_spec=custom_spec,
                        odmr_multipeak=odmr_multipeak,
                        trace=trace,
                    )
                    model_note = "robust_fit"
                elif exp_type == "Ramsey":
                    sel = fit_ramsey_physics_first(x, y, multistart=multistart)
                    result = sel.best
                    model_note = sel.selected_by
                else:
                    result = _fit_non_ramsey(
                        exp_type,
                        x,
                        y,
                        n_starts=multistart,
                        custom_spec=custom_spec,
                        odmr_multipeak=odmr_multipeak,
                        trace=trace,
                    )
                    model_note = "single_model"
            except Exception as exc:
                try:
                    if trace.experiment_type == "TimeTrace" or exp_type == "GaussianPeaks":
                        # A spin-echo decay is no substitute for a time trace or a spatial profile.
                        raise ValueError(
                            "The selected fit failed; no spin-decay fallback is applied to this data type."
                        ) from exc
                    result = _fit_non_ramsey("SpinEcho", x, y, n_starts=multistart)
                    model_note = f"fallback_after_error:{type(exc).__name__}: {exc}"
                except Exception as fallback_exc:
                    reason = (
                        f"Fit failed: {type(exc).__name__}: {exc}; "
                        f"fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
                    )
                    rj, sr, vr = _build_result_row(
                        mat_file,
                        exp_type,
                        profile,
                        None,
                        "FAIL",
                        reason,
                        file_meta,
                        trace_metadata,
                        provenance,
                        "fit_error",
                        relative_path=relative_path,
                        scan_dim=trace.scan_dim,
                        fit_allowed=trace.fit_allowed,
                    )
                    with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                        json.dump(rj, f, indent=2, default=_json_default)
                    finish(rj, sr, vr, stem, mat_file)
                    continue

            status, reason = _fit_status(result, profile, model_note)

            _plot_fit(
                fit_png,
                x,
                y,
                result,
                title=f"{mat_file.name} | {detector_label} | {result.model_name} | R²={result.r2:.4f}",
                y_label=trace.y_label,
                status=status,
                reason=reason,
                dpi=dpi,
            )

            rj, sr, vr = _build_result_row(
                mat_file,
                exp_type,
                profile,
                result,
                status,
                reason,
                file_meta,
                trace_metadata,
                provenance,
                model_note,
                relative_path=relative_path,
                scan_dim=trace.scan_dim,
                fit_allowed=trace.fit_allowed,
            )
            with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
                json.dump(rj, f, indent=2, default=_json_default)
            if write_origin_bundles:
                nv = _extract_nv_metrics(exp_type, result, trace_metadata)
                _write_origin_bundle(
                    output_dir,
                    stem,
                    x,
                    np.asarray(result.extras.get("plot_y", y)),
                    result.y_fit,
                    {
                        "sample_metadata": file_meta,
                        "trace_metadata": trace_metadata,
                        "provenance": provenance,
                        "nv_metrics": nv,
                        "fit_space": {
                            k: v
                            for k, v in result.extras.items()
                            if k.startswith("t1_norm") or k in ("t1_prep_mode", "plot_y_label")
                        },
                    },
                )
            finish(rj, sr, vr, stem, mat_file, x, y, result)

    except KeyboardInterrupt:
        state = "cancelled"
    except BaseException as exc:
        state = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.finish(state, error)
        reports()
    return {
        "state": state,
        "done": len(summary_rows),
        "total": n_total,
        "rows": [r["summary"] for r in record.data["rows"] if r["state"] in ("completed", "preserved")],
        "manifest_path": str(record.path),
        "run_id": record.id,
        "unfinished_count": sum(r["state"] == "pending" for r in record.data["rows"]),
        "discovery_complete": record.data["discovery_complete"],
    }


def run_batch(input_dir: Path, output_dir: Path, *args, **kwargs):
    with output_lock(output_dir):
        return _run_batch_locked(Path(input_dir), Path(output_dir), *args, **kwargs)


def replay_batch(manifest_path, selected_rows=None, resume_unfinished=False, **runtime):
    manifest = load_manifest(manifest_path)
    if resume_unfinished:
        completed = {r["row_id"] for r in manifest["rows"] if r["state"] in ("completed", "preserved")}
        selected_rows = (
            None
            if not manifest.get("discovery_complete", True)
            else [r["row_id"] for r in manifest["rows"] if r["row_id"] not in completed]
        )
    if resume_unfinished:
        manifest["_resume_unfinished"] = True
    options = dict(manifest["options"])
    for key in ("input_dir", "output_dir", "single_file", "metadata_file", "exclude_file", "plugin_file"):
        if options.get(key) is not None:
            options[key] = Path(options[key])
    return run_batch(**options, resume_manifest=manifest, selected_rows=selected_rows, **runtime)


def _write_reports(output_dir, summary_rows, validation_rows):
    # --- Summary CSVs ---
    _SUMMARY_FIELDS = [
        "file",
        "relative_path",
        "detector_id",
        "detector_label",
        "detector_count",
        "experiment_type",
        "scan_family",
        "scan_dim",
        "fit_allowed",
        "profile",
        "model",
        "r2",
        "rmse",
        "aic",
        "bic",
        "durbin_watson",
        "bound_hits_count",
        "success",
        "status",
        "status_reason",
        "model_note",
        "save_type",
        "run_status",
        "iteration_completion",
        "completed_iterations",
        "target_iterations",
        "checkpoint_count",
        "warnings",
        "sample_id",
        "condition",
        "power",
        "output_stem",
    ]
    atomic_csv(output_dir / "batch_summary.csv", _SUMMARY_FIELDS, summary_rows)

    _VALIDATION_FIELDS = [
        "file",
        "relative_path",
        "detector_id",
        "detector_label",
        "detector_count",
        "experiment_type",
        "scan_family",
        "scan_dim",
        "status",
        "reason",
        "r2",
        "r2_min",
        "bound_hits",
        "bound_hits_max",
        "model",
        "model_note",
    ]
    atomic_csv(output_dir / "validation_report.csv", _VALIDATION_FIELDS, validation_rows)

    # Aggregate statistics
    agg: dict[tuple[str, str], list[float]] = {}
    for row in summary_rows:
        if row.get("status") not in {"PASS", "WARN"}:
            continue
        try:
            r2_val = float(row.get("r2"))
        except Exception:
            continue
        key = (str(row.get("experiment_type", "")), str(row.get("condition", "")))
        agg.setdefault(key, []).append(r2_val)
    aggregate = [
        {
            "experiment_type": exp,
            "condition": condition,
            "n": len(values),
            "mean_r2": float(np.mean(values)),
            "std_r2": float(np.std(values)),
        }
        for (exp, condition), values in sorted(agg.items())
    ]
    atomic_csv(
        output_dir / "aggregate_report.csv", ["experiment_type", "condition", "n", "mean_r2", "std_r2"], aggregate
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SmartFitter Pro batch pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--single-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./batch_out"))
    parser.add_argument(
        "--mode", choices=["contrast", "signal", "raw_signal", "reference", "difference"], default="contrast"
    )
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
    parser.add_argument(
        "--metadata-file", type=Path, default=None, help="CSV/JSON mapping filename to metadata fields."
    )
    parser.add_argument("--exclude-file", type=Path, default=None, help="CSV/JSON with exclude flags per filename.")
    parser.add_argument("--plugin-file", type=Path, default=None, help="Custom model JSON spec.")
    parser.add_argument(
        "--model-override",
        type=str,
        default=None,
        choices=[
            "Rabi",
            "SpinEcho",
            "DynamicDecoupling",
            "ODMR",
            "Ramsey",
            "T1",
            "DEERFrequency",
            "DEERDuration",
            "DEERDecay",
            "DEERPosition",
            "TimeTraceDecay",
            "GaussianPeaks",
        ],
        help="Force a specific model for ALL files.",
    )
    parser.add_argument(
        "--multistart", type=int, default=12, help="Number of random-seed restarts per fit (default: 12)."
    )
    parser.add_argument("--robust", action="store_true", help="Use subset-based robust fitting.")
    parser.add_argument("--odmr-multipeak", action="store_true", help="Use automatic multi-Lorentzian for ODMR data.")
    parser.add_argument("--dpi", type=int, default=180, help="Output plot resolution (default: 180).")
    parser.add_argument("--no-origin-bundle", action="store_true", help="Skip writing per-file origin bundle CSVs.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find .mat files recursively under --input-dir and use collision-safe output names.",
    )
    parser.add_argument("--skip-checkpoints", action="store_true", help="Skip files whose stem contains __checkpoint.")
    parser.add_argument(
        "--manifest", type=Path, help="Replay recorded sources and options; other fit flags are ignored."
    )
    parser.add_argument("--row", action="append", default=None, help="Recorded row ID to rerun (repeatable).")
    args = parser.parse_args()

    def _cli_progress(idx, total, name, msg):
        print(f"  [{idx}/{total}] {name}: {msg}")

    if args.manifest:
        result = replay_batch(args.manifest, selected_rows=args.row, progress_callback=_cli_progress)
    else:
        result = run_batch(
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
    print(
        f"Batch {result['state']}: {result['done']}/{result['total']} requested rows committed. Record: {result['manifest_path']}"
    )
    if result["state"] == "cancelled":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
