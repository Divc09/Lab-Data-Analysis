from __future__ import annotations

import csv
import json
import re
import sys
from io import BytesIO
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import (
    QEvent,
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QRectF,
    Qt,
    QSettings,
    QThread,
    Signal,
    QMimeData,
    QTimer,
)
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QShortcut
from scipy.signal import find_peaks
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QStatusBar,
    QStackedWidget,
    QTabWidget,
)
from scipy.optimize import curve_fit

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - optional runtime dependency during partial installs
    pg = None

try:
    import qtawesome as qta
except Exception:  # pragma: no cover - optional runtime dependency during partial installs
    qta = None

from .diagnostics import aic as _aic, bic as _bic, r_squared as _r_squared, rmse as _rmse, durbin_watson as _dw
from .analysis_tools import AnalysisStep, ProcessedSeries, apply_steps, process_series
from .fit_engine import FitResult, fit_model_multistart
from .fit_workflows import FitWorkflowConfig, fit_non_ramsey
from .io_mat import DataMode, ExperimentTrace, load_saved_data_mat
from .models import (
    MODEL_REGISTRY,
    build_custom_expression_model,
    odmr_lorentzian,
    odmr_multi_lorentzian,
    rabi_dual_model,
    rabi_model,
    rabi_shifted_model,
    ramsey_hyperfine_n14,
    ramsey_simple,
    spin_echo_model,
)
from .preprocess import apply_roi, bin_trace, smooth_trace
from .profiles import PROFILES, FitProfile, classify_status
from .rabi_utils import build_rabi_nv_metrics, rabi_envelope_bounds, rabi_envelope_metrics
from .ramsey import estimate_ramsey_frequency, fit_ramsey_physics_first
from .sessions import load_session, resolve_source, save_session, source_descriptor


@dataclass
class FitContext:
    trace: ExperimentTrace | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    y_smooth: np.ndarray | None = None
    fit_result: FitResult | None = None
    fit_target: np.ndarray | None = None
    status: str = "N/A"
    status_reason: str = ""
    profile: FitProfile | None = None
    odmr_peaks: np.ndarray | None = None
    excluded_points: set[int] | None = None
    exclusion_ranges: list[tuple[float, float]] | None = None
    loaded_traces: dict[str, ExperimentTrace] | None = None
    sample_metadata: dict[str, str] | None = None
    active_preset_name: str = "default"
    custom_model_spec: dict[str, Any] | None = None
    analysis_steps: list[AnalysisStep] = field(default_factory=list)
    processed: ProcessedSeries | None = None
    analysis_x: np.ndarray | None = None
    analysis_y: np.ndarray | None = None


@dataclass
class ParamMeta:
    symbol: str
    units: str
    description: str


@dataclass
class PlotOptions:
    show_data: bool = True
    show_smoothed: bool = True
    show_fit: bool = True
    show_rabi_envelope: bool = True
    show_peaks: bool = True
    show_signal_layer: bool = False
    show_reference_layer: bool = False
    show_difference_layer: bool = False
    show_iteration_layer: bool = False
    show_iteration_mean: bool = True
    show_iteration_std: bool = False
    iteration_std_mode: str = "Band"
    hide_failed_iterations: bool = True
    show_residual: bool = True
    show_fft_panel: bool = False
    show_legend: bool = True
    show_confidence: bool = False
    legend_loc: str = "best"
    legend_font_size: int = 9
    annotation_mode: str = "Compact"
    annotation_loc: str = "upper right"
    annotation_font_size: int = 9
    annotation_show_file_model: bool = True
    annotation_show_fit_metrics: bool = True
    annotation_show_t2rho: bool = True
    annotation_show_nv_metrics: bool = True
    annotation_show_params: bool = False
    legend_compact_labels: bool = True
    plot_style: str = "Line + scatter"
    fig_width: float = 10.0
    fig_height: float = 6.0
    save_dpi: int = 220
    export_png: bool = True
    export_pdf: bool = False
    export_svg: bool = False
    export_report_png: bool = False
    export_report_pdf: bool = False
    export_json: bool = True
    export_csv: bool = True
    export_origin_bundle: bool = False


PARAM_METADATA: dict[str, ParamMeta] = {
    "y0": ParamMeta("y_0", "arb.", "Vertical offset / baseline level."),
    "m": ParamMeta("m", "arb./ns", "Linear drift slope across scan window."),
    "A": ParamMeta("A", "arb.", "Oscillation/contrast amplitude."),
    "A1": ParamMeta("A_1", "arb.", "Primary oscillation amplitude."),
    "A2": ParamMeta("A_2", "arb.", "Secondary oscillation amplitude."),
    "C": ParamMeta("C", "arb.", "ODMR resonance amplitude."),
    "f": ParamMeta("f", "GHz", "Oscillation frequency (detuning or Rabi frequency)."),
    "f1": ParamMeta("f_1", "GHz", "First oscillation frequency component."),
    "f2": ParamMeta("f_2", "GHz", "Second oscillation frequency component."),
    "phi": ParamMeta("phi", "rad", "Initial oscillation phase."),
    "phi1": ParamMeta("phi_1", "rad", "First oscillation phase."),
    "phi2": ParamMeta("phi_2", "rad", "Second oscillation phase."),
    "tau": ParamMeta("tau", "ns", "Decay time constant."),
    "T": ParamMeta("T", "ns", "Envelope decay timescale."),
    "t0": ParamMeta("t_0", "ns", "Effective oscillation start time."),
    "tau_ramp": ParamMeta(r"\tau_r", "ns", "Microwave startup / pulse-area buildup timescale."),
    "beta": ParamMeta(r"\beta", "-", "Envelope stretch exponent."),
    "alpha": ParamMeta(r"\alpha", "1/ns^2", "Linear chirp term."),
    "T2": ParamMeta("T_2", "ns", "Spin-echo coherence time."),
    "T2_star": ParamMeta("T_2*", "ns", "Ramsey dephasing time."),
    "n": ParamMeta("n", "-", "Stretch exponent (1=exp, 2=Gaussian-like)."),
    "w": ParamMeta("gamma", "axis units", "Half width at half maximum."),
    "x0": ParamMeta("x_0", "axis units", "ODMR resonance center."),
}


def _param_meta(name: str) -> ParamMeta:
    return PARAM_METADATA.get(name, ParamMeta(name, "", "Model parameter."))


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except Exception:
        return default


def _spin_echo_seed_and_bounds(experiment_type: str, tspan: float) -> tuple[list[float], list[float], np.ndarray]:
    lower = [-0.2, -0.2, 1.0, 0.5]
    if experiment_type == "DynamicDecoupling":
        upper = [0.2, 0.2, max(200000.0, 8.0 * tspan), 5.0]
        seed0 = np.array([0.0, 0.0, max(1000.0, tspan / 3.0), 1.5])
    else:
        upper = [0.2, 0.2, 20000.0, 5.0]
        seed0 = np.array([0.0, 0.0, max(100.0, tspan / 3.0), 1.5])
    return lower, upper, seed0


def _nv_metrics(experiment_type: str, result: FitResult, trace_metadata: dict[str, object] | None = None) -> dict[str, float]:
    semantic_experiment_type = experiment_type
    if experiment_type == "DEERFrequency":
        experiment_type = "ODMR"
    elif experiment_type == "DEERDuration":
        experiment_type = "Rabi"
    elif experiment_type == "DEERDecay":
        experiment_type = "SpinEcho"
    p = {n: float(v) for n, v in zip(result.param_names, result.params)}
    e = {n: float(v) for n, v in zip(result.param_names, result.errors)}
    out: dict[str, float] = {}
    if experiment_type == "Ramsey":
        if "tau" in p:
            out["T2_star_ns"] = p["tau"]
        if "T2_star" in p:
            out["T2_star_ns"] = p["T2_star"]
        if "n" in p:
            out["stretch_n"] = p["n"]
        if "f" in p:
            out["detuning_MHz"] = p["f"] * 1000.0
        if "f1" in p:
            out["f1_MHz"] = p["f1"] * 1000.0
        if "f2" in p:
            out["f2_MHz"] = p["f2"] * 1000.0
    elif experiment_type == "Rabi":
        physical_params = p
        if isinstance(result.extras, dict) and result.extras.get("rabi_physical_reference_params"):
            physical_params = {
                k: float(v) for k, v in dict(result.extras.get("rabi_physical_reference_params", {})).items()
            }
        out.update(build_rabi_nv_metrics(physical_params, metadata=trace_metadata, errors=e))
        if isinstance(result.extras, dict):
            out.update(rabi_envelope_metrics(result.extras.get("rabi_envelope")))
        if "f1" in p:
            out["f1_MHz"] = p["f1"] * 1000.0
        if "f2" in p:
            out["f2_MHz"] = p["f2"] * 1000.0
        if "tau" in p:
            out["decay_ns"] = p["tau"]
        if "t0" in p:
            out["t0_ns"] = p["t0"]
    elif experiment_type in ("SpinEcho", "T1", "DynamicDecoupling"):
        if experiment_type == "T1":
            if "T1" in p:
                out["T1_ns"] = p["T1"]
        elif "T2" in p:
            if experiment_type == "SpinEcho":
                out["T2_ns"] = p["T2"]
            else:
                out["T2_dd_ns"] = p["T2"]
        if "n" in p:
            out["stretch_n"] = p["n"]
        if experiment_type == "DynamicDecoupling" and trace_metadata is not None:
            dd_count = trace_metadata.get("dd_pulse_count")
            rf_ghz = trace_metadata.get("rf_frequency_ghz")
            pi_ns = trace_metadata.get("pi_pulse_ns")
            if dd_count is not None:
                out["dd_pulse_count"] = float(dd_count)
            if rf_ghz is not None:
                out["rf_frequency_GHz"] = float(rf_ghz)
            if pi_ns is not None:
                out["pi_pulse_ns"] = float(pi_ns)
    elif experiment_type == "ODMR":
        if "x0" in p:
            out["center_freq"] = p["x0"]
        if "w" in p:
            out["fwhm"] = 2.0 * p["w"]
    if semantic_experiment_type.startswith("DEER") and trace_metadata is not None:
        for key in ("rf_frequency_ghz", "rf2_frequency_ghz", "rf2_duration_ns", "pi_pulse_ns", "tau_ns"):
            if trace_metadata.get(key) is not None:
                out[key] = float(trace_metadata[key])  # type: ignore[arg-type]
    return out


class PreferencesDialog(QDialog):
    def __init__(self, parent: QWidget | None, opts: PlotOptions):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.width_spin = NoScrollDoubleSpinBox()
        self.width_spin.setRange(4.0, 30.0)
        self.width_spin.setDecimals(1)
        self.width_spin.setValue(opts.fig_width)
        form.addRow("Figure width (in)", self.width_spin)

        self.height_spin = NoScrollDoubleSpinBox()
        self.height_spin.setRange(3.0, 20.0)
        self.height_spin.setDecimals(1)
        self.height_spin.setValue(opts.fig_height)
        form.addRow("Figure height (in)", self.height_spin)

        self.dpi_spin = NoScrollSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(opts.save_dpi)
        form.addRow("Save DPI", self.dpi_spin)

        self.legend_font_spin = NoScrollSpinBox()
        self.legend_font_spin.setRange(6, 20)
        self.legend_font_spin.setValue(opts.legend_font_size)
        form.addRow("Legend font size", self.legend_font_spin)

        self.residual_chk = QCheckBox("Show residual by default")
        self.residual_chk.setChecked(opts.show_residual)
        form.addRow(self.residual_chk)

        self.legend_chk = QCheckBox("Show legend by default")
        self.legend_chk.setChecked(opts.show_legend)
        form.addRow(self.legend_chk)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class NoScrollSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoScrollComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SmoothScrollArea(QScrollArea):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scroll_anim = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._scroll_anim.setDuration(130)
        self._scroll_anim.setEasingCurve(QEasingCurve.OutCubic)

    def wheelEvent(self, event):
        bar = self.verticalScrollBar()
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        target = bar.value() - int(delta * 0.55)
        target = max(bar.minimum(), min(bar.maximum(), target))
        self._scroll_anim.stop()
        self._scroll_anim.setStartValue(bar.value())
        self._scroll_anim.setEndValue(target)
        self._scroll_anim.start()
        event.accept()


class IterationSelectionDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        *,
        count: int,
        selected: set[int],
        show_mean: bool,
    ):
        super().__init__(parent)
        self.setWindowTitle("Iterations")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        self.mean_chk = QCheckBox("Mean")
        self.mean_chk.setChecked(show_mean)
        layout.addWidget(self.mean_chk)

        button_row = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        button_row.addWidget(btn_all)
        button_row.addWidget(btn_none)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(220)
        for idx in range(count):
            item = QListWidgetItem(f"Iteration {idx + 1}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if idx in selected else Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_all(self):
        for idx in range(self.list_widget.count()):
            self.list_widget.item(idx).setCheckState(Qt.Checked)

    def _select_none(self):
        for idx in range(self.list_widget.count()):
            self.list_widget.item(idx).setCheckState(Qt.Unchecked)

    def selected_indices(self) -> set[int]:
        return {
            idx
            for idx in range(self.list_widget.count())
            if self.list_widget.item(idx).checkState() == Qt.Checked
        }

    def show_mean(self) -> bool:
        return self.mean_chk.isChecked()



class BatchFitWorker(QThread):
    """Worker thread for batch fitting."""
    progress = Signal(int, int, str, str)   # idx, total, filename, msg
    finished = Signal(str)                  # output_dir path
    error = Signal(str)

    def __init__(self, kwargs: dict):
        super().__init__()
        self._kwargs = kwargs

    def run(self):
        from .pro_batch import run_batch
        try:
            self._kwargs["progress_callback"] = self._emit_progress
            run_batch(**self._kwargs)
            self.finished.emit(str(self._kwargs["output_dir"]))
        except Exception as e:
            self.error.emit(str(e))

    def _emit_progress(self, idx, total, name, msg):
        self.progress.emit(idx, total, name, msg)


class BatchFitDialog(QDialog):
    """Dialog for running batch fitting from the GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Fit")
        self.setMinimumWidth(560)
        self._worker: BatchFitWorker | None = None
        layout = QVBoxLayout(self)

        # --- Folder selectors ---
        folder_grp = QGroupBox("Folders")
        folder_gl = QGridLayout(folder_grp)
        folder_gl.addWidget(QLabel("Input folder:"), 0, 0)
        self.input_dir_edit = QLineEdit()
        folder_gl.addWidget(self.input_dir_edit, 0, 1)
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(lambda: self._browse(self.input_dir_edit))
        folder_gl.addWidget(btn_in, 0, 2)
        folder_gl.addWidget(QLabel("Output folder:"), 1, 0)
        self.output_dir_edit = QLineEdit()
        folder_gl.addWidget(self.output_dir_edit, 1, 1)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(lambda: self._browse(self.output_dir_edit))
        folder_gl.addWidget(btn_out, 1, 2)
        layout.addWidget(folder_grp)

        # --- Options ---
        opts_grp = QGroupBox("Options")
        opts_form = QFormLayout(opts_grp)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["contrast", "raw_signal", "difference"])
        opts_form.addRow("Data mode:", self.mode_combo)

        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(1, 100)
        self.bin_spin.setValue(1)
        opts_form.addRow("Bin size:", self.bin_spin)

        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(1, 100)
        self.smooth_spin.setValue(1)
        opts_form.addRow("Smooth window:", self.smooth_spin)

        self.multistart_spin = QSpinBox()
        self.multistart_spin.setRange(1, 100)
        self.multistart_spin.setValue(12)
        opts_form.addRow("Multistart seeds:", self.multistart_spin)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Auto-detect", "Rabi", "SpinEcho", "DynamicDecoupling", "ODMR", "Ramsey", "T1", "DEERFrequency", "DEERDuration", "DEERDecay", "DEERPosition"])
        opts_form.addRow("Model override:", self.model_combo)

        self.robust_chk = QCheckBox("Robust fit (subset-based)")
        opts_form.addRow(self.robust_chk)

        self.odmr_multi_chk = QCheckBox("ODMR multi-peak")
        opts_form.addRow(self.odmr_multi_chk)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(180)
        opts_form.addRow("Plot DPI:", self.dpi_spin)

        layout.addWidget(opts_grp)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        layout.addWidget(self.log_text)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Batch")
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    def _browse(self, line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if d:
            line_edit.setText(d)

    def _on_run(self):
        input_dir = self.input_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "Missing folders", "Please select both input and output folders.")
            return

        model_ov = self.model_combo.currentText()
        kwargs = {
            "input_dir": Path(input_dir),
            "output_dir": Path(output_dir),
            "mode": self.mode_combo.currentText(),
            "bin_size": self.bin_spin.value(),
            "smooth_window": self.smooth_spin.value(),
            "multistart": self.multistart_spin.value(),
            "model_override": model_ov if model_ov != "Auto-detect" else None,
            "robust": self.robust_chk.isChecked(),
            "odmr_multipeak": self.odmr_multi_chk.isChecked(),
            "dpi": self.dpi_spin.value(),
        }

        self.run_btn.setEnabled(False)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self._worker = BatchFitWorker(kwargs)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, idx, total, name, msg):
        pct = int(100 * (idx + 1) / max(1, total))
        self.progress_bar.setValue(pct)
        self.log_text.append(f"[{idx+1}/{total}] {name}: {msg}")

    def _on_finished(self, out_dir: str):
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.log_text.append(f"\n✓ Batch complete → {out_dir}")
        import os
        os.startfile(out_dir)

    def _on_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.log_text.append(f"\n✗ ERROR: {msg}")
        QMessageBox.critical(self, "Batch Error", msg)


class FitWorker(QThread):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SmartFitterMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmartFitter")
        self.resize(1480, 900)
        self.ctx = FitContext()
        self.ctx.excluded_points = set()
        self.ctx.exclusion_ranges = []
        self.ctx.loaded_traces = {}
        self.ctx.sample_metadata = {}
        self.settings = QSettings("BacklundLab", "SmartFitterPy")
        self.plot_opts = PlotOptions()
        self.span_selector: SpanSelector | None = None
        self.main_splitter: QSplitter | None = None
        self._startup_splitter_applied = False
        self._drag_depth = 0
        self._is_loading = False
        self._mode_override_active = False
        self._syncing_plot_controls = False
        self._selected_plot_point: dict[str, float] | None = None
        self._selected_iteration_indices: set[int] = set()
        self._scan_cursor: tuple[float, float] | None = None
        self._ax_secondary = None
        self._secondary_series: list[np.ndarray] = []
        self._secondary_plot_data: tuple[np.ndarray, np.ndarray, str] | None = None
        self._scan_colorbar = None
        self._scan_colorbar_ax = None
        self._crosshair_v = None
        self._crosshair_h = None
        self.live_selection_marker = None
        self.live_selection_vline = None
        self.live_selection_hline = None
        self._live_secondary_view = None
        self._live_secondary_update = None
        self.equation_map: dict[str, str] = {
            "RamseySimple": r"$y_0 + A e^{-(t/\tau)^n}\cos(2\pi f t + \phi)$",
            "RamseyHyperfine": r"$y_0 + A e^{-(t/\tau)^n}\frac{1}{3}\sum_{k=-1}^{1}\cos(2\pi(f+k\Delta f)t+\phi)$",
            "RamseyAuto": "Automatic selection across Ramsey models",
            "Rabi": r"$c + a\cos(2\pi f[t-\tau_r(1-e^{-t/\tau_r})]+\phi)$",
            "SpinEcho": r"$y_0 + A e^{-(t/T_2)^n}$",
            "T1": r"$y_0 + A(1-e^{-(t/T_1)^n})$",
            "ODMR": r"$y_0 + C\frac{w^2}{(x-x_0)^2+w^2}$",
            "CustomModel": "User equation",
        }
        self.fit_worker: FitWorker | None = None
        self._param_undo_stack: list[dict] = []
        self._param_redo_stack: list[dict] = []
        self._analysis_undo_stack: list[dict[str, Any]] = []
        self._analysis_redo_stack: list[dict[str, Any]] = []
        self.equation_expr_template: dict[str, str] = {
            "RamseySimple": "y0 + A*np.exp(-((np.abs(t/tau))**n))*np.cos(2*np.pi*f*t + phi)",
            "RamseyHyperfine": "y0 + A*np.exp(-((np.abs(t/tau))**n))*((np.cos(2*np.pi*(f-0.00216)*t+phi)+np.cos(2*np.pi*f*t+phi)+np.cos(2*np.pi*(f+0.00216)*t+phi))/3.0)",
            "Rabi": "c + a*np.cos(2*np.pi*f*(t - tau_ramp*(1.0 - np.exp(-t/tau_ramp))) + phi)",
            "SpinEcho": "y0 + A*np.exp(-((np.abs(t/T2))**n))",
            "T1": "y0 + A*(1.0 - np.exp(-((np.abs(t/T1))**n)))",
            "ODMR": "y0 + C*(w**2/((t-x0)**2 + w**2))",
        }
        self._build_ui()
        self._load_preferences()
        self._apply_plot_controls_to_state()
        self._apply_figure_size()
        self._apply_theme()
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("axes_leave_event", self._on_plot_leave)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+O"), self, self.on_load)
        QShortcut(QKeySequence("Ctrl+F"), self, self.on_fit)
        QShortcut(QKeySequence("Ctrl+Shift+F"), self, self.on_robust_fit)
        QShortcut(QKeySequence("Ctrl+S"), self, self.on_save_results)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self.copy_export_figure)
        QShortcut(QKeySequence("Escape"), self, self.clear_selected_marker)

        # Drag-and-drop support
        self.setAcceptDrops(True)
        if self.centralWidget() is not None:
            self.centralWidget().setAcceptDrops(True)
            self.centralWidget().installEventFilter(self)
        self.installEventFilter(self)

    def _apply_theme(self):
        dark_qss = """
        QMainWindow, QWidget {
            background-color: #171918;
            color: #e8ebe8;
            font-family: "Aptos", "Calibri", sans-serif;
            font-size: 10pt;
        }
        QGroupBox {
            border: 1px solid #353b36;
            border-radius: 4px;
            margin-top: 8px;
            font-weight: 600;
            padding-top: 12px;
            background-color: #202321;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
            color: #d9dfd9;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
            background-color: #1d211f;
            border: 1px solid #3d4540;
            border-radius: 4px;
            color: #f2f5f2;
            padding: 4px;
            selection-background-color: #3d6f5f;
        }
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QTextEdit:focus {
            border: 1px solid #52947c;
            background-color: #222824;
        }
        QPushButton {
            background-color: #282d2a;
            border: 1px solid #444c46;
            border-radius: 4px;
            padding: 5px 10px;
            color: #f0f0f0;
        }
        QPushButton:hover {
            background-color: #343b36;
            border-color: #5b665e;
        }
        QPushButton:pressed {
            background-color: #1c211e;
        }
        QPushButton#PrimaryAction {
            background-color: #367963;
            border-color: #55a384;
            color: #ffffff;
            font-weight: 600;
        }
        QPushButton#ModeNav {
            border: 0;
            border-radius: 4px;
            padding: 8px 10px;
            text-align: left;
            background-color: transparent;
            color: #cfcfcf;
        }
        QPushButton#ModeNav:checked {
            background-color: #29322d;
            color: #ffffff;
            border-left: 3px solid #5aa989;
        }
        QWidget#ModeRail {
            background-color: #121513;
            border-right: 1px solid #303832;
        }
        QLabel#PaneTitle {
            color: #f2f2f2;
            font-weight: 700;
            font-size: 12pt;
            padding: 3px 0;
        }
        QLabel#MutedLabel {
            color: #a7a7a7;
        }
        QTableWidget {
            gridline-color: #343434;
            background-color: #202321;
            alternate-background-color: #252a27;
            selection-background-color: #367963;
            border: 1px solid #353b36;
        }
        QHeaderView::section {
            background-color: #292f2b;
            color: #d9dfd9;
            border: 1px solid #3b443d;
            padding: 4px;
        }
        QScrollBar:vertical {
            background: #171918;
            width: 12px;
        }
        QScrollBar::handle:vertical {
            background: #4a554d;
            min-height: 20px;
            border-radius: 4px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 13px;
            height: 13px;
        }
        QTabWidget::pane {
            border: 1px solid #303030;
            top: -1px;
        }
        QTabBar::tab {
            background: #202020;
            color: #cfcfcf;
            border: 1px solid #303030;
            padding: 6px 10px;
        }
        QTabBar::tab:selected {
            background: #2b332e;
            color: #ffffff;
        }
        QStatusBar { background: #121513; color: #b9c4bb; border-top: 1px solid #303832; }
        QToolTip { background: #272e2a; color: #f2f5f2; border: 1px solid #566159; padding: 4px; }
        """
        self.setStyleSheet(dark_qss)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.root_widget = root

        self.drop_banner = QLabel("Drop .mat files anywhere in the window to load them")
        self.drop_banner.setAlignment(Qt.AlignCenter)
        self.drop_banner.setVisible(False)
        self.drop_banner.setStyleSheet(
            "QLabel { background-color: rgba(33, 150, 243, 0.18); "
            "border: 2px dashed #42a5f5; border-radius: 8px; padding: 10px; "
            "color: #d7ecff; font-weight: bold; }"
        )
        layout.addWidget(self.drop_banner)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        layout.addWidget(content, 1)

        self.mode_rail = QWidget()
        self.mode_rail.setObjectName("ModeRail")
        self.mode_rail.setFixedWidth(168)
        rail_layout = QVBoxLayout(self.mode_rail)
        rail_layout.setContentsMargins(10, 12, 10, 12)
        rail_layout.setSpacing(6)
        rail_title = QLabel("SmartFitter")
        rail_title.setObjectName("PaneTitle")
        rail_layout.addWidget(rail_title)
        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.setExclusive(True)
        self.mode_buttons: dict[str, QPushButton] = {}
        for idx, (key, label, tip, icon_name) in enumerate(
            [
                ("single", "Single File", "Load, inspect, fit, and export one trace.", "fa5s.file-import"),
                ("batch", "Batch", "Process folders and review batch results.", "fa5s.table"),
                ("scan", "Scan Explorer", "Inspect 1D/2D scans, linecuts, and cursor readouts.", "fa5s.crosshairs"),
                ("results", "Results", "Review fit diagnostics, parameters, and export settings.", "fa5s.chart-line"),
            ]
        ):
            btn = QPushButton(label)
            btn.setObjectName("ModeNav")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            self._set_button_icon(btn, icon_name)
            self.mode_button_group.addButton(btn, idx)
            self.mode_buttons[key] = btn
            rail_layout.addWidget(btn)
        rail_layout.addStretch(1)
        content_layout.addWidget(self.mode_rail)

        self.mode_stack = QStackedWidget()
        content_layout.addWidget(self.mode_stack, 1)

        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(10, 10, 10, 10)
        single_layout.setSpacing(8)
        self.mode_stack.addWidget(single_page)

        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax_main = self.fig.add_subplot(2, 1, 1)
        self.ax_res = self.fig.add_subplot(2, 1, 2)
        self._ax_main_default_pos = self.ax_main.get_position().frozen()
        self._ax_res_default_pos = self.ax_res.get_position().frozen()
        self.toolbar = NavigationToolbar(self.canvas, root)
        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_scroll = SmoothScrollArea()
        left_scroll.setWidgetResizable(True)
        left_layout.addWidget(left_scroll)
        left_inner = QWidget()
        left_scroll.setWidget(left_inner)
        left_inner_layout = QVBoxLayout(left_inner)
        splitter.addWidget(left)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self._build_plot_quick_toolbar(center_layout)
        self.plot_tabs = QTabWidget()
        live_page = QWidget()
        live_layout = QVBoxLayout(live_page)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self.live_plot_widget = None
        self.live_image_item = None
        self.live_curve_items: list[Any] = []
        if pg is not None:
            pg.setConfigOptions(antialias=True, background="#181818", foreground="#d8d8d8")
            self.live_plot_widget = pg.PlotWidget()
            self.live_plot_widget.setAcceptDrops(False)
            self.live_plot_widget.viewport().setAcceptDrops(False)
            self.live_plot_widget.setLabel("bottom", "X")
            self.live_plot_widget.setLabel("left", "Signal")
            self.live_plot_widget.showGrid(x=True, y=True, alpha=0.22)
            self.live_plot_widget.scene().sigMouseMoved.connect(self._on_live_plot_mouse_moved)
            self.live_plot_widget.scene().sigMouseClicked.connect(self._on_live_plot_mouse_clicked)
            self.live_plot_widget.setToolTip("Click a sample to select it. Drag to pan, wheel to zoom, and hover for coordinates.")
            live_layout.addWidget(self.live_plot_widget)
        else:
            missing = QLabel("Install pyqtgraph to enable the live inspection plot.")
            missing.setObjectName("MutedLabel")
            missing.setAlignment(Qt.AlignCenter)
            live_layout.addWidget(missing)
        self.plot_tabs.addTab(live_page, "Live inspect")
        export_page = QWidget()
        export_layout = QVBoxLayout(export_page)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.addWidget(self.toolbar)
        export_layout.addWidget(self.canvas)
        self.plot_tabs.addTab(export_page, "Export figure")
        center_layout.addWidget(self.plot_tabs)
        self.live_readout_lbl = QLabel("Point readout: click a point in Live inspect")
        self.live_readout_lbl.setObjectName("MutedLabel")
        self.live_readout_lbl.setMinimumHeight(24)
        self.live_readout_lbl.setToolTip("Shows the nearest data coordinates. Click the selected point again or press Escape to remove its marker.")
        readout_row = QHBoxLayout()
        readout_row.setContentsMargins(0, 0, 0, 0)
        readout_row.addWidget(self.live_readout_lbl, 1)
        self.clear_marker_btn = QPushButton("Clear marker")
        self.clear_marker_btn.setToolTip("Remove the yellow selected-point marker and its legend entry. This never changes the data.")
        self.clear_marker_btn.setEnabled(False)
        self.clear_marker_btn.clicked.connect(self.clear_selected_marker)
        readout_row.addWidget(self.clear_marker_btn)
        center_layout.addLayout(readout_row)
        splitter.addWidget(center)

        single_layout.addWidget(splitter)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 6)
        splitter.setStretchFactor(2, 3)

        scroll = SmoothScrollArea()
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)
        right_inner = QWidget()
        scroll.setWidget(right_inner)
        right_inner_layout = QVBoxLayout(right_inner)

        data_box = QGroupBox("Data")
        self.data_box = data_box
        data_gl = QGridLayout(data_box)
        self.btn_load = QPushButton("Load .mat")
        self.btn_load.setToolTip("Load a MATLAB file containing legacy savedData or new data/scanInfo output.")
        self.btn_load.clicked.connect(self.on_load)
        data_gl.addWidget(self.btn_load, 0, 0, 1, 2)
        self.load_help_lbl = QLabel("Open one file to inspect and fit it, or drag files into the window to load them quickly.")
        self.load_help_lbl.setWordWrap(True)
        self.load_help_lbl.setStyleSheet("QLabel { color: #aab6c8; }")
        self.load_help_lbl.setProperty("drawerHidden", True)
        self.load_help_lbl.setVisible(False)
        data_gl.addWidget(self.load_help_lbl, 1, 0, 1, 2)

        self.profile_combo = NoScrollComboBox()
        self.profile_combo.addItems(["Auto", "Ramsey", "Rabi", "SpinEcho", "DynamicDecoupling", "T1", "ODMR", "DEERFrequency", "DEERDuration", "DEERDecay", "DEERPosition", "LineScan"])
        self.profile_combo.setToolTip("Choose the experiment family. Auto follows the detected type from the file.")
        self.profile_combo.currentTextChanged.connect(self.apply_profile_defaults)
        data_gl.addWidget(QLabel("Experiment type"), 2, 0)
        data_gl.addWidget(self.profile_combo, 2, 1)

        self.mode_combo = NoScrollComboBox()
        self.mode_combo.addItems(["contrast", "signal", "reference", "difference"])
        self.mode_combo.setToolTip("Observable view used for plotting and fitting.")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        data_gl.addWidget(QLabel("Observable view"), 3, 0)
        data_gl.addWidget(self.mode_combo, 3, 1)
        self.profile_hint_lbl = QLabel("No file loaded.")
        self.profile_hint_lbl.setWordWrap(True)
        self.profile_hint_lbl.setStyleSheet("QLabel { color: #8fd3ff; }")
        data_gl.addWidget(self.profile_hint_lbl, 4, 0, 1, 2)
        left_inner_layout.addWidget(data_box)

        overlay_box = QGroupBox("Overlays")
        self.overlay_box = overlay_box
        ov_gl = QGridLayout(overlay_box)
        btn_add_overlay = QPushButton("Add Overlay Files")
        btn_add_overlay.setToolTip("Add additional traces to overlay and compare.")
        btn_add_overlay.clicked.connect(self.on_add_overlays)
        ov_gl.addWidget(btn_add_overlay, 0, 0)
        btn_remove_overlay = QPushButton("Remove Selected")
        btn_remove_overlay.setToolTip("Remove selected overlay traces from registry.")
        btn_remove_overlay.clicked.connect(self.on_remove_overlays)
        ov_gl.addWidget(btn_remove_overlay, 0, 1)
        btn_clear_overlay = QPushButton("Clear Overlays")
        btn_clear_overlay.setToolTip("Clear all overlays except currently loaded trace.")
        btn_clear_overlay.clicked.connect(self.on_clear_overlays)
        ov_gl.addWidget(btn_clear_overlay, 1, 0, 1, 2)
        self.overlay_list = QListWidget()
        self.overlay_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.overlay_list.setToolTip("Loaded traces. Selected traces are shown in overlay.")
        self.overlay_list.itemDoubleClicked.connect(self.on_overlay_set_primary)
        ov_gl.addWidget(self.overlay_list, 2, 0, 1, 2)
        self.baseline_combo = NoScrollComboBox()
        self.baseline_combo.addItem("None")
        self.baseline_combo.setToolTip("Optional baseline trace to subtract before fitting.")
        self.baseline_combo.currentTextChanged.connect(self._refresh_processed)
        ov_gl.addWidget(QLabel("Baseline subtract"), 3, 0)
        ov_gl.addWidget(self.baseline_combo, 3, 1)
        self.compare_metric_combo = NoScrollComboBox()
        self.compare_metric_combo.addItems(["r2", "T2_star_ns", "T2_dd_ns", "T1_ns", "pi_time_ns", "pi_time_ns_nominal", "first_peak_ns", "delay_ns", "rabi_freq_MHz", "center_freq", "fwhm"])
        self.compare_metric_combo.setToolTip("Metric used in overlay comparison table.")
        ov_gl.addWidget(QLabel("Compare metric"), 4, 0)
        ov_gl.addWidget(self.compare_metric_combo, 4, 1)
        btn_compare = QPushButton("Build Comparison")
        btn_compare.clicked.connect(self.on_compare_traces)
        ov_gl.addWidget(btn_compare, 5, 0, 1, 2)
        left_inner_layout.addWidget(overlay_box)

        prep_box = QGroupBox("Preprocess")
        self.prep_box = prep_box
        prep_gl = QGridLayout(prep_box)
        self.bin_spin = NoScrollSpinBox()
        self.bin_spin.setRange(1, 500)
        self.bin_spin.setValue(1)
        self.bin_spin.setToolTip("Average neighboring points by this bin size.")
        self.bin_spin.valueChanged.connect(self._refresh_processed)
        prep_gl.addWidget(QLabel("Bin size"), 0, 0)
        prep_gl.addWidget(self.bin_spin, 0, 1)

        self.smooth_spin = NoScrollSpinBox()
        self.smooth_spin.setRange(1, 201)
        self.smooth_spin.setValue(1)
        self.smooth_spin.setToolTip("Savitzky-Golay window. 1=off.")
        self.smooth_spin.valueChanged.connect(self._refresh_processed)
        prep_gl.addWidget(QLabel("Smooth window"), 1, 0)
        prep_gl.addWidget(self.smooth_spin, 1, 1)

        self.roi_min = QLineEdit("")
        self.roi_min.setToolTip("Minimum ROI bound in ns.")
        self.roi_max = QLineEdit("")
        self.roi_max.setToolTip("Maximum ROI bound in ns.")
        self.roi_min.editingFinished.connect(self._refresh_processed)
        self.roi_max.editingFinished.connect(self._refresh_processed)
        prep_gl.addWidget(QLabel("ROI min (ns)"), 2, 0)
        prep_gl.addWidget(self.roi_min, 2, 1)
        prep_gl.addWidget(QLabel("ROI max (ns)"), 3, 0)
        prep_gl.addWidget(self.roi_max, 3, 1)
        left_inner_layout.addWidget(prep_box)

        strategy_box = QGroupBox("Fit")
        st_layout = QVBoxLayout(strategy_box)
        self.strategy_layout = st_layout
        self.strategy_box = strategy_box
        strategy_help = QLabel("Start with the detected experiment type and default analysis mode. Use advanced controls only when you need to override model assumptions or manual constraints.")
        strategy_help.setWordWrap(True)
        strategy_help.setStyleSheet("QLabel { color: #aab6c8; }")
        strategy_help.setProperty("drawerHidden", True)
        strategy_help.setVisible(False)
        st_layout.addWidget(strategy_help)

        # Common settings
        common_form = QFormLayout()
        
        self.model_combo = NoScrollComboBox()
        self.model_combo.addItems(["RamseyAuto", "RamseySimple", "RamseyHyperfine", "Rabi", "SpinEcho", "T1", "ODMR", "DEERFrequency", "DEERDuration", "DEERDecay", "LineScan", "CustomModel"])
        self.model_combo.setToolTip("Analysis mode used for fitting.")
        self.model_combo.currentTextChanged.connect(self._update_contextual_visibility)
        common_form.addRow("Analysis mode", self.model_combo)

        self.multistart_spin = NoScrollSpinBox()
        self.multistart_spin.setRange(4, 120)
        self.multistart_spin.setValue(20)
        self.multistart_spin.setToolTip("How many seeded fit attempts to try before keeping the best result.")
        common_form.addRow("Multistart", self.multistart_spin)
        self.multistart_help_lbl = QLabel("Multistart tries the optimizer from several seeded guesses and keeps the strongest solution. Higher values can improve stability but take longer.")
        self.multistart_help_lbl.setWordWrap(True)
        self.multistart_help_lbl.setStyleSheet("QLabel { color: #aab6c8; }")
        self.multistart_help_lbl.setProperty("drawerHidden", True)
        self.multistart_help_lbl.setVisible(False)

        st_layout.addLayout(common_form)
        st_layout.addWidget(self.multistart_help_lbl)

        # Rabi specific
        self.rabi_container = QGroupBox("Rabi Settings")
        rabi_layout = QFormLayout(self.rabi_container)
        
        self.rabi_mode_combo = NoScrollComboBox()
        self.rabi_mode_combo.addItems([
            "Phase-ramp (recommended physical fit)",
            "Long damped Rabi",
            "Chirped cosine (best overlay)",
            "Constant-frequency damped cosine",
            "Auto compare (advanced)",
        ])
        self.rabi_mode_combo.setCurrentIndex(0)
        self.rabi_mode_combo.setToolTip("Choose the internal Rabi fit model family.")
        self.rabi_mode_combo.currentTextChanged.connect(self._update_rabi_model_help)
        rabi_layout.addRow("Rabi model", self.rabi_mode_combo)
        self.rabi_model_help_lbl = QLabel("")
        self.rabi_model_help_lbl.setWordWrap(True)
        self.rabi_model_help_lbl.setStyleSheet("QLabel { color: #aab6c8; }")
        rabi_layout.addRow(self.rabi_model_help_lbl)
        
        self.rabi_pi_lock_chk = QCheckBox("Lock to pi / pi/2")
        self.rabi_pi_lock_chk.setToolTip("When enabled, convert specified pi or pi/2 times to frequency and lock f.")
        rabi_layout.addRow(self.rabi_pi_lock_chk)
        
        self.rabi_pi_ns = NoScrollDoubleSpinBox()
        self.rabi_pi_ns.setDecimals(3)
        self.rabi_pi_ns.setRange(0.001, 1e9)
        self.rabi_pi_ns.setValue(200.0)
        rabi_layout.addRow("pi time (ns)", self.rabi_pi_ns)
        
        self.rabi_pi2_ns = NoScrollDoubleSpinBox()
        self.rabi_pi2_ns.setDecimals(3)
        self.rabi_pi2_ns.setRange(0.001, 1e9)
        self.rabi_pi2_ns.setValue(100.0)
        rabi_layout.addRow("pi/2 time (ns)", self.rabi_pi2_ns)

        self.rabi_dead_time_ns = NoScrollDoubleSpinBox()
        self.rabi_dead_time_ns.setDecimals(3)
        self.rabi_dead_time_ns.setRange(-1e6, 1e9)
        self.rabi_dead_time_ns.setValue(0.0)
        self.rabi_dead_time_ns.setToolTip("Optional manual timing-offset override used when locking pi or pi/2. Reported delay on fitted Rabi traces is derived from the first fitted peak.")
        self.rabi_dead_time_ns.valueChanged.connect(self._on_rabi_dead_time_changed)
        rabi_layout.addRow("Delay override (ns)", self.rabi_dead_time_ns)
        self.rabi_sem_weight_chk = QCheckBox("Use iteration SEM weighting when available")
        self.rabi_sem_weight_chk.setChecked(True)
        self.rabi_sem_weight_chk.setToolTip("When full per-iteration contrast data are available, weight the fit by the SEM of each programmed pulse point.")
        rabi_layout.addRow(self.rabi_sem_weight_chk)
        
        st_layout.addWidget(self.rabi_container)

        self.t1_container = QGroupBox("T1 Settings")
        t1_layout = QFormLayout(self.t1_container)
        self.t1_prep_combo = NoScrollComboBox()
        self.t1_prep_combo.addItems(["Auto", "Direct T1", "Normalize then 1-data"])
        self.t1_prep_combo.setToolTip("Choose whether T1 should be fit directly or after normalization followed by 1-data.")
        self.t1_prep_combo.currentTextChanged.connect(self._update_equation_display)
        t1_layout.addRow("T1 preprocessing", self.t1_prep_combo)
        self.t1_model_combo = NoScrollComboBox()
        self.t1_model_combo.addItems(["Regular exponential", "Stretched exponential"])
        self.t1_model_combo.setCurrentText("Stretched exponential")
        self.t1_model_combo.setToolTip("Choose between a regular exponential and a stretched exponential T1 model.")
        self.t1_model_combo.currentTextChanged.connect(self._update_equation_display)
        t1_layout.addRow("T1 model family", self.t1_model_combo)
        st_layout.addWidget(self.t1_container)

        # ODMR specific
        self.odmr_container = QGroupBox("ODMR Settings")
        odmr_layout = QFormLayout(self.odmr_container)
        
        self.odmr_display_combo = NoScrollComboBox()
        self.odmr_display_combo.addItems(["Peaks (measured view)", "Dips (inverted view)"])
        self.odmr_display_combo.setToolTip("ODMR display only: show measured peaks or invert them to a dip view.")
        self.odmr_display_combo.currentTextChanged.connect(self._plot_raw)
        odmr_layout.addRow("ODMR display", self.odmr_display_combo)
        self.odmr_polarity_combo = NoScrollComboBox()
        self.odmr_polarity_combo.addItems(["Auto", "Peak", "Dip"])
        self.odmr_polarity_combo.setCurrentText("Peak")
        self.odmr_polarity_combo.setToolTip("Choose whether ODMR should be fit as positive peaks, negative dips, or selected automatically.")
        odmr_layout.addRow("ODMR fit polarity", self.odmr_polarity_combo)
        
        self.odmr_ranges = QLineEdit("")
        self.odmr_ranges.setToolTip("Optional fit-center ranges, comma-separated. Example: 2.18-2.19,2.205-2.21")
        odmr_layout.addRow("ODMR peak fit ranges", self.odmr_ranges)
        
        self.odmr_peak_pick_only = QCheckBox("ODMR multi-peak: peak-pick only")
        self.odmr_peak_pick_only.setChecked(False)
        self.odmr_peak_pick_only.setToolTip("If multiple ODMR resonances are found, report peak locations instead of single-peak fit.")
        odmr_layout.addRow(self.odmr_peak_pick_only)
        
        self.odmr_multipeak_fit_chk = QCheckBox("ODMR multi-peak fit")
        self.odmr_multipeak_fit_chk.setChecked(False)
        self.odmr_multipeak_fit_chk.setToolTip("Fit multiple Lorentzian peaks to ODMR data.")
        odmr_layout.addRow(self.odmr_multipeak_fit_chk)
        
        self.odmr_peak_count_spin = NoScrollSpinBox()
        self.odmr_peak_count_spin.setRange(1, 8)
        self.odmr_peak_count_spin.setValue(2)
        self.odmr_peak_count_spin.setToolTip("Number of ODMR peaks for multi-peak fit.")
        odmr_layout.addRow("ODMR peak count", self.odmr_peak_count_spin)
        
        self.odmr_auto_peak_count_chk = QCheckBox("Auto peak count from peak-pick")
        self.odmr_auto_peak_count_chk.setChecked(True)
        odmr_layout.addRow(self.odmr_auto_peak_count_chk)
        
        self.odmr_cal_scale = NoScrollDoubleSpinBox()
        self.odmr_cal_scale.setDecimals(6)
        self.odmr_cal_scale.setRange(1e-9, 1e9)
        self.odmr_cal_scale.setValue(1.0)
        self.odmr_cal_scale.setToolTip("Axis scale to physical units (e.g., MHz per axis unit).")
        odmr_layout.addRow("ODMR axis scale", self.odmr_cal_scale)
        
        self.odmr_cal_offset = NoScrollDoubleSpinBox()
        self.odmr_cal_offset.setDecimals(6)
        self.odmr_cal_offset.setRange(-1e9, 1e9)
        self.odmr_cal_offset.setValue(0.0)
        self.odmr_cal_offset.setToolTip("Axis offset in calibrated units.")
        odmr_layout.addRow("ODMR axis offset", self.odmr_cal_offset)
        
        st_layout.addWidget(self.odmr_container)

        # Custom Model specific
        self.custom_container = QWidget()
        custom_layout = QHBoxLayout(self.custom_container)
        custom_layout.setContentsMargins(0,0,0,0)
        
        btn_custom_model = QPushButton("Load Custom JSON")
        btn_custom_model.setToolTip("Load user-defined model spec (name, expression, params, bounds, seed).")
        btn_custom_model.clicked.connect(self.on_load_custom_model)
        custom_layout.addWidget(btn_custom_model)
        
        btn_seed_custom = QPushButton("Seed from Current")
        btn_seed_custom.clicked.connect(self.on_seed_custom_from_current_model)
        custom_layout.addWidget(btn_seed_custom)
        
        st_layout.addWidget(self.custom_container)

        # Equation Preview (Latex)
        self.eq_preview_fig = Figure(figsize=(4.0, 0.5))
        self.eq_preview_fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        self.eq_preview_canvas = FigureCanvas(self.eq_preview_fig)
        self.eq_preview_canvas.setMaximumHeight(50)
        self.eq_preview_ax = self.eq_preview_fig.add_subplot(1, 1, 1)
        self.eq_preview_ax.axis("off")
        st_layout.addWidget(self.eq_preview_canvas)

        # Equation Editor (Unified)
        st_layout.addWidget(QLabel("Model Expression (Edit to Custom):"))
        self.equation_display = QTextEdit()
        self.equation_display.setMaximumHeight(80)
        self.equation_display.setToolTip("Equation source code. Edit to switch to CustomModel.")
        self.equation_display.setAcceptRichText(False)
        self.equation_display.textChanged.connect(self._on_equation_edited)
        st_layout.addWidget(self.equation_display)

        self.fit_smoothed_chk = QCheckBox("Fit smoothed data")
        self.fit_smoothed_chk.setToolTip("If checked, fit the smoothed signal rather than raw points.")
        st_layout.addWidget(self.fit_smoothed_chk)

        self.quality_gate_chk = QCheckBox("Use profile quality gate")
        self.quality_gate_chk.setChecked(True)
        self.quality_gate_chk.setToolTip("Classify PASS/WARN/FAIL using experiment-specific quality thresholds. This does not change the fit itself.")
        st_layout.addWidget(self.quality_gate_chk)
        self.quality_gate_help_lbl = QLabel("Fit = run once on the current trace. Robust Fit = compare the current fit against trimmed-data candidates to find a more stable solution when early or late points distort the result.")
        self.quality_gate_help_lbl.setWordWrap(True)
        self.quality_gate_help_lbl.setStyleSheet("QLabel { color: #aab6c8; }")
        self.quality_gate_help_lbl.setProperty("drawerHidden", True)
        self.quality_gate_help_lbl.setVisible(False)
        st_layout.addWidget(self.quality_gate_help_lbl)
        
        right_inner_layout.addWidget(strategy_box)

        mask_box = QGroupBox("Exclusions")
        self.mask_box = mask_box
        mk_gl = QGridLayout(mask_box)
        self.btn_select_point = QPushButton("Select")
        self.btn_select_point.setToolTip("Switch to the live plot. Click any sample there to pin its coordinates.")
        self.btn_select_point.clicked.connect(self._activate_live_selection)
        mk_gl.addWidget(self.btn_select_point, 0, 0)
        self.btn_exclude_selected = QPushButton("Exclude selected point")
        self.btn_exclude_selected.setToolTip("Exclude the currently selected 1D sample from the next fit.")
        self.btn_exclude_selected.clicked.connect(self.on_exclude_selected_point)
        mk_gl.addWidget(self.btn_exclude_selected, 0, 1)
        self.mask_xmin = QLineEdit("")
        self.mask_xmax = QLineEdit("")
        self.mask_xmin.setPlaceholderText("x min")
        self.mask_xmax.setPlaceholderText("x max")
        self.mask_xmin.setToolTip("Lower x bound for a range to exclude from fitting.")
        self.mask_xmax.setToolTip("Upper x bound for a range to exclude from fitting.")
        mk_gl.addWidget(QLabel("Range"), 1, 0)
        range_row = QHBoxLayout()
        range_row.addWidget(self.mask_xmin)
        range_row.addWidget(self.mask_xmax)
        mk_gl.addLayout(range_row, 1, 1)
        btn_add_mask = QPushButton("Add range")
        btn_add_mask.setToolTip("Exclude all 1D samples between the x bounds above.")
        btn_add_mask.clicked.connect(self.on_add_exclusion_range)
        mk_gl.addWidget(btn_add_mask, 2, 0)
        self.btn_remove_exclusion = QPushButton("Remove selected")
        self.btn_remove_exclusion.setToolTip("Remove the highlighted exclusion entry below.")
        self.btn_remove_exclusion.clicked.connect(self.on_remove_selected_exclusion)
        mk_gl.addWidget(self.btn_remove_exclusion, 2, 1)
        btn_clear_mask = QPushButton("Clear exclusions")
        btn_clear_mask.setToolTip("Remove every point and range exclusion.")
        btn_clear_mask.clicked.connect(self.on_clear_exclusions)
        mk_gl.addWidget(btn_clear_mask, 3, 0, 1, 2)
        self.mask_mode_combo = NoScrollComboBox()
        self.mask_mode_combo.addItems(["Inspect point"])
        self.mask_mode_combo.setCurrentText("Inspect point")
        self.mask_mode_combo.currentTextChanged.connect(self._update_mask_mode)
        self.mask_mode_combo.setVisible(False)
        self.point_readout_lbl = QLabel("Point readout: none")
        self.point_readout_lbl.setWordWrap(True)
        self.point_readout_lbl.setStyleSheet("QLabel { color: #8fd3ff; }")
        self.point_readout_lbl.setToolTip("Last selected or hovered sample coordinate.")
        mk_gl.addWidget(self.point_readout_lbl, 4, 0, 1, 2)
        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(96)
        self.mask_list.setToolTip("Current excluded points and x ranges. Select one and click Remove selected.")
        mk_gl.addWidget(self.mask_list, 5, 0, 1, 2)
        left_inner_layout.addWidget(mask_box)

        analysis_box = QGroupBox("Analysis transforms")
        self.analysis_box = analysis_box
        analysis_layout = QGridLayout(analysis_box)
        self.analysis_kind_combo = NoScrollComboBox()
        self.analysis_kind_combo.addItems(["Baseline", "Detrend", "Normalize", "Derivative", "Integral", "Resample", "Expression"])
        self.analysis_kind_combo.currentTextChanged.connect(self._sync_analysis_step_fields)
        self.analysis_option_combo = NoScrollComboBox()
        self.analysis_value_edit = QLineEdit()
        self.analysis_value_edit.setPlaceholderText("Optional value")
        self.analysis_value_edit.setToolTip("Resample uses a point count; Expression uses x, y, abs, sin, cos, exp, log, sqrt, or clip.")
        add_step = QPushButton("Add transform")
        add_step.clicked.connect(self.on_add_analysis_step)
        remove_step = QPushButton("Remove selected")
        remove_step.clicked.connect(self.on_remove_analysis_step)
        clear_steps = QPushButton("Clear transforms")
        clear_steps.clicked.connect(self.on_clear_analysis_steps)
        self.analysis_list = QListWidget()
        self.analysis_list.setMaximumHeight(100)
        self.analysis_list.setToolTip("Ordered display and measurement transforms. Fitting remains on the untransformed processed trace.")
        analysis_layout.addWidget(QLabel("Transform"), 0, 0)
        analysis_layout.addWidget(self.analysis_kind_combo, 0, 1)
        analysis_layout.addWidget(QLabel("Option"), 1, 0)
        analysis_layout.addWidget(self.analysis_option_combo, 1, 1)
        analysis_layout.addWidget(self.analysis_value_edit, 2, 0, 1, 2)
        analysis_layout.addWidget(add_step, 3, 0)
        analysis_layout.addWidget(remove_step, 3, 1)
        analysis_layout.addWidget(clear_steps, 4, 0, 1, 2)
        analysis_layout.addWidget(self.analysis_list, 5, 0, 1, 2)
        left_inner_layout.addWidget(analysis_box)
        self._sync_analysis_step_fields()

        meta_box = QGroupBox("Sample Metadata")
        self.meta_box = meta_box
        md_gl = QGridLayout(meta_box)
        self.meta_sample_id = QLineEdit("")
        self.meta_condition = QLineEdit("")
        self.meta_power = QLineEdit("")
        self.meta_date = QLineEdit("")
        self.meta_notes = QLineEdit("")
        md_gl.addWidget(QLabel("Sample ID"), 0, 0)
        md_gl.addWidget(self.meta_sample_id, 0, 1)
        md_gl.addWidget(QLabel("Condition"), 1, 0)
        md_gl.addWidget(self.meta_condition, 1, 1)
        md_gl.addWidget(QLabel("Power"), 2, 0)
        md_gl.addWidget(self.meta_power, 2, 1)
        md_gl.addWidget(QLabel("Date"), 3, 0)
        md_gl.addWidget(self.meta_date, 3, 1)
        md_gl.addWidget(QLabel("Notes"), 4, 0)
        md_gl.addWidget(self.meta_notes, 4, 1)
        left_inner_layout.addWidget(meta_box)

        plot_box = QGroupBox("Plot")
        self.plot_box = plot_box
        plot_gl = QGridLayout(plot_box)
        self.show_data_chk = QCheckBox("Data")
        self.show_data_chk.setChecked(True)
        self.show_data_chk.setToolTip("Show/hide raw measured points.")
        self.show_data_chk.toggled.connect(self._refresh_plot_only)
        self.show_smoothed_chk = QCheckBox("Smoothed")
        self.show_smoothed_chk.setChecked(True)
        self.show_smoothed_chk.setToolTip("Show/hide smoothed trace overlay.")
        self.show_smoothed_chk.toggled.connect(self._refresh_plot_only)
        self.show_fit_chk = QCheckBox("Fit")
        self.show_fit_chk.setChecked(True)
        self.show_fit_chk.setToolTip("Show/hide model fit curve.")
        self.show_fit_chk.toggled.connect(self._refresh_plot_only)
        self.show_rabi_envelope_chk = QCheckBox("Rabi envelope")
        self.show_rabi_envelope_chk.setChecked(True)
        self.show_rabi_envelope_chk.setToolTip("Show/hide dashed upper and lower Rabi envelope bounds when available.")
        self.show_rabi_envelope_chk.toggled.connect(self._refresh_plot_only)
        self.show_peaks_chk = QCheckBox("ODMR peaks")
        self.show_peaks_chk.setChecked(True)
        self.show_peaks_chk.setToolTip("Show/hide ODMR detected peak markers.")
        self.show_peaks_chk.toggled.connect(self._refresh_plot_only)
        self.show_residual_chk = QCheckBox("Residual")
        self.show_residual_chk.setChecked(True)
        self.show_residual_chk.setToolTip("Show/hide residual subplot.")
        self.show_residual_chk.toggled.connect(self._refresh_plot_only)
        self.show_fft_panel_chk = QCheckBox("Persistent FFT")
        self.show_fft_panel_chk.setChecked(False)
        self.show_fft_panel_chk.setToolTip("Show FFT panel below the main trace.")
        self.show_fft_panel_chk.toggled.connect(self._refresh_plot_only)
        self.show_legend_chk = QCheckBox("Legend")
        self.show_legend_chk.setChecked(True)
        self.show_legend_chk.setToolTip("Show/hide legend on the main plot.")
        self.show_legend_chk.toggled.connect(self._refresh_plot_only)
        plot_gl.addWidget(self.show_data_chk, 0, 0)
        plot_gl.addWidget(self.show_smoothed_chk, 0, 1)
        plot_gl.addWidget(self.show_fit_chk, 1, 0)
        plot_gl.addWidget(self.show_rabi_envelope_chk, 1, 1)
        self.plot_style_combo = NoScrollComboBox()
        self.plot_style_combo.addItems(["Scatter", "Line", "Line + scatter"])
        self.plot_style_combo.setCurrentText("Line + scatter")
        self.plot_style_combo.setToolTip("Choose how 1D traces are drawn on the plot.")
        self.plot_style_combo.currentTextChanged.connect(self._refresh_plot_only)
        plot_gl.addWidget(self.show_peaks_chk, 2, 0)
        plot_gl.addWidget(self.show_residual_chk, 2, 1)
        plot_gl.addWidget(self.show_fft_panel_chk, 3, 0)
        plot_gl.addWidget(QLabel("Plot style"), 3, 1)
        plot_gl.addWidget(self.plot_style_combo, 4, 0, 1, 2)
        plot_gl.addWidget(self.show_legend_chk, 5, 0)
        self.show_confidence_chk = QCheckBox("Confidence band")
        self.show_confidence_chk.setChecked(False)
        self.show_confidence_chk.setToolTip("Show ±1σ confidence band around the fit curve.")
        self.show_confidence_chk.toggled.connect(self._refresh_plot_only)
        plot_gl.addWidget(self.show_confidence_chk, 5, 1)
        self.legend_compact_chk = QCheckBox("Compact legend labels")
        self.legend_compact_chk.setChecked(True)
        self.legend_compact_chk.setToolTip("Use shorter legend labels on the live plot.")
        self.legend_compact_chk.toggled.connect(self._refresh_plot_only)
        plot_gl.addWidget(self.legend_compact_chk, 6, 0, 1, 2)

        self.show_signal_layer_chk = QCheckBox("Signal layer")
        self.show_signal_layer_chk.setChecked(False)
        self.show_signal_layer_chk.setToolTip("Overlay signal counts without changing the active fit observable.")
        self.show_signal_layer_chk.toggled.connect(self._refresh_plot_only)
        self.show_reference_layer_chk = QCheckBox("Reference layer")
        self.show_reference_layer_chk.setChecked(False)
        self.show_reference_layer_chk.setToolTip("Overlay reference counts without changing the active fit observable.")
        self.show_reference_layer_chk.toggled.connect(self._refresh_plot_only)
        self.show_difference_layer_chk = QCheckBox("Difference layer")
        self.show_difference_layer_chk.setChecked(False)
        self.show_difference_layer_chk.setToolTip("Overlay reference-signal without changing the active fit observable.")
        self.show_difference_layer_chk.toggled.connect(self._refresh_plot_only)
        self.show_iteration_layer_chk = QCheckBox("Iteration")
        self.show_iteration_layer_chk.setChecked(False)
        self.show_iteration_layer_chk.setToolTip("Overlay selected individual iteration traces when available.")
        self.show_iteration_layer_chk.toggled.connect(self._refresh_plot_only)
        self.show_iteration_mean_chk = QCheckBox("Mean")
        self.show_iteration_mean_chk.setChecked(True)
        self.show_iteration_mean_chk.setToolTip("Overlay the mean of the selected complete iterations.")
        self.show_iteration_mean_chk.toggled.connect(self._refresh_plot_only)
        self.show_iteration_std_chk = QCheckBox("Std")
        self.show_iteration_std_chk.setChecked(False)
        self.show_iteration_std_chk.setToolTip("Overlay selected-iteration standard deviation.")
        self.show_iteration_std_chk.toggled.connect(self._refresh_plot_only)
        plot_gl.addWidget(self.show_signal_layer_chk, 7, 0)
        plot_gl.addWidget(self.show_reference_layer_chk, 7, 1)
        plot_gl.addWidget(self.show_difference_layer_chk, 8, 0)
        plot_gl.addWidget(self.show_iteration_layer_chk, 8, 1)
        plot_gl.addWidget(self.show_iteration_mean_chk, 9, 0)
        plot_gl.addWidget(self.show_iteration_std_chk, 9, 1)

        self.iteration_select_btn = QPushButton("Iterations...")
        self.iteration_select_btn.setToolTip("Choose multiple complete iterations to plot.")
        self.iteration_select_btn.clicked.connect(self._open_iteration_dialog)
        self.iteration_summary_lbl = QLabel("Mean only")
        self.iteration_summary_lbl.setObjectName("MutedLabel")
        self.iteration_summary_lbl.setToolTip("Shows selected iteration count.")
        self.hide_failed_iterations_chk = QCheckBox("Hide failed")
        self.hide_failed_iterations_chk.setChecked(True)
        self.hide_failed_iterations_chk.setToolTip("Ignore failed iteration cells in per-point inspection.")
        self.hide_failed_iterations_chk.toggled.connect(self._refresh_plot_only)
        self.iteration_std_mode_combo = NoScrollComboBox()
        self.iteration_std_mode_combo.addItems(["Band", "Error bars"])
        self.iteration_std_mode_combo.setToolTip("Choose how standard deviation is displayed.")
        self.iteration_std_mode_combo.currentTextChanged.connect(self._refresh_plot_only)
        plot_gl.addWidget(QLabel("Iteration"), 10, 0)
        iter_row = QHBoxLayout()
        iter_row.addWidget(self.iteration_select_btn)
        iter_row.addWidget(self.iteration_summary_lbl)
        iter_row.addWidget(self.hide_failed_iterations_chk)
        plot_gl.addLayout(iter_row, 10, 1)
        plot_gl.addWidget(QLabel("Std mode"), 11, 0)
        plot_gl.addWidget(self.iteration_std_mode_combo, 11, 1)

        self.legend_loc_combo = NoScrollComboBox()
        self.legend_loc_combo.addItems(["best", "upper right", "upper left", "lower right", "lower left"])
        self.legend_loc_combo.setToolTip("Legend anchor location on plot.")
        self.legend_loc_combo.currentTextChanged.connect(self._refresh_plot_only)
        plot_gl.addWidget(QLabel("Legend location"), 12, 0)
        plot_gl.addWidget(self.legend_loc_combo, 12, 1)

        self.legend_font_spin = NoScrollSpinBox()
        self.legend_font_spin.setRange(6, 20)
        self.legend_font_spin.setValue(9)
        self.legend_font_spin.setToolTip("Legend text size.")
        self.legend_font_spin.valueChanged.connect(self._refresh_plot_only)
        plot_gl.addWidget(QLabel("Legend font"), 13, 0)
        plot_gl.addWidget(self.legend_font_spin, 13, 1)

        self.fig_w_spin = NoScrollDoubleSpinBox()
        self.fig_w_spin.setRange(4.0, 30.0)
        self.fig_w_spin.setDecimals(1)
        self.fig_w_spin.setValue(10.0)
        self.fig_w_spin.setToolTip("Live figure width in inches.")
        self.fig_w_spin.valueChanged.connect(self._apply_figure_size)
        self.fig_h_spin = NoScrollDoubleSpinBox()
        self.fig_h_spin.setRange(3.0, 20.0)
        self.fig_h_spin.setDecimals(1)
        self.fig_h_spin.setValue(6.0)
        self.fig_h_spin.setToolTip("Live figure height in inches.")
        self.fig_h_spin.valueChanged.connect(self._apply_figure_size)
        self.save_dpi_spin = NoScrollSpinBox()
        self.save_dpi_spin.setRange(72, 600)
        self.save_dpi_spin.setValue(220)
        self.save_dpi_spin.setToolTip("Export resolution (dots per inch).")
        plot_gl.addWidget(QLabel("Figure W/H"), 14, 0)
        wh_row = QHBoxLayout()
        wh_row.addWidget(self.fig_w_spin)
        wh_row.addWidget(self.fig_h_spin)
        plot_gl.addLayout(wh_row, 14, 1)
        plot_gl.addWidget(QLabel("Save DPI"), 15, 0)
        plot_gl.addWidget(self.save_dpi_spin, 15, 1)
        right_inner_layout.addWidget(plot_box)

        annotation_box = QGroupBox("Annotations")
        self.annotation_box = annotation_box
        ann_gl = QGridLayout(annotation_box)
        ann_help = QLabel("Control the on-plot info box separately from the legend. Compact mode is intended for routine fitting; detailed mode is for screenshots and presentation exports.")
        ann_help.setWordWrap(True)
        ann_help.setStyleSheet("QLabel { color: #aab6c8; }")
        ann_help.setProperty("drawerHidden", True)
        ann_help.setVisible(False)
        ann_gl.addWidget(ann_help, 0, 0, 1, 2)
        self.annotation_mode_combo = NoScrollComboBox()
        self.annotation_mode_combo.addItems(["Off", "Compact", "Detailed"])
        self.annotation_mode_combo.setCurrentText("Compact")
        self.annotation_mode_combo.currentTextChanged.connect(self._refresh_plot_only)
        ann_gl.addWidget(QLabel("Annotation mode"), 1, 0)
        ann_gl.addWidget(self.annotation_mode_combo, 1, 1)
        self.annotation_loc_combo = NoScrollComboBox()
        self.annotation_loc_combo.addItems(["upper right", "upper left", "lower right", "lower left"])
        self.annotation_loc_combo.setCurrentText("upper right")
        self.annotation_loc_combo.currentTextChanged.connect(self._refresh_plot_only)
        ann_gl.addWidget(QLabel("Annotation corner"), 2, 0)
        ann_gl.addWidget(self.annotation_loc_combo, 2, 1)
        self.annotation_font_spin = NoScrollSpinBox()
        self.annotation_font_spin.setRange(6, 18)
        self.annotation_font_spin.setValue(9)
        self.annotation_font_spin.valueChanged.connect(self._refresh_plot_only)
        ann_gl.addWidget(QLabel("Annotation font"), 3, 0)
        ann_gl.addWidget(self.annotation_font_spin, 3, 1)
        self.ann_file_model_chk = QCheckBox("File + model")
        self.ann_file_model_chk.setChecked(True)
        self.ann_file_model_chk.toggled.connect(self._refresh_plot_only)
        ann_gl.addWidget(self.ann_file_model_chk, 4, 0)
        self.ann_metrics_chk = QCheckBox("Fit metrics")
        self.ann_metrics_chk.setChecked(True)
        self.ann_metrics_chk.toggled.connect(self._refresh_plot_only)
        ann_gl.addWidget(self.ann_metrics_chk, 4, 1)
        self.ann_t2rho_chk = QCheckBox("T2rho")
        self.ann_t2rho_chk.setChecked(True)
        self.ann_t2rho_chk.setToolTip("Show/hide the Rabi envelope T2rho annotation separately from other NV metrics.")
        self.ann_t2rho_chk.toggled.connect(self._refresh_plot_only)
        ann_gl.addWidget(self.ann_t2rho_chk, 5, 0)
        self.ann_nv_metrics_chk = QCheckBox("NV metrics")
        self.ann_nv_metrics_chk.setChecked(True)
        self.ann_nv_metrics_chk.toggled.connect(self._refresh_plot_only)
        ann_gl.addWidget(self.ann_nv_metrics_chk, 5, 1)
        self.ann_params_chk = QCheckBox("Key parameters")
        self.ann_params_chk.setChecked(False)
        self.ann_params_chk.toggled.connect(self._refresh_plot_only)
        ann_gl.addWidget(self.ann_params_chk, 6, 0, 1, 2)
        right_inner_layout.addWidget(annotation_box)

        self.scan_tools_box = QGroupBox("Scan")
        scan_gl = QGridLayout(self.scan_tools_box)
        self.scan_help_lbl = QLabel("Cursor shows the current inspected scan location. Linecut view extracts a 1D profile through a 2D map; choose None to view only the map. It is hidden for 1D scans because the loaded trace is already the linecut.")
        self.scan_help_lbl.setWordWrap(True)
        self.scan_help_lbl.setStyleSheet("QLabel { color: #aab6c8; }")
        self.scan_help_lbl.setProperty("drawerHidden", True)
        self.scan_help_lbl.setVisible(False)
        scan_gl.addWidget(self.scan_help_lbl, 0, 0, 1, 2)
        self.scan_cursor_lbl = QLabel("Cursor: none")
        self.scan_cursor_lbl.setWordWrap(True)
        self.scan_cursor_lbl.setStyleSheet("QLabel { color: #8fd3ff; }")
        scan_gl.addWidget(self.scan_cursor_lbl, 1, 0, 1, 2)
        self.scan_linecut_combo = NoScrollComboBox()
        self.scan_linecut_combo.addItems(["None", "Cursor horizontal", "Cursor vertical", "Best-point horizontal", "Best-point vertical"])
        self.scan_linecut_combo.currentTextChanged.connect(self._refresh_plot_only)
        self.scan_linecut_lbl = QLabel("Linecut view")
        scan_gl.addWidget(self.scan_linecut_lbl, 2, 0)
        scan_gl.addWidget(self.scan_linecut_combo, 2, 1)
        self.scan_secondary_combo = NoScrollComboBox()
        self.scan_secondary_combo.addItems(["None", "Contrast", "Signal", "Reference"])
        self.scan_secondary_combo.currentTextChanged.connect(self._refresh_plot_only)
        self.scan_secondary_lbl = QLabel("Secondary y-axis")
        scan_gl.addWidget(self.scan_secondary_lbl, 3, 0)
        scan_gl.addWidget(self.scan_secondary_combo, 3, 1)
        self.scan_summary_text = QTextEdit()
        self.scan_summary_text.setReadOnly(True)
        self.scan_summary_text.setMaximumHeight(120)
        scan_gl.addWidget(self.scan_summary_text, 4, 0, 1, 2)
        right_inner_layout.addWidget(self.scan_tools_box)

        action_box = QGroupBox("Actions")
        self.action_box = action_box
        act_gl = QGridLayout(action_box)
        btn_fft = QPushButton("Show FFT")
        btn_fft.setToolTip("Estimate dominant oscillation frequency from current ROI.")
        btn_fft.clicked.connect(self.on_fft)
        act_gl.addWidget(btn_fft, 0, 0)
        btn_head = QPushButton("Fit Head 20%")
        btn_head.setToolTip("Set ROI to first 20% of current x range for initial parameter finding.")
        btn_head.clicked.connect(self._crop_to_head)
        act_gl.addWidget(btn_head, 0, 1)
        btn_full = QPushButton("ROI Full")
        btn_full.clicked.connect(self._reset_roi_full)
        act_gl.addWidget(btn_full, 1, 0)
        btn_reset = QPushButton("Reset Profile Defaults")
        btn_reset.clicked.connect(self.apply_profile_defaults)
        act_gl.addWidget(btn_reset, 1, 1)
        right_inner_layout.addWidget(action_box)

        param_box = QGroupBox("Advanced")
        self.param_box = param_box
        param_layout = QVBoxLayout(param_box)
        self.param_table = QTableWidget(0, 5)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Lock"])
        self.param_table.setToolTip("Edit parameter values and lock selected terms.")
        self.param_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.param_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.param_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.param_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.param_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.param_table.horizontalHeader().setMinimumSectionSize(50)
        self.param_table.verticalHeader().setDefaultSectionSize(24)
        self.param_table.setMinimumHeight(220)
        param_layout.addWidget(self.param_table)
        right_inner_layout.addWidget(param_box)

        export_box = QGroupBox("Export")
        self.export_box = export_box
        self.export_box = export_box
        ex_gl = QGridLayout(export_box)
        self.btn_fit = QPushButton("Run Fit")
        self.btn_fit.setObjectName("PrimaryAction")
        self.btn_fit.clicked.connect(self.on_fit)
        ex_gl.addWidget(self.btn_fit, 0, 0)
        
        self.btn_robust_fit = QPushButton("Robust Fit")
        self.btn_robust_fit.setToolTip("Run multiple fits with varying data ranges to find the most stable result.")
        self.btn_robust_fit.clicked.connect(self.on_robust_fit)
        ex_gl.addWidget(self.btn_robust_fit, 0, 1)
        self.fit_help_lbl = QLabel("Fit runs the current model once on the current trace. Robust Fit compares that result to trimmed-data candidates and keeps the most stable overall solution.")
        self.fit_help_lbl.setWordWrap(True)
        self.fit_help_lbl.setStyleSheet("QLabel { color: #aab6c8; }")
        self.fit_help_lbl.setProperty("drawerHidden", True)
        self.fit_help_lbl.setVisible(False)
        ex_gl.addWidget(self.fit_help_lbl, 0, 2, 1, 2)

        btn_save = QPushButton("Save Results")
        btn_save.setObjectName("PrimaryAction")
        btn_save.setToolTip("Save plot, residual plot, JSON and CSV report with extracted NV metrics.")
        btn_save.clicked.connect(self.on_save_results)
        ex_gl.addWidget(btn_save, 1, 2)
        self.exp_png_chk = QCheckBox("PNG")
        self.exp_png_chk.setChecked(True)
        self.exp_pdf_chk = QCheckBox("PDF")
        self.exp_svg_chk = QCheckBox("SVG")
        self.exp_report_png_chk = QCheckBox("Report PNG")
        self.exp_report_pdf_chk = QCheckBox("Report PDF")
        self.exp_json_chk = QCheckBox("JSON")
        self.exp_json_chk.setChecked(True)
        self.exp_csv_chk = QCheckBox("CSV")
        self.exp_csv_chk.setChecked(True)
        self.exp_origin_chk = QCheckBox("Origin bundle")
        self.exp_origin_chk.setChecked(False)
        ex_gl.addWidget(self.exp_png_chk, 2, 0)
        ex_gl.addWidget(self.exp_pdf_chk, 2, 1)
        ex_gl.addWidget(self.exp_svg_chk, 3, 0)
        ex_gl.addWidget(self.exp_report_png_chk, 3, 1)
        ex_gl.addWidget(self.exp_report_pdf_chk, 4, 0)
        ex_gl.addWidget(self.exp_json_chk, 4, 1)
        ex_gl.addWidget(self.exp_csv_chk, 5, 0)
        ex_gl.addWidget(self.exp_origin_chk, 5, 1)
        self.report_content_lbl = QLabel("Report content")
        ex_gl.addWidget(self.report_content_lbl, 6, 0, 1, 2)
        self.report_metric_checks: dict[str, QCheckBox] = {}
        self.fit_export_widgets = [self.btn_fit, self.btn_robust_fit, self.fit_help_lbl, self.exp_report_png_chk, self.exp_report_pdf_chk, self.report_content_lbl]
        report_metric_defs = [
            ("r2", "R²"),
            ("rmse", "RMSE"),
            ("T2_star_ns", "T2*"),
            ("T1_ns", "T1"),
            ("tau", "tau"),
            ("pi_time_ns", "pi"),
            ("pi_over_2_time_ns", "pi/2"),
            ("pi_time_ns_nominal", "pi nominal"),
            ("pi_over_2_time_ns_nominal", "pi/2 nominal"),
            ("rabi_period_ns", "period"),
            ("t0_ns", "t0"),
            ("first_peak_ns", "first peak"),
            ("delay_ns", "delay"),
            ("tau_ramp_ns", "ramp tau"),
            ("T2rho_ns", "T2rho"),
            ("rabi_envelope_beta", "env beta"),
            ("rabi_envelope_r2", "env R2"),
            ("rabi_freq_MHz", "Rabi freq"),
            ("detuning_MHz", "detuning"),
            ("D", "D"),
            ("E", "E"),
            ("fwhm", "linewidth"),
        ]
        rr = 7
        cc = 0
        for key, label in report_metric_defs:
            chk = QCheckBox(label)
            chk.setChecked(key in {"r2", "rmse", "T2_star_ns", "T1_ns", "T2rho_ns", "pi_time_ns", "delay_ns", "rabi_freq_MHz", "D", "E", "fwhm"})
            self.report_metric_checks[key] = chk
            self.fit_export_widgets.append(chk)
            ex_gl.addWidget(chk, rr, cc)
            cc += 1
            if cc > 1:
                cc = 0
                rr += 1
        right_inner_layout.addWidget(export_box)

        self.metadata_status_lbl = QLabel("Metadata: no file")
        self.metadata_status_lbl.setWordWrap(True)
        self.metadata_status_lbl.setStyleSheet("QLabel { color: #c9d7e8; }")
        right_inner_layout.addWidget(self.metadata_status_lbl)
        self.metadata_progress_bar = QProgressBar()
        self.metadata_progress_bar.setRange(0, 100)
        self.metadata_progress_bar.setValue(0)
        self.metadata_progress_bar.setToolTip("Completed iterations divided by target iterations when scan metadata provides it.")
        right_inner_layout.addWidget(self.metadata_progress_bar)

        self.status_lbl = QLabel("Status: N/A")
        self.status_lbl.setMinimumHeight(24)
        self._set_status("N/A")
        right_inner_layout.addWidget(self.status_lbl)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setToolTip("Model diagnostics, parameter estimates, and quality assessment.")
        self.summary_text.setMinimumHeight(200)
        right_inner_layout.addWidget(self.summary_text, 1)
        left_inner_layout.addStretch(1)
        right_inner_layout.addStretch(1)

        self._set_group_collapsible(data_box)
        self._set_group_collapsible(overlay_box, checked=False)
        self._set_group_collapsible(prep_box, checked=False)
        self._set_group_collapsible(strategy_box)
        self._set_group_collapsible(mask_box, checked=False)
        self._set_group_collapsible(meta_box, checked=False)
        self._set_group_collapsible(plot_box, checked=False)
        self._set_group_collapsible(annotation_box, checked=False)
        self._set_group_collapsible(self.scan_tools_box, checked=False)
        self._set_group_collapsible(action_box, checked=False)
        self._set_group_collapsible(param_box, checked=False)
        self._set_group_collapsible(export_box)

        self._build_batch_workspace()
        self._build_scan_workspace()
        self._build_results_workspace()
        self._install_control_tooltips()
        self.mode_button_group.idClicked.connect(self._set_workspace_mode)
        self._set_workspace_mode(0)
        self._build_menu()
        self._update_contextual_visibility()
        self._update_mask_mode()

    def _message(self, title: str, text: str):
        QMessageBox.information(self, title, text)

    def _set_button_icon(self, button: QPushButton, icon_name: str):
        if qta is None:
            return
        try:
            button.setIcon(qta.icon(icon_name, color="#d8d8d8"))
        except Exception:
            pass

    def _make_quick_toggle(self, text: str, tooltip: str, slot) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setCheckable(True)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setToolTip(tooltip)
        btn.toggled.connect(slot)
        return btn

    def _build_plot_quick_toolbar(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(6)

        self.quick_data_btn = self._make_quick_toggle("Data", "Show measured data points.", self._on_quick_data_toggled)
        self.quick_fit_btn = self._make_quick_toggle("Fit", "Show fitted model curve.", self._on_quick_fit_toggled)
        self.quick_smooth_btn = self._make_quick_toggle("Smooth", "Show smoothed trace.", self._on_quick_smooth_toggled)
        self.quick_signal_btn = self._make_quick_toggle("Signal", "Overlay signal counts on the right axis.", self._on_quick_signal_toggled)
        self.quick_reference_btn = self._make_quick_toggle("Reference", "Overlay reference counts on the right axis.", self._on_quick_reference_toggled)
        self.quick_difference_btn = self._make_quick_toggle("Diff", "Overlay reference-signal.", self._on_quick_difference_toggled)
        self.quick_iterations_btn = self._make_quick_toggle("Iterations", "Show selected individual iterations.", self._on_quick_iterations_toggled)
        self.quick_mean_btn = self._make_quick_toggle("Mean", "Show mean of selected iterations.", self._on_quick_mean_toggled)
        self.quick_std_btn = self._make_quick_toggle("Std", "Show standard deviation of selected iterations.", self._on_quick_std_toggled)

        for btn in (
            self.quick_data_btn,
            self.quick_fit_btn,
            self.quick_smooth_btn,
            self.quick_signal_btn,
            self.quick_reference_btn,
            self.quick_difference_btn,
            self.quick_iterations_btn,
            self.quick_mean_btn,
            self.quick_std_btn,
        ):
            row.addWidget(btn)

        self.quick_iteration_select_btn = QPushButton("Iterations...")
        self.quick_iteration_select_btn.setToolTip("Choose multiple complete iterations to plot.")
        self.quick_iteration_select_btn.clicked.connect(self._open_iteration_dialog)
        row.addWidget(self.quick_iteration_select_btn)

        self.quick_std_mode_combo = NoScrollComboBox()
        self.quick_std_mode_combo.addItems(["Band", "Error bars"])
        self.quick_std_mode_combo.setToolTip("Choose how standard deviation is displayed.")
        self.quick_std_mode_combo.currentTextChanged.connect(self._on_quick_std_mode_changed)
        row.addWidget(self.quick_std_mode_combo)
        self.copy_figure_btn = QPushButton("Copy figure")
        self.copy_figure_btn.setToolTip("Copy the current export figure to the clipboard (Ctrl+Shift+C).")
        self.copy_figure_btn.clicked.connect(self.copy_export_figure)
        row.addWidget(self.copy_figure_btn)
        row.addStretch(1)
        layout.addLayout(row)

    def _labeled_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("PaneTitle")
        return label

    def _set_workspace_mode(self, idx: int):
        idx = max(0, min(idx, self.mode_stack.count() - 1))
        self.mode_stack.setCurrentIndex(idx)
        btn = self.mode_button_group.button(idx)
        if btn is not None:
            btn.setChecked(True)
        if idx == 1:
            self._load_batch_summary_from_dir()
        elif idx == 2:
            self._sync_scan_workspace()
        elif idx == 3:
            self._sync_results_workspace()

    def _build_batch_workspace(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(self._labeled_title("Batch Analysis"))

        folders = QGroupBox("Folders")
        folders_layout = QGridLayout(folders)
        self.batch_input_dir_edit = QLineEdit()
        self.batch_output_dir_edit = QLineEdit()
        btn_input = QPushButton("Browse")
        btn_output = QPushButton("Browse")
        btn_input.clicked.connect(lambda: self._browse_batch_dir(self.batch_input_dir_edit))
        btn_output.clicked.connect(lambda: self._browse_batch_dir(self.batch_output_dir_edit))
        folders_layout.addWidget(QLabel("Input"), 0, 0)
        folders_layout.addWidget(self.batch_input_dir_edit, 0, 1)
        folders_layout.addWidget(btn_input, 0, 2)
        folders_layout.addWidget(QLabel("Output"), 1, 0)
        folders_layout.addWidget(self.batch_output_dir_edit, 1, 1)
        folders_layout.addWidget(btn_output, 1, 2)
        layout.addWidget(folders)

        controls = QGroupBox("Run Settings")
        controls_layout = QGridLayout(controls)
        self.batch_mode_combo = NoScrollComboBox()
        self.batch_mode_combo.addItems(["contrast", "signal", "reference", "difference"])
        self.batch_model_combo = NoScrollComboBox()
        self.batch_model_combo.addItems(["Auto-detect", "Rabi", "SpinEcho", "DynamicDecoupling", "ODMR", "Ramsey", "T1", "DEERFrequency", "DEERDuration", "DEERDecay", "DEERPosition"])
        self.batch_bin_spin = NoScrollSpinBox()
        self.batch_bin_spin.setRange(1, 500)
        self.batch_bin_spin.setValue(1)
        self.batch_smooth_spin = NoScrollSpinBox()
        self.batch_smooth_spin.setRange(1, 201)
        self.batch_smooth_spin.setValue(1)
        self.batch_multistart_spin = NoScrollSpinBox()
        self.batch_multistart_spin.setRange(1, 120)
        self.batch_multistart_spin.setValue(20)
        self.batch_robust_chk = QCheckBox("Robust")
        self.batch_odmr_multi_chk = QCheckBox("ODMR multi-peak")
        self.batch_recursive_chk = QCheckBox("Recursive")
        self.batch_recursive_chk.setChecked(False)
        self.batch_recursive_chk.setToolTip("Process MAT files in subfolders and use collision-safe output names.")
        self.batch_skip_checkpoints_chk = QCheckBox("Skip checkpoints")
        self.batch_skip_checkpoints_chk.setChecked(False)
        self.batch_skip_checkpoints_chk.setToolTip("Skip files whose name contains __checkpoint.")
        self.batch_dpi_spin = NoScrollSpinBox()
        self.batch_dpi_spin.setRange(72, 600)
        self.batch_dpi_spin.setValue(180)
        self.batch_run_btn = QPushButton("Run Batch")
        self.batch_run_btn.setObjectName("PrimaryAction")
        self._set_button_icon(self.batch_run_btn, "fa5s.play")
        self.batch_run_btn.clicked.connect(self._on_batch_run)
        controls_layout.addWidget(QLabel("Mode"), 0, 0)
        controls_layout.addWidget(self.batch_mode_combo, 0, 1)
        controls_layout.addWidget(QLabel("Model"), 0, 2)
        controls_layout.addWidget(self.batch_model_combo, 0, 3)
        controls_layout.addWidget(QLabel("Bin"), 1, 0)
        controls_layout.addWidget(self.batch_bin_spin, 1, 1)
        controls_layout.addWidget(QLabel("Smooth"), 1, 2)
        controls_layout.addWidget(self.batch_smooth_spin, 1, 3)
        controls_layout.addWidget(QLabel("Multistart"), 2, 0)
        controls_layout.addWidget(self.batch_multistart_spin, 2, 1)
        controls_layout.addWidget(QLabel("DPI"), 2, 2)
        controls_layout.addWidget(self.batch_dpi_spin, 2, 3)
        controls_layout.addWidget(self.batch_robust_chk, 3, 0)
        controls_layout.addWidget(self.batch_odmr_multi_chk, 3, 1)
        controls_layout.addWidget(self.batch_recursive_chk, 3, 2)
        controls_layout.addWidget(self.batch_skip_checkpoints_chk, 4, 0)
        controls_layout.addWidget(self.batch_run_btn, 4, 3)
        layout.addWidget(controls)

        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        layout.addWidget(self.batch_progress_bar)

        results_row = QHBoxLayout()
        self.batch_filter_combo = NoScrollComboBox()
        self.batch_filter_combo.addItems(["All", "PASS", "WARN", "FAIL"])
        self.batch_filter_combo.currentTextChanged.connect(self._apply_batch_filter)
        self.batch_search_edit = QLineEdit()
        self.batch_search_edit.setPlaceholderText("Search file, model, metric, or path")
        self.batch_search_edit.textChanged.connect(self._apply_batch_filter)
        btn_load_summary = QPushButton("Load Summary")
        self._set_button_icon(btn_load_summary, "fa5s.folder-open")
        btn_load_summary.clicked.connect(self._choose_batch_summary)
        results_row.addWidget(QLabel("Filter"))
        results_row.addWidget(self.batch_filter_combo)
        results_row.addWidget(self.batch_search_edit, 2)
        results_row.addStretch(1)
        results_row.addWidget(btn_load_summary)
        layout.addLayout(results_row)

        self.batch_results_table = QTableWidget(0, 7)
        self.batch_results_table.setHorizontalHeaderLabels(["File", "Status", "Model", "R2", "RMSE", "Metric", "Path"])
        self.batch_results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.batch_results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.batch_results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.batch_results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.batch_results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.batch_results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.batch_results_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.batch_results_table.itemSelectionChanged.connect(self._update_batch_preview)
        self.batch_results_table.itemDoubleClicked.connect(self._open_selected_batch_result)
        layout.addWidget(self.batch_results_table, 1)

        lower = QSplitter(Qt.Horizontal)
        self.batch_log_text = QTextEdit()
        self.batch_log_text.setReadOnly(True)
        self.batch_log_text.setMaximumHeight(150)
        self.batch_preview_text = QTextEdit()
        self.batch_preview_text.setReadOnly(True)
        self.batch_preview_text.setMaximumHeight(150)
        lower.addWidget(self.batch_log_text)
        lower.addWidget(self.batch_preview_text)
        layout.addWidget(lower)

        self._batch_rows: list[dict[str, str]] = []
        self._batch_worker: BatchFitWorker | None = None
        self.mode_stack.addWidget(page)

    def _build_scan_workspace(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(self._labeled_title("Scan Explorer"))
        controls = QHBoxLayout()
        self.scan_workspace_linecut_combo = NoScrollComboBox()
        self.scan_workspace_linecut_combo.addItems(["None", "Cursor horizontal", "Cursor vertical", "Best-point horizontal", "Best-point vertical"])
        self.scan_workspace_linecut_combo.currentTextChanged.connect(self._on_scan_workspace_linecut_changed)
        self.scan_workspace_secondary_combo = NoScrollComboBox()
        self.scan_workspace_secondary_combo.addItems(["None", "Contrast", "Signal", "Reference"])
        self.scan_workspace_secondary_combo.currentTextChanged.connect(self._on_scan_workspace_secondary_changed)
        self.scan_colormap_combo = NoScrollComboBox()
        self.scan_colormap_combo.addItems(["viridis", "cividis", "magma", "turbo"])
        self.scan_colormap_combo.setToolTip("Color map for 2D scan review and exported scan figures.")
        self.scan_colormap_combo.currentTextChanged.connect(self._refresh_plot_only)
        self.scan_reverse_chk = QCheckBox("Reverse")
        self.scan_reverse_chk.toggled.connect(self._refresh_plot_only)
        self.scan_robust_limits_chk = QCheckBox("Robust range")
        self.scan_robust_limits_chk.setChecked(True)
        self.scan_robust_limits_chk.setToolTip("Use 2nd–98th percentile color limits so isolated outliers do not flatten a scan.")
        self.scan_robust_limits_chk.toggled.connect(self._refresh_plot_only)
        controls.addWidget(QLabel("Linecut"))
        controls.addWidget(self.scan_workspace_linecut_combo)
        controls.addWidget(QLabel("Secondary"))
        controls.addWidget(self.scan_workspace_secondary_combo)
        controls.addWidget(QLabel("Colors"))
        controls.addWidget(self.scan_colormap_combo)
        controls.addWidget(self.scan_reverse_chk)
        controls.addWidget(self.scan_robust_limits_chk)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.scan_workspace_plot = None
        self.scan_workspace_image = None
        if pg is not None:
            self.scan_workspace_plot = pg.PlotWidget()
            self.scan_workspace_plot.setAcceptDrops(False)
            self.scan_workspace_plot.viewport().setAcceptDrops(False)
            self.scan_workspace_plot.setLabel("bottom", "X")
            self.scan_workspace_plot.setLabel("left", "Signal")
            self.scan_workspace_plot.showGrid(x=True, y=True, alpha=0.22)
            layout.addWidget(self.scan_workspace_plot, 1)
        else:
            missing = QLabel("Install pyqtgraph to enable the scan explorer plot.")
            missing.setObjectName("MutedLabel")
            missing.setAlignment(Qt.AlignCenter)
            layout.addWidget(missing, 1)
        self.scan_workspace_summary = QTextEdit()
        self.scan_workspace_summary.setReadOnly(True)
        self.scan_workspace_summary.setMaximumHeight(150)
        layout.addWidget(self.scan_workspace_summary)
        self.mode_stack.addWidget(page)

    def _build_results_workspace(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(self._labeled_title("Results"))
        result_actions = QHBoxLayout()
        self.results_filter_edit = QLineEdit()
        self.results_filter_edit.setPlaceholderText("Filter result fields")
        self.results_filter_edit.textChanged.connect(self._sync_results_workspace)
        copy_results = QPushButton("Copy selected")
        copy_results.clicked.connect(self.copy_selected_results)
        result_actions.addWidget(self.results_filter_edit, 1)
        result_actions.addWidget(copy_results)
        layout.addLayout(result_actions)
        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.results_table)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text, 1)
        self.mode_stack.addWidget(page)

    def copy_selected_results(self) -> None:
        rows = sorted({item.row() for item in self.results_table.selectedItems()})
        if not rows:
            return
        lines = []
        for row in rows:
            key = self.results_table.item(row, 0)
            value = self.results_table.item(row, 1)
            lines.append(f"{key.text() if key else ''}\t{value.text() if value else ''}")
        QApplication.clipboard().setText("\n".join(lines))
        self.statusBar().showMessage("Selected results copied", 3000)

    def _browse_batch_dir(self, line_edit: QLineEdit):
        start = line_edit.text().strip() or str(self.settings.value("session/last_open_dir", ""))
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if folder:
            line_edit.setText(folder)
            if line_edit is self.batch_output_dir_edit:
                self._load_batch_summary_from_dir()

    def _on_batch_run(self):
        input_dir = self.batch_input_dir_edit.text().strip()
        output_dir = self.batch_output_dir_edit.text().strip()
        if not input_dir or not output_dir:
            self._message("Batch", "Select both input and output folders.")
            return
        if not Path(input_dir).exists():
            self._message("Batch", "Input folder does not exist.")
            return
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_ov = self.batch_model_combo.currentText()
        kwargs = {
            "input_dir": Path(input_dir),
            "output_dir": out,
            "mode": self.batch_mode_combo.currentText(),
            "bin_size": int(self.batch_bin_spin.value()),
            "smooth_window": int(self.batch_smooth_spin.value()),
            "multistart": int(self.batch_multistart_spin.value()),
            "model_override": model_ov if model_ov != "Auto-detect" else None,
            "robust": self.batch_robust_chk.isChecked(),
            "odmr_multipeak": self.batch_odmr_multi_chk.isChecked(),
            "dpi": int(self.batch_dpi_spin.value()),
            "recursive": self.batch_recursive_chk.isChecked(),
            "skip_checkpoints": self.batch_skip_checkpoints_chk.isChecked(),
        }
        self.batch_run_btn.setEnabled(False)
        self.batch_progress_bar.setValue(0)
        self.batch_log_text.clear()
        self.batch_preview_text.clear()
        self._batch_worker = BatchFitWorker(kwargs)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

    def _on_batch_progress(self, idx: int, total: int, name: str, msg: str):
        pct = int(100 * (idx + 1) / max(1, total))
        self.batch_progress_bar.setValue(pct)
        self.batch_log_text.append(f"[{idx + 1}/{total}] {name}: {msg}")

    def _on_batch_finished(self, out_dir: str):
        self.batch_run_btn.setEnabled(True)
        self.batch_progress_bar.setValue(100)
        self.batch_log_text.append(f"Batch complete: {out_dir}")
        self.batch_output_dir_edit.setText(out_dir)
        self._load_batch_summary_from_dir()

    def _on_batch_error(self, msg: str):
        self.batch_run_btn.setEnabled(True)
        self.batch_log_text.append(f"ERROR: {msg}")
        self._message("Batch error", msg)

    def _choose_batch_summary(self):
        start = self.batch_output_dir_edit.text().strip() or str(self.settings.value("session/last_open_dir", ""))
        path, _ = QFileDialog.getOpenFileName(self, "Open batch summary", start, "CSV files (*.csv)")
        if path:
            self.batch_output_dir_edit.setText(str(Path(path).parent))
            self._load_batch_summary(Path(path))

    def _load_batch_summary_from_dir(self):
        if not hasattr(self, "batch_output_dir_edit"):
            return
        out = Path(self.batch_output_dir_edit.text().strip())
        if not out.exists():
            return
        for name in ("batch_summary.csv", "validation_report.csv", "aggregate_report.csv"):
            candidate = out / name
            if candidate.exists():
                self._load_batch_summary(candidate)
                return

    def _load_batch_summary(self, path: Path):
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as f:
                rows = [dict(r) for r in csv.DictReader(f)]
        except Exception as exc:
            self._message("Batch summary", f"Could not read summary:\n{exc}")
            return
        self._batch_rows = rows
        self.batch_preview_text.setPlainText(f"Loaded {len(rows)} rows from {path}")
        self._apply_batch_filter()

    def _apply_batch_filter(self):
        if not hasattr(self, "batch_results_table"):
            return
        wanted = self.batch_filter_combo.currentText() if hasattr(self, "batch_filter_combo") else "All"
        rows = self._batch_rows if hasattr(self, "_batch_rows") else []
        if wanted != "All":
            rows = [r for r in rows if (r.get("status") or r.get("Status") or "").upper() == wanted]
        query = self.batch_search_edit.text().strip().lower() if hasattr(self, "batch_search_edit") else ""
        if query:
            rows = [r for r in rows if query in " ".join(str(value) for value in r.values()).lower()]
        self.batch_results_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            file_name = row.get("file") or row.get("filename") or row.get("File") or row.get("name") or ""
            status = (row.get("status") or row.get("Status") or "").upper()
            model = row.get("model") or row.get("Model") or row.get("analysis_mode") or ""
            r2 = row.get("r2") or row.get("R2") or row.get("R^2") or ""
            rmse = row.get("rmse") or row.get("RMSE") or ""
            metric = row.get("metric") or row.get("validation") or row.get("status_reason") or row.get("reason") or ""
            path = row.get("relative_path") or row.get("path") or row.get("source_path") or ""
            for col, value in enumerate([file_name, status, model, r2, rmse, metric, path]):
                item = QTableWidgetItem(str(value))
                if col == 1:
                    item.setTextAlignment(Qt.AlignCenter)
                    color = {"PASS": "#6fcf97", "WARN": "#e0a458", "FAIL": "#e57373"}.get(status)
                    if color:
                        item.setForeground(QColor(color))
                self.batch_results_table.setItem(row_idx, col, item)
        self.batch_results_table.resizeRowsToContents()

    def _update_batch_preview(self):
        row = self.batch_results_table.currentRow()
        if row < 0:
            return
        values = []
        for col in range(self.batch_results_table.columnCount()):
            header = self.batch_results_table.horizontalHeaderItem(col).text()
            item = self.batch_results_table.item(row, col)
            values.append(f"{header}: {item.text() if item else ''}")
        self.batch_preview_text.setPlainText("\n".join(values))

    def _open_selected_batch_result(self, item: QTableWidgetItem) -> None:
        row = item.row()
        path_item = self.batch_results_table.item(row, 6)
        if path_item is None:
            return
        path = Path(path_item.text())
        if not path.is_absolute() and hasattr(self, "batch_input_dir_edit"):
            path = Path(self.batch_input_dir_edit.text()) / path
        if path.exists() and self._load_file(str(path)):
            self._set_workspace_mode(0)

    def _on_scan_workspace_linecut_changed(self, text: str):
        if hasattr(self, "scan_linecut_combo"):
            self.scan_linecut_combo.setCurrentText(text)
        self._sync_scan_workspace()

    def _on_scan_workspace_secondary_changed(self, text: str):
        if hasattr(self, "scan_secondary_combo"):
            self.scan_secondary_combo.setCurrentText(text)
        self._sync_scan_workspace()

    def _sync_scan_workspace(self):
        if not hasattr(self, "scan_workspace_summary"):
            return
        if self.ctx.trace is None:
            self.scan_workspace_summary.setPlainText("No file loaded.")
            return
        if hasattr(self, "scan_workspace_linecut_combo"):
            self.scan_workspace_linecut_combo.blockSignals(True)
            self.scan_workspace_linecut_combo.setCurrentText(self.scan_linecut_combo.currentText())
            self.scan_workspace_linecut_combo.blockSignals(False)
            self.scan_workspace_secondary_combo.blockSignals(True)
            self.scan_workspace_secondary_combo.setCurrentText(self.scan_secondary_combo.currentText())
            self.scan_workspace_secondary_combo.blockSignals(False)
        self.scan_workspace_summary.setPlainText(self.scan_summary_text.toPlainText())
        if pg is None or self.scan_workspace_plot is None:
            return
        self.scan_workspace_plot.clear()
        trace = self.ctx.trace
        if trace.scan_dim == "scan2d" and trace.z2d is not None:
            image = pg.ImageItem(np.asarray(trace.z2d, dtype=float).T)
            self.scan_workspace_plot.addItem(image)
            scan_extent = self._scan2d_extent(trace)
            if scan_extent is not None:
                _x_edges, _y_edges, extent = scan_extent
                xmin, xmax, ymin, ymax = extent
                image.setRect(QRectF(float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)))
            if self._scan_cursor is not None:
                cx, cy = self._scan_cursor
                self.scan_workspace_plot.addLine(x=cx, pen=pg.mkPen("#d6a75d", width=1))
                self.scan_workspace_plot.addLine(y=cy, pen=pg.mkPen("#d6a75d", width=1))
            linecut = self._extract_scan_linecut()
            if linecut is not None:
                lx, ly, _ = linecut
                self.scan_workspace_plot.plot(lx, ly, pen=pg.mkPen("#3d8b74", width=2))
        elif self.ctx.x is not None and self.ctx.y is not None:
            self.scan_workspace_plot.plot(self.ctx.x, self.ctx.y, pen=pg.mkPen("#3d8b74", width=2), symbol="o", symbolSize=5)

    def _sync_results_workspace(self):
        if not hasattr(self, "results_text"):
            return
        self.results_text.setPlainText(self.summary_text.toPlainText() if hasattr(self, "summary_text") else "")
        rows: list[tuple[str, str]] = []
        if self.ctx.trace is not None:
            rows.extend(
                [
                    ("File", self.ctx.trace.file_name),
                    ("Experiment", self.ctx.trace.experiment_type),
                    ("Observable", self.mode_combo.currentText()),
                    ("Scan", self.ctx.trace.scan_dim),
                ]
            )
        if self.ctx.fit_result is not None:
            result = self.ctx.fit_result
            rows.extend(
                [
                    ("Model", result.model_name),
                    ("R2", f"{result.r2:.6g}"),
                    ("RMSE", f"{result.rmse:.6g}"),
                    ("Status", self.ctx.status),
                ]
            )
            nv = _nv_metrics(self._active_profile().name, result, self._effective_trace_metadata())
            rows.extend((self._metric_display_label(k), self._format_value_with_units(k, v)) for k, v in nv.items())
        query = self.results_filter_edit.text().strip().lower() if hasattr(self, "results_filter_edit") else ""
        if query:
            rows = [(key, value) for key, value in rows if query in f"{key} {value}".lower()]
        self.results_table.setRowCount(len(rows))
        for idx, (key, value) in enumerate(rows):
            self.results_table.setItem(idx, 0, QTableWidgetItem(key))
            self.results_table.setItem(idx, 1, QTableWidgetItem(value))
        self.results_table.resizeRowsToContents()

    def _reset_live_secondary_axis(self):
        if pg is None or getattr(self, "live_plot_widget", None) is None:
            return
        plot_item = self.live_plot_widget.getPlotItem()
        view = getattr(self, "_live_secondary_view", None)
        if view is not None:
            try:
                view.clear()
                plot_item.scene().removeItem(view)
            except Exception:
                pass
            self._live_secondary_view = None
        try:
            plot_item.showAxis("right", False)
        except Exception:
            pass

    def _ensure_live_secondary_axis(self, label: str):
        if pg is None or getattr(self, "live_plot_widget", None) is None:
            return None
        if self._live_secondary_view is not None:
            return self._live_secondary_view
        plot_item = self.live_plot_widget.getPlotItem()
        view = pg.ViewBox()
        self._live_secondary_view = view
        plot_item.scene().addItem(view)
        right_axis = plot_item.getAxis("right")
        right_axis.setLabel(label)
        right_axis.linkToView(view)
        plot_item.showAxis("right")
        view.setXLink(plot_item)

        def update_view():
            if self._live_secondary_view is not view:
                return
            view.setGeometry(plot_item.vb.sceneBoundingRect())
            view.linkedViewChanged(plot_item.vb, view.XAxis)

        self._live_secondary_update = update_view
        plot_item.vb.sigResized.connect(update_view)
        update_view()
        return view

    def _sync_live_plot(self, fit_result: FitResult | None = None, y_display: np.ndarray | None = None):
        if pg is None or getattr(self, "live_plot_widget", None) is None:
            return
        self.live_plot_widget.clear()
        self._reset_live_secondary_axis()
        self.live_selection_marker = None
        self.live_selection_vline = None
        self.live_selection_hline = None
        if self.ctx.trace is None:
            return
        if self.ctx.trace.scan_dim == "scan2d" and self.ctx.trace.z2d is not None:
            image = pg.ImageItem(np.asarray(self.ctx.trace.z2d, dtype=float).T)
            self.live_plot_widget.addItem(image)
            scan_extent = self._scan2d_extent(self.ctx.trace)
            if scan_extent is not None:
                _x_edges, _y_edges, extent = scan_extent
                xmin, xmax, ymin, ymax = extent
                image.setRect(QRectF(float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)))
            if self._scan_cursor is not None:
                cx, cy = self._scan_cursor
                self.live_selection_vline = pg.InfiniteLine(pos=cx, angle=90, pen=pg.mkPen("#d6a75d", width=1))
                self.live_selection_hline = pg.InfiniteLine(pos=cy, angle=0, pen=pg.mkPen("#d6a75d", width=1))
                self.live_plot_widget.addItem(self.live_selection_vline)
                self.live_plot_widget.addItem(self.live_selection_hline)
                self.live_selection_marker = pg.ScatterPlotItem(
                    [cx],
                    [cy],
                    symbol="o",
                    size=10,
                    brush=pg.mkBrush(0, 0, 0, 0),
                    pen=pg.mkPen("#d6a75d", width=2),
                )
                self.live_plot_widget.addItem(self.live_selection_marker)
            self.live_plot_widget.setLabel("bottom", self._friendly_axis_label(self.ctx.trace.scan_axes[0] if self.ctx.trace.scan_axes else "X"))
            self.live_plot_widget.setLabel("left", self._friendly_axis_label(self.ctx.trace.scan_axes[1] if len(self.ctx.trace.scan_axes) > 1 else "Y"))
            return
        if self.ctx.x is None or self.ctx.y is None:
            return
        plot_x = self.ctx.analysis_x if self.ctx.analysis_x is not None else self.ctx.x
        plot_source = y_display if y_display is not None else (self.ctx.analysis_y if self.ctx.analysis_y is not None else self.ctx.y)
        y_plot, y_label = self._display_y(plot_source)
        self.live_plot_widget.plot(plot_x, y_plot, pen=pg.mkPen("#5aa989", width=2), symbol="o", symbolSize=5, symbolBrush="#5aa989")
        if self.ctx.trace is not None and self.ctx.excluded_points:
            indices = sorted(index for index in self.ctx.excluded_points if 0 <= index < len(self.ctx.trace.x_ns))
            if indices:
                excluded_y, _ = self._display_y(self.ctx.trace.y[indices])
                self.live_plot_widget.plot(self.ctx.trace.x_ns[indices], excluded_y, pen=None, symbol="x", symbolSize=10, symbolPen="#e0a458")
        active_mode = self._friendly_mode_name(self._current_data_mode())
        live_secondary_view = None
        for mode, enabled, color in (
            ("signal", self.plot_opts.show_signal_layer, "#7aa2f7"),
            ("reference", self.plot_opts.show_reference_layer, "#f6bd60"),
            ("difference", self.plot_opts.show_difference_layer, "#9ece6a"),
        ):
            if enabled:
                y_full, _ = self._series_for_mode(self.ctx.trace, mode)
                lx, ly = self._processed_trace_series(self.ctx.trace, y_full)
                if len(lx):
                    needs_secondary = mode in {"signal", "reference"} or (mode == "difference" and active_mode not in {"signal", "reference", "difference"})
                    if needs_secondary:
                        if live_secondary_view is None:
                            live_secondary_view = self._ensure_live_secondary_axis("Counts")
                        if live_secondary_view is not None:
                            live_secondary_view.addItem(pg.PlotDataItem(lx, ly, pen=pg.mkPen(color, width=1)))
                    else:
                        self.live_plot_widget.plot(lx, ly, pen=pg.mkPen(color, width=1))
        if self.plot_opts.show_iteration_layer or self.plot_opts.show_iteration_mean or self.plot_opts.show_iteration_std:
            ix, matrix, indices, _ = self._selected_iteration_matrix(self.ctx.trace, self._friendly_mode_name(self._current_data_mode()))
            if len(ix) and matrix.size:
                if self.plot_opts.show_iteration_layer:
                    for col, _iter_idx in enumerate(indices):
                        self.live_plot_widget.plot(ix, matrix[:, col], pen=pg.mkPen("#c678dd", width=1, alpha=80))
                finite_counts = np.sum(np.isfinite(matrix), axis=1)
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(matrix, axis=1)
                    std = np.nanstd(matrix, axis=1, ddof=1 if matrix.shape[1] > 1 else 0)
                mean[finite_counts == 0] = np.nan
                std[finite_counts == 0] = np.nan
                if self.plot_opts.show_iteration_mean:
                    self.live_plot_widget.plot(ix, mean, pen=pg.mkPen("#e0af68", width=2))
                if self.plot_opts.show_iteration_std:
                    if self.plot_opts.iteration_std_mode == "Error bars":
                        try:
                            error = pg.ErrorBarItem(x=ix, y=mean, top=std, bottom=std, beam=0.0, pen=pg.mkPen("#c678dd", width=1, alpha=110))
                            self.live_plot_widget.addItem(error)
                        except Exception:
                            self.live_plot_widget.plot(ix, mean + std, pen=pg.mkPen("#c678dd", width=1, alpha=80))
                            self.live_plot_widget.plot(ix, mean - std, pen=pg.mkPen("#c678dd", width=1, alpha=80))
                    else:
                        self.live_plot_widget.plot(ix, mean + std, pen=pg.mkPen("#c678dd", width=1, alpha=80))
                        self.live_plot_widget.plot(ix, mean - std, pen=pg.mkPen("#c678dd", width=1, alpha=80))
        if (
            self.ctx.y_smooth is not None
            and self.plot_opts.show_smoothed
            and self.smooth_spin.value() > 1
            and len(self.ctx.y_smooth) == len(self.ctx.x)
        ):
            y_sm, _ = self._display_y(self.ctx.y_smooth)
            self.live_plot_widget.plot(self.ctx.x, y_sm, pen=pg.mkPen("#c0954c", width=2))
        if fit_result is not None and self.plot_opts.show_fit:
            fit_y, _ = self._display_y(fit_result.y_fit)
            self.live_plot_widget.plot(self.ctx.x, fit_y, pen=pg.mkPen("#d6d6d6", width=2))
            if self.plot_opts.show_rabi_envelope and isinstance(fit_result.extras, dict):
                bounds = rabi_envelope_bounds(self.ctx.x, fit_result.extras.get("rabi_envelope"))
                if bounds is not None:
                    upper, lower = bounds
                    upper_y, _ = self._display_y(np.asarray(upper, dtype=float))
                    lower_y, _ = self._display_y(np.asarray(lower, dtype=float))
                    env_pen = pg.mkPen("#b57edc", width=1.5, style=Qt.DashLine)
                    self.live_plot_widget.plot(self.ctx.x, upper_y, pen=env_pen)
                    self.live_plot_widget.plot(self.ctx.x, lower_y, pen=env_pen)
        if self._selected_plot_point is not None and "x" in self._selected_plot_point and "y" in self._selected_plot_point:
            sx = float(self._selected_plot_point["x"])
            sy_raw = float(self._selected_plot_point["y"])
            sy_disp, _ = self._display_y(np.asarray([sy_raw], dtype=float))
            self.live_selection_vline = pg.InfiniteLine(pos=sx, angle=90, pen=pg.mkPen("#d6a75d", width=1))
            self.live_plot_widget.addItem(self.live_selection_vline)
            self.live_selection_marker = pg.ScatterPlotItem(
                [sx],
                [float(sy_disp[0])],
                symbol="o",
                size=11,
                brush=pg.mkBrush(0, 0, 0, 0),
                pen=pg.mkPen("#d6a75d", width=2),
            )
            self.live_plot_widget.addItem(self.live_selection_marker)
        self.live_plot_widget.setLabel("bottom", self._friendly_axis_label(self.ctx.trace.x_label))
        self.live_plot_widget.setLabel("left", y_label)

    def _on_live_plot_mouse_clicked(self, event):
        if pg is None or self.live_plot_widget is None:
            return
        try:
            if event.button() != Qt.LeftButton:
                return
            point = self.live_plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        except Exception:
            return
        self._select_point_from_plot(float(point.x()), float(point.y()))
        try:
            event.accept()
        except Exception:
            pass

    def _on_live_plot_mouse_moved(self, pos):
        if pg is None or self.live_plot_widget is None:
            return
        if self.ctx.x is None or self.ctx.y is None:
            return
        try:
            point = self.live_plot_widget.plotItem.vb.mapSceneToView(pos)
        except Exception:
            return
        if self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan2d" and self.ctx.trace.z2d is not None:
            x2d = self.ctx.trace.x2d
            y2d = self.ctx.trace.y2d
            row = int(np.argmin(np.abs(y2d[:, 0] - point.y())))
            col = int(np.argmin(np.abs(x2d[0, :] - point.x())))
            z = float(self.ctx.trace.z2d[row, col])
            self._update_point_readout(f"Point readout: x={point.x():.6g}, y={point.y():.6g}, z={z:.6g}")
            return
        nearest = self._nearest_trace_point(float(point.x()))
        if nearest is not None:
            _, px, py = nearest
            self._update_point_readout(f"Point readout: x={self._format_inspect_value(px)}, y={self._format_inspect_value(py)}")

    def _activate_live_selection(self):
        if hasattr(self, "plot_tabs"):
            self.plot_tabs.setCurrentIndex(0)
        self._update_point_readout("Point readout: click a point in Live inspect")

    def _select_point_from_plot(self, xdata: float, ydata: float | None = None, *, refresh: bool = True):
        if self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan2d" and self.ctx.trace.z2d is not None and ydata is not None:
            row = int(np.argmin(np.abs(self.ctx.trace.y2d[:, 0] - ydata)))
            col = int(np.argmin(np.abs(self.ctx.trace.x2d[0, :] - xdata)))
            px = float(self.ctx.trace.x2d[row, col])
            py = float(self.ctx.trace.y2d[row, col])
            z = float(self.ctx.trace.z2d[row, col])
            if self._selected_plot_point is not None and np.isclose(float(self._selected_plot_point.get("x", np.nan)), px) and np.isclose(float(self._selected_plot_point.get("y", np.nan)), py):
                self.clear_selected_marker(refresh=refresh)
                return
            self._scan_cursor = (px, py)
            self._selected_plot_point = {"x": px, "y": py, "z": z}
            self.clear_marker_btn.setEnabled(True)
            self.scan_cursor_lbl.setText(f"Cursor: x={px:.6g}, y={py:.6g}, z={z:.6g}")
            self._update_point_readout(
                f"Point readout: x={self._format_inspect_value(px)}, "
                f"y={self._format_inspect_value(py)}, z={self._format_inspect_value(z)}"
            )
        else:
            nearest = self._nearest_trace_point(float(xdata))
            if nearest is None:
                return
            _, px, py = nearest
            if self._selected_plot_point is not None and np.isclose(float(self._selected_plot_point.get("x", np.nan)), px):
                self.clear_selected_marker(refresh=refresh)
                return
            self._selected_plot_point = {"x": px, "y": py}
            self.clear_marker_btn.setEnabled(True)
            iter_text = self._point_iteration_readout(px)
            secondary = self._nearest_secondary_point(float(xdata))
            if secondary is not None:
                _, sx, sy, sec_label = secondary
                self._selected_plot_point["secondary_x"] = sx
                self._selected_plot_point["secondary_y"] = sy
                self._selected_plot_point["secondary_label"] = sec_label
                readout = (
                    f"Point readout: x={self._format_inspect_value(px)}, "
                    f"primary={self._format_inspect_value(py)}, "
                    f"secondary={self._format_inspect_value(sy)}"
                )
                if iter_text:
                    readout += f", {iter_text}"
                self._update_point_readout(readout)
            else:
                readout = f"Point readout: x={self._format_inspect_value(px)}, y={self._format_inspect_value(py)}"
                if iter_text:
                    readout += f", {iter_text}"
                self._update_point_readout(readout)
            if self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan1d":
                self.scan_cursor_lbl.setText(f"Cursor: x={self._format_inspect_value(px)}, y={self._format_inspect_value(py)}")
        if refresh:
            self._refresh_plot_only()

    def clear_selected_marker(self, _checked: bool = False, *, refresh: bool = True):
        """Clear only the pinned inspection marker; measured data is untouched."""
        self._selected_plot_point = None
        self._scan_cursor = None
        if hasattr(self, "clear_marker_btn"):
            self.clear_marker_btn.setEnabled(False)
        self._update_point_readout("Point readout: none")
        if hasattr(self, "scan_cursor_lbl"):
            self.scan_cursor_lbl.setText("Cursor: none")
        if refresh and self.ctx.trace is not None:
            self._refresh_plot_only()


    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        save_session_action = QAction("Save Analysis Session...", self)
        save_session_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_session_action.triggered.connect(self.on_save_session)
        file_menu.addAction(save_session_action)
        load_session_action = QAction("Open Analysis Session...", self)
        load_session_action.setShortcut(QKeySequence("Ctrl+Alt+O"))
        load_session_action.triggered.connect(self.on_load_session)
        file_menu.addAction(load_session_action)
        file_menu.addSeparator()
        recent_action = QAction("Open Recent...", self)
        recent_action.triggered.connect(self.on_open_recent)
        file_menu.addAction(recent_action)
        fav_action = QAction("Open Favorite Folder...", self)
        fav_action.triggered.connect(self.on_open_favorite_folder)
        file_menu.addAction(fav_action)

        batch_action = QAction("Batch Fit…", self)
        batch_action.triggered.connect(self._open_batch_dialog)
        file_menu.addAction(batch_action)

        edit_menu = self.menuBar().addMenu("Edit")
        undo_action = QAction("Undo analysis change", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self._undo)
        edit_menu.addAction(undo_action)
        redo_action = QAction("Redo analysis change", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self._redo)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        copy_figure_action = QAction("Copy Export Figure", self)
        copy_figure_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        copy_figure_action.triggered.connect(self.copy_export_figure)
        edit_menu.addAction(copy_figure_action)


        preset_menu = self.menuBar().addMenu("Presets")
        save_preset_action = QAction("Save Analysis Preset...", self)
        save_preset_action.triggered.connect(self.on_save_preset)
        preset_menu.addAction(save_preset_action)
        load_preset_action = QAction("Load Analysis Preset...", self)
        load_preset_action.triggered.connect(self.on_load_preset)
        preset_menu.addAction(load_preset_action)
        apply_preset_action = QAction("Apply Current Preset", self)
        apply_preset_action.triggered.connect(lambda: self.apply_profile_defaults(reload_data=False))
        preset_menu.addAction(apply_preset_action)
        custom_action = QAction("Load Custom Model JSON...", self)
        custom_action.triggered.connect(self.on_load_custom_model)
        preset_menu.addAction(custom_action)

        menu = self.menuBar().addMenu("Options")
        pref_action = QAction("Preferences...", self)
        pref_action.triggered.connect(self._open_preferences)
        menu.addAction(pref_action)
        save_pref_action = QAction("Save Current Preferences", self)
        save_pref_action.triggered.connect(self._save_preferences)
        menu.addAction(save_pref_action)

    def _set_group_collapsible(self, box: QGroupBox, checked: bool = True):
        box.setCheckable(True)
        box.setChecked(checked)
        box.setToolTip(f"Click the title to {'collapse' if checked else 'expand'} this section.")
        box._drawer_anim = QPropertyAnimation(box, b"maximumHeight", box)
        box._drawer_anim.setDuration(0)
        box._drawer_anim.setEasingCurve(QEasingCurve.Linear)
        box._drawer_finish_connected = False

        def set_children_visible(state: bool):
            lay = box.layout()
            if lay is None:
                return
            for i in range(lay.count()):
                item = lay.itemAt(i)
                w = item.widget()
                if w is not None:
                    w.setVisible(False if w.property("drawerHidden") else state)
                elif item.layout() is not None:
                    item.layout().setEnabled(state)

        def toggle_children(state: bool):
            collapsed_h = 30
            box._drawer_anim.stop()
            if getattr(box, "_drawer_finish_connected", False):
                try:
                    box._drawer_anim.finished.disconnect()
                except Exception:
                    pass
                box._drawer_finish_connected = False
            if state:
                set_children_visible(True)
                target = max(box.sizeHint().height(), collapsed_h + 20)
                box._drawer_anim.setStartValue(max(collapsed_h, box.height()))
                box._drawer_anim.setEndValue(target)
                box._drawer_anim.finished.connect(lambda: box.setMaximumHeight(16777215))
                box._drawer_finish_connected = True
            else:
                box._drawer_anim.setStartValue(box.height() if box.height() > collapsed_h else box.sizeHint().height())
                box._drawer_anim.setEndValue(collapsed_h)
                box._drawer_anim.finished.connect(lambda: set_children_visible(False))
                box._drawer_finish_connected = True
            box._drawer_anim.start()
        box.toggled.connect(toggle_children)
        set_children_visible(checked)
        if not checked:
            box.setMaximumHeight(30)

    def _tip(self, widget: QWidget, text: str):
        widget.setToolTip(text)
        widget.setStatusTip(text)

    def _install_control_tooltips(self):
        tips = {
            self.btn_load: "Open a MATLAB .mat file. You can also drag one or more .mat files onto the window.",
            self.profile_combo: "Experiment family. Auto chooses from file metadata; override only if detection is wrong.",
            self.mode_combo: "Observable to plot and fit: contrast is usually safest for NV measurements.",
            self.overlay_list: "Checked traces are drawn as overlays. Double-click a trace to make it primary.",
            self.baseline_combo: "Optional trace to subtract from the active trace before fitting.",
            self.compare_metric_combo: "Metric used when building an overlay comparison.",
            self.bin_spin: "Average neighboring samples before fitting. Increase only for noisy dense traces.",
            self.smooth_spin: "Smooth the plotted trace. Value 1 leaves the data unchanged.",
            self.roi_min: "Lower x bound kept for fitting and plotting.",
            self.roi_max: "Upper x bound kept for fitting and plotting.",
            self.model_combo: "Fit model used by Run Fit. Profile defaults pick the usual model for the detected experiment.",
            self.multistart_spin: "Number of optimizer starts. Higher is slower but can avoid bad local fits.",
            self.rabi_mode_combo: "Rabi model family. Phase-ramp is the physical default.",
            self.btn_fit: "Fit the active trace using the current profile, ROI, and model settings.",
            self.btn_robust_fit: "Try trimmed data subsets and keep the most stable fit.",
            self.show_confidence_chk: "Draw an approximate +/-1 sigma fit band when parameter errors are available.",
            self.show_signal_layer_chk: "Overlay signal counts without changing the fit observable.",
            self.show_reference_layer_chk: "Overlay reference counts without changing the fit observable.",
            self.show_iteration_layer_chk: "Overlay selected complete iteration traces when iteration data are available.",
            self.annotation_mode_combo: "Controls the on-plot text box in the exported Matplotlib figure.",
            self.exp_png_chk: "Save the current Matplotlib export figure as PNG.",
            self.exp_json_chk: "Save machine-readable fit results and metadata.",
            self.exp_csv_chk: "Save a compact CSV summary of fit results and metadata.",
            self.exp_origin_chk: "Save a table formatted for Origin plotting.",
            self.scan_linecut_combo: "For 2D scans, draw a horizontal or vertical linecut through the cursor or best point.",
            self.scan_secondary_combo: "For 1D scans, overlay a second observable on the right axis.",
            self.batch_mode_combo: "Observable used for every file in a batch run.",
            self.batch_model_combo: "Force one model family for batch files, or leave on Auto-detect.",
            self.batch_run_btn: "Run batch fitting on the input folder and write outputs to the output folder.",
        }
        for widget, text in tips.items():
            self._tip(widget, text)


    def _apply_figure_size(self):
        self.plot_opts.fig_width = float(self.fig_w_spin.value())
        self.plot_opts.fig_height = float(self.fig_h_spin.value())
        self.fig.set_size_inches(self.plot_opts.fig_width, self.plot_opts.fig_height, forward=True)
        self.canvas.draw_idle()

    def _apply_plot_controls_to_state(self):
        self.plot_opts.show_data = self.show_data_chk.isChecked()
        self.plot_opts.show_smoothed = self.show_smoothed_chk.isChecked()
        self.plot_opts.show_fit = self.show_fit_chk.isChecked()
        self.plot_opts.show_rabi_envelope = self.show_rabi_envelope_chk.isChecked()
        self.plot_opts.show_peaks = self.show_peaks_chk.isChecked()
        self.plot_opts.show_signal_layer = self.show_signal_layer_chk.isChecked()
        self.plot_opts.show_reference_layer = self.show_reference_layer_chk.isChecked()
        self.plot_opts.show_difference_layer = self.show_difference_layer_chk.isChecked()
        self.plot_opts.show_iteration_layer = self.show_iteration_layer_chk.isChecked()
        self.plot_opts.show_iteration_mean = self.show_iteration_mean_chk.isChecked()
        self.plot_opts.show_iteration_std = self.show_iteration_std_chk.isChecked()
        self.plot_opts.iteration_std_mode = self.iteration_std_mode_combo.currentText()
        self.plot_opts.hide_failed_iterations = self.hide_failed_iterations_chk.isChecked()
        self.plot_opts.show_residual = self.show_residual_chk.isChecked()
        self.plot_opts.show_fft_panel = self.show_fft_panel_chk.isChecked()
        self.plot_opts.show_legend = self.show_legend_chk.isChecked()
        self.plot_opts.show_confidence = self.show_confidence_chk.isChecked()
        self.plot_opts.legend_loc = self.legend_loc_combo.currentText()
        self.plot_opts.legend_font_size = int(self.legend_font_spin.value())
        self.plot_opts.annotation_mode = self.annotation_mode_combo.currentText()
        self.plot_opts.annotation_loc = self.annotation_loc_combo.currentText()
        self.plot_opts.annotation_font_size = int(self.annotation_font_spin.value())
        self.plot_opts.annotation_show_file_model = self.ann_file_model_chk.isChecked()
        self.plot_opts.annotation_show_fit_metrics = self.ann_metrics_chk.isChecked()
        self.plot_opts.annotation_show_t2rho = self.ann_t2rho_chk.isChecked()
        self.plot_opts.annotation_show_nv_metrics = self.ann_nv_metrics_chk.isChecked()
        self.plot_opts.annotation_show_params = self.ann_params_chk.isChecked()
        self.plot_opts.legend_compact_labels = self.legend_compact_chk.isChecked()
        self.plot_opts.plot_style = self.plot_style_combo.currentText()
        self.plot_opts.fig_width = float(self.fig_w_spin.value())
        self.plot_opts.fig_height = float(self.fig_h_spin.value())
        self.plot_opts.save_dpi = int(self.save_dpi_spin.value())
        self._sync_quick_toolbar_from_state()

    def _sync_quick_toolbar_from_state(self):
        if not hasattr(self, "quick_data_btn"):
            return
        self._syncing_plot_controls = True
        try:
            pairs = (
                (self.quick_data_btn, self.show_data_chk.isChecked()),
                (self.quick_fit_btn, self.show_fit_chk.isChecked()),
                (self.quick_smooth_btn, self.show_smoothed_chk.isChecked()),
                (self.quick_signal_btn, self.show_signal_layer_chk.isChecked()),
                (self.quick_reference_btn, self.show_reference_layer_chk.isChecked()),
                (self.quick_difference_btn, self.show_difference_layer_chk.isChecked()),
                (self.quick_iterations_btn, self.show_iteration_layer_chk.isChecked()),
                (self.quick_mean_btn, self.show_iteration_mean_chk.isChecked()),
                (self.quick_std_btn, self.show_iteration_std_chk.isChecked()),
            )
            for button, checked in pairs:
                button.setChecked(bool(checked))
            self.quick_std_mode_combo.setCurrentText(self.iteration_std_mode_combo.currentText())
            has_iterations = self._iteration_count() > 0
            for widget in (
                self.quick_iterations_btn,
                self.quick_mean_btn,
                self.quick_std_btn,
                self.quick_iteration_select_btn,
                self.quick_std_mode_combo,
            ):
                widget.setEnabled(has_iterations)
        finally:
            self._syncing_plot_controls = False

    def _on_quick_data_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_data_chk.setChecked(checked)

    def _on_quick_fit_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_fit_chk.setChecked(checked)

    def _on_quick_smooth_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_smoothed_chk.setChecked(checked)

    def _on_quick_signal_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_signal_layer_chk.setChecked(checked)

    def _on_quick_reference_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_reference_layer_chk.setChecked(checked)

    def _on_quick_difference_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_difference_layer_chk.setChecked(checked)

    def _on_quick_iterations_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_iteration_layer_chk.setChecked(checked)

    def _on_quick_mean_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_iteration_mean_chk.setChecked(checked)

    def _on_quick_std_toggled(self, checked: bool):
        if not self._syncing_plot_controls:
            self.show_iteration_std_chk.setChecked(checked)

    def _on_quick_std_mode_changed(self, text: str):
        if not self._syncing_plot_controls:
            self.iteration_std_mode_combo.setCurrentText(text)

    def _selected_report_metrics(self) -> set[str]:
        if not hasattr(self, "report_metric_checks"):
            return set()
        return {k for k, w in self.report_metric_checks.items() if w.isChecked()}

    def _report_label(self, key: str) -> str:
        labels = {
            "r2": r"$R^2$",
            "rmse": "RMSE",
            "T2_star_ns": r"$T_2^*$ (ns)",
            "T1_ns": r"$T_1$ (ns)",
            "tau": r"$\tau$ (ns)",
            "pi_time_ns": r"$\pi$ time (ns)",
            "pi_over_2_time_ns": r"$\pi/2$ time (ns)",
            "pi_time_ns_nominal": r"$\pi$ nominal (ns)",
            "pi_over_2_time_ns_nominal": r"$\pi/2$ nominal (ns)",
            "rabi_period_ns": "Rabi period (ns)",
            "t0_ns": r"$t_0$ (ns)",
            "first_peak_ns": "First peak (ns)",
            "delay_ns": "Delay (ns)",
            "tau_ramp_ns": r"$\tau_r$ (ns)",
            "T2rho_ns": r"$T_{2\rho}$ (ns)",
            "rabi_envelope_decay_ns": "Rabi envelope decay (ns)",
            "rabi_envelope_beta": "Rabi envelope beta",
            "rabi_envelope_r2": "Rabi envelope R2",
            "rabi_dead_time_ns": "Derived delay (ns)",
            "rabi_freq_MHz": r"$\Omega_R/2\pi$ (MHz)",
            "detuning_MHz": r"$\Delta/2\pi$ (MHz)",
            "D": "D",
            "E": "E",
            "fwhm": "FWHM",
        }
        return labels.get(key, key)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._startup_splitter_applied:
            QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _apply_initial_splitter_sizes(self):
        if self._startup_splitter_applied or self.main_splitter is None:
            return
        total = self.main_splitter.width()
        if total <= 0:
            return
        saved = self.settings.value("layout/main_splitter_sizes", [])
        try:
            saved_sizes = [int(value) for value in saved]
        except (TypeError, ValueError):
            saved_sizes = []
        mins = [330, 700, 420]
        if len(saved_sizes) == 3 and sum(saved_sizes) > 0:
            sizes = saved_sizes
        elif total >= sum(mins):
            left, center, right = mins
            center += total - sum(mins)
            sizes = [left, center, right]
        else:
            weights = np.array([0.24, 0.48, 0.28], dtype=float)
            sizes = np.maximum(1, np.floor(weights * total).astype(int)).tolist()
            sizes[1] += total - sum(sizes)
        self.main_splitter.setSizes(sizes)
        self._startup_splitter_applied = True

    def _snapshot_summary_lines(self, metadata: dict[str, object] | None) -> list[str]:
        if not metadata or metadata.get("schema") != "snapshot_run":
            return []
        lines = ["Snapshot metadata:"]
        if metadata.get("save_type"):
            lines.append(f"  Save type: {metadata['save_type']}")
        if metadata.get("run_status"):
            lines.append(f"  Run status: {metadata['run_status']}")
        completed = metadata.get("completed_iterations")
        target = metadata.get("target_iterations")
        current = metadata.get("current_iteration")
        if completed is not None or target is not None or current is not None:
            progress_text = f"{completed if completed is not None else '?'} / {target if target is not None else '?'}"
            if current is not None:
                progress_text += f" (current={current})"
            lines.append(f"  Iterations: {progress_text}")
        if metadata.get("checkpoint_count") is not None:
            lines.append(f"  Checkpoints saved: {metadata['checkpoint_count']}")
        if metadata.get("scan_notes"):
            lines.append(f"  Notes: {metadata['scan_notes']}")
        return lines

    def _trace_summary_lines(self, trace: ExperimentTrace) -> list[str]:
        metadata = trace.metadata or {}
        completed = metadata.get("completed_iterations")
        target = metadata.get("target_iterations")
        warnings = metadata.get("warnings")
        lines = [
            trace.file_name,
            f"Detected experiment: {trace.experiment_type}",
            f"Scan type: {trace.scan_dim}",
            f"Axes: {', '.join(trace.scan_axes) if trace.scan_axes else trace.x_label}",
            f"Fit allowed: {'yes' if trace.fit_allowed else 'no'}",
            f"Source: {trace.source_path}",
        ]
        if metadata.get("save_type") or metadata.get("run_status"):
            lines.append(f"Run state: {metadata.get('run_status', 'unknown')} ({metadata.get('save_type', 'unknown')})")
        if completed is not None or target is not None:
            lines.append(f"Iteration completion: {completed if completed is not None else '?'} / {target if target is not None else '?'}")
        for key, label in (
            ("rf_frequency_ghz", "RF"),
            ("rf2_frequency_ghz", "RF2"),
            ("rf2_duration_ns", "RF2 duration"),
            ("pi_pulse_ns", "pi pulse"),
            ("tau_ns", "tau"),
        ):
            if metadata.get(key) is not None:
                unit = " GHz" if "frequency" in key else " ns"
                lines.append(f"{label}: {metadata[key]}{unit}")
        dead_time = (trace.metadata or {}).get("rabi_dead_time_ns")
        if trace.experiment_type == "Rabi" and dead_time is not None:
            lines.append(f"Rabi dead time default: {float(dead_time):.6g} ns")
        if trace.y_sem is not None:
            lines.append("Iteration SEM: available")
        if trace.iteration_contrast is not None:
            lines.append(f"Complete iteration traces: {trace.iteration_contrast.shape[1]}")
        if warnings:
            if isinstance(warnings, list):
                lines.extend([f"Warning: {w}" for w in warnings])
            else:
                lines.append(f"Warning: {warnings}")
        lines.extend(self._snapshot_summary_lines(trace.metadata if trace.metadata is not None else {}))
        return lines

    def _show_loaded_trace_summary(self):
        if self.ctx.trace is None:
            return
        self._update_metadata_status_panel(self.ctx.trace)
        self.summary_text.setPlainText("\n".join(self._trace_summary_lines(self.ctx.trace)))

    def _update_metadata_status_panel(self, trace: ExperimentTrace):
        metadata = trace.metadata or {}
        completed = metadata.get("completed_iterations")
        target = metadata.get("target_iterations")
        run_status = str(metadata.get("run_status", "unknown"))
        save_type = str(metadata.get("save_type", "unknown"))
        scan_state = "fit" if trace.fit_allowed else "plot-only"
        bits = [
            f"{trace.experiment_type} | {trace.scan_dim} | {scan_state}",
            f"run={run_status}",
            f"save={save_type}",
        ]
        if metadata.get("scan_notes"):
            bits.append(str(metadata["scan_notes"]))
        self.metadata_status_lbl.setText("Metadata: " + " | ".join(bits))
        if completed is not None and target not in (None, 0):
            pct = int(round(100.0 * float(completed) / max(float(target), 1.0)))
            self.metadata_progress_bar.setValue(max(0, min(100, pct)))
            self.metadata_progress_bar.setFormat(f"{completed}/{target} iterations")
        else:
            self.metadata_progress_bar.setValue(0)
            self.metadata_progress_bar.setFormat("iterations unavailable")

    def _recommended_mode_for_profile(self, profile: FitProfile) -> str:
        if profile.name == "ODMR":
            return "contrast"
        return self._friendly_mode_name(profile.mode)

    def _current_data_mode(self) -> str:
        text = self.mode_combo.currentText()
        if text == "signal":
            return "raw_signal"
        return text

    def _friendly_mode_name(self, mode: str) -> str:
        return "signal" if mode == "raw_signal" else mode

    def _update_profile_hint(self):
        if self.ctx.trace is None:
            self.profile_hint_lbl.setText("No file loaded.")
            return
        profile = self._active_profile()
        recommended_mode = self._recommended_mode_for_profile(profile)
        recommended_model = profile.candidate_models[0] if profile.candidate_models else "n/a"
        current_mode = self._friendly_mode_name(self._current_data_mode())
        override_note = "manual mode override active" if self._mode_override_active else "using profile default mode"
        self.profile_hint_lbl.setText(
            f"Detected: {self.ctx.trace.experiment_type} | Recommended mode: {recommended_mode} | "
            f"Recommended model: {recommended_model} | Current mode: {current_mode} ({override_note})"
        )

    def _trace_is_scan(self) -> bool:
        return bool(self.ctx.trace is not None and self.ctx.trace.scan_dim in {"scan1d", "scan2d"})

    def _iteration_count(self) -> int:
        if self.ctx.trace is None:
            return 0
        mode = self._friendly_mode_name(self._current_data_mode()) if hasattr(self, "mode_combo") else self.ctx.trace.mode
        matrix, _ = self._iteration_source_for_mode(self.ctx.trace, mode)
        if matrix is not None and matrix.ndim == 2:
            return int(matrix.shape[1])
        for fallback in (
            self.ctx.trace.iteration_contrast,
            self.ctx.trace.iteration_signal,
            self.ctx.trace.iteration_reference,
        ):
            if fallback is not None:
                fallback_matrix = np.asarray(fallback)
                if fallback_matrix.ndim == 2:
                    return int(fallback_matrix.shape[1])
        return 0

    def _update_iteration_controls(self):
        if not hasattr(self, "iteration_select_btn"):
            return
        count = self._iteration_count()
        if count <= 0:
            self._selected_iteration_indices.clear()
        else:
            self._selected_iteration_indices = {idx for idx in self._selected_iteration_indices if 0 <= idx < count}
            if not self._selected_iteration_indices:
                self._selected_iteration_indices = {0}
        for widget in (
            self.iteration_select_btn,
            self.show_iteration_layer_chk,
            self.show_iteration_mean_chk,
            self.show_iteration_std_chk,
            self.hide_failed_iterations_chk,
            self.iteration_std_mode_combo,
        ):
            widget.setEnabled(count > 0)
        self._update_iteration_summary()
        self._sync_quick_toolbar_from_state()

    def _update_iteration_summary(self):
        if not hasattr(self, "iteration_summary_lbl"):
            return
        count = self._iteration_count()
        selected = sorted(idx for idx in self._selected_iteration_indices if 0 <= idx < count)
        if count <= 0:
            text = "No iterations"
        elif not selected:
            text = "Mean only"
        elif len(selected) == count:
            text = f"All {count}"
        elif len(selected) <= 3:
            text = ", ".join(str(idx + 1) for idx in selected)
        else:
            text = f"{len(selected)} selected"
        self.iteration_summary_lbl.setText(text)

    def _open_iteration_dialog(self):
        count = self._iteration_count()
        if count <= 0:
            self._message("Iterations", "No complete iteration matrix is available for this trace.")
            return
        dlg = IterationSelectionDialog(
            self,
            count=count,
            selected={idx for idx in self._selected_iteration_indices if 0 <= idx < count},
            show_mean=self.show_iteration_mean_chk.isChecked(),
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self._selected_iteration_indices = dlg.selected_indices()
        self.show_iteration_mean_chk.setChecked(dlg.show_mean())
        self._update_iteration_summary()
        self._refresh_plot_only()

    def _update_trace_mode_visibility(self):
        is_scan = self._trace_is_scan()
        is_scan1d = bool(self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan1d")
        is_scan2d = bool(self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan2d")
        is_fit_trace = bool(self.ctx.trace is None or self.ctx.trace.fit_allowed)
        self.strategy_box.setVisible(is_fit_trace)
        self.scan_tools_box.setVisible(is_scan)
        self.param_box.setVisible(is_fit_trace)
        self.scan_help_lbl.setVisible(False)
        self.scan_linecut_lbl.setVisible(is_scan2d)
        self.scan_linecut_combo.setVisible(is_scan2d)
        self.scan_secondary_lbl.setVisible(is_scan1d)
        self.scan_secondary_combo.setVisible(is_scan1d)
        for widget in getattr(self, "fit_export_widgets", []):
            widget.setVisible(False if widget.property("drawerHidden") else is_fit_trace)
        if is_scan2d:
            self.exp_report_png_chk.setVisible(False)
            self.exp_report_pdf_chk.setVisible(False)
        else:
            self.exp_report_png_chk.setVisible(is_fit_trace)
            self.exp_report_pdf_chk.setVisible(is_fit_trace)

    def _effective_trace_metadata(self) -> dict[str, object]:
        if self.ctx.trace is None:
            return {}
        metadata = dict(self.ctx.trace.metadata or {})
        metadata["rabi_dead_time_ns"] = float(self.rabi_dead_time_ns.value())
        return metadata

    def _on_mode_changed(self, *_args):
        if self.ctx.trace is None or self._is_loading:
            return
        self._mode_override_active = True
        self._update_profile_hint()
        self._reload_current_file()

    def _on_rabi_dead_time_changed(self):
        if self.ctx.fit_result is not None:
            self.on_fit_finished(self.ctx.fit_result)
        elif self.ctx.trace is not None:
            self._show_loaded_trace_summary()

    def _refresh_plot_only(self):
        view_state = self._capture_view_state()
        self._apply_plot_controls_to_state()
        if self.ctx.fit_result is not None and self.ctx.y is not None:
            plot_y = self.ctx.fit_target if self.ctx.fit_target is not None else self.ctx.y
            if self._fit_matches_current_data(self.ctx.fit_result, plot_y):
                self._plot_fit(self.ctx.fit_result, plot_y)
            else:
                self._clear_stale_fit("processed data changed")
                self._plot_raw()
        else:
            self._plot_raw()
        self._restore_view_state(view_state)
        self.canvas.draw_idle()

    def _fit_matches_current_data(self, fit_result: FitResult | None, y_target: np.ndarray | None = None) -> bool:
        if fit_result is None or self.ctx.x is None:
            return False
        n = len(self.ctx.x)
        arrays = [fit_result.y_fit]
        if y_target is not None:
            arrays.append(y_target)
        for values in arrays:
            try:
                if len(np.asarray(values, dtype=float)) != n:
                    return False
            except Exception:
                return False
        return True

    def _clear_stale_fit(self, reason: str = "processed data changed"):
        self.ctx.fit_result = None
        self.ctx.fit_target = None
        self.ctx.odmr_peaks = None
        self.ctx.status = "N/A"
        self.ctx.status_reason = reason
        if hasattr(self, "status_lbl"):
            self._set_status("N/A", reason)

    @staticmethod
    def _arrays_differ(a: np.ndarray | None, b: np.ndarray | None) -> bool:
        if a is None or b is None:
            return a is not b
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        return a_arr.shape != b_arr.shape or not np.allclose(a_arr, b_arr, equal_nan=True)

    def _open_batch_dialog(self):
        self._set_workspace_mode(1)

    def _open_preferences(self):
        dlg = PreferencesDialog(self, self.plot_opts)
        if dlg.exec() == QDialog.Accepted:
            self.fig_w_spin.setValue(float(dlg.width_spin.value()))
            self.fig_h_spin.setValue(float(dlg.height_spin.value()))
            self.save_dpi_spin.setValue(int(dlg.dpi_spin.value()))
            self.legend_font_spin.setValue(int(dlg.legend_font_spin.value()))
            self.show_residual_chk.setChecked(dlg.residual_chk.isChecked())
            self.show_legend_chk.setChecked(dlg.legend_chk.isChecked())
            self._apply_plot_controls_to_state()
            self._save_preferences()
            self._refresh_plot_only()

    def _save_preferences(self):
        self._apply_plot_controls_to_state()
        if self.main_splitter is not None:
            self.settings.setValue("layout/main_splitter_sizes", self.main_splitter.sizes())
        s = self.settings
        s.setValue("plot/fig_width", self.plot_opts.fig_width)
        s.setValue("plot/fig_height", self.plot_opts.fig_height)
        s.setValue("plot/save_dpi", self.plot_opts.save_dpi)
        s.setValue("plot/show_residual", self.plot_opts.show_residual)
        s.setValue("plot/show_legend", self.plot_opts.show_legend)
        s.setValue("plot/show_data", self.plot_opts.show_data)
        s.setValue("plot/show_smoothed", self.plot_opts.show_smoothed)
        s.setValue("plot/show_fit", self.plot_opts.show_fit)
        s.setValue("plot/show_rabi_envelope", self.plot_opts.show_rabi_envelope)
        s.setValue("plot/show_peaks", self.plot_opts.show_peaks)
        s.setValue("plot/show_signal_layer", self.plot_opts.show_signal_layer)
        s.setValue("plot/show_reference_layer", self.plot_opts.show_reference_layer)
        s.setValue("plot/show_difference_layer", self.plot_opts.show_difference_layer)
        s.setValue("plot/show_iteration_layer", self.plot_opts.show_iteration_layer)
        s.setValue("plot/show_iteration_mean", self.plot_opts.show_iteration_mean)
        s.setValue("plot/show_iteration_std", self.plot_opts.show_iteration_std)
        s.setValue("plot/iteration_std_mode", self.plot_opts.iteration_std_mode)
        s.setValue("plot/hide_failed_iterations", self.plot_opts.hide_failed_iterations)
        s.setValue("plot/show_fft_panel", self.plot_opts.show_fft_panel)
        s.setValue("plot/legend_loc", self.plot_opts.legend_loc)
        s.setValue("plot/legend_font_size", self.plot_opts.legend_font_size)
        s.setValue("plot/legend_compact_labels", self.plot_opts.legend_compact_labels)
        s.setValue("plot/annotation_mode", self.plot_opts.annotation_mode)
        s.setValue("plot/annotation_loc", self.plot_opts.annotation_loc)
        s.setValue("plot/annotation_font_size", self.plot_opts.annotation_font_size)
        s.setValue("plot/annotation_file_model", self.plot_opts.annotation_show_file_model)
        s.setValue("plot/annotation_fit_metrics", self.plot_opts.annotation_show_fit_metrics)
        s.setValue("plot/annotation_t2rho", self.plot_opts.annotation_show_t2rho)
        s.setValue("plot/annotation_nv_metrics", self.plot_opts.annotation_show_nv_metrics)
        s.setValue("plot/annotation_params", self.plot_opts.annotation_show_params)
        s.setValue("plot/plot_style", self.plot_opts.plot_style)
        s.setValue("export/png", self.exp_png_chk.isChecked())
        s.setValue("export/pdf", self.exp_pdf_chk.isChecked())
        s.setValue("export/svg", self.exp_svg_chk.isChecked())
        s.setValue("export/report_png", self.exp_report_png_chk.isChecked())
        s.setValue("export/report_pdf", self.exp_report_pdf_chk.isChecked())
        s.setValue("export/json", self.exp_json_chk.isChecked())
        s.setValue("export/csv", self.exp_csv_chk.isChecked())
        s.setValue("export/origin", self.exp_origin_chk.isChecked())
        for key, chk in self.report_metric_checks.items():
            s.setValue(f"report/{key}", chk.isChecked())

    def _load_preferences(self):
        s = self.settings
        self.fig_w_spin.setValue(float(s.value("plot/fig_width", self.plot_opts.fig_width)))
        self.fig_h_spin.setValue(float(s.value("plot/fig_height", self.plot_opts.fig_height)))
        self.save_dpi_spin.setValue(int(s.value("plot/save_dpi", self.plot_opts.save_dpi)))
        self.show_residual_chk.setChecked(str(s.value("plot/show_residual", "true")).lower() != "false")
        self.show_legend_chk.setChecked(str(s.value("plot/show_legend", "true")).lower() != "false")
        self.show_data_chk.setChecked(str(s.value("plot/show_data", "true")).lower() != "false")
        self.show_smoothed_chk.setChecked(str(s.value("plot/show_smoothed", "true")).lower() != "false")
        self.show_fit_chk.setChecked(str(s.value("plot/show_fit", "true")).lower() != "false")
        self.show_rabi_envelope_chk.setChecked(str(s.value("plot/show_rabi_envelope", "true")).lower() != "false")
        self.show_peaks_chk.setChecked(str(s.value("plot/show_peaks", "true")).lower() != "false")
        self.show_signal_layer_chk.setChecked(str(s.value("plot/show_signal_layer", "false")).lower() == "true")
        self.show_reference_layer_chk.setChecked(str(s.value("plot/show_reference_layer", "false")).lower() == "true")
        self.show_difference_layer_chk.setChecked(str(s.value("plot/show_difference_layer", "false")).lower() == "true")
        self.show_iteration_layer_chk.setChecked(str(s.value("plot/show_iteration_layer", "false")).lower() == "true")
        self.show_iteration_mean_chk.setChecked(str(s.value("plot/show_iteration_mean", "true")).lower() != "false")
        self.show_iteration_std_chk.setChecked(str(s.value("plot/show_iteration_std", "false")).lower() == "true")
        self.iteration_std_mode_combo.setCurrentText(str(s.value("plot/iteration_std_mode", self.plot_opts.iteration_std_mode)))
        self.hide_failed_iterations_chk.setChecked(str(s.value("plot/hide_failed_iterations", "true")).lower() != "false")
        self.show_fft_panel_chk.setChecked(str(s.value("plot/show_fft_panel", "false")).lower() == "true")
        self.legend_loc_combo.setCurrentText(str(s.value("plot/legend_loc", self.plot_opts.legend_loc)))
        self.legend_font_spin.setValue(int(s.value("plot/legend_font_size", self.plot_opts.legend_font_size)))
        self.legend_compact_chk.setChecked(str(s.value("plot/legend_compact_labels", "true")).lower() != "false")
        self.annotation_mode_combo.setCurrentText(str(s.value("plot/annotation_mode", "Compact")))
        self.annotation_loc_combo.setCurrentText(str(s.value("plot/annotation_loc", "upper right")))
        self.annotation_font_spin.setValue(int(s.value("plot/annotation_font_size", 9)))
        self.ann_file_model_chk.setChecked(True)
        self.ann_metrics_chk.setChecked(str(s.value("plot/annotation_fit_metrics", "true")).lower() != "false")
        self.ann_t2rho_chk.setChecked(str(s.value("plot/annotation_t2rho", "true")).lower() != "false")
        self.ann_nv_metrics_chk.setChecked(str(s.value("plot/annotation_nv_metrics", "true")).lower() != "false")
        self.ann_params_chk.setChecked(str(s.value("plot/annotation_params", "false")).lower() == "true")
        self.plot_style_combo.setCurrentText(str(s.value("plot/plot_style", "Line + scatter")))
        self.exp_png_chk.setChecked(str(s.value("export/png", "true")).lower() != "false")
        self.exp_pdf_chk.setChecked(str(s.value("export/pdf", "false")).lower() == "true")
        self.exp_svg_chk.setChecked(str(s.value("export/svg", "false")).lower() == "true")
        self.exp_report_png_chk.setChecked(str(s.value("export/report_png", "false")).lower() == "true")
        self.exp_report_pdf_chk.setChecked(str(s.value("export/report_pdf", "false")).lower() == "true")
        self.exp_json_chk.setChecked(str(s.value("export/json", "true")).lower() != "false")
        self.exp_csv_chk.setChecked(str(s.value("export/csv", "true")).lower() != "false")
        self.exp_origin_chk.setChecked(str(s.value("export/origin", "false")).lower() == "true")
        for key, chk in self.report_metric_checks.items():
            default_val = "true" if chk.isChecked() else "false"
            chk.setChecked(str(s.value(f"report/{key}", default_val)).lower() != "false")
        # Export defaults should always start from PNG-only; users can enable other formats per save.
        self.exp_png_chk.setChecked(True)
        self.exp_pdf_chk.setChecked(False)
        self.exp_svg_chk.setChecked(False)
        self.exp_report_png_chk.setChecked(False)
        self.exp_report_pdf_chk.setChecked(False)
        self._update_contextual_visibility()
        self._sync_quick_toolbar_from_state()

    def _update_contextual_visibility(self):
        model = self.model_combo.currentText()
        is_odmr = "ODMR" in model or model == "DEERFrequency"
        is_rabi = "Rabi" in model or model == "DEERDuration"
        is_t1 = model == "T1"
        is_custom = model == "CustomModel"
        is_fit_trace = bool(self.ctx.trace is None or self.ctx.trace.fit_allowed)
        
        self.rabi_container.setVisible(is_fit_trace and is_rabi)
        self.t1_container.setVisible(is_fit_trace and is_t1)
        self.odmr_container.setVisible(is_fit_trace and is_odmr)
        self.custom_container.setVisible(is_fit_trace and is_custom)
        self.eq_preview_canvas.setVisible(is_fit_trace or is_t1)
        self.equation_display.setVisible(is_fit_trace or is_t1)
        self.quality_gate_chk.setVisible(is_fit_trace)
        self.fit_smoothed_chk.setVisible(is_fit_trace)
        self._update_rabi_model_help()
        self._update_trace_mode_visibility()
        self._update_equation_display()

    def _update_rabi_model_help(self):
        if not hasattr(self, "rabi_model_help_lbl"):
            return
        mode = self.rabi_mode_combo.currentText() if hasattr(self, "rabi_mode_combo") else ""
        if mode.startswith("Phase-ramp"):
            text = "Phase-ramp fits microwave turn-on for short calibration runs and auto-switches to long damped Rabi when a trace is long and strongly decayed."
        elif mode.startswith("Long"):
            text = "Long damped Rabi uses peak-derived frequency seeds and calibration-weighted residuals for long decayed scans where late baseline points dominate ordinary residuals."
        elif mode.startswith("Chirped"):
            text = "Chirped cosine allows the effective oscillation frequency to drift across the trace. It usually gives the prettiest overlay, but the startup parameter is less directly physical."
        elif mode.startswith("Constant-frequency"):
            text = "Constant-frequency damped cosine is the legacy fixed-frequency model. It is useful as a reference, but it often leaves structured early-time residuals on your Rabi data."
        else:
            text = "Auto compare fits phase-ramp, long damped, constant-frequency, and chirped candidates and keeps the strongest result by the built-in acceptance rules."
        self.rabi_model_help_lbl.setText(text)

    def _update_equation_display(self):
        m = self.model_combo.currentText()
        if m == "T1":
            stretch = self.t1_model_combo.currentText().startswith("Stretched")
            normalized = self.t1_prep_combo.currentText().startswith("Normalize")
            if stretch and normalized:
                latex_eq = r"$y_0 + A e^{-(t/T_1)^n}$ after normalize then $(1-\mathrm{data})$"
                code_eq = "y0 + A*np.exp(-((np.abs(t/T1))**n))  # after normalize then 1-data"
            elif stretch:
                latex_eq = r"$y_0 + A(1-e^{-(t/T_1)^n})$"
                code_eq = "y0 + A*(1.0 - np.exp(-((np.abs(t/T1))**n)))"
            elif normalized:
                latex_eq = r"$y_0 + A e^{-t/T_1}$ after normalize then $(1-\mathrm{data})$"
                code_eq = "y0 + A*np.exp(-np.abs(t/T1))  # after normalize then 1-data"
            else:
                latex_eq = r"$y_0 + A(1-e^{-t/T_1})$"
                code_eq = "y0 + A*(1.0 - np.exp(-np.abs(t/T1)))"
        else:
            latex_eq = self.equation_map.get(m, "Model equation unavailable.")
            code_eq = self.equation_expr_template.get(m, "")
        
        # 1. Update Preview (Latex)
        if m == "CustomModel" and self.ctx.custom_model_spec:
            latex_eq = f"Custom: {self.ctx.custom_model_spec.get('name', 'User Model')}"
        
        self.eq_preview_ax.clear()
        self.eq_preview_ax.axis("off")
        disp = latex_eq if ("$" in latex_eq or "Custom" in latex_eq) else f"${latex_eq}$"
        try:
            self.eq_preview_ax.text(0.01, 0.5, disp, fontsize=11, va="center", ha="left")
        except Exception:
            self.eq_preview_ax.text(0.01, 0.5, latex_eq, fontsize=10, va="center", ha="left")
        self.eq_preview_fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        self.eq_preview_canvas.draw_idle()

        # 2. Update Editor (Code)
        self.equation_display.blockSignals(True)
        if m == "CustomModel" and self.ctx.custom_model_spec:
            code_eq = str(self.ctx.custom_model_spec.get("expression", ""))
        
        if self.equation_display.toPlainText() != code_eq:
            self.equation_display.setPlainText(code_eq)
        self.equation_display.blockSignals(False)

    def _on_equation_edited(self):
        m = self.model_combo.currentText()
        new_expr = self.equation_display.toPlainText()
        
        if m != "CustomModel":
            self.on_seed_custom_from_current_model(initial_expr=new_expr)
        else:
            if self.ctx.custom_model_spec:
                self.ctx.custom_model_spec["expression"] = new_expr

    def on_seed_custom_from_current_model(self, initial_expr: str | None = None):
        m = self.model_combo.currentText()
        expr = initial_expr
        if expr is None:
            if m == "CustomModel":
                expr = self.equation_display.toPlainText().strip()
            else:
                expr = self.equation_expr_template.get(m, "")
        
        if not expr:
            expr = "y0 + A*np.cos(2*np.pi*f*t)"

        # Basic default params for custom model
        default = {
            "name": f"CustomFrom{m}",
            "expression": expr,
            "param_names": ["y0", "A", "f", "phi"],
            "lower": [-1.0, -1.0, 1e-6, -2 * np.pi],
            "upper": [1.0, 1.0, 0.2, 2 * np.pi],
            "seed": [0.0, 0.02, 0.003, 0.0],
        }
        self.ctx.custom_model_spec = default
        
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentText("CustomModel")
        self.model_combo.blockSignals(False)
        self._update_contextual_visibility()

    def _update_mask_mode(self):
        mode = self.mask_mode_combo.currentText()
        if self.span_selector is not None:
            self.span_selector.set_active(False)
        self._clear_crosshairs()
        self.canvas.draw_idle()

    def _analysis_snapshot(self) -> dict[str, Any]:
        return {"excluded_points": set(self.ctx.excluded_points or set()), "exclusion_ranges": list(self.ctx.exclusion_ranges or []), "steps": [step.to_dict() for step in self.ctx.analysis_steps]}

    def _record_analysis_state(self) -> None:
        self._analysis_undo_stack.append(self._analysis_snapshot())
        self._analysis_redo_stack.clear()
        if len(self._analysis_undo_stack) > 100:
            self._analysis_undo_stack.pop(0)

    def _restore_analysis_state(self, snapshot: dict[str, Any]) -> None:
        self.ctx.excluded_points = set(snapshot.get("excluded_points") or set())
        self.ctx.exclusion_ranges = [tuple(value) for value in snapshot.get("exclusion_ranges") or []]
        self.ctx.analysis_steps = [AnalysisStep.from_dict(value) for value in snapshot.get("steps") or []]
        self._refresh_mask_list()
        self._refresh_analysis_list()
        self._refresh_processed()

    def _undo(self) -> None:
        if self._analysis_undo_stack:
            self._analysis_redo_stack.append(self._analysis_snapshot())
            self._restore_analysis_state(self._analysis_undo_stack.pop())
        else:
            self._undo_params()

    def _redo(self) -> None:
        if self._analysis_redo_stack:
            self._analysis_undo_stack.append(self._analysis_snapshot())
            self._restore_analysis_state(self._analysis_redo_stack.pop())
        else:
            self._redo_params()

    def _sync_analysis_step_fields(self) -> None:
        kind = self.analysis_kind_combo.currentText() if hasattr(self, "analysis_kind_combo") else ""
        options = {"Baseline": ["median", "mean", "linear", "first"], "Detrend": ["linear"], "Normalize": ["minmax", "zscore", "area"], "Derivative": ["gradient"], "Integral": ["cumulative"], "Resample": ["uniform"], "Expression": ["custom"]}
        self.analysis_option_combo.blockSignals(True)
        self.analysis_option_combo.clear()
        self.analysis_option_combo.addItems(options.get(kind, []))
        self.analysis_option_combo.blockSignals(False)
        if kind == "Resample":
            self.analysis_value_edit.setPlaceholderText("Point count (default 200)")
            self.analysis_value_edit.setText(self.analysis_value_edit.text() or "200")
        elif kind == "Expression":
            self.analysis_value_edit.setPlaceholderText("Expression, e.g. y / max(abs(y))")
            if self.analysis_value_edit.text() == "200":
                self.analysis_value_edit.clear()
        else:
            self.analysis_value_edit.clear()
            self.analysis_value_edit.setPlaceholderText("No additional value")

    def on_add_analysis_step(self) -> None:
        kind_map = {"Baseline": "baseline", "Detrend": "detrend", "Normalize": "normalize", "Derivative": "derivative", "Integral": "integral", "Resample": "resample", "Expression": "expression"}
        kind = kind_map[self.analysis_kind_combo.currentText()]
        option = self.analysis_option_combo.currentText()
        params: dict[str, Any] = {"method": option} if kind in {"baseline", "normalize"} else {}
        if kind == "resample":
            try:
                params["count"] = max(2, int(self.analysis_value_edit.text() or "200"))
            except ValueError:
                self._message("Analysis transform", "Resample point count must be a whole number.")
                return
        elif kind == "expression":
            params["expression"] = self.analysis_value_edit.text().strip() or "y"
        self._record_analysis_state()
        self.ctx.analysis_steps.append(AnalysisStep(kind, params))
        try:
            self._refresh_processed()
        except ValueError as exc:
            self.ctx.analysis_steps.pop()
            self._analysis_undo_stack.pop()
            self._message("Analysis transform", str(exc))
            return
        self._refresh_analysis_list()

    def on_remove_analysis_step(self) -> None:
        item = self.analysis_list.currentItem()
        if item is None or item.data(Qt.UserRole) is None:
            return
        index = int(item.data(Qt.UserRole))
        if 0 <= index < len(self.ctx.analysis_steps):
            self._record_analysis_state()
            self.ctx.analysis_steps.pop(index)
            self._refresh_analysis_list()
            self._refresh_processed()

    def on_clear_analysis_steps(self) -> None:
        if self.ctx.analysis_steps:
            self._record_analysis_state()
            self.ctx.analysis_steps.clear()
            self._refresh_analysis_list()
            self._refresh_processed()

    def _refresh_analysis_list(self) -> None:
        if not hasattr(self, "analysis_list"):
            return
        self.analysis_list.clear()
        for index, step in enumerate(self.ctx.analysis_steps):
            details = ", ".join(f"{key}={value}" for key, value in step.params.items())
            item = QListWidgetItem(f"{index + 1}. {step.kind}" + (f" ({details})" if details else ""))
            item.setData(Qt.UserRole, index)
            self.analysis_list.addItem(item)
        if not self.ctx.analysis_steps:
            self.analysis_list.addItem("No display transforms")

    def _on_range_selected(self, xmin: float, xmax: float):
        if self.ctx.exclusion_ranges is None:
            self.ctx.exclusion_ranges = []
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        self._record_analysis_state()
        self.ctx.exclusion_ranges.append((float(lo), float(hi)))
        self._refresh_mask_list()
        self._refresh_processed()

    def _collect_sample_metadata(self) -> dict[str, str]:
        return {
            "sample_id": self.meta_sample_id.text().strip(),
            "condition": self.meta_condition.text().strip(),
            "power": self.meta_power.text().strip(),
            "date": self.meta_date.text().strip(),
            "notes": self.meta_notes.text().strip(),
        }

    def _update_recent_files(self, path: str):
        rec = list(self.settings.value("session/recent_files", []))
        if path in rec:
            rec.remove(path)
        rec.insert(0, path)
        rec = rec[:25]
        self.settings.setValue("session/recent_files", rec)

    def _update_recent_folder(self, folder: str):
        fav = list(self.settings.value("session/favorite_folders", []))
        if folder in fav:
            fav.remove(folder)
        fav.insert(0, folder)
        fav = fav[:10]
        self.settings.setValue("session/favorite_folders", fav)

    def on_open_recent(self):
        rec = list(self.settings.value("session/recent_files", []))
        if not rec:
            self._message("Recent files", "No recent files stored yet.")
            return
        p = rec[0]
        if Path(p).exists():
            self._load_file(p)
        else:
            self._message("Recent files", f"File not found:\n{p}")

    def on_open_favorite_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose favorite folder")
        if not folder:
            fav = list(self.settings.value("session/favorite_folders", []))
            if fav:
                folder = fav[0]
            else:
                return
        self._update_recent_folder(folder)
        p, _ = QFileDialog.getOpenFileName(self, "Open .mat from favorite folder", folder, "MAT files (*.mat)")
        if p:
            self._load_file(p)

    def _update_overlay_widgets(self):
        self.overlay_list.clear()
        self.baseline_combo.blockSignals(True)
        self.baseline_combo.clear()
        self.baseline_combo.addItem("None")
        if self.ctx.loaded_traces is None:
            self.ctx.loaded_traces = {}
        for name in sorted(self.ctx.loaded_traces.keys()):
            item = QListWidgetItem(name)
            item.setCheckState(Qt.Checked)
            self.overlay_list.addItem(item)
            self.baseline_combo.addItem(name)
        self.baseline_combo.blockSignals(False)

    def on_add_overlays(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select overlay MAT files", "", "MAT files (*.mat)")
        if not paths:
            return
        mode: DataMode = self._current_data_mode()  # type: ignore[assignment]
        if self.ctx.loaded_traces is None:
            self.ctx.loaded_traces = {}
        for p in paths:
            try:
                tr = load_saved_data_mat(p, mode=mode)
            except Exception:
                continue
            self.ctx.loaded_traces[tr.file_name] = tr
            self._update_recent_files(str(Path(p).resolve()))
        self._update_overlay_widgets()
        self._plot_raw()

    def on_remove_overlays(self):
        if not self.ctx.loaded_traces:
            return
        selected = [it.text() for it in self.overlay_list.selectedItems()]
        if not selected:
            return
        current_name = self.ctx.trace.file_name if self.ctx.trace is not None else ""
        for nm in selected:
            if nm == current_name:
                continue
            self.ctx.loaded_traces.pop(nm, None)
        self._update_overlay_widgets()
        self._plot_raw()

    def on_clear_overlays(self):
        if self.ctx.loaded_traces is None:
            self.ctx.loaded_traces = {}
        current = self.ctx.trace.file_name if self.ctx.trace is not None else None
        keep = {}
        if current and current in self.ctx.loaded_traces:
            keep[current] = self.ctx.loaded_traces[current]
        self.ctx.loaded_traces = keep
        self._update_overlay_widgets()
        self._plot_raw()

    def on_compare_traces(self):
        metric = self.compare_metric_combo.currentText()
        rows: list[str] = []
        vals: list[float] = []
        labels: list[str] = []
        if self.ctx.loaded_traces is None:
            self.ctx.loaded_traces = {}
        for name, tr in self.ctx.loaded_traces.items():
            x = tr.x_ns.copy()
            y = tr.y.copy()
            x, y = bin_trace(x, y, max(1, self.bin_spin.value()))
            try:
                if tr.experiment_type == "Ramsey":
                    res = fit_ramsey_physics_first(x, y).best
                else:
                    fit_name = tr.experiment_type if tr.experiment_type in {"Rabi", "SpinEcho", "DynamicDecoupling", "T1", "ODMR", "DEERFrequency", "DEERDuration", "DEERDecay"} else "SpinEcho"
                    res = self._run_single_fit_background(
                        fit_name,
                        x,
                        y,
                        {},
                        {
                            "multistart": int(self.multistart_spin.value()),
                            "rabi_use_sem": self.rabi_sem_weight_chk.isChecked(),
                            "rabi_pi_lock": False,
                            "rabi_pi_ns": float(self.rabi_pi_ns.value()),
                            "rabi_pi2_ns": float(self.rabi_pi2_ns.value()),
                            "rabi_dead_time_ns": float(self.rabi_dead_time_ns.value()),
                            "rabi_mode": self.rabi_mode_combo.currentText(),
                            "odmr_multipeak": self.odmr_multipeak_fit_chk.isChecked(),
                            "odmr_polarity": self.odmr_polarity_combo.currentText(),
                            "odmr_peak_ranges": tuple(self._parse_odmr_ranges()),
                            "odmr_peak_count": int(self.odmr_peak_count_spin.value()),
                            "odmr_auto_peak": self.odmr_auto_peak_count_chk.isChecked(),
                            "t1_prep_mode": self.t1_prep_combo.currentText(),
                            "t1_model_family": self.t1_model_combo.currentText(),
                            "custom_spec": self.ctx.custom_model_spec,
                            "custom_expr": self.equation_display.toPlainText().strip(),
                        },
                    )
                nv = _nv_metrics("Ramsey" if tr.experiment_type == "Ramsey" else tr.experiment_type, res, tr.metadata or {})
                mv = nv.get(metric, res.r2 if metric == "r2" else np.nan)
                rows.append(f"{name}: {metric} = {mv:.6g}")
                if np.isfinite(mv):
                    vals.append(float(mv))
                    labels.append(name)
            except Exception as exc:
                rows.append(f"{name}: compare failed ({exc})")
        if rows:
            self.summary_text.setPlainText("Comparison\n" + "\n".join(rows))
        if vals:
            self.ax_res.clear()
            self.ax_res.plot(np.arange(len(vals)), vals, "o-")
            self.ax_res.set_xticks(np.arange(len(vals)))
            self.ax_res.set_xticklabels([f"{i+1}" for i in range(len(vals))], rotation=0)
            self.ax_res.set_ylabel(metric)
            self.ax_res.set_xlabel("Trace index")
            self.ax_res.grid(True, alpha=0.3)
            self.ax_res.set_title("Comparison plot (see summary for filename map)")
            self.canvas.draw_idle()

    def on_overlay_set_primary(self, item: QListWidgetItem):
        nm = item.text()
        if not self.ctx.loaded_traces or nm not in self.ctx.loaded_traces:
            return
        self.ctx.trace = self.ctx.loaded_traces[nm]
        self.ctx.fit_result = None
        self.ctx.fit_target = None
        self.ctx.odmr_peaks = None
        tr = self.ctx.trace
        if len(tr.x_ns) == 0:
            self._message("Overlay", "Selected overlay trace is empty in the current observable mode.")
            return
        self.roi_min.setText(f"{np.min(tr.x_ns):.6g}")
        self.roi_max.setText(f"{np.max(tr.x_ns):.6g}")
        self.rabi_dead_time_ns.setValue(float((tr.metadata or {}).get("rabi_dead_time_ns", 0.0) or 0.0))
        self._update_contextual_visibility()
        self._refresh_processed()
        self._show_loaded_trace_summary()

    def on_add_exclusion_range(self):
        if self.ctx.exclusion_ranges is None:
            self.ctx.exclusion_ranges = []
        lo = _safe_float(self.mask_xmin.text(), np.nan)
        hi = _safe_float(self.mask_xmax.text(), np.nan)
        if not np.isfinite(lo) or not np.isfinite(hi):
            self._message("Mask", "Enter valid x min and x max.")
            return
        if hi < lo:
            lo, hi = hi, lo
        self._record_analysis_state()
        self.ctx.exclusion_ranges.append((lo, hi))
        self._refresh_mask_list()
        self._refresh_processed()

    def on_exclude_selected_point(self):
        if self.ctx.x is None or self._selected_plot_point is None or "x" not in self._selected_plot_point:
            self._message("Exclusions", "Select a 1D point in the live plot first.")
            return
        if self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan2d":
            self._message("Exclusions", "Point exclusions apply to 1D fit traces. Use linecuts or ranges for scan review.")
            return
        processed = self.ctx.processed
        if processed is None or len(processed.analysis_x) == 0:
            return
        displayed_index = int(np.argmin(np.abs(processed.analysis_x - float(self._selected_plot_point["x"]))))
        if displayed_index >= len(processed.source_groups):
            return
        raw_indices = set(int(value) for value in processed.source_groups[displayed_index])
        self._record_analysis_state()
        self.ctx.excluded_points.update(raw_indices)
        self._refresh_mask_list()
        self._refresh_processed()

    def on_remove_selected_exclusion(self):
        selected = self.mask_list.selectedItems() if hasattr(self.mask_list, "selectedItems") else []
        if not selected:
            return
        ranges_to_remove: list[int] = []
        points_to_remove: list[int] = []
        for item in selected:
            data = item.data(Qt.UserRole)
            if not data:
                continue
            kind, value = data
            if kind == "range":
                ranges_to_remove.append(int(value))
            elif kind == "point":
                points_to_remove.append(int(value))
        self._record_analysis_state()
        for idx in sorted(set(ranges_to_remove), reverse=True):
            if self.ctx.exclusion_ranges is not None and 0 <= idx < len(self.ctx.exclusion_ranges):
                self.ctx.exclusion_ranges.pop(idx)
        if self.ctx.excluded_points is not None:
            for idx in points_to_remove:
                self.ctx.excluded_points.discard(idx)
        self._refresh_mask_list()
        self._refresh_processed()

    def on_clear_exclusions(self):
        if not (self.ctx.excluded_points or self.ctx.exclusion_ranges):
            return
        self._record_analysis_state()
        self.ctx.excluded_points = set()
        self.ctx.exclusion_ranges = []
        self._refresh_mask_list()
        self._refresh_processed()

    def _refresh_mask_list(self):
        if not hasattr(self, "mask_list"):
            return
        self.mask_list.clear()
        if self.ctx.exclusion_ranges:
            for idx, (a, b) in enumerate(self.ctx.exclusion_ranges):
                item = QListWidgetItem(f"Range: {a:.6g} to {b:.6g}")
                item.setData(Qt.UserRole, ("range", idx))
                self.mask_list.addItem(item)
        if self.ctx.excluded_points:
            for idx in sorted(self.ctx.excluded_points):
                if self.ctx.x is not None and 0 <= idx < len(self.ctx.x):
                    label = f"Point: index {idx}, x={float(self.ctx.x[idx]):.6g}"
                else:
                    label = f"Point: index {idx}"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, ("point", idx))
                self.mask_list.addItem(item)
        if self.mask_list.count() == 0:
            self.mask_list.addItem("No exclusions")

    def _apply_exclusions(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = np.ones(len(x), dtype=bool)
        if self.ctx.exclusion_ranges:
            for a, b in self.ctx.exclusion_ranges:
                keep &= ~((x >= a) & (x <= b))
        if self.ctx.excluded_points:
            for i in self.ctx.excluded_points:
                if 0 <= i < len(keep):
                    keep[i] = False
        return x[keep], y[keep]

    def _on_plot_click(self, event):
        if event.inaxes not in self._inspectable_axes():
            return
        mode = self.mask_mode_combo.currentText()
        if event.xdata is None:
            return
        if mode == "Inspect point":
            self._select_point_from_plot(float(event.xdata), float(event.ydata) if event.ydata is not None else None)

    def on_save_preset(self):
        out, _ = QFileDialog.getSaveFileName(self, "Save analysis preset", "", "Preset JSON (*.json)")
        if not out:
            return
        self._apply_plot_controls_to_state()
        payload = {
            "preset_name": Path(out).stem,
            "profile": self.profile_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "model": self.model_combo.currentText(),
            "rabi_mode": self.rabi_mode_combo.currentText(),
            "t1_prep_mode": self.t1_prep_combo.currentText(),
            "t1_model_family": self.t1_model_combo.currentText(),
            "roi_min": self.roi_min.text(),
            "roi_max": self.roi_max.text(),
            "bin_size": self.bin_spin.value(),
            "smooth_window": self.smooth_spin.value(),
            "locks": self._collect_locks(),
            "plot_options": self.plot_opts.__dict__,
            "odmr": {
                "display": self.odmr_display_combo.currentText(),
                "polarity": self.odmr_polarity_combo.currentText(),
                "ranges": self.odmr_ranges.text(),
                "peak_pick_only": self.odmr_peak_pick_only.isChecked(),
                "multipeak_fit": self.odmr_multipeak_fit_chk.isChecked(),
                "peak_count": self.odmr_peak_count_spin.value(),
                "auto_peak_count": self.odmr_auto_peak_count_chk.isChecked(),
                "cal_scale": self.odmr_cal_scale.value(),
                "cal_offset": self.odmr_cal_offset.value(),
            },
        }
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.ctx.active_preset_name = payload["preset_name"]
        self._message("Preset", f"Saved preset:\n{out}")

    def copy_export_figure(self) -> None:
        if self.ctx.trace is None:
            self._message("Copy figure", "Load data before copying a figure.")
            return
        buffer = BytesIO()
        self.fig.savefig(buffer, format="png", dpi=max(300, int(self.plot_opts.save_dpi)), bbox_inches="tight", pad_inches=0.12)
        image = QImage.fromData(buffer.getvalue(), "PNG")
        if image.isNull():
            self._message("Copy figure", "The figure could not be rendered for the clipboard.")
            return
        QApplication.clipboard().setImage(image)
        self.statusBar().showMessage("Export figure copied to clipboard", 4000)

    def _session_payload(self, session_path: str) -> dict[str, Any]:
        if self.ctx.trace is None:
            raise ValueError("Load data before saving a session.")
        trace = self.ctx.trace
        overlays = [source_descriptor(overlay.source_path, session_path) for name, overlay in (self.ctx.loaded_traces or {}).items() if name != trace.file_name]
        return {
            "primary_source": source_descriptor(trace.source_path, session_path), "profile": self.profile_combo.currentText(),
            "mode": self.mode_combo.currentText(), "model": self.model_combo.currentText(),
            "roi": [self.roi_min.text(), self.roi_max.text()], "bin_size": self.bin_spin.value(), "smooth_window": self.smooth_spin.value(),
            "excluded_raw_indices": sorted(self.ctx.excluded_points or set()), "exclusion_ranges": list(self.ctx.exclusion_ranges or []),
            "analysis_steps": [step.to_dict() for step in self.ctx.analysis_steps], "plot_options": self.plot_opts.__dict__,
            "sample_metadata": self._collect_sample_metadata(), "locks": self._collect_locks(),
            "overlay_sources": overlays,
            "view": {"xlim": list(self.ax_main.get_xlim()), "ylim": list(self.ax_main.get_ylim())},
        }

    def on_save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save analysis session", "", "SmartFitter session (*.nvfit-session.json)")
        if not path:
            return
        if not path.endswith(".nvfit-session.json"):
            path += ".nvfit-session.json"
        try:
            save_session(path, self._session_payload(path))
        except Exception as exc:
            self._message("Save session", str(exc))
            return
        self.settings.setValue("session/last_session", path)
        self.statusBar().showMessage(f"Saved analysis session: {Path(path).name}", 5000)

    def on_load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open analysis session", str(self.settings.value("session/last_session", "")), "SmartFitter session (*.nvfit-session.json)")
        if not path:
            return
        try:
            document = load_session(path)
            source, changed = resolve_source(dict(document["primary_source"]), path)
        except Exception as exc:
            self._message("Open session", str(exc))
            return
        if source is None:
            replacement, _ = QFileDialog.getOpenFileName(self, "Locate the session's primary MAT file", str(Path(path).parent), "MAT files (*.mat)")
            source = Path(replacement) if replacement else None
        if source is None or not self._load_file(str(source)):
            return
        for descriptor in document.get("overlay_sources") or []:
            overlay_path, _changed = resolve_source(dict(descriptor), path)
            if overlay_path is None or overlay_path.resolve() == source.resolve():
                continue
            try:
                overlay = load_saved_data_mat(str(overlay_path), mode=self._current_data_mode())
                self.ctx.loaded_traces[overlay.file_name] = overlay
            except Exception:
                continue
        self._update_overlay_widgets()
        self.profile_combo.setCurrentText(str(document.get("profile", "Auto")))
        self.mode_combo.setCurrentText(str(document.get("mode", "contrast")))
        self.model_combo.setCurrentText(str(document.get("model", self.model_combo.currentText())))
        roi = document.get("roi") or ["", ""]
        self.roi_min.setText(str(roi[0] if len(roi) else "")); self.roi_max.setText(str(roi[1] if len(roi) > 1 else ""))
        self.bin_spin.setValue(int(document.get("bin_size", 1))); self.smooth_spin.setValue(int(document.get("smooth_window", 1)))
        self.ctx.excluded_points = set(int(value) for value in document.get("excluded_raw_indices") or [])
        self.ctx.exclusion_ranges = [tuple(value) for value in document.get("exclusion_ranges") or []]
        self.ctx.analysis_steps = [AnalysisStep.from_dict(value) for value in document.get("analysis_steps") or []]
        for key, value in dict(document.get("sample_metadata") or {}).items():
            widget = getattr(self, f"meta_{key}", None)
            if widget is not None: widget.setText(str(value))
        self._refresh_mask_list(); self._refresh_analysis_list(); self._refresh_processed()
        view = dict(document.get("view") or {})
        if view.get("xlim") and view.get("ylim"):
            self.ax_main.set_xlim(*view["xlim"]); self.ax_main.set_ylim(*view["ylim"]); self.canvas.draw_idle()
        self.settings.setValue("session/last_session", path)
        self.statusBar().showMessage("Session restored" + ("; source file has changed" if changed else ""), 6000)

    def on_load_preset(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load analysis preset", "", "Preset JSON (*.json)")
        if not p:
            return
        try:
            payload = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as exc:
            self._message("Preset", f"Failed to load preset:\n{exc}")
            return
        self.profile_combo.setCurrentText(str(payload.get("profile", self.profile_combo.currentText())))
        self.mode_combo.setCurrentText(str(payload.get("mode", self.mode_combo.currentText())))
        self.model_combo.setCurrentText(str(payload.get("model", self.model_combo.currentText())))
        self.rabi_mode_combo.setCurrentText(str(payload.get("rabi_mode", self.rabi_mode_combo.currentText())))
        self.t1_prep_combo.setCurrentText(str(payload.get("t1_prep_mode", self.t1_prep_combo.currentText())))
        self.t1_model_combo.setCurrentText(str(payload.get("t1_model_family", self.t1_model_combo.currentText())))
        self.roi_min.setText(str(payload.get("roi_min", self.roi_min.text())))
        self.roi_max.setText(str(payload.get("roi_max", self.roi_max.text())))
        self.bin_spin.setValue(int(payload.get("bin_size", self.bin_spin.value())))
        self.smooth_spin.setValue(int(payload.get("smooth_window", self.smooth_spin.value())))
        odmr = payload.get("odmr", {})
        display = str(odmr.get("display", self.odmr_display_combo.currentText()))
        if display == "Peaks (inverted view)":
            display = "Dips (inverted view)"
        elif display == "Dips (baseline view)":
            display = "Peaks (measured view)"
        self.odmr_display_combo.setCurrentText(display)
        self.odmr_polarity_combo.setCurrentText(str(odmr.get("polarity", self.odmr_polarity_combo.currentText())))
        self.odmr_ranges.setText(str(odmr.get("ranges", self.odmr_ranges.text())))
        self.odmr_peak_pick_only.setChecked(bool(odmr.get("peak_pick_only", self.odmr_peak_pick_only.isChecked())))
        self.odmr_multipeak_fit_chk.setChecked(bool(odmr.get("multipeak_fit", self.odmr_multipeak_fit_chk.isChecked())))
        self.odmr_peak_count_spin.setValue(int(odmr.get("peak_count", self.odmr_peak_count_spin.value())))
        self.odmr_auto_peak_count_chk.setChecked(bool(odmr.get("auto_peak_count", self.odmr_auto_peak_count_chk.isChecked())))
        self.odmr_cal_scale.setValue(float(odmr.get("cal_scale", self.odmr_cal_scale.value())))
        self.odmr_cal_offset.setValue(float(odmr.get("cal_offset", self.odmr_cal_offset.value())))
        plot_opts = payload.get("plot_options", {})
        for k, v in plot_opts.items():
            if hasattr(self.plot_opts, k):
                setattr(self.plot_opts, k, v)
        self.fig_w_spin.setValue(float(self.plot_opts.fig_width))
        self.fig_h_spin.setValue(float(self.plot_opts.fig_height))
        self.save_dpi_spin.setValue(int(self.plot_opts.save_dpi))
        self.legend_font_spin.setValue(int(self.plot_opts.legend_font_size))
        self.show_legend_chk.setChecked(bool(self.plot_opts.show_legend))
        self.show_residual_chk.setChecked(bool(self.plot_opts.show_residual))
        self.show_data_chk.setChecked(bool(self.plot_opts.show_data))
        self.show_smoothed_chk.setChecked(bool(self.plot_opts.show_smoothed))
        self.show_fit_chk.setChecked(bool(self.plot_opts.show_fit))
        self.show_rabi_envelope_chk.setChecked(bool(getattr(self.plot_opts, "show_rabi_envelope", True)))
        self.show_peaks_chk.setChecked(bool(self.plot_opts.show_peaks))
        self.show_signal_layer_chk.setChecked(bool(self.plot_opts.show_signal_layer))
        self.show_reference_layer_chk.setChecked(bool(self.plot_opts.show_reference_layer))
        self.show_difference_layer_chk.setChecked(bool(self.plot_opts.show_difference_layer))
        self.show_iteration_layer_chk.setChecked(bool(self.plot_opts.show_iteration_layer))
        self.hide_failed_iterations_chk.setChecked(bool(self.plot_opts.hide_failed_iterations))
        self.ann_t2rho_chk.setChecked(bool(getattr(self.plot_opts, "annotation_show_t2rho", True)))
        self.legend_loc_combo.setCurrentText(str(self.plot_opts.legend_loc))
        self.ctx.active_preset_name = str(payload.get("preset_name", Path(p).stem))
        if self.ctx.custom_model_spec is not None:
            self.equation_display.setPlainText(str(self.ctx.custom_model_spec.get("expression", "")))
        self._update_contextual_visibility()
        self._refresh_processed()

    def on_load_custom_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load custom model spec", "", "JSON (*.json)")
        if not p:
            return
        try:
            spec = json.loads(Path(p).read_text(encoding="utf-8"))
            required = {"name", "expression", "param_names", "lower", "upper", "seed"}
            if not required.issubset(set(spec.keys())):
                raise ValueError(f"Missing keys: {sorted(required - set(spec.keys()))}")
            self.ctx.custom_model_spec = spec
            if "CustomModel" not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                self.model_combo.addItem("CustomModel")
            self.model_combo.setCurrentText("CustomModel")
            self._update_contextual_visibility()
            self._update_equation_display()
            self._message("Custom model", f"Loaded custom model: {spec['name']}")
        except Exception as exc:
            self._message("Custom model", f"Failed to load:\n{exc}")

    def _active_profile(self) -> FitProfile:
        p = self.profile_combo.currentText()
        if p == "Auto":
            if self.ctx.trace is None:
                return PROFILES["LineScan"]
            exp = self.ctx.trace.experiment_type
            if exp == "LineScan" and "t1" in self.ctx.trace.file_name.lower():
                exp = "T1"
            return PROFILES.get(exp if exp in PROFILES else "LineScan")
        return PROFILES.get(p if p in PROFILES else "LineScan")

    def apply_profile_defaults(self, *_args, reload_data: bool = True):
        prof = self._active_profile()
        self.ctx.profile = prof
        recommended_mode = self._recommended_mode_for_profile(prof)
        if not self._mode_override_active:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(recommended_mode)
            self.mode_combo.blockSignals(False)
        self.bin_spin.setValue(prof.bin_size)
        self.smooth_spin.setValue(prof.smooth_window)
        if self.ctx.trace is not None:
            model = prof.candidate_models[0]
            if model in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                self.model_combo.setCurrentText(model)
        self._update_contextual_visibility()
        self._update_profile_hint()
        if reload_data:
            self._reload_current_file()

    def on_load(self):
        start_dir = str(self.settings.value("session/last_open_dir", ""))
        file_path, _ = QFileDialog.getOpenFileName(self, "Load MAT File", start_dir, "MAT files (*.mat)")
        if not file_path:
            return
        self.settings.setValue("session/last_open_dir", str(Path(file_path).parent))
        self._load_file(file_path)

    def _load_file(self, file_path: str, *, preserve_analysis: bool = False):
        if self._is_loading:
            return False
        preserve_mode_override = self._mode_override_active
        self._is_loading = True
        try:
            mode: DataMode = self._current_data_mode()  # type: ignore[assignment]
            trace = load_saved_data_mat(file_path, mode=mode)
        except Exception as exc:
            self._is_loading = False
            self._message("Load error", str(exc))
            return False
        if len(trace.x_ns) == 0 or len(trace.y) == 0:
            self._is_loading = False
            self._message("Load error", f"Loaded trace is empty after applying observable mode '{self._friendly_mode_name(mode)}'.")
            return False
        self.ctx.trace = trace
        self.ctx.fit_result = None
        self.ctx.fit_target = None
        self.ctx.odmr_peaks = None
        self._selected_plot_point = None
        self._scan_cursor = None
        if hasattr(self, "clear_marker_btn"):
            self.clear_marker_btn.setEnabled(False)
        if not preserve_analysis:
            self.ctx.excluded_points = set()
            self.ctx.exclusion_ranges = []
            self.ctx.analysis_steps = []
            self._analysis_undo_stack.clear()
            self._analysis_redo_stack.clear()
        if self.ctx.loaded_traces is None:
            self.ctx.loaded_traces = {}
        self.ctx.loaded_traces[trace.file_name] = trace
        self._update_overlay_widgets()
        self._update_recent_files(str(Path(file_path).resolve()))
        self.roi_min.setText(f"{np.min(trace.x_ns):.6g}")
        self.roi_max.setText(f"{np.max(trace.x_ns):.6g}")

        exp_guess = trace.experiment_type
        if exp_guess == "LineScan" and "t1" in trace.file_name.lower():
            exp_guess = "T1"
        if self.profile_combo.currentText() == "Auto":
            self.profile_combo.blockSignals(True)
            self.profile_combo.setCurrentText(exp_guess if exp_guess in PROFILES else "LineScan")
            self.profile_combo.blockSignals(False)
        self._mode_override_active = preserve_mode_override
        self.apply_profile_defaults(reload_data=False)
        self._update_contextual_visibility()
        self._refresh_processed()
        self.rabi_dead_time_ns.setValue(float((trace.metadata or {}).get("rabi_dead_time_ns", 0.0) or 0.0))
        self._update_iteration_controls()
        self._update_profile_hint()
        self._show_loaded_trace_summary()
        self._is_loading = False
        return True

    def _reload_current_file(self):
        if self.ctx.trace is None or self._is_loading:
            return
        path = Path(self.ctx.trace.source_path)
        if path.exists():
            preserve_mode_override = self._mode_override_active
            if self._load_file(str(path), preserve_analysis=True):
                self._mode_override_active = preserve_mode_override
                self._update_profile_hint()
        else:
            self._refresh_processed()

    def _parse_roi(self) -> tuple[float | None, float | None]:
        def parse_one(s: str) -> float | None:
            s = s.strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None

        return parse_one(self.roi_min.text()), parse_one(self.roi_max.text())

    def _parse_odmr_ranges(self) -> list[tuple[float, float]]:
        text = self.odmr_ranges.text().strip()
        if not text:
            return []
        out: list[tuple[float, float]] = []
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        range_pattern = re.compile(rf"^\s*({number})\s*(?:-|:|\bto\b)\s*({number})\s*$", re.IGNORECASE)
        for token in text.split(","):
            match = range_pattern.fullmatch(token)
            if match is None:
                continue
            lo, hi = float(match.group(1)), float(match.group(2))
            if hi < lo:
                lo, hi = hi, lo
            out.append((lo, hi))
        return out

    def _odmr_calibrate(self, xval: float) -> float:
        return float(self.odmr_cal_offset.value()) + float(self.odmr_cal_scale.value()) * float(xval)

    def _display_y(self, y: np.ndarray) -> tuple[np.ndarray, str]:
        if self.ctx.trace is not None and self.ctx.trace.experiment_type in {"ODMR", "DEERFrequency"}:
            if self.odmr_display_combo.currentText().startswith("Dips"):
                return -y, "ODMR signal [a.u.] (dips)"
            return y, "ODMR signal [a.u.] (peaks)"
        if self.ctx.trace is None:
            return y, "Signal"
        mode = self._friendly_mode_name(self._current_data_mode()) if hasattr(self, "mode_combo") else self.ctx.trace.mode
        if mode == "contrast":
            return y, "Contrast [a.u.]"
        if mode == "signal":
            return y, "Signal [counts]"
        if mode == "reference":
            return y, "Reference [counts]"
        if mode == "difference":
            return y, "Difference [counts]"
        return y, (self.ctx.trace.y_label or "Signal")

    def _series_for_mode(self, trace: ExperimentTrace, mode: str) -> tuple[np.ndarray, str]:
        mode_key = mode.strip().lower()
        if mode_key == "contrast":
            ref = trace.reference.astype(float, copy=True)
            ref[ref == 0] = np.nan
            return (trace.reference - trace.signal) / ref, "Contrast [a.u.]"
        if mode_key == "signal":
            return trace.signal.astype(float, copy=True), "Signal [counts]"
        if mode_key == "reference":
            return trace.reference.astype(float, copy=True), "Reference [counts]"
        return (trace.reference - trace.signal).astype(float, copy=True), "Difference [counts]"

    def _iteration_source_for_mode(self, trace: ExperimentTrace, mode: str) -> tuple[np.ndarray | None, str]:
        mode_key = mode.strip().lower()
        source: np.ndarray | None
        if mode_key == "signal":
            source = trace.iteration_signal
            label = "Signal iterations [counts]"
        elif mode_key == "reference":
            source = trace.iteration_reference
            label = "Reference iterations [counts]"
        elif mode_key == "difference":
            if trace.iteration_reference is None or trace.iteration_signal is None:
                return None, ""
            source = trace.iteration_reference - trace.iteration_signal
            label = "Difference iterations [counts]"
        else:
            source = trace.iteration_contrast
            label = "Contrast iterations [a.u.]"
        if source is None:
            return None, ""
        matrix = np.asarray(source, dtype=float).copy()
        if self.plot_opts.hide_failed_iterations and trace.iteration_valid is not None and trace.iteration_valid.shape == matrix.shape:
            matrix[~trace.iteration_valid] = np.nan
        return matrix, label

    def _iteration_series_for_mode(self, trace: ExperimentTrace, mode: str, iteration_index: int | None) -> tuple[np.ndarray | None, str]:
        if iteration_index is None:
            return None, ""
        matrix, label = self._iteration_source_for_mode(trace, mode)
        if matrix is None or iteration_index >= matrix.shape[1]:
            return None, ""
        return np.asarray(matrix[:, iteration_index], dtype=float), label

    def _processed_trace_series(self, trace: ExperimentTrace, y_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        processed = process_series(
            trace.x_ns,
            np.asarray(y_full, dtype=float),
            bin_size=self.bin_spin.value(),
            roi=self._parse_roi(),
            excluded_raw_indices=self.ctx.excluded_points,
            exclusion_ranges=self.ctx.exclusion_ranges,
        )
        return processed.fit_x, processed.fit_y

    def _selected_iteration_matrix(self, trace: ExperimentTrace, mode: str) -> tuple[np.ndarray, np.ndarray, list[int], str]:
        matrix, label = self._iteration_source_for_mode(trace, mode)
        if matrix is None or matrix.ndim != 2 or matrix.shape[1] == 0:
            return np.array([]), np.empty((0, 0)), [], ""
        count = matrix.shape[1]
        selected = sorted(idx for idx in self._selected_iteration_indices if 0 <= idx < count)
        cols: list[np.ndarray] = []
        x_ref: np.ndarray | None = None
        kept: list[int] = []
        for idx in selected:
            ix, iy = self._processed_trace_series(trace, matrix[:, idx])
            if len(ix) == 0:
                continue
            if x_ref is None:
                x_ref = ix
            elif len(ix) != len(x_ref) or not np.allclose(ix, x_ref, equal_nan=True):
                iy = np.interp(x_ref, ix, iy, left=np.nan, right=np.nan)
                ix = x_ref
            cols.append(np.asarray(iy, dtype=float))
            kept.append(idx)
        if x_ref is None or not cols:
            return np.array([]), np.empty((0, 0)), [], label
        return x_ref, np.column_stack(cols), kept, label

    def _add_observable_layers(self, primary_series: list[np.ndarray]):
        if self.ctx.trace is None or self.ctx.x is None:
            return
        trace = self.ctx.trace
        active_mode = self._friendly_mode_name(self._current_data_mode())
        layer_specs = [
            ("signal", self.plot_opts.show_signal_layer, "#7aa2f7"),
            ("reference", self.plot_opts.show_reference_layer, "#f6bd60"),
            ("difference", self.plot_opts.show_difference_layer, "#9ece6a"),
        ]
        secondary_layers: list[tuple[np.ndarray, np.ndarray, str, str]] = []
        for mode, enabled, color in layer_specs:
            if not enabled:
                continue
            y_full, label = self._series_for_mode(trace, mode)
            lx, ly = self._processed_trace_series(trace, y_full)
            if len(lx) == 0:
                continue
            needs_secondary = mode in {"signal", "reference"} or (mode == "difference" and active_mode not in {"signal", "reference", "difference"})
            if needs_secondary:
                secondary_layers.append((lx, ly, label, color))
            else:
                primary_series.append(np.asarray(ly, dtype=float))
                self._plot_series(self.ax_main, lx, ly, self._legend_label(label, mode.title()), role="overlay", color=color, alpha=0.65)
        if secondary_layers:
            if self._ax_secondary is None:
                self._ax_secondary = self.ax_main.twinx()
            for lx, ly, label, color in secondary_layers:
                self._secondary_series.append(np.asarray(ly, dtype=float))
                self._plot_series(self._ax_secondary, lx, ly, self._legend_label(label, label.split()[0]), role="overlay", color=color, alpha=0.7)
            labels = [label for _, _, label, _ in secondary_layers]
            ylabel = "Counts" if any("counts" in label.lower() for label in labels) else labels[0]
            self._ax_secondary.set_ylabel(ylabel)
            self._ax_secondary.grid(False)
            self._ax_secondary.set_autoscalex_on(False)
            self._ax_secondary.set_xlim(self.ax_main.get_xlim())
            self._set_axis_ylim_from_series(self._ax_secondary, self._secondary_series)
            self._secondary_plot_data = (secondary_layers[0][0], secondary_layers[0][1], secondary_layers[0][2])

    def _add_iteration_layer(self, primary_series: list[np.ndarray]):
        if self.ctx.trace is None or self.ctx.x is None:
            return
        mode = self._friendly_mode_name(self._current_data_mode())
        ix, matrix, indices, label = self._selected_iteration_matrix(self.ctx.trace, mode)
        if len(ix) == 0 or matrix.size == 0:
            return
        if self.plot_opts.show_iteration_layer:
            for col, iter_idx in enumerate(indices):
                iy = matrix[:, col]
                primary_series.append(np.asarray(iy, dtype=float))
                label_text = f"Iteration {iter_idx + 1}" if col < 5 else "_nolegend_"
                short_text = f"Iter {iter_idx + 1}" if col < 5 else "_nolegend_"
                self._plot_series(
                    self.ax_main,
                    ix,
                    iy,
                    self._legend_label(label_text, short_text),
                    role="overlay",
                    color="#c678dd",
                    alpha=0.32,
                )
        if self.plot_opts.show_iteration_mean or self.plot_opts.show_iteration_std:
            finite_counts = np.sum(np.isfinite(matrix), axis=1)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(matrix, axis=1)
            mean[finite_counts == 0] = np.nan
            primary_series.append(np.asarray(mean, dtype=float))
            if self.plot_opts.show_iteration_mean:
                self._plot_series(
                    self.ax_main,
                    ix,
                    mean,
                    self._legend_label("Iteration mean", "Iter mean"),
                    role="overlay",
                    color="#e0af68",
                    alpha=0.9,
                )
            if self.plot_opts.show_iteration_std:
                with np.errstate(invalid="ignore"):
                    std = np.nanstd(matrix, axis=1, ddof=1 if matrix.shape[1] > 1 else 0)
                std[finite_counts == 0] = np.nan
                if self.plot_opts.iteration_std_mode == "Error bars":
                    self.ax_main.errorbar(
                        ix,
                        mean,
                        yerr=std,
                        fmt="none",
                        ecolor="#c678dd",
                        elinewidth=0.8,
                        capsize=2,
                        alpha=0.45,
                        label=self._legend_label("Iteration std", "Iter std"),
                    )
                else:
                    lo = mean - std
                    hi = mean + std
                    primary_series.extend([np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)])
                    self.ax_main.fill_between(
                        ix,
                        lo,
                        hi,
                        color="#c678dd",
                        alpha=0.16,
                        label=self._legend_label("Iteration +/- std", "Iter +/- std"),
                    )

    def _friendly_axis_label(self, axis_label: str | None) -> str:
        text = (axis_label or "").strip().lower()
        mapping = {
            "duration": "Pulse duration [ns]",
            "pulse duration (ns)": "Pulse duration [ns]",
            "frequency": "RF frequency [GHz]",
            "rf frequency": "RF frequency [GHz]",
            "stage x": "Stage X [um]",
            "stage y": "Stage Y [um]",
            "stage z": "Stage Z [um]",
        }
        return mapping.get(text, axis_label or "X")

    def _metric_display_label(self, key: str) -> str:
        labels = {
            "r2": "R^2",
            "rmse": "RMSE",
            "durbin_watson": "Durbin-Watson",
            "rabi_freq_MHz": "Rabi frequency [MHz]",
            "rabi_period_ns": "Rabi period [ns]",
            "pi_time_ns": "Pi time [ns]",
            "pi_over_2_time_ns": "Pi/2 time [ns]",
            "pi_time_ns_nominal": "Nominal pi time [ns]",
            "pi_over_2_time_ns_nominal": "Nominal pi/2 time [ns]",
            "rabi_dead_time_ns": "Derived delay [ns]",
            "first_peak_ns": "First peak [ns]",
            "delay_ns": "Delay [ns]",
            "tau_ramp_ns": "Ramp timescale [ns]",
            "T2rho_ns": "T2rho [ns]",
            "rabi_envelope_decay_ns": "Rabi envelope decay [ns]",
            "rabi_envelope_beta": "Rabi envelope beta",
            "rabi_envelope_amp": "Rabi envelope amplitude",
            "rabi_envelope_floor": "Rabi envelope floor",
            "rabi_envelope_r2": "Rabi envelope R^2",
            "rabi_envelope_rmse": "Rabi envelope RMSE",
            "center_freq": "Center frequency [GHz]",
            "fwhm": "FWHM [axis units]",
            "t0_ns": "Delay [ns]",
            "decay_ns": "Decay [ns]",
            "T1_ns": "T1 [ns]",
        }
        return labels.get(key, key.replace("_", " "))

    def _format_value_with_units(self, key: str, value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        if abs(value) >= 1e4 or abs(value) < 1e-3:
            return f"{value:.3g}"
        return f"{value:.4g}"

    def _friendly_param_name(self, name: str) -> str:
        labels = {
            "y0": "Baseline",
            "A": "Amplitude",
            "A1": "Amplitude 1",
            "A2": "Amplitude 2",
            "C": "Peak or dip amplitude",
            "f": "Frequency [GHz]",
            "f1": "Frequency 1 [GHz]",
            "f2": "Frequency 2 [GHz]",
            "phi": "Phase [rad]",
            "phi1": "Phase 1 [rad]",
            "phi2": "Phase 2 [rad]",
            "tau": "Decay [ns]",
            "T": "Envelope decay [ns]",
            "t0": "Delay [ns]",
            "tau_ramp": "Ramp timescale [ns]",
            "beta": "Envelope stretch exponent",
            "alpha": "Chirp [1/ns²]",
            "T2": "T2 [ns]",
            "T1": "T1 [ns]",
            "T2_star": "T2* [ns]",
            "n": "Stretch exponent",
            "w": "Width [axis units]",
            "x0": "Center [axis units]",
        }
        return labels.get(name, name)

    def _plot_style_for(self, role: str) -> tuple[str, dict[str, object]]:
        style = self.plot_opts.plot_style
        if role in {"fit", "smoothed", "overlay", "linecut"}:
            return "-", {"lw": 2.0}
        if style == "Scatter":
            return "o", {"ms": 4, "linestyle": "None", "alpha": 0.65}
        if style == "Line":
            return "-", {"lw": 1.8, "alpha": 0.9}
        return "o-", {"ms": 3.6, "lw": 1.2, "alpha": 0.78}

    def _plot_series(self, ax, x: np.ndarray, y: np.ndarray, label: str, role: str = "data", **kwargs):
        fmt, defaults = self._plot_style_for(role)
        plot_kwargs = {**defaults, **kwargs}
        ax.plot(x, y, fmt, label=label, **plot_kwargs)

    def _plot_rabi_envelope(self, ax, x: np.ndarray, fit_result: FitResult, primary_series: list[np.ndarray] | None = None):
        if not self.plot_opts.show_rabi_envelope or not self.plot_opts.show_fit:
            return
        if not isinstance(fit_result.extras, dict):
            return
        bounds = rabi_envelope_bounds(x, fit_result.extras.get("rabi_envelope"))
        if bounds is None:
            return
        upper, lower = bounds
        upper_disp, _ = self._display_y(np.asarray(upper, dtype=float))
        lower_disp, _ = self._display_y(np.asarray(lower, dtype=float))
        if primary_series is not None:
            primary_series.extend([np.asarray(upper_disp, dtype=float), np.asarray(lower_disp, dtype=float)])
        ax.plot(x, upper_disp, "--", lw=1.5, color="#b57edc", alpha=0.95, label=self._legend_label("Rabi envelope", "Envelope"))
        ax.plot(x, lower_disp, "--", lw=1.5, color="#b57edc", alpha=0.95, label="_nolegend_")

    def _annotation_lines_for_result(self, fit_result: FitResult) -> list[str]:
        lines: list[str] = []
        if self.plot_opts.annotation_show_file_model and self.ctx.trace is not None:
            lines.append(self.ctx.trace.file_name)
            lines.append(f"Analysis mode: {fit_result.model_name}")
        if self.plot_opts.annotation_show_fit_metrics:
            lines.append(f"R^2: {fit_result.r2:.4f}")
            lines.append(f"RMSE: {fit_result.rmse:.4g}")
            lines.append(f"Durbin-Watson: {fit_result.durbin_watson:.3f}")
        nv = _nv_metrics(self._active_profile().name, fit_result, self._effective_trace_metadata())
        if self.plot_opts.annotation_show_t2rho and "T2rho_ns" in nv:
            lines.append(f"{self._metric_display_label('T2rho_ns')}: {self._format_value_with_units('T2rho_ns', nv['T2rho_ns'])}")
        if self.plot_opts.annotation_show_nv_metrics:
            priority = {
                "rabi_freq_MHz": 1,
                "pi_time_ns": 2,
                "pi_over_2_time_ns": 3,
                "first_peak_ns": 4,
                "delay_ns": 5,
                "rabi_envelope_beta": 6,
                "rabi_envelope_r2": 7,
            }
            separate_t2rho_keys = {"T2rho_ns", "rabi_envelope_decay_ns"}
            items = [(key, value) for key, value in nv.items() if key not in separate_t2rho_keys]
            items.sort(key=lambda item: priority.get(item[0], 99))
            for key, value in items[: (4 if self.plot_opts.annotation_mode == "Compact" else 8)]:
                lines.append(f"{self._metric_display_label(key)}: {self._format_value_with_units(key, value)}")
        if self.plot_opts.annotation_show_params:
            max_params = 3 if self.plot_opts.annotation_mode == "Compact" else 6
            pairs = list(zip(fit_result.param_names, fit_result.params))
            priority = {"t0": 0, "f": 1, "tau": 2, "A": 3, "x0": 4, "w": 5}
            pairs.sort(key=lambda item: priority.get(item[0], 99))
            for name, value in pairs[:max_params]:
                lines.append(f"{self._friendly_param_name(name)}: {self._format_value_with_units(name, float(value))}")
        if fit_result.selection_note and self.plot_opts.annotation_mode == "Detailed":
            lines.append(f"Selection: {fit_result.selection_note}")
        return lines

    def _update_point_readout(self, text: str):
        if hasattr(self, "point_readout_lbl"):
            self.point_readout_lbl.setText(text)
        if hasattr(self, "live_readout_lbl"):
            self.live_readout_lbl.setText(text)

    def _capture_view_state(self) -> tuple[float, float] | None:
        if not self.ax_main.has_data():
            return None
        return self.ax_main.get_xlim()

    def _view_state_fits_current_x_data(self, state: tuple[float, float]) -> bool:
        """Return whether an old view still belongs to the processed trace."""
        if self.ctx.x is None:
            return False
        finite_x = np.asarray(self.ctx.x, dtype=float)
        finite_x = finite_x[np.isfinite(finite_x)]
        if finite_x.size == 0:
            return False
        data_lo = float(np.min(finite_x))
        data_hi = float(np.max(finite_x))
        data_span = data_hi - data_lo
        if data_span <= 0.0:
            pad = max(abs(data_lo) * 0.05, 0.5)
        else:
            # Matplotlib's default x margin is 5%. Permit that normal
            # autoscaled view, but reject a range left over from a larger ROI
            # or a previously displayed trace.
            pad = 0.051 * data_span
        state_lo, state_hi = sorted((float(state[0]), float(state[1])))
        tolerance = max(np.finfo(float).eps * max(abs(data_lo), abs(data_hi), 1.0) * 16.0, 1e-12)
        return state_lo >= data_lo - pad - tolerance and state_hi <= data_hi + pad + tolerance

    def _restore_view_state(self, state: tuple[float, float] | None):
        if state is None or not self._view_state_fits_current_x_data(state):
            return
        try:
            self.ax_main.set_xlim(state)
            if self._ax_secondary is not None:
                self._ax_secondary.set_xlim(state)
        except Exception:
            pass

    def _reset_export_navigation_history(self):
        """Discard toolbar views whose limits predate the rebuilt axes."""
        try:
            self.toolbar.update()
        except Exception:
            pass

    def _nearest_trace_point(self, xdata: float) -> tuple[int, float, float] | None:
        x = self.ctx.analysis_x if self.ctx.analysis_x is not None else self.ctx.x
        y = self.ctx.analysis_y if self.ctx.analysis_y is not None else self.ctx.y
        if x is None or y is None or len(x) == 0:
            return None
        idx = int(np.argmin(np.abs(x - float(xdata))))
        return idx, float(x[idx]), float(y[idx])

    def _raw_index_for_processed_x(self, x_value: float) -> int | None:
        if self.ctx.trace is None or len(self.ctx.trace.x_ns) == 0:
            return None
        return int(np.argmin(np.abs(self.ctx.trace.x_ns - float(x_value))))

    def _point_iteration_readout(self, x_value: float) -> str:
        if self.ctx.trace is None:
            return ""
        raw_idx = self._raw_index_for_processed_x(x_value)
        if raw_idx is None:
            return ""
        trace = self.ctx.trace
        parts: list[str] = []
        if raw_idx < len(trace.signal):
            parts.append(f"signal={self._format_inspect_value(float(trace.signal[raw_idx]))}")
        if raw_idx < len(trace.reference):
            parts.append(f"reference={self._format_inspect_value(float(trace.reference[raw_idx]))}")
        if trace.y_sem is not None and raw_idx < len(trace.y_sem):
            parts.append(f"SEM={self._format_inspect_value(float(trace.y_sem[raw_idx]))}")
        if trace.iteration_valid is not None and raw_idx < trace.iteration_valid.shape[0]:
            valid_count = int(np.sum(trace.iteration_valid[raw_idx]))
            parts.append(f"valid_iter={valid_count}/{trace.iteration_valid.shape[1]}")
        matrix, _ = self._iteration_source_for_mode(trace, self._friendly_mode_name(self._current_data_mode()))
        if matrix is not None and raw_idx < matrix.shape[0]:
            selected = sorted(idx for idx in self._selected_iteration_indices if 0 <= idx < matrix.shape[1])
            if 0 < len(selected) <= 4:
                for iter_idx in selected:
                    value = matrix[raw_idx, iter_idx]
                    if np.isfinite(value):
                        parts.append(f"iter{iter_idx + 1}={self._format_inspect_value(float(value))}")
            elif len(selected) > 4:
                values = matrix[raw_idx, selected]
                finite = values[np.isfinite(values)]
                if len(finite):
                    parts.append(
                        f"selected_iter={len(finite)}/{len(selected)}, "
                        f"mean={self._format_inspect_value(float(np.nanmean(finite)))}, "
                        f"std={self._format_inspect_value(float(np.nanstd(finite, ddof=1 if len(finite) > 1 else 0)))}"
                    )
        return ", ".join(parts)

    def _nearest_secondary_point(self, xdata: float) -> tuple[int, float, float, str] | None:
        if self._secondary_plot_data is None:
            return None
        sec_x, sec_y, sec_label = self._secondary_plot_data
        if len(sec_x) == 0:
            return None
        idx = int(np.argmin(np.abs(sec_x - float(xdata))))
        return idx, float(sec_x[idx]), float(sec_y[idx]), sec_label

    def _format_inspect_value(self, value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        if abs(value) >= 100:
            return str(int(round(value)))
        return f"{value:.6f}"

    def _inspectable_axes(self) -> set:
        axes = {self.ax_main}
        if self._ax_secondary is not None:
            axes.add(self._ax_secondary)
        return axes

    def _ensure_crosshairs(self, x: float, y: float):
        if self._crosshair_v is None:
            self._crosshair_v = self.ax_main.axvline(x, color="#90caf9", lw=0.9, ls="--", alpha=0.7, visible=False)
        if self._crosshair_h is None:
            self._crosshair_h = self.ax_main.axhline(y, color="#90caf9", lw=0.9, ls="--", alpha=0.7, visible=False)

    def _clear_crosshairs(self):
        if self._crosshair_v is not None:
            self._crosshair_v.set_visible(False)
        if self._crosshair_h is not None:
            self._crosshair_h.set_visible(False)
        self.canvas.draw_idle()

    def _scan_best_points(self) -> dict[str, tuple[float, ...]]:
        if self.ctx.trace is None:
            return {}
        if self.ctx.trace.scan_dim == "scan2d" and self.ctx.trace.z2d is not None:
            z = np.asarray(self.ctx.trace.z2d, dtype=float)
            max_idx = np.unravel_index(int(np.nanargmax(z)), z.shape)
            min_idx = np.unravel_index(int(np.nanargmin(z)), z.shape)
            return {
                "max": (float(self.ctx.trace.x2d[max_idx]), float(self.ctx.trace.y2d[max_idx]), float(z[max_idx])),
                "min": (float(self.ctx.trace.x2d[min_idx]), float(self.ctx.trace.y2d[min_idx]), float(z[min_idx])),
            }
        if self.ctx.trace.scan_dim == "scan1d" and self.ctx.x is not None and self.ctx.y is not None and len(self.ctx.x):
            max_idx = int(np.nanargmax(self.ctx.y))
            min_idx = int(np.nanargmin(self.ctx.y))
            return {
                "max": (float(self.ctx.x[max_idx]), float(self.ctx.y[max_idx])),
                "min": (float(self.ctx.x[min_idx]), float(self.ctx.y[min_idx])),
            }
        return {}

    def _measure_width(self, x: np.ndarray, y: np.ndarray, peak_idx: int, peak_kind: str) -> float | None:
        if len(x) < 3 or not (0 <= peak_idx < len(x)):
            return None
        if peak_kind == "max":
            level = 0.5 * (float(np.max(y)) + float(np.min(y)))
            mask = y >= level
        else:
            level = 0.5 * (float(np.max(y)) + float(np.min(y)))
            mask = y <= level
        idx = np.where(mask)[0]
        if len(idx) < 2:
            return None
        return float(x[idx[-1]] - x[idx[0]])

    def _update_scan_summary(self):
        if self.ctx.trace is None or not self._trace_is_scan():
            self.scan_summary_text.setPlainText("")
            self.scan_cursor_lbl.setText("Cursor: none")
            return
        trace = self.ctx.trace
        points = self._scan_best_points()
        lines: list[str] = [f"Observable: {self.mode_combo.currentText()}"]
        if trace.scan_dim == "scan1d":
            secondary = self.scan_secondary_combo.currentText().strip().lower()
            current = self.mode_combo.currentText().strip().lower()
            if secondary and secondary != "none" and secondary != current:
                lines.append(f"Secondary axis: {secondary}")
        if "max" in points:
            if trace.scan_dim == "scan2d":
                lines.append(f"Max point: x={points['max'][0]:.6g}, y={points['max'][1]:.6g}, z={points['max'][2]:.6g}")
                lines.append(f"Min point: x={points['min'][0]:.6g}, y={points['min'][1]:.6g}, z={points['min'][2]:.6g}")
                if self._scan_cursor is None:
                    self.scan_cursor_lbl.setText("Cursor: click the map in Inspect point mode to place the crosshair.")
            else:
                x = self.ctx.x if self.ctx.x is not None else np.array([])
                y = self.ctx.y if self.ctx.y is not None else np.array([])
                if len(x):
                    max_idx = int(np.nanargmax(y))
                    min_idx = int(np.nanargmin(y))
                    w_max = self._measure_width(x, y, max_idx, "max")
                    w_min = self._measure_width(x, y, min_idx, "min")
                    lines.append(f"Peak position: x={points['max'][0]:.6g}, y={points['max'][1]:.6g}")
                    lines.append(f"Dip position: x={points['min'][0]:.6g}, y={points['min'][1]:.6g}")
                    if self._selected_plot_point is None:
                        self.scan_cursor_lbl.setText(f"Cursor: best point x={points['max'][0]:.6g}, y={points['max'][1]:.6g} (click in Inspect point mode to inspect another point)")
                    if w_max is not None:
                        lines.append(f"Peak width: {w_max:.6g}")
                    if w_min is not None:
                        lines.append(f"Dip width: {w_min:.6g}")
        self.scan_summary_text.setPlainText("\n".join(lines))

    def _extract_scan_linecut(self) -> tuple[np.ndarray, np.ndarray, str] | None:
        if self.ctx.trace is None or self.ctx.trace.scan_dim != "scan2d" or self.ctx.trace.z2d is None:
            return None
        x2d = np.asarray(self.ctx.trace.x2d, dtype=float)
        y2d = np.asarray(self.ctx.trace.y2d, dtype=float)
        z2d = np.asarray(self.ctx.trace.z2d, dtype=float)
        mode = self.scan_linecut_combo.currentText()
        if mode.strip().lower() == "none":
            return None
        points = self._scan_best_points()
        if self._scan_cursor is None:
            self._scan_cursor = (points.get("max", (float(x2d[0, 0]), float(y2d[0, 0]), 0.0))[0], points.get("max", (0.0, float(y2d[0, 0]), 0.0))[1])
        cx, cy = self._scan_cursor
        if mode.startswith("Best-point") and "max" in points:
            cx, cy = points["max"][0], points["max"][1]
        row = int(np.argmin(np.abs(y2d[:, 0] - cy)))
        col = int(np.argmin(np.abs(x2d[0, :] - cx)))
        if "horizontal" in mode.lower():
            return x2d[row, :], z2d[row, :], f"Horizontal linecut @ y={y2d[row, 0]:.6g}"
        return y2d[:, col], z2d[:, col], f"Vertical linecut @ x={x2d[0, col]:.6g}"

    def _on_plot_motion(self, event):
        if self.mask_mode_combo.currentText() != "Inspect point" or event.inaxes not in self._inspectable_axes():
            return
        if event.xdata is None:
            return
        if self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan2d" and event.ydata is not None:
            self._ensure_crosshairs(float(event.xdata), float(event.ydata))
            self._crosshair_v.set_xdata([event.xdata, event.xdata])
            self._crosshair_h.set_ydata([event.ydata, event.ydata])
            self._crosshair_v.set_visible(True)
            self._crosshair_h.set_visible(True)
            z = float(self.ctx.trace.z2d[int(np.argmin(np.abs(self.ctx.trace.y2d[:, 0] - event.ydata))), int(np.argmin(np.abs(self.ctx.trace.x2d[0, :] - event.xdata)))])
            self._update_point_readout(
                f"Point readout: x={self._format_inspect_value(float(event.xdata))}, "
                f"y={self._format_inspect_value(float(event.ydata))}, "
                f"z={self._format_inspect_value(z)}"
            )
        else:
            nearest = self._nearest_trace_point(float(event.xdata))
            if nearest is None:
                return
            _, px, py = nearest
            self._ensure_crosshairs(px, py)
            self._crosshair_v.set_xdata([px, px])
            self._crosshair_h.set_ydata([py, py])
            self._crosshair_v.set_visible(True)
            self._crosshair_h.set_visible(True)
            secondary = self._nearest_secondary_point(float(event.xdata))
            if secondary is not None:
                _, sx, sy, _ = secondary
                self._update_point_readout(
                    f"Point readout: x={self._format_inspect_value(px)}, "
                    f"primary={self._format_inspect_value(py)}, "
                    f"secondary={self._format_inspect_value(sy)}"
                )
            else:
                self._update_point_readout(f"Point readout: x={self._format_inspect_value(px)}, y={self._format_inspect_value(py)}")
        self.canvas.draw_idle()

    def _on_plot_leave(self, event):
        if self.mask_mode_combo.currentText() == "Inspect point":
            self._clear_crosshairs()

    def _legend_label(self, default_label: str, short_label: str) -> str:
        if self.plot_opts.legend_compact_labels:
            return short_label
        return default_label

    def _annotation_anchor(self) -> tuple[float, float, str, str]:
        loc = self.plot_opts.annotation_loc
        mapping = {
            "upper right": (0.98, 0.98, "right", "top"),
            "upper left": (0.02, 0.98, "left", "top"),
            "lower right": (0.98, 0.02, "right", "bottom"),
            "lower left": (0.02, 0.02, "left", "bottom"),
        }
        return mapping.get(loc, mapping["upper right"])

    def _draw_plot_annotation(self, fit_result: FitResult):
        if self.plot_opts.annotation_mode == "Off":
            return
        lines = self._annotation_lines_for_result(fit_result)
        if not lines:
            return
        x, y, ha, va = self._annotation_anchor()
        self.ax_main.text(
            x,
            y,
            "\n".join(lines),
            transform=self.ax_main.transAxes,
            fontsize=self.plot_opts.annotation_font_size,
            ha=ha,
            va=va,
            color="#f5fbff",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": (0.08, 0.10, 0.15, 0.82), "edgecolor": "#93c5fd"},
        )

    def _clear_secondary_axis(self):
        self._secondary_plot_data = None
        self._secondary_series = []
        if self._ax_secondary is not None:
            try:
                self._ax_secondary.remove()
            except Exception:
                pass
            self._ax_secondary = None

    def _clear_scan_colorbar(self):
        if self._scan_colorbar is not None:
            try:
                self._scan_colorbar.remove()
            except Exception:
                pass
            self._scan_colorbar = None
        if self._scan_colorbar_ax is not None:
            try:
                self._scan_colorbar_ax.remove()
            except Exception:
                pass
            self._scan_colorbar_ax = None

    def _reset_main_axes_layout(self):
        self.ax_main.set_position(self._ax_main_default_pos)
        self.ax_main.set_aspect("auto")
        try:
            self.ax_main.set_box_aspect(None)
        except Exception:
            pass
        self.ax_res.set_position(self._ax_res_default_pos)

    def _scan2d_axis_edges(self, centers: np.ndarray) -> np.ndarray:
        values = np.asarray(centers, dtype=float).ravel()
        if values.size == 0:
            return np.array([], dtype=float)
        if values.size == 1:
            center = float(values[0])
            return np.array([center - 0.5, center + 0.5], dtype=float)
        deltas = np.diff(values)
        mids = 0.5 * (values[:-1] + values[1:])
        first = values[0] - 0.5 * deltas[0]
        last = values[-1] + 0.5 * deltas[-1]
        return np.concatenate([[first], mids, [last]]).astype(float)

    def _scan2d_extent(self, trace: ExperimentTrace) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]] | None:
        if trace.x2d is None or trace.y2d is None or trace.z2d is None:
            return None
        x2d = np.asarray(trace.x2d, dtype=float)
        y2d = np.asarray(trace.y2d, dtype=float)
        if x2d.ndim != 2 or y2d.ndim != 2 or x2d.size == 0 or y2d.size == 0:
            return None
        x_edges = self._scan2d_axis_edges(x2d[0, :])
        y_edges = self._scan2d_axis_edges(y2d[:, 0])
        if len(x_edges) < 2 or len(y_edges) < 2:
            return None
        xmin, xmax = float(np.nanmin(x_edges)), float(np.nanmax(x_edges))
        ymin, ymax = float(np.nanmin(y_edges)), float(np.nanmax(y_edges))
        return x_edges, y_edges, (xmin, xmax, ymin, ymax)

    def _fit_axes_box_to_aspect(self, rect: tuple[float, float, float, float] | list[float], aspect: float) -> list[float]:
        left, bottom, width, height = [float(v) for v in rect]
        if not np.isfinite(aspect) or aspect <= 0 or width <= 0 or height <= 0:
            return [left, bottom, width, height]
        fig_width, fig_height = self.fig.get_size_inches()
        fig_aspect = float(fig_width) / float(fig_height) if fig_height > 0 else 1.0
        normalized_aspect = aspect / fig_aspect
        current = width / height
        if current > normalized_aspect:
            new_width = height * normalized_aspect
            left += 0.5 * (width - new_width)
            width = new_width
        else:
            new_height = width / normalized_aspect
            bottom += 0.5 * (height - new_height)
            height = new_height
        return [left, bottom, width, height]

    def _configure_scan2d_layout(self, show_linecut: bool, extent: tuple[float, float, float, float]) -> None:
        xmin, xmax, ymin, ymax = extent
        x_span = abs(float(xmax) - float(xmin))
        y_span = abs(float(ymax) - float(ymin))
        aspect = x_span / y_span if y_span > 0 else 1.0
        if show_linecut:
            main_rect = self._fit_axes_box_to_aspect([0.09, 0.45, 0.66, 0.42], aspect)
            self.ax_res.set_position([0.09, 0.12, 0.66, 0.20])
            self.ax_res.set_visible(True)
        else:
            main_rect = self._fit_axes_box_to_aspect([0.09, 0.12, 0.66, 0.76], aspect)
            self.ax_res.set_visible(False)
        self.ax_main.set_position(main_rect)
        cbar_pad = 0.016
        cbar_width = 0.018
        cbar_left = min(0.95 - cbar_width, main_rect[0] + main_rect[2] + cbar_pad)
        self._scan_colorbar_ax = self.fig.add_axes([cbar_left, main_rect[1], cbar_width, main_rect[3]])

    def _current_observable_label(self) -> str:
        if self.ctx.trace is None:
            return "Signal"
        mode = self._friendly_mode_name(self._current_data_mode()) if hasattr(self, "mode_combo") else self.ctx.trace.mode
        if mode == "contrast":
            return "Contrast [a.u.]"
        if mode == "signal":
            return "Signal [counts]"
        if mode == "reference":
            return "Reference [counts]"
        if mode == "difference":
            return "Difference [counts]"
        return self.ctx.trace.y_label or "Signal"

    def _plot_scan_secondary_axis(self):
        if self.ctx.trace is None or self.ctx.trace.scan_dim != "scan1d" or self.ctx.x is None:
            return
        secondary = self.scan_secondary_combo.currentText().strip().lower()
        primary = self.mode_combo.currentText().strip().lower()
        if secondary in {"", "none", primary, "difference"}:
            return
        if self.ctx.trace.mode == secondary:
            sec_x = self.ctx.x
            sec_y = self.ctx.y
        else:
            sec_y_full, sec_label = self._series_for_mode(self.ctx.trace, secondary)
            sec_x, sec_y = self.ctx.trace.x_ns.copy(), sec_y_full.copy()
            sec_x, sec_y = bin_trace(sec_x, sec_y, self.bin_spin.value())
            sec_x, sec_y = apply_roi(sec_x, sec_y, *self._parse_roi())
            sec_x, sec_y = self._apply_exclusions(sec_x, sec_y)
            if len(sec_x) == 0:
                return
        _, sec_label = self._series_for_mode(self.ctx.trace, secondary)
        if self._ax_secondary is None:
            self._ax_secondary = self.ax_main.twinx()
        self._plot_series(self._ax_secondary, sec_x, sec_y, self._legend_label(f"{secondary.title()} axis", secondary.title()), role="overlay", color="#ff8a65", alpha=0.75)
        existing_label = self._ax_secondary.get_ylabel()
        if existing_label and existing_label != sec_label:
            self._ax_secondary.set_ylabel(f"{existing_label} / {sec_label}")
        else:
            self._ax_secondary.set_ylabel(sec_label)
        self._ax_secondary.grid(False)
        self._ax_secondary.set_autoscalex_on(False)
        self._ax_secondary.set_xlim(self.ax_main.get_xlim())
        self._secondary_series.append(np.asarray(sec_y, dtype=float))
        self._set_axis_ylim_from_series(self._ax_secondary, self._secondary_series)
        self._secondary_plot_data = (np.asarray(sec_x, dtype=float), np.asarray(sec_y, dtype=float), sec_label)

    def _apply_combined_legend(self):
        if not self.plot_opts.show_legend:
            return
        handles, labels = self.ax_main.get_legend_handles_labels()
        if self._ax_secondary is not None:
            h2, l2 = self._ax_secondary.get_legend_handles_labels()
            handles += h2
            labels += l2
        if handles:
            self.ax_main.legend(handles, labels, loc=self.plot_opts.legend_loc, fontsize=self.plot_opts.legend_font_size)

    def _autoscale_current_axes(self):
        if self._ax_secondary is not None:
            try:
                self._ax_secondary.set_xlim(self.ax_main.get_xlim())
            except Exception:
                pass

    def _set_axis_ylim_from_series(self, ax, series_list: list[np.ndarray]):
        finite_parts = []
        for arr in series_list:
            vals = np.asarray(arr, dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                finite_parts.append(vals)
        if not finite_parts:
            return
        vals = np.concatenate(finite_parts)
        ymin = float(np.min(vals))
        ymax = float(np.max(vals))
        if np.isclose(ymin, ymax):
            pad = max(1e-6, 0.05 * max(abs(ymin), 1.0))
        else:
            pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    def _reset_roi_full(self):
        if self.ctx.trace is None:
            return
        self.roi_min.setText(f"{np.min(self.ctx.trace.x_ns):.6g}")
        self.roi_max.setText(f"{np.max(self.ctx.trace.x_ns):.6g}")
        self._refresh_processed()

    def _crop_to_head(self):
        if self.ctx.trace is None:
            return
        x0, x1 = float(np.min(self.ctx.trace.x_ns)), float(np.max(self.ctx.trace.x_ns))
        self.roi_min.setText(f"{x0:.6g}")
        self.roi_max.setText(f"{x0 + 0.2 * (x1 - x0):.6g}")
        self._refresh_processed()

    def _refresh_processed(self):
        if self.ctx.trace is None:
            return
        tr = self.ctx.trace
        old_x = None if self.ctx.x is None else self.ctx.x.copy()
        old_y = None if self.ctx.y is None else self.ctx.y.copy()
        old_y_sm = None if self.ctx.y_smooth is None else self.ctx.y_smooth.copy()
        y = tr.y.copy()
        # Optional baseline/reference subtraction from overlay trace.
        baseline_name = self.baseline_combo.currentText().strip() if hasattr(self, "baseline_combo") else "None"
        if baseline_name and baseline_name != "None" and self.ctx.loaded_traces and baseline_name in self.ctx.loaded_traces:
            btr = self.ctx.loaded_traces[baseline_name]
            bx, by = btr.x_ns.copy(), btr.y.copy()
            order = np.argsort(bx)
            bx = bx[order]
            by = by[order]
            y = y - np.interp(tr.x_ns, bx, by, left=by[0], right=by[-1])
        processed = process_series(
            tr.x_ns,
            y,
            bin_size=self.bin_spin.value(),
            roi=self._parse_roi(),
            excluded_raw_indices=self.ctx.excluded_points,
            exclusion_ranges=self.ctx.exclusion_ranges,
            steps=self.ctx.analysis_steps,
        )
        x, y = processed.fit_x, processed.fit_y
        y_sm = smooth_trace(y, self.smooth_spin.value())
        if len(x) == 0 or len(y) == 0:
            if self.ctx.fit_result is not None and not self._is_loading:
                self._clear_stale_fit("processed data changed")
            self.ctx.x = np.array([])
            self.ctx.y = np.array([])
            self.ctx.y_smooth = np.array([])
            self.ctx.analysis_x = np.array([])
            self.ctx.analysis_y = np.array([])
            self.ctx.processed = processed
            self.summary_text.setPlainText("No points remain after ROI / exclusion / observable filtering.")
            self.ax_main.clear()
            self.ax_res.clear()
            self.canvas.draw_idle()
            return
        processed_changed = (
            self._arrays_differ(old_x, x)
            or self._arrays_differ(old_y, y)
            or (self.fit_smoothed_chk.isChecked() and self._arrays_differ(old_y_sm, y_sm))
        )
        if self.ctx.fit_result is not None and processed_changed and not self._is_loading:
            self._clear_stale_fit("processed data changed")
        self.ctx.x = x
        self.ctx.y = y
        self.ctx.y_smooth = y_sm
        self.ctx.processed = processed
        self.ctx.analysis_x = processed.analysis_x
        self.ctx.analysis_y = processed.analysis_y
        self._refresh_analysis_list()
        self._plot_raw()

    def _plot_raw(self):
        if self.ctx.x is None or self.ctx.y is None:
            return
        is_scan2d = bool(self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan2d")
        view_state = self._capture_view_state() if (not self._is_loading and not is_scan2d) else None
        self._apply_plot_controls_to_state()
        self._clear_secondary_axis()
        self._clear_scan_colorbar()
        self.ax_main.clear()
        self.ax_res.clear()
        self._reset_main_axes_layout()
        self._crosshair_v = None
        self._crosshair_h = None
        primary_series: list[np.ndarray] = []
        if self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan2d" and self.ctx.trace.z2d is not None:
            scan_extent = self._scan2d_extent(self.ctx.trace)
            if scan_extent is None:
                self.summary_text.setPlainText("2D scan map is unavailable because scan axes are malformed.")
                return
            x_edges, y_edges, extent = scan_extent
            xmin, xmax, ymin, ymax = extent
            linecut = self._extract_scan_linecut()
            self._configure_scan2d_layout(show_linecut=linecut is not None, extent=extent)
            cmap = self.scan_colormap_combo.currentText() if hasattr(self, "scan_colormap_combo") else "viridis"
            if hasattr(self, "scan_reverse_chk") and self.scan_reverse_chk.isChecked():
                cmap += "_r"
            mesh = self.ax_main.pcolormesh(x_edges, y_edges, self.ctx.trace.z2d, shading="flat", cmap=cmap)
            if hasattr(self, "scan_robust_limits_chk") and self.scan_robust_limits_chk.isChecked():
                finite = np.asarray(self.ctx.trace.z2d, dtype=float)
                finite = finite[np.isfinite(finite)]
                if len(finite) > 1:
                    low, high = np.percentile(finite, [2, 98])
                    if high > low:
                        mesh.set_clim(float(low), float(high))
            self.ax_main.set_xlim(xmin, xmax)
            self.ax_main.set_ylim(ymin, ymax)
            # The axes box is already fitted to the scan's physical aspect by
            # _configure_scan2d_layout. Use datalim so a previously removed
            # twinx axis cannot make Matplotlib reject the deferred Qt draw.
            self.ax_main.set_aspect("equal", adjustable="datalim")
            self.ax_main.set_xlabel(self._friendly_axis_label(self.ctx.trace.scan_axes[0] if len(self.ctx.trace.scan_axes) > 0 else "X"))
            self.ax_main.set_ylabel(self._friendly_axis_label(self.ctx.trace.scan_axes[1] if len(self.ctx.trace.scan_axes) > 1 else "Y"))
            self.ax_main.set_title(f"{self.ctx.trace.file_name} | 2D scan map")
            self._scan_colorbar = self.fig.colorbar(mesh, cax=self._scan_colorbar_ax)
            self._scan_colorbar.set_label(self._current_observable_label())
            self._update_scan_summary()
            points = self._scan_best_points()
            if self._scan_cursor is None and "max" in points:
                self._scan_cursor = (points["max"][0], points["max"][1])
            if self._scan_cursor is not None:
                cx, cy = self._scan_cursor
                self.ax_main.axvline(cx, color="#90caf9", lw=1.0, ls="--")
                self.ax_main.axhline(cy, color="#90caf9", lw=1.0, ls="--")
                self.scan_cursor_lbl.setText(f"Cursor: x={cx:.6g}, y={cy:.6g}")
            self.ax_main.set_xlim(xmin, xmax)
            self.ax_main.set_ylim(ymin, ymax)
            self.ax_res.set_visible(linecut is not None)
            if linecut is not None:
                lx, ly, title = linecut
                self._plot_series(self.ax_res, lx, ly, "Linecut", role="linecut", color="#4fc3f7")
                self.ax_res.set_title(title)
                self.ax_res.set_xlabel(self._friendly_axis_label(self.ctx.trace.scan_axes[0] if "horizontal" in title.lower() else self.ctx.trace.scan_axes[1]))
                self.ax_res.set_ylabel(self._current_observable_label())
                self.ax_res.grid(True, alpha=0.3)
        else:
            plot_x = self.ctx.analysis_x if self.ctx.analysis_x is not None else self.ctx.x
            plot_y = self.ctx.analysis_y if self.ctx.analysis_y is not None else self.ctx.y
            y_disp, y_lab = self._display_y(plot_y)
            primary_series.append(np.asarray(y_disp, dtype=float))
            if self.plot_opts.show_data:
                self._plot_series(self.ax_main, plot_x, y_disp, self._legend_label("Data", "Data"), role="data")
            if not self.ctx.analysis_steps:
                self._add_observable_layers(primary_series)
                self._add_iteration_layer(primary_series)
            if self.ctx.trace is not None and self.ctx.excluded_points:
                indices = sorted(index for index in self.ctx.excluded_points if 0 <= index < len(self.ctx.trace.x_ns))
                if indices:
                    excluded_x = self.ctx.trace.x_ns[indices]
                    excluded_y, _ = self._display_y(self.ctx.trace.y[indices])
                    self.ax_main.plot(excluded_x, excluded_y, "x", color="#e0a458", ms=6, mew=1.5, label=self._legend_label("Excluded", "Excluded"))
            # Overlay selected traces.
            if self.ctx.loaded_traces:
                for i in range(self.overlay_list.count()):
                    item = self.overlay_list.item(i)
                    if item.checkState() != Qt.Checked:
                        continue
                    nm = item.text()
                    if self.ctx.trace is not None and nm == self.ctx.trace.file_name:
                        continue
                    tr = self.ctx.loaded_traces.get(nm)
                    if tr is None:
                        continue
                    tx, ty = tr.x_ns.copy(), tr.y.copy()
                    tx, ty = bin_trace(tx, ty, self.bin_spin.value())
                    tx, ty = apply_roi(tx, ty, *self._parse_roi())
                    ty_disp, _ = self._display_y(ty)
                    primary_series.append(np.asarray(ty_disp, dtype=float))
                    self._plot_series(self.ax_main, tx, ty_disp, self._legend_label(f"Overlay:{nm}", f"Ov:{nm}"), role="overlay", alpha=0.5)
            if (
                not self.ctx.analysis_steps
                and
                self.plot_opts.show_smoothed
                and self.smooth_spin.value() > 1
                and self.ctx.y_smooth is not None
                and len(self.ctx.y_smooth) == len(self.ctx.x)
            ):
                ysm_disp, _ = self._display_y(self.ctx.y_smooth)
                primary_series.append(np.asarray(ysm_disp, dtype=float))
                self._plot_series(self.ax_main, self.ctx.x, ysm_disp, self._legend_label("Smoothed", "Smooth"), role="smoothed")
            if self._selected_plot_point is not None and "x" in self._selected_plot_point and "y" in self._selected_plot_point:
                py_disp, _ = self._display_y(np.asarray([self._selected_plot_point["y"]], dtype=float))
                primary_series.append(np.asarray(py_disp, dtype=float))
                self.ax_main.plot([self._selected_plot_point["x"]], py_disp, "o", ms=8, mec="#ffeb3b", mfc="none", mew=1.8, label=self._legend_label("Selected point", "Point"))
                self.ax_main.annotate(
                    f"x={self._format_inspect_value(float(self._selected_plot_point['x']))}\ny={self._format_inspect_value(float(self._selected_plot_point['y']))}",
                    (self._selected_plot_point["x"], float(py_disp[0])),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                    color="#ffeb3b",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": (0.08, 0.10, 0.15, 0.75), "edgecolor": "#ffeb3b"},
                )
            self.ax_main.set_xlabel(self._friendly_axis_label(self.ctx.trace.x_label if self.ctx.trace else "X"))
            self.ax_main.set_ylabel(y_lab)
            if self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan1d":
                self._plot_scan_secondary_axis()
                if self._ax_secondary is not None and self._selected_plot_point is not None and "secondary_y" in self._selected_plot_point:
                    sx = float(self._selected_plot_point.get("secondary_x", self._selected_plot_point["x"]))
                    sy = float(self._selected_plot_point["secondary_y"])
                    self._ax_secondary.plot([sx], [sy], "s", ms=7, mec="#ff8a65", mfc="none", mew=1.6)
                    self._ax_secondary.annotate(
                        self._format_inspect_value(sy),
                        (sx, sy),
                        textcoords="offset points",
                        xytext=(8, -14),
                        fontsize=8,
                        color="#ff8a65",
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": (0.08, 0.10, 0.15, 0.75), "edgecolor": "#ff8a65"},
                    )
            self._set_axis_ylim_from_series(self.ax_main, primary_series)
            self.ax_main.grid(True, alpha=0.3)
            self._apply_combined_legend()
            self._update_scan_summary()
            is_scan1d = bool(self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan1d")
            self.ax_res.set_visible((self.plot_opts.show_residual or self.plot_opts.show_fft_panel) and not is_scan1d)
            if self.plot_opts.show_fft_panel:
                y_fft = (
                    self.ctx.y_smooth
                    if (self.ctx.y_smooth is not None and self.smooth_spin.value() > 1 and len(self.ctx.y_smooth) == len(self.ctx.x))
                    else self.ctx.y
                )
                if len(self.ctx.x) > 3:
                    dt = float(np.median(np.diff(self.ctx.x)))
                    yf = np.abs(np.fft.rfft(y_fft - np.mean(y_fft)))
                    ff = np.fft.rfftfreq(len(y_fft), dt)
                    self.ax_res.plot(ff * 1000.0, yf, "-", lw=1.2)
                    self.ax_res.set_xlabel("Frequency (MHz)")
                    self.ax_res.set_ylabel("FFT amplitude")
                    self.ax_res.grid(True, alpha=0.3)
            elif self.plot_opts.show_residual:
                self.ax_res.set_xlabel(self._friendly_axis_label(self.ctx.trace.x_label if self.ctx.trace else "X"))
                self.ax_res.set_ylabel("Residual")
                self.ax_res.grid(True, alpha=0.3)
        self._autoscale_current_axes()
        self._restore_view_state(view_state)
        self._reset_export_navigation_history()
        self.canvas.draw_idle()
        self._sync_live_plot()
        if self.mode_stack.currentIndex() == 2:
            self._sync_scan_workspace()
        elif self.mode_stack.currentIndex() == 3:
            self._sync_results_workspace()

    def on_fft(self):
        if self.ctx.x is None or self.ctx.y is None:
            return
        y_use = (
            self.ctx.y_smooth
            if (self.smooth_spin.value() > 1 and self.ctx.y_smooth is not None and len(self.ctx.y_smooth) == len(self.ctx.x))
            else self.ctx.y
        )
        f0 = estimate_ramsey_frequency(self.ctx.x, y_use)
        self._message("FFT estimate", f"Dominant frequency ≈ {f0:.6g} GHz ({f0*1000:.3f} MHz)")

    def _collect_locks(self) -> dict[str, tuple[float, bool, float | None, float | None]]:
        locks: dict[str, tuple[float, bool, float | None, float | None]] = {}
        for r in range(self.param_table.rowCount()):
            name_item = self.param_table.item(r, 0)
            value_item = self.param_table.item(r, 1)
            min_item = self.param_table.item(r, 2)
            max_item = self.param_table.item(r, 3)
            lock_item = self.param_table.item(r, 4)
            if not (name_item and value_item and lock_item):
                continue
            try:
                val = float(value_item.text())
            except ValueError:
                continue
            lo = None
            hi = None
            if min_item is not None and min_item.text().strip():
                try:
                    lo = float(min_item.text())
                except ValueError:
                    lo = None
            if max_item is not None and max_item.text().strip():
                try:
                    hi = float(max_item.text())
                except ValueError:
                    hi = None
            locks[name_item.text()] = (val, lock_item.checkState() == Qt.Checked, lo, hi)
        return locks

    @staticmethod
    def _apply_locks(
        param_names: list[str],
        lower: list[float],
        upper: list[float],
        seed: np.ndarray,
        locks: dict[str, tuple[float, bool, float | None, float | None]],
    ):
        for i, name in enumerate(param_names):
            if name not in locks:
                continue
            v, is_locked, lo, hi = locks[name]
            if lo is not None:
                lower[i] = lo
            if hi is not None:
                upper[i] = hi
            if is_locked:
                lower[i] = v
                upper[i] = v
                seed[i] = v

    def _populate_param_table(self, names: list[str], values: np.ndarray):
        prev = self._collect_locks()
        self.param_table.setRowCount(len(names))
        for i, (n, v) in enumerate(zip(names, values)):
            meta = _param_meta(n)
            it0 = QTableWidgetItem(n)
            it1 = QTableWidgetItem(f"{float(v):.8g}")
            it2 = QTableWidgetItem("")
            it3 = QTableWidgetItem("")
            it4 = QTableWidgetItem("")
            it4.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            was_locked = n in prev and prev[n][1]
            it4.setCheckState(Qt.Checked if was_locked else Qt.Unchecked)
            if n in prev:
                if prev[n][2] is not None:
                    it2.setText(f"{prev[n][2]:.8g}")
                if prev[n][3] is not None:
                    it3.setText(f"{prev[n][3]:.8g}")
            it0.setToolTip(meta.description)
            it1.setToolTip(meta.description)
            it2.setToolTip(meta.description)
            it3.setToolTip(meta.description)
            self.param_table.setItem(i, 0, it0)
            self.param_table.setItem(i, 1, it1)
            self.param_table.setItem(i, 2, it2)
            self.param_table.setItem(i, 3, it3)
            self.param_table.setItem(i, 4, it4)
        self.param_table.resizeColumnsToContents()
        self.param_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def _save_param_snapshot(self):
        """Save current parameter state for undo."""
        snapshot = self._collect_locks()
        if snapshot:
            self._param_undo_stack.append(snapshot)
            self._param_redo_stack.clear()
            # Keep stack manageable
            if len(self._param_undo_stack) > 50:
                self._param_undo_stack.pop(0)

    def _undo_params(self):
        """Undo last parameter change."""
        if not self._param_undo_stack:
            return
        current = self._collect_locks()
        self._param_redo_stack.append(current)
        prev = self._param_undo_stack.pop()
        self._restore_param_snapshot(prev)

    def _redo_params(self):
        """Redo last undone parameter change."""
        if not self._param_redo_stack:
            return
        current = self._collect_locks()
        self._param_undo_stack.append(current)
        nxt = self._param_redo_stack.pop()
        self._restore_param_snapshot(nxt)

    def _restore_param_snapshot(self, snapshot: dict):
        """Restore parameter table from a locks snapshot."""
        for i in range(self.param_table.rowCount()):
            name_item = self.param_table.item(i, 0)
            if name_item is None:
                continue
            name = name_item.text()
            if name not in snapshot:
                continue
            val, locked, lo, hi = snapshot[name]
            val_item = self.param_table.item(i, 1)
            lo_item = self.param_table.item(i, 2)
            hi_item = self.param_table.item(i, 3)
            lock_item = self.param_table.item(i, 4)
            if val_item:
                val_item.setText(f"{val:.8g}")
            if lo_item:
                lo_item.setText(f"{lo:.8g}" if lo is not None else "")
            if hi_item:
                hi_item.setText(f"{hi:.8g}" if hi is not None else "")
            if lock_item:
                lock_item.setCheckState(Qt.Checked if locked else Qt.Unchecked)

    def on_fit(self):
        self._start_fit(robust=False)

    def on_robust_fit(self):
        self._start_fit(robust=True)

    def _start_fit(self, robust: bool = False):
        if self.ctx.x is None or self.ctx.y is None:
            return
        if self.ctx.trace is not None and not self.ctx.trace.fit_allowed:
            self.ctx.fit_result = None
            self.ctx.odmr_peaks = None
            self._set_status("N/A", "scan data plotted only; fitting disabled")
            self.summary_text.setPlainText("\n".join(self._trace_summary_lines(self.ctx.trace) + ["No fit performed for spatial scan data."]))
            return
        x = self.ctx.x
        y_raw = self.ctx.y
        y_target = self.ctx.y_smooth if (self.fit_smoothed_chk.isChecked() and self.ctx.y_smooth is not None) else y_raw
        model_name = self.model_combo.currentText()
        locks = self._collect_locks()

        # ODMR peak picking mode for multi-resonance spectra.
        if (
            self.ctx.trace is not None
            and self.ctx.trace.experiment_type in {"ODMR", "DEERFrequency"}
            and self.odmr_peak_pick_only.isChecked()
            and not self.odmr_multipeak_fit_chk.isChecked()
        ):
            y_peak = -y_target if self.odmr_polarity_combo.currentText() == "Dip" else y_target
            prominence = max(1e-9, 0.08 * float(np.ptp(y_peak)))
            peaks, _ = find_peaks(y_peak, prominence=prominence)
            peak_x = x[peaks]
            ranges = self._parse_odmr_ranges()
            if ranges:
                chosen = []
                for lo, hi in ranges:
                    mask = (peak_x >= lo) & (peak_x <= hi)
                    if np.any(mask):
                        local = peaks[mask]
                        best = local[np.argmax(y_peak[local])]
                        chosen.append(best)
                    else:
                        local = np.flatnonzero((x >= lo) & (x <= hi))
                        if len(local):
                            chosen.append(int(local[np.argmax(y_peak[local])]))
                peaks = np.array(sorted(set(chosen)), dtype=int)
                peak_x = x[peaks]
            if len(peak_x) > 1:
                self.ctx.odmr_peaks = peak_x
                self.ctx.fit_result = None
                self._plot_raw()
                if len(peaks):
                    y_disp, _ = self._display_y(y_target)
                    if self.plot_opts.show_peaks:
                        self.ax_main.plot(x[peaks], y_disp[peaks], "rx", ms=8, mew=2, label="Detected peaks")
                    if self.plot_opts.show_legend:
                        self.ax_main.legend(loc=self.plot_opts.legend_loc, fontsize=self.plot_opts.legend_font_size)
                    self.canvas.draw_idle()
                self._set_status("PASS", f"ODMR peak-pick mode, {len(peak_x)} peaks")
                self.summary_text.setPlainText(
                    "ODMR multi-peak detection (no single-peak fit)\n"
                    f"Detected peak locations ({len(peak_x)}):\n" + "\n".join([f"  {v:.6g}" for v in peak_x])
                )
                return

        # Disable UI during fit
        self._save_param_snapshot()
        self._set_ui_busy(True)
        self._set_status("Fitting", f"{'Robust ' if robust else ''}please wait...")

        # Prepare worker arguments to avoid accessing self in thread
        is_ramsey_auto = (model_name == "RamseyAuto")
        
        # We need to capture the current state for the worker
        # Note: We pass methods and data, but be careful with self access in the worker.
        # Actually, it's safer to have a static-like function or pass everything needed.
        # _fit_single_model uses self.multistart_spin, self.rabi_pi_lock_chk etc.
        # So we should gather all those configs now.
        
        fit_config = {
            "multistart": int(self.multistart_spin.value()),
            "rabi_use_sem": self.rabi_sem_weight_chk.isChecked(),
            "rabi_pi_lock": self.rabi_pi_lock_chk.isChecked(),
            "rabi_pi_ns": float(self.rabi_pi_ns.value()),
            "rabi_pi2_ns": float(self.rabi_pi2_ns.value()),
            "rabi_dead_time_ns": float(self.rabi_dead_time_ns.value()),
            "rabi_mode": self.rabi_mode_combo.currentText(),
            "odmr_multipeak": self.odmr_multipeak_fit_chk.isChecked(),
            "odmr_display": self.odmr_display_combo.currentText(),
            "odmr_polarity": self.odmr_polarity_combo.currentText(),
            "odmr_peak_ranges": tuple(self._parse_odmr_ranges()),
            "odmr_peak_count": int(self.odmr_peak_count_spin.value()),
            "odmr_auto_peak": self.odmr_auto_peak_count_chk.isChecked(),
            "odmr_cal_scale": float(self.odmr_cal_scale.value()),
            "t1_prep_mode": self.t1_prep_combo.currentText(),
            "t1_model_family": self.t1_model_combo.currentText(),
            "custom_spec": self.ctx.custom_model_spec,
            "custom_expr": self.custom_expr_edit.text().strip() if hasattr(self, "custom_expr_edit") else self.equation_display.toPlainText().strip()
        }

        # We need a worker function that doesn't depend on 'self' if possible, 
        # or we accept that we are reading from self (which is risky if UI changes).
        # Better to wrap the logic.
        
        if is_ramsey_auto:
            self.fit_worker = FitWorker(fit_ramsey_physics_first, x, y_target)
        else:
            if robust:
                self.fit_worker = FitWorker(self._run_robust_fit_background, model_name, x, y_target, locks, fit_config)
            else:
                self.fit_worker = FitWorker(self._run_single_fit_background, model_name, x, y_target, locks, fit_config)

        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.error.connect(self.on_fit_error)
        self.fit_worker.progress.connect(lambda msg: self._set_status("Fitting", msg))
        self.fit_worker.start()

    def _set_ui_busy(self, busy: bool):
        self.btn_fit.setEnabled(not busy)
        self.btn_robust_fit.setEnabled(not busy)
        self.btn_load.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.profile_combo.setEnabled(not busy)

    def on_fit_error(self, msg: str):
        self._set_ui_busy(False)
        self._set_status("FAIL", msg)
        self._message("Fit error", f"Fit failed:\n{msg}")

    def _set_status(self, status: str, reason: str = ""):
        """Set the status label text with color coding."""
        color_map = {
            "PASS": "#4caf50",   # green
            "WARN": "#ff9800",   # orange
            "FAIL": "#f44336",   # red
            "N/A": "#9e9e9e",    # grey
        }
        # Detect fitting state
        if "fitting" in status.lower() or "Fitting" in status:
            color = "#42a5f5"  # blue
        else:
            color = color_map.get(status, "#e0e0e0")
        text = f"Status: {status}"
        if reason:
            text += f" ({reason})"
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; }}")
        self.statusBar().showMessage(f"{status}: {reason}" if reason else status, 5000)

    def _drag_paths_from_event(self, event) -> list[str]:
        if not event.mimeData().hasUrls():
            return []
        paths = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local.lower().endswith(".mat"):
                paths.append(local)
        return paths

    def _set_drop_active(self, active: bool, text: str | None = None):
        self.drop_banner.setVisible(active)
        if text:
            self.drop_banner.setText(text)

    def _load_dropped_files(self, paths: list[str]):
        if not paths:
            self._message("Drop ignored", "Only MATLAB .mat files can be dropped here.")
            return
        self._set_drop_active(False)
        primary = paths[0]
        try:
            preview = load_saved_data_mat(primary, mode=self._current_data_mode())
        except Exception as exc:
            self._message("Drop ignored", f"Failed to load dropped file:\n{exc}")
            return
        if len(preview.x_ns) == 0 or len(preview.y) == 0:
            self._message("Drop ignored", "Dropped file produced an empty trace in the current observable mode.")
            return
        if not self._load_file(primary):
            return
        extras = paths[1:]
        if not extras:
            return
        added = 0
        for extra in extras:
            try:
                trace = load_saved_data_mat(extra, mode=self._current_data_mode())
            except Exception:
                continue
            self.ctx.loaded_traces[trace.file_name] = trace
            added += 1
        self._update_overlay_widgets()
        if added:
            self._message("Overlay files added", f"Loaded {Path(primary).name} and added {added} extra dropped file(s) as overlays.")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched not in {self, self.centralWidget()}:
            return super().eventFilter(watched, event)
        if event.type() == QEvent.DragEnter:
            paths = self._drag_paths_from_event(event)
            if paths:
                self._drag_depth += 1
                self._set_drop_active(True, f"Release to load {len(paths)} MATLAB file(s)")
                event.acceptProposedAction()
                return True
        elif event.type() == QEvent.DragMove:
            paths = self._drag_paths_from_event(event)
            if paths:
                event.acceptProposedAction()
                return True
        elif event.type() == QEvent.DragLeave:
            self._drag_depth = max(0, self._drag_depth - 1)
            if self._drag_depth == 0:
                self._set_drop_active(False)
        elif event.type() == QEvent.Drop:
            paths = self._drag_paths_from_event(event)
            if paths:
                self._drag_depth = 0
                self._load_dropped_files(paths)
                event.acceptProposedAction()
                return True
            self._set_drop_active(False)
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event):
        paths = self._drag_paths_from_event(event)
        if paths:
            self._drag_depth = 1
            self._set_drop_active(True, f"Release to load {len(paths)} MATLAB file(s)")
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = self._drag_paths_from_event(event)
        if paths:
            self._load_dropped_files(paths)
            event.acceptProposedAction()
        else:
            self._set_drop_active(False)
            event.ignore()

    def on_fit_finished(self, result_obj):
        self._set_ui_busy(False)
        # result_obj is either FitResult or RamseySelectionResult
        
        result: FitResult | None = None
        summary_header = "Single-model fit"
        summary_extra = ""
        robust_summary = ""

        if hasattr(result_obj, "selected_by"):
            # RamseySelectionResult
            result = result_obj.best
            summary_header = f"Selected by: {result_obj.selected_by}"
            summary_extra = "\n".join(
                [f"- {c.model_name}: R²={c.r2:.4f}, BIC={c.bic:.2f}, bound_hits={sum(c.bound_hits)}" for c in result_obj.candidates]
            )
        else:
            result = result_obj
            # Check for robust fit comparison in result.message
            if result is not None and result.message and "Robust fit candidate" in result.message:
                msg_lines = result.message.split("\n")
                robust_idx = next((i for i, l in enumerate(msg_lines) if "Robust fit candidate" in l), None)
                if robust_idx is not None:
                    robust_summary = "\n".join(msg_lines[robust_idx:])
                    summary_header = "Robust fit (best subset selected)"
        
        # Check if fallback happened in worker or if we need to do it here?
        # The worker should handle exceptions or return the result. 
        # If worker threw exception, on_fit_error is called.
        
        if result is None:
             self._set_status("FAIL", "No result")
             return

        # Proceed with updating UI (copy-paste from original on_fit)
        self.ctx.fit_result = result
        if isinstance(result.extras, dict) and result.extras.get("detected_peaks"):
            self.ctx.odmr_peaks = np.asarray(result.extras.get("detected_peaks", []), dtype=float)
        # ... (rest of the update logic) ...
        # We need to recover y_raw and y_target because they might have visually changed? 
        # No, ctx should be stable.
        y_raw = self.ctx.y
        fit_plot_y = result.extras.get("plot_y") if isinstance(result.extras, dict) else None
        if fit_plot_y is not None:
            fit_plot_y = np.asarray(fit_plot_y, dtype=float)
        display_y = fit_plot_y if fit_plot_y is not None else y_raw
        self.ctx.fit_target = display_y

        self._populate_param_table(result.param_names, result.params)
        self._plot_fit(result, display_y)

        profile = self._active_profile()
        if self.quality_gate_chk.isChecked():
            status, reason = classify_status(r2=result.r2, bound_hits=int(sum(result.bound_hits)), profile=profile)
        else:
            status, reason = "N/A", "Quality gate disabled"
        self.ctx.status = status
        self.ctx.status_reason = reason
        self._set_status(status, reason)

        lines = [
            summary_header,
            f"Profile: {profile.name}",
            f"Analysis mode: {result.model_name}",
            f"R²: {result.r2:.4f}",
            f"RMSE: {result.rmse:.6g}",
            f"AIC/BIC: {result.aic:.2f} / {result.bic:.2f}",
            f"Durbin-Watson: {result.durbin_watson:.3f}",
            f"Bound hits: {sum(result.bound_hits)}",
            f"Success: {result.success}",
            f"Status: {status} | {reason}",
        ]
        if result.selection_note:
            lines.append(f"Selection: {result.selection_note}")
        if self.ctx.trace is not None:
            snapshot_lines = self._snapshot_summary_lines(self.ctx.trace.metadata if self.ctx.trace.metadata is not None else {})
            if snapshot_lines:
                lines += [""] + snapshot_lines
        for n, v, e in zip(result.param_names, result.params, result.errors):
            meta = _param_meta(n)
            unit_txt = f" {meta.units}" if meta.units else ""
            lines.append(f"  {n} ({meta.symbol}): {v:.6g}{unit_txt} ± {e:.2g}   [{meta.description}]")
        nv = _nv_metrics(profile.name, result, self._effective_trace_metadata())
        if profile.name == "ODMR":
            # Calibrated ODMR metrics and optional ZFS estimates.
            pmap = {n: float(v) for n, v in zip(result.param_names, result.params)}
            centers = []
            widths = []
            if result.model_name == "ODMRMulti":
                i = 1
                while f"x0_{i}" in pmap:
                    centers.append(self._odmr_calibrate(pmap[f"x0_{i}"]))
                    widths.append(abs(float(self.odmr_cal_scale.value()) * pmap.get(f"w{i}", 0.0)) * 2.0)
                    i += 1
            else:
                if "x0" in pmap:
                    centers.append(self._odmr_calibrate(pmap["x0"]))
                if "w" in pmap:
                    widths.append(abs(float(self.odmr_cal_scale.value()) * pmap["w"]) * 2.0)
            if centers:
                nv["center_calibrated"] = float(np.mean(centers))
                nv["num_centers"] = float(len(centers))
            if widths:
                nv["fwhm_calibrated_mean"] = float(np.mean(widths))
            if len(centers) >= 2:
                cs = sorted(centers)
                nv["D"] = float(0.5 * (cs[-1] + cs[0]))
                nv["E"] = float(0.5 * (cs[-1] - cs[0]))
        if nv:
            lines += ["", "NV-relevant metrics:"]
            lines += [f"  {k}: {v:.6g}" for k, v in nv.items()]
        if profile.name == "T1" and isinstance(result.extras, dict):
            prep_mode = result.extras.get("t1_prep_mode")
            family = result.extras.get("t1_model_family")
            if prep_mode or family:
                lines += ["", "T1 fit mode:"]
                if prep_mode:
                    lines.append(f"  preprocessing: {prep_mode}")
                if family:
                    lines.append(f"  model family: {family}")
        if summary_extra:
            lines += ["", "Candidate comparison:", summary_extra]
        if robust_summary:
            lines += ["", robust_summary]
        self.summary_text.setPlainText("\n".join(lines))
        self._sync_results_workspace()

    def _run_single_fit_background(self, model_name, x, y, locks, config):
        rng = np.random.default_rng(22)
        tspan = float(np.max(x) - np.min(x))
        yamp = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
        y0 = float(np.median(y))
        f0 = estimate_ramsey_frequency(x, y)
        n_starts = int(config["multistart"])

        if model_name == "RamseySimple":
            names = ["y0", "A", "f", "phi", "tau", "n"]
            lower = [-0.2, 0.0, 2e-4, -2 * np.pi, 50.0, 0.5]
            upper = [0.1, 0.2, 0.03, 2 * np.pi, 12000.0, 5.0]
            seed0 = np.array([y0, max(1e-4, yamp), f0, 0.0, max(120.0, tspan / 3), 2.0])
            self._apply_locks(names, lower, upper, seed0, locks)
            scale = [0.05 * (u - l) for l, u in zip(lower, upper)]
            seeds = [seed0 + rng.normal(0, scale) for _ in range(n_starts)]
            return fit_model_multistart(model_name=model_name, model_func=ramsey_simple, x=x, y=y, param_names=names, lower=lower, upper=upper, seeds=seeds)

        if model_name == "RamseyHyperfine":
            names = ["y0", "A", "f", "phi", "tau", "n"]
            lower = [-0.2, 0.0, 2e-4, -2 * np.pi, 50.0, 0.5]
            upper = [0.1, 0.2, 0.03, 2 * np.pi, 12000.0, 5.0]
            seed0 = np.array([y0, max(1e-4, yamp), f0, 0.0, max(120.0, tspan / 3), 2.0])
            self._apply_locks(names, lower, upper, seed0, locks)
            scale = [0.05 * (u - l) for l, u in zip(lower, upper)]
            seeds = [seed0 + rng.normal(0, scale) for _ in range(n_starts)]
            return fit_model_multistart(model_name=model_name, model_func=ramsey_hyperfine_n14, x=x, y=y, param_names=names, lower=lower, upper=upper, seeds=seeds)

        return fit_non_ramsey(
            model_name,
            x,
            y,
            config=FitWorkflowConfig(
                multistart=n_starts,
                rabi_use_sem=bool(config.get("rabi_use_sem", True)),
                rabi_pi_lock=bool(config["rabi_pi_lock"]),
                rabi_pi_ns=float(config["rabi_pi_ns"]),
                rabi_pi2_ns=float(config["rabi_pi2_ns"]),
                rabi_dead_time_ns=float(config["rabi_dead_time_ns"]),
                rabi_mode=str(config["rabi_mode"]),
                odmr_multipeak=bool(config["odmr_multipeak"]),
                odmr_polarity=str(config.get("odmr_polarity", "Auto")),
                odmr_peak_ranges=tuple(config.get("odmr_peak_ranges", ())),
                odmr_peak_count=int(config["odmr_peak_count"]),
                odmr_auto_peak=bool(config["odmr_auto_peak"]),
                t1_prep_mode=str(config.get("t1_prep_mode", "Auto")),
                t1_model_family=str(config.get("t1_model_family", "Stretched exponential")),
                custom_spec=config["custom_spec"],
                custom_expr=str(config["custom_expr"]),
            ),
            locks=locks,
            custom_spec=config["custom_spec"],
            trace_experiment_type=(self.ctx.trace.experiment_type if self.ctx.trace is not None else model_name),
            trace=self.ctx.trace,
        )

    def _run_robust_fit_background(self, model_name, x, y, locks, config):
        """Robust fit: run fits on multiple data subsets, compare ALL on full data."""
        # 1. Baseline fit (full data)
        base_res = self._run_single_fit_background(model_name, x, y, locks, config)

        # 2. Define data subsets to try
        n = len(x)
        cuts = [
            ("full",        0,              0),
            ("trim_start4", int(n * 0.04),  0),
            ("trim_start8", int(n * 0.08),  0),
            ("trim_end5",   0,              int(n * 0.05)),
            ("trim_both4",  int(n * 0.04),  int(n * 0.04)),
            ("trim_start12",int(n * 0.12),  0),
            ("trim_end10",  0,              int(n * 0.10)),
        ]

        # Each entry: (label, FitResult on subset, FitResult re-evaluated on full)
        candidates = [("full", base_res, base_res)]  # base already on full data

        sub_config = config.copy()
        sub_config["multistart"] = max(1, int(config["multistart"] // 2))

        for idx, (label, onset, offset) in enumerate(cuts[1:]):  # skip 'full' — already done
            if n - onset - offset < 10:
                continue
            # Emit progress to UI
            if self.fit_worker is not None:
                self.fit_worker.progress.emit(f"Robust: fitting subset {idx+1}/{len(cuts)-1} ({label})...")
            xv = x[onset : n - offset if offset else n]
            yv = y[onset : n - offset if offset else n]
            try:
                sub_res = self._run_single_fit_background(model_name, xv, yv, locks, sub_config)
                # Re-evaluate this candidate's params on the FULL data
                model_func = MODEL_REGISTRY.get("SpinEcho" if sub_res.model_name == "T1" else sub_res.model_name)
                if model_func is not None:
                    x_eval = x
                    if isinstance(sub_res.extras, dict) and sub_res.extras.get("x_offset_ns") is not None and sub_res.model_name in {"RabiPhaseRamp", "RabiCosine", "RabiChirp", "RabiLongDamped"}:
                        x_eval = x - float(sub_res.extras["x_offset_ns"])
                    y_fit_full = model_func(x_eval, *sub_res.params)
                    full_eval = FitResult(
                        model_name=sub_res.model_name,
                        param_names=sub_res.param_names,
                        params=sub_res.params,
                        errors=sub_res.errors,
                        y_fit=y_fit_full,
                        r2=_r_squared(y, y_fit_full),
                        rmse=_rmse(y, y_fit_full),
                        aic=_aic(y, y_fit_full, len(sub_res.params)),
                        bic=_bic(y, y_fit_full, len(sub_res.params)),
                        durbin_watson=_dw(y - y_fit_full),
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

        # 3. Compare ALL candidates using full-data R² (apples-to-apples)
        best_label, _, best_full = max(candidates, key=lambda t: t[2].r2)

        # 4. Build a summary of all candidates for the user
        summary_lines = ["Robust fit candidate comparison (all scored on full data):"]
        for label, sub_res, full_res in candidates:
            marker = " ★" if label == best_label else ""
            summary_lines.append(
                f"  {label}: subset_R²={sub_res.r2:.4f}, full_R²={full_res.r2:.4f}, "
                f"RMSE={full_res.rmse:.6g}, BIC={full_res.bic:.2f}{marker}"
            )
        best_full.message = best_full.message + "\n" + "\n".join(summary_lines)
        return best_full


    def _plot_fit(self, fit_result: FitResult, y_raw: np.ndarray):
        x = self.ctx.x
        if x is None:
            return
        raw_x = np.asarray(x, dtype=float)
        raw_y = np.asarray(y_raw, dtype=float)
        fit_for_display = np.asarray(fit_result.y_fit, dtype=float)
        if self.ctx.analysis_steps:
            x, y_raw, _ = apply_steps(raw_x, raw_y, self.ctx.analysis_steps)
            _fit_x, fit_for_display, _ = apply_steps(raw_x, fit_for_display, self.ctx.analysis_steps)
        view_state = self._capture_view_state() if not self._is_loading else None
        self._apply_plot_controls_to_state()
        self._clear_secondary_axis()
        self._clear_scan_colorbar()
        self.ax_main.clear()
        self.ax_res.clear()
        self._reset_main_axes_layout()
        self._crosshair_v = None
        self._crosshair_h = None
        primary_series: list[np.ndarray] = []
        if isinstance(fit_result.extras, dict) and fit_result.extras.get("plot_y_label"):
            y_disp = np.asarray(y_raw, dtype=float)
            yfit_disp = fit_for_display
            y_lab = str(fit_result.extras.get("plot_y_label"))
        else:
            y_disp, y_lab = self._display_y(y_raw)
            yfit_disp, _ = self._display_y(fit_for_display)
        primary_series.append(np.asarray(y_disp, dtype=float))
        if self.plot_opts.show_data:
            self._plot_series(self.ax_main, x, y_disp, self._legend_label("Data", "Data"), role="data")
        if not self.ctx.analysis_steps:
            self._add_observable_layers(primary_series)
            self._add_iteration_layer(primary_series)
        if self.ctx.loaded_traces:
            for i in range(self.overlay_list.count()):
                item = self.overlay_list.item(i)
                if item.checkState() != Qt.Checked:
                    continue
                nm = item.text()
                if self.ctx.trace is not None and nm == self.ctx.trace.file_name:
                    continue
                tr = self.ctx.loaded_traces.get(nm)
                if tr is None:
                    continue
                tx, ty = tr.x_ns.copy(), tr.y.copy()
                tx, ty = bin_trace(tx, ty, self.bin_spin.value())
                tx, ty = apply_roi(tx, ty, *self._parse_roi())
                ty_disp, _ = self._display_y(ty)
                primary_series.append(np.asarray(ty_disp, dtype=float))
                self._plot_series(self.ax_main, tx, ty_disp, self._legend_label(f"Overlay:{nm}", f"Ov:{nm}"), role="overlay", alpha=0.5)
        if self.plot_opts.show_fit:
            primary_series.append(np.asarray(yfit_disp, dtype=float))
            self._plot_series(self.ax_main, x, yfit_disp, self._legend_label(f"Fit: {fit_result.model_name}", fit_result.model_name), role="fit")
            self._plot_rabi_envelope(self.ax_main, x, fit_result, primary_series)
            # Confidence band (±1σ from parameter errors)
            if self.plot_opts.show_confidence and fit_result.errors is not None:
                model_func = MODEL_REGISTRY.get(
                    "SpinEcho" if fit_result.model_name == "T1" else fit_result.model_name
                )
                if model_func is not None:
                    try:
                        x_eval = x
                        if isinstance(fit_result.extras, dict) and fit_result.extras.get("x_offset_ns") is not None and fit_result.model_name in {"RabiPhaseRamp", "RabiCosine", "RabiChirp", "RabiLongDamped"}:
                            x_eval = x - float(fit_result.extras["x_offset_ns"])
                        y_upper = model_func(x_eval, *(fit_result.params + fit_result.errors))
                        y_lower = model_func(x_eval, *(fit_result.params - fit_result.errors))
                        yu_disp, _ = self._display_y(y_upper)
                        yl_disp, _ = self._display_y(y_lower)
                        # Ensure upper >= lower for fill_between
                        band_lo = np.minimum(yu_disp, yl_disp)
                        band_hi = np.maximum(yu_disp, yl_disp)
                        primary_series.extend([np.asarray(band_lo, dtype=float), np.asarray(band_hi, dtype=float)])
                        self.ax_main.fill_between(
                            x, band_lo, band_hi,
                            alpha=0.2, color="#42a5f5", label="±1σ confidence"
                        )
                    except Exception:
                        pass  # Gracefully skip if model evaluation fails
        if self.plot_opts.show_peaks and self.ctx.odmr_peaks is not None and len(self.ctx.odmr_peaks):
            peak_indices = [int(np.argmin(np.abs(x - px))) for px in self.ctx.odmr_peaks]
            peak_indices = [idx for idx in peak_indices if 0 <= idx < len(x)]
            if peak_indices:
                self.ax_main.plot(x[peak_indices], y_disp[peak_indices], "rx", ms=8, mew=1.8, label=self._legend_label("Detected peaks", "Peaks"))
        if self._selected_plot_point is not None and "x" in self._selected_plot_point and "y" in self._selected_plot_point:
            py_disp, _ = self._display_y(np.asarray([self._selected_plot_point["y"]], dtype=float))
            primary_series.append(np.asarray(py_disp, dtype=float))
            self.ax_main.plot([self._selected_plot_point["x"]], py_disp, "o", ms=8, mec="#ffeb3b", mfc="none", mew=1.8, label=self._legend_label("Selected point", "Point"))
        self.ax_main.set_xlabel(self._friendly_axis_label(self.ctx.trace.x_label if self.ctx.trace else "X"))
        self.ax_main.set_ylabel(y_lab)
        if self.ctx.trace is not None and self.ctx.trace.scan_dim == "scan1d":
            self._plot_scan_secondary_axis()
        self._set_axis_ylim_from_series(self.ax_main, primary_series)
        self.ax_main.grid(True, alpha=0.3)
        self._draw_plot_annotation(fit_result)
        self._apply_combined_legend()
        self.ax_main.set_title(f"{self.ctx.trace.file_name if self.ctx.trace else ''} | " + rf"$R^2$={fit_result.r2:.3f}")

        fit_target = self.ctx.fit_target if self.ctx.fit_target is not None else raw_y
        residual = fit_target - fit_result.y_fit
        self.ax_res.set_visible((self.plot_opts.show_residual or self.plot_opts.show_fft_panel) and not self.ctx.analysis_steps)
        if self.plot_opts.show_fft_panel:
            y_fft = y_raw
            if len(x) > 3:
                dt = float(np.median(np.diff(x)))
                yf = np.abs(np.fft.rfft(y_fft - np.mean(y_fft)))
                ff = np.fft.rfftfreq(len(y_fft), dt)
                self.ax_res.plot(ff * 1000.0, yf, "-", lw=1.2)
                self.ax_res.set_xlabel("Frequency (MHz)")
                self.ax_res.set_ylabel("FFT amplitude")
                self.ax_res.grid(True, alpha=0.3)
        elif self.plot_opts.show_residual:
            self.ax_res.plot(raw_x, residual, ".", ms=3)
            self.ax_res.axhline(0.0, ls="--", lw=1, color="k")
            self.ax_res.set_xlabel(self.ctx.trace.x_label if self.ctx.trace else "X")
            self.ax_res.set_ylabel("Residual")
            self.ax_res.grid(True, alpha=0.3)
        self._autoscale_current_axes()
        self._restore_view_state(view_state)
        self._reset_export_navigation_history()
        self.canvas.draw_idle()
        self._sync_live_plot(fit_result=fit_result, y_display=y_raw)
        if self.mode_stack.currentIndex() == 3:
            self._sync_results_workspace()

    def on_save_results(self):
        if self.ctx.trace is None or self.ctx.x is None or self.ctx.y is None:
            self._message("Save", "Load data first.")
            return
        self._apply_plot_controls_to_state()
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            return
        out = Path(out_dir)
        stem = Path(self.ctx.trace.file_name).stem

        # Save current figure snapshot (works for 1D and 2D scan displays).
        fit_png = out / f"{stem}_gui_fit.png"
        self.fig.set_size_inches(self.plot_opts.fig_width, self.plot_opts.fig_height, forward=True)
        if self.ctx.trace.scan_dim != "scan2d":
            self.fig.tight_layout(pad=0.5)
        saved_paths: list[Path] = []
        if self.exp_png_chk.isChecked():
            self.fig.savefig(fit_png, dpi=self.plot_opts.save_dpi, bbox_inches="tight", pad_inches=0.12)
            saved_paths.append(fit_png)
        fit_pdf = out / f"{stem}_gui_fit.pdf"
        fit_svg = out / f"{stem}_gui_fit.svg"
        if self.exp_pdf_chk.isChecked():
            self.fig.savefig(fit_pdf, dpi=self.plot_opts.save_dpi, bbox_inches="tight", pad_inches=0.12)
            saved_paths.append(fit_pdf)
        if self.exp_svg_chk.isChecked():
            self.fig.savefig(fit_svg, dpi=self.plot_opts.save_dpi, bbox_inches="tight", pad_inches=0.12)
            saved_paths.append(fit_svg)

        profile = self._active_profile()
        result = self.ctx.fit_result
        trace_metadata = self._effective_trace_metadata()
        nv = _nv_metrics(profile.name, result, trace_metadata) if result is not None else {}
        sample_md = self._collect_sample_metadata()
        provenance = {
            "app_name": "SmartFitterPy",
            "app_version": "2026.02",
            "timestamp_utc": _now_utc_iso(),
            "preset_name": self.ctx.active_preset_name,
            "baseline_trace": self.baseline_combo.currentText(),
            "exclusion_ranges": list(self.ctx.exclusion_ranges or []),
            "excluded_points_count": int(len(self.ctx.excluded_points or set())),
            "rabi_dead_time_ns": float(self.rabi_dead_time_ns.value()),
            "analysis_steps": [step.to_dict() for step in self.ctx.analysis_steps],
            "analysis_provenance": ([] if self.ctx.processed is None else self.ctx.processed.provenance),
        }
        payload = {
            "file": self.ctx.trace.file_name,
            "profile": profile.name,
            "scan_dim": self.ctx.trace.scan_dim,
            "scan_axes": list(self.ctx.trace.scan_axes),
            "model": (result.model_name if result is not None else "none"),
            "selection_note": (result.selection_note if result is not None else ""),
            "status": self.ctx.status,
            "status_reason": self.ctx.status_reason,
            "metrics": (
                {
                    "r2": result.r2,
                    "rmse": result.rmse,
                    "aic": result.aic,
                    "bic": result.bic,
                    "durbin_watson": result.durbin_watson,
                    "bound_hits_count": int(sum(result.bound_hits)),
                }
                if result is not None
                else {}
            ),
            "params": ({n: float(v) for n, v in zip(result.param_names, result.params)} if result is not None else {}),
            "errors": ({n: float(e) for n, e in zip(result.param_names, result.errors)} if result is not None else {}),
            "fit_extras": (dict(result.extras) if result is not None and isinstance(result.extras, dict) else {}),
            "nv_metrics": nv,
            "odmr_peak_locations": ([] if self.ctx.odmr_peaks is None else [float(v) for v in self.ctx.odmr_peaks]),
            "sample_metadata": sample_md,
            "trace_metadata": trace_metadata,
            "provenance": provenance,
        }
        result_json_path = out / f"{stem}_gui_result.json"
        if self.exp_json_chk.isChecked():
            with result_json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            saved_paths.append(result_json_path)

        csv_path = out / f"{stem}_gui_result.csv"
        if self.exp_csv_chk.isChecked():
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["file", self.ctx.trace.file_name])
                writer.writerow(["profile", profile.name])
                writer.writerow(["scan_dim", self.ctx.trace.scan_dim])
                writer.writerow(["scan_axes", ",".join(self.ctx.trace.scan_axes)])
                writer.writerow(["model", (result.model_name if result is not None else "none")])
                writer.writerow(["selection_note", (result.selection_note if result is not None else "")])
                writer.writerow(["status", self.ctx.status])
                writer.writerow(["status_reason", self.ctx.status_reason])
                if result is not None:
                    writer.writerow(["r2", result.r2])
                    writer.writerow(["rmse", result.rmse])
                    writer.writerow(["aic", result.aic])
                    writer.writerow(["bic", result.bic])
                    writer.writerow(["durbin_watson", result.durbin_watson])
                    for n, v in zip(result.param_names, result.params):
                        writer.writerow([f"param:{n}", v])
                    for k, v in dict(result.extras).items():
                        writer.writerow([f"fit_extra:{k}", v])
                for k, v in nv.items():
                    writer.writerow([f"nv:{k}", v])
                if self.ctx.odmr_peaks is not None:
                    for i, v in enumerate(self.ctx.odmr_peaks):
                        writer.writerow([f"odmr_peak_{i+1}", float(v)])
                for k, v in sample_md.items():
                    writer.writerow([f"sample:{k}", v])
                for k, v in provenance.items():
                    writer.writerow([f"provenance:{k}", json.dumps(v) if isinstance(v, (list, dict)) else v])
            saved_paths.append(csv_path)

        # Save an annotated report figure with key extracted metrics.
        if self.ctx.trace.fit_allowed and (self.exp_report_png_chk.isChecked() or self.exp_report_pdf_chk.isChecked()):
            report_png = out / f"{stem}_gui_report.png"
            fig = Figure(figsize=(self.plot_opts.fig_width, self.plot_opts.fig_height))
            selected_report = self._selected_report_metrics()
            if self.plot_opts.show_residual:
                gs = fig.add_gridspec(2, 2, width_ratios=[4.5, 1.7], height_ratios=[3.0, 1.4])
                ax = fig.add_subplot(gs[0, 0])
                axr = fig.add_subplot(gs[1, 0], sharex=ax)
                ax_stats = fig.add_subplot(gs[:, 1])
            else:
                gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.7])
                ax = fig.add_subplot(gs[0, 0])
                axr = None
                ax_stats = fig.add_subplot(gs[0, 1])
            x = self.ctx.x
            y = self.ctx.fit_target if (result is not None and isinstance(result.extras, dict) and result.extras.get("plot_y_label")) else self.ctx.y
            if result is not None and isinstance(result.extras, dict) and result.extras.get("plot_y_label"):
                y_disp = np.asarray(y, dtype=float)
                y_lab = str(result.extras.get("plot_y_label"))
            else:
                y_disp, y_lab = self._display_y(y)
            if self.plot_opts.show_data:
                ax.plot(x, y_disp, "o", ms=4, alpha=0.65, label="Data")
            if result is not None:
                if isinstance(result.extras, dict) and result.extras.get("plot_y_label"):
                    yf_disp = np.asarray(result.y_fit, dtype=float)
                else:
                    yf_disp, _ = self._display_y(result.y_fit)
                if self.plot_opts.show_fit:
                    ax.plot(x, yf_disp, "-", lw=2, label=f"Fit: {result.model_name}")
                    if self.plot_opts.show_rabi_envelope and isinstance(result.extras, dict):
                        bounds = rabi_envelope_bounds(x, result.extras.get("rabi_envelope"))
                        if bounds is not None:
                            upper, lower = bounds
                            upper_disp, _ = self._display_y(np.asarray(upper, dtype=float))
                            lower_disp, _ = self._display_y(np.asarray(lower, dtype=float))
                            ax.plot(x, upper_disp, "--", lw=1.4, color="#b57edc", label="Rabi envelope")
                            ax.plot(x, lower_disp, "--", lw=1.4, color="#b57edc", label="_nolegend_")
                if self.plot_opts.show_residual and axr is not None:
                    res = np.asarray(y, dtype=float) - result.y_fit
                    axr.plot(x, res, ".", ms=3)
                    axr.axhline(0.0, ls="--", lw=1, color="k")
                    axr.set_ylabel("Residual")
                    axr.set_xlabel(self.ctx.trace.x_label)
            ax.set_xlabel(self.ctx.trace.x_label)
            ax.set_ylabel(y_lab)
            ax.grid(True, alpha=0.3)
            if self.plot_opts.show_legend:
                ax.legend(loc=self.plot_opts.legend_loc, fontsize=self.plot_opts.legend_font_size)
            key_lines = [f"Profile: {profile.name}", f"Status: {self.ctx.status}"]
            if result is not None:
                if "r2" in selected_report:
                    key_lines.append(f"{self._report_label('r2')}: {result.r2:.4f}")
                if "rmse" in selected_report:
                    key_lines.append(f"{self._report_label('rmse')}: {result.rmse:.4g}")
                for n, v, e in zip(result.param_names[:6], result.params[:6], result.errors[:6]):
                    key_lines.append(f"{self._friendly_param_name(n)}: {self._format_value_with_units(n, float(v))} +/- {e:.2g}")
            for k, v in nv.items():
                if k in selected_report:
                    key_lines.append(f"{self._report_label(k)}: {v:.6g}")
            if self.ctx.odmr_peaks is not None and len(self.ctx.odmr_peaks):
                key_lines.append("ODMR peaks: " + ", ".join([f"{p:.6g}" for p in self.ctx.odmr_peaks[:8]]))
            ax_stats.axis("off")
            ax_stats.text(
                0.01,
                0.99,
                "\n".join(key_lines),
                transform=ax_stats.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
            fig.tight_layout(pad=0.5)
            if self.exp_report_png_chk.isChecked():
                FigureCanvas(fig).print_figure(report_png, dpi=self.plot_opts.save_dpi, bbox_inches="tight", pad_inches=0.12)
                saved_paths.append(report_png)
            if self.exp_report_pdf_chk.isChecked():
                report_pdf = out / f"{stem}_gui_report.pdf"
                FigureCanvas(fig).print_figure(report_pdf, dpi=self.plot_opts.save_dpi, bbox_inches="tight", pad_inches=0.12)
                saved_paths.append(report_pdf)

        # Origin-friendly bundle CSV.
        if self.exp_origin_chk.isChecked():
            origin_csv = out / f"{stem}_origin_bundle.csv"
            with origin_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "x_original",
                        "y_original",
                        "signal_original",
                        "reference_original",
                        "x_processed",
                        "y_processed",
                        "y_fit",
                        "residual",
                        "x_analysis",
                        "y_analysis",
                    ]
                )
                tr = self.ctx.trace
                y_fit = result.y_fit if result is not None else np.full_like(self.ctx.y, np.nan, dtype=float)
                residual = self.ctx.y - y_fit if result is not None else np.full_like(self.ctx.y, np.nan, dtype=float)
                n = max(len(tr.x_ns), len(self.ctx.x))
                for i in range(n):
                    xo = tr.x_ns[i] if i < len(tr.x_ns) else ""
                    yo = tr.y[i] if i < len(tr.y) else ""
                    so = tr.signal[i] if i < len(tr.signal) else ""
                    ro = tr.reference[i] if i < len(tr.reference) else ""
                    xp = self.ctx.x[i] if i < len(self.ctx.x) else ""
                    yp = self.ctx.y[i] if i < len(self.ctx.y) else ""
                    yf = y_fit[i] if i < len(y_fit) else ""
                    rv = residual[i] if i < len(residual) else ""
                    ax = self.ctx.analysis_x[i] if self.ctx.analysis_x is not None and i < len(self.ctx.analysis_x) else ""
                    ay = self.ctx.analysis_y[i] if self.ctx.analysis_y is not None and i < len(self.ctx.analysis_y) else ""
                    writer.writerow([xo, yo, so, ro, xp, yp, yf, rv, ax, ay])
            origin_meta = out / f"{stem}_origin_bundle_meta.json"
            with origin_meta.open("w", encoding="utf-8") as f:
                json.dump({"sample_metadata": sample_md, "trace_metadata": trace_metadata, "provenance": provenance, "nv_metrics": nv}, f, indent=2)
            saved_paths += [origin_csv, origin_meta]

        self._save_preferences()
        msg = "Saved:\n" + "\n".join([str(p) for p in saved_paths]) if saved_paths else "No export files selected."
        self._message("Save complete", msg)


def run_gui():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    print("--- Launching SmartFitter v2026.02 (Refactored) ---")
    win = SmartFitterMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()

