import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent
from PySide6.QtWidgets import QApplication

from nvfit.fit_engine import FitResult
from nvfit.gui_app import IterationSelectionDialog, SmartFitterMainWindow


TDD_ROOT = Path(r"D:\BacklundLabResearch\Data\Experiments\429_TDD")
JUNE5_2D_STAGE = Path(r"C:\Users\Div\Box\Backlund lab shared\data\Div\June5th_2DStage_SPC1_NVFinding3Large200um5umstep.mat")


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _fixture(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath(*parts)


def test_odmr_display_defaults_to_measured_peaks_and_only_dip_view_inverts():
    _app()
    win = SmartFitterMainWindow()
    win.ctx.trace = SimpleNamespace(experiment_type="ODMR")
    y = np.array([0.1, 0.8, 0.2])

    assert win.odmr_display_combo.currentText() == "Peaks (measured view)"
    assert win.odmr_polarity_combo.currentText() == "Peak"
    displayed, label = win._display_y(y)
    assert np.array_equal(displayed, y)
    assert label.endswith("(peaks)")

    win.odmr_display_combo.setCurrentText("Dips (inverted view)")
    displayed, label = win._display_y(y)
    assert np.array_equal(displayed, -y)
    assert label.endswith("(dips)")


def test_odmr_range_parser_accepts_negative_and_scientific_ranges():
    _app()
    win = SmartFitterMainWindow()
    win.odmr_ranges.setText("-2.2--2.1, 2.18e0 to 2.19e0, 3:4")

    assert win._parse_odmr_ranges() == [(-2.2, -2.1), (2.18, 2.19), (3.0, 4.0)]


def test_gui_drop_loading_and_annotation_smoke():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    assert win.exp_png_chk.isChecked()
    assert not win.exp_pdf_chk.isChecked()
    assert not win.exp_svg_chk.isChecked()
    assert not win.exp_report_png_chk.isChecked()
    assert not win.exp_report_pdf_chk.isChecked()

    primary = str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat"))
    overlay = str(_fixture("TestData", "NewCodebaseTests", "317RabiFullRun.mat"))
    win._load_dropped_files([primary, overlay])

    assert win.ctx.trace is not None
    assert win.ctx.trace.file_name == Path(primary).name
    assert Path(overlay).name in win.ctx.loaded_traces

    fake_result = FitResult(
        model_name="RabiShifted",
        param_names=["y0", "A", "f", "t0"],
        params=np.array([0.03, 0.02, 0.029, 24.5]),
        errors=np.array([0.001, 0.001, 0.0005, 1.0]),
        y_fit=win.ctx.y.copy(),
        r2=0.91,
        rmse=0.002,
        aic=1.0,
        bic=2.0,
        durbin_watson=1.1,
        bound_hits=[False, False, False, False],
        success=True,
        message="ok",
        selection_note="selected shifted single-frequency Rabi model",
    )
    win.ann_params_chk.setChecked(True)
    win.ann_metrics_chk.setChecked(True)
    win._apply_plot_controls_to_state()
    lines = win._annotation_lines_for_result(fake_result)
    assert any("R^2" in line for line in lines)
    assert any("Analysis mode:" in line for line in lines)
    assert any("Delay [ns]" in line for line in lines)
    assert "Recommended mode" in win.profile_hint_lbl.text()


def test_gui_drag_path_filter_accepts_mat_only():
    _app()
    win = SmartFitterMainWindow()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(r"C:\tmp\one.mat"), QUrl.fromLocalFile(r"C:\tmp\two.txt")])
    event = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    paths = win._drag_paths_from_event(event)
    assert [Path(p) for p in paths] == [Path(r"C:\tmp\one.mat")]


def test_gui_embedded_plot_views_defer_file_drops_to_window():
    _app()
    win = SmartFitterMainWindow()

    for plot_widget in (win.live_plot_widget, win.scan_workspace_plot):
        if plot_widget is not None:
            assert not plot_widget.acceptDrops()
            assert not plot_widget.viewport().acceptDrops()
    assert win.acceptDrops()


def test_gui_inspect_click_updates_point_readout():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    win.mask_mode_combo.setCurrentText("Inspect point")
    idx = 10
    event = SimpleNamespace(inaxes=win.ax_main, xdata=float(win.ctx.x[idx]), ydata=float(win.ctx.y[idx]))
    win._on_plot_click(event)

    assert win._selected_plot_point is not None
    assert abs(win._selected_plot_point["x"] - float(win.ctx.x[idx])) < 1e-9
    assert "Point readout" in win.point_readout_lbl.text()
    assert win.live_readout_lbl.text() == win.point_readout_lbl.text()

    point_count = len(win.ctx.x)
    win._on_plot_click(event)
    assert win._selected_plot_point is None
    assert not any(label in {"Point", "Selected point"} for label in win.ax_main.get_legend_handles_labels()[1])
    assert len(win.ctx.x) == point_count


def test_gui_live_plot_click_selection_pins_nearest_point():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    idx = 12
    win._select_point_from_plot(float(win.ctx.x[idx]), refresh=True)

    assert win._selected_plot_point is not None
    assert abs(win._selected_plot_point["x"] - float(win.ctx.x[idx])) < 1e-9
    assert "Point readout" in win.live_readout_lbl.text()
    assert win.live_selection_marker is not None
    assert any(line.get_label() == "Point" for line in win.ax_main.lines)


def test_gui_scan_visibility_and_summary_for_2d_scan():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    assert win.ctx.trace is not None
    assert win.ctx.trace.scan_dim == "scan2d"
    assert not win.scan_tools_box.isHidden()
    assert win.strategy_box.isHidden()
    assert win.btn_fit.isHidden()
    assert "Max point" in win.scan_summary_text.toPlainText()
    assert win._scan_colorbar is not None
    assert win._scan_colorbar_ax is not None

    x_edges, y_edges, extent = win._scan2d_extent(win.ctx.trace)
    assert np.allclose(win.ax_main.get_xlim(), (extent[0], extent[1]))
    assert np.allclose(win.ax_main.get_ylim(), (extent[2], extent[3]))
    pos = win.ax_main.get_position()
    fig_w, fig_h = win.fig.get_size_inches()
    rendered_aspect = (pos.width * fig_w) / (pos.height * fig_h)
    expected_aspect = abs((extent[1] - extent[0]) / (extent[3] - extent[2]))
    assert rendered_aspect == pytest.approx(expected_aspect, rel=0.03)

    win.scan_linecut_combo.setCurrentText("Best-point horizontal")
    win._refresh_plot_only()
    assert np.allclose(win.ax_main.get_xlim(), (extent[0], extent[1]))
    assert np.allclose(win.ax_main.get_ylim(), (extent[2], extent[3]))


def test_gui_2d_scan_draws_after_removing_twinned_observable_axis():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    win.show_signal_layer_chk.setChecked(True)
    win.show_reference_layer_chk.setChecked(True)
    win._refresh_plot_only()
    assert win._ax_secondary is not None
    win.canvas.draw()

    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))
    assert win._ax_secondary is None
    assert win.ax_main.get_adjustable() == "datalim"
    win.canvas.draw()

    _x_edges, _y_edges, extent = win._scan2d_extent(win.ctx.trace)
    assert np.allclose(win.ax_main.get_xlim(), (extent[0], extent[1]))
    assert np.allclose(win.ax_main.get_ylim(), (extent[2], extent[3]))


@pytest.mark.skipif(not JUNE5_2D_STAGE.exists(), reason="June 5 2D stage scan file is not available")
def test_gui_june5_stage_scan_export_extent_and_colorbar_layout():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(JUNE5_2D_STAGE))

    assert win.ctx.trace is not None
    assert win.ctx.trace.scan_dim == "scan2d"
    assert win.ctx.trace.z2d.shape == (41, 41)

    _x_edges, _y_edges, extent = win._scan2d_extent(win.ctx.trace)
    assert np.allclose(extent, (-452.5, -247.5, 547.5, 752.5))
    assert np.allclose(win.ax_main.get_xlim(), (extent[0], extent[1]))
    assert np.allclose(win.ax_main.get_ylim(), (extent[2], extent[3]))
    assert win._scan_colorbar is not None
    assert win._scan_colorbar_ax is not None

    main_pos = win.ax_main.get_position()
    cbar_pos = win._scan_colorbar_ax.get_position()
    assert cbar_pos.x0 > main_pos.x1
    assert cbar_pos.height == pytest.approx(main_pos.height)

    fig_w, fig_h = win.fig.get_size_inches()
    rendered_aspect = (main_pos.width * fig_w) / (main_pos.height * fig_h)
    assert rendered_aspect == pytest.approx(1.0, rel=0.03)


def test_gui_live_scan_click_selection_updates_cursor_and_crosshair():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    x0 = float(win.ctx.trace.x2d[0, 0])
    y0 = float(win.ctx.trace.y2d[0, 0])
    z0 = float(win.ctx.trace.z2d[0, 0])
    win._select_point_from_plot(x0, y0, refresh=True)

    assert win._scan_cursor == (x0, y0)
    assert win._selected_plot_point["z"] == z0
    assert "z=" in win.live_readout_lbl.text()
    assert win.live_selection_vline is not None
    assert win.live_selection_hline is not None


def test_gui_exclusion_workflow_uses_selected_point_and_ranges():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    initial_count = len(win.ctx.x)
    idx = 8
    win._select_point_from_plot(float(win.ctx.x[idx]), refresh=False)
    selected_x = float(win.ctx.x[idx])
    win.on_exclude_selected_point()

    assert len(win.ctx.x) == initial_count - 1
    assert win.mask_list.count() >= 1
    assert any("Point:" in win.mask_list.item(i).text() for i in range(win.mask_list.count()))

    win.mask_xmin.setText(str(selected_x - 1))
    win.mask_xmax.setText(str(selected_x + 1))
    win.on_add_exclusion_range()
    assert any("Range:" in win.mask_list.item(i).text() for i in range(win.mask_list.count()))

    win.mask_list.setCurrentRow(0)
    before_remove = win.mask_list.count()
    win.on_remove_selected_exclusion()
    assert win.mask_list.count() <= before_remove

    win.on_clear_exclusions()
    assert win.ctx.excluded_points == set()
    assert win.ctx.exclusion_ranges == []
    assert win.mask_list.item(0).text() == "No exclusions"


def test_gui_scan_mode_roundtrip_restores_contrast_and_ylabel():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "Z_Scan4pm.mat")))

    y_contrast = win.ctx.y.copy()
    win.mode_combo.setCurrentText("signal")
    assert win.ctx.trace is not None and win.ctx.trace.mode == "signal"
    assert "Signal" in win.ax_main.get_ylabel()

    win.mode_combo.setCurrentText("reference")
    assert win.ctx.trace is not None and win.ctx.trace.mode == "reference"
    assert "Reference" in win.ax_main.get_ylabel()

    win.mode_combo.setCurrentText("contrast")
    assert win.ctx.trace is not None and win.ctx.trace.mode == "contrast"
    assert np.allclose(win.ctx.y, y_contrast, equal_nan=True)
    assert "Contrast" in win.ax_main.get_ylabel()


def test_gui_scan_secondary_axis_appears_for_scan1d():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "Z_Scan4pm.mat")))
    win.scan_secondary_combo.setCurrentText("Signal")
    win._refresh_plot_only()

    assert win._ax_secondary is not None
    assert "Signal" in win._ax_secondary.get_ylabel()


def test_gui_export_x_limits_reset_when_roi_becomes_smaller():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi10us.mat")))

    old_limits = win.ax_main.get_xlim()
    win.toolbar.push_current()
    win.roi_min.setText("100")
    win.roi_max.setText("200")
    win._refresh_processed()

    new_limits = win.ax_main.get_xlim()
    assert new_limits != pytest.approx(old_limits)
    assert new_limits[0] == pytest.approx(95.0)
    assert new_limits[1] == pytest.approx(205.0)

    # "Home" must not resurrect the pre-ROI limits from the toolbar stack.
    win.toolbar.home()
    assert win.ax_main.get_xlim() == pytest.approx(new_limits)


def test_gui_export_x_limits_preserve_zoom_inside_current_scan():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi10us.mat")))

    win.ax_main.set_xlim(120.0, 160.0)
    win._refresh_plot_only()

    assert win.ax_main.get_xlim() == pytest.approx((120.0, 160.0))


def test_gui_export_inspect_crosshair_does_not_add_zero_to_axis_limits():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "Z_Scan4pm.mat")))

    initial_xlim = win.ax_main.get_xlim()
    initial_ylim = win.ax_main.get_ylim()
    idx = len(win.ctx.x) // 2
    win.mask_mode_combo.setCurrentText("Inspect point")
    win._on_plot_motion(
        SimpleNamespace(
            inaxes=win.ax_main,
            xdata=float(win.ctx.x[idx]),
            ydata=float(win.ctx.y[idx]),
        )
    )
    win.canvas.draw()

    assert win.ax_main.get_xlim() == pytest.approx(initial_xlim)
    assert win.ax_main.get_ylim() == pytest.approx(initial_ylim)
    assert win.ax_main.dataLim.x0 > 10000.0


def test_gui_t1_controls_and_equation_are_visible_for_t1():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "T1_20260324_041805.mat")))

    assert win.ctx.trace is not None
    assert win.ctx.trace.experiment_type == "T1"
    assert not win.t1_container.isHidden()
    assert win.model_combo.currentText() == "T1"
    win.t1_prep_combo.setCurrentText("Normalize then 1-data")
    win.t1_model_combo.setCurrentText("Stretched exponential")
    win._update_equation_display()
    assert "T_1" in win.equation_display.toPlainText() or "T1" in win.equation_display.toPlainText()


def test_gui_t1_normalized_fit_uses_transformed_plot_series():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "T1_20260324_041805.mat")))

    transformed = np.linspace(1.0, 0.0, len(win.ctx.y))
    result = FitResult(
        model_name="T1NormStretch",
        param_names=["y0", "A", "T1", "n"],
        params=np.array([0.0, 1.0, 120000.0, 1.2]),
        errors=np.array([0.01, 0.01, 1000.0, 0.1]),
        y_fit=transformed * 0.95,
        r2=0.88,
        rmse=0.01,
        aic=1.0,
        bic=2.0,
        durbin_watson=1.2,
        bound_hits=[False, False, False, False],
        success=True,
        message="ok",
        selection_note="selected normalized_1_minus_data T1 with stretched exponential model",
        extras={"plot_y": transformed, "plot_y_label": "Normalized 1 - data [a.u.]", "t1_prep_mode": "normalized_1_minus_data", "t1_model_family": "stretched"},
    )
    win.on_fit_finished(result)

    assert np.allclose(win.ctx.fit_target, transformed)
    assert win.ax_main.get_ylabel() == "Normalized 1 - data [a.u.]"


def test_gui_refresh_plot_only_clears_stale_fit_arrays_after_data_length_changes():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    result = FitResult(
        model_name="RabiShifted",
        param_names=["y0", "A", "f", "t0"],
        params=np.array([0.03, 0.02, 0.029, 24.5]),
        errors=np.array([0.001, 0.001, 0.0005, 1.0]),
        y_fit=win.ctx.y.copy(),
        r2=0.91,
        rmse=0.002,
        aic=1.0,
        bic=2.0,
        durbin_watson=1.1,
        bound_hits=[False, False, False, False],
        success=True,
        message="ok",
        selection_note="selected shifted single-frequency Rabi model",
    )
    win.on_fit_finished(result)

    win.ctx.x = win.ctx.x[::2]
    win.ctx.y = win.ctx.y[::2]
    win._refresh_plot_only()

    assert win.ctx.fit_result is None
    assert win.ctx.fit_target is None
    assert "processed data changed" in win.ctx.status_reason


def test_gui_preprocess_change_invalidates_existing_fit():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    result = FitResult(
        model_name="RabiShifted",
        param_names=["y0", "A", "f", "t0"],
        params=np.array([0.03, 0.02, 0.029, 24.5]),
        errors=np.array([0.001, 0.001, 0.0005, 1.0]),
        y_fit=win.ctx.y.copy(),
        r2=0.91,
        rmse=0.002,
        aic=1.0,
        bic=2.0,
        durbin_watson=1.1,
        bound_hits=[False, False, False, False],
        success=True,
        message="ok",
        selection_note="selected shifted single-frequency Rabi model",
    )
    win.on_fit_finished(result)

    win.bin_spin.setValue(2)
    win._refresh_plot_only()

    assert win.ctx.fit_result is None
    assert win.ctx.fit_target is None


def test_gui_workspace_modes_batch_summary_and_results(tmp_path):
    _app()
    win = SmartFitterMainWindow()
    messages = []
    win._message = lambda title, text: messages.append((title, text))

    assert win.mode_stack.count() == 4
    win._set_workspace_mode(1)
    assert win.mode_stack.currentIndex() == 1

    win._on_batch_run()
    assert messages and messages[-1][0] == "Batch"

    summary = tmp_path / "batch_summary.csv"
    summary.write_text(
        "file,status,model,r2,rmse,status_reason,path\n"
        "one.mat,PASS,Rabi,0.99,0.001,ok,C:/tmp/one.mat\n"
        "two.mat,WARN,ODMR,0.82,0.02,low r2,C:/tmp/two.mat\n",
        encoding="utf-8",
    )
    win.batch_output_dir_edit.setText(str(tmp_path))
    win._load_batch_summary_from_dir()
    assert win.batch_results_table.rowCount() == 2

    win.batch_filter_combo.setCurrentText("PASS")
    assert win.batch_results_table.rowCount() == 1
    assert win.batch_results_table.item(0, 0).text() == "one.mat"

    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))
    fake_result = FitResult(
        model_name="RabiShifted",
        param_names=["y0", "A", "f", "t0"],
        params=np.array([0.03, 0.02, 0.029, 24.5]),
        errors=np.array([0.001, 0.001, 0.0005, 1.0]),
        y_fit=win.ctx.y.copy(),
        r2=0.91,
        rmse=0.002,
        aic=1.0,
        bic=2.0,
        durbin_watson=1.1,
        bound_hits=[False, False, False, False],
        success=True,
        message="ok",
        selection_note="selected shifted single-frequency Rabi model",
    )
    win.on_fit_finished(fake_result)
    win._set_workspace_mode(3)
    assert win.mode_stack.currentIndex() == 3
    assert "Analysis mode: RabiShifted" in win.results_text.toPlainText()
    assert win.results_table.rowCount() > 0


def test_gui_options_drawers_defaults_and_tooltips():
    _app()
    win = SmartFitterMainWindow()

    assert win.data_box.isChecked()
    assert not win.overlay_box.isChecked()
    assert not win.prep_box.isChecked()
    assert win.strategy_box.isChecked()
    assert not win.mask_box.isChecked()
    assert not win.plot_box.isChecked()
    assert not win.annotation_box.isChecked()
    assert win.export_box.isChecked()
    assert not win.mask_mode_combo.isVisible()

    required = [
        win.btn_load,
        win.profile_combo,
        win.mode_combo,
        win.model_combo,
        win.smooth_spin,
        win.btn_fit,
        win.btn_robust_fit,
        win.btn_exclude_selected,
        win.show_confidence_chk,
        win.show_signal_layer_chk,
        win.annotation_mode_combo,
        win.exp_png_chk,
        win.batch_run_btn,
        win.scan_linecut_combo,
    ]
    assert all(widget.toolTip() for widget in required)
    assert "Long damped Rabi" in [win.rabi_mode_combo.itemText(i) for i in range(win.rabi_mode_combo.count())]
    win.rabi_mode_combo.setCurrentText("Long damped Rabi")
    assert "long decayed scans" in win.rabi_model_help_lbl.text()

    win.mask_box.setChecked(True)
    assert not win.btn_exclude_selected.isHidden()
    assert not win.mask_list.isHidden()


def test_gui_metadata_panel_and_observable_layers_do_not_change_mode():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    assert "run=checkpoint" in win.metadata_status_lbl.text()
    assert win.metadata_progress_bar.value() > 0
    assert win.ctx.trace is not None and win.ctx.trace.mode == "contrast"

    win.show_signal_layer_chk.setChecked(True)
    win.show_reference_layer_chk.setChecked(True)
    win._refresh_plot_only()

    assert win.ctx.trace is not None and win.ctx.trace.mode == "contrast"
    assert win._ax_secondary is not None
    assert "Counts" in win._ax_secondary.get_ylabel()
    assert max(line.get_ydata().max() for line in win.ax_main.lines if len(line.get_ydata())) < 2.0


def test_gui_plot_toolbar_syncs_with_drawer_controls():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    win.quick_signal_btn.setChecked(True)
    win.quick_reference_btn.setChecked(True)
    win.quick_iterations_btn.setChecked(True)
    win.quick_std_btn.setChecked(True)
    win.quick_std_mode_combo.setCurrentText("Error bars")

    assert win.show_signal_layer_chk.isChecked()
    assert win.show_reference_layer_chk.isChecked()
    assert win.show_iteration_layer_chk.isChecked()
    assert win.show_iteration_std_chk.isChecked()
    assert win.iteration_std_mode_combo.currentText() == "Error bars"

    win.show_reference_layer_chk.setChecked(False)
    win._apply_plot_controls_to_state()
    assert not win.quick_reference_btn.isChecked()


def test_gui_rabi_envelope_control_plot_and_annotation():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    fake_result = FitResult(
        model_name="RabiCosine",
        param_names=["y0", "A", "T", "beta", "f", "phi"],
        params=np.array([0.035, 0.025, 520.0, 1.2, 0.028, 0.0]),
        errors=np.array([0.001, 0.001, 40.0, 0.2, 0.0005, 0.1]),
        y_fit=win.ctx.y.copy(),
        r2=0.94,
        rmse=0.002,
        aic=1.0,
        bic=2.0,
        durbin_watson=1.2,
        bound_hits=[False, False, False, False, False, False],
        success=True,
        message="ok",
        extras={
            "rabi_envelope": {
                "success": True,
                "message": "ok",
                "params": {"floor": 0.001, "amp": 0.025, "T2rho": 520.0, "beta": 1.2},
                "errors": {"floor": 0.0001, "amp": 0.001, "T2rho": 40.0, "beta": 0.2},
                "r2": 0.9,
                "rmse": 0.001,
                "baseline_mode": "constant",
                "baseline_intercept": 0.035,
                "baseline_slope": 0.0,
            }
        },
    )

    win.on_fit_finished(fake_result)
    win._apply_plot_controls_to_state()
    assert win.plot_opts.show_rabi_envelope
    assert any(line.get_label() in {"Envelope", "Rabi envelope"} for line in win.ax_main.lines)

    lines = win._annotation_lines_for_result(fake_result)
    assert any("T2rho [ns]" in line for line in lines)
    win.ann_nv_metrics_chk.setChecked(False)
    win._apply_plot_controls_to_state()
    lines = win._annotation_lines_for_result(fake_result)
    assert any("T2rho [ns]" in line for line in lines)
    assert not any("Rabi frequency" in line for line in lines)
    win.ann_t2rho_chk.setChecked(False)
    win._apply_plot_controls_to_state()
    lines = win._annotation_lines_for_result(fake_result)
    assert not any("T2rho [ns]" in line for line in lines)

    win.show_rabi_envelope_chk.setChecked(False)
    win._refresh_plot_only()
    assert not any(line.get_label() in {"Envelope", "Rabi envelope"} for line in win.ax_main.lines)


def test_gui_multi_iteration_overlay_and_point_readout():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    assert win._iteration_count() > 1
    win._selected_iteration_indices = {0, 1}
    win.show_iteration_layer_chk.setChecked(True)
    win.show_iteration_mean_chk.setChecked(False)
    win.show_iteration_std_chk.setChecked(False)
    win._refresh_plot_only()
    idx = 5
    win._select_point_from_plot(float(win.ctx.x[idx]), refresh=True)

    iter_lines = [line for line in win.ax_main.lines if line.get_label().startswith("Iter ")]
    assert len(iter_lines) >= 2
    assert "iter1=" in win.point_readout_lbl.text()
    assert "iter2=" in win.point_readout_lbl.text()


def test_gui_iteration_overlay_follows_signal_and_reference_modes():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    for mode in ("signal", "reference"):
        win.mode_combo.setCurrentText(mode)
        win._selected_iteration_indices = {0, 1}
        win.show_iteration_layer_chk.setChecked(True)
        win.show_iteration_mean_chk.setChecked(True)
        win.show_iteration_std_chk.setChecked(False)
        win._refresh_plot_only()

        assert win._iteration_count() == 17
        assert win.iteration_select_btn.isEnabled()
        iter_lines = [line for line in win.ax_main.lines if line.get_label().startswith("Iter ")]
        assert len(iter_lines) >= 2
        assert any(line.get_label() in {"Iter mean", "Iteration mean"} for line in win.ax_main.lines)


def test_gui_iteration_mean_std_band_and_errorbars():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    win._selected_iteration_indices = {0, 1, 2}
    win.show_iteration_layer_chk.setChecked(False)
    win.show_iteration_mean_chk.setChecked(True)
    win.show_iteration_std_chk.setChecked(True)
    win.iteration_std_mode_combo.setCurrentText("Band")
    win._refresh_plot_only()

    assert any(collection.get_label() in {"Iter +/- std", "Iteration +/- std"} for collection in win.ax_main.collections)

    win.iteration_std_mode_combo.setCurrentText("Error bars")
    win._refresh_plot_only()

    assert win.ax_main.containers


def test_gui_iteration_dialog_disabled_without_matrix():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    assert win._iteration_count() == 0
    assert not win.iteration_select_btn.isEnabled()
    assert not win.quick_iteration_select_btn.isEnabled()


def test_gui_iteration_dialog_all_none_selection():
    _app()
    dlg = IterationSelectionDialog(None, count=4, selected={1}, show_mean=True)

    assert dlg.selected_indices() == {1}
    assert dlg.show_mean()
    dlg._select_all()
    assert dlg.selected_indices() == {0, 1, 2, 3}
    dlg._select_none()
    assert dlg.selected_indices() == set()


def test_gui_scan_explorer_workspace_syncs_with_loaded_scan():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    win._set_workspace_mode(2)
    assert win.mode_stack.currentIndex() == 2
    assert "Max point" in win.scan_workspace_summary.toPlainText()

    win.scan_workspace_linecut_combo.setCurrentText("Best-point horizontal")
    assert win.scan_linecut_combo.currentText() == "Best-point horizontal"


@pytest.mark.skipif(not TDD_ROOT.exists(), reason="429_TDD data directory is not available")
def test_gui_deer_profiles_and_plot_only_position():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    win._load_file(str(TDD_ROOT / "May12" / "1820_May12_DEERDecayScanLogUpdatedNoRebuildSequenceAtEachPoint.mat"))
    assert win.ctx.trace is not None and win.ctx.trace.experiment_type == "DEERDecay"
    assert win.model_combo.currentText() == "DEERDecay"
    assert not win.strategy_box.isHidden()

    win._load_file(str(TDD_ROOT / "May13" / "0637_May13_DEERPositionSweep.mat"))
    assert win.ctx.trace is not None and win.ctx.trace.experiment_type == "DEERPosition"
    assert win.ctx.trace.scan_dim == "scan1d"
    assert win.strategy_box.isHidden()


def test_gui_selected_marker_toggles_and_clears_without_changing_data():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi10us.mat")))
    original_count = len(win.ctx.x)
    x0 = float(win.ctx.x[0])

    win._select_point_from_plot(x0, refresh=True)
    assert win._selected_plot_point is not None
    assert win.clear_marker_btn.isEnabled()
    assert any(label in {"Point", "Selected point"} for label in win.ax_main.get_legend_handles_labels()[1])

    win._select_point_from_plot(x0, refresh=True)
    assert win._selected_plot_point is None
    assert not win.clear_marker_btn.isEnabled()
    assert not any(label in {"Point", "Selected point"} for label in win.ax_main.get_legend_handles_labels()[1])
    assert len(win.ctx.x) == original_count

    win._select_point_from_plot(x0, refresh=False)
    win.clear_selected_marker(refresh=False)
    assert win._selected_plot_point is None
    assert len(win.ctx.x) == original_count


def test_gui_copy_export_figure_places_image_on_clipboard():
    app = _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi10us.mat")))
    win.copy_export_figure()
    assert not app.clipboard().image().isNull()
