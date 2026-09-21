import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, Qt, QTimer, QUrl
from PySide6.QtGui import QDragEnterEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog

from nvfit.fit_engine import FitResult
from nvfit.app_state import FitOutcome, FitRequest
from nvfit.gui_app import IterationSelectionDialog, SmartFitterMainWindow
from nvfit.scan_spots import SpotFit


TDD_ROOT = Path(r"D:\BacklundLabResearch\Data\Experiments\429_TDD")
JUNE5_2D_STAGE = Path(r"C:\Users\Div\Box\Backlund lab shared\data\Div\June5th_2DStage_SPC1_NVFinding3Large200um5umstep.mat")


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _fixture(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath(*parts)


def _write_gui_dual_snapshot(path: Path, *, dimension: int, reference_only: bool = False) -> None:
    if dimension == 1:
        x = np.arange(4, dtype=float)
        values1 = np.empty((4, 1), dtype=object)
        values2 = np.empty((4, 1), dtype=object)
        ref1 = 100.0 + x
        ref2 = 60.0 + x
        sig1 = np.zeros_like(x) if reference_only else 70.0 + x
        sig2 = np.zeros_like(x) if reference_only else 20.0 + 2 * x
        for i in range(4):
            values1[i, 0] = np.array([ref1[i], sig1[i]])
            values2[i, 0] = np.array([ref2[i], sig2[i]])

        def metric(y_values: np.ndarray, label: str) -> dict:
            return {"x": x, "y": y_values, "xLabel": "Stage Z", "yLabel": label}

        scan_info = {
            "bounds": np.array([0.0, 3.0]), "stepSize": 1.0, "nSteps": 4,
            "parameter": "stage z", "identifier": "PI Stage",
        }
        analysis1 = {
            "dimension": 1,
            "averageContrast": metric((ref1 - sig1) / ref1, "Contrast"),
            "averageSignal": metric(sig1, "Signal [counts]"),
            "averageReference": metric(ref1, "Reference [counts]"),
        }
        analysis2 = {
            "dimension": 1,
            "averageContrast": metric((ref2 - sig2) / ref2, "Contrast"),
            "averageSignal": metric(sig2, "Signal [counts]"),
            "averageReference": metric(ref2, "Reference [counts]"),
        }
        iterations = np.ones(4)
        failed = np.zeros((4, 1))
    else:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 11.0])
        ref1 = np.array([[100.0, 101.0, 102.0], [103.0, 104.0, 105.0]])
        sig1 = np.zeros_like(ref1) if reference_only else np.array([[80.0, 79.0, 78.0], [77.0, 76.0, 75.0]])
        ref2 = np.array([[50.0, 51.0, 52.0], [53.0, 54.0, 55.0]])
        sig2 = np.zeros_like(ref2) if reference_only else np.array([[10.0, 12.0, 14.0], [16.0, 18.0, 20.0]])
        values1 = np.empty((2, 3), dtype=object)
        values2 = np.empty((2, 3), dtype=object)
        for index in np.ndindex(values1.shape):
            values1[index] = np.array([ref1[index], sig1[index]])
            values2[index] = np.array([ref2[index], sig2[index]])

        def metric(z_values: np.ndarray, label: str) -> dict:
            return {"x": x, "y": y, "z": z_values, "xLabel": "Stage X", "yLabel": "Stage Y", "zLabel": label}

        scan_info = np.empty(2, dtype=object)
        scan_info[0] = {"bounds": np.array([0.0, 2.0]), "stepSize": 1.0, "nSteps": 3, "parameter": "stage x", "identifier": "PI Stage"}
        scan_info[1] = {"bounds": np.array([10.0, 11.0]), "stepSize": 1.0, "nSteps": 2, "parameter": "stage y", "identifier": "PI Stage"}
        analysis1 = {
            "dimension": 2,
            "averageContrast": metric((ref1 - sig1) / ref1, "Contrast"),
            "averageSignal": metric(sig1, "Signal [counts]"),
            "averageReference": metric(ref1, "Reference [counts]"),
        }
        analysis2 = {
            "dimension": 2,
            "averageContrast": metric((ref2 - sig2) / ref2, "Contrast"),
            "averageSignal": metric(sig2, "Signal [counts]"),
            "averageReference": metric(ref2, "Reference [counts]"),
        }
        iterations = np.ones((2, 3))
        failed = np.zeros((2, 3))

    savemat(
        path,
        {
            "Data": {
                "values": values1, "detector2Values": values2,
                "iteration": iterations, "failedPoints": failed, "detectorMode": "dual",
            },
            "scanInfo": scan_info,
            "params": {"detectorMode": "dual"},
            "progress": {"status": "completed", "completedIterations": 1, "targetIterations": 1},
            "metadata": {"runStatus": "completed", "saveType": "final"},
            "analysis": {**analysis1, "detector2": analysis2},
        },
    )


def test_startup_empty_state_and_loaded_action_availability():
    _app()
    win = SmartFitterMainWindow()

    assert win.plot_stack.currentIndex() == 0
    assert not win.command_fit_btn.isEnabled()
    assert not win.btn_fit.isEnabled()
    assert not win.btn_save.isEnabled()
    assert not win.copy_figure_btn.isEnabled()
    assert win.profile_combo.currentText() == "Auto"
    assert win.model_combo.currentText() == "LineScan"
    assert win.strategy_box.isHidden()

    assert win._load_file(str(_fixture("TestData", "Rabi_2206GHz4ns.mat")))
    assert win.plot_stack.currentIndex() == 1
    assert win.command_fit_btn.isEnabled()
    assert win.btn_fit.isEnabled()
    assert win.btn_save.isEnabled()
    assert win.copy_figure_btn.isEnabled()


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
    assert "Recommended mode" in win.profile_hint_lbl.toolTip()


def test_gui_drag_path_filter_accepts_mat_only():
    _app()
    win = SmartFitterMainWindow()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(r"C:\tmp\one.mat"), QUrl.fromLocalFile(r"C:\tmp\two.txt")])
    event = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    paths = win._drag_paths_from_event(event)
    assert [Path(p) for p in paths] == [Path(r"C:\tmp\one.mat")]


def test_gui_uses_one_wysiwyg_plot_and_window_owns_file_drops():
    _app()
    win = SmartFitterMainWindow()

    assert win.live_plot_widget is None
    assert win.scan_workspace_plot is None
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


def test_gui_wysiwyg_plot_click_selection_pins_nearest_point_without_legend_entry():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    idx = 12
    win._select_point_from_plot(float(win.ctx.x[idx]), refresh=True)

    assert win._selected_plot_point is not None
    assert abs(win._selected_plot_point["x"] - float(win.ctx.x[idx])) < 1e-9
    assert "Point readout" in win.live_readout_lbl.text()
    assert any(line.get_label() == "_nolegend_" for line in win.ax_main.lines)
    assert not any(label in {"Point", "Selected point"} for label in win.ax_main.get_legend_handles_labels()[1])


def test_gui_ctrl_click_keeps_multiple_1d_points_highlighted():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    first = 4
    second = 18
    first_event = SimpleNamespace(
        inaxes=win.ax_main,
        xdata=float(win.ctx.x[first]),
        ydata=float(win.ctx.y[first]),
        key=None,
    )
    additive_event = SimpleNamespace(
        inaxes=win.ax_main,
        xdata=float(win.ctx.x[second]),
        ydata=float(win.ctx.y[second]),
        key="control",
    )
    win._on_plot_click(first_event)
    win._on_plot_click(additive_event)

    assert len(win._selected_plot_points) == 2
    assert [point["x"] for point in win._selected_plot_points] == [
        float(win.ctx.x[first]),
        float(win.ctx.x[second]),
    ]
    assert "2 points selected" in win.point_readout_lbl.text()
    assert sum(line.get_label() == "_nolegend_" for line in win.ax_main.lines) >= 2

    # The same modifier toggles one point off without clearing the other.
    win._on_plot_click(additive_event)
    assert len(win._selected_plot_points) == 1
    assert win._selected_plot_point["x"] == pytest.approx(float(win.ctx.x[first]))


def test_gui_spot_table_clipboard_is_powerpoint_ready_and_orders_xy():
    app = _app()
    win = SmartFitterMainWindow()
    spot = SpotFit(
        spot_id=1,
        polarity="bright",
        center_x=227.0,
        center_y=598.0,
        sigma_major=0.8243 / np.sqrt(2.0 * np.log(2.0)),
        sigma_minor=0.7259 / np.sqrt(2.0 * np.log(2.0)),
        angle_deg=10.014,
        amplitude=100.0,
        background=2.0,
        contrast_percent=50.0,
        snr=10.0,
        r_squared=0.9666733,
        integrated_signal=1000.0,
        component_pixels=12,
        fit_success=True,
        source="Reference",
    )
    win._scan_spot_results = {"detector1": [spot]}

    rows = win._spot_presentation_rows()
    assert rows[0] == ["#", "Detector", "Center X, Y", "R major × minor", "R²"]
    assert rows[1][2] == "227, 598"
    assert rows[1][3] == "0.8243 × 0.7259 @ 10°"
    assert rows[1][4] == "0.9667"

    win._copy_spot_table()
    text = app.clipboard().text()
    assert text.splitlines()[0] == "#\tDetector\tCenter X, Y\tR major × minor\tR²"
    assert "227, 598" in text
    assert "detector_id" not in text
    assert "<table" in win._spot_presentation_html(rows)


def test_gui_spot_table_converts_reversed_stored_scan_axes_to_physical_xy():
    _app()
    win = SmartFitterMainWindow()
    win._primary_dataset = SimpleNamespace(
        detectors={"detector1": SimpleNamespace(detector_id="detector1", scan_axes=("y", "x"))}
    )
    spot = SpotFit(
        spot_id=1,
        polarity="bright",
        center_x=227.0,
        center_y=598.0,
        sigma_major=0.5,
        sigma_minor=0.4,
        angle_deg=0.0,
        amplitude=1.0,
        background=0.0,
        contrast_percent=1.0,
        snr=1.0,
        r_squared=0.5,
        integrated_signal=1.0,
        component_pixels=4,
        fit_success=True,
    )
    win._scan_spot_results = {"detector1": [spot]}

    assert win._spot_presentation_rows()[1][2] == "598, 227"


def test_gui_scan_visibility_and_summary_for_2d_scan():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    assert win.ctx.trace is not None
    assert win.ctx.trace.scan_dim == "scan2d"
    assert not win.scan_quick_bar.isHidden()
    assert not win.scan_tools_box.isHidden()
    assert win.strategy_box.isHidden()
    assert win.btn_fit.isHidden()
    assert "Max point" in win.scan_summary_text.toPlainText()
    assert win._scan_colorbar is not None
    assert win._scan_colorbar_ax is not None

    x_edges, y_edges, extent = win._scan2d_extent(win.ctx.trace)
    assert np.allclose(win.ax_main.get_xlim(), (extent[0], extent[1]))
    assert np.allclose(win.ax_main.get_ylim(), (extent[2], extent[3]))


def test_gui_2d_quick_controls_update_color_limits_and_view():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))
    assert win.ctx.trace.experiment_type == "Scan2D"
    assert win.scan_swap_axes_chk.isChecked()
    assert "Stage X" in win.scan_swap_axes_chk.text()
    assert "Stage Z" in win.scan_swap_axes_chk.text()
    assert "Stage Z" in win.ax_main.get_xlabel()
    assert "Stage X" in win.ax_main.get_ylabel()
    swapped_extent = win._scan2d_extent(win.ctx.trace)[2]
    win.scan_swap_axes_chk.setChecked(False)
    assert "Stage X" in win.ax_main.get_xlabel()
    assert "Stage Z" in win.ax_main.get_ylabel()
    unswapped_extent = win._scan2d_extent(win.ctx.trace)[2]
    assert swapped_extent == pytest.approx((unswapped_extent[2], unswapped_extent[3], unswapped_extent[0], unswapped_extent[1]))
    win.scan_swap_axes_chk.setChecked(True)
    win.scan_pixel_geometry_combo.setCurrentText("Physical pixel size")
    win._refresh_plot_only()

    finite = np.asarray(win.ctx.trace.z2d, dtype=float)
    finite = finite[np.isfinite(finite)]
    low, high = [float(value) for value in np.percentile(finite, [10, 90])]
    win.scan_colormap_combo.setCurrentText("magma")
    win.scan_scale_combo.setCurrentText("Manual")
    win.scan_vmin_edit.setText(f"{low:.12g}")
    win.scan_vmax_edit.setText(f"{high:.12g}")
    win._apply_manual_scan_color_limits()

    mesh = win.ax_main.collections[0]
    assert mesh.get_cmap().name == "magma"
    assert mesh.get_clim() == pytest.approx((low, high))
    assert win.scan_low_percentile_spin.isEnabled() is False

    _x_edges, _y_edges, extent = win._scan2d_extent(win.ctx.trace)
    xlim = (extent[0] + 0.15 * (extent[1] - extent[0]), extent[1] - 0.15 * (extent[1] - extent[0]))
    ylim = (extent[2] + 0.20 * (extent[3] - extent[2]), extent[3] - 0.20 * (extent[3] - extent[2]))
    for edit, value in ((win.scan_xmin_edit, xlim[0]), (win.scan_xmax_edit, xlim[1]), (win.scan_ymin_edit, ylim[0]), (win.scan_ymax_edit, ylim[1])):
        edit.setText(f"{value:.12g}")
    win._apply_scan_view_limits()
    assert win.scan_pixel_geometry_combo.currentText() == "Physical pixel size"
    assert win.ax_main.get_xlim() == pytest.approx(xlim)
    assert win.ax_main.get_ylim() == pytest.approx(ylim)

    before_xspan = np.ptp(win.ax_main.get_xlim())
    event = SimpleNamespace(inaxes=win.ax_main, xdata=float(np.mean(xlim)), ydata=float(np.mean(ylim)), button="up")
    win._on_scan_scroll(event)
    assert np.ptp(win.ax_main.get_xlim()) < before_xspan

    win._reset_scan_view()
    assert win.ax_main.get_xlim() == pytest.approx((extent[0], extent[1]))
    assert win.ax_main.get_ylim() == pytest.approx((extent[2], extent[3]))
    pos = win.ax_main.get_position()
    fig_w, fig_h = win.fig.get_size_inches()
    rendered_aspect = (pos.width * fig_w) / (pos.height * fig_h)
    expected_aspect = abs((extent[1] - extent[0]) / (extent[3] - extent[2]))
    assert rendered_aspect == pytest.approx(expected_aspect, rel=0.03)

    win.scan_linecut_combo.setCurrentText("Best-point horizontal")
    win._refresh_plot_only()
    assert np.allclose(win.ax_main.get_xlim(), (extent[0], extent[1]))
    assert np.allclose(win.ax_main.get_ylim(), (extent[2], extent[3]))


def test_gui_2d_pixel_geometry_survives_axis_swap_for_rectangular_map():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    trace = win.ctx.trace
    assert trace is not None and trace.x2d is not None
    # Make the 41-column first scan axis use 2-unit pixels while the
    # 11-column second axis retains its 5-unit pixels.
    first_axis = float(trace.x2d[0, 0]) + 2.0 * np.arange(trace.x2d.shape[1])
    trace.x2d = np.tile(first_axis, (trace.x2d.shape[0], 1))

    for geometry in ("Physical pixel size", "Square pixels"):
        win.scan_pixel_geometry_combo.setCurrentText(geometry)
        for swapped in (False, True):
            win.scan_swap_axes_chk.setChecked(swapped)
            win._refresh_plot_only()
            win.canvas.draw()

            x, y, z, _x_label, _y_label = win._scan2d_display_data(trace)
            pos = win.ax_main.get_position()
            fig_w, fig_h = win.fig.get_size_inches()
            screen_cell_ratio = ((pos.width * fig_w) / z.shape[1]) / ((pos.height * fig_h) / z.shape[0])
            expected = 1.0 if geometry == "Square pixels" else abs(float(np.median(np.diff(x))) / float(np.median(np.diff(y))))
            assert screen_cell_ratio == pytest.approx(expected, rel=0.03)

            _x_edges, _y_edges, extent = win._scan2d_extent(trace)
            assert win.ax_main.get_xlim() == pytest.approx((extent[0], extent[1]))
            assert win.ax_main.get_ylim() == pytest.approx((extent[2], extent[3]))


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
    assert win.ax_main.get_adjustable() == "box"
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
    win.scan_pixel_geometry_combo.setCurrentText("Physical pixel size")
    win._refresh_plot_only()

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


def test_gui_scan_click_selection_updates_cursor_on_wysiwyg_plot():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    x0 = float(win.ctx.trace.x2d[0, 0])
    y0 = float(win.ctx.trace.y2d[0, 0])
    z0 = float(win.ctx.trace.z2d[0, 0])
    win._select_point_from_plot(y0, x0, refresh=True)

    assert win._scan_cursor == (y0, x0)
    assert win._selected_plot_point["z"] == z0
    assert "z=" in win.live_readout_lbl.text()
    assert any(np.allclose(line.get_xdata(), [y0, y0]) for line in win.ax_main.lines)


def test_gui_cursor_linecuts_leave_navigation_mode_and_follow_selected_cell():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    win.scan_pan_btn.click()
    assert "pan" in str(win.toolbar.mode).lower()

    win.scan_linecut_combo.setCurrentText("Cursor horizontal")
    assert not str(win.toolbar.mode)
    assert not win.scan_pan_btn.isChecked()
    assert not win.scan_zoom_btn.isChecked()

    row, col = 3, 7
    x0 = float(win.ctx.trace.x2d[row, col])
    y0 = float(win.ctx.trace.y2d[row, col])
    win._on_plot_click(SimpleNamespace(dblclick=False, inaxes=win.ax_main, xdata=y0, ydata=x0))

    assert win._scan_cursor == (y0, x0)
    assert win.ax_res.get_visible()
    assert np.allclose(win.ax_res.lines[0].get_xdata(), win.ctx.trace.y2d[:, col])
    assert np.allclose(win.ax_res.lines[0].get_ydata(), win.ctx.trace.z2d[:, col])

    win.scan_linecut_combo.setCurrentText("Cursor vertical")
    assert win.ax_res.get_visible()
    assert np.allclose(win.ax_res.lines[0].get_xdata(), win.ctx.trace.x2d[row, :])
    assert np.allclose(win.ax_res.lines[0].get_ydata(), win.ctx.trace.z2d[row, :])


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

    assert win.mode_stack.count() == 2
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
    win._set_workspace_mode(0)
    assert win.mode_stack.currentIndex() == 0
    assert "Analysis mode: RabiShifted" in win.summary_text.toPlainText()
    assert not win.results_box.isHidden()


def test_gui_options_drawers_defaults_and_tooltips():
    _app()
    win = SmartFitterMainWindow()

    assert win.data_box.isHidden()
    assert not win.overlay_box.isCheckable()
    assert not win.strategy_box.isCheckable()
    assert not win.param_box.isCheckable()
    assert win.prep_box.isHidden()
    assert win.advanced_box.isHidden()
    assert win.plot_box.isHidden()
    assert win.annotation_box.isHidden()
    assert win.export_box.isHidden()
    assert win.analysis_tools_btn.isVisibleTo(win.left_panel)
    assert win.file_details_btn.toolTip()
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

    win.advanced_box.setChecked(True)
    assert not win.btn_exclude_selected.isHidden()
    assert not win.mask_list.isHidden()


def test_gui_layout_is_two_workspaces_without_horizontal_panel_scrolling():
    app = _app()
    win = SmartFitterMainWindow()
    win.resize(1366, 768)
    win.show()
    app.processEvents()

    assert win.mode_stack.count() == 2
    assert win.left_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert win.right_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert win.live_plot_widget is None
    assert sum(win.main_splitter.sizes()) <= win.main_splitter.width() + 4


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


def test_gui_auto_detection_updates_each_new_primary_instead_of_sticking_on_ramsey():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    win._load_file(str(_fixture("TestData", "RamseyAtResonance2203.mat")))
    assert win.profile_combo.currentText() == "Auto"
    assert win.model_combo.currentText() == "RamseyAuto"

    win._load_file(str(_fixture("TestData", "Rabi_2206GHz4ns.mat")))
    assert win.profile_combo.currentText() == "Auto"
    assert win.model_combo.currentText() == "Rabi"


def test_gui_overlay_registry_starts_primary_only_and_explicit_selection_fits_combined_range():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    long_path = _fixture("TestData", "Rabi10us.mat")
    short_path = _fixture("TestData", "Rabi_2206GHz4ns.mat")

    win._load_file(str(long_path))
    win._load_file(str(short_path))

    assert {long_path.name, short_path.name} <= set(win.ctx.loaded_traces)
    assert win.ctx.trace.file_name == short_path.name
    assert win._checked_overlay_names() == set()
    assert not any(descriptor.series_id.startswith("overlay:") for descriptor in win._series_registry)

    win.on_show_all_overlays()
    assert win._checked_overlay_names() == {long_path.name}
    assert any(descriptor.series_id.startswith("overlay:") for descriptor in win._series_registry)
    assert win.ax_main.get_xlim()[1] > 4500.0

    win.on_show_primary_only()
    assert win._checked_overlay_names() == set()
    assert win.ax_main.get_xlim()[1] < 500.0


def test_gui_mixed_dimension_registry_keeps_2d_primary_visible_and_selectable():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    first_1d = _fixture("TestData", "NewCodebaseTests", "Z_Scan4pm.mat")
    second_1d = _fixture("TestData", "Rabi_2206GHz4ns.mat")
    scan_2d = _fixture("TestData", "1305_XZScan.mat")

    win._load_dropped_files([str(first_1d), str(second_1d)])
    assert win._checked_overlay_names() == {second_1d.name}

    win._load_dropped_files([str(scan_2d)])
    win.canvas.draw()

    assert win.ctx.trace is not None and win.ctx.trace.file_name == scan_2d.name
    assert win.ctx.trace.scan_dim == "scan2d"
    assert len(win.ax_main.collections) == 1
    assert win._scan_colorbar is not None
    assert win._checked_overlay_names() == set()
    assert {first_1d.name, second_1d.name, scan_2d.name} <= set(win.ctx.loaded_traces)

    first_item = next(
        win.overlay_list.item(index)
        for index in range(win.overlay_list.count())
        if win._overlay_item_name(win.overlay_list.item(index)) == first_1d.name
    )
    assert not bool(first_item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
    win.overlay_box.setChecked(True)
    win.overlay_list.setCurrentItem(first_item)
    win.btn_plot_selected_file.click()
    win.canvas.draw()

    assert win.ctx.trace is not None and win.ctx.trace.file_name == first_1d.name
    assert win.ctx.trace.scan_dim == "scan1d"
    assert len(win.ax_main.lines) >= 1
    scan_item = next(
        win.overlay_list.item(index)
        for index in range(win.overlay_list.count())
        if win._overlay_item_name(win.overlay_list.item(index)) == scan_2d.name
    )
    assert not bool(scan_item.flags() & Qt.ItemFlag.ItemIsUserCheckable)


def test_gui_reference_only_primary_defaults_to_reference_and_normal_file_restores_contrast(tmp_path):
    _app()
    reference_only = tmp_path / "reference_only.mat"
    normal = tmp_path / "normal_contrast.mat"
    _write_gui_dual_snapshot(reference_only, dimension=2, reference_only=True)
    _write_gui_dual_snapshot(normal, dimension=1)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    assert win._load_file(str(reference_only))
    win.canvas.draw()
    assert win.mode_combo.currentText() == "reference"
    assert win.ctx.trace is not None and win.ctx.trace.mode == "reference"
    assert np.allclose(win.ctx.trace.y, win.ctx.trace.reference)
    assert len(win.ax_main.collections) == 1

    assert win._load_file(str(normal))
    assert win.mode_combo.currentText() == "contrast"
    assert win.ctx.trace is not None and win.ctx.trace.mode == "contrast"

    reference_item = next(
        win.overlay_list.item(index)
        for index in range(win.overlay_list.count())
        if win._overlay_item_name(win.overlay_list.item(index)) == reference_only.name
    )
    win.overlay_box.setChecked(True)
    win.overlay_list.setCurrentItem(reference_item)
    win.btn_plot_selected_file.click()
    win.canvas.draw()
    assert win.ctx.trace is not None and win.ctx.trace.file_name == reference_only.name
    assert win.mode_combo.currentText() == "reference"
    assert win.ctx.trace.mode == "reference"
    assert len(win.ax_main.collections) == 1


def test_gui_responsive_layout_collapses_panels_before_scaling_text():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi_2206GHz4ns.mat")))
    win.presentation_state.annotation_style.font_size = 14
    win._set_annotation_visible(True, refresh=True)

    win.show()
    _app().processEvents()
    win.resize(900, 560)
    win._apply_responsive_scaling(force=True)
    _app().processEvents()

    assert win._ui_scale == 1.0
    assert win._plot_scale == 1.0
    assert win._responsive_layout_mode == "files"
    assert not win.left_panel.isHidden()
    assert win.right_panel.isHidden()
    assert win.files_drawer_btn.isVisible()
    assert win.inspector_drawer_btn.isVisible()
    assert win._plot_annotation_artist is not None
    assert win._plot_annotation_artist.get_fontsize() == pytest.approx(14)
    assert win.plot_surface.width() >= win.plot_stack.width() - 4
    assert win.plot_surface.height() >= win.plot_stack.height() - 4


def test_gui_plot_toolbar_syncs_with_drawer_controls():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))

    for button in (
        win.quick_data_btn,
        win.quick_fit_btn,
        win.quick_legend_btn,
        win.quick_edit_plot_btn,
        win.copy_figure_btn,
    ):
        assert win.plot_quick_grid.indexOf(button) >= 0
    for button in (win.quick_smooth_btn, win.quick_annotation_btn, win.quick_signal_btn, win.quick_reference_btn, win.quick_iterations_btn):
        assert not button.isHidden()
        assert win.plot_quick_grid.indexOf(button) >= 0
    assert win.quick_more_btn.isHidden()

    win.quick_legend_btn.setChecked(False)
    assert not win.show_legend_chk.isChecked()
    assert not win.presentation_state.legend.visible

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
    assert win.annotation_mode_combo.currentText() == "Off"
    assert not win.quick_annotation_btn.isChecked()
    assert win._plot_annotation_artist is None

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
    assert win.annotation_mode_combo.currentText() != "Off"
    assert win.quick_annotation_btn.isChecked()
    assert win._plot_annotation_artist is not None
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
    assert win._selected_iteration_indices == set(range(win._iteration_count()))
    assert win.iteration_summary_lbl.text() == f"All {win._iteration_count()}"
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


def test_gui_map_controls_replace_separate_scan_workspace():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))

    assert win.mode_stack.count() == 2
    assert not win.scan_quick_bar.isHidden()
    assert "Max point" in win.scan_summary_text.toPlainText()
    win.scan_linecut_combo.setCurrentText("Best-point horizontal")
    assert win.ax_res.get_visible()


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
    assert not win.clear_marker_btn.isHidden()
    assert not any(label in {"Point", "Selected point"} for label in win.ax_main.get_legend_handles_labels()[1])

    win._select_point_from_plot(x0, refresh=True)
    assert win._selected_plot_point is None
    assert win.clear_marker_btn.isHidden()
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
    image = app.clipboard().image()
    assert not image.isNull()
    assert win.fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert image.pixelColor(0, 0).red() >= 250
    assert image.pixelColor(0, 0).green() >= 250
    assert image.pixelColor(0, 0).blue() >= 250


def _fake_fit(win: SmartFitterMainWindow, model_name: str = "Rabi") -> FitResult:
    return FitResult(
        model_name=model_name,
        param_names=["y0", "A", "f", "phi"],
        params=np.array([0.0, 0.02, 0.003, 0.0]),
        errors=np.array([0.001, 0.001, 0.0001, 0.01]),
        y_fit=np.asarray(win.ctx.y, dtype=float).copy(),
        r2=0.95,
        rmse=0.002,
        aic=1.0,
        bic=2.0,
        durbin_watson=1.9,
        bound_hits=[False] * 4,
        success=True,
        message="ok",
    )


def test_gui_discards_late_fit_result_after_analysis_changes():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi10us.mat")))
    request = FitRequest(1, win._current_fit_signature())
    win._active_fit_request = request
    win.ctx.x = win.ctx.x + 1.0

    win.on_fit_finished(FitOutcome(request, _fake_fit(win)))

    assert win.ctx.fit_result is None
    assert "discarded" in win.status_lbl.text().lower()


def test_gui_failed_quality_fit_keeps_values_and_reveals_results_summary():
    app = _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win.resize(1366, 768)
    win.show()
    win._load_file(str(_fixture("TestData", "NewCodebaseTests", "323RabiFullRun__checkpoint.mat")))
    result = _fake_fit(win, model_name="RabiAdaptive")
    result.r2 = 0.10
    result.success = False
    result.message = "maximum evaluations reached"

    win.right_scroll.verticalScrollBar().setValue(0)
    win.on_fit_finished(result)
    app.processEvents()
    app.processEvents()

    summary = win.summary_text.toPlainText()
    assert win.ctx.fit_result is result
    assert "WARNING: This fit did not pass all quality checks." in summary
    assert "maximum evaluations reached" in summary
    assert "f (" in summary
    assert "FAIL" in win.status_lbl.text()
    assert win.inspector_tabs.currentIndex() == win.inspector_tab_indices["results"]
    assert not win.results_box.isHidden()


def test_gui_fit_exception_preserves_previous_summary_values():
    app = _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi10us.mat")))
    win.on_fit_finished(_fake_fit(win, model_name="RabiAdaptive"))

    win.on_fit_error("all starts failed")
    app.processEvents()

    summary = win.summary_text.toPlainText()
    assert "FIT ERROR: all starts failed" in summary
    assert "previous fitted values are retained below" in summary
    assert "Analysis mode: RabiAdaptive" in summary


def test_gui_undo_history_is_chronological_across_parameters_and_masks():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi10us.mat")))
    win._populate_param_table(["A"], np.array([1.0]))
    win.param_table.item(0, 1).setText("2")
    win._record_analysis_state("range exclusion")
    win.ctx.exclusion_ranges.append((0.0, 1.0))

    win._undo()
    assert win.ctx.exclusion_ranges == []
    assert float(win.param_table.item(0, 1).text()) == 2.0
    win._undo()
    assert float(win.param_table.item(0, 1).text()) == 1.0


def test_gui_session_v3_restores_plot_locks_and_fit_state(tmp_path, monkeypatch):
    _app()
    session_path = tmp_path / "complete.nvfit-session.json"
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "Rabi10us.mat")))
    win.show_legend_chk.setChecked(False)
    win.ctx.exclusion_ranges = [(float(win.ctx.x[2]), float(win.ctx.x[3]))]
    win._refresh_processed()
    win.ctx.pending_locks = {"A": (0.02, True, 0.0, 0.1)}
    win.on_fit_finished(_fake_fit(win))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(session_path), ""))
    win.on_save_session()

    restored = SmartFitterMainWindow()
    restored._message = lambda *args, **kwargs: None
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(session_path), ""))
    restored.on_load_session()

    assert restored.ctx.trace is not None
    assert not restored.show_legend_chk.isChecked()
    assert restored.ctx.exclusion_ranges == win.ctx.exclusion_ranges
    assert restored.ctx.pending_locks["A"][1]
    assert restored.ctx.fit_result is not None


def test_gui_clearing_2d_cursor_does_not_recreate_best_point():
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    win._load_file(str(_fixture("TestData", "1305_XZScan.mat")))
    x0 = float(win.ctx.trace.x2d[0, 0])
    y0 = float(win.ctx.trace.y2d[0, 0])
    win._select_point_from_plot(x0, y0, refresh=True)
    win.clear_selected_marker(refresh=True)

    assert win._scan_cursor is None
    assert win._selected_plot_point is None


def test_gui_dual_detector_selection_invalidates_fit_and_compares_1d(tmp_path):
    _app()
    path = tmp_path / "dual_1d.mat"
    _write_gui_dual_snapshot(path, dimension=1)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    assert win._load_file(str(path))
    assert win._active_detector_id() == "detector1"
    assert win.detector_combo.count() == 2
    detector1_y = win.ctx.y.copy()
    win.ctx.exclusion_ranges = [(0.5, 1.5)]
    win.ctx.pending_locks = {"A": (0.1, True, 0.0, 1.0)}
    win.ctx.fit_result = _fake_fit(win)

    win.detector_combo.setCurrentIndex(win.detector_combo.findData("detector2"))

    assert win.ctx.trace.detector_id == "detector2"
    assert win.ctx.fit_result is None
    assert win.ctx.exclusion_ranges == [(0.5, 1.5)]
    assert win.ctx.pending_locks["A"][1]
    assert len(win.ctx.y) != 0
    assert not np.isclose(detector1_y[0], win.ctx.y[0])

    win.compare_detectors_chk.setChecked(True)
    assert any(line.get_label() == "Detector 1" for line in win.ax_main.lines)
    win.mode_combo.setCurrentText("signal")
    assert win._ax_secondary is not None
    assert win._ax_secondary.get_ylabel().startswith("Detector 1")


def test_gui_both_detector_scope_applies_smoothing_and_series_edits_to_both(tmp_path):
    app = _app()
    path = tmp_path / "dual_scope.mat"
    _write_gui_dual_snapshot(path, dimension=1)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    assert win._load_file(str(path))

    win.detector_scope_combo.setCurrentIndex(win.detector_scope_combo.findData("both"))
    win.smooth_spin.setValue(3)
    win.show_smoothed_chk.setChecked(True)
    win._refresh_processed()

    ids = {descriptor.series_id for descriptor in win._series_registry}
    assert win.compare_detectors_chk.isChecked()
    assert "smoothed" in ids
    assert "detector:detector2:smoothed" in ids
    assert "both detectors" in win.data_info_lbl.text()

    data = next(descriptor for descriptor in win._series_registry if descriptor.series_id == "data")
    win._open_plot_element_popover({"kind": "series", "descriptor": data, "route": ("series", "main", None)})
    app.processEvents()
    visible = next(
        check for check in win._plot_element_popover.findChildren(QCheckBox)
        if check.text() == "Show series"
    )
    visible.setChecked(False)
    app.processEvents()

    assert not win.presentation_state.series["data"].show_series
    assert not win.presentation_state.series["detector:detector1"].show_series
    assert not win.presentation_state.series["detector:detector2"].show_series
    assert win.ctx.dirty
    win._undo()
    assert win.presentation_state.series.get("data") is None or win.presentation_state.series["data"].show_series


def test_gui_dual_fit_queue_visits_each_detector_and_restores_origin(tmp_path, monkeypatch):
    path = tmp_path / "dual_fit_queue.mat"
    _write_gui_dual_snapshot(path, dimension=1)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    assert win._load_file(str(path))
    win.detector_scope_combo.setCurrentIndex(win.detector_scope_combo.findData("both"))
    win._dual_fit_origin = "detector1"
    win._dual_fit_queue = ["detector2"]
    win._dual_fit_robust = True
    calls = []
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda _delay, callback: callback()))
    monkeypatch.setattr(win, "_start_fit", lambda robust=False, **kwargs: calls.append((win._active_detector_id(), robust, kwargs)))

    win._continue_dual_fit()
    assert calls == [("detector2", True, {"_detector_chain": True})]
    assert win._active_detector_id() == "detector2"

    win._dual_fit_summaries = {"detector1": "ok", "detector2": "ok"}
    win._continue_dual_fit()
    assert win._active_detector_id() == "detector1"
    assert win._dual_fit_origin is None


def test_gui_dual_2d_comparison_has_synchronized_maps_values_and_linecuts(tmp_path):
    _app()
    path = tmp_path / "dual_2d.mat"
    _write_gui_dual_snapshot(path, dimension=2)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    assert win._load_file(str(path))
    win.compare_detectors_chk.setChecked(True)

    assert win._detector_compare_ax is not None and win._detector_compare_ax.get_visible()
    assert win._scan_colorbar is not None
    assert win._detector_compare_colorbar is not None
    assert win.scan_compare_scale_combo.currentText() == "Independent"
    assert np.allclose(win.ax_main.get_xlim(), win._detector_compare_ax.get_xlim())
    assert np.allclose(win.ax_main.get_ylim(), win._detector_compare_ax.get_ylim())

    x0 = float(win.ctx.trace.x2d[0, 0])
    y0 = float(win.ctx.trace.y2d[0, 0])
    win._select_point_from_plot(x0, y0, refresh=True)
    assert "secondary_z" in win._selected_plot_point
    assert "Detector 1=" in win.live_readout_lbl.text()
    assert "Detector 2=" in win.live_readout_lbl.text()

    # Clicking the comparison map while Detector 1 remains focused must use
    # Detector 2 as the source and Detector 1 as the counterpart, not compare
    # the selected cell with itself.
    win._select_point_from_plot(x0, y0, refresh=True, detector_id="detector2")
    assert win._selected_plot_point.get("detector_id") == "detector2"
    assert win._selected_plot_point.get("secondary_z") != pytest.approx(win._selected_plot_point.get("z"))
    assert "Detector 1=" in win.live_readout_lbl.text()
    assert "Detector 2=" in win.live_readout_lbl.text()

    win.scan_linecut_combo.setCurrentText("Cursor horizontal")
    assert win.ax_res.get_visible()
    assert len(win.ax_res.lines) >= 2


def test_gui_dual_2d_plot_order_is_stable_when_focus_changes(tmp_path):
    _app()
    path = tmp_path / "dual_order.mat"
    _write_gui_dual_snapshot(path, dimension=2)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    assert win._load_file(str(path))
    win.compare_detectors_chk.setChecked(True)
    assert win.ax_main.get_title() == "Detector 1"
    compare_title = win._detector_compare_ax.get_title()
    assert compare_title
    assert "ACTIVE" not in win.ax_main.get_title()

    win._activate_detector("detector2")
    assert win.ax_main.get_title() == "Detector 1"
    assert win._detector_compare_ax.get_title() == compare_title
    assert "ACTIVE" not in win.ax_main.get_title()


def test_gui_xy_spot_detection_uses_signal_and_targets_both_detectors(tmp_path):
    _app()
    path = tmp_path / "dual_spots.mat"
    _write_gui_dual_snapshot(path, dimension=2)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    assert win._load_file(str(path))

    x = np.linspace(-2.0, 2.0, 41)
    y = np.linspace(-2.0, 2.0, 41)
    xx, yy = np.meshgrid(x, y)
    for detector_id, center in (("detector1", (-0.6, 0.4)), ("detector2", (0.7, -0.5))):
        trace = win._primary_dataset.detectors[detector_id]
        signal = 1000.0 + 300.0 * np.exp(
            -0.5 * (((xx - center[0]) / 0.35) ** 2 + ((yy - center[1]) / 0.22) ** 2)
        )
        trace.x2d, trace.y2d = xx, yy
        trace.z2d = np.zeros_like(signal)
        trace.signal2d = signal
        trace.reference2d = np.full_like(signal, 1000.0)

    win.ctx.trace = win._primary_dataset.detectors["detector1"]
    win.detector_scope_combo.setCurrentIndex(win.detector_scope_combo.findData("both"))
    win.spot_threshold_spin.setValue(2.0)
    win._detect_scan_spots()

    assert set(win._scan_spot_results) == {"detector1", "detector2"}
    assert all(win._scan_spot_results[detector_id] for detector_id in ("detector1", "detector2"))
    assert all(
        win._scan_spot_diagnostics[detector_id]["source_key"] == "signal"
        for detector_id in ("detector1", "detector2")
    )
    assert win._detector_compare_ax is not None
    assert win.ax_main.patches
    assert win._detector_compare_ax.patches
    win.spot_overlay_chk.setChecked(False)
    assert not win.ax_main.patches
    assert not win._detector_compare_ax.patches

    win.spot_overlay_chk.setChecked(True)
    win._compare_detector_spots()
    assert win._spot_comparison_pair is not None
    assert "Local profile correlation" in win.spot_compare_lbl.text()
    original_count = sum(len(spots) for spots in win._scan_spot_results.values())
    win.spot_results_table.selectRow(0)
    win._remove_selected_spots()
    assert sum(len(spots) for spots in win._scan_spot_results.values()) == original_count - 1
    win._restore_removed_spots()
    assert sum(len(spots) for spots in win._scan_spot_results.values()) == original_count


def test_gui_dual_2d_comparison_supports_default_equal_physical_axes(tmp_path):
    _app()
    path = tmp_path / "dual_2d_equal_axes.mat"
    _write_gui_dual_snapshot(path, dimension=2)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    assert win.scan_pixel_geometry_combo.currentText() == "Physical pixel size"
    assert win._load_file(str(path))
    win.compare_detectors_chk.setChecked(True)
    # Matplotlib validates the shared-axis aspect constraint during the
    # deferred canvas draw, not when set_aspect() is called.
    win.canvas.draw()

    assert win._detector_compare_ax is not None
    assert win.ax_main.get_aspect() == pytest.approx(1.0)
    assert win._detector_compare_ax.get_aspect() == pytest.approx(1.0)
    assert win.ax_main.get_adjustable() == "box"
    assert win._detector_compare_ax.get_adjustable() == "box"
    assert np.allclose(win.ax_main.get_xlim(), win._detector_compare_ax.get_xlim())
    assert np.allclose(win.ax_main.get_ylim(), win._detector_compare_ax.get_ylim())


def test_gui_dual_map_stacks_equal_axes_in_narrow_plot_shape(tmp_path):
    path = tmp_path / "dual_2d_stacked.mat"
    _write_gui_dual_snapshot(path, dimension=2)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    assert win._load_file(str(path))
    win.compare_detectors_chk.setChecked(True)
    win._dual_map_stacked = True
    win._refresh_plot_only()
    win.canvas.draw()

    main = win.ax_main.get_position()
    comparison = win._detector_compare_ax.get_position()
    assert main.width == pytest.approx(comparison.width, rel=0.02)
    assert main.height == pytest.approx(comparison.height, rel=0.02)
    assert main.y0 > comparison.y0
    assert win.ax_main.get_aspect() == pytest.approx(1.0)
    assert win._detector_compare_ax.get_aspect() == pytest.approx(1.0)


def test_gui_automatic_export_sizes_follow_plot_mode(tmp_path):
    one_d = tmp_path / "dual_1d_export.mat"
    two_d = tmp_path / "dual_2d_export.mat"
    _write_gui_dual_snapshot(one_d, dimension=1)
    _write_gui_dual_snapshot(two_d, dimension=2)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    assert win._load_file(str(one_d))
    assert win._automatic_export_size() == (8.5, 5.5)
    assert win._load_file(str(two_d))
    win.compare_detectors_chk.setChecked(False)
    assert win._automatic_export_size() == (6.5, 6.0)
    win.compare_detectors_chk.setChecked(True)
    assert win._automatic_export_size() == (11.0, 5.5)
    win.fig_w_spin.setValue(9.25)
    assert win.plot_opts.export_size_mode == "custom"
    win._restore_automatic_export_size()
    assert win.plot_opts.export_size_mode == "auto"


def test_gui_reduced_motion_drawers_reach_immediate_end_state():
    win = SmartFitterMainWindow()
    win._reduce_motion = True
    win._set_workspace_panel_visible("files", False)
    assert win.left_panel.isHidden()
    win._set_workspace_panel_visible("files", True)
    assert not win.left_panel.isHidden()
    assert win.left_panel.maximumWidth() > 10000


def test_gui_reference_only_1d_comparison_can_transition_to_dual_2d_comparison(tmp_path):
    _app()
    reference_1d = tmp_path / "reference_only_1d.mat"
    scan_2d = tmp_path / "dual_2d_after_twinned_axis.mat"
    _write_gui_dual_snapshot(reference_1d, dimension=1, reference_only=True)
    _write_gui_dual_snapshot(scan_2d, dimension=2)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None

    assert win._load_file(str(reference_1d))
    assert win.mode_combo.currentText() == "reference"
    win.compare_detectors_chk.setChecked(True)
    win.canvas.draw()
    assert win._ax_secondary is not None
    assert win.ax_main in win.ax_main._twinned_axes

    assert win._load_file(str(scan_2d))
    win.canvas.draw()

    assert win.ctx.trace is not None and win.ctx.trace.scan_dim == "scan2d"
    assert win.compare_detectors_chk.isChecked()
    assert win._ax_secondary is None
    assert win.ax_main not in win.ax_main._twinned_axes
    assert len(win.ax_main.collections) == 1
    assert win._detector_compare_ax is not None
    assert len(win._detector_compare_ax.collections) == 1


def test_gui_dual_session_labels_and_detector_safe_exports(tmp_path, monkeypatch):
    _app()
    path = tmp_path / "dual_session.mat"
    session_path = tmp_path / "dual.nvfit-session.json"
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    _write_gui_dual_snapshot(path, dimension=1)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    assert win._load_file(str(path))
    win._detector_labels = {"detector1": "Green APD", "detector2": "Red APD"}
    win._apply_detector_labels()
    win._refresh_detector_controls("detector1")
    win.compare_detectors_chk.setChecked(True)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(export_dir))
    win.exp_png_chk.setChecked(True)
    win.exp_json_chk.setChecked(True)
    win.exp_origin_chk.setChecked(True)
    win.on_save_results()
    win.detector_combo.setCurrentIndex(win.detector_combo.findData("detector2"))
    win.on_save_results()

    assert (export_dir / "dual_session__detector1_gui_result.json").exists()
    assert (export_dir / "dual_session__detector2_gui_result.json").exists()
    assert (export_dir / "dual_session__detector1__comparison_gui_fit.png").exists()
    assert (export_dir / "dual_session__detector2__comparison_gui_fit.png").exists()
    assert (export_dir / "dual_session__detectors_raw.csv").exists()
    tidy = (export_dir / "dual_session__detectors_raw.csv").read_text(encoding="utf-8")
    assert "Green APD" in tidy and "Red APD" in tidy

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(session_path), ""))
    win.on_save_session()
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    assert payload["version"] == 3

    restored = SmartFitterMainWindow()
    restored._message = lambda *args, **kwargs: None
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(session_path), ""))
    restored.on_load_session()
    assert restored.ctx.trace.detector_id == "detector2"
    assert restored.compare_detectors_chk.isChecked()
    assert restored._detector_label("detector1") == "Green APD"
    assert restored._detector_label("detector2") == "Red APD"


def test_gui_session_v3_restores_independent_detector_fit_states(tmp_path, monkeypatch):
    path = tmp_path / "dual_fit_session.mat"
    session_path = tmp_path / "dual_fit_session.nvfit-session.json"
    _write_gui_dual_snapshot(path, dimension=1)
    win = SmartFitterMainWindow()
    win._message = lambda *args, **kwargs: None
    assert win._load_file(str(path))
    win.detector_scope_combo.setCurrentIndex(win.detector_scope_combo.findData("both"))

    detector1_result = _fake_fit(win)
    detector1_result.params = detector1_result.params.copy()
    detector1_result.params[0] = 1.25
    win.ctx.fit_result = detector1_result
    win.ctx.fit_target = win.ctx.y.copy()
    win._set_status("PASS", "detector 1 fit")
    win._store_active_detector_state()

    win._activate_detector("detector2")
    detector2_result = _fake_fit(win)
    detector2_result.params = detector2_result.params.copy()
    detector2_result.params[0] = 2.5
    win.ctx.fit_result = detector2_result
    win.ctx.fit_target = win.ctx.y.copy()
    win._set_status("WARN", "detector 2 review")
    win._store_active_detector_state()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(session_path), ""))
    win.on_save_session()
    restored = SmartFitterMainWindow()
    restored._message = lambda *args, **kwargs: None
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(session_path), ""))
    restored.on_load_session()

    assert restored.detector_scope_combo.currentData() == "both"
    assert set(restored._detector_analysis_states) == {"detector1", "detector2"}
    assert restored._detector_analysis_states["detector1"].fit_result.params[0] == pytest.approx(1.25)
    assert restored._detector_analysis_states["detector2"].fit_result.params[0] == pytest.approx(2.5)
    assert restored.ctx.trace.detector_id == "detector2"
    assert restored.ctx.status == "WARN"
