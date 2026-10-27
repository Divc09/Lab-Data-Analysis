import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog
from scipy.io import loadmat, savemat

from nvfit.io_mat import load_experiment_dataset
from nvfit.map_comparison import build_map_overlay, compatible_axes
from nvfit.sessions import save_session
from test_time_trace_support import window


def _write_gui_dual_snapshot(path, *, dimension=2, x=None):
    assert dimension == 2
    x = np.arange(3, dtype=float) if x is None else np.asarray(x, dtype=float)
    y = np.array([10., 11.])
    reference = np.arange(2 * len(x), dtype=float).reshape(2, len(x)) + 100
    values, values2 = np.empty(reference.shape, dtype=object), np.empty(reference.shape, dtype=object)
    for index in np.ndindex(reference.shape):
        values[index] = [reference[index], .8 * reference[index]]
        values2[index] = [2 * reference[index], 1.4 * reference[index]]
    def metrics(ref, sig):
        def metric(z, label):
            return {"x": x, "y": y, "z": z, "xLabel": "Stage X", "yLabel": "Stage Y", "zLabel": label}
        return {"dimension": 2, "averageReference": metric(ref, "Reference"), "averageSignal": metric(sig, "Signal"), "averageContrast": metric((ref-sig)/ref, "Contrast")}
    scan = np.empty(2, dtype=object)
    for i, (axis, label) in enumerate(((x, "stage x"), (y, "stage y"))):
        scan[i] = {"bounds": [axis[0], axis[-1]], "stepSize": axis[1] - axis[0], "nSteps": len(axis), "parameter": label}
    savemat(path, {"Data": {"values": values, "detector2Values": values2, "iteration": np.ones(reference.shape), "failedPoints": np.zeros(reference.shape), "detectorMode": "dual"},
                   "scanInfo": scan, "params": {"detectorMode": "dual", "scanMode": "parameter", "perSecond": False},
                   "analysis": {**metrics(reference, .8 * reference), "detector2": metrics(2 * reference, 1.4 * reference)}})


def test_equal_registered_maps_have_white_overlap_and_missing_cells_are_transparent():
    values = np.array([[0., 3., np.nan], [2., 1., 0.]])
    layer = (np.array([0., 1., 2.]), np.array([10., 11.]), values)
    overlay = build_map_overlay(layer, layer, background_percentile=0)
    np.testing.assert_allclose(overlay.rgba[0, 1], [1, 1, 1, 1])
    assert overlay.rgba[0, 2, 3] == 0
    np.testing.assert_array_equal(overlay.primary, values)


def test_different_grids_copy_native_cells_without_extrapolation_or_axis_warp():
    first = (np.array([0., 1.]), np.array([10., 12.]), np.array([[1., 2.], [3., 4.]]))
    second = (np.array([2., 0.]), np.array([12., 10.]), np.array([[8., 6.], [4., 2.]]))
    result = build_map_overlay(first, second, background_percentile=0)
    assert result.x_edges[0] == -1 and result.x_edges[-1] == 3
    assert result.y_edges[0] == 9 and result.y_edges[-1] == 13
    assert np.all(np.isnan(result.primary[:, 0]))
    assert np.all(np.isnan(result.primary[:, -1]))
    np.testing.assert_array_equal(result.primary[0, 1:-1], [1., 2., 2.])
    np.testing.assert_array_equal(result.secondary[0], [2., 2., 2., 4., 4.])
    assert np.all(result.rgba[:, :, 3] == 1)


def test_shared_ranges_retain_relative_intensity_and_independent_ranges_show_shapes():
    first = (np.array([0., 1.]), np.array([0., 1.]), np.array([[0., 1.], [0., 0.]]))
    second = (first[0], first[1], first[2] * 10)
    independent = build_map_overlay(first, second, background_percentile=0)
    shared = build_map_overlay(first, second, background_percentile=0, shared=True)
    np.testing.assert_array_equal(independent.rgba[0, 1, :3], [1, 1, 1])
    np.testing.assert_allclose(shared.rgba[0, 1, :3], [.1, 1, .66])
    assert shared.ranges[0] == shared.ranges[1] == (0, 10)


def test_coordinate_and_plane_validation():
    assert compatible_axes(("Stage X", "Stage Z"), ("z (µm)", "x (µm)"))
    assert not compatible_axes(("Stage X", "Stage Y"), ("Stage X", "Stage Z"))
    layer = (np.array([0, 1, 0]), np.array([0, 1]), np.ones((2, 3)))
    with pytest.raises(ValueError, match="monotonic"):
        build_map_overlay(layer, layer)


def test_gui_overlay_preserves_geometry_and_survives_view_switches(window, tmp_path):
    path = tmp_path / "dual.mat"
    _write_gui_dual_snapshot(path, dimension=2)
    assert window._load_file(str(path))
    window._show_map_overlay()
    for swap in (False, True):
        for geometry in ("Physical pixel size", "Square pixels"):
            window.scan_swap_axes_chk.setChecked(swap)
            window.scan_pixel_geometry_combo.setCurrentText(geometry)
            window.canvas.draw()
            assert window._scan_colorbar is None
            assert window._detector_compare_ax is None
            x, y, z, *_ = window._scan2d_display_data(window._canonical_scan_traces()[0])
            assert window.ax_main.collections[0].get_array().shape == (*z.shape, 4)
            expected_aspect = window._scan2d_geometry(window._scan2d_axis_edges(x), window._scan2d_axis_edges(y))[0]
            assert window.ax_main.get_aspect() == expected_aspect
    window.map_compare_view_combo.setCurrentText("Maps + overlay")
    window.canvas.draw()
    assert window._map_overlay_ax is not None
    assert "map_overlay" in window._active_presentation_axis_roles()
    assert window._axis_role_for(window._map_overlay_ax) == "map_overlay"
    assert window._presentation_axis_map()["map_overlay"] is window._map_overlay_ax
    window._on_scan_scroll(SimpleNamespace(inaxes=window._map_overlay_ax, xdata=10.5, ydata=1., button="up"))
    assert window._scan_view_limits == (window._map_overlay_ax.get_xlim(), window._map_overlay_ax.get_ylim())
    window.map_compare_view_combo.setCurrentText("Separate maps")
    window.canvas.draw()
    assert window._map_overlay_ax is None
    assert window._scan_colorbar is not None
    assert not window.test_messages


def _assert_view_contains(axis, xlim, ylim):
    # Equal-pixel maps keep a fixed axes box and widen one data range instead.
    (x0, x1), (y0, y1) = sorted(axis.get_xlim()), sorted(axis.get_ylim())
    assert x0 <= xlim[0] + 1e-9 and x1 >= xlim[1] - 1e-9
    assert y0 <= ylim[0] + 1e-9 and y1 >= ylim[1] - 1e-9


def test_zooming_a_detector_map_keeps_both_map_boxes_fixed(window, tmp_path):
    path = tmp_path / "dual.mat"
    _write_gui_dual_snapshot(path)
    assert window._load_file(str(path))
    window.compare_detectors_chk.setChecked(True)
    window.map_compare_view_combo.setCurrentText("Separate maps")
    window.canvas.draw()
    axes = (window.ax_main, window._detector_compare_ax)
    boxes = [axis.get_position().bounds for axis in axes]
    window._on_scan_scroll(SimpleNamespace(inaxes=window.ax_main, xdata=1., ydata=10.5, button="up"))
    window._scan_view_limits = ((0.9, 1.1), (9.75, 11.25))
    window._refresh_plot_only()
    window.canvas.draw()
    for axis, box in zip((window.ax_main, window._detector_compare_ax), boxes):
        np.testing.assert_allclose(axis.get_position().bounds, box)
        _assert_view_contains(axis, (0.9, 1.1), (9.75, 11.25))


@pytest.mark.parametrize("previous_view", ["Single map", "Separate maps", "Maps + overlay", "cross-file"])
def test_overlay_button_restores_previous_layout_sources_zoom_and_linecut(window, tmp_path, previous_view):
    first, second = tmp_path / "first.mat", tmp_path / "second.mat"
    _write_gui_dual_snapshot(first)
    assert window._load_file(str(first))
    window.scan_swap_axes_chk.setChecked(False)
    window.scan_linecut_combo.setCurrentText("Cursor horizontal")
    if previous_view == "cross-file":
        _write_gui_dual_snapshot(second)
        dataset = load_experiment_dataset(second)
        window._loaded_datasets[second.name] = dataset
        window.ctx.loaded_traces[second.name] = dataset.detectors["detector1"]
        window._update_overlay_widgets()
        window.map_compare_source_combo.setCurrentIndex(window._map_comparison_source_index((second.name, "detector2")))
    else:
        if previous_view == "Separate maps":
            window.compare_detectors_chk.setChecked(True)
        window.map_compare_view_combo.setCurrentText(previous_view)
    window._scan_cursor = (1., 10.)
    window._scan_view_limits = ((0.25, 1.75), (9.75, 11.25))
    window._refresh_plot_only()
    window.canvas.draw()
    assert window.ax_res.get_visible()
    source = window.map_compare_source_combo.currentData()
    view = window.map_compare_view_combo.currentText()
    compare = window.compare_detectors_chk.isChecked()
    previous_pixels = np.array(window.ax_main.collections[0].get_array(), copy=True)
    axes_count = len(window.fig.axes)
    for _ in range(3):
        window.quick_map_overlay_btn.click()
        window.canvas.draw()
        assert window.quick_map_overlay_btn.isChecked()
        assert window.map_compare_view_combo.currentText() == "RGB overlay"
        assert window._scan_colorbar is None
        window.quick_map_overlay_btn.click()
        window.canvas.draw()
        assert not window.quick_map_overlay_btn.isChecked()
        assert window.map_compare_view_combo.currentText() == view
        assert window.map_compare_source_combo.currentData() == source
        assert window.compare_detectors_chk.isChecked() == compare
        assert len(window.fig.axes) == axes_count
        assert window.ax_res.get_visible()
        assert window._scan_colorbar is not None
        np.testing.assert_array_equal(window.ax_main.collections[0].get_array(), previous_pixels)
        for axis in (window.ax_main, window._detector_compare_ax, window._map_overlay_ax):
            if axis is not None:
                _assert_view_contains(axis, (0.25, 1.75), (9.75, 11.25))
        assert window._scan_view_limits == ((0.25, 1.75), (9.75, 11.25))


def test_detector_and_primary_only_controls_exit_overlay_without_reenabling_it(window, tmp_path):
    path = tmp_path / "dual.mat"
    _write_gui_dual_snapshot(path)
    assert window._load_file(str(path))
    window.quick_map_overlay_btn.click()
    window.quick_compare_detectors_btn.click()
    window.map_overlay_palette_combo.setCurrentText("Red / cyan")
    window.canvas.draw()
    assert window.map_compare_view_combo.currentText() == "Single map"
    assert not window.compare_detectors_chk.isChecked()
    assert not window.quick_map_overlay_btn.isChecked()
    assert window._detector_compare_ax is None
    assert window._map_overlay_data is None
    assert window._scan_colorbar is not None
    window.quick_compare_detectors_btn.click()
    window.canvas.draw()
    assert window.map_compare_view_combo.currentText() == "Separate maps"
    assert window._detector_compare_ax is not None
    window.quick_map_overlay_btn.click()
    window.on_show_primary_only()
    window.canvas.draw()
    assert window.map_compare_view_combo.currentText() == "Single map"
    assert window._canonical_scan_traces() == (window.ctx.trace, None)
    assert not window.quick_map_overlay_btn.isChecked()


def test_single_map_picker_exits_cross_file_overlay_and_separate_maps_recovers_it(window, tmp_path):
    first, second = tmp_path / "first.mat", tmp_path / "second.mat"
    for path in (first, second):
        _write_gui_dual_snapshot(path)
    assert window._load_file(str(first))
    dataset = load_experiment_dataset(second)
    window._loaded_datasets[second.name] = dataset
    window.ctx.loaded_traces[second.name] = dataset.detectors["detector1"]
    window._update_overlay_widgets()
    window.map_compare_source_combo.setCurrentIndex(window._map_comparison_source_index((second.name, "detector2")))
    window.quick_map_overlay_btn.click()
    window.map_compare_view_combo.setCurrentText("Single map")
    window.map_overlay_background_spin.setValue(15)
    window.canvas.draw()
    assert window._canonical_scan_traces() == (window.ctx.trace, None)
    assert window._map_overlay_data is None
    assert not window.quick_map_overlay_btn.isChecked()
    window.map_compare_view_combo.setCurrentText("Separate maps")
    window.canvas.draw()
    assert window._detector_compare_ax is not None
    assert window._canonical_scan_traces()[1].source_path == str(second.resolve())
    window.map_compare_source_combo.setCurrentIndex(0)
    window.map_compare_view_combo.setCurrentText("Single map")
    window.map_compare_view_combo.setCurrentText("Separate maps")
    window.canvas.draw()
    assert window.compare_detectors_chk.isChecked()
    assert window._detector_compare_ax is not None


def _write_single_map(path):
    _write_gui_dual_snapshot(path)
    payload = loadmat(path, simplify_cells=True)
    payload["Data"].pop("detector2Values")
    payload["Data"]["detectorMode"] = "single"
    payload["params"]["detectorMode"] = "single"
    payload["analysis"].pop("detector2")
    savemat(path, {key: value for key, value in payload.items() if not key.startswith("__")})


def test_unavailable_overlay_leaves_the_current_view_and_zoom_unchanged(window, tmp_path):
    path = tmp_path / "single.mat"
    _write_single_map(path)
    assert window._load_file(str(path))
    window._scan_view_limits = ((10., 10.5), (0.25, 1.75))
    window._refresh_plot_only()
    view = window.map_compare_view_combo.currentText()
    axes_count = len(window.fig.axes)
    window.quick_map_overlay_btn.click()
    window.canvas.draw()
    assert not window.quick_map_overlay_btn.isChecked()
    assert not window.compare_detectors_chk.isChecked()
    assert window.map_compare_view_combo.currentText() == view
    assert len(window.fig.axes) == axes_count
    assert window._scan_view_limits == ((10., 10.5), (0.25, 1.75))
    assert window._map_overlay_return_state is None


def test_auto_selected_cross_file_overlay_restores_single_map_and_pixel_geometry(window, tmp_path):
    first, second = tmp_path / "single.mat", tmp_path / "second.mat"
    _write_single_map(first)
    _write_gui_dual_snapshot(second, x=[0., 2., 4.])
    assert window._load_file(str(first))
    dataset = load_experiment_dataset(second)
    window._loaded_datasets[second.name] = dataset
    window.ctx.loaded_traces[second.name] = dataset.detectors["detector1"]
    window._update_overlay_widgets()
    window.scan_pixel_geometry_combo.setCurrentText("Square pixels")
    full_view = (window.ax_main.get_xlim(), window.ax_main.get_ylim())
    window.quick_map_overlay_btn.click()
    window.canvas.draw()
    assert window.map_compare_source_combo.currentData() is not None
    assert window.scan_pixel_geometry_combo.currentText() == "Physical pixel size"
    window.quick_map_overlay_btn.click()
    window.canvas.draw()
    assert window.map_compare_source_combo.currentData() is None
    assert window.scan_pixel_geometry_combo.currentText() == "Square pixels"
    assert window._detector_compare_ax is None
    assert (window.ax_main.get_xlim(), window.ax_main.get_ylim()) == full_view


def test_overlay_return_zoom_tracks_axis_swap_and_survives_observable_reload(window, tmp_path):
    path = tmp_path / "dual.mat"
    _write_gui_dual_snapshot(path)
    assert window._load_file(str(path))
    window.scan_swap_axes_chk.setChecked(False)
    window._scan_view_limits = ((0.25, 1.75), (9.75, 11.25))
    window._refresh_plot_only()
    window.quick_map_overlay_btn.click()
    window.scan_swap_axes_chk.setChecked(True)
    window.mode_combo.setCurrentText("reference")
    window.quick_map_overlay_btn.click()
    window.canvas.draw()
    assert not window.compare_detectors_chk.isChecked()
    assert window._detector_compare_ax is None
    _assert_view_contains(window.ax_main, (9.75, 11.25), (0.25, 1.75))
    window.toolbar.home()
    assert window._scan_view_limits is None


def test_loading_another_primary_discards_the_previous_overlay_return_view(window, tmp_path):
    first, second = tmp_path / "first.mat", tmp_path / "second.mat"
    _write_gui_dual_snapshot(first)
    _write_gui_dual_snapshot(second, x=[20., 25., 30.])
    assert window._load_file(str(first))
    window.quick_map_overlay_btn.click()
    assert window._load_file(str(second))
    assert window._map_overlay_return_state is None
    window.quick_map_overlay_btn.click()
    window.canvas.draw()
    assert window.ctx.trace.file_name == second.name
    assert window._scan_colorbar is not None
    np.testing.assert_allclose(window.ax_main.get_ylim(), (17.5, 32.5))


def test_removing_the_only_comparison_source_clears_overlay_mode(window, tmp_path):
    first, second = tmp_path / "single.mat", tmp_path / "second.mat"
    _write_single_map(first)
    _write_gui_dual_snapshot(second)
    assert window._load_file(str(first))
    dataset = load_experiment_dataset(second)
    window._loaded_datasets[second.name] = dataset
    window.ctx.loaded_traces[second.name] = dataset.detectors["detector1"]
    window._update_overlay_widgets()
    window.quick_map_overlay_btn.click()
    window.on_clear_overlays()
    window.canvas.draw()
    assert not window.quick_map_overlay_btn.isChecked()
    assert window.map_compare_view_combo.currentText() == "Separate maps"
    assert window._map_overlay_data is None
    assert window._scan_colorbar is not None
    assert window._map_overlay_return_state is None


@pytest.mark.parametrize("view", ["Single map", "RGB overlay", "Maps + overlay"])
def test_session_restores_map_mode_and_overlay_button_remains_reversible(window, tmp_path, monkeypatch, view):
    path = tmp_path / "dual.mat"
    _write_gui_dual_snapshot(path)
    assert window._load_file(str(path))
    window.map_compare_view_combo.setCurrentText(view)
    session = tmp_path / "maps.nvfit-session.json"
    save_session(session, window._session_payload(str(session)))
    # Load into a different detector-comparison state so checkbox signals must
    # not overwrite the view that the session has just restored.
    window.compare_detectors_chk.setChecked(view == "Single map")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(session), ""))
    window.on_load_session()
    window.canvas.draw()
    assert window.map_compare_view_combo.currentText() == view
    assert window.quick_map_overlay_btn.isChecked() == (view == "RGB overlay")
    if view == "RGB overlay":
        window.quick_map_overlay_btn.click()
        window.canvas.draw()
        assert window.map_compare_view_combo.currentText() == "Separate maps"
        assert window._detector_compare_ax is not None
        assert window._scan_colorbar is not None
    else:
        window.quick_map_overlay_btn.click()
        window.quick_map_overlay_btn.click()
        window.canvas.draw()
        assert window.map_compare_view_combo.currentText() == view
    assert not window.test_messages


def test_cross_file_comparison_picker_session_and_export(window, tmp_path, monkeypatch):
    first, second = tmp_path / "first.mat", tmp_path / "second.mat"
    _write_gui_dual_snapshot(first, dimension=2)
    _write_gui_dual_snapshot(second, dimension=2)
    assert window._load_file(str(first))
    dataset = load_experiment_dataset(second)
    window._loaded_datasets[dataset.file_name] = dataset
    window.ctx.loaded_traces[dataset.file_name] = dataset.detectors["detector1"]
    window._update_overlay_widgets()
    source = (second.name, "detector2")
    window.map_compare_source_combo.setCurrentIndex(window._map_comparison_source_index(source))
    window.map_compare_view_combo.setCurrentText("RGB overlay")
    window.canvas.draw()
    assert window._canonical_scan_traces()[1].source_path == str(second.resolve())
    metadata = window._effective_trace_metadata()["map_comparison"]
    assert metadata["sources"][1]["detector_id"] == "detector2"
    assert metadata["sources"][1]["path"] == str(second.resolve())
    session = tmp_path / "maps.nvfit-session.json"
    save_session(session, window._session_payload(str(session)))
    window.map_compare_view_combo.setCurrentText("Separate maps")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(session), ""))
    window.on_load_session()
    window.canvas.draw()
    assert window.map_compare_source_combo.currentData() == source
    assert window.map_compare_view_combo.currentText() == "RGB overlay"
    assert window._effective_trace_metadata()["map_comparison"]["sources"][1]["path"] == str(second.resolve())
    out = tmp_path / "export"
    out.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(out))
    window.exp_png_chk.setChecked(False)
    window.exp_pdf_chk.setChecked(False)
    window.exp_svg_chk.setChecked(False)
    window.exp_origin_chk.setChecked(True)
    window.on_save_results()
    assert next(out.glob('*__map_comparison.csv')).exists()
    result = json.loads(next(out.glob('*_gui_result.json')).read_text())
    assert result["trace_metadata"]["map_comparison"]["sources"][1]["path"] == str(second.resolve())


@pytest.mark.parametrize("scan_mode", ["parameter", "time", None])
def test_spatial_scan_with_stale_time_trace_metadata_remains_a_map(tmp_path, scan_mode):
    path = tmp_path / "stage.mat"
    _write_gui_dual_snapshot(path)
    payload = loadmat(path, simplify_cells=True)
    payload["Data"]["timeTrace"] = {"elapsedSeconds": np.arange(100), "sourceExperiment": "FluorescenceTrace"}
    payload["metadata"] = {"timeTrace": payload["Data"]["timeTrace"]}
    if scan_mode is None:
        payload["params"].pop("scanMode")
    else:
        payload["params"]["scanMode"] = scan_mode
    savemat(path, {key: val for key, val in payload.items() if not key.startswith('__')})
    dataset = load_experiment_dataset(path)
    assert all(trace.scan_dim == "scan2d" for trace in dataset.detectors.values())


def test_gui_different_grids_keep_full_union_coverage_and_correct_inspection(window, tmp_path):
    first, second = tmp_path / "first.mat", tmp_path / "second.mat"
    _write_gui_dual_snapshot(first)
    _write_gui_dual_snapshot(second, x=[0., 2., 4.])
    assert window._load_file(str(first))
    dataset = load_experiment_dataset(second)
    window._loaded_datasets[second.name] = dataset
    window.ctx.loaded_traces[second.name] = dataset.detectors["detector1"]
    window._update_overlay_widgets()
    window.scan_swap_axes_chk.setChecked(False)
    source = (second.name, "detector1")
    window.map_compare_source_combo.setCurrentIndex(window._map_comparison_source_index(source))
    window.scan_pixel_geometry_combo.setCurrentText("Square pixels")
    window.map_compare_view_combo.setCurrentText("RGB overlay")
    window.canvas.draw()
    assert window.scan_pixel_geometry_combo.currentText() == "Physical pixel size"
    assert window.ax_main.get_xlim() == (-1., 5.)
    window._select_point_from_plot(4., 10.)
    assert window._selected_plot_point["source_path"] == str(second.resolve())
    window.ax_main.set_xlim(0., 2.)
    window._reset_scan_view()
    assert window.ax_main.get_xlim() == (-1., 5.)


def test_cross_file_opposite_storage_order_preserves_registered_pixels(window, tmp_path):
    first, second = tmp_path / "xy.mat", tmp_path / "yx.mat"
    _write_gui_dual_snapshot(first)
    payload = loadmat(first, simplify_cells=True)
    payload["scanInfo"] = payload["scanInfo"][::-1]
    for field in ("values", "detector2Values", "iteration", "failedPoints"):
        payload["Data"][field] = payload["Data"][field].T
    for analysis in (payload["analysis"], payload["analysis"]["detector2"]):
        for field in ("averageReference", "averageSignal", "averageContrast"):
            metric = analysis[field]
            metric["x"], metric["y"] = metric["y"], metric["x"]
            metric["xLabel"], metric["yLabel"] = metric["yLabel"], metric["xLabel"]
            metric["z"] = metric["z"].T
    savemat(second, {key: val for key, val in payload.items() if not key.startswith('__')})
    assert window._load_file(str(first))
    window.mode_combo.setCurrentText("reference")
    dataset = load_experiment_dataset(second, mode="reference")
    window._loaded_datasets[second.name] = dataset
    window.ctx.loaded_traces[second.name] = dataset.detectors["detector1"]
    window._update_overlay_widgets()
    window.map_compare_source_combo.setCurrentIndex(window._map_comparison_source_index((second.name, "detector1")))
    window.map_compare_view_combo.setCurrentText("RGB overlay")
    for swap in (False, True):
        window.scan_swap_axes_chk.setChecked(swap)
        window.canvas.draw()
        overlay = window._map_overlay_data
        np.testing.assert_array_equal(overlay.primary, overlay.secondary)
        brightest = np.unravel_index(np.argmax(overlay.primary), overlay.primary.shape)
        np.testing.assert_array_equal(overlay.rgba[brightest], [1., 1., 1., 1.])
