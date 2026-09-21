import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog

from nvfit.fit_engine import FitResult
from nvfit.gui_app import (
    PRESENTATION_EXPORT_DPI,
    PRESENTATION_FIGURE_HEIGHT,
    PRESENTATION_FIGURE_WIDTH,
    PlotOptions,
    SmartFitterMainWindow,
)
from nvfit.plot_presentation import (
    AnnotationOverride,
    AxisStyle,
    PlotPresentationState,
    SeriesOverride,
    TickStyle,
    apply_tick_style,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _fixture(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "TestData" / name


def _window(name: str = "Rabi10us.mat") -> SmartFitterMainWindow:
    _app()
    win = SmartFitterMainWindow()
    win._message = lambda *_args, **_kwargs: None
    win.presentation_defaults = win.presentation_factory.clone()
    win.presentation_state = win.presentation_defaults.clone()
    assert win._load_file(str(_fixture(name)))
    return win


def _fake_fit(win: SmartFitterMainWindow, r2: float = 0.91) -> FitResult:
    return FitResult(
        model_name="RabiCosine",
        param_names=["y0", "A", "f", "phi"],
        params=np.array([0.03, 0.02, 0.029, 0.0]),
        errors=np.array([0.001, 0.001, 0.0005, 0.1]),
        y_fit=win.ctx.y.copy(),
        r2=r2,
        rmse=0.002,
        aic=1.0,
        bic=2.0,
        durbin_watson=1.1,
        bound_hits=[False] * 4,
        success=True,
        message="ok",
    )


def test_plot_defaults_are_presentation_oriented():
    options = PlotOptions()
    assert options.plot_style == "Line + scatter"
    assert not options.show_smoothed
    assert not options.show_iteration_mean
    assert not options.show_rabi_envelope
    assert options.fig_width == PRESENTATION_FIGURE_WIDTH == 8.5
    assert options.fig_height == PRESENTATION_FIGURE_HEIGHT == 5.5
    assert options.save_dpi == PRESENTATION_EXPORT_DPI == 300


def test_presentation_state_round_trip_and_manual_ticks():
    state = PlotPresentationState()
    state.series["data"] = SeriesOverride(label="Measured", color="#ff0000", order=2)
    state.annotations["metric:r2"] = AnnotationOverride(label="Goodness", precision=5)
    axis = state.axis("main")
    axis.x_ticks = TickStyle(mode="Manual", manual_positions=[0.0, 2.0], manual_labels=["zero", "two"])

    restored = PlotPresentationState.from_json(state.to_json())
    assert restored.series["data"].label == "Measured"
    assert restored.annotations["metric:r2"].precision == 5
    assert restored.axis("main").x_ticks.manual_labels == ["zero", "two"]

    ax = Figure().subplots()
    apply_tick_style(ax, "x", restored.axis("main").x_ticks)
    assert list(ax.get_xticks()) == [0.0, 2.0]
    assert [item.get_text() for item in ax.get_xticklabels()] == ["zero", "two"]


def test_legacy_secondary_axis_defaults_migrate_to_the_right_side():
    legacy = PlotPresentationState(version=1)
    legacy.axes["secondary"] = AxisStyle()

    restored = PlotPresentationState.from_json(legacy.to_json())
    secondary = restored.axis("secondary")

    assert restored.version == 2
    assert not secondary.x_ticks.bottom
    assert not secondary.x_ticks.label_bottom
    assert not secondary.y_ticks.left
    assert secondary.y_ticks.right
    assert not secondary.y_ticks.label_left
    assert secondary.y_ticks.label_right


def test_unified_editor_applies_series_annotation_axis_and_tick_edits():
    win = _window()
    win.on_fit_finished(_fake_fit(win))
    win.presentation_state.series["data"] = SeriesOverride(
        label="Measured counts",
        color="#ff0000",
        linestyle="--",
        marker="s",
        linewidth=2.5,
        alpha=0.6,
        order=1,
    )
    win.presentation_state.annotations["metric:r2"] = AnnotationOverride(
        label="Fit quality",
        show_plot=True,
        show_report=False,
        format_mode="Decimal",
        precision=2,
        order=0,
    )
    win.presentation_state.annotations["custom:test"] = AnnotationOverride(
        static_text="Operator note",
        show_plot=True,
        show_report=True,
        order=1,
    )
    main = win.presentation_state.axis("main")
    main.title.mode = "Custom"
    main.title.text = "Custom Rabi title"
    main.x_ticks.mode = "Manual"
    main.x_ticks.manual_positions = [0.0, 1000.0]
    main.x_ticks.manual_labels = ["start", "1 us"]
    win._refresh_plot_only()

    data = next(desc for desc in win._series_registry if desc.series_id == "data")
    assert data.handle.get_color() == "#ff0000"
    assert data.handle.get_linestyle() == "--"
    assert "Measured counts" in [text.get_text() for text in win._active_legend.get_texts()]
    assert "Fit quality: 0.91" in win._plot_annotation_artist.get_text()
    assert "Operator note" in win._plot_annotation_artist.get_text()
    assert win.ax_main.get_title() == "Custom Rabi title"
    assert [item.get_text() for item in win.ax_main.get_xticklabels()] == ["start", "1 us"]

    win._open_plot_editor(section="annotation")
    assert win.plot_editor.tabs.count() == 3
    assert not win.annotation_box.isVisible()
    assert win.plot_editor.annotation_table.rowCount() >= 6


def test_typing_axis_text_switches_from_auto_to_custom_and_updates_live_plot():
    win = _window()
    win._open_plot_editor(section="axes", axis_role="main")
    editor = win.plot_editor

    generated_xlabel = win.ax_main.get_xlabel()
    assert editor.text_controls["xlabel"]["text"].text() == generated_xlabel
    edits = {
        "title": "Custom experiment title",
        "xlabel": "Custom time axis",
        "ylabel": "Custom response axis",
    }
    for key, text in edits.items():
        control = editor.text_controls[key]["text"]
        control.setText(text)
        control.textEdited.emit(text)
        _app().processEvents()
        assert editor.text_controls[key]["mode"].currentText() == "Custom"
        assert getattr(win.presentation_state.axis("main"), key).text == text

    assert win.ax_main.get_title() == edits["title"]
    assert win.ax_main.get_xlabel() == edits["xlabel"]
    assert win.ax_main.get_ylabel() == edits["ylabel"]

    editor.text_controls["xlabel"]["mode"].setCurrentText("Auto")
    _app().processEvents()
    assert win.ax_main.get_xlabel() == generated_xlabel
    assert editor.text_controls["xlabel"]["text"].text() == generated_xlabel


def test_secondary_observable_axis_uses_right_ticks_and_labels():
    win = _window("NewCodebaseTests/323RabiFullRun__checkpoint.mat")
    win.show_signal_layer_chk.setChecked(True)
    _app().processEvents()

    secondary = win._ax_secondary
    assert secondary is not None
    style = win.presentation_state.axis("secondary")
    assert not style.y_ticks.left and style.y_ticks.right
    assert not style.y_ticks.label_left and style.y_ticks.label_right
    assert secondary.yaxis.get_label_position() == "right"
    assert all(not tick.label1.get_visible() for tick in secondary.yaxis.majorTicks)
    assert any(tick.label2.get_visible() for tick in secondary.yaxis.majorTicks)
    assert win.ax_main.yaxis.get_label_position() == "left"

    win._open_plot_editor(section="axes", axis_role="secondary")
    controls = win.plot_editor.tick_controls["y"]
    assert not controls["left"].isChecked()
    assert controls["right"].isChecked()
    assert not controls["label_left"].isChecked()
    assert controls["label_right"].isChecked()


def test_colorbar_axis_uses_right_side_defaults():
    win = _window("1305_XZScan.mat")
    assert win._scan_colorbar_ax is not None
    style = win.presentation_state.axis("colorbar")

    assert not style.y_ticks.left and style.y_ticks.right
    assert not style.y_ticks.label_left and style.y_ticks.label_right
    assert win._scan_colorbar_ax.yaxis.get_label_position() == "right"


def test_plot_and_legend_visibility_are_independent_and_hidden_data_does_not_set_ylim():
    win = _window()
    win.presentation_state.series["data"] = SeriesOverride(show_series=True, show_legend=False)
    win._refresh_plot_only()
    assert win.ax_main.lines[0].get_visible()
    legend_labels = [] if win._active_legend is None else [text.get_text() for text in win._active_legend.get_texts()]
    assert "Data" not in legend_labels

    win.presentation_state.series["data"] = SeriesOverride(show_series=False, show_legend=True)
    win._refresh_plot_only()
    data = next(desc for desc in win._series_registry if desc.series_id == "data")
    assert not data.handle.get_visible()
    assert "Data" not in ([] if win._active_legend is None else [text.get_text() for text in win._active_legend.get_texts()])


def test_double_click_targets_open_the_relevant_editor_section():
    win = _window()
    win.canvas.draw()
    bbox = win._active_legend.get_window_extent(win.canvas.get_renderer())
    event = MouseEvent("button_press_event", win.canvas, bbox.x0 + 3, bbox.y0 + 3, button=1, dblclick=True)
    assert win._editable_plot_target(event) == ("series", None, None)

    captured = []
    win._open_plot_element_popover = lambda element: captured.append(element)
    win._on_plot_click(event)
    assert captured and captured[0]["kind"] == "legend"

    win.canvas.draw()
    xlabel_bbox = win.ax_main.xaxis.label.get_window_extent(win.canvas.get_renderer())
    label_event = MouseEvent("button_press_event", win.canvas, xlabel_bbox.x0 + 1, xlabel_bbox.y0 + 1, button=1, dblclick=True)
    assert win._editable_plot_target(label_event) == ("axes", "main", "xlabel")


def test_current_file_overrides_reset_on_new_source_and_explicit_defaults_persist(tmp_path):
    win = _window()
    win.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    win.presentation_state.series["data"] = SeriesOverride(label="Current file")
    same_source = str(win.ctx.trace.source_path)
    win._reset_presentation_for_source(same_source)
    assert win.presentation_state.series["data"].label == "Current file"

    win._save_presentation_defaults()
    saved = PlotPresentationState.from_json(str(win.settings.value("presentation/defaults_v1")))
    assert saved.series["data"].label == "Current file"

    win.presentation_state.series["data"].label = "Temporary"
    win._reset_presentation_for_source(str(_fixture("ODMRRedo.mat")))
    assert win.presentation_state.series["data"].label == "Current file"


def test_dynamic_annotation_value_updates_without_losing_row_customization():
    win = _window()
    win.on_fit_finished(_fake_fit(win, r2=0.91))
    win.presentation_state.annotations["metric:r2"] = AnnotationOverride(
        label="Quality",
        show_plot=True,
        format_mode="Decimal",
        precision=3,
        order=0,
    )
    win.on_fit_finished(_fake_fit(win, r2=0.87654))
    assert "Quality: 0.877" in win._plot_annotation_artist.get_text()
    assert win.presentation_state.annotations["metric:r2"].label == "Quality"


def test_report_uses_shared_series_and_annotation_presentation(tmp_path, monkeypatch):
    win = _window()
    win.on_fit_finished(_fake_fit(win, r2=0.934))
    win.presentation_state.series["data"] = SeriesOverride(label="Report measurements")
    win.presentation_state.annotations["metric:r2"] = AnnotationOverride(
        label="Report quality",
        show_plot=False,
        show_report=True,
        format_mode="Decimal",
        precision=3,
        order=0,
    )
    for check in (win.exp_png_chk, win.exp_pdf_chk, win.exp_svg_chk, win.exp_json_chk, win.exp_csv_chk, win.exp_origin_chk, win.exp_report_pdf_chk):
        check.setChecked(False)
    win.exp_report_png_chk.setChecked(True)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(tmp_path))
    captured = {}

    def capture_report(canvas, *_args, **_kwargs):
        figure = canvas.figure
        captured["legend"] = [text.get_text() for ax in figure.axes for legend in ([ax.get_legend()] if ax.get_legend() else []) for text in legend.get_texts()]
        captured["texts"] = [text.get_text() for ax in figure.axes for text in ax.texts]

    monkeypatch.setattr("nvfit.gui_app.FigureCanvas.print_figure", capture_report)
    win.on_save_results()

    assert "Report measurements" in captured["legend"]
    assert any("Report quality: 0.934" in text for text in captured["texts"])
