from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from matplotlib.ticker import AutoMinorLocator, EngFormatter, FixedFormatter, FixedLocator, FormatStrFormatter, MultipleLocator, NullLocator, ScalarFormatter
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


LINE_STYLES = {
    "Solid": "-",
    "Dashed": "--",
    "Dotted": ":",
    "Dash-dot": "-.",
    "None": "None",
}
LINE_STYLE_NAMES = {value: key for key, value in LINE_STYLES.items()}
MARKERS = {
    "Default": None,
    "None": "None",
    "Circle": "o",
    "Square": "s",
    "Triangle": "^",
    "Diamond": "D",
    "X": "x",
    "Plus": "+",
    "Point": ".",
}
MARKER_NAMES = {value: key for key, value in MARKERS.items()}
FORMAT_MODES = ["Automatic", "Decimal", "Scientific", "Engineering"]


@dataclass
class SeriesOverride:
    label: str | None = None
    show_series: bool = True
    show_legend: bool = True
    color: str | None = None
    linestyle: str | None = None
    marker: str | None = None
    linewidth: float | None = None
    alpha: float | None = None
    order: int | None = None


@dataclass
class LegendStyle:
    visible: bool = True
    title: str = ""
    location: str = "best"
    custom_anchor: bool = False
    anchor_x: float = 0.98
    anchor_y: float = 0.98
    font_size: int = 9
    columns: int = 1
    text_color: str = "#26332d"
    frame_visible: bool = True
    frame_color: str = "#ffffff"
    edge_color: str = "#c8d2cc"
    frame_alpha: float = 0.94
    border_pad: float = 0.4
    label_spacing: float = 0.5


@dataclass
class AnnotationOverride:
    label: str | None = None
    show_plot: bool | None = None
    show_report: bool | None = None
    format_mode: str = "Automatic"
    precision: int = 4
    units: str | None = None
    order: int | None = None
    static_text: str | None = None


@dataclass
class AnnotationStyle:
    mode: str = "Compact"
    visible: bool = True
    location: str = "upper right"
    custom_anchor: bool = False
    anchor_x: float = 0.98
    anchor_y: float = 0.98
    horizontal_alignment: str = "right"
    vertical_alignment: str = "top"
    font_size: int = 9
    text_color: str = "#f5fbff"
    background_color: str = "#141923"
    edge_color: str = "#93c5fd"
    background_alpha: float = 0.82
    padding: float = 0.35


@dataclass
class TextStyle:
    mode: str = "Auto"
    text: str = ""
    visible: bool = True
    font_size: int = 10
    font_weight: str = "normal"
    color: str = "#26332d"
    padding: float = 4.0


@dataclass
class TickStyle:
    mode: str = "Automatic"
    major_interval: float = 1.0
    minor_mode: str = "Off"
    minor_count: int = 1
    format_mode: str = "Automatic"
    precision: int = 4
    show_offset: bool = True
    label_size: int = 9
    label_color: str = "#526159"
    rotation: float = 0.0
    direction: str = "out"
    major_length: float = 3.5
    major_width: float = 0.8
    minor_length: float = 2.0
    minor_width: float = 0.6
    tick_color: str = "#526159"
    bottom: bool = True
    top: bool = False
    left: bool = True
    right: bool = False
    label_bottom: bool = True
    label_top: bool = False
    label_left: bool = True
    label_right: bool = False
    manual_positions: list[float] = field(default_factory=list)
    manual_labels: list[str] = field(default_factory=list)


@dataclass
class AxisStyle:
    title: TextStyle = field(default_factory=lambda: TextStyle(font_size=11, font_weight="normal", padding=6.0))
    xlabel: TextStyle = field(default_factory=lambda: TextStyle(font_size=10, padding=4.0))
    ylabel: TextStyle = field(default_factory=lambda: TextStyle(font_size=10, padding=4.0))
    x_ticks: TickStyle = field(default_factory=TickStyle)
    y_ticks: TickStyle = field(default_factory=TickStyle)


def axis_style_for_role(role: str) -> AxisStyle:
    """Return factory defaults that respect the physical side of an axis role."""
    style = AxisStyle()
    if role in {"secondary", "colorbar"}:
        style.x_ticks.bottom = False
        style.x_ticks.top = False
        style.x_ticks.label_bottom = False
        style.x_ticks.label_top = False
        style.y_ticks.left = False
        style.y_ticks.right = True
        style.y_ticks.label_left = False
        style.y_ticks.label_right = True
    return style


@dataclass
class PlotPresentationState:
    version: int = 2
    legend: LegendStyle = field(default_factory=LegendStyle)
    annotation_style: AnnotationStyle = field(default_factory=AnnotationStyle)
    series: dict[str, SeriesOverride] = field(default_factory=dict)
    annotations: dict[str, AnnotationOverride] = field(default_factory=dict)
    axes: dict[str, AxisStyle] = field(default_factory=dict)

    def clone(self) -> "PlotPresentationState":
        return copy.deepcopy(self)

    def axis(self, role: str) -> AxisStyle:
        if role not in self.axes:
            self.axes[role] = axis_style_for_role(role)
        return self.axes[role]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> "PlotPresentationState":
        raw = json.loads(payload) if isinstance(payload, str) else dict(payload)
        source_version = int(raw.get("version", 1))
        state = cls()
        state.legend = LegendStyle(**dict(raw.get("legend") or {}))
        state.annotation_style = AnnotationStyle(**dict(raw.get("annotation_style") or {}))
        state.series = {str(k): SeriesOverride(**dict(v)) for k, v in dict(raw.get("series") or {}).items()}
        state.annotations = {str(k): AnnotationOverride(**dict(v)) for k, v in dict(raw.get("annotations") or {}).items()}
        for role, values in dict(raw.get("axes") or {}).items():
            values = dict(values)
            role = str(role)
            style = AxisStyle(
                title=TextStyle(**dict(values.get("title") or {})),
                xlabel=TextStyle(**dict(values.get("xlabel") or {})),
                ylabel=TextStyle(**dict(values.get("ylabel") or {})),
                x_ticks=TickStyle(**dict(values.get("x_ticks") or {})),
                y_ticks=TickStyle(**dict(values.get("y_ticks") or {})),
            )
            # Version 1 used main-axis tick sides for every role. Migrate only
            # untouched legacy tick settings so deliberate custom sides survive.
            if source_version < 2 and role in {"secondary", "colorbar"}:
                legacy_ticks = asdict(TickStyle())
                if asdict(style.x_ticks) == legacy_ticks:
                    style.x_ticks = axis_style_for_role(role).x_ticks
                if asdict(style.y_ticks) == legacy_ticks:
                    style.y_ticks = axis_style_for_role(role).y_ticks
            state.axes[role] = style
        return state


@dataclass
class SeriesDescriptor:
    series_id: str
    default_label: str
    artists: list[Any]
    handle: Any
    axis: Any
    axis_role: str
    default_color: str | None
    default_linestyle: str | None
    default_marker: str | None
    default_linewidth: float | None
    default_alpha: float | None
    supports_line: bool = True
    supports_marker: bool = True
    x_values: list[Any] = field(default_factory=list)
    y_values: list[Any] = field(default_factory=list)
    creation_order: int = 0


@dataclass
class AnnotationDescriptor:
    key: str
    default_label: str
    value: Any
    default_units: str = ""
    default_show_plot: bool = True
    default_show_report: bool = True
    numeric: bool = False
    creation_order: int = 0


def valid_color(value: str, fallback: str) -> str:
    color = QColor(str(value))
    return color.name() if color.isValid() else fallback


def resolved_series(descriptor: SeriesDescriptor, override: SeriesOverride | None) -> dict[str, Any]:
    override = override or SeriesOverride()
    return {
        "label": descriptor.default_label if override.label is None else override.label,
        "show_series": bool(override.show_series),
        "show_legend": bool(override.show_legend),
        "color": descriptor.default_color if override.color is None else override.color,
        "linestyle": descriptor.default_linestyle if override.linestyle is None else override.linestyle,
        "marker": descriptor.default_marker if override.marker is None else override.marker,
        "linewidth": descriptor.default_linewidth if override.linewidth is None else override.linewidth,
        "alpha": descriptor.default_alpha if override.alpha is None else override.alpha,
        "order": descriptor.creation_order if override.order is None else override.order,
    }


def apply_series_override(descriptor: SeriesDescriptor, override: SeriesOverride | None) -> dict[str, Any]:
    values = resolved_series(descriptor, override)
    visible = values["show_series"]
    for artist in descriptor.artists:
        try:
            artist.set_visible(visible)
        except Exception:
            pass
        if values["color"]:
            if descriptor.supports_line:
                for method in ("set_color", "set_edgecolor", "set_facecolor"):
                    try:
                        getattr(artist, method)(values["color"])
                        break
                    except Exception:
                        continue
            else:
                for method in ("set_facecolor", "set_edgecolor"):
                    try:
                        getattr(artist, method)(values["color"])
                    except Exception:
                        continue
        if descriptor.supports_line and values["linestyle"] is not None:
            try:
                artist.set_linestyle(values["linestyle"])
            except Exception:
                pass
        if descriptor.supports_marker and values["marker"] is not None:
            try:
                artist.set_marker(values["marker"])
            except Exception:
                pass
        if values["linewidth"] is not None:
            try:
                artist.set_linewidth(values["linewidth"])
            except Exception:
                pass
        if values["alpha"] is not None:
            try:
                artist.set_alpha(values["alpha"])
            except Exception:
                pass
    return values


def format_annotation_value(descriptor: AnnotationDescriptor, override: AnnotationOverride | None) -> str:
    override = override or AnnotationOverride()
    value = descriptor.value
    if value is None:
        return ""
    if descriptor.numeric:
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value)
        else:
            if not math.isfinite(number):
                text = "n/a"
            elif override.format_mode == "Decimal":
                text = f"{number:.{override.precision}f}"
            elif override.format_mode == "Scientific":
                text = f"{number:.{override.precision}e}"
            elif override.format_mode == "Engineering":
                text = EngFormatter(places=override.precision).format_eng(number)
            else:
                text = f"{number:.{override.precision}g}"
    else:
        text = str(value)
    units = descriptor.default_units if override.units is None else override.units
    return f"{text} {units}".rstrip()


def apply_tick_style(axis, dimension: str, style: TickStyle) -> None:
    axis_obj = axis.xaxis if dimension == "x" else axis.yaxis
    if style.mode == "Fixed interval" and style.major_interval > 0:
        axis_obj.set_major_locator(MultipleLocator(style.major_interval))
    elif style.mode == "Manual" and style.manual_positions:
        axis_obj.set_major_locator(FixedLocator(style.manual_positions))
        labels = list(style.manual_labels)
        if len(labels) < len(style.manual_positions):
            labels.extend([f"{v:g}" for v in style.manual_positions[len(labels) :]])
        axis_obj.set_major_formatter(FixedFormatter(labels[: len(style.manual_positions)]))
    else:
        if style.format_mode == "Decimal":
            axis_obj.set_major_formatter(FormatStrFormatter(f"%.{style.precision}f"))
        elif style.format_mode == "Scientific":
            axis_obj.set_major_formatter(FormatStrFormatter(f"%.{style.precision}e"))
        elif style.format_mode == "Engineering":
            axis_obj.set_major_formatter(EngFormatter(places=style.precision))
        elif style.format_mode == "Automatic":
            formatter = axis_obj.get_major_formatter()
            if isinstance(formatter, ScalarFormatter):
                formatter.set_useOffset(style.show_offset)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-4, 5))
    if style.mode != "Manual" and style.format_mode != "Automatic":
        if style.format_mode == "Engineering":
            axis_obj.set_major_formatter(EngFormatter(places=style.precision))
        elif style.format_mode == "Scientific":
            axis_obj.set_major_formatter(FormatStrFormatter(f"%.{style.precision}e"))
        elif style.format_mode == "Decimal":
            axis_obj.set_major_formatter(FormatStrFormatter(f"%.{style.precision}f"))
    if style.minor_mode == "Automatic":
        axis_obj.set_minor_locator(AutoMinorLocator())
    elif style.minor_mode == "Fixed divisions" and style.minor_count > 0:
        axis_obj.set_minor_locator(AutoMinorLocator(style.minor_count + 1))
    else:
        axis_obj.set_minor_locator(NullLocator())
    axis.tick_params(
        axis=dimension,
        which="major",
        direction=style.direction,
        length=style.major_length,
        width=style.major_width,
        colors=style.tick_color,
        labelcolor=style.label_color,
        labelsize=style.label_size,
        bottom=style.bottom if dimension == "x" else None,
        top=style.top if dimension == "x" else None,
        left=style.left if dimension == "y" else None,
        right=style.right if dimension == "y" else None,
        labelbottom=style.label_bottom if dimension == "x" else None,
        labeltop=style.label_top if dimension == "x" else None,
        labelleft=style.label_left if dimension == "y" else None,
        labelright=style.label_right if dimension == "y" else None,
    )
    axis.tick_params(
        axis=dimension,
        which="minor",
        direction=style.direction,
        length=style.minor_length,
        width=style.minor_width,
        colors=style.tick_color,
    )
    for label in axis_obj.get_ticklabels():
        label.set_rotation(style.rotation)
        label.set_color(style.label_color)
        label.set_fontsize(style.label_size)


class ColorEdit(QLineEdit):
    """Editable Matplotlib color with a double-click QColorDialog shortcut."""

    def __init__(self, value: str = ""):
        super().__init__(value)
        self.setToolTip("Enter a Matplotlib color or double-click to choose one.")
        self.textChanged.connect(self._update_swatch)
        self._update_swatch(value)

    def _update_swatch(self, value: str) -> None:
        color = QColor(value)
        if not color.isValid():
            self.setStyleSheet("")
            return
        foreground = "#111111" if color.lightness() > 150 else "#ffffff"
        self.setStyleSheet(f"QLineEdit {{ background: {color.name()}; color: {foreground}; }}")

    def mouseDoubleClickEvent(self, event) -> None:
        initial = QColor(self.text())
        color = QColorDialog.getColor(initial if initial.isValid() else QColor("#000000"), self, "Choose color")
        if color.isValid():
            self.setText(color.name())
        super().mouseDoubleClickEvent(event)


class PlotEditorDialog(QDialog):
    """Modeless editor backed by the owning SmartFitter window's presentation state."""

    SERIES_COLUMNS = ["Label", "Plot", "Legend", "Color", "Line", "Marker", "Width", "Opacity"]
    ANNOTATION_COLUMNS = ["Plot", "Report", "Label / custom text", "Live value", "Format", "Precision", "Units"]

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self._loading = False
        self._axis_role = "main"
        self.setWindowTitle("Plot Editor")
        self.setModal(False)
        self.resize(1020, 700)
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        self._build_series_tab()
        self._build_annotation_tab()
        self._build_axes_tab()
        buttons = QHBoxLayout()
        self.save_defaults_btn = QPushButton("Save as defaults")
        self.reset_current_btn = QPushButton("Reset current file")
        self.factory_btn = QPushButton("Restore factory defaults")
        close_btn = QPushButton("Close")
        self.save_defaults_btn.clicked.connect(self.owner._save_presentation_defaults)
        self.reset_current_btn.clicked.connect(self.owner._reset_presentation_current)
        self.factory_btn.clicked.connect(self.owner._restore_presentation_factory)
        close_btn.clicked.connect(self.close)
        buttons.addWidget(self.save_defaults_btn)
        buttons.addWidget(self.reset_current_btn)
        buttons.addWidget(self.factory_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)
        self.refresh()

    @property
    def state(self) -> PlotPresentationState:
        return self.owner.presentation_state

    def _combo(self, items: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        return combo

    def _spin(self, lo: int, hi: int, value: int = 0) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        return spin

    def _dspin(self, lo: float, hi: float, value: float, decimals: int = 2) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _build_series_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        legend_box = QGroupBox("Legend")
        form = QGridLayout(legend_box)
        self.legend_visible = QCheckBox("Show legend")
        self.legend_title = QLineEdit()
        self.legend_location = self._combo(["best", "upper right", "upper left", "lower right", "lower left", "center right", "center left", "upper center", "lower center", "center"])
        self.legend_custom_anchor = QCheckBox("Custom anchor")
        self.legend_anchor_x = self._dspin(-2.0, 3.0, 0.98, 3)
        self.legend_anchor_y = self._dspin(-2.0, 3.0, 0.98, 3)
        self.legend_font = self._spin(5, 36, 9)
        self.legend_columns = self._spin(1, 10, 1)
        self.legend_text_color = ColorEdit()
        self.legend_frame = QCheckBox("Frame")
        self.legend_frame_color = ColorEdit()
        self.legend_edge_color = ColorEdit()
        self.legend_alpha = self._dspin(0.0, 1.0, 0.94, 2)
        self.legend_border_pad = self._dspin(0.0, 5.0, 0.4, 2)
        self.legend_label_spacing = self._dspin(0.0, 5.0, 0.5, 2)
        form.addWidget(self.legend_visible, 0, 0)
        form.addWidget(QLabel("Title"), 0, 1)
        form.addWidget(self.legend_title, 0, 2)
        form.addWidget(QLabel("Location"), 0, 3)
        form.addWidget(self.legend_location, 0, 4)
        form.addWidget(self.legend_custom_anchor, 1, 0)
        form.addWidget(QLabel("Anchor X/Y"), 1, 1)
        anchor = QHBoxLayout()
        anchor.addWidget(self.legend_anchor_x)
        anchor.addWidget(self.legend_anchor_y)
        form.addLayout(anchor, 1, 2)
        form.addWidget(QLabel("Font / columns"), 1, 3)
        fc = QHBoxLayout()
        fc.addWidget(self.legend_font)
        fc.addWidget(self.legend_columns)
        form.addLayout(fc, 1, 4)
        form.addWidget(QLabel("Text color"), 2, 0)
        form.addWidget(self.legend_text_color, 2, 1)
        form.addWidget(self.legend_frame, 2, 2)
        form.addWidget(self.legend_frame_color, 2, 3)
        form.addWidget(self.legend_edge_color, 2, 4)
        form.addWidget(QLabel("Frame opacity"), 3, 0)
        form.addWidget(self.legend_alpha, 3, 1)
        form.addWidget(QLabel("Border / row spacing"), 3, 2)
        spacing = QHBoxLayout()
        spacing.addWidget(self.legend_border_pad)
        spacing.addWidget(self.legend_label_spacing)
        form.addLayout(spacing, 3, 3, 1, 2)
        layout.addWidget(legend_box)
        self.series_table = QTableWidget(0, len(self.SERIES_COLUMNS))
        self.series_table.setHorizontalHeaderLabels(self.SERIES_COLUMNS)
        self.series_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.series_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.series_table)
        row_buttons = QHBoxLayout()
        for text, callback in (
            ("Move up", lambda: self._move_series(-1)),
            ("Move down", lambda: self._move_series(1)),
            ("Reset selected", self._reset_selected_series),
            ("Reset series", self._reset_all_series),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row_buttons.addWidget(button)
        row_buttons.addStretch(1)
        layout.addLayout(row_buttons)
        self.tabs.addTab(tab, "Series && Legend")
        for widget in (
            self.legend_visible, self.legend_title, self.legend_location, self.legend_custom_anchor,
            self.legend_anchor_x, self.legend_anchor_y, self.legend_font, self.legend_columns,
            self.legend_text_color, self.legend_frame, self.legend_frame_color,
            self.legend_edge_color, self.legend_alpha,
            self.legend_border_pad, self.legend_label_spacing,
        ):
            signal = getattr(widget, "toggled", None) or getattr(widget, "textChanged", None) or getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None)
            signal.connect(self._legend_changed)
        self.series_table.cellChanged.connect(self._series_changed)
        self.series_table.cellDoubleClicked.connect(self._series_double_clicked)

    def _build_annotation_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        style_box = QGroupBox("Annotation box")
        form = QGridLayout(style_box)
        self.annotation_mode = self._combo(["Off", "Compact", "Detailed", "Custom"])
        self.annotation_location = self._combo(["upper right", "upper left", "lower right", "lower left"])
        self.annotation_custom_anchor = QCheckBox("Custom anchor")
        self.annotation_anchor_x = self._dspin(-2.0, 3.0, 0.98, 3)
        self.annotation_anchor_y = self._dspin(-2.0, 3.0, 0.98, 3)
        self.annotation_halign = self._combo(["left", "center", "right"])
        self.annotation_valign = self._combo(["top", "center", "bottom"])
        self.annotation_font = self._spin(5, 36, 9)
        self.annotation_text_color = ColorEdit()
        self.annotation_bg_color = ColorEdit()
        self.annotation_edge_color = ColorEdit()
        self.annotation_alpha = self._dspin(0.0, 1.0, 0.82, 2)
        self.annotation_padding = self._dspin(0.0, 3.0, 0.35, 2)
        fields = [
            ("Preset", self.annotation_mode), ("Location", self.annotation_location),
            ("", self.annotation_custom_anchor), ("Anchor X", self.annotation_anchor_x),
            ("Anchor Y", self.annotation_anchor_y), ("Horizontal", self.annotation_halign),
            ("Vertical", self.annotation_valign), ("Font", self.annotation_font),
            ("Text color", self.annotation_text_color), ("Background", self.annotation_bg_color),
            ("Border", self.annotation_edge_color), ("Opacity", self.annotation_alpha),
            ("Padding", self.annotation_padding),
        ]
        for idx, (label, widget) in enumerate(fields):
            row, col = divmod(idx, 3)
            cell = QHBoxLayout()
            if label:
                cell.addWidget(QLabel(label))
            cell.addWidget(widget)
            form.addLayout(cell, row, col)
        layout.addWidget(style_box)
        self.annotation_table = QTableWidget(0, len(self.ANNOTATION_COLUMNS))
        self.annotation_table.setHorizontalHeaderLabels(self.ANNOTATION_COLUMNS)
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.annotation_table)
        controls = QHBoxLayout()
        for text, callback in (
            ("Add custom line", self._add_custom_annotation),
            ("Remove custom line", self._remove_custom_annotation),
            ("Move up", lambda: self._move_annotation(-1)),
            ("Move down", lambda: self._move_annotation(1)),
            ("Reset annotations", self._reset_annotations),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.tabs.addTab(tab, "Annotation Lines")
        for widget in (
            self.annotation_mode, self.annotation_location, self.annotation_custom_anchor,
            self.annotation_anchor_x, self.annotation_anchor_y, self.annotation_halign,
            self.annotation_valign, self.annotation_font, self.annotation_text_color,
            self.annotation_bg_color, self.annotation_edge_color, self.annotation_alpha,
            self.annotation_padding,
        ):
            signal = getattr(widget, "toggled", None) or getattr(widget, "textChanged", None) or getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None)
            signal.connect(self._annotation_style_changed)
        self.annotation_table.cellChanged.connect(self._annotation_changed)

    def _build_axes_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        role_row = QHBoxLayout()
        role_row.addWidget(QLabel("Axis"))
        self.axis_role_combo = QComboBox()
        self.axis_role_combo.currentTextChanged.connect(self._axis_role_changed)
        role_row.addWidget(self.axis_role_combo)
        role_row.addStretch(1)
        layout.addLayout(role_row)
        text_box = QGroupBox("Titles and labels")
        text_grid = QGridLayout(text_box)
        self.text_controls: dict[str, dict[str, Any]] = {}
        for row, (key, label) in enumerate((("title", "Title"), ("xlabel", "X label"), ("ylabel", "Y label"))):
            mode = self._combo(["Auto", "Custom"])
            edit = QLineEdit()
            visible = QCheckBox("Show")
            size = self._spin(5, 48, 10)
            weight = self._combo(["normal", "bold"])
            color = ColorEdit()
            padding = self._dspin(0.0, 30.0, 4.0, 1)
            self.text_controls[key] = {"mode": mode, "text": edit, "visible": visible, "size": size, "weight": weight, "color": color, "padding": padding}
            text_grid.addWidget(QLabel(label), row, 0)
            text_grid.addWidget(mode, row, 1)
            text_grid.addWidget(edit, row, 2)
            text_grid.addWidget(visible, row, 3)
            text_grid.addWidget(size, row, 4)
            text_grid.addWidget(weight, row, 5)
            text_grid.addWidget(color, row, 6)
            text_grid.addWidget(padding, row, 7)
            mode.currentTextChanged.connect(lambda value, k=key: self._axis_text_mode_changed(k, value))
            edit.textEdited.connect(lambda value, k=key: self._axis_text_edited(k, value))
            for widget in (visible, size, weight, color, padding):
                signal = getattr(widget, "toggled", None) or getattr(widget, "textChanged", None) or getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None)
                signal.connect(self._axis_changed)
        layout.addWidget(text_box)
        self.tick_tabs = QTabWidget()
        self.tick_controls: dict[str, dict[str, Any]] = {}
        for dimension in ("x", "y"):
            pane = QWidget()
            grid = QGridLayout(pane)
            controls: dict[str, Any] = {
                "mode": self._combo(["Automatic", "Fixed interval", "Manual"]),
                "interval": self._dspin(1e-12, 1e12, 1.0, 6),
                "minor_mode": self._combo(["Off", "Automatic", "Fixed divisions"]),
                "minor": self._spin(1, 10, 1),
                "format": self._combo(FORMAT_MODES),
                "precision": self._spin(0, 12, 4),
                "offset": QCheckBox("Show offset"),
                "size": self._spin(5, 36, 9),
                "color": ColorEdit(),
                "rotation": self._dspin(-180.0, 180.0, 0.0, 1),
                "direction": self._combo(["out", "in", "inout"]),
                "major_length": self._dspin(0.0, 20.0, 3.5, 1),
                "major_width": self._dspin(0.1, 8.0, 0.8, 1),
                "minor_length": self._dspin(0.0, 20.0, 2.0, 1),
                "minor_width": self._dspin(0.1, 8.0, 0.6, 1),
                "tick_color": ColorEdit(),
            }
            controls["offset"].setChecked(True)
            labels = [
                ("Mode", "mode"), ("Major interval", "interval"), ("Minor ticks", "minor_mode"),
                ("Number format", "format"), ("Precision", "precision"), ("", "offset"),
                ("Label size", "size"), ("Label color", "color"), ("Rotation", "rotation"),
                ("Direction", "direction"), ("Major length", "major_length"), ("Major width", "major_width"),
                ("Minor length", "minor_length"), ("Minor width", "minor_width"), ("Tick color", "tick_color"),
            ]
            for idx, (label, key) in enumerate(labels):
                row, pair = divmod(idx, 3)
                col = pair * 2
                if label:
                    grid.addWidget(QLabel(label), row, col)
                grid.addWidget(controls[key], row, col + 1)
            sides = QGroupBox("Visible sides")
            sides_layout = QHBoxLayout(sides)
            for key in ("bottom", "top", "left", "right", "label_bottom", "label_top", "label_left", "label_right"):
                check = QCheckBox(key.replace("_", " ").title())
                controls[key] = check
                sides_layout.addWidget(check)
            grid.addWidget(sides, 5, 0, 1, 6)
            minor_row = QHBoxLayout()
            minor_row.addWidget(QLabel("Fixed minor divisions"))
            minor_row.addWidget(controls["minor"])
            minor_row.addStretch(1)
            grid.addLayout(minor_row, 6, 0, 1, 6)
            manual = QTableWidget(0, 2)
            manual.setHorizontalHeaderLabels(["Position", "Label"])
            controls["manual"] = manual
            grid.addWidget(manual, 7, 0, 1, 6)
            manual_buttons = QHBoxLayout()
            add = QPushButton("Add tick")
            remove = QPushButton("Remove tick")
            add.clicked.connect(lambda _checked=False, d=dimension: self._add_manual_tick(d))
            remove.clicked.connect(lambda _checked=False, d=dimension: self._remove_manual_tick(d))
            manual_buttons.addWidget(add)
            manual_buttons.addWidget(remove)
            manual_buttons.addStretch(1)
            grid.addLayout(manual_buttons, 8, 0, 1, 6)
            self.tick_controls[dimension] = controls
            for key, widget in controls.items():
                if key == "manual":
                    widget.cellChanged.connect(self._axis_changed)
                    continue
                signal = getattr(widget, "toggled", None) or getattr(widget, "textChanged", None) or getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None)
                signal.connect(self._axis_changed)
            self.tick_tabs.addTab(pane, f"{dimension.upper()} ticks")
        layout.addWidget(self.tick_tabs)
        reset = QPushButton("Reset selected axis")
        reset.clicked.connect(self._reset_axis)
        layout.addWidget(reset)
        self.tabs.addTab(tab, "Titles && Axes")

    def open_for(self, section: str = "series", axis_role: str | None = None, dimension: str | None = None) -> None:
        self.refresh()
        index = {"series": 0, "annotation": 1, "axes": 2}.get(section, 0)
        self.tabs.setCurrentIndex(index)
        if axis_role and self.axis_role_combo.findText(axis_role) >= 0:
            self.axis_role_combo.setCurrentText(axis_role)
        if dimension in {"x", "y"}:
            self.tick_tabs.setCurrentIndex(0 if dimension == "x" else 1)
        elif dimension in {"title", "xlabel", "ylabel"}:
            self.text_controls[dimension]["text"].setFocus()
            self.text_controls[dimension]["text"].selectAll()
        self.show()
        self.raise_()
        self.activateWindow()

    def refresh(self) -> None:
        self._loading = True
        try:
            self._refresh_legend()
            self._refresh_series_table()
            self._refresh_annotation_style()
            self._refresh_annotation_table()
            roles = self.owner._active_presentation_axis_roles()
            current = self.axis_role_combo.currentText() or self._axis_role
            self.axis_role_combo.clear()
            self.axis_role_combo.addItems(roles or ["main"])
            if current in roles:
                self.axis_role_combo.setCurrentText(current)
            self._axis_role = self.axis_role_combo.currentText() or "main"
            self._refresh_axis_controls()
        finally:
            self._loading = False

    def _refresh_legend(self) -> None:
        s = self.state.legend
        self.legend_visible.setChecked(s.visible)
        self.legend_title.setText(s.title)
        self.legend_location.setCurrentText(s.location)
        self.legend_custom_anchor.setChecked(s.custom_anchor)
        self.legend_anchor_x.setValue(s.anchor_x)
        self.legend_anchor_y.setValue(s.anchor_y)
        self.legend_font.setValue(s.font_size)
        self.legend_columns.setValue(s.columns)
        self.legend_text_color.setText(s.text_color)
        self.legend_frame.setChecked(s.frame_visible)
        self.legend_frame_color.setText(s.frame_color)
        self.legend_edge_color.setText(s.edge_color)
        self.legend_alpha.setValue(s.frame_alpha)
        self.legend_border_pad.setValue(s.border_pad)
        self.legend_label_spacing.setValue(s.label_spacing)

    def _refresh_series_table(self) -> None:
        descriptors = self.owner._presentation_series_descriptors()
        descriptors = sorted(descriptors, key=lambda d: resolved_series(d, self.state.series.get(d.series_id))["order"])
        self.series_table.setRowCount(len(descriptors))
        for row, desc in enumerate(descriptors):
            values = resolved_series(desc, self.state.series.get(desc.series_id))
            label = QTableWidgetItem(str(values["label"]))
            label.setData(Qt.UserRole, desc.series_id)
            self.series_table.setItem(row, 0, label)
            for col, key in ((1, "show_series"), (2, "show_legend")):
                item = QTableWidgetItem()
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if values[key] else Qt.Unchecked)
                self.series_table.setItem(row, col, item)
            self.series_table.setItem(row, 3, QTableWidgetItem(str(values["color"] or "")))
            line = self._combo(list(LINE_STYLES))
            line.setCurrentText(LINE_STYLE_NAMES.get(values["linestyle"], "Solid"))
            line.setEnabled(desc.supports_line)
            line.currentTextChanged.connect(self._series_changed)
            self.series_table.setCellWidget(row, 4, line)
            marker = self._combo(list(MARKERS))
            marker.setCurrentText(MARKER_NAMES.get(values["marker"], "Default"))
            marker.setEnabled(desc.supports_marker)
            marker.currentTextChanged.connect(self._series_changed)
            self.series_table.setCellWidget(row, 5, marker)
            self.series_table.setItem(row, 6, QTableWidgetItem("" if values["linewidth"] is None else f"{values['linewidth']:.3g}"))
            self.series_table.setItem(row, 7, QTableWidgetItem("" if values["alpha"] is None else f"{values['alpha']:.3g}"))

    def _refresh_annotation_style(self) -> None:
        s = self.state.annotation_style
        self.annotation_mode.setCurrentText(s.mode)
        self.annotation_location.setCurrentText(s.location)
        self.annotation_custom_anchor.setChecked(s.custom_anchor)
        self.annotation_anchor_x.setValue(s.anchor_x)
        self.annotation_anchor_y.setValue(s.anchor_y)
        self.annotation_halign.setCurrentText(s.horizontal_alignment)
        self.annotation_valign.setCurrentText(s.vertical_alignment)
        self.annotation_font.setValue(s.font_size)
        self.annotation_text_color.setText(s.text_color)
        self.annotation_bg_color.setText(s.background_color)
        self.annotation_edge_color.setText(s.edge_color)
        self.annotation_alpha.setValue(s.background_alpha)
        self.annotation_padding.setValue(s.padding)

    def _annotation_rows(self):
        descriptors = list(self.owner._presentation_annotation_descriptors())
        known = {d.key for d in descriptors}
        for key, override in self.state.annotations.items():
            if key not in known and override.static_text is not None:
                descriptors.append(AnnotationDescriptor(key, "", "", default_show_plot=True, default_show_report=True, creation_order=10_000))
        return sorted(descriptors, key=lambda d: (self.state.annotations.get(d.key).order if self.state.annotations.get(d.key) and self.state.annotations[d.key].order is not None else d.creation_order))

    def _refresh_annotation_table(self) -> None:
        rows = self._annotation_rows()
        self.annotation_table.setRowCount(len(rows))
        for row, desc in enumerate(rows):
            override = self.state.annotations.get(desc.key) or AnnotationOverride()
            for col, default, value in (
                (0, desc.default_show_plot, override.show_plot),
                (1, desc.default_show_report, override.show_report),
            ):
                item = QTableWidgetItem()
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if (default if value is None else value) else Qt.Unchecked)
                self.annotation_table.setItem(row, col, item)
            label_text = override.static_text if override.static_text is not None else (override.label if override.label is not None else desc.default_label)
            label = QTableWidgetItem(label_text)
            label.setData(Qt.UserRole, desc.key)
            self.annotation_table.setItem(row, 2, label)
            value = QTableWidgetItem("" if override.static_text is not None else format_annotation_value(desc, override))
            value.setFlags(value.flags() & ~Qt.ItemIsEditable)
            self.annotation_table.setItem(row, 3, value)
            fmt = self._combo(FORMAT_MODES)
            fmt.setCurrentText(override.format_mode)
            fmt.setEnabled(desc.numeric and override.static_text is None)
            fmt.currentTextChanged.connect(self._annotation_changed)
            self.annotation_table.setCellWidget(row, 4, fmt)
            precision = QTableWidgetItem(str(override.precision))
            if not desc.numeric or override.static_text is not None:
                precision.setFlags(precision.flags() & ~Qt.ItemIsEditable)
            self.annotation_table.setItem(row, 5, precision)
            units = desc.default_units if override.units is None else override.units
            units_item = QTableWidgetItem(units)
            if override.static_text is not None:
                units_item.setFlags(units_item.flags() & ~Qt.ItemIsEditable)
            self.annotation_table.setItem(row, 6, units_item)

    def _refresh_axis_controls(self) -> None:
        style = self.state.axis(self._axis_role)
        for key in ("title", "xlabel", "ylabel"):
            value: TextStyle = getattr(style, key)
            c = self.text_controls[key]
            c["mode"].setCurrentText(value.mode)
            generated = self.owner._presentation_axis_text(self._axis_role, key)
            c["text"].setText(value.text if value.mode == "Custom" else generated)
            c["visible"].setChecked(value.visible)
            c["size"].setValue(value.font_size)
            c["weight"].setCurrentText(value.font_weight)
            c["color"].setText(value.color)
            c["padding"].setValue(value.padding)
        for dimension, value in (("x", style.x_ticks), ("y", style.y_ticks)):
            c = self.tick_controls[dimension]
            c["mode"].setCurrentText(value.mode)
            c["interval"].setValue(value.major_interval)
            c["minor_mode"].setCurrentText(value.minor_mode)
            c["minor"].setValue(value.minor_count)
            c["format"].setCurrentText(value.format_mode)
            c["precision"].setValue(value.precision)
            c["offset"].setChecked(value.show_offset)
            c["size"].setValue(value.label_size)
            c["color"].setText(value.label_color)
            c["rotation"].setValue(value.rotation)
            c["direction"].setCurrentText(value.direction)
            c["major_length"].setValue(value.major_length)
            c["major_width"].setValue(value.major_width)
            c["minor_length"].setValue(value.minor_length)
            c["minor_width"].setValue(value.minor_width)
            c["tick_color"].setText(value.tick_color)
            for side in ("bottom", "top", "left", "right", "label_bottom", "label_top", "label_left", "label_right"):
                c[side].setChecked(bool(getattr(value, side)))
            manual: QTableWidget = c["manual"]
            manual.setRowCount(len(value.manual_positions))
            for row, position in enumerate(value.manual_positions):
                manual.setItem(row, 0, QTableWidgetItem(f"{position:.12g}"))
                label = value.manual_labels[row] if row < len(value.manual_labels) else f"{position:g}"
                manual.setItem(row, 1, QTableWidgetItem(label))

    def _emit_change(self) -> None:
        if not self._loading:
            self.owner._on_presentation_changed()

    def _legend_changed(self, *_args) -> None:
        if self._loading:
            return
        s = self.state.legend
        s.visible = self.legend_visible.isChecked()
        s.title = self.legend_title.text()
        s.location = self.legend_location.currentText()
        s.custom_anchor = self.legend_custom_anchor.isChecked()
        s.anchor_x, s.anchor_y = self.legend_anchor_x.value(), self.legend_anchor_y.value()
        s.font_size, s.columns = self.legend_font.value(), self.legend_columns.value()
        s.text_color = valid_color(self.legend_text_color.text(), s.text_color)
        s.frame_visible = self.legend_frame.isChecked()
        s.frame_color = valid_color(self.legend_frame_color.text(), s.frame_color)
        s.edge_color = valid_color(self.legend_edge_color.text(), s.edge_color)
        s.frame_alpha = self.legend_alpha.value()
        s.border_pad = self.legend_border_pad.value()
        s.label_spacing = self.legend_label_spacing.value()
        self._emit_change()

    def _series_changed(self, *_args) -> None:
        if self._loading:
            return
        for row in range(self.series_table.rowCount()):
            item = self.series_table.item(row, 0)
            if item is None:
                continue
            sid = str(item.data(Qt.UserRole))
            line_combo = self.series_table.cellWidget(row, 4)
            marker_combo = self.series_table.cellWidget(row, 5)
            try:
                linewidth = float(self.series_table.item(row, 6).text())
            except (TypeError, ValueError):
                linewidth = None
            try:
                alpha = min(1.0, max(0.0, float(self.series_table.item(row, 7).text())))
            except (TypeError, ValueError):
                alpha = None
            self.state.series[sid] = SeriesOverride(
                label=item.text(),
                show_series=self.series_table.item(row, 1).checkState() == Qt.Checked,
                show_legend=self.series_table.item(row, 2).checkState() == Qt.Checked,
                color=self.series_table.item(row, 3).text().strip() or None,
                linestyle=LINE_STYLES.get(line_combo.currentText()) if line_combo else None,
                marker=MARKERS.get(marker_combo.currentText()) if marker_combo else None,
                linewidth=linewidth,
                alpha=alpha,
                order=row,
            )
        self._emit_change()

    def _series_double_clicked(self, row: int, column: int) -> None:
        if column != 3:
            return
        item = self.series_table.item(row, column)
        initial = QColor(item.text() or "#000000")
        color = QColorDialog.getColor(initial, self, "Series color")
        if color.isValid():
            item.setText(color.name())

    def _move_series(self, delta: int) -> None:
        row = self.series_table.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= self.series_table.rowCount():
            return
        self._series_changed()
        ids = [str(self.series_table.item(r, 0).data(Qt.UserRole)) for r in range(self.series_table.rowCount())]
        ids[row], ids[target] = ids[target], ids[row]
        for order, sid in enumerate(ids):
            self.state.series.setdefault(sid, SeriesOverride()).order = order
        self._loading = True
        self._refresh_series_table()
        self._loading = False
        self.series_table.selectRow(target)
        self._emit_change()

    def _reset_selected_series(self) -> None:
        row = self.series_table.currentRow()
        if row >= 0:
            sid = str(self.series_table.item(row, 0).data(Qt.UserRole))
            self.state.series.pop(sid, None)
            self.refresh()
            self._emit_change()

    def _reset_all_series(self) -> None:
        self.state.series.clear()
        self.refresh()
        self._emit_change()

    def _annotation_style_changed(self, *_args) -> None:
        if self._loading:
            return
        s = self.state.annotation_style
        previous_mode = s.mode
        s.mode = self.annotation_mode.currentText()
        if s.mode != previous_mode and s.mode in {"Off", "Compact", "Detailed"}:
            for override in self.state.annotations.values():
                if override.static_text is None:
                    override.show_plot = None
        s.visible = s.mode != "Off"
        s.location = self.annotation_location.currentText()
        s.custom_anchor = self.annotation_custom_anchor.isChecked()
        s.anchor_x, s.anchor_y = self.annotation_anchor_x.value(), self.annotation_anchor_y.value()
        s.horizontal_alignment, s.vertical_alignment = self.annotation_halign.currentText(), self.annotation_valign.currentText()
        s.font_size = self.annotation_font.value()
        s.text_color = valid_color(self.annotation_text_color.text(), s.text_color)
        s.background_color = valid_color(self.annotation_bg_color.text(), s.background_color)
        s.edge_color = valid_color(self.annotation_edge_color.text(), s.edge_color)
        s.background_alpha = self.annotation_alpha.value()
        s.padding = self.annotation_padding.value()
        self._emit_change()

    def _annotation_changed(self, *_args) -> None:
        if self._loading:
            return
        for row in range(self.annotation_table.rowCount()):
            key_item = self.annotation_table.item(row, 2)
            if key_item is None:
                continue
            key = str(key_item.data(Qt.UserRole))
            prior = self.state.annotations.get(key) or AnnotationOverride()
            fmt = self.annotation_table.cellWidget(row, 4)
            try:
                precision = max(0, min(12, int(self.annotation_table.item(row, 5).text())))
            except (TypeError, ValueError):
                precision = prior.precision
            label = key_item.text()
            if prior.static_text is not None:
                prior.static_text = label
            else:
                prior.label = label
            prior.show_plot = self.annotation_table.item(row, 0).checkState() == Qt.Checked
            prior.show_report = self.annotation_table.item(row, 1).checkState() == Qt.Checked
            prior.format_mode = fmt.currentText() if fmt else prior.format_mode
            prior.precision = precision
            prior.units = self.annotation_table.item(row, 6).text()
            prior.order = row
            self.state.annotations[key] = prior
        if self.state.annotation_style.mode in {"Compact", "Detailed"}:
            self.state.annotation_style.mode = "Custom"
            self._loading = True
            self.annotation_mode.setCurrentText("Custom")
            self._loading = False
        self._emit_change()

    def _add_custom_annotation(self) -> None:
        key = f"custom:{uuid.uuid4().hex}"
        self.state.annotations[key] = AnnotationOverride(static_text="Custom note", show_plot=True, show_report=True, order=len(self._annotation_rows()))
        self.refresh()
        self.annotation_table.selectRow(self.annotation_table.rowCount() - 1)
        self._emit_change()

    def _remove_custom_annotation(self) -> None:
        row = self.annotation_table.currentRow()
        if row < 0:
            return
        key = str(self.annotation_table.item(row, 2).data(Qt.UserRole))
        override = self.state.annotations.get(key)
        if override is not None and override.static_text is not None:
            self.state.annotations.pop(key, None)
            self.refresh()
            self._emit_change()

    def _move_annotation(self, delta: int) -> None:
        self._annotation_changed()
        row = self.annotation_table.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= self.annotation_table.rowCount():
            return
        keys = [str(self.annotation_table.item(r, 2).data(Qt.UserRole)) for r in range(self.annotation_table.rowCount())]
        keys[row], keys[target] = keys[target], keys[row]
        for order, key in enumerate(keys):
            self.state.annotations.setdefault(key, AnnotationOverride()).order = order
        self.refresh()
        self.annotation_table.selectRow(target)
        self._emit_change()

    def _reset_annotations(self) -> None:
        self.state.annotations.clear()
        self.refresh()
        self._emit_change()

    def _axis_role_changed(self, role: str) -> None:
        if not role:
            return
        self._axis_role = role
        if not self._loading:
            self._loading = True
            self._refresh_axis_controls()
            self._loading = False

    def _axis_text_mode_changed(self, key: str, mode: str) -> None:
        if self._loading:
            return
        if mode == "Custom":
            # The Auto field displays the live generated text. Preserve it as
            # the initial custom value when the user explicitly changes mode.
            getattr(self.state.axis(self._axis_role), key).text = self.text_controls[key]["text"].text()
        self._axis_changed()
        if mode == "Auto":
            self._loading = True
            self.text_controls[key]["text"].setText(self.owner._presentation_axis_text(self._axis_role, key))
            self._loading = False

    def _axis_text_edited(self, key: str, _text: str) -> None:
        if self._loading:
            return
        controls = self.text_controls[key]
        if controls["mode"].currentText() != "Custom":
            self._loading = True
            controls["mode"].setCurrentText("Custom")
            self._loading = False
        self._axis_changed()

    def _axis_changed(self, *_args) -> None:
        if self._loading:
            return
        style = self.state.axis(self._axis_role)
        for key in ("title", "xlabel", "ylabel"):
            c = self.text_controls[key]
            value: TextStyle = getattr(style, key)
            value.mode = c["mode"].currentText()
            value.text = c["text"].text()
            value.visible = c["visible"].isChecked()
            value.font_size = c["size"].value()
            value.font_weight = c["weight"].currentText()
            value.color = valid_color(c["color"].text(), value.color)
            value.padding = c["padding"].value()
        for dimension, value in (("x", style.x_ticks), ("y", style.y_ticks)):
            c = self.tick_controls[dimension]
            value.mode = c["mode"].currentText()
            value.major_interval = c["interval"].value()
            value.minor_mode = c["minor_mode"].currentText()
            value.minor_count = c["minor"].value()
            value.format_mode = c["format"].currentText()
            value.precision = c["precision"].value()
            value.show_offset = c["offset"].isChecked()
            value.label_size = c["size"].value()
            value.label_color = valid_color(c["color"].text(), value.label_color)
            value.rotation = c["rotation"].value()
            value.direction = c["direction"].currentText()
            value.major_length = c["major_length"].value()
            value.major_width = c["major_width"].value()
            value.minor_length = c["minor_length"].value()
            value.minor_width = c["minor_width"].value()
            value.tick_color = valid_color(c["tick_color"].text(), value.tick_color)
            for side in ("bottom", "top", "left", "right", "label_bottom", "label_top", "label_left", "label_right"):
                setattr(value, side, c[side].isChecked())
            manual: QTableWidget = c["manual"]
            positions: list[float] = []
            labels: list[str] = []
            manual.blockSignals(True)
            for row in range(manual.rowCount()):
                position_item = manual.item(row, 0)
                try:
                    position = float(position_item.text())
                except (AttributeError, TypeError, ValueError):
                    if position_item is not None:
                        position_item.setBackground(QColor("#5c1f1f"))
                        position_item.setToolTip("Enter a finite numeric tick position.")
                    continue
                if not math.isfinite(position) or position in positions:
                    if position_item is not None:
                        position_item.setBackground(QColor("#5c1f1f"))
                        position_item.setToolTip("Tick positions must be finite and unique.")
                    continue
                if position_item is not None:
                    position_item.setBackground(QColor("transparent"))
                    position_item.setToolTip("")
                positions.append(position)
                label_item = manual.item(row, 1)
                labels.append(label_item.text() if label_item else f"{position:g}")
            manual.blockSignals(False)
            ordered = sorted(zip(positions, labels), key=lambda item: item[0])
            value.manual_positions = [item[0] for item in ordered]
            value.manual_labels = [item[1] for item in ordered]
        self._emit_change()

    def _add_manual_tick(self, dimension: str) -> None:
        table: QTableWidget = self.tick_controls[dimension]["manual"]
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem("0"))
        table.setItem(row, 1, QTableWidgetItem("0"))

    def _remove_manual_tick(self, dimension: str) -> None:
        table: QTableWidget = self.tick_controls[dimension]["manual"]
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)
            self._axis_changed()

    def _reset_axis(self) -> None:
        self.state.axes[self._axis_role] = axis_style_for_role(self._axis_role)
        self._emit_change()
        self.refresh()
