# SmartFitter App Reference

This document is the single source of truth for how the current `nvfit` app works, how to run it, and how to safely modify it later.

## 1) What this app is

`nvfit` is a Python toolkit for NV-center experiment analysis with:

- A GUI app (`nvfit.gui_app`) for interactive loading, plotting, fitting, and exporting.
- A batch CLI app (`nvfit.pro_batch`) for automated processing of folders of `.mat` files.
- A shared core for I/O, preprocessing, model definitions, fitting, and diagnostics.

Primary goals:

- Robust loading of MATLAB `savedData` and `data` + `scanInfo` structures.
- Correct experiment type detection from metadata first (axis labels / scan structure), filename second.
- Physics-aware fitting workflows (especially Ramsey).
- Useful outputs for lab review: plots, fit metrics, extracted NV-relevant parameters.

## 2) Environment and dependencies

From `requirements.txt`:

- `numpy`
- `scipy`
- `matplotlib`
- `h5py`
- `PySide6`
- `pytest`

## 3) Run commands

From repository root:

- GUI:
  - `python -m nvfit.gui_app`
- Batch:
  - `python -m nvfit.pro_batch --input-dir "D:\BacklundLabResearch\Code\Lab_Data_Analysis\TestData" --output-dir "D:\BacklundLabResearch\Code\Lab_Data_Analysis\batch_out"`
- Tests:
  - `python -m pytest -q`

## 4) High-level architecture

Core files and responsibilities:

- `nvfit/io_mat.py`
  - Loads MATLAB files from both supported root schemas, extracts vectors, builds axes, infers experiment type.
  - Provides `ExperimentTrace` with scan metadata (`scan_dim`, `scan_axes`, `fit_allowed`, optional 2D grids).
- `nvfit/preprocess.py`
  - ROI slicing, binning, smoothing.
- `nvfit/models.py`
  - Model functions (Ramsey/Rabi/SpinEcho/ODMR).
- `nvfit/fit_engine.py`
  - Generic multistart nonlinear fitting (`least_squares`) and diagnostics packaging.
- `nvfit/ramsey.py`
  - Physics-first Ramsey model selection.
- `nvfit/diagnostics.py`
  - R2/RMSE/AIC/BIC/Durbin-Watson/bound-hit helpers.
- `nvfit/profiles.py`
  - Profile presets and quality-gate thresholds.
- `nvfit/gui_app.py`
  - Main GUI, plot controls, preferences, export behavior.
- `nvfit/pro_batch.py`
  - Batch processing and report generation.

## 5) Data model and type detection

`io_mat.py` key concepts:

- `DataMode`: `contrast`, `raw_signal`, `difference`
- `ExperimentTrace` includes:
  - file/path
  - `experiment_type`
  - `scan_axes`, `scan_dim` (`time`, `scan1d`, `scan2d`)
  - `fit_allowed`
  - 1D vectors (`x_ns`, `y`, `signal`, `reference`)
  - optional raw acquisition axis (`raw_x`, `raw_x_label`)
  - trace metadata dictionary (`metadata`)
  - optional 2D grids (`x2d`, `y2d`, `z2d`)

Type inference behavior:

1. Prefer scan metadata (`loadedScan.type` or new `scanInfo` metadata) for classification:
   - RF frequency -> ODMR
   - Pulse duration -> Rabi/Ramsey (filename hints for Ramsey)
   - XY8 / CPMG / DD notes -> DynamicDecoupling
   - Stage axes -> spatial scan
2. Fallback to filename keywords.

Spatial scan behavior:

- `scan1d` and `scan2d` are plot-only (no fit).
- Time-domain experiments keep fit enabled.

## 6) Models and parameter meaning

Defined in `models.py`:

- `ramsey_simple(t, y0, A, f, phi, tau, n)`
- `ramsey_hyperfine_n14(...)`
- `ramsey_extended(...)`
- `rabi_model(t, y0, m, A, f, phi, tau)`
- `rabi_adaptive_pulse_area_model(t, c, a, T, beta, f0, chirp1, chirp2, delay)`
- `rabi_phase_ramp_model(t, c, a, T, beta, f, tau_ramp, phi)`
- `rabi_dual_model(...)`
- `spin_echo_model(t, y0, A, T2, n)`
- `odmr_lorentzian(t, y0, C, w, x0)`

Dynamic-decoupling behavior:

- `DynamicDecoupling` uses the stretched-exponential decay model path (`spin_echo_model`)
- the fit axis is converted to total evolution time when new-format sequence metadata provides the required timing fields

Important conventions:

- Time axes are generally in ns in this app.
- Frequencies in these models are in 1/ns (reported as GHz, often converted to MHz in metrics).
- Stretched decays use `abs(t/T)` internally where needed to avoid non-finite behavior during optimizer steps.

GUI parameter metadata:

- `gui_app.py` includes `PARAM_METADATA` mapping parameter names to:
  - symbol (display text)
  - units
  - plain-language description

This metadata is used in:

- Parameter priors/locks table
- Fit summary text
- Tooltips

## 7) Fitting engine behavior

`fit_engine.fit_model_multistart(...)`:

- Uses `scipy.optimize.least_squares` with:
  - robust loss: `soft_l1`
  - configurable `f_scale`
  - multistart seeds + bounds
- Returns `FitResult`:
  - params/errors/y_fit
  - r2/rmse/aic/bic/durbin_watson
  - bound-hit flags
  - success/message

## 8) GUI behavior summary

Main class: `SmartFitterMainWindow` in `gui_app.py`.

### 8.1 Analyze and Batch workspaces

- **Analyze**: a compact Data/Prepare panel, one WYSIWYG Matplotlib plot, and a contextual Fit Results or 2D Map Controls inspector.
- **Batch**: folder setup, progress/cancellation, searchable results, a run log, and selected-result details.
- Reversible fit exclusions and transforms remain available under the collapsed **Advanced** section without plot interaction modes.

Analyze side panes are scrollable with collapsible section groups and no horizontal scrolling. UI spacing, side-panel minima, toolbar wrapping, and Matplotlib text/marker sizes scale with the available display and current window size.
Mouse-wheel safety is enforced for combo/spin controls: wheel changes are ignored unless the widget has focus.

### 8.2 Plot controls

The Plot section keeps the high-frequency data-layer toggles, overall trace style, figure size, and export DPI. Detailed presentation editing is centralized in the modeless **Plot Editor**, opened from **Edit plot...**, the plot context menu, or by double-clicking a legend, annotation, title, axis label, or tick label.

The Plot Editor contains:

- **Series & Legend**: editable labels and order, independent plot/legend visibility, color, line, marker, width, opacity, and global legend frame/layout controls.
- **Annotation Lines**: ordered dynamic fit/metadata rows with separate live-plot and report visibility, editable labels, units, number formats, and custom static lines. Dynamic values update after a refit.
- **Titles & Axes**: auto or custom titles/labels plus per-axis major/minor tick spacing, formatting, placement, appearance, and exact manual position/label tables.

Edits update the WYSIWYG figure immediately. Current-file overrides survive redraws and same-file mode reloads but reset when a different primary file is selected. **Save as defaults** explicitly promotes the current presentation to future files; reset actions restore the saved or factory presentation.

Base plot toggles/settings include:

- The always-visible quick bar exposes Data, Fit, Smooth, Legend, Annotations, Signal, Reference, Iterations, and iteration selection
- Show data/smoothed/fit/ODMR peaks
- Show residual subplot
- Observable and iteration layers
- Confidence and Rabi-envelope layers
- Overall plot style
- Figure width/height
- Save DPI

Factory presentation defaults are line-plus-scatter traces with smoothing,
iteration mean, and the Rabi envelope disabled on an 8.5 × 5.5 inch white
figure at 300 DPI. Existing preferences are migrated only when they still
match previous factory values.

Presentation settings affect the live plot, clipboard image, PNG/PDF/SVG outputs, and annotated report figures. Batch CLI plots remain independent.

### 8.3 Parameter editor

The parameter table is:

- `Parameter | Value | Min | Max | Lock`

Behavior:

- `Value`: seed/lock value
- `Min` and `Max`: optional per-parameter bounds
- `Lock`: hard-fixes parameter to `Value`

### 8.4 Rabi mode

In Fit Strategy:

- `Adaptive pulse-area (recommended)` fits an explicit timing delay, a stretched decay envelope, and a low-order smooth pulse-area calibration. `chirp1` and `chirp2` describe effective rotation rate versus programmed duration; they are not microwave-carrier chirp parameters.
- `Phase-ramp` fits an exponential amplitude turn-on plus a stretched decay envelope.
- `Long damped Rabi` emphasizes early calibration lobes in long, strongly decayed scans.
- `Chirped cosine` and `Constant-frequency damped cosine` remain comparison models.
- `Auto compare` uses fit quality and BIC while rejecting pathological adaptive solutions.

For adaptive fits, `delay` is fitted directly. Saved `RFRampTime` metadata is retained separately as `saved_rf_ramp_time_ns`; it is not confused with the exponential turn-on timescale `tau_ramp`.

### 8.5 ODMR mode

Supports:

- Display measured peaks by default or invert them to a dip view
- Multi-peak detection mode (peak-pick only)
- Optional peak-center range constraints for peak picking and Lorentzian fitting

Behavior:

- Multi-peak + peak-pick mode reports peak locations (no forced single-peak Lorentzian fit).

### 8.6 Equation rendering and custom expressions

- Active model equations are rendered with Matplotlib mathtext in Fit Strategy.
- `CustomModel` exposes editable expression source and live preview.

### 8.7 Preferences/options

Persistent controls live directly in the Plot, Annotation, Export, and Map sections. `Options > Save Current Preferences` stores them without a duplicate preferences dialog.

Preferences are persisted via `QSettings("BacklundLab", "SmartFitterPy")`.

Stored examples and migrated legacy settings:

- figure size
- save DPI
- legend visibility/location/font
- residual visibility
- series visibility toggles
- export format toggles
- report-content selector checkboxes

Legacy legend/annotation preferences and session/preset fields are still read and mapped into the Plot Editor. Detailed per-series, line-by-line annotation, title, and tick overrides do not change the analysis-session schema.

## 9) Export and output schema

GUI save (`on_save_results`) writes:

- `<stem>_gui_fit.png`
- `<stem>_gui_report.png` (non-2D scan mode)
- `<stem>_gui_result.json`
- `<stem>_gui_result.csv`

Report figure behavior:

- Metrics are displayed in a dedicated right-side panel to avoid blocking data traces.
- Included metrics are selected via Fit + Export report-content checkboxes and persisted in settings.

JSON contains (typical):

- file/profile/scan metadata
- selected model
- status + reason
- fit metrics
- params + errors
- NV metrics
- ODMR peak locations (if any)

Batch output (`pro_batch.py`):

- Per-file `<stem>_fit.png`, `<stem>_result.json`
- Aggregates:
  - `batch_summary.csv`
  - `validation_report.csv`

For spatial scans, batch writes plot-only results with explicit no-fit status.

## 10) Profiles and quality gates

From `profiles.py`:

- Profiles: Ramsey, Rabi, SpinEcho, DynamicDecoupling, T1, ODMR, LineScan, Scan2D
- Each profile defines:
  - default observable mode
  - preprocessing defaults
  - candidate model names
  - quality thresholds (`r2_min`, `max_bound_hits`)

`classify_status(...)`:

- `FAIL` if `r2` below profile minimum
- `WARN` if too many bound hits
- else `PASS`

## 11) How to safely modify this app

Use this checklist before/after edits:

1. Preserve `ExperimentTrace` contract in `io_mat.py`.
2. If adding model params:
   - update model function
   - update fit seed/bounds path
   - update `PARAM_METADATA` for GUI readability.
3. Keep plot behavior centralized:
   - reflect changes in `_plot_raw`, `_plot_fit`, and save/report path.
4. Keep GUI options synchronized with persisted settings.
5. For spatial scans, do not accidentally re-enable fitting unless intentionally designed.
6. Update/extend tests when adding logic.
7. Run:
   - `python -m pytest -q`
   - optional batch smoke test on `TestData`.

## 12) Extension recipes

### 12.1 Add a new experiment model

1. Add function in `models.py`.
2. Add fit path in:
   - GUI: `_fit_single_model`
   - batch: `_fit_non_ramsey` or dedicated selection flow
3. Add profile entry in `profiles.py` if needed.
4. Add param metadata in `gui_app.py`.
5. Add model tests in `tests/test_models.py`.

### 12.2 Add a new GUI option

1. Add widget in `_build_ui`.
2. Connect to plot/config state methods (`_apply_plot_controls_to_state`, `_refresh_plot_only`).
3. Persist with `QSettings` (`_save_preferences`, `_load_preferences`).
4. Confirm export path respects the option.

### 12.3 Change output schema

1. Update JSON/CSV payload in GUI and/or batch.
2. Update tests that validate schema (`tests/test_batch_schema.py` and related checks).
3. Keep backward compatibility notes if external analysis scripts rely on keys.

## 13) Common troubleshooting

- Recursive GUI reload errors:
  - Ensure profile default application does not recursively call file reload loops.
- Poor Rabi fits:
  - Start with `Rabi mode = Adaptive pulse-area (recommended)`.
  - Compare the residual scale with iteration SEM before adding more model complexity.
  - Use `Phase-ramp` only for a plausible first-order amplitude-settling transient and `Chirped cosine` only as an empirical comparison.
- ODMR many-peak traces:
  - Use peak-pick mode and optional ranges rather than forcing single Lorentzian fit.
- Spatial scans misfit:
  - Verify `scan_dim` detection in loader and `fit_allowed=False` logic.

## 14) AI handoff notes

If an AI agent modifies this project, it should:

- Read this file first.
- Then inspect:
  - `nvfit/gui_app.py`
  - `nvfit/io_mat.py`
  - `nvfit/pro_batch.py`
  - `nvfit/models.py`
  - `nvfit/profiles.py`
- Preserve existing behavior unless explicitly requested:
  - metadata-first experiment detection
  - scan plot-only handling
  - explicit Rabi mode selection
  - persisted GUI preferences

## 15) New integrated capabilities (current release)

This release adds the following production features:

- Overlay and comparison:
  - multi-file overlay registry with an explicit primary trace, per-file checkboxes, Show all, and Primary only controls
  - newly opened primary files start without implicit overlays; explicitly added/checked overlays autoscale to their combined full axis range
  - baseline subtraction against selected overlay trace (interpolated x-alignment)
  - comparison summary and quick comparison plot for selected metric.

- ODMR upgrades:
  - optional multi-peak Lorentzian fitting (`ODMRMulti`)
  - user-selected or auto peak count
  - calibrated axis conversion (`offset + scale * x`)
  - derived D/E estimates when >=2 resonance centers are available.

- Presets and reproducibility:
  - save/load analysis presets to JSON (ROI, model, locks, plot controls, ODMR options)
  - exports include provenance (timestamp, version, preset, exclusions, baseline trace).

- Data workflow:
  - recent files and favorite folder helpers
  - sample metadata fields in GUI (sample/condition/power/date/notes)
  - metadata included in JSON/CSV exports.

- Interactive cleaning:
  - exclusion ranges by x window
  - Ctrl+click nearest-point exclusion
  - exclusions applied before fitting and exported in provenance.

- Rabi/Ramsey:
  - Rabi pi/pi2 calibration uncertainty reported from fitted frequency uncertainty
  - optional persistent FFT panel in GUI subplot area.

- Export/reporting:
  - PNG/PDF/SVG figure export from GUI
  - report PNG/PDF export
  - Origin-friendly bundle (`*_origin_bundle.csv` + metadata JSON).

- Batch workflow:
  - `--single-file` mode
  - metadata ingestion via `--metadata-file` (CSV/JSON map)
  - exclusion flags via `--exclude-file` (CSV/JSON map)
  - custom plugin model via `--plugin-file` (JSON model spec)
  - `aggregate_report.csv` generated (excluding explicitly excluded files).

## 16) Analysis sessions, transforms, and clipboard

- `File > Save Analysis Session...` writes an atomic v2 `.nvfit-session.json` document and can read v1 sessions. It records source fingerprints, overlays/baselines, profile/model choices, locks, preprocessing, masks, transforms, plot/map state, metadata, view limits, and valid fit state. Missing sources can be located interactively; changed sources invalidate stale fits.
- Analysis transforms are display/measurement tools: baseline removal, detrend, normalization, derivative, integral, uniform resampling, and safe vector expressions. The fitting pipeline remains on the pre-transform processed data, so fit parameters retain their experiment-model meaning.
- Plot clicks create a temporary yellow inspection marker that is excluded from the legend. Clicking the same point again, pressing `Escape`, or using `Clear` removes it without changing or excluding measured data.
- Double-clicking presentation objects takes priority over point inspection and opens the Plot Editor at the matching series, annotation, title/label, or tick section.
- `Copy figure` and `Ctrl+Shift+C` render the current Matplotlib export figure at at least 300 DPI and place a PNG image on the system clipboard.
- A labeled 2D Map Controls inspector appears beside scan maps. It controls colormap/reversal, editable robust percentiles or manual color limits, exact X/Y view bounds, full-view reset, pan, box zoom, pointer-centered wheel zoom, default-on X/Y swapping, optional equal-axis scaling, linecuts, and peak/dip navigation. Settings persist in `QSettings`.

## 17) Plugin spec for custom model

`--plugin-file` and GUI custom-model loader expect JSON with:

```json
{
  "name": "MyCustomModel",
  "expression": "y0 + A*np.cos(2*np.pi*f*t+phi)",
  "param_names": ["y0", "A", "f", "phi"],
  "lower": [-0.2, 0.0, 1e-5, -6.283],
  "upper": [0.2, 0.2, 0.2, 6.283],
  "seed": [0.0, 0.02, 0.003, 0.0]
}
```

Expression rules:

- expression can reference `t`, `np`, and each parameter name
- evaluator is restricted (`__builtins__` disabled)
- output must be numeric and vectorizable over `t`.

## 17) New batch CLI examples

- Single file:
  - `python -m nvfit.pro_batch --single-file "D:\\...\\Rabi10us.mat" --output-dir "D:\\...\\out_single"`

- Batch with metadata/exclude maps:
  - `python -m nvfit.pro_batch --input-dir "D:\\...\\TestData" --output-dir "D:\\...\\out" --metadata-file "D:\\...\\meta.csv" --exclude-file "D:\\...\\exclude.csv"`

- Batch custom model:
  - `python -m nvfit.pro_batch --input-dir "D:\\...\\TestData" --output-dir "D:\\...\\out_custom" --plugin-file "D:\\...\\custom_model.json"`

