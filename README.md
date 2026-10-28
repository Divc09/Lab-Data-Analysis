# SmartFitter: NV-Center Lab Data Analysis

SmartFitter is a Python application for loading, visualizing, fitting, and exporting data from nitrogen-vacancy (NV) center experiments saved as MATLAB `.mat` files. It provides an interactive desktop GUI for individual measurements and a command-line batch pipeline for folders of experiments.

## SmartFitter v2 (new interface)

`smartfitter/` is a new desktop interface built on the same `nvfit` loaders, fits, spot detection, map overlays and batch pipeline. SmartFitter v2 is now the default app: `python -m nvfit.gui_app` and `Launch SmartFitter.vbs` start it. The original Qt app is deprecated and kept for reference; start it with `python -m nvfit.gui_app --legacy`.

Launch it:

- Double-click **SmartFitter** on the desktop or in the Start menu. Create those shortcuts once with `.venv\Scripts\python.exe -m smartfitter.install` (add `--associate` to put SmartFitter in the "Open with" list for `.mat` files; `--remove` undoes both).
- Or run `python -m smartfitter [files...]`. A second launch hands its files to the window that is already open.

Workspaces:

- **Analyze**: 1D experiments (fit, residuals, iterations, SEM band, envelope, signal/reference right axis, FFT, pinned points, measure, box-select exclusion), time traces (stacked detectors, decay fit, rolling mean, A/B window statistics, retries/events, Lomb–Scargle periodogram), and 2D maps (physical or square pixels, axis swap, colormaps/limits/log, linecuts, spot detection and detector comparison, RGB overlays).
- **Plot editing**: double-click a title or axis label to retype it, drag the legend or results box, double-click a legend entry to restyle a series, or use the full editor (series, legend, axes and ticks, results lines, added text/lines/regions, presets, save as default).
- **Compare**: overlay files with normalization, fit them all, and plot a fitted value against time, file order, or a number in the file name.
- **Figure**: multi-panel figures at slide or journal size; copy, SVG, PNG or PDF.
- **Batch**: runs `nvfit.pro_batch` with live progress and a filterable table.

Watch a folder from the Library to see new files and checkpoint updates as they are saved. Keyboard shortcuts match the original app (Ctrl+O, Ctrl+Shift+O, Ctrl+K, Ctrl+F, Ctrl+Shift+F, Ctrl+S, Ctrl+Shift+S, Ctrl+Alt+O, Ctrl+Z, Ctrl+Y, Ctrl+Shift+C, Ctrl+Home, Alt+1–7, Alt+Down / Alt+Up, Esc).

Library:

- Drag the edges of the library and inspector panels to resize them. Double-click a handle to reset its width.
- Alt+Down / Alt+Up open the next or previous file in the filtered list.
- Click the folder name to list recent folders.
- "Refit on update" re-runs the fit when a watched checkpoint file changes.

Compare:

- Save CSV exports the trend table.
- Fits that did not pass are outlined in red on the trend plot.

Export:

- PDF export is now vector, converted from the same SVG (via svglib/reportlab). PNG and SVG are unchanged.

Other:

- Folder scans are cached in `%APPDATA%\SmartFitter\peek_cache.json`, so reopening large folders is faster.
- Fit uncertainties: the covariance estimate in `nvfit/fit_engine.py` now scales the Jacobian by column and uses an SVD cutoff. Badly scaled parameters (e.g. Rabi T, β, delay) previously got unrealistically small errors. Unidentifiable parameters report NaN, shown as "n/a" in the app.
- Auto fits 1D line scans with one Gaussian peak on a constant background, in Analyze, Compare and Batch. Choose GaussianPeaks with more peaks, a custom model, or `--model-override` for anything else.

Iteration sensitivity (Analyze → Fit, or Process):

- **Check iteration sensitivity…** fits the current full selection and refits its mean with each iteration omitted in turn, then automatically deselects flagged iterations and refits the main plot. Select all first to use every available complete iteration. This supports T1 and other 1D models with iteration data.
- The flag rule is the absolute percentage change in the selected parameter (T1/T2/timescale by default), with an adjustable threshold of 10%. Failed or noncomparable omission fits are not automatically excluded. If every iteration is flagged, the selection is preserved. The rule identifies sensitivity, not proven statistical outliers.
- Checked **Excluded** rows immediately change the main selection and refit. **Exclude listed & refit** applies a typed range. **Test combined omission** adds a comparison row for the current excluded group. Restore any iteration in **Process → Iterations used for mean + fit**, or use **Restore baseline iterations** in the popup. Undo restores the previous selection and fit.
- The popup provides **Error bars** and **Rescale plot** controls, a scrollable table, compact source labels with full provenance on hover, and an automatically recorded exclusion reason. The Fit/Process panel retains its scroll position during updates.
- Diagnostics and exclusion reasons persist in sessions and JSON exports; **Export sensitivity CSV** saves the parameter comparison and source iteration identities. Changed settings/source files mark old diagnostics stale. These correlated refits do not provide independent uncertainty estimates or justify exclusion on their own.

## Highlights

- Loads legacy MATLAB `savedData` files and newer single- or dual-detector snapshot exports.
- Supports Rabi, Ramsey, spin echo, dynamic decoupling, T1, ODMR, DEER, line scans, and 2D stage scans.
- Loads measured time traces from the acquisition code, including reference-only fluorescence and dual detectors.
- Compares 2D maps with pink/teal RGB overlays, across detectors or loaded files.
- Provides experiment-aware fitting, diagnostics, exclusions, transformations, overlays, iteration views, and derived NV metrics.
- Includes a unified Plot Editor for legends, annotations, titles, axes, ticks, secondary axes, and report presentation.
- Exports publication-ready PNG, PDF, and SVG figures plus JSON, CSV, and report outputs.
- Includes representative test data and an offscreen GUI regression suite.

## Requirements

- Python 3.12 or newer
- Windows, macOS, or Linux

The GUI uses Qt through PySide6. The included one-click launcher is Windows-specific; the Python launch command works on every supported platform.

## Quick start

Clone the repository and enter its directory:

```bash
git clone https://github.com/Divc09/Lab-Data-Analysis.git
cd Lab-Data-Analysis
```

Create an isolated Python environment.

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pins in `requirements.txt` are the tested environment (Windows 11, Python 3.14.5). For running the tests or building the portable app, install `requirements-dev.txt` instead; for the research scripts under `scripts/`, `requirements-scripts.txt`.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start SmartFitter:

```bash
python -m smartfitter
# legacy Qt app (deprecated): python -m nvfit.gui_app --legacy
```

On Windows, after creating `.venv`, you can also double-click `Launch SmartFitter.vbs`.

## Basic GUI workflow

1. Open or drag in a supported `.mat` file.
2. Leave the analysis profile on **Auto**, or choose an experiment-specific profile.
3. For a dual-detector file, choose Detector 1 or Detector 2, then use **Apply to** to target the active detector or both detector streams. Labels are editable for the current session.
4. Select the observable: contrast, signal, reference, or difference.
5. Adjust the fit range in the **Model** inspector tab; binning, smoothing, exclusions, and transforms stay in the adjacent **Process** tab.
6. Run the fit for supported 1D experiments. In **Both detectors** mode, SmartFitter fits each stream independently with the same model/settings and retains both results.
7. Double-click plot text, legends, annotations, measured/fit series, markers, or colorbar labels for a compact live editor; use **Edit plot** or the **Plot** inspector tab for complete control.
8. Export the figure, fit data, metadata, or report.

Spatial scans are plot-only for the 1D fit engine. Their map controls provide color scaling, view bounds, pan and zoom, peak/dip navigation, and cursor or best-point linecuts. In a 1D plot, Ctrl/Shift-click keeps multiple samples highlighted while a plain click retains the simple single-point pin. For XY scans, **Fluorescent spots > Detect and fit** removes the slowly varying background, detects bright or dark features, and fits rotated elliptical Gaussians. It reports each center, half-maximum radii, angle, contrast, SNR, and fit quality, and draws a toggleable fitted-shape overlay. Select rows in the spot table to **Remove selected** or **Keep selected only**; **Restore removed** and Ctrl+Z both undo curation. **Compare detector spots** pairs one spot from each detector, reports center/radius/orientation/amplitude differences and local profile correlation, and highlights the pair on both maps. **Copy spot table** places a compact X,Y-ordered table on the clipboard for PowerPoint, while saved CSV output remains complete. **Auto fluorescence** analyzes raw signal counts even when the displayed contrast is flat; **Display analyzed source** makes that fitted map visible. Detector scope can apply the operation independently to both channels.

For everyday GUI navigation, **Ctrl+K** focuses the file search. Search words can match filenames, folders, experiment types, dimensions, or detectors; **Enter** opens the first match and **Escape** clears the search. **Ctrl+Shift+O** adds files, including from an empty workspace. **Ctrl+Home** or the plot toolbar's Home button restores the full plot view. Fit-range bounds are available in the Model inspector.

## Time traces

Open a time-trace MAT file normally and leave the profile on **Auto**. SmartFitter detects **Time trace** independently of the fixed source sequence (ODMR, Rabi, DD, or fluorescence). The horizontal axis and ROI use **seconds**, with zero at the first measurement midpoint. Measured timestamps take precedence over cadence, bounds, and exported analysis axes; missing timestamps produce a load error instead of an invented regular axis. Spatial scans that retain old time-trace metadata are still recognized from their current scan declaration.

Reference-only fluorescence automatically selects **Reference** and offers only the recorded fluorescence observable. Paired traces retain contrast, signal, reference, and difference. Both detectors, overlays between time traces, exclusions, binning, smoothing, transformations, sessions, and figure exports remain available. Recorded counts or counts/s are preserved without conversion by the point cadence. Successful samples that required acquisition retries remain in the trace.

Use **Events** to show saved optimization/pause intervals. File details include the source sequence, fixed parameter/value, timing mode, cadence, run state, and numeric timing diagnostics. Inspecting a sample also shows scheduled start, lateness, and retries where saved. Scheduled starts are relative to acquisition trace start; elapsed timestamps and event intervals are relative to the first measurement midpoint.

The optional **TimeTraceDecay** model fits `background + amplitude * exp(-elapsed_s / tau_s)`. Its decay time is in seconds and describes drift/bleaching, rather than a spin-relaxation lifetime. FFT controls are unavailable for measured time traces because the acquisition can sample irregularly. **Save Results** with CSV selected writes a `<name>__time_trace.csv` table containing both detector streams, original point indices, elapsed seconds, recorded observables, timing diagnostics, and units. JSON retains the timing/event metadata; unavailable fluorescence signal/contrast fields stay empty in the time-trace CSV.

## Overlaid 2D comparisons

Load a 2D file and click **2D overlay** above the plot. Click it again to restore the previous map layout, sources, and zoom. A dual-detector file uses its detector pair; otherwise add another compatible 2D file. The button opens the **Map** inspector, where **Compare with** selects a detector from another loaded file. The picker includes maps of the same physical plane (XY, XZ, YZ, or matching custom axes), including opposite stored axis orders.

Choose **RGB overlay** for one blended map, **Maps + overlay** for both source maps plus the blend, **Separate maps** for the comparison layout, or **Single map** for the active detector alone. Turning **Detectors** off or clicking **Primary only** also returns to a single map. Pink/teal, green/magenta, and red/cyan palettes are available. **Independent** normalization emphasizes each map's shape; **Shared** uses one recorded-value range and requires matching observables and units. The overlay subtracts the selected background percentile (10% by default), then scales to the recorded maximum. The legend identifies the colors and sources, and the plot labels its normalization.

Maps retain their measured coordinates, axis swaps, and missing cells. Different grids are combined by splitting native pixel rectangles at their common boundaries and copying each recorded cell's value; there is no intensity interpolation or extrapolation into unmeasured coverage. Comparisons with different grids use physical pixel scaling. Same-grid comparisons also retain the existing square-pixel option. Source plots keep their quantitative colorbars.

Sessions save the comparison source, view, palette, background percentile, and ranges. Normal figure export and **Copy figure** include the displayed comparison; JSON records both source paths/detectors and the normalization. Selecting **Origin bundle** also writes `<name>__map_comparison.csv` with cell boundaries, both recorded values, RGB values, and coverage.

## Batch analysis

Process every `.mat` file in a folder:

```bash
python -m nvfit.pro_batch --input-dir "path/to/data" --output-dir "path/to/results"
```

Process one file:

```bash
python -m nvfit.pro_batch --single-file "path/to/data.mat" --output-dir "path/to/results"
```

Run `python -m nvfit.pro_batch --help` for all options. Batch results include fit figures, JSON records, CSV summaries, validation reports, and aggregate reports.

Batch processes both available detector streams by default. Use `--detector detector1` or `--detector detector2` to restrict it. Dual-file outputs use `__detector1` and `__detector2` suffixes; single-detector output names remain unchanged.

## Portable build (no Python needed)

`powershell -File tools\build_portable.ps1` builds `dist\SmartFitter\SmartFitter.exe` (PyInstaller onedir, no PySide6) and `dist\SmartFitter-portable-<version>.zip`. Lab-mates unzip and run `SmartFitter.exe` (same options as `python -m smartfitter`; settings stay in `%APPDATA%\SmartFitter`). It uses the WebView2 runtime that ships with Windows 11 and falls back to an Edge app window. Update notices are informational only: git checkouts compare with `origin/main` (update with `git pull --ff-only origin main`), portable builds compare with the latest GitHub release.

## Tests

With the virtual environment active:

```bash
python -m pytest -q
```

Some tests use optional lab-local datasets and are skipped automatically when those files are unavailable. The checked-in `TestData` fixtures cover the portable regression suite.

## Repository layout

- `nvfit/` — GUI, data loaders, preprocessing, models, fitting, diagnostics, export, and batch analysis.
- `tests/` — automated unit and offscreen GUI tests.
- `TestData/` — compact MATLAB fixtures used by the tests.
- `scripts/` — specialized research analysis and presentation scripts.
- `USER_GUIDE.txt` — detailed end-user instructions.
- `nvfit/APP_REFERENCE.md` — implementation and application behavior reference.

## Data and scientific use

The included fixtures are representative experimental data intended for testing and examples. Fit suitability, uncertainty, and physical interpretation remain the responsibility of the user. Review outputs and diagnostics before using results in publications or experimental decisions.

## Contributing

Contributions are welcome. Please:

1. Create a branch for the change.
2. Add focused tests for new behavior or bug fixes.
3. Run `python -m pytest -q`.
4. Open a pull request describing the motivation, implementation, and validation.

Avoid committing private experimental data, credentials, local environments, generated reports, or machine-specific configuration.

## License

This project is available under the [MIT License](LICENSE).
