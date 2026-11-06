# SmartFitter: NV-Center Lab Data Analysis

SmartFitter is a Python application for loading, visualizing, fitting, and exporting data from nitrogen-vacancy (NV) center experiments saved as MATLAB `.mat` files. It provides an interactive desktop GUI for individual measurements and a command-line batch pipeline for folders of experiments.

## Install

### Windows: download and run (no Python needed)

1. Go to the [latest release](https://github.com/Divc09/Lab-Data-Analysis/releases/latest) and download `SmartFitter-portable-<version>.zip`.
2. Right-click the zip, choose **Extract All…**, and pick a folder you keep (for example `Documents\SmartFitter`). Do not run it from inside the zip.
3. Open the extracted folder and double-click **SmartFitter.exe**.

Notes:

- The first time, Windows SmartScreen may say "Windows protected your PC" because the app is not code-signed. Click **More info → Run anyway**.
- Optional: pin `SmartFitter.exe` to the taskbar or Start menu, or right-click a `.mat` file → **Open with** → choose `SmartFitter.exe`.
- To update, download the new zip and replace the old folder. Your settings are kept in `%APPDATA%\SmartFitter`. The app shows a notice when a newer release exists.
- Needs Windows 10 or 11. It uses the Microsoft WebView2 runtime, which is already installed on Windows 11 and most up-to-date Windows 10 PCs.

### macOS: app download (Apple Silicon, no Python needed)

1. From the [latest release](https://github.com/Divc09/Lab-Data-Analysis/releases/latest), download `SmartFitter-<version>-mac-apple-silicon.dmg` (for Macs with M1 or newer chips).
2. Open the `.dmg` and drag **SmartFitter** into **Applications**.
3. Double-click SmartFitter in Applications. The first time, macOS says it "cannot verify" the app because it is not signed by an Apple-registered developer. Click **Done**, open **System Settings → Privacy & Security**, scroll down, and click **Open Anyway** next to SmartFitter. You only do this once.

### macOS: double-click launcher (any Mac, no Python needed)

Use this if the app above does not open, or on an older Intel Mac. It installs everything it needs (a private copy of Python, inside `~/Library/Application Support/SmartFitter`) and updates itself to each new release.

1. From the [latest release](https://github.com/Divc09/Lab-Data-Analysis/releases/latest), download `SmartFitter-mac-launcher.zip` and double-click it to unzip.
2. Double-click **SmartFitter.command**. The first time, allow it the same way: **System Settings → Privacy & Security → Open Anyway**.
3. A Terminal window shows progress. The first start takes a minute or two and needs internet; later starts are quick. Keep the Terminal window open while you use SmartFitter (you can minimize it).

To uninstall, delete `~/Library/Application Support/SmartFitter` (and the app or the `.command` file).

### Linux, or running from source on a Mac

Run SmartFitter from Python (3.12 or newer):

```bash
git clone https://github.com/Divc09/Lab-Data-Analysis.git
cd Lab-Data-Analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m smartfitter
```

Next time, just `cd Lab-Data-Analysis`, `source .venv/bin/activate`, and `python -m smartfitter`. Windows is the main tested platform; the Mac builds are checked automatically when they are built, and Linux is not regularly tested. On Linux, if no desktop window can open, run `python -m smartfitter --browser` to use your web browser instead.

To update: `git pull`, then `python -m pip install -r requirements.txt`.

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
- Fit ± values: fits use a robust soft-L1 loss (it keeps occasional glitches from dragging the result), and the ± is the matching robust covariance (Huber's H1 sandwich, as in statsmodels' RLM), computed from the plain model Jacobian. For an ordinary (linear-loss) fit it is exactly the usual least-squares error. The previous formula mixed SciPy's loss-rescaled Jacobian with the unweighted residual variance and overstated errors: in simulations of 26-point T1 scans its 1σ intervals covered the true T1 73-89 % of the time; the new ones cover 67-70 % (ideal 68 %). On 33 October 2026 T1 files the new ±T1 is a median 0.84× the iteration-jackknife error (previously 1.21×); the jackknife also includes iteration-to-iteration drift that a single averaged trace cannot show.
- Auto fits 1D line scans with one Gaussian peak on a constant background, in Analyze, Compare and Batch. Choose GaussianPeaks with more peaks, a custom model, or `--model-override` for anything else.

Iteration sensitivity (Analyze → Fit, or Process):

- **Check iteration sensitivity…** fits the current full selection and refits its mean with each iteration omitted in turn, then automatically deselects flagged iterations and refits the main plot. Select all first to use every available complete iteration. This supports T1 and other 1D models with iteration data.
- The flag rule is the absolute percentage change in the selected parameter (T1/T2/timescale by default), with an adjustable threshold of 10%. Failed or noncomparable omission fits are not automatically excluded. If every iteration is flagged, the selection is preserved. The rule identifies sensitivity, not proven statistical outliers.
- Checked **Excluded** rows immediately change the main selection and refit. **Exclude listed & refit** applies a typed range. **Test combined omission** adds a comparison row for the current excluded group. Restore any iteration in **Process → Iterations used for mean + fit**, or use **Restore baseline iterations** in the popup. Undo restores the previous selection and fit.
- The popup provides **Error bars** and **Rescale plot** controls, a scrollable table, compact source labels with full provenance on hover, and an automatically recorded exclusion reason. The Fit/Process panel retains its scroll position during updates.
- Diagnostics and exclusion reasons persist in sessions and JSON exports; **Export sensitivity CSV** saves the parameter comparison and source iteration identities. Changed settings/source files mark old diagnostics stale. These correlated refits do not provide independent uncertainty estimates or justify exclusion on their own.

Plan the measurement (Analyze → Fit → Result → **Plan the measurement**; `nvfit/planning.py`):

- **When is it done?** refits the mean of the first 2, 3, … N selected iterations and projects how many iterations reach a target error, assuming errors fall as 1/√N. Lifetimes, widths, amplitudes and frequencies take a % target; centres and offsets (x0, y0, phase, …) take an absolute target in their own unit. The measured exponent of error vs N is shown; a ⚠ means the error is not shrinking with averaging (drift, model mismatch or a poorly constrained parameter), so more iterations may not help. Time left uses the time per iteration you enter, or an estimate from the start stamp in the file name to the last save (includes overhead). **Rerun when the file updates** repeats the check after each watched checkpoint refit.
- **Where to measure next** ranks scan positions by how much one more point (same averaging as the existing ones) would reduce the chosen parameter's error, using the linearized fit (greedy, repeats allowed), and compares it with adding the same number of points spread like the current scan. **Copy list** copies the positions in the file's native x unit. For decays the search reaches up to 5× the fitted lifetime (at most 10× the scan) unless you set **Up to**.
- **Magnetic sensitivity** (ODMR fits): δB per point = residual noise ÷ (γe × steepest fitted slope), γe = 28.024 GHz/T, and η = δB × √(time per point) in T/√Hz, next to the CW shot-noise limit 4/(3√3) · FWHM / (γe C √R) (Dréau et al., PRB 84, 195204, 2011) with R the mean reference count rate. Time per point defaults to the wall-clock estimate, so the measured η is conservative.
- All three are planning estimates around the current fit (model assumed right, noise like the current residuals), not validated predictions.

## Highlights

- Loads legacy MATLAB `savedData` files and newer single- or dual-detector snapshot exports.
- Supports Rabi, Ramsey, spin echo, dynamic decoupling, T1, ODMR, DEER, line scans, and 2D stage scans.
- Loads measured time traces from the acquisition code, including reference-only fluorescence and dual detectors.
- Compares 2D maps with pink/teal RGB overlays, across detectors or loaded files.
- Provides experiment-aware fitting, diagnostics, exclusions, transformations, overlays, iteration views, and derived NV metrics.
- Includes a unified Plot Editor for legends, annotations, titles, axes, ticks, secondary axes, and report presentation.
- Exports publication-ready PNG, PDF, and SVG figures plus JSON, CSV, and report outputs.
- Includes representative test data and an offscreen GUI regression suite.

## Install from source (Windows, for development)

Most Windows users should use the download above. To run from source instead you need Python 3.12 or newer. Clone the repository and enter its directory:

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

## Building the downloads (maintainers)

**Releasing:** raise `APP_VERSION` in `smartfitter/core.py` (e.g. to `2.1`), commit and push, then push a matching tag (`git tag v2.1` then `git push origin v2.1`). The **Build apps** GitHub Actions workflow (`.github/workflows/build.yml`) then builds the Windows zip, the Mac `.dmg` and `SmartFitter-mac-launcher.zip`, smoke-tests them, and attaches them to the `v2.1` release (creating it if needed). You can also run the workflow by hand from the Actions tab to get the files without releasing. Mac builds come only from this workflow (they must be built on a Mac); the Windows zip can also be built locally:

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
