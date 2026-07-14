# SmartFitter: NV-Center Lab Data Analysis

SmartFitter is a Python application for loading, visualizing, fitting, and exporting data from nitrogen-vacancy (NV) center experiments saved as MATLAB `.mat` files. It provides an interactive desktop GUI for individual measurements and a command-line batch pipeline for folders of experiments.

## Highlights

- Loads legacy MATLAB `savedData` files and newer `data` + `scanInfo` snapshots.
- Supports Rabi, Ramsey, spin echo, dynamic decoupling, T1, ODMR, DEER, line scans, and 2D stage scans.
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

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start SmartFitter:

```bash
python -m nvfit.gui_app
```

On Windows, after creating `.venv`, you can also double-click `Launch SmartFitter.vbs`.

## Basic GUI workflow

1. Open or drag in a supported `.mat` file.
2. Leave the analysis profile on **Auto**, or choose an experiment-specific profile.
3. Select the observable: contrast, signal, reference, or difference.
4. Adjust the region of interest, binning, smoothing, exclusions, or transforms as needed.
5. Run the fit for supported 1D experiments.
6. Double-click a legend, annotation, title, axis label, or tick label to customize the figure.
7. Export the figure, fit data, metadata, or report.

Spatial scans are plot-only. Their map controls provide color scaling, view bounds, pan and zoom, peak/dip navigation, and cursor or best-point linecuts.

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
