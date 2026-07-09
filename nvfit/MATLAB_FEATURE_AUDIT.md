# SmartFitter MATLAB Feature Audit

This checklist consolidates behavior from:
- `SmartFitter_Interactive.m`
- `SmartFitter_Interactive2.m`
- `SmartFitter_Pro.m`

## 1) Data and file handling

- [x] `.mat` file load via legacy `savedData` root struct
- [x] `.mat` file load via new `data` + `scanInfo` root structs
- [x] Scan-axis extraction from `savedData.loadedScan` (`starts`, `ends`, `nsteps`)
- [x] Scan-axis extraction from new `scanInfo.bounds` + `stepSize` + `nSteps`
- [x] Fallback signal/reference fields:
  - [x] prefer `signal` and `reference`
  - [x] fallback `signalVector` and `referenceVector`
- [x] New-format iteration reduction from `data.values`
- [x] NaN/Inf filtering after contrast computation
- [x] Experiment type auto-detection by filename (`odmr`, `rabi`, `ramsey`, `echo`)
- [x] Dynamic-decoupling detection from notes/filename (`XY8`, `CPMG`, DD)

## 2) Data preprocessing modes

- [x] Contrast mode: `(ref - sig)/ref`
- [x] Raw signal mode
- [x] Difference mode `(sig-ref)` or `(ref-sig)` conventions
- [x] Binning / averaging
- [x] ROI bounds (`minX`, `maxX`)
- [x] Optional smoothing (Savitzky-Golay in V2/Pro)

## 3) GUI workflow features

- [x] Plot + controls panel layout
- [x] Model dropdown
- [x] Bounds widgets + full reset
- [x] Guess parameters from current ROI
- [x] Lock/unlock parameter support
- [x] FFT helper for frequency estimation
- [x] Fit execution button
- [x] Save high-resolution figure
- [x] Fit status with R² shown in title/annotation

## 4) Supported model families in MATLAB versions

- [x] Rabi (damped cosine + optional drift)
- [x] Ramsey (simple stretched-exponential cosine)
- [x] Ramsey hyperfine variant
- [x] SpinEcho stretched exponential
- [x] DynamicDecoupling stretched exponential on total evolution time
- [x] ODMR Lorentzian-like resonance
- [x] Generic linescan Gaussian

## 5) Current MATLAB pain points to fix in Python rewrite

- [x] Duplicated model and extraction logic across files
- [x] Inconsistent unit scaling (`ns`, `us`, `GHz/MHz`) between branches
- [x] Weak/fragile defaults and bounds for Ramsey in noisy traces
- [x] Little model-selection rigor (single-model trial then manual tweaks)
- [x] Limited diagnostics beyond R²
- [x] GUI logic tightly coupled to fitting internals (hard to test)

## 6) Python rewrite acceptance requirements

- [x] One canonical data contract from `.mat` to fitting engine across both MATLAB schemas
- [x] Shared model library for GUI and batch pipelines
- [x] Physics-first constrained models and robust multistart fitting
- [x] Validation metrics beyond R² (AIC/BIC, residual checks, bound-hit flags)
- [x] Reproducible exports (JSON/CSV + figures + diagnostics)

