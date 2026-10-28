# SmartFitter audit fixes: batch 2

Implemented on 2026-10-27 against `dd66a94`, following Claude's earlier fixes.
Scope: F19, F16, F24, F28, F33, UX01 and UX04. The original audit remains a historical report.

## Changes

| ID | Result | Main implementation |
| --- | --- | --- |
| F19 | Relocated sessions remap the captured overlay paths in Figure panels. Saved baseline references already passed through recursive path remapping; a regression now covers both together. Restoring Figure configuration clones the saved document. | `state.js` |
| F16 | FFT checks the current processed grid and the displayed series, including gaps, duplicates, finite values, length and spacing. An invalid saved FFT toggle produces no spectrum or empty FFT panel; the menu is disabled with a reason. Residuals can still appear when valid. | `plot1d.js`, `spectrum.js`, `analyze.js` |
| F24 | Batch distinguishes `PLOT_ONLY`, `UNCONVERGED` and `FALLBACK` from the usual fitted `PASS`/`WARN`/`FAIL`. JSON and both CSV reports carry the same status/reason and retain fallback error details. Plot-only rows have no optimizer-success claim. | `nvfit/pro_batch.py`, `batchstatus.js`, `batch.js` |
| F28 | Data-sized minimum/maximum argument spreads were replaced by finite-value iteration in maps, Compare normalization/trends, Analyze ranges/linecuts, iteration trends, Stacks and AutoFind bounds. Entirely missing automatic map limits fall back to `[0, 1]`. | `numeric.js` and its callers |
| F33 | Each file's loads have a monotonic revision and immutable processing snapshot. Only the newest request that still owns the open file and matches its settings may apply detector payloads or metadata. Closed/replaced files and stale failures are ignored. | `state.js` |
| UX01 | Fit actions use a responsive grid with full-width rows in a narrow inspector. Shortcut badges can wrap. Narrow topbars place workspace navigation on its own row so the Save action stays reachable. | `app.css` |
| UX04 | Toolbar Save, Ctrl+S, Ctrl+Shift+C and command-palette Save/Copy share a workspace routing table. Figure export offers a format menu. Empty workspaces disable their toolbar export. | `workspaceactions.js`, `app.js` and workspace exports |

## Batch outcome contract

- `PASS` / `WARN` / `FAIL`: the quality gate for a converged fit using the requested fitting workflow, or `FAIL` for a load/preprocessing/fit error.
- `PLOT_ONLY`: a plot was generated without a fitted model. The CSV `success` field is blank.
- `UNCONVERGED`: the optimizer returned a result but reported failure. This takes precedence when a fallback result is also unconverged; its reason and `model_note` retain the fallback evidence.
- `FALLBACK`: a converged substitute model was used after the requested workflow raised an error. The reason includes the original error and the ordinary quality assessment.
- `EXCLUDED`: the existing exclusion status remains available.

Batch filters and status colors expose the new outcomes. The Qt summary viewer recognizes them too. Send fitted rows to Compare accepts only ordinary PASS/WARN fitted rows and rejects unsuccessful/no-model rows in older summaries. Retry includes failures, unconverged results and fallback results. Aggregate R² statistics include ordinary PASS/WARN fits only.

Existing summary files are not rewritten or automatically reclassified. External consumers that enumerate status values must accept the new values and the additive `model_note` column.

## Workspace actions

| Workspace | Save / Ctrl+S | Copy / Ctrl+Shift+C |
| --- | --- | --- |
| Analyze | Save analysis using its existing export options | Analyze figure |
| Figure | Choose SVG, PNG or PDF for the assembled figure | Assembled figure as PNG |
| Compare | Current trend table as CSV | Compare overlay figure |
| Day | Displayed summary rows as CSV | Displayed summary table |
| Stacks | Stack results and provenance as CSV | Stack results table |
| Batch | All loaded summary rows as CSV | All loaded summary rows |
| AutoFind | Selected runs as a figure set; choose output directory when needed | Unavailable; no implicit Analyze copy |
| Start | Save session | Unavailable |

Ctrl+Shift+S continues to save a session in every workspace. Existing workspace-specific export buttons remain available.

## FFT definition

The browser uses a radix-2 FFT with mean subtraction and zero padding to the next power of two. It keeps the actual input sample count for magnitude normalization and actual native x spacing for frequency scaling. Positive frequencies through Nyquist are shown; magnitude is `abs(FFT) / n`, without doubling. Zero padding changes the frequency grid, not the physical resolution of the acquisition. Frequency units retain the prior ns/us-to-MHz and seconds-to-Hz convention. Irregular inputs are rejected; no implicit interpolation is applied.

## Verification

Portable regressions are in `tests/test_audit_batch2.py` and `tests/test_audit_batch2.mjs`, with narrow `.gitignore` exceptions so they are visible to Git. They cover reversed multi-detector responses, request snapshots, closed/replaced files, stale failures, relocated baselines and Figure overlays, 300,000-value maps, Compare normalization, invalid FFT grids, normalized-fit masking of processing gaps, direct-DFT agreement, 512/4096/5000-sample tones, batch outcome files and workspace routing. The tracked time-trace test now expects PLOT_ONLY.

Browser checks used an isolated server and temporary application preferences. Verified:

- Complete fit-button labels and shortcut badges at 1000×640, 1280×720, 1366×768 and 1600×900 CSS viewports. Reduced 1024×576, 800×450 and 640×360 viewports exercised layout behavior under less available CSS space; the native Windows DPI setting was not changed.
- Figure's top action offered SVG/PNG/PDF and wrote an assembled SVG plus session sidecar. Compare Ctrl+S wrote a trend CSV. Batch's top action wrote all seven synthetic outcome rows to CSV. Test picker destinations bypassed only the native file chooser; application rendering and filesystem writes used the normal APIs.
- A relocated two-file session reopened through the real API with both Figure overlays and baseline paths remapped. Subtracting identical fixture copies produced zero as expected.
- A real Rabi fixture with one middle point excluded kept source `irregular:false`, but disabled the FFT menu with an explicit irregular-spacing reason and produced zero FFT traces.
- Every workspace displayed its own Save label; empty Day/Stacks/AutoFind/Batch exports were disabled. A synthetic batch with no eligible normal fits disabled Send fitted rows to Compare.

Screenshots and exported QA files are under `artifacts/audit-batch2/` (ignored generated evidence). The native clipboard and Windows file-picker UI were not exercised end to end. These are software checks, not instrument or bench validation.

Local ignored legacy tests were updated for the intentional new status/shortcut behavior and the numeric-helper import in a VM harness. This does not promote the rest of the local ignored suite into the tracked repository.

Final clean suite result: **378 Python tests passed, 9 skipped; all 81 JavaScript tests passed.** Python reported three warnings: a deprecated Qt activation helper in an existing GUI test and two NumPy constant-signal correlation warnings. `git diff --check` passed.

Commands used: `python -B -m pytest -q -p no:cacheprovider` with isolated APPDATA/LOCALAPPDATA/MPLCONFIGDIR and `QT_QPA_PLATFORM=offscreen`; `node --test tests/*.mjs`.
