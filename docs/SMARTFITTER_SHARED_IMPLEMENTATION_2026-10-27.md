# Shared analysis implementation log

Companion to `SMARTFITTER_SHARED_ANALYSIS_DESIGN_2026-10-27.md`. Started from the dirty checkout after Batch 2 and existing iteration-sensitivity changes. No commit/push requested.

## Completed checkpoint: prepared fitting

- Reproduced F08, F09, F10 and F14 before implementation; numerical evidence is recorded in the design note.
- Added immutable `nvfit.prepared_fit.PreparedFit`: chosen target, aligned SEM, config, controls and full-target T1 normalization are reused by ordinary/robust fits. Robust candidates are evaluated on the full target with its weights; custom evaluators are retained.
- Ramsey accepts explicit model choice, multistart and parameter controls, with observable-scaled amplitude/background bounds.
- FitResult retains optimizer covariance, free mask and uncertainty metadata. Confidence bands use this covariance, including fixed parameters and ODMR rescaling. Fit-result serialization retains uncertainty.
- Web dispatch, batch robust and Qt fitting consume the shared path. Existing sensitivity-related provenance/defaults were retained.
- Focused check: 29 passed (`test_shared_analysis.py`, `test_audit_batch2.py`, `test_audit_fixes.py`). This is a checkpoint, not a claim that the remaining result/batch/server work is complete.

## Completed implementation checkpoints: identity and batch/server

- Added matching Python/JavaScript canonical descriptors and signatures, with source-content revisions and dependent sources. Compare analysis items fix detector identity and support captured processing variants; stale fits are excluded from curves/trends, retained as historical records, and refittable.
- Versioned session serialization includes bootstrap, peaks, DEER/scientific tools, Stack state, results and signatures. Relocation remaps references and parent-fit signatures. Unknown legacy results are unverified.
- Baseline/global operations convert compatible physical units and reject incompatible quantities. Compare uses each trend result's unit. Map display/export share a captured request/result including raw map source and view state.
- Atomic batch manifests capture source hashes, options, auxiliary mappings/custom model, code hashes, row IDs and archived outputs. Progress increments after row commit. Replay validates source hashes and reuses recorded options; cancellation checkpoints completed work and does not adopt older CSV results.
- Batch-to-Compare imports stored curves/parameters through `batch_import`; it does not mutate a dirty Analyze file or refit through a different route.
- Per-launch server tokens are required on API calls. Host/Origin checks, browser fragment bootstrap and authenticated second-launch handoff are implemented.
- Focused current checks: 9 shared-analysis Python tests, 4 shared-analysis JavaScript tests, and 5 batch/server tests pass. Related legacy suites require fixture/signature updates and normalization expectations; final validation is still pending.

## Final verification and behavior

All requested groups are implemented: AN01/F08/F09/F10/F14; F06/F07/F18/F25/F15; F21–F23 and F30. The design note above preceded the implementation. Reproductions of the first four are recorded there.

Additional integration repairs:

- ODMR controls are converted into the optimizer's scaled y space and returned in native units; its covariance is rescaled with the same parameter transformation. Locked parameters report zero analytic error and are labelled locked in the inspector. Qt keeps Ramsey candidate and robust-subset summaries.
- Captured Compare variants use the owning source's current revision for freshness, retain exact processed targets through sessions, and label detector/observable/processing differences. Stale curves are excluded from comparison copies/tables as well as plots. Compare actions sit below readable identities; the overlay legend uses compact labels.
- Aggregate Stack/global results use the shared envelope. Session relocation remaps aggregate source revisions and parent-fit signatures. Stack source/option changes produce historical-result warnings; exports include their signatures. Mixed Stack coordinate units are rejected explicitly.
- Export validates source revisions before writing. Map export consumes the exact captured overlay used by the screen, with source, palette, scale, swap and displayed view state recorded.
- Batch has Resume unfinished rows in addition to recorded-row reruns. Incomplete discovery is marked explicitly. Counts come from committed rows, including failure finalization; original run records and outputs remain available. Manifests also record numerical-library versions.

Final automated checks:

- **413 Python tests passed, 9 skipped, 3 warnings** (`.venv/Scripts/python.exe -B -m pytest tests -q -p no:cacheprovider`, isolated app/settings/Matplotlib directories, Qt offscreen).
- **100 JavaScript tests passed** (`node --test tests/*.mjs`).
- `git diff --check` passed. The three Python warnings are an existing deprecated Qt activation helper and two NumPy constant-signal correlation warnings.
- Focused post-integration boundary suite: 69 passed. Synthetic tests cover selected-target T1 normalization, robust custom/SEM/locks, explicit Ramsey controls and starts, covariance-derived bands, scaled ODMR controls, compatible/incompatible units, stale/dual/captured Compare contexts, relocated tool/Stack sessions, map cells, batch replay/resume/cancellation/import and API token checks.

Real-browser checks against an isolated local server:

1. A real Rabi fit rendered with a current signature. Changing binning marked its Compare result STALE, removed its fit/trend, and enabled refitting.
2. A synthetic dual-detector copy of a real T1 file fit both detectors independently. Normalized robust fitting retained a T1 lock at 40000 ns. Both Compare rows and trend values remained distinct.
3. Ten bootstrap resamples and their parent-fit relationship survived an actual save/load session. A built map Stack also survived restoration and passed server source validation; changing the selected observable showed a historical-result warning.
4. A real two-row batch completed. The UI imported both archived detector fits, retained their source record, and preserved the dirty Analyze detector/mode/processing/model settings exactly. Restored captured items remained current.
5. A signal-source map overlay with swapped axes was exported through Save analysis. The displayed PNG used the captured overlay; CSV values matched all **451** captured primary cells. JSON retained the shown source and swap state.

Evidence is in `artifacts/shared-analysis/` (ignored generated QA files): `compare-final.png`, `compare-stale.png`, `map-signal-overlay.png`, `stack-historical-after-option-change.png`, session round trips, batch manifests/archived results, map exports, and final test logs.

## Use and compatibility

- Existing results without a verifiable envelope remain historical/unverified until refitted. They are preserved for inspection.
- Batch reruns/resume use a saved manifest and its options. Changed source contents require a new batch. Legacy CSV-only summaries can be read, but cannot supply a reproducible stored-fit import.
- Compatible baseline/global/Compare units are converted explicitly. Stack operations require consistent recorded coordinate units; files that omit physical units remain labelled as recorded stage units.
- Confidence bands show a one-sigma model-mean estimate from the fit's retained covariance. They remain an approximate optimizer linearization. A missing/unidentifiable covariance yields an unavailable band.
- Reopen SmartFitter through its launcher after restarting the server. API tokens rotate per launch and bootstrap from the launch URL fragment into browser session storage; they are excluded from analysis/session exports.
- Local ignored legacy tests were adapted for the intentional normalization/identity/shortcut behavior and VM imports. They were not promoted wholesale into the tracked repository. Portable regressions added for this task are explicitly unignored.
- One initial test invocation used the Windows Python launcher after redirecting LOCALAPPDATA, causing it to bootstrap Python 3.15 under the temporary QA directory and fail for missing pytest. All reported checks were rerun using the existing repository `.venv`.

These checks validate software behavior on fixtures; they are not instrument/bench validation. Existing Claude/user changes and generated artifacts were preserved. No commit or push was made.
