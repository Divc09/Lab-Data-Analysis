# SmartFitter codebase audit and improvement roadmap

**Audit date:** October 27, 2026  
**Repository:** `D:\BacklundLabResearch\Code\Lab_Data_Analysis`  
**Baseline:** branch `main`, commit `45c76712c1b57d8e9a807b7f2c018554211e1bd6`, including the existing local edits.  
**Deliverable:** findings and recommendations only. No application, test, configuration, or existing documentation files were edited for this audit.

## 1. Main conclusions

SmartFitter already has substantial capabilities: detector-aware MAT loading, experiment-specific fitting, iteration selection and pooling, editable plots, comparisons, map overlays, spot fitting, time-trace tools, AutoFind inspection, stacks, batch processing, and figure composition. The largest opportunity is to make those capabilities behave consistently across workspaces and preserve exactly which data and settings produced each result.

The first implementation round should address silent numerical/provenance errors and broken controls. Cosmetic redesign should follow those repairs. In particular:

1. Fix parameter locks, ignored seeds, and the custom-expression variable mismatch.
2. Prevent detector substitution and preserve stage coordinates when samples are missing.
3. Make fit validity depend on the source, detector, observable, processing, and model settings everywhere, including Compare, Day, sessions, and exports.
4. Make the selected fit target and its uncertainty model consistent across ordinary, robust, bootstrap, global, and batch fits.
5. Repair session completeness, batch provenance, output-name collisions, and map export fidelity.
6. Reorganize the workbench around a stable plot, contextual controls, persistent result identity, and recoverable sessions.

### Evidence labels

- **Reproduced:** demonstrated by a current-run numerical, state, API, or visible browser check. A synthetic reproduction establishes a software defect, not its frequency in historical experimental data.
- **Source-confirmed:** the current implementation directly establishes the behavior; the complete user workflow was not reproduced during this audit.
- **Candidate:** a plausible risk requiring a focused experiment before calling it a bug.
- **Proposal:** an intentional feature or design change.

**Priority:** P1 = address first because results, saved work, or core workflows are affected; P2 = important usability, reliability, or analysis improvement; P3 = optional expansion or cleanup. No P0 emergency is asserted. **Effort:** S = localized change, M = several components, L = data-model/workflow change; these are relative estimates, not delivery promises.

### Review coverage and limits

The repository inventory contains **74 tracked source/build/UI files and 35,829 lines**. Review covered the v2 shell and workspace modules, backend/API, fitting and uncertainty code, MAT readers, transforms, iterations, maps, spots, stacks, AutoFind, sessions, exports, launch/build/update paths, documentation, and local tests. The legacy Qt interface was reviewed structurally and through its current regression suite, with targeted inspection of shared workflows. This is a broad codebase audit; it does not imply every line of the 12,553-line legacy GUI or every possible interaction was manually exercised.

Current-run verification:

| Check | Result | Meaning |
|---|---|---|
| Full local Python suite | **355 passed, 9 skipped, 3 warnings**, 255.83 s | Existing regression coverage passes in this checkout. |
| Focused loader/map/time-trace/fit/uncertainty lane | **89 passed, 3 skipped**, 53.47 s | Additional focused run; overlaps the full suite and is not an independent test count. |
| Local JavaScript tests | **71 passed, 0 failed** | State, plotting, units, comparisons, maps, accessibility helpers, and exports have passing local regressions. |
| Numerical edge cases | Reproductions recorded below | Revealed bugs not covered by the existing passing tests. |
| Session-state probes | Reproductions recorded below | Confirmed missing scientific state, incomplete relocation, and cross-session undo. |
| Visible v2 browser | Real T1 files, Analyze and Compare | Confirmed stale Compare fits and inspected the default 1280×720 layout. Other dimensions are proposed acceptance checks, not claimed as tested. |
| API origin probe | Read-only request accepted with a foreign Origin and `text/plain` | Supports the local-API hardening finding; no hostile webpage or destructive operation was executed. |

Tests ran with bytecode/cache settings and application preferences isolated under the system temporary directory. The full-suite warnings were one Qt deprecation and two NumPy divide warnings for a deliberately constant T1 input. Skips mean those checks were not validated. No bench measurements, acquisition correctness, physical lifetime interpretation, fresh portable build, or macOS/Linux runtime validation is claimed.

The tested local environment was Windows with Python **3.14.5**, NumPy **2.4.2**, SciPy **1.16.3**, and pytest **9.0.2**. The principal commands were the repository interpreter running `-B -m pytest -q -p no:cacheprovider` with a temporary `--basetemp`, and `node --test tests/*.mjs`. These results describe this local environment, not every allowed dependency combination.

## 2. Fixes and correctness findings

### F01 — Parameter locking makes the shared fitter reject the fit

**P1 · Reproduced · M**  
**Source:** [fit_workflows.py:148](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/fit_workflows.py:148), [fit_engine.py:116](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/fit_engine.py:116).

`_apply_locks()` implements a lock by setting the lower and upper bound to the same value. The shared `least_squares` path requires a strictly positive interval. A synthetic TimeTraceDecay fit with `tau_s` locked to 2 failed with: `All 2 fit attempts failed: Each lower bound must be strictly less than each upper bound.` The SpinEcho inspector explicitly recommends locking `n`, so this affects a documented workflow.

**Fix:** optimize only free parameters, reconstruct the complete vector inside the model wrapper, and calculate degrees of freedom from the free count. Handle the all-locked case explicitly. **Acceptance:** locked T1, echo exponent, Gaussian center, custom-model parameter, and time-trace decay fits run; locked values remain exact and are labeled fixed rather than given estimated standard errors.

### F02 — Unlocked parameter values do not become starting seeds

**P1 · Reproduced · S–M**  
**Source:** [fit_workflows.py:157](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/fit_workflows.py:157), [analyze.js:973](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:973).

The shared lock/seed adapter reads `value` but assigns it to `seed[i]` only when locked. A requested unlocked seed of 42 left a synthetic seed at 1. Consequently, editing the Seeds & bounds value or using “Values from current fit” can appear to work while leaving the actual starting point unchanged.

**Fix:** apply finite supplied seed values independently of locks, validate against bounds, and include that seed as an unperturbed start. **Acceptance:** an instrumented optimizer sees the requested value; invalid seed/bound combinations produce an actionable field error.

### F03 — The default custom expression uses `x`, while the backend accepts `t`

**P1 · Reproduced · S**  
**Source:** [state.js:35](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/state.js:35), [analyze.js:126](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:126), [models.py:147](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/models.py:147).

The UI supplies and describes `y0 + A*exp(-x/tau)`. The backend expression validator permits `t` and rejects `x`: the default expression raised `Unsupported name in custom expression: x`; the equivalent `t` expression was accepted. Simply changing the UI to `t` is incomplete because the client parameter detector currently reserves `x`, but not `t`, and would treat `t` as a fit parameter. Regex parameter discovery can also misread scientific-notation fragments such as `1e3`.

**Fix:** define one expression contract, preferably accept both `x` and `t` as aliases, and derive parameter names from the validated AST in the backend. Validate output shape and finiteness before fitting. **Acceptance:** the untouched default expression fits; expressions using scientific notation produce only genuine parameters; load/save/import uses the same contract.

### F04 — Missing samples can shift a 1D stage scan onto the wrong coordinates

**P1 · Reproduced · M**  
**Source:** [io_mat.py:339](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/io_mat.py:339), [io_mat.py:410](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/io_mat.py:410).

`_finalize_trace()` filters and sorts the sample coordinates and values, then the scan1d branch replaces the coordinates with the first `L1` elements of the original scan vector. Input positions `[0,1,2,3]` with values `[10,NaN,30,40]` produced positions `[0,1,2]` and values `[10,30,40]`; the correct retained positions are `[0,2,3]`. The same reset deserves checks for descending axes and iteration-array alignment.

**Fix:** carry sample indices through filtering/sorting and never assign a fresh coordinate prefix after removing samples. **Acceptance:** missing first/middle/last points and reversed stage sweeps retain their exact measured coordinate/value pairs in plots, fits, inspection, and export.

### F05 — An explicitly unavailable detector is silently substituted

**P1 · Reproduced · M**  
**Source:** [core.py:164](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:164), [core.py:305](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:305), [state.js:82](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/state.js:82).

`Workspace.trace()` chooses the first loaded detector whenever the requested ID is absent from `ds.detectors`. A test request for detector2 returned detector1 data. This is particularly dangerous when `available_detectors` includes a detector that failed to load: the client requests both IDs and can store the substituted payload under the failed detector's key. Downstream fits, maps, and tracking may then appear to contain two independent channels.

**Fix:** select a default only when the request omits a detector. An explicit missing/failed ID must return its structured load error. Render partial detector success without fabricating the other channel. **Acceptance:** corrupt detector2 remains visibly unavailable while detector1 works; no detector2 curve/result is created from detector1.

### F06 — Compare retains stale fits after analysis changes

**P1 · Reproduced in the visible UI · L**  
**Source:** [compare.js:284](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/compare.js:284), [compare.js:341](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/compare.js:341), [analyze.js:47](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:47).

Compare reads its separate `results[path]` cache without verifying the current analysis. I fitted the two local T1 fixtures in Compare, switched `T1_20260324_054548.mat` from Contrast to Signal in Analyze, and returned to Compare. The trend still reported the earlier approximately `520 ± 210 µs`, R² 0.93 result; its old contrast fit was drawn near zero against the new normalized signal, without a stale warning. Processing, detector, source updates, fit settings, and asynchronous fits need the same dependency treatment.

**Fix:** attach an immutable analysis signature to every result and check it whenever displayed, exported, trended, or restored. Preserve old results as explicitly historical; provide Refit stale. **Acceptance:** observable/detector/processing/range/source changes mark the correct individual and global results stale, including undo, failed refit, session reopen, and watched checkpoint updates.

### F07 — Compare cannot represent both detectors of one file independently

**P1 · Source-confirmed limitation · L**  
**Source:** [state.js:14](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/state.js:14), [batch.js:84](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/batch.js:84).

Membership and results are keyed by path. Batch-to-Compare explicitly skips additional detector rows to avoid overwriting the first. A file also cannot hold two independently processed analysis variants in the Compare set.

**Fix:** introduce an analysis-item ID containing source identity, detector, observable, and processing variant; distinguish source files from analysis items in the UI. **Acceptance:** detector1 and detector2 of the same MAT file have independent curves, labels, results, trend points, exports, and session entries.

### F08 — Normalized T1 can silently fit the original series instead of the requested target

**P1 · Reproduced at the preparation helper · M**  
**Source:** [fit_workflows.py:269](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/fit_workflows.py:269), [core.py:482](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:482).

When lengths match, `_prepare_t1_series()` replaces its `y` argument with `trace.y` before normalization. With original values `[0,.2,.5,.9,1]` and a requested squared target `[0,.04,.25,.81,1]`, it normalized the original values. Same-length transforms and baseline subtraction can therefore be bypassed. Batch smoothing has the same exposure because it passes a smoothed `y` and the original trace. The smoothed v2 path has a special correction, but the general target contract remains inconsistent.

**Fix:** pass the selected fit target explicitly and normalize exactly that target. Keep any legacy preparation mode explicit and recorded. **Acceptance:** transforms, baseline subtraction, smoothing, ROI, and binning each produce the expected fitted target; no branch changes targets merely because array lengths match.

### F09 — Robust fitting bypasses the selected fit configuration

**P1 · Source-confirmed · M–L**  
**Source:** [core.py:556](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:556), [pro_batch.py:136](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:136).

The robust dispatch forwards the family, arrays, start count, and ODMR multipeak flag, but omits locks, custom specification, T1 preparation/shape, Rabi model/calibration settings, and SEM configuration. The robust helper then calls its own default workflows. It can therefore solve a different problem from the one shown in Fit settings. Normalized T1 subset scoring also reevaluates model-space curves against the original full `y`, which requires an explicit common-space conversion.

**Fix:** let robust fitting wrap the same prepared request and model callable used by ordinary fitting. Score all candidates in a documented common target space. **Acceptance:** ordinary versus robust differs only in subset strategy; requested locks, model, transform, units, weights, and normalization remain identical and appear in provenance.

### F10 — Ramsey fitting ignores several general Fit controls

**P1 · Source-confirmed · M**  
**Source:** [core.py:563](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:563), [ramsey.py:95](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/ramsey.py:95).

The Ramsey call accepts arrays and optional SEM, with internally fixed starts and bounds. General seed/bound/lock settings and the UI multistart count are not passed. Selecting a specific Ramsey model still fits all three candidates before choosing the requested one. Hard-coded amplitude limits also assume contrast-like scales, even though the application allows Signal and Reference.

**Fix:** accept the shared fit configuration, fit only the chosen model when explicit, and scale baseline/amplitude seeds and bounds to the selected observable. **Acceptance:** locked Ramsey parameters and multistart work; count-rate observables can be fitted without contrast-sized bounds; explicit-model latency reflects one model.

### F11 — Bootstrap counts unconverged fits as successful

**P1 · Reproduced · S–M**  
**Source:** [bootstrap.py:67](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/bootstrap.py:67), [ext_uncertainty.py:16](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/ext_uncertainty.py:16).

Draws are retained when parameter names match and parameters are finite; `res.success` is never checked. A stub returning `success=False` for every draw still reported `n_success=4/4` and zero spread. The endpoint also locks only the Ramsey auto-model choice, so other Auto families can change choices across bootstrap draws; name matching alone is not a sufficient model identity.

**Fix:** validate baseline and draw convergence, model identity, and finite predictions; record failure reasons and the accepted fraction. Resolve Auto once where the intended bootstrap conditions on a model. **Acceptance:** failed optimizers do not count as successes; a model-family change is reported or prevented; too few valid draws cannot produce a seemingly trustworthy interval.

### F12 — Underdetermined fits can receive finite covariance errors

**P1 · Reproduced · M**  
**Source:** [fit_engine.py:31](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/fit_engine.py:31), [core.py:517](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:517).

The compact SVD omits additional right-null directions when the Jacobian has more columns than rows. For `J=[[1,0,1],[0,1,1]]`, rank 2 with 3 parameters, the helper returned finite variances `[1.25,1.25,.25]`. The general API requires five points but does not require more points than free parameters for all model families; RamseyExtended has eleven parameters. Clamping degrees of freedom to one hides the insufficient-data condition.

**Fix:** check free-parameter count and numerical rank before uncertainty estimation; include the complete null space when necessary. **Acceptance:** underdetermined/saturated fits are rejected or explicitly marked unidentifiable, with unavailable errors and no PASS implication based on fit quality alone.

### F13 — Global fitting relaxes incompatible shared bounds

**P1 · Reproduced · S**  
**Source:** [global_fit.py:209](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/global_fit.py:209).

When per-dataset shared bounds have an empty intersection, the implementation substitutes their union. Synthetic shared `tau_s` bounds `[.5,1]` and `[2,3]` produced a successful fit at approximately 2.141, violating the first dataset's requested constraint.

**Fix:** reject the request and identify the conflicting parameter and datasets. **Acceptance:** incompatible bounds never widen silently; a shared value lies within every applicable interval.

### F14 — Confidence bands use a different uncertainty calculation from the fit

**P1 · Source-confirmed · M**  
**Source:** [core.py:574](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:574), [fit_engine.py:146](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/fit_engine.py:146).

`_conf_band()` recomputes an unweighted finite-difference Jacobian and residual covariance, independent of the optimizer's weights, robust-loss Jacobian, and fixed-parameter mask. Consequently the plotted band and reported parameter errors need not describe the same fitted objective. The calculation's “locked” handling only recognizes zero model sensitivity; a fixed but sensitive parameter needs explicit exclusion.

**Fix:** retain covariance, free-parameter mapping, target space, and weighting metadata in the fit result and derive bands from that result. Label the interval level and distinguish model-mean confidence from prediction scatter. **Acceptance:** a weighted linear reference case gives analytically consistent parameter errors and bands; locks contribute no uncertainty; unavailable covariance gives an explicit unavailable band.

### F15 — Cross-file operations need strict unit and observable compatibility

**P1 · Source-confirmed · M–L**  
**Source:** [core.py:363](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:363), [ext_globalfit.py:15](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/ext_globalfit.py:15), [compare.js:225](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/compare.js:225), [compare.js:260](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/compare.js:260).

Baseline subtraction interpolates raw numeric x values without checking physical units or scan domain. Global fitting passes each dataset's native numbers into one shared-parameter model without conversion or compatibility checks. Compare's x overlay falls back to scale 1 for incompatible units; its trend applies the first result's unit conversion to every value instead of converting each source separately. These paths can combine unlike quantities even though Analyze overlays already have stronger compatibility checks.

**Fix:** use a common dimensional-compatibility service; convert compatible quantities to canonical units and reject incompatible domains. Validate y units and observables as well as x. **Acceptance:** equivalent ns/µs traces give the same physical shared time constant and trend; frequency/spatial/time inputs cannot be silently mixed; baseline interpolation records conversion, overlap, and method.

### F16 — FFT remains available after processing makes the x grid irregular

**P2 · Source-confirmed · M**  
**Source:** [plot1d.js:204](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/plot1d.js:204), [plot1d.js:377](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/plot1d.js:377), [core.py:315](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:315).

FFT availability depends on a source metadata flag, while exclusions, missing samples, and partial bins can make the processed grid nonuniform. The implementation computes a DFT from sample indices and one average spacing. It also caps its denominator at 2048 while summing all input samples, giving unusual folding behavior for longer traces.

**Fix:** validate the actual processed grid before enabling FFT; offer a clearly defined irregular-sampling method or explicit resampling option. Replace the browser's nested-loop DFT with a tested FFT implementation when warranted. **Acceptance:** a removed middle point cannot masquerade as a uniform FFT; long synthetic tones have the expected frequency, scaling, and documented resolution.

### F17 — Session reopening preserves undo history from the previous workspace

**P1 · Reproduced · S–M**  
**Source:** [state.js:127](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/state.js:127), [state.js:180](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/state.js:180).

`applySession()` replaces files but does not reset the module-level undo/redo stacks. A state probe opened a new session with Reference and bin 9, then Undo restored Contrast and bin 1 from a previous workspace using the same path.

**Fix:** scope history to a workspace/session epoch, reset at replacement boundaries, and make importing/merging a separate operation. **Acceptance:** Undo after Open session cannot inject earlier workspace state; Close all/reopen has similarly explicit history semantics.

### F18 — Sessions and ordinary exports omit scientific-tool results

**P1 · Reproduced for session serialization; source-confirmed for export · M–L**  
**Source:** [state.js:162](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/state.js:162), [science.js:61](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/science.js:61), [analyze.js:1157](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:1157), [stack.js:1](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/stack.js:1).

A serialization probe populated bootstrap, peak-pick, DEER-ratio, and NV-field state; none appeared in the saved file state. Time-trace scientific helpers live under `_sci` too. Stack and AutoFind selections/results live in module-local state outside `sessionDoc()`. Users can invest substantial analysis time and save a session while these results disappear on reopening.

**Fix:** define versioned persistent state for each workspace and tool, including input signatures, assumptions, warnings, and export tables; exclude only transient busy flags and rebuildable caches. **Acceptance:** each advertised analysis has a save/reopen/export round trip or an explicit “not saved” indicator.

### F19 — Figure relocation remaps the panel source but leaves overlay paths behind

**P1 · Reproduced · S–M**  
**Source:** [state.js:232](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/state.js:232), [figure.js:20](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/figure.js:20).

A relocated panel changed from `old` to `new`, but its `overlays:['ref']` remained unchanged when `ref -> newref` was provided. Linked panel regeneration then uses outdated overlay references. Existing session tests check the panel path but do not cover this nested array.

**Fix:** remap and validate the entire source-reference graph, including panel overlays and baseline paths, through one schema-aware migration. **Acceptance:** moving a folder containing a multi-file figure preserves every linked source and explicitly reports unresolved references.

### F20 — Batch output naming is not collision-safe

**P1 · Reproduced · M**  
**Source:** [pro_batch.py:693](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:693), [pro_batch.py:837](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:837).

Flattening relative folders with `__` maps both `a/b.mat` and `a__b.mat` to `a__b`. Their figures, result JSON, and bundles can overwrite each other while the summary keeps both rows. Detector suffixes can introduce a similar collision with a preexisting single-detector filename.

**Fix:** preserve output subdirectories or use a reversible escaped encoding/short stable source hash; reserve a run-local unique output ID before writing. **Acceptance:** adversarial filenames produce distinct outputs and manifest references, including detector suffixes and case-insensitive Windows paths.

### F21 — Previous batch results lack a source/options manifest

**P1 · Source-confirmed · M**  
**Source:** [batch.js:48](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/batch.js:48), [batch.js:75](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/batch.js:75), [core.py:899](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:899).

“Load previous results” sets the recorded input root to the current input control and may discard the remembered run options. Open, refit, Figure, and Send to Compare then resolve paths or settings using mutable controls. A previous output directory is insufficient to reconstruct the original analysis reliably.

**Fix:** save a run manifest with source roots, individual source identities, detector/observable, full options, code/environment versions, and output paths. Make relocated-root remapping deliberate. **Acceptance:** loading output after switching to another input folder resolves the original sources and original settings or clearly asks for remapping.

### F22 — Batch-to-Compare refits do not reproduce the batch calculation

**P1 · Source-confirmed · M–L**  
**Source:** [batch.js:88](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/batch.js:88), [core.py:519](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:519), [pro_batch.py:930](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:930).

The transfer updates only some processing/settings on an already-open file. Existing ROI, exclusions, transforms, baseline, model-family settings, and bounds can remain. It puts a batch family override in `fitSpec.model`, whereas the ordinary fit path generally uses `fitSpec.experiment` for family overrides. Batch smoothing changes the fitted `y`; the v2 path requires `fit_smoothed`. The UI acknowledges that it refits, but the result is not guaranteed to represent the batch row.

**Fix:** import full stored results/curves where possible; otherwise build a fresh request from the manifest and label the operation as a new fit. **Acceptance:** transfer from a deliberately dirty Analyze state either matches batch parameters within a documented numerical tolerance or explains each intentional difference.

### F23 — Batch cancellation/progress can misrepresent completed work

**P1 · Source-confirmed · M**  
**Source:** [core.py:905](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:905), [pro_batch.py:782](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:782), [pro_batch.py:1097](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:1097).

The progress callback marks `done=idx+1` even for “Processing...” before fitting. Input files are all decoded into a work list before per-item progress/cancellation. Summary CSVs are written at the end; cancellation instead reads the output directory's existing summary, which can be absent or belong to an older run. The displayed live table is populated on completion rather than progressively from completed rows.

**Fix:** use explicit discovered/loading/fitting/written stages, stream completed rows into a run-specific journal, and checkpoint summaries atomically. Cancellation should be cooperative between bounded operations and return this run's completed rows. **Acceptance:** cancel a fresh run and a rerun in a populated output directory; both report only actual completed items, with recoverable partial results.

### F24 — Batch PASS can mean plot-only or an unconverged/fallback fit

**P1 · Source-confirmed · M**  
**Source:** [pro_batch.py:953](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:953), [pro_batch.py:1044](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:1044), [pro_batch.py:1068](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:1068), [batch.js:81](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/batch.js:81).

Plot-only maps/line scans are marked PASS, so “Send fitted rows” can select rows with no fit. Fit quality classification does not incorporate `result.success`, unlike the v2 ordinary fit path. A failed requested fit can fall back to SpinEcho and still receive a passing quality status, with the model change recorded less prominently than the PASS badge.

**Fix:** separate load/plot outcome, optimizer convergence, model suitability, and quality review. Use explicit PLOT_ONLY, FIT_PASS, FIT_WARN, FIT_FAIL, and FALLBACK states or equivalent independent fields. **Acceptance:** plot-only rows are never described as fitted; an unconverged or substituted model cannot get an unqualified fitted PASS.

### F25 — Map exports can disagree with the displayed source

**P1 · Source-confirmed · M**  
**Source:** [analyze.js:727](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:727), [analyze.js:1166](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:1166), [core.py:1049](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:1049).

Overlay rendering forwards `M.source`, but the export's map-comparison request omits the source fields. CSV recomputation consequently defaults to the selected observable instead of, for example, a displayed raw Signal map. The backend expects `req.map_comparison` for JSON provenance, while the client supplies only a recomputation request; ordinary `plot_state` also omits the map configuration. Images can therefore be accompanied by insufficient or inconsistent numerical provenance.

**Fix:** define one immutable map-comparison request/result used for rendering, CSV, and JSON, including paths, detector IDs, source observables, recorded units, axis order, normalization, background, and coverage. **Acceptance:** display Signal while the main observable is Contrast; exported cell values and metadata must match Signal, and the session can reconstruct the same comparison.

### F26 — Cross-file map panels and linecuts do not consistently include the second file

**P2 · Source-confirmed · M**  
**Source:** [plot2d.js:73](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/plot2d.js:73), [plot2d.js:180](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/plot2d.js:180).

The overlay API can use `map.compareWith` from another file, but `mapPanels()` and `linecuts()` enumerate only the primary file's detectors. For cross-file comparisons, “Maps + overlay” therefore does not inherently mean the two actual source maps plus their blend, and the second file does not participate in the same linecut path.

**Fix:** represent panels and cuts by `{path,detector,source}` rather than detector alone. **Acceptance:** two single-detector files show primary, secondary, and overlay panels; each cut identifies its source and uses native coordinates without extrapolation.

### F27 — Map shared scaling and spot comparison need stronger compatibility checks

**P2 · Source-confirmed · M**  
**Source:** [core.py:737](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:737), [core.py:767](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:767).

Shared overlay scaling validates the plane but not matching value units/observables. Spot profile correlation treats equal array shapes as a common physical grid and applies the first map's index mask to both. Equal shapes can have shifted, reversed, or different coordinate grids. `distance_px` also divides a two-axis physical distance by only the x pixel spacing.

**Fix:** check units and coordinate registration; compute correlation only on a documented common measured support, or explicitly label a resampling method. Use separate pixel-axis offsets or a clearly defined physical distance. **Acceptance:** shifted/reversed equal-size grids cannot produce a falsely registered correlation; anisotropic pixels have correctly defined distances.

### F28 — Large arrays can exceed JavaScript argument limits

**P2 · Source-confirmed risk · S–M**  
**Source:** [plot2d.js:45](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/plot2d.js:45), [compare.js:26](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/compare.js:26), [analyze.js:877](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:877).

Several reductions use `Math.min(...array)` / `Math.max(...array)` on full measured arrays or flattened maps. Browser engines have a finite argument limit, so sufficiently large maps/traces can throw instead of displaying. Percentile sorting and repeatedly copying maps also add avoidable rendering work.

**Fix:** use iterative reductions and bounded/typed data structures; cache source-dependent statistics separately from presentation. **Acceptance:** a large synthetic map and long trace can change limits, normalize, and open fitting controls without argument-count failures. Measure memory and render latency at realistic sizes before choosing limits.

### F29 — “Acquisition time” is file modification time

**P2 · Source-confirmed · M**  
**Source:** [compare.js:64](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/compare.js:64), [day.js:15](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/day.js:15), [scan_stack.py:105](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/scan_stack.py:105).

Compare uses library peek `mtime`, defaulting to zero if missing, while labeling the axis Acquisition time. Day also orders by file modification time. Copies, checkpoint saves, restored files, and exports can change this timestamp. Stack code already distinguishes filename and mtime provenance, so the application has a useful pattern to reuse.

**Fix:** prefer acquisition metadata, show the source and timezone, and explicitly label save-time/mtime fallback. Missing timestamps must stay missing rather than become the Unix epoch. **Acceptance:** copying/touching a file does not silently alter an acquisition-time trend; an all-equal timestamp set does not imply a meaningful drift slope.

### F30 — The local HTTP API accepts untrusted-origin write-capable requests

**P1 · Read-only acceptance reproduced; write risk source-confirmed · M**  
**Source:** [server.py:277](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/server.py:277), [server.py:317](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/server.py:317).

The server binds to loopback, which limits direct network exposure, but POST dispatch has no session token or Origin/Host validation and does not require JSON content type. A read-only `list_dir` request using `Origin: https://example.invalid` and `Content-Type: text/plain` was accepted. The same dispatcher exposes file writes, uploads, exports, and local open actions. Cross-origin response reading is separately restricted by browser CORS; that does not establish protection against blind side effects.

**Fix:** use an unguessable per-instance token, an explicit endpoint allowlist, strict request validation, and host/origin checks compatible with the desktop bridge. Add request-size limits. **Acceptance:** untokened or foreign-origin requests are rejected before invoking any endpoint; the ordinary app and authorized second-instance handoff still work. No destructive exploit was attempted in this audit.

### F31 — Most tests and all fixtures are absent from the tracked checkout

**P1 · Reproduced by Git inventory · M**  
**Source:** [.gitignore:9](D:/BacklundLabResearch/Code/Lab_Data_Analysis/.gitignore:9), [README.md:168](D:/BacklundLabResearch/Code/Lab_Data_Analysis/README.md:168).

There are 27 local Python test files and 11 local JavaScript test files. `git ls-files tests` lists only `test_map_comparison.py` and `test_time_trace_support.py`; `tests/*` ignores the others, including the session and fit-engine tests. `TestData/` is ignored too. This contradicts README descriptions of checked-in fixtures and means the passing local suite cannot be reproduced from an ordinary fresh clone. The earlier backlog's statement about an unignored session regression is also stale in this checkout.

**Fix:** deliberately track the portable tests and small synthetic/public fixtures, with large/private lab data kept optional. Add a documented Python + Node verification command and CI configuration. **Acceptance:** a fresh clone in a clean environment runs the promised portable suite; missing private datasets produce explicit skips without hiding missing public fixtures.

### F32 — Documentation and defaults disagree about line-scan Auto behavior

**P2 · Source-confirmed · S–M**  
**Source:** [core.py:521](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/core.py:521), [analyze.js:859](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:859), [pro_batch.py:978](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/pro_batch.py:978).

The v2 backend maps LineScan Auto to GaussianPeaks; the inspector says LineScan Auto is plot-only; batch deliberately plots LineScan Auto. This makes the same “Auto” choice mean different things across interfaces. The app reference/README/user guide and old backlog also need a current feature/status matrix, rather than accumulating conflicting claims.

**Fix:** choose a consistent policy or expose the intentional distinction prominently. Generate default labels/help from the model/profile registry where possible. **Acceptance:** an unmodified line-scan workflow behaves as the displayed help promises in Analyze, Compare, Day, and Batch.

### F33 — Out-of-order data loads can overwrite newer settings

**P1 · Reproduced with controlled response order · M**  
**Source:** [state.js:82](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/state.js:82), [analyze.js:47](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:47), [analyze.js:60](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:60).

`loadData()` applies all responses after awaiting them without checking whether the file's current mode/processing still matches the request. A controlled state probe started Contrast then Signal loads, completed Signal first, and completed Contrast last. The file ended with `mode:'signal'` and a Contrast data payload. Rapid control changes, network/disk latency, or watch refresh can expose this ordering.

**Fix:** snapshot the complete request, assign a per-file revision, and apply results only if that revision still owns the file. Cancel requests where practical and handle failed loads transactionally. **Acceptance:** deliberately reverse response order for mode, processing, source update, and closed-file cases; only the newest valid state renders or fits.

### F34 — Resampling can retain incorrect raw-point ownership

**P2 · Reproduced · M**  
**Source:** [analysis_tools.py:127](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/analysis_tools.py:127), [analysis_tools.py:182](D:/BacklundLabResearch/Code/Lab_Data_Analysis/nvfit/analysis_tools.py:182), [analyze.js:138](D:/BacklundLabResearch/Code/Lab_Data_Analysis/smartfitter/static/js/analyze.js:138).

Resampled points are remapped to source groups only when the output length changes. Resampling `[0,8,9,10]` to four regular points produced x `[0,3.333,6.667,10]` while retaining groups `[[0],[1],[2],[3]]`, as though the interior derived points still directly belonged to the original points at 8 and 9. A nearest-point mapping is itself only an approximation for interpolated samples. Excluding a displayed derived point can consequently target a misleading raw sample.

**Fix:** carry explicit interpolation support/weights or distinguish derived samples from directly measured bins. Define whether selection excludes interpolation support or a measured coordinate interval. **Acceptance:** changing the grid without changing length updates provenance correctly; readout and exclusion never imply that an interpolated point was measured there.

## 3. UI, UX, fitting features, and engineering roadmap

The following sections are proposals unless explicitly labeled otherwise. They extend the source findings above and are not claims that the corresponding behavior is broken in every workflow.

### 3.1 UI and layout recommendations

The inspected 1280×720 Analyze view had a useful central white plot and compact library, but three rows of plot controls consumed substantial height, the right-side Run fit/Robust fit row clipped its text/shortcut, and Series controls lay below the initial visible main area. Compare used separate file, plot, and fitting panels with two vertically stacked figures, leaving the trend below the initial viewport. These are observations of that layout, not measurements across all monitor/DPI combinations.

| ID | Priority / effort | Recommendation and reason | Acceptance / useful completion criterion |
|---|---|---|---|
| UX01 | P1 / M | **Create one responsive action row.** Keep Run fit prominent; move Robust fit into an adjacent menu or let it wrap to a complete second row. Shortcut badges should yield space before action text. | At 1000×640, 1280×720, 1366×768, 1600×900, and 125–200% display scaling, action names remain readable and controls do not overlap. These sizes still need testing. |
| UX02 | P2 / M | **Use a plot-first layout with compact/expanded control modes.** Put common layers on one row; group uncertainty, diagnostics, and iteration layers into labeled popovers. Keep every existing capability accessible. | Opening Series or the editor preserves a useful plot area, with visible indications when more controls are below. No removed shortcuts or hidden enabled analysis state. |
| UX03 | P2 / M | **Make Compare's main area switchable between Overlay, Trend, and Split.** A user studying a trend should not have to scroll past a large overlay each time. Add an adjustable splitter on larger screens. | Each figure can occupy the central area alone; the choice persists; the trend's controls remain beside the trend. |
| UX04 | P1 / M | **Make Save and Copy follow the current workspace.** The top Save currently calls `saveResults(active())` in every view, including Compare/Figure, so its destination is ambiguous. Use explicit Save analysis, Save session, Export figure, and Export trend actions. | The button label, tooltip, shortcut behavior, output preview, and current workspace agree. A Figure user cannot accidentally export the earlier Analyze file with the main Save button. |
| UX05 | P1 / M–L | **Add a named-session header, dirty indicator, and recovery autosave.** Distinguish saving analysis state from exporting figures. Single-file Close currently discards state immediately; Close all has only a fit-oriented warning. | A deliberately crashed session can be recovered; closing unsaved fits, curation, annotations, or scientific tools offers a clear recovery path. Autosave does not overwrite a named session or raw data. |
| UX06 | P2 / M | **Create one persistent result inspector.** Show source, detector, observable, target, model, status, and key values at its top. Keep detailed parameters, diagnostics, and history in collapsible sections below. | A result's identity and stale state remain visible while switching tabs; the plot and inspector reference the same analysis item. |
| UX07 | P2 / M | **Use validated numeric inputs with units beside the field.** `num()` strips unrecognized characters and can turn a mistyped quantity into a plausible different number. Add inline parse/range errors and optional explicit unit entry. | Invalid text stays visible with an error; `1 ms` is deliberately converted or rejected in a native-ns field; incompatible units are never silently discarded. |
| UX08 | P2 / M | **Preserve inspector focus, scroll, and expanded sections.** Full DOM replacement on state emissions can interrupt reading or editing and rebuild unchanged controls. | A background load, watch tick, or fit update does not move the active field or reset an unrelated section. Enter commits once; blur does not create duplicate history entries or requests. |
| UX09 | P2 / S–M | **Offer explicit map interaction modes: Inspect, Horizontal cut, Vertical cut, Cross, and Line.** A readout-only click should be available without changing layout. | Cursor behavior is obvious before clicking; reset and double-click are reliable in each mode; the chosen mode persists per analysis. |
| UX10 | P2 / M | **Improve accessibility and keyboard-only work.** Extend the existing labeled fields to raw fit-range inputs; give modals dialog semantics, initial focus, focus trapping/restoration, and menus keyboard navigation. Add selected/pressed semantics and live feedback for async status. | File opening, fitting, exclusion management, editor, and export are usable without pointer input; a screen reader can identify modal purpose and selected controls. |
| UX11 | P2 / S–M | **Show persistent, actionable errors.** Toasts are useful acknowledgement, but fit/load/export failures need a retained error row with Retry, Details, and Copy diagnostics. | A disappeared toast cannot leave a failed operation looking successful; details identify the affected file/detector and preserve the previous valid result. |
| UX12 | P2 / M | **Make file rows explicit about source status.** Show acquisition/save/checkpoint state, available versus failed detectors, duplicate/checkpoint relationships, and fitted/stale/unsaved indicators. Add deliberate multi-file Close and Restore closed. | A partially loaded dual-detector file and a checkpoint duplicate are identifiable before fitting or pooling. Removing a comparison does not unexpectedly destroy a source analysis. |
| UX13 | P2 / M | **Unify appearance editing.** Keep double-click editing and the full editor, but route both through the same state/actions. Add a searchable appearance panel, reset-this-control, and visible units/axis-role labels. | Direct edits, side-panel edits, undo, sessions, and PNG/SVG/PDF reflect the same settings. Appearance changes do not invalidate a scientific fit. |
| UX14 | P3 / S–M | **Reduce first-use clutter without losing discoverability.** Replace the permanent large shortcut card with a compact recent-session area, task examples, and searchable shortcut help. Add concise onboarding for Compare versus Analyze overlays. | Returning users reach recent work quickly; first-time users can identify which workspace performs their intended task without learning internal terminology. |

### 3.2 Fitting and scientific-analysis improvements

These proposals build on the correctness fixes; they should not be used to disguise unresolved target, unit, or covariance problems.

| ID | Priority / effort | Improvement | Required scientific/UX behavior |
|---|---|---|---|
| AN01 | P1 / L | **One prepared analysis request for every fitting route.** Include source hash/revision, detector, observable, exact sample indices, selected iterations, preprocessing, fit target, model, bounds, locks, loss, weights, and seed. | Analyze, Compare, Day, Batch, global/individual iteration fits, and bootstrap either reuse this request or explicitly describe a difference. Save the request with the result. |
| AN02 | P1 / M | **Model-aware fit readiness checks.** Check distinct x values, span, finite data, free parameters, rank, range overlap, valid bounds, weight availability, and domain before starting. | Fail early with a fixable explanation such as “7 points cannot identify 11 free parameters,” not a generic optimizer error. |
| AN03 | P2 / M | **Expose optimizer and model-selection evidence.** Show requested versus actual model, starts attempted/succeeded, termination, bound proximity, candidate scores, and why Auto chose a model. | A lower-R² model selected for a physical/complexity reason is explainable. Do not compare information criteria across different data subsets or incompatible target scales without accounting for that difference. |
| AN04 | P2 / M | **Residual diagnostics that lead to action.** Add normalized/standardized residuals, residual autocorrelation, residual-versus-x and residual-versus-signal views, and optional leave-one-iteration-out sensitivity. | Diagnostics distinguish baseline drift, model mismatch, heteroscedasticity, and influential iterations. A PASS badge remains a software check, not proof of a physical model. |
| AN05 | P2 / M–L | **Offer explicit noise/weight choices.** Start with unweighted and empirical iteration SEM, then consider Poisson/count likelihood only when count and integration-time metadata support it. Provide robust-loss controls separately from subset fitting. | The selected objective, actual applied weights, fallback reason, and uncertainty assumptions are visible and exported. Counts/s alone must not be treated as Poisson counts. |
| AN06 | P2 / M | **Distinguish uncertainty types.** Show SD, SEM, fitted parameter standard error, bootstrap interval, model-mean confidence band, and prediction interval as separate choices. | Labels include coverage and method; unknown errors remain unknown. Smoothed/resampled data carry a correlation warning and do not acquire unjustified independent-point precision. |
| AN07 | P2 / L | **Add profile likelihood and block/hierarchical bootstrap options.** Useful for asymmetric decay-time uncertainty, weakly identified stretch exponents, and correlated iterations/time traces. | Report intervals or lower/upper limits when a symmetric ± estimate is misleading; expose block/run grouping and accepted-draw counts. Validate methods on known synthetic inputs before enabling by default. |
| AN08 | P2 / M | **Parameter correlation and identifiability view.** Display covariance correlation, numerical rank, bound proximity, and fit-window coverage relative to decay/envelope time. | Large uncertainty, a near-flat direction, or a lifetime extending beyond the measured window is visible even with high R². Allow sensitivity plots without changing the saved fit. |
| AN09 | P2 / M | **Fit-range and preprocessing sensitivity explorer.** Compare a small, declared set of ranges/binning choices while keeping one accepted result and a provenance table. | Show parameter stability and residual changes; never silently choose a favorable range or delete inconvenient points. Every candidate records its retained sample set. |
| AN10 | P2 / M | **Global fit with units, full configuration, and inspectable constraints.** Support individual versus shared bounds, meaningful model names, actual fit range/target, and exportable shared/independent comparisons. | Stale/global state depends on configuration, not only membership. Show when SEM weighting is unavailable and avoid labeling unweighted RSS/dof as a calibrated χ² statistic. |
| AN11 | P2 / M | **Strengthen ODMR/Rabi workflows.** Add visible center-range chips and peak assignments, per-peak residual review, consistent peak-count controls, parameter locking in multipeak fits, and saved Rabi calibration context. | The selected number of peaks is respected or Auto's departure is explained; first-peak/π/π⁄2/envelope definitions remain distinct and reproducible. Calibration quality includes source and timing metadata. |
| AN12 | P1 / M | **Keep physical interpretation calibrated.** Allow users to label optical relaxation as apparent/descriptive, retain acquisition sequence/charge/spectral context, and show NV-field assumptions and degeneracies alongside values. | The application does not turn an optical decay into an intrinsic spin T1 claim or a fitted spot into proof of a single emitter. Derived metrics retain units, assumptions, and uncertainty provenance in exports. |

### 3.3 Useful feature additions

| ID | Priority / effort | Feature | Concrete scope and acceptance |
|---|---|---|---|
| FE01 | P1 / L | **Analysis variants and a result library** | Keep raw source files separate from named analyses such as D1 contrast/raw, D1 contrast/ROI, D2 reference. Compare any compatible variants without overwriting. Session/export IDs stay stable across file moves. |
| FE02 | P2 / M–L | **Reusable workflow recipes** | Save selection rules, processing, model/settings, presentation, and export destinations as a recipe with preview. Apply to new files only after validating domains, units, and available detectors. |
| FE03 | P2 / L | **Whole-figure templates** | Save panel roles, dimensions, spacing, shared-axis rules, annotations, legend/result placement, and styles. Bind new compatible sources through named slots. Keep source-specific values separate from reusable design. |
| FE04 | P2 / M | **Linked versus frozen figure panels** | Let each panel explicitly follow its current analysis or freeze a versioned snapshot. Show an update badge when a linked analysis changes, refresh thumbnails, and preserve the exact exported version in the sidecar. |
| FE05 | P2 / M | **Time-trace quality and event review** | Extend the existing Sampling card with a cadence/gap timeline, retries, acquisition pauses, exclusion overlays, and an explicit raw/processed scope selector for statistics, periodograms, steps, ratios, and drift. Preserve measured timestamps. |
| FE06 | P2 / M–L | **Map ROI and registered comparison tools** | Add rectangular/polygon measurement ROIs, integrated/background-corrected counts, annotated linecut export, and optional registration estimates displayed as a separate transform. Preserve the measured grids and clearly distinguish any derived registered image. |
| FE07 | P2 / M | **Spot tracking review and curation** | Let users choose a seed spot, reject or reconnect ambiguous links, mark split/merge/missing detections, and inspect alternatives. A detector-offset plot should show which matched physical feature produced each point. |
| FE08 | P2 / M | **Checkpoint/run grouping** | Recognize successive checkpoints and final files as a run family, compare convergence, and prevent accidental pooling of overlapping observations. Ask for a deliberate override when duplication cannot be resolved from metadata. |
| FE09 | P2 / M | **MAT schema inspector and v7.3 support across the app** | Reuse the existing AutoFind HDF5 decoding experience to support v7.3 in the main loader, which currently calls SciPy `loadmat` directly. Show root schema, detector structure, axes, recorded units, and unsupported fields before fitting. |
| FE10 | P2 / M | **Portable analysis bundle and reproducible CLI** | Export a session, manifest, fit tables, selected figures, source descriptors, environment/code versions, and an executable recipe. Raw data inclusion should be explicit. A dry-run replay reports missing/changed sources before computing. |

### 3.4 Architecture, performance, packaging, and reliability

| ID | Priority / effort | Recommendation | Evidence / acceptance |
|---|---|---|---|
| EN01 | P1 / L | **Centralize result identity, stale checks, and state transitions.** | Current results are split across per-file fits, per-mode fits, Compare results, global state, iteration fits, scientific cards, and module-local workspaces. One dependency graph should determine validity, persistence, undo, and export. |
| EN02 | P1 / M–L | **Introduce a cancellable job service.** | Fit, bootstrap, iteration fit, and global fit are long synchronous API calls; batch uses a separate daemon-thread path. Use bounded workers, IDs, revision checks, progress stages, and cooperative cancellation, with explicit shutdown behavior. |
| EN03 | P1 / M | **Version and validate API/session/export schemas.** | v2 sessions check format but not supported version; arbitrary public API methods are auto-exposed. Validate field types, finite numbers, bounds, array shapes, size limits, and schema versions. Return structured errors without relying on raw tracebacks in normal UI flows. |
| EN04 | P2 / M | **Measure file-to-first-plot performance.** | `openFile()` loads a full initial payload, then loads all needed detectors again; source-mode caches can decode the same MAT multiple times. Record file listing, header peek, decode, preprocessing, JSON size, transport, and first completed render before optimizing. |
| EN05 | P2 / M–L | **Separate raw decoding from observable processing.** | A single immutable decoded source could feed detector/mode views, reducing duplicate I/O and memory. Keep native arrays/metadata authoritative; bound caches by bytes as well as item count. Close/release derived data deliberately. |
| EN06 | P2 / M | **Use virtualized lists and display-only decimation where needed.** | Large library/iteration/spot tables and full-resolution JSON can dominate interaction. Draw a bounded preview while retaining exact samples for fits, readout, and export. State plainly when a display is decimated. |
| EN07 | P1 / M | **Make output writes transactional and provenance complete.** | Figure/data exports and batch files mostly write directly. Stage a complete export in a run directory, validate it, then publish a manifest atomically; show per-file failures and overwrite policy. Include timezone, full options, source hash, and code revision. |
| EN08 | P2 / M | **Document and test a supported environment.** | `requirements.txt` has no version ranges; the build script installs current PyInstaller; the deck script imports `pptx` without that dependency being declared. Separate runtime/dev/research-script dependencies and record a known-good lock/constraints file. Validate a clean source install and a clean-machine portable launch. |
| EN09 | P2 / M–L | **Decide the legacy Qt support boundary and remove duplicated science gradually.** | `gui_app.py` contains 12,553 lines and 398 main-window methods, including duplicated lock/fit/export logic. Its lock adapter repeats F01/F02. Preserve supported legacy workflows, route shared calculations through one service, and move legacy-only dependencies behind the legacy launch. |
| EN10 | P2 / M | **Treat Windows-only integration and build/update identity explicitly.** | Explorer/startfile, PowerShell clipboard, backslash path construction, Edge fallback, and launcher registration are Windows-specific despite broad README platform claims. Add platform adapters or narrow support claims. Use one version/build identity across GUI, batch provenance, portable zip, and update comparison; test second-instance handoff and shutdown separately. |

## 4. Earlier backlog: current status reconciliation

The October 14 backlog was used as a lead list and checked against current source. It should not be treated as a current bug inventory without this reconciliation.

| Earlier item | Current audit conclusion |
|---|---|
| Compare session fit loss | Result serialization/restoration exists and the local round-trip tests pass. Keep this behavior; new work is validity and complete dependency remapping. |
| Map double-click unexpectedly creates a cut | The deferred/cancellable click handling and map regressions exist. This audit does not re-report the original issue as an unfixed bug. |
| Compare stale results | Still present; now visibly reproduced in F06. |
| Both detectors in Compare | Still a path-identity limitation; F07. |
| Batch manifest | Still needed; F21/F22. |
| Folder names containing periods | The current drop classifier recognizes `.mat`/`.json` explicitly, so the earlier generic extension heuristic has been changed. Native OS folder dropping still deserves integration tests; it was not reproduced here. |
| Arbitrary right-hand comparison axis | Analyze's per-file Series controls already expose a right axis, and a local JavaScript regression passes. Extend/validate its unit behavior as needed; do not present this as wholly absent. Compare workspace parity remains a separate design choice. |
| Per-series uncertainty | Analyze comparisons already have per-file band/bars choices. More styling and consistent detector/workspace coverage are enhancements, not a completely missing feature. |
| Time-trace cadence readout | A Sampling card already displays median/range/gap/retry information. FE05 proposes a richer timeline and scope consistency. |
| Autosave/recovery and complete templates | Still valuable additions; UX05 and FE03. |
| Session regression tracked in Git | The file exists locally but is currently ignored; F31 supersedes the earlier tracking claim. |

## 5. Suggested implementation order

### Round A — Protect the numerical result

Fix F01–F05, F08, F11–F15, F33, and F34 with focused reproductions promoted to portable tests. Establish one prepared fit request and strict detector/unit handling. Add the custom-expression default test and missing-stage-coordinate test early; these give a high return for a small regression investment.

**Exit criterion:** no detector substitution, no coordinate reassignment, controls affect the actual fit, and unsupported uncertainty/data regimes are explicit.

### Round B — Protect saved and compared work

Implement result identity/signatures (F06/F07), session/history completeness (F17–F19), batch manifest/collision/cancellation semantics (F20–F24), and faithful map export (F25–F27). Address the local API boundary and tracked test gap (F30/F31) alongside this work.

**Exit criterion:** the same saved analysis is identifiable and reproducible across workspaces, source moves, restart, batch transfer, and export; stale data cannot appear current.

### Round C — Reduce daily interaction cost

Deliver UX01–UX13, starting with readable actions, context-aware Save, dirty/recovery state, compact plot controls, Compare plot modes, and persistent diagnostics. Measure responsiveness before replacing infrastructure.

**Exit criterion:** complete open → select detector/observable → process → fit → compare → save/export tasks at laptop dimensions and high DPI using pointer and keyboard.

### Round D — Expand scientific and publication workflows

Add the highest-value requests from AN03–AN12 and FE01–FE10. Start with reusable recipes, result variants, frozen figure panels, checkpoint grouping, and portable provenance bundles. Profile likelihood, advanced noise models, registration, and tracking curation should each have synthetic validation and explicit assumptions.

**Exit criterion:** every new analysis tool has a defined input/target/unit contract, source-backed uncertainty, session persistence, export support, and an inspectable failure path.

## 6. Validation plan for future fixes

### Minimum portable regression matrix

| Area | Cases to add or retain |
|---|---|
| MAT loading | Legacy/snapshot/v7.3; one/dual/partially failed detector; missing/NaN samples; descending/irregular axes; stage metadata conflicting with names; reference-only data. |
| Fit controls | Every supported family with free seeds, lower/upper bounds, one/all locked parameters, insufficient points, constant data, explicit model versus Auto, count-rate versus contrast observable. |
| Processing | ROI before/after binning contract; partial bins; sample exclusions; baseline units/overlap; transforms; same-length resampling; smoothed fit target and correlation note. |
| Uncertainty | Known linear/nonlinear synthetic reference; rank deficiency; underdetermined data; robust and weighted fits; bootstrap failed draws/model changes; fixed parameter degrees of freedom; band/parameter consistency. |
| State/races | Reversed API response completion; mode/processing changes during fit; close/session replacement during work; undo/redo boundaries; stale cache after failed refit; watched-source revision changes. |
| Compare/global | Two detectors of one file; multiple analysis variants; mixed compatible units; incompatible domains; shared bound conflict; mixed observable rejection; hidden rows; stale trends; timestamp provenance. |
| Maps | Asymmetric grids and reversed storage axes; physical/square geometry; different-grid union/coverage; source identity in all panels/cuts; raw-source overlay export; log readout; missing cells; large-map reductions. |
| Sessions/figures | Relocated panel overlays/baselines; changed/missing sources; bootstrap/NV/DEER/time tools; Stack/AutoFind state; linked/frozen panels; unsupported session version; snapshot fallback explicitly labeled. |
| Batch/export | Output naming collisions; same basename in subfolders; partial cancellation; previous output directory; exact options replay; plot-only/fallback/convergence statuses; vector-PDF fallback notice; interrupted/overwritten output. |
| Desktop UX | Cold/warm source and portable launch; native file/folder drops including periods; second-instance handoff; clipboard; shutdown during long work; laptop/high-DPI layout; keyboard and modal focus. |

### Reference reproductions from this audit

These are minimal behavioral checks, not newly added repository tests.

| Probe | Observed result | Expected behavior |
|---|---|---|
| Lock TimeTraceDecay `tau_s=2` | Strict-bound optimizer error | Exact fixed parameter and successful free-parameter fit. |
| Apply unlocked seed `tau=42` to initial seed 1 | Seed remains 1 | Requested finite seed is applied. |
| Default custom expression in `x` | `Unsupported name ... x` | Default UI expression is accepted. |
| Stage `[0,1,2,3]`, values `[10,NaN,30,40]` | x `[0,1,2]`, y `[10,30,40]` | x `[0,2,3]`. |
| Request detector2 when only detector1 loaded | Detector1 data returned | Explicit unavailable-detector error. |
| Rank-2, 2×3 Jacobian | Three finite variances | Full null-space/insufficient-data handling. |
| Four bootstrap draws all `success=False` | `n_success=4`, zero spread | No successful draws and an unreliable/unavailable interval. |
| Shared `tau_s` intervals `[.5,1]` and `[2,3]` | Success at approximately 2.141 | Conflicting-bound error. |
| Normalize squared T1 target with same-length original trace | Original trace used | Selected target normalized. |
| Relocate panel `old→new`, overlay `ref→newref` | Panel changes; overlay remains `ref` | Every nested source remapped. |
| Open a new session after previous history | Undo restores previous mode/bin | New session has isolated history. |
| Populate scientific-tool state then serialize | No `boot`, `peakPick`, `deerRatio`, `_sci` keys | Versioned saved tool state or clear unsaved status. |
| Recursive `a/b.mat` and `a__b.mat` | Both output stems `a__b` | Unique output destinations. |
| Contrast load A, Signal load B; finish B then A | Current Signal mode with Contrast payload | A discarded as stale. |
| Resample x `[0,8,9,10]` to four points | New interior x with old direct groups | Correct derived-sample provenance/selection semantics. |

The visible stale-Compare reproduction used `TestData/NewCodebaseTests/T1_20260324_044350.mat` and `T1_20260324_054548.mat`. These files are local ignored fixtures, so future portable tests should replace the reproduction with compact synthetic sources or deliberately tracked fixtures.

## 7. Coverage map and maintenance notes

| Component | Reviewed behavior and principal follow-up |
|---|---|
| `nvfit/io_mat.py` | Root schemas, classification, units, detector errors, measured timestamps, iterations, stage-axis finalization. F04/F05, shared v7.3 decoding. |
| `models.py`, `fit_engine.py`, `fit_workflows.py`, `ramsey.py`, `rabi_utils.py`, `diagnostics.py`, `profiles.py` | Model registry, parameter controls, multistart, bounds, uncertainty, target preparation, calibration metrics and quality gates. F01–F03/F08–F16, AN01–AN12. |
| `analysis_tools.py`, `preprocess.py`, `iterations.py` | Source groups, ROI/binning/transforms, smoothing, selection and exact-grid pooling. Preserve native samples; fix derived ownership and duplicate-run handling. |
| `bootstrap.py`, `global_fit.py` | Resampling, convergence, shared constraints, information criteria and covariance. F11–F15, consistent intervals and objective provenance. |
| `trace_tools.py`, `deer_tools.py`, `nv_field.py` and v2 scientific extensions | Raw-domain helper contracts, noise assumptions, timestamp pairing, physical assumptions and ambiguity. Preserve existing caveats; add saved results and scope controls. Validate short/constant/duplicate-time edge cases before broadening tools. |
| `map_comparison.py`, `scan_spots.py`, `scan_stack.py` | Native cell geometry, monotonic axes, missing coverage, Gaussian fits, linking and drift. Preserve no-extrapolation guarantees; improve source identity, registration checks and track curation. |
| `autofind.py`, `autofind_figures.py`, `ext_autofind.py`, `autofind.js` | Lazy source indexing, bounded caches, saved/reconstructed evidence labels, raster paging, async epochs and publication export. Useful patterns for other workspaces; add full session persistence and clean portable fixtures. |
| `core.py`, `server.py`, extensions | Cache identity, JSON adapters, load/fit contracts, exported data, jobs and local API. Share immutable requests, strict detector errors and validated schemas; measure payload/first-render costs. |
| `state.js`, `app.js`, `analyze.js`, `iterations.js`, `overlaypicker.js` | Workspace routing, history, sessions, pending loads, watcher, detector/mode/processing actions, iteration and overlay UI. F06/F17–F19/F33, centralized transactions and recovery. |
| `compare.js`, `batch.js`, `day.js`, `stack.js`, `figure.js` | Result identity, trends, run transfer, summary, tracking and figure regeneration. F07/F20–F29, contextual export and versioned tool state. |
| `plot1d.js`, `plot2d.js`, `fitdisplay.js`, `unitdisplay.js`, `plotstyle.js`, `ploteditor.js`, `seriescontrol.js`, `science.js`, `export.js`, `util.js`, CSS/HTML | Display spaces/units, curve/band mapping, native maps, direct editing, accessibility, responsive layout and image export. Preserve existing parity improvements; repair scope/provenance consistency and clipping. |
| `gui_app.py`, `plot_presentation.py`, `app_state.py`, `sessions.py` | Legacy desktop state, detector workflow, map view restoration, presentation editing, fit threading and sessions. Regression suite passes locally; duplicated calculation paths remain a maintenance risk. |
| Launchers, `install.py`, `updates.py`, portable spec/build scripts, requirements | Platform integration, bundled assets, version/check behavior and dependency reproducibility. Reviewed structurally; no registry changes, install, update, or fresh executable build performed. |
| `scripts/analyze_429_tdd.py`, `scripts/build_429_tdd_deck.py` | Specialized report/deck entry points, hard-coded default data locations, dependency/export usage. Treat as research recipes with pinned inputs and provenance; do not infer current experimental findings from static report-generation code. Not rerun against lab data. |
| README/user guide/app reference/backlog/tests | Current behavior claims and portable coverage compared with source/Git. Reconcile drift and explicitly track the portable verification assets. |

### Additional bounded follow-ups

These are useful candidates or smaller proposals, not extra confirmed high-priority defects:

- **Watcher transactions:** retry checkpoint reads after size/mtime stabilize, invalidate fits when auto-refit is off or fails, and requeue changed metadata peeks. The current watcher clears some peek entries but requests new peeks primarily for newly discovered files.
- **Result removal semantics:** distinguish hiding, removing a comparison item, closing a source, and deleting an analysis result. Remove unused payloads after explicit disposal; do not silently resurrect cached Compare results on re-add.
- **Scientific-tool scope:** whole-trace statistics are already labeled, but periodogram/step/ratio/drift requests do not pass `proc`. Show Raw acquisition versus Processed selection explicitly and include that scope in saved results.
- **Stack request validity:** the built-stack key contains selected paths/detector/source, while trajectory/detection options and source revisions need their own signatures. Guard late responses and stale results when settings change.
- **Spot source provenance:** Auto fluorescence selects a map based on variation. Return and display the actual selected source and recorded units rather than only the word Auto. Apply a clear policy to failed Gaussian spot fits before using them for tracking.
- **Map log hover:** log mode replaces heatmap z with `log10(z)` while the hover template retains the recorded-value label. Display both recorded value and log color coordinate so the readout is unambiguous.
- **Irregular linecut edges:** arbitrary-line extent uses an average half-cell width; use the same monotonic cell-edge authority as overlays for nonuniform grids. Avoid distance-in-pixels claims on anisotropic grids without a definition.
- **Figure limits:** panel-label strings cover only ten letters; extend to larger figures or validate limits. Purge obsolete Plotly panels, version asynchronous renders, and validate positive physical sizes/DPI.
- **PDF fidelity:** vector conversion silently falls back to raster when a PNG is available. Report that fallback, validate missing requested outputs, and compare vector/raster text, Greek symbols, clipping, and colorbars against screen output.
- **CLI parity:** `batch_cli()` omits DPI, Origin-bundle selection, and detector labels. A copied command should reproduce every material option or state its differences. Quote platform paths robustly.
- **Short time traces:** ratio accepts two-point traces but automatic `rolling_sigma()` requires four; choose a documented fallback or validate the minimum up front. Periodogram should reject insufficient span, duplicate-only timestamps, constant signals, and invalid frequency ranges with helpful messages.
- **Source refresh identity:** mtime-only raw-cache keys can miss replacements that preserve timestamps. Consider size plus a revision/fingerprint strategy appropriate for large files, and document when hashing occurs.
- **Instance lifecycle:** use a tokenized handshake and atomic ownership record rather than treating any responding `index.html` on a recorded port as the app. Exercise simultaneous starts, Edge fallback, and closing the only window during a running job.
- **Research-script reproducibility:** save selection/ranking criteria and source checksums beside generated reports/decks; make experiment-specific defaults explicit CLI arguments. Keep descriptive software output separate from scientific conclusions.

## 8. Preservation record

The audit started with existing modifications in `nvfit/gui_app.py`, `nvfit/io_mat.py`, `nvfit/pro_batch.py`, and `tests/test_map_comparison.py`, plus an untracked `.claude/` directory. They were preserved. This new audit document is the only intended repository write. No commit, push, package installation, acquisition-file edit, or application fix was performed.

Final verification compared SHA-256 hashes of all pre-existing tracked files against the start-of-audit baseline: **zero changed files**. Git status retained the original four modified files and untracked `.claude/` directory, with this audit document as the sole added repository file. Temporary test logs, synthetic probe output, isolated settings, and the temporary UI server were kept outside the repository; the document includes the essential results so it does not depend on those temporary artifacts. The temporary browser tab was closed after inspection.
