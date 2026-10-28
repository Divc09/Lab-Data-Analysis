# Shared fitting and result identity

Design note written before implementation for F08/F09/F10/F14 (AN01), F06/F07/F18/F25/F15, F21–F23 and F30. Existing uncommitted work, including iteration sensitivity, is preserved.

## Reproductions before changes

- **F08:** supplied squared T1 target `[0,.04,.25,.81,1]` normalized as `[1,.8,.5,.1,0]`, matching the original unsquared trace instead of the requested target.
- **F09:** robust TimeTraceDecay returned background about 1 despite a background lock at 0; selected SEM weighting was absent from its result.
- **F10:** requesting Simple Ramsey, three starts and locked y0=123 still invoked all three models with 18/18/24 starts and contrast-sized bounds.
- **F14:** a weighted linear fit with an exact intercept had zero intercept error but a nonzero confidence band at x=0. Its band at x=10 also disagreed with the fit-derived slope uncertainty.

## One prepared fit request

Prepare the selected x/y target, aligned SEM, range, family/model choice, seeds/bounds/locks, multistart, calibration and T1 normalization once. Ordinary fitting, robust subsets, Ramsey and bootstrap use this request. Subsets slice arrays and weights, retain settings, and score predictions in the full request's target space. Freeze T1 normalization against the full selected target, never recover it from an unrelated raw trace. Explicit Ramsey selection fits one candidate. The optimizer owns covariance and the free-parameter mapping; model-mean bands propagate that same covariance, including any parameter rescaling. Missing/unidentifiable covariance gives an unavailable band, not a second estimate.

## One result envelope

An analysis descriptor contains source path, detector, observable and canonical processing. Its deterministic analysis ID distinguishes both detectors and independently processed variants. A result adds kind, fit/tool settings, source revision and dependent sources (baseline, combined inputs, comparison maps). A canonical signature identifies the complete calculation. Display-only style is excluded from scientific signatures, but map/export view state is recorded separately.

Use the same envelope for fits, bootstrap, peak-pick, DEER/tool results, Stacks and global fits. Preserve historical results; check freshness centrally before curves, trends, copy/export or session restoration. A failed refit never makes an earlier result current. Async replies retain the request signature and cannot be relabelled as a newer calculation. Unknown signatures in older sessions are explicitly unverified.

Compare membership becomes analysis items, not file paths. Detector is fixed per item. Linked items follow the source's current analysis settings and become stale when these change; captured variants retain their own processing. Data, labels, fit results, trend points and selection all use item identity. Batch sends both detector rows. A versioned session migration converts old path entries, remaps relocated paths/dependencies and preserves envelopes, scientific tools and Stack state. Changed sources keep results marked historical/stale.

## Units and maps

Validate domains, observables and y units before cross-file operations. Convert compatible x units explicitly before baseline interpolation, pooled fitting and Compare; convert each trend value and error from that result's own unit. Reject incompatible quantities with the affected sources named. Global-fit output records its calculation units.

A map view descriptor records source/detector/observable, shown map source, normalization, limits, geometry, swap, crop and decorations. Screen, Figure and exports consume the same descriptor. Export provenance includes every participating analysis and the shown bounds; saved images must use the displayed composition rather than a separately rebuilt default map.

## Batch and local server

Persist an atomic run manifest with source revisions, effective options, stable output stems and per-row completion records. Reruns reference that manifest and selected row identities, preserving options and unaffected results. Cancellation checks occur at work boundaries; completed rows and their outputs are committed before progress increments. Partial results belong to the current run, never a leftover CSV from an earlier run.

Give each server launch a random token. Require it on API calls, validate local Host/Origin, and provide it through the launcher/browser bootstrap and private instance record for second-launch handoff. Keep it out of saved analysis sessions and exports.

## Delivery and checks

Implement and checkpoint in order: shared request/covariance; shared result identity, units and persistence; Compare/map consumers; batch manifest/cancellation; server token. Add portable synthetic regressions at each boundary, then run the broader suites and real browser workflows. Record completed work and evidence in the companion implementation log; this note does not claim that the changes are already implemented.
