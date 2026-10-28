// Fit-space/display-space mapping. The stored fit and its parameters never change.
export const isNormalizedT1 = (fit) => fit?.extras?.t1_prep_mode === 'normalized_1_minus_data';
export const hasPlotTransform = (fit) => !!fit?.plot_y && (isNormalizedT1(fit) || !fit.extras?.t1_prep_mode);
export function t1Normalization(fit) {
  const e = fit?.extras || {};
  return isNormalizedT1(fit) && Number.isFinite(e.t1_norm_lo) && Number.isFinite(e.t1_norm_span) && e.t1_norm_span > 0
    ? { lo: e.t1_norm_lo, span: e.t1_norm_span, source: e.t1_norm_source, stored: true } : null;
}
export function displaySpace(f) {
  const fit = f.fit[f.detector];
  if (isNormalizedT1(fit)) return f.display?.t1 === 'normalized' || !t1Normalization(fit) ? 'normalized' : 'original';
  return hasPlotTransform(fit) ? 'fit' : 'original';
}
export const toNormalized = (y, n) => y.map((v) => v === null ? NaN : 1 - (v - n.lo) / n.span);
export const toOriginal = (y, n) => y.map((v) => v === null ? NaN : n.lo + n.span * (1 - v));
const numbers = (a) => (a || []).map((v) => v === null ? NaN : v);

// Without a normalized fit, use the same base choice and span floor as _prepare_t1_series.
export function normalizationFromData(d) {
  const xs = d?.proc?.fit_x || d?.proc?.x || [];
  const y = (d?.proc?.fit_y || d?.proc?.y || []).filter((v, i) => Number.isFinite(v) && Number.isFinite(xs[i]));
  const base = y;
  if (!base.length) return null;
  const lo = base.reduce((a, b) => Math.min(a, b), Infinity), hi = base.reduce((a, b) => Math.max(a, b), -Infinity);
  const span = Math.max(hi - lo, 1e-9);
  return { lo, span, source: 'y', stored: false };
}

export function displaySeries(f, det, space, mainFit = null) {
  const d = f.data[det], fit = f.fit[det];
  if (!d?.proc) return null;
  const n = t1Normalization(fit) || (!isNormalizedT1(fit) ? normalizationFromData(d) : null);
  let x = numbers(d.proc.x), y = numbers(d.proc.y), note = '';
  if (space === 'normalized') {
    if (!n && !isNormalizedT1(fit)) return null;
    if (isNormalizedT1(fit)) {
      if (n && fit.provenance?.fit_target === 'smoothed') y = toNormalized(y, n); // Keep measured mean distinct from the smoothed fit target.
      else { x = numbers(fit.x); y = numbers(fit.plot_y); }
    }
    else { y = toNormalized(n.source === 'trace.y' ? numbers(d.raw.trace_y || d.raw.y) : y, n); }
    note = n?.stored ? 'own fit normalization' : n ? 'own data min–max; not a normalized fit' : 'legacy normalized fit; refit for original units';
  } else if (space === 'fit') {
    // An unknown plot_y transform is only comparable to an explicitly matching fitted space.
    if (!hasPlotTransform(fit) || fit.plot_y_label !== mainFit?.plot_y_label) return null;
    x = numbers(fit.x); y = numbers(fit.plot_y); note = fit.plot_y_label || 'fit space';
  }
  let fitY = fit ? numbers(fit.dense?.y || fit.y_fit) : null;
  let residuals = fit ? numbers(fit.residuals) : null;
  let band = fit?.dense?.band ? numbers(fit.dense.band) : null;
  if (fitY && isNormalizedT1(fit) && space === 'original') {
    if (!n?.stored) { fitY = null; band = null; residuals = null; }
    else {
      fitY = toOriginal(fitY, n);
      residuals = residuals.map((v) => -n.span * v);
      band = band?.map((v) => n.span * v);
      note = `fit mapped back from normalized space (${n.source})`;
    }
  } else if (fitY && !isNormalizedT1(fit) && space === 'normalized') {
    fitY = toNormalized(fitY, n);
    residuals = residuals.map((v) => -v / n.span);
    band = band?.map((v) => v / n.span);
  }
  return { x, y, fitY, residuals, band, note, normalization: n };
}

export function displayProvenance(f) {
  const fit = f.fit[f.detector];
  if (!isNormalizedT1(fit)) return hasPlotTransform(fit) ? `Display: ${fit.plot_y_label || 'fit space'}. Only matching fit-space overlays are shown.` : '';
  const n = t1Normalization(fit);
  if (!n) return 'Legacy normalized fit has no inverse metadata: showing normalized space. Refit to use original units.';
  return displaySpace(f) === 'original'
    ? `Original ${f.mode} units; fit and ±1σ mapped back with lo + span × (1 − fit). Base: ${n.source}. A and y0 remain normalized.`
    : `Normalized 1 − data; each file uses its own lo/span. A and y0 are normalized; T1 stays in the file’s x units.`;
}
