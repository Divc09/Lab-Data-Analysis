// 1D figure builder (pulse experiments, line scans) and time-trace figure builder.
import { S, detLabel, plotFile } from './state.js';
import { PALETTES, baseStyle, baseLayout, axisLayout, legendLayout, scatterTrace, resultsAnnotation, userDecorations, friendlyAxis, MODE_LABEL, seriesStyle } from './plotstyle.js';

import { displaySpace, displaySeries, isNormalizedT1, isDirectT1, toOriginal, toNormalized, normScale } from './fitdisplay.js';
import { xDisplay, resolveXAxis } from './unitdisplay.js';
import { comparisonFiles, filePresentation, legendLabel } from './seriescontrol.js';
import { uniformSampling, magnitudeSpectrum } from './spectrum.js';
import { cmapColor, ITER_SCALES, DEFAULT_ITER_SCALE } from './plot2d.js';

const nonNull = (a) => (a || []).map((v) => (v === null ? NaN : v));

export function fftAvailability(f) {
  const proc = f.data[f.detector]?.proc;
  const processed = uniformSampling(proc?.x, proc?.y);
  if (!processed.ok) return processed;
  const shown = displaySeries(f, f.detector, displaySpace(f), f.fit[f.detector]);
  return uniformSampling(shown?.x, shown?.y);
}

// Legend-free colourbar mapping iteration number (1..count) to the trace colour scale.
function iterColorbar(cmap, count, st) {
  return { type: 'scatter', mode: 'markers', x: [null], y: [null], showlegend: false, hoverinfo: 'skip', meta: { key: 'iterScale' },
    marker: { color: [1, count], cmin: 1, cmax: Math.max(2, count), colorscale: Array.from({ length: 17 }, (_, i) => [i / 16, cmapColor(cmap, i / 16)]), showscale: true,
      colorbar: { title: { text: 'Iteration', side: 'right' }, thickness: 10, len: 0.45, y: 0.5, x: 1.01, xpad: 4, outlinewidth: 0, tickfont: { size: Math.max(9, st.base - 3) } } } };
}

function modeFor(f, styleMode) {
  return styleMode === 'pts' ? 'markers' : styleMode === 'line' ? 'lines' : 'lines+markers';
}

export function yTitle(f) {
  const d = f.data[f.detector];
  const fit = f.fit[f.detector];
  const decay = isDirectT1(fit) && displaySpace(f) === 'normalized';
  if (displaySpace(f) !== 'original' && fit?.plot_y_label && !decay) return displayOf(f).inv ? `${fit.plot_y_label} (inverted)` : fit.plot_y_label;
  const unit = f.info.unit ? ` [${f.info.unit}]` : ' [a.u.]';
  const base = f.mode === 'contrast' ? 'Contrast [dimensionless]' : `${MODE_LABEL[f.mode] || f.mode}${unit}`;
  const shown = decay && fit.extras.t1_prep_mode !== 'direct_decay' ? `1 − ${base}` : base;
  return displayOf(f).inv ? `${shown} (inverted)` : shown;
}

const displayOf = (f) => {
  const d = f.display || {};
  return { inv: !!d.invert, xs: Number.isFinite(d.xscale) ? d.xscale : 1, xo: Number.isFinite(d.xoffset) ? d.xoffset : 0,
    cal: Number.isFinite(d.xscale) || Number.isFinite(d.xoffset) };
};

export function xTitle(f) {
  const d = xDisplay(f, comparisonFiles(f).map(plotFile));
  const t = friendlyAxis(f.info.x_label, d.unit, f.info.type);
  return d.calibrated ? `${t} (calibrated)` : t;
}

// Returns { data, layout } for Plotly plus helpers the view needs.
export function figure1D(f, { forExport = false } = {}) {
  // Resolve the display unit on the real file state: resolveXAxis records the unit in f.display and rescales
  // stored coordinates (pins, decorations, axis limits). On plotFile's shallow copy the record was lost while the
  // shared coordinate objects were still rescaled, so every redraw shrank e.g. pinned points toward x = 0.
  const xd = resolveXAxis(f, comparisonFiles(f).map(plotFile));
  f = plotFile(f);
  const st = baseStyle(f.style);
  const pal = PALETTES[st.palette] || PALETTES.lab;
  const L = f.layers;
  const det = f.detector;
  const d = f.data[det];
  if (!d || !d.proc) return null;
  const fit = f.fit[det];
  const traces = [];
  const fftReady = L.fft && fftAvailability(f).ok;
  const showRes = L.res && fit && !fftReady;
  const bottom = showRes || fftReady;
  const countsPanel = L.sigref && !f.info.reference_only;
  const domainMain = countsPanel ? (bottom ? [0.62, 1] : [0.42, 1]) : (bottom ? [0.3, 1] : [0, 1]);
  const mode = modeFor(f, f.styleMode);

  // display-only transforms (never touch stored data or fits): invert flips y, xscale/xoffset calibrate x
  const dsp = displayOf(f), primaryStyle = filePresentation(f, f);
  const TX = (a, unit = f.info.x_unit || '') => xd.transform(a, unit);
  const primaryName = legendLabel(f, f);
  const TY = (a) => (dsp.inv ? a.map((v) => -v) : a);
  const space = displaySpace(f), transformed = space !== 'original';
  const shown = displaySeries(f, det, space, fit);
  const xr = shown.x, yr = shown.y;
  const x = TX(xr), y = TY(yr);

  // uncertainty
  const semByX = new Map((d.proc.x || []).map((v, i) => [v, d.proc.sem?.[i]]));
  const sem = d.proc.sem && (space === 'original' || (space === 'normalized' && shown.normalization))
    ? xr.map((v) => { const s = semByX.get(v); return s == null ? NaN : s / (space === 'normalized' ? normScale(shown.normalization) : 1); }) : null;
  if (L.band && sem) {
    if (f.errorMode === 'bars') {
      // drawn on the data trace below
    } else {
      const s = seriesStyle(st, 'band', { name: '±1 SEM', color: pal[0] });
      traces.push({ type: 'scatter', x: [...x, ...[...x].reverse()], y: [...y.map((v, i) => v + sem[i]), ...y.map((v, i) => v - sem[i]).reverse()],
        fill: 'toself', fillcolor: hexA(s.color, 0.16), line: { width: 0 }, name: s.name, hoverinfo: 'skip', showlegend: s.legend, visible: s.visible === false ? 'legendonly' : true, meta: { key: 'band' } });
    }
  }
  // Iteration layers use the same selection, binning, exclusions and observable as the mean.
  // Nonlinear display steps have no propagated SD; do not put raw statistics on a transformed axis.
  const its = d.iterations;
  const hasSteps = (f.proc?.steps || []).some((s) => s.enabled !== false);
  const norm = shown.normalization;
  // 'gray' keeps the single series colour; any other scale colours traces by iteration number.
  const iterCmap = f.iterationView?.cmap === 'gray' ? null : ITER_SCALES[f.iterationView?.cmap] ? f.iterationView.cmap : DEFAULT_ITER_SCALE;
  const iterationY = (a) => TY(space === 'normalized' && norm ? toNormalized(nonNull(a), norm) : nonNull(a));
  if (its && (L.iter || L.meanSd) && !hasSteps && space !== 'fit' && (!transformed || norm)) {
    const ix = TX(nonNull(its.x || d.raw.x));
    const sel = its.selected ?? f.proc?.iterations ?? f.iterSel ?? its.values.map((_, i) => i);
    const vals = its.processed_values || its.values;
    if (L.iter) {
      const s = seriesStyle(st, 'iter', { name: `Selected iterations (${sel.length})`, color: '#8C95A4', width: iterCmap ? 1.2 : 1, opacity: iterCmap ? 0.85 : 0.6 });
      sel.forEach((k, j) => traces.push({ type: 'scatter', mode: 'lines', x: ix, y: iterationY(vals[k]),
        line: { color: iterCmap ? cmapColor(iterCmap, k / Math.max(1, vals.length - 1)) : s.color, width: s.width, dash: iterCmap ? 'solid' : ['solid', 'dot', 'dash'][k % 3] }, opacity: s.opacity,
        name: s.name, legendgroup: 'iter', showlegend: j === 0 && s.legend, hovertemplate: `iteration ${k + 1}: %{y:.5g}<extra></extra>`, meta: { key: 'iter', iteration: k + 1 }, visible: s.visible === false ? 'legendonly' : true }));
      if (iterCmap && sel.length) traces.push(iterColorbar(iterCmap, vals.length, st));
    }
    if (L.meanSd && d.proc.sd && sel.length > 1) {
      const mean = iterationY(d.proc.y), sd = nonNull(d.proc.sd).map((v) => space === 'normalized' ? v / normScale(norm) : v);
      const s = seriesStyle(st, 'meanSd', { name: 'Selected mean ± SD', color: pal[0] });
      traces.push({ type: 'scatter', x: [...ix, ...[...ix].reverse()], y: [...mean.map((v, i) => v + sd[i]), ...mean.map((v, i) => v - sd[i]).reverse()], fill: 'toself',
        fillcolor: hexA(s.color, 0.14), line: { width: 0 }, name: s.name, showlegend: s.legend, hoverinfo: 'skip', meta: { key: 'meanSd' } });
    }
  }
  if (L.iterFits && !hasSteps && space !== 'fit') {
    const result = f.iterationFits?.[det];
    if (result?.mode === f.mode) for (const row of result.rows) {
      const r = row.fit; if (!r || (its?.selected && !its.selected.includes(row.index))) continue;
      let fy = nonNull(r.dense?.y || r.y_fit);
      if (isNormalizedT1(r)) {
        const rn = r.extras;
        if (!Number.isFinite(rn.t1_norm_lo) || !Number.isFinite(rn.t1_norm_span)) continue;
        fy = toOriginal(fy, { lo: rn.t1_norm_lo, span: rn.t1_norm_span });
      }
      if (space === 'normalized') { if (!norm) continue; fy = toNormalized(fy, norm); }
      const nIt = its?.values.length || result.rows.length;
      traces.push({ type: 'scatter', mode: 'lines', x: TX(nonNull(r.dense?.x || r.x)), y: TY(fy),
        line: { color: iterCmap ? cmapColor(iterCmap, row.index / Math.max(1, nIt - 1)) : '#5B6474', width: 1.4, dash: ['dash', 'dot', 'dashdot'][row.index % 3] },
        name: `Individual fits${result.fit_spec.fit_smoothed ? ' · SMOOTHED' : ''}`, legendgroup: 'iterFits', showlegend: row === result.rows.find((r) => r.fit),
        hovertemplate: `iteration ${row.iteration} · ${row.status}<br>%{x:.5g} ${xd.unit}, %{y:.5g}<extra></extra>`, meta: { key: 'iterFits', iteration: row.iteration } });
    }
  }
  // overlays (other files, same observable)
  const rightSeries = [];
  S.overlays.filter((p) => p !== f.path && S.files[p]).forEach((p) => {
    const o = S.files[p];
    if (o.mode !== f.mode || o.info.dim === 'scan2d' || o.info.type === 'TimeTrace') return;
    const odet = o.info.detectors.includes(det) ? det : o.detector;
    const os = displaySeries(o, odet, space, fit);
    if (!os) return;
    const cfg = filePresentation(f, o), name = legendLabel(f, o);
    const ox = TX(os.x, o.info.x_unit || '');
    if (!ox) return; // Never overlay incompatible physical x dimensions.
    const label = `${name}${space !== 'original' ? ` · ${os.normalization?.stored ? 'own fit norm.' : 'own data norm.'}` : ''}`;
    const ofit = o.fit[odet], hasFit = cfg.fit && ofit && os.fitY;
    // Per-series axis (left / right overlay axis y9) and uncertainty style (none / band / bars) from the Series panel.
    const yx = cfg.axis === 'right' ? { yaxis: 'y9' } : {};
    if (cfg.axis === 'right') rightSeries.push(cfg.label);
    const od = o.data[odet], oy = TY(os.y);
    let osem = null;
    if (cfg.err && cfg.err !== 'none' && od?.proc?.sem && (space === 'original' || (space === 'normalized' && os.normalization))) {
      const m = new Map((od.proc.x || []).map((v, i) => [v, od.proc.sem[i]]));
      osem = os.x.map((v) => { const s = m.get(v); return s == null ? NaN : s / (space === 'normalized' ? normScale(os.normalization) : 1); });
    }
    if (osem && cfg.err === 'band' && cfg.data) traces.push({ type: 'scatter', x: [...ox, ...[...ox].reverse()], y: [...oy.map((v, i) => v + osem[i]), ...oy.map((v, i) => v - osem[i]).reverse()],
      fill: 'toself', fillcolor: hexA(cfg.color, 0.15), line: { width: 0 }, name: `${cfg.label} ±1 SEM`, showlegend: false, hoverinfo: 'skip', legendgroup: p, meta: { key: `ovband:${p}` }, ...yx });
    traces.push(scatterTrace(st, `ov:${p}`, { name: label, color: cfg.color, mode: cfg.mode, marker: cfg.marker, dash: cfg.dash }, ox, oy,
      { visible: cfg.data, showlegend: cfg.legend && (cfg.data || !hasFit), legendgroup: p,
        ...(osem && cfg.err === 'bars' ? { error_y: { type: 'data', array: osem, visible: true, thickness: 1, width: 2, color: hexA(cfg.color, 0.65) } } : {}), ...yx,
        hovertemplate: `${cfg.label} · ${detLabel(o, odet)} · ${o.mode}<br>${os.note}<br>%{x:.6g} ${xd.unit}, %{y:.6g}<extra></extra>` }));
    if (ofit && os.fitY && f.display?.overlayFits !== false) traces.push(scatterTrace(st, `ovfit:${p}`,
      { name: label, color: cfg.color, mode: 'lines', dash: cfg.fitDash, width: 2 }, TX(nonNull(ofit.dense?.x || ofit.x), o.info.x_unit || ''), TY(os.fitY),
      { visible: !!hasFit, showlegend: cfg.legend && !cfg.data && !!hasFit, legendgroup: p, ...yx }));
  });
  // second detector on the same plot
  if (f.compareDetectors && f.info.detectors.length > 1) {
    const other = f.info.detectors.find((k) => k !== det);
    const od = f.data[other];
    const os = displaySeries(f, other, space, fit);
    if (od?.proc && os) {
      const right = false; // Matching observables share physical units; a second detector does not need a second scale.
      traces.push(scatterTrace(st, `det:${other}`, { name: detLabel(f, other), color: pal[2], mode }, TX(os.x), TY(os.y), right ? { yaxis: 'y2' } : {}));
      const ofit = f.fit[other];
      if (ofit && L.fit && os.fitY) traces.push(scatterTrace(st, `fit:${other}`, { name: `Fit · ${detLabel(f, other)}`, color: pal[3], mode: 'lines', width: 2 },
        TX(nonNull(ofit.dense?.x || ofit.x)), TY(os.fitY), right ? { yaxis: 'y2' } : {}));
    }
  }
  // main data
  if (L.data) {
    const extra = { visible: primaryStyle.data, showlegend: primaryStyle.legend, legendgroup: f.path };
    if (L.band && sem && f.errorMode === 'bars') extra.error_y = { type: 'data', array: sem, visible: true, thickness: 1, width: 2, color: hexA(pal[0], 0.6) };
    const name = f.info.detectors.length > 1 ? `Data · ${detLabel(f, det)}` : (its ? `Data · mean of ${its.count} iterations` : 'Data');
    const sourceIndices = transformed ? new Map(d.proc.x.map((v, i) => [v, i])) : null;
    traces.push(scatterTrace(st, 'data', { name: primaryName, color: primaryStyle.color, mode: primaryStyle.mode, marker: primaryStyle.marker, dash: primaryStyle.dash }, x, y, { customdata: xr.map((v, i) => sourceIndices?.get(v) ?? i), ...extra }));
  }
  if (L.smooth && d.proc.smooth && (space === 'original' || (space === 'normalized' && norm))) traces.push(scatterTrace(st, 'smooth', { name: `Smoothed (${d.proc.smoothing_window ?? f.proc.smooth} pt)`, color: '#222222', mode: 'lines', width: 1.6 }, TX(nonNull(d.proc.x)), iterationY(d.proc.smooth)));
  // fit
  if (fit && L.fit && shown.fitY) {
    const fx = TX(nonNull(fit.dense?.x || fit.x)), fy = TY(shown.fitY);
    const fb = shown.band || [];
    if (L.conf && fit.dense?.band && fb.length === fx.length && fb.every(Number.isFinite)) {
      const s = seriesStyle(st, 'conf', { name: 'Model mean ±1σ', color: pal[1] });
      traces.push({ type: 'scatter', x: [...fx, ...[...fx].reverse()], y: [...fy.map((v, i) => v + fb[i]), ...fy.map((v, i) => v - fb[i]).reverse()],
        fill: 'toself', fillcolor: hexA(s.color, 0.18), line: { width: 0 }, name: s.name, hoverinfo: 'skip', showlegend: s.legend, visible: s.visible === false ? 'legendonly' : true, meta: { key: 'conf' } });
    }
    traces.push(scatterTrace(st, 'fit', { name: primaryName, color: primaryStyle.color, mode: 'lines', dash: primaryStyle.fitDash, width: Math.max(2, st.line + 0.4) }, fx, fy,
      { visible: primaryStyle.fit, showlegend: primaryStyle.legend && !primaryStyle.data, legendgroup: f.path, hovertemplate: `fit %{x:.6g} ${xd.unit}, %{y:.6g}<extra>${fit.model}</extra>` }));
    if (L.envelope && fit.envelope) {
      const s = seriesStyle(st, 'envelope', { name: 'Decay envelope', color: '#7A4FD6', dash: 'dash', width: 1.4 });
      traces.push({ type: 'scatter', mode: 'lines', x: TX(nonNull(fit.envelope.x)), y: TY(nonNull(fit.envelope.upper)), line: { color: s.color, dash: s.dash, width: s.width }, name: s.name, legendgroup: 'env', meta: { key: 'envelope' }, showlegend: s.legend });
      traces.push({ type: 'scatter', mode: 'lines', x: TX(nonNull(fit.envelope.x)), y: TY(nonNull(fit.envelope.lower)), line: { color: s.color, dash: s.dash, width: s.width }, name: s.name, legendgroup: 'env', showlegend: false, meta: { key: 'envelope' } });
    }
  }
  // Counts have their own aligned panel, never a contrast/normalized axis or an arbitrary overlaid scale.
  if (countsPanel) {
    const rx = TX(nonNull(d.raw.x));
    traces.push(scatterTrace(st, 'signal', { name: 'Signal (selected mean)', color: '#eb6834', mode: 'lines', width: 1.1 }, rx, nonNull(d.raw.signal), { yaxis: 'y2' }));
    traces.push(scatterTrace(st, 'reference', { name: 'Reference (selected mean)', color: '#1baf7a', mode: 'lines', width: 1.1 }, rx, nonNull(d.raw.reference), { yaxis: 'y2' }));
  }
  // pinned points
  if (f.pins.length) {
    traces.push({ type: 'scatter', mode: 'markers', x: f.pins.map((p) => p.x), y: f.pins.map((p) => p.y), name: 'Pinned', showlegend: false, hoverinfo: 'skip',
      marker: { size: 14, color: 'rgba(0,0,0,0)', line: { color: '#E0A400', width: 2 } }, meta: { key: 'pins' } });
  }
  // residuals
  if (showRes && L.data) {
    const rx = TX(nonNull(fit.x));
    traces.push({ type: 'scatter', mode: 'markers', x: rx, y: TY(shown.residuals || []), xaxis: 'x', yaxis: 'y3', name: 'Residuals', showlegend: false,
      marker: { size: Math.max(3, st.marker - 2), color: '#5B6474' }, hovertemplate: 'residual %{y:.4g}<extra></extra>', meta: { key: 'res' } });
  }
  // FFT of the processed data (regular sampling only)
  let fftInfo = null;
  if (fftReady) {
    fftInfo = fft(xr, yr, f.info.x_unit); // uncalibrated x keeps the frequency unit tied to x_unit
    traces.push({ type: 'scatter', mode: 'lines', x: fftInfo.f, y: fftInfo.mag, xaxis: 'x2', yaxis: 'y3', name: 'FFT', showlegend: false, line: { color: '#7A4FD6', width: 1.4 }, meta: { key: 'fft' } });
  }

  const deco = userDecorations(st);
  const annotations = [...deco.annotations];
  if (L.results) { const a = resultsAnnotation(st, fit, f.info); if (a) annotations.push(a); }
  if (isNormalizedT1(fit)) annotations.push({ xref: 'paper', yref: 'paper', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false,
    text: `${space === 'original' ? 'Original units · fit mapped from normalized space' : 'Normalized 1 − data · each file uses its own normalization'} · A, y0 normalized`, font: { size: Math.max(9, st.base - 3), color: '#5B6474' }, name: 'fit-space' });
  const decayView = isDirectT1(fit) && space === 'normalized';
  if (decayView) annotations.push({ xref: 'paper', yref: 'paper', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false,
    text: fit.extras.t1_prep_mode === 'direct_decay' ? 'Plotted as decay · data already decays, shown unchanged'
      : `Plotted as decay · 1 − ${f.mode}, no rescaling · fitted directly`, font: { size: Math.max(9, st.base - 3), color: '#5B6474' }, name: 'fit-space' });
  const shapes = [...deco.shapes];
  // ODMR peak-pick-only results (analyze.js peakPickCard): detected sample points, no model.
  const picked = f.peakPick?.[det];
  if (picked?.peaks?.length && L.picked !== false && !fit?.plot_y) {
    const px = TX(picked.peaks.map((p) => p.x)), py = TY(picked.peaks.map((p) => p.y));
    if (px) traces.push({ type: 'scatter', mode: 'markers', x: px, y: py, name: `Picked ${picked.polarity === 'dip' ? 'dips' : 'peaks'} (${picked.peaks.length})`,
      marker: { symbol: 'x-thin', size: 13, color: '#D62728', line: { width: 2.5, color: '#D62728' } }, meta: { key: 'picked' },
      hovertemplate: 'picked %{x:.7g}<extra></extra>' });
  }
  if (L.peaks && fit) fit.params.filter((p) => /^(x0|center)(_\d+)?$/.test(p.name)).forEach((p) =>
    shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: TX([p.value])[0], x1: TX([p.value])[0], y0: domainMain[0], y1: 1, line: { color: '#E06A1B', width: 1, dash: 'dot' } }));
  if (f.measure && f.measure.length === 2) {
    const [a, b] = f.measure;
    shapes.push({ type: 'line', x0: a.x, y0: a.y, x1: b.x, y1: b.y, line: { color: '#C0392B', width: 1.5, dash: 'dot' } });
  }
  const range = fit ? TX([fit.x?.[0], fit.x?.[fit.x.length - 1]]).sort((a, b) => a - b) : null;
  const hasCounts = countsPanel;
  const layout = baseLayout(st, {
    xaxis: { ...axisLayout(st, 'x', xTitle(f), { range: f.xrange }), domain: [0, 1], anchor: countsPanel ? 'y2' : 'y' },
    yaxis: { ...axisLayout(st, 'y', yTitle(f), { range: f.yrange }), domain: domainMain, ...(f.mode !== 'contrast' && !transformed ? { exponentformat: 'SI' } : {}) },
    legend: legendLayout(st),
    showlegend: L.legend,
    annotations, shapes,
    uirevision: f.path + det + f.mode + space + xd.unit + !!st.x.log,
  });
  if (isNormalizedT1(fit) || decayView) layout.margin.t = Math.max(layout.margin.t, 38);
  if (rightSeries.length) {
    // Independent right-hand axis for series the user moved there (Series panel); the left axis stops mirroring ticks.
    layout.yaxis9 = { ...axisLayout(st, 'yRight', `${yTitle(f)} · right: ${rightSeries.join(', ')}`, { range: f.yRightRange }), overlaying: 'y', side: 'right', anchor: 'x', ...(st.yRight?.ticks?.length ? {} : { tickmode: 'auto', nticks: 8 }),
      showgrid: false, mirror: false, ...(f.mode !== 'contrast' && !transformed ? { exponentformat: 'SI' } : {}) };
    layout.yaxis.mirror = false;
    layout.margin.r = Math.max(layout.margin.r || 0, 78);
  }
  if (countsPanel) layout.yaxis2 = { ...axisLayout(st, 'y2', `Signal / Reference<br>[${f.info.unit || 'recorded units'}]`), domain: bottom ? [0.31, 0.51] : [0, 0.27], anchor: 'x', exponentformat: 'SI' };
  if (fit?.provenance?.fit_target === 'smoothed') {
    annotations.push({ xref: 'paper', yref: 'paper', x: 1, y: 1.02, xanchor: 'right', yanchor: 'bottom', showarrow: false,
      text: `SMOOTHED FIT · ${fit.provenance.smoothing_window} pt · errors underestimated`, font: { size: Math.max(9, st.base - 3), color: '#5B6474' } });
    layout.margin.t = Math.max(layout.margin.t, 55);
  }
  if (hasSteps && (L.iter || L.meanSd || L.iterFits)) annotations.push({ xref: 'paper', yref: 'paper', x: 0, y: 1.02, showarrow: false, xanchor: 'left', yanchor: 'bottom', text: 'Iteration layers hidden: transformed-axis statistics are not propagated.', font: { size: 10, color: '#5B6474' } });
  if (bottom) {
    if (fftInfo) {
      layout.xaxis2 = { ...axisLayout(st, 'x2', `Frequency [${fftInfo.unit}]`), anchor: 'y3', domain: [0, 1] };
      layout.yaxis3 = { ...axisLayout(st, 'y3', '|FFT|'), domain: [0, 0.2], anchor: 'x2' };
      layout.xaxis.title.text = layout.xaxis.title.text; // keep
    } else {
      layout.yaxis3 = { ...axisLayout(st, 'y3', space === 'normalized' && !decayView ? 'Resid.<br>[a.u.]' : space === 'fit' ? `Resid. · ${fit.plot_y_label || 'fit space'}` : `Resid.<br>[${f.mode === 'contrast' ? '1' : f.info.unit || 'a.u.'}]`), domain: [0, 0.2], anchor: 'x', zeroline: true, zerolinecolor: '#C9CED6' };
      layout.xaxis.anchor = 'y3';
      layout.yaxis.anchor = 'x';
    }
  }
  const xbreakBlocked = bottom ? 'residual panel or FFT' : hasCounts ? 'signal/reference counts panel' : '';
  const fileOrder = comparisonFiles(f).map(plotFile).map((o) => o.path), explicitOrder = st.seriesOrder || [];
  const rank = (t) => {
    const key = t.meta?.key;
    if (explicitOrder.includes(key)) return explicitOrder.indexOf(key);
    const path = key === 'data' || key === 'fit' ? f.path : key?.replace(/^ovfit:|^ov:/, '');
    return explicitOrder.length + (fileOrder.includes(path) ? fileOrder.indexOf(path) * 2 + (key === 'fit' || key?.startsWith('ovfit:') ? 1 : 0) : fileOrder.length * 2);
  };
  traces.sort((a, b) => rank(a) - rank(b));
  applyInset(f, st, traces, layout, shapes, annotations, domainMain, 1);
  applyXBreak(f, st, traces, layout, shapes, annotations, domainMain, xbreakBlocked);
  if (forExport) layout.uirevision = undefined;
  return { data: traces, layout, range, xbreakBlocked, fftPeak: fftInfo ? { f: fftInfo.peak, unit: fftInfo.unit } : null };
}

// ---- presentation extras: inset zoom (xaxis4/yaxis4) and broken x axis (xaxis5) ----
const mainAx = (t) => (!t.xaxis || t.xaxis === 'x') && (!t.yaxis || t.yaxis === 'y');
const NO_INSET = new Set(['band', 'conf', 'meanSd', 'iter', 'iterScale', 'iterFits', 'pins', 'res', 'fft']);

function applyInset(f, st, traces, layout, shapes, annotations, dom, xEnd) {
  const ins = f.style?.inset;
  if (!ins?.enabled || !(ins.x1 > ins.x0)) return;
  const { x0, x1 } = ins;
  const src = traces.filter((t) => mainAx(t) && !NO_INSET.has(t.meta?.key));
  const copies = [];
  let lo = Infinity, hi = -Infinity;
  for (const t of src) {
    if (!t.x || !t.y || t.fill) continue;
    const xs = [], ys = [];
    for (let i = 0; i < t.x.length; i++) {
      const inR = t.x[i] >= x0 && t.x[i] <= x1;
      const near = (t.x[i + 1] >= x0 && t.x[i + 1] <= x1) || (t.x[i - 1] >= x0 && t.x[i - 1] <= x1);
      if (inR || near) { xs.push(t.x[i]); ys.push(t.y[i]); }
      if (inR && Number.isFinite(t.y[i])) { lo = Math.min(lo, t.y[i]); hi = Math.max(hi, t.y[i]); }
    }
    if (!xs.length) continue;
    const c = { ...t, x: xs, y: ys, xaxis: 'x4', yaxis: 'y4', showlegend: false, customdata: undefined, error_y: undefined };
    copies.push(c);
  }
  if (!copies.length) return;
  let y0, y1;
  if (ins.y && Number.isFinite(ins.y.y0) && Number.isFinite(ins.y.y1)) { y0 = ins.y.y0; y1 = ins.y.y1; }
  else { const pad = (hi - lo || 1) * 0.08; y0 = lo - pad; y1 = hi + pad; }
  if (!Number.isFinite(y0) || !Number.isFinite(y1)) return;
  traces.push(...copies);
  const span = dom[1] - dom[0];
  const w = Math.min(0.7, Math.max(0.1, ins.size || 0.35)) * xEnd, hh = Math.min(0.7, Math.max(0.1, ins.size || 0.35)) * 1.15 * span;
  const pos = ins.pos || 'ur';
  const xa = pos[1] === 'r' ? xEnd - 0.02 - w : 0.085;
  const ya = pos[0] === 'u' ? dom[1] - 0.03 * span - hh : dom[0] + 0.13 * span;
  const small = Math.round(st.base * 0.72);
  const base = (ax) => ({ ...axisLayout(st, ax, ''), title: { text: '' }, automargin: false, showgrid: false, mirror: 'ticks', nticks: 4, tickfont: { size: small }, exponentformat: 'SI' });
  layout.xaxis4 = { ...base('x4'), domain: [xa, xa + w], anchor: 'y4', range: [x0, x1], autorange: false };
  layout.yaxis4 = { ...base('y4'), domain: [ya, ya + hh], anchor: 'x4', range: [y0, y1], autorange: false };
  // white backing (covers main traces and tick-label margin) and the zoom-region marker on the main plot
  shapes.unshift({ type: 'rect', xref: 'paper', yref: 'paper', x0: xa - 0.08, x1: xa + w + 0.012, y0: ya - 0.12 * span, y1: ya + hh + 0.012 * span, fillcolor: '#FFFFFF', line: { width: 0 }, layer: 'below' });
  shapes.push({ type: 'rect', xref: 'x', yref: 'y', x0, x1, y0, y1, line: { color: '#5B6474', width: 1, dash: 'dash' }, fillcolor: 'rgba(0,0,0,0)', layer: 'above' });
  layout.shapes = shapes;
}

function applyXBreak(f, st, traces, layout, shapes, annotations, dom, blocked) {
  const b = f.style?.xbreak;
  if (!b?.enabled || blocked || !(b.to > b.from)) return;
  let lo = st.x?.min ?? f.xrange?.[0], hi = st.x?.max ?? f.xrange?.[1];
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = Infinity; hi = -Infinity;
    for (const t of traces) if (mainAx(t) && t.x) for (const v of t.x) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  const { from, to } = b;
  if (!(lo < from && to < hi)) return;
  const gap = 0.02, avail = 1 - gap;
  const lw = Math.min(0.85, Math.max(0.15, avail * (from - lo) / ((from - lo) + (hi - to))));
  const title = layout.xaxis.title?.text ?? '';
  const keep = { ...layout.xaxis };
  layout.xaxis = { ...keep, domain: [0, lw], range: [lo, from], autorange: false, title: { text: '' } };
  layout.xaxis5 = { ...keep, domain: [lw + gap, 1], range: [to, hi], autorange: false, title: { text: '' }, anchor: 'y' };
  const inPanel = (x0, x1, a, c) => x1 >= a && x0 <= c;
  const dupT = [];
  for (const t of traces) if (mainAx(t)) dupT.push({ ...t, xaxis: 'x5', showlegend: false });
  traces.push(...dupT);
  // shapes / annotations bound to x: left panel keeps those in range, right panel gets a copy where they fall in its range
  const newShapes = [];
  for (const s of shapes) {
    if ((s.xref ?? 'x') !== 'x' || s.x0 === undefined) { newShapes.push(s); continue; }
    const a = Math.min(s.x0, s.x1 ?? s.x0), c = Math.max(s.x0, s.x1 ?? s.x0);
    if (inPanel(a, c, lo, from)) newShapes.push(s);
    if (inPanel(a, c, to, hi)) newShapes.push({ ...s, xref: 'x5', name: undefined });
  }
  const px = [lw, lw + gap], yb = dom[0], tick = 0.018;
  for (const p of px) newShapes.push({ type: 'line', xref: 'paper', yref: 'paper', x0: p - 0.007, x1: p + 0.007, y0: yb - tick, y1: yb + tick, line: { color: '#1D2330', width: 1.2 } });
  if (st.mirror) newShapes.push({ type: 'line', xref: 'paper', yref: 'paper', x0: 1, x1: 1, y0: dom[0], y1: dom[1], line: { color: '#1D2330', width: 1 } },
    { type: 'line', xref: 'paper', yref: 'paper', x0: lw + gap, x1: lw + gap, y0: dom[0], y1: dom[1], line: { color: '#1D2330', width: 1 } });
  layout.shapes = newShapes;
  const anns = [];
  for (const a of annotations) {
    if ((a.xref ?? 'x') !== 'x' || a.x === undefined) { anns.push(a); continue; }
    if (a.x >= lo && a.x <= from) anns.push(a);
    else if (a.x >= to && a.x <= hi) anns.push({ ...a, xref: 'x5' });
  }
  anns.push({ xref: 'paper', yref: 'paper', x: 0.5, y: dom[0], yanchor: 'top', yshift: -Math.round(st.base * 2.7), text: title, showarrow: false, font: { size: st.base } });
  layout.annotations = anns;
}

// Frequency axis units follow the x unit: ns -> MHz, us -> MHz, s -> Hz, otherwise cycles per x unit ("1/GHz", "1/µm").
function fftUnit(xu) {
  const u = (xu || '').trim();
  if (u === 'ns') return { scale: 1000, unit: 'MHz' };
  if (u === 'µs' || u === 'us') return { scale: 1, unit: 'MHz' };
  if (u === 's') return { scale: 1, unit: 'Hz' };
  return { scale: 1, unit: u ? `1/${u}` : '1/x unit' };
}

function fft(x, y, xUnit) {
  const { scale, unit } = fftUnit(xUnit);
  const spectrum = magnitudeSpectrum(x, y);
  return { ...spectrum, f: spectrum.f.map((v) => v * scale), peak: spectrum.peak === null ? null : spectrum.peak * scale, unit };
}

export function hexA(hex, a) {
  const c = hex.replace('#', '');
  const r = parseInt(c.slice(0, 2), 16), g = parseInt(c.slice(2, 4), 16), b = parseInt(c.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

// ---------------- time traces: one panel per detector, shared time axis ----------------
export function rolling(x, y, w) {
  const out = new Array(y.length);
  let lo = 0, hi = 0, sum = 0, cnt = 0;
  for (let i = 0; i < x.length; i++) {
    while (hi < x.length && x[hi] <= x[i] + w / 2) { if (Number.isFinite(y[hi])) { sum += y[hi]; cnt++; } hi++; }
    while (x[lo] < x[i] - w / 2) { if (Number.isFinite(y[lo])) { sum -= y[lo]; cnt--; } lo++; }
    out[i] = cnt ? sum / cnt : NaN;
  }
  return out;
}

export function figureTT(f) {
  f = plotFile(f);
  const st = baseStyle(f.style);
  const pal = PALETTES[st.palette] || PALETTES.lab;
  const tt = f.tt;
  const scale = tt.unit === 'min' ? 1 / 60 : 1;
  const dets = tt.layout === 'single' ? [f.detector] : f.info.detectors;
  const layoutKind = dets.length === 1 ? 'single' : tt.layout;
  const traces = [];
  const annotations = [], shapes = [];
  const n = layoutKind === 'stacked' ? dets.length : 1;
  const gap = 0.07;
  const h = (1 - gap * (n - 1)) / n;
  const layout = baseLayout(st, { legend: legendLayout(st), showlegend: f.layers.legend, uirevision: f.path + tt.layout });
  const detColors = [pal[0], pal[2], pal[3]];
  dets.forEach((det, k) => {
    const d = f.data[det];
    if (!d?.proc) return;
    const panel = layoutKind === 'stacked' ? k : 0;
    const yax = layoutKind === 'right' && k === 1 ? 'y2' : (panel === 0 ? 'y' : `y${panel + 2}`);
    const x = nonNull(d.proc.x).map((v) => v * scale);
    let y = nonNull(d.proc.y);
    if (tt.norm === 'start' || layoutKind === 'percent') {
      const a = tt.windows.a || [0, 60];
      const base = mean(y.filter((_, i) => d.proc.x[i] >= a[0] && d.proc.x[i] <= a[1]));
      y = y.map((v) => 100 * v / base);
    }
    const color = detColors[k % detColors.length];
    traces.push(scatterTrace(st, `tt:${det}`, { name: detLabel(f, det), color, mode: f.styleMode === 'both' ? 'lines' : (f.styleMode === 'pts' ? 'markers' : 'lines'), width: 1, size: 4 }, x, y, { yaxis: yax }));
    if (tt.showRolling) traces.push(scatterTrace(st, `roll:${det}`, { name: `Rolling mean ${tt.rolling} s · ${detLabel(f, det)}`, color: '#1D2330', mode: 'lines', width: 1.6 }, x, rolling(nonNull(d.proc.x), y, tt.rolling), { yaxis: yax }));
    const fit = f.fit[det];
    if (fit && f.layers.fit && tt.norm === 'none' && layoutKind !== 'percent') {
      traces.push(scatterTrace(st, `fit:${det}`, { name: `Decay fit · ${detLabel(f, det)}`, color: pal[1], mode: 'lines', width: 2, dash: k ? 'dash' : 'solid' },
        nonNull(fit.dense?.x || fit.x).map((v) => v * scale), nonNull(fit.dense?.y || fit.y_fit), { yaxis: yax }));
    }
    const unit = f.info.unit ? ` [${f.info.unit}]` : '';
    const title = layoutKind === 'percent' || tt.norm === 'start' ? `% of window A` : `${detLabel(f, det)}${unit}`;
    const axisName = yax === 'y' ? 'yaxis' : `yaxis${yax.slice(1)}`;
    if (!layout[axisName]) {
      const dom = layoutKind === 'stacked' ? [1 - (panel + 1) * h - panel * gap, 1 - panel * h - panel * gap] : [0, 1];
      layout[axisName] = { exponentformat: 'SI', ...axisLayout(st, yax === 'y2' && layoutKind === 'right' ? 'y2' : 'y', layoutKind === 'shared' ? `${f.info.y_label || 'Fluorescence'}${unit}` : title), domain: dom, anchor: 'x', exponentformat: 'SI' };
      if (yax === 'y2') Object.assign(layout[axisName], { overlaying: 'y', side: 'right' });
    }
    if (tt.showRetries) for (const r of d.time?.retries || []) {
      shapes.push({ type: 'line', xref: 'x', yref: `${yax === 'y' ? 'y' : yax} domain`, x0: r.t * scale, x1: r.t * scale, y0: 0, y1: 0.06, line: { color: '#C0392B', width: 2 } });
    }
  });
  layout.xaxis = { ...axisLayout(st, 'x', `Elapsed time [${tt.unit}], zero at first measurement midpoint`, { range: f.xrange }), anchor: layoutKind === 'stacked' && n > 1 ? `y${n + 1}` : 'y', domain: [0, layoutKind === 'right' ? 0.92 : 1] };
  if (layoutKind === 'stacked') for (let p = 1; p < n; p++) layout[`yaxis${p + 2}`] && (layout[`yaxis${p + 2}`].anchor = 'x');
  if (tt.showWindows) for (const [key, w] of Object.entries(tt.windows)) {
    if (!w) continue;
    shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: w[0] * scale, x1: w[1] * scale, y0: 0, y1: 1, fillcolor: '#7A4FD6', opacity: 0.08, line: { width: 0 }, layer: 'below' });
    annotations.push({ x: (w[0] + w[1]) / 2 * scale, xref: 'x', y: 1, yref: 'paper', yanchor: 'bottom', text: key.toUpperCase(), showarrow: false, font: { color: '#5B3FB0', size: 12 } });
  }
  if (tt.showEvents) for (const ev of f.data[f.detector]?.time?.events || []) {
    const s = Number(ev.start ?? ev.startSeconds ?? ev[0]), e = Number(ev.end ?? ev.endSeconds ?? ev[1]);
    if (Number.isFinite(s)) shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: s * scale, x1: (Number.isFinite(e) ? e : s) * scale, y0: 0, y1: 1, fillcolor: '#E0A400', opacity: 0.15, line: { width: 0 }, layer: 'below' });
  }
  const deco = userDecorations(st);
  layout.shapes = [...shapes, ...deco.shapes];
  layout.annotations = [...annotations, ...deco.annotations];
  return { data: traces, layout };
}

const mean = (a) => { const v = a.filter(Number.isFinite); return v.reduce((p, q) => p + q, 0) / (v.length || 1); };
