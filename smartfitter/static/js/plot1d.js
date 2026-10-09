// 1D figure builder (pulse experiments, line scans) and time-trace figure builder.
import { S, detLabel } from './state.js';
import { PALETTES, baseStyle, baseLayout, axisLayout, legendLayout, scatterTrace, resultsAnnotation, userDecorations, friendlyAxis, MODE_LABEL, seriesStyle } from './plotstyle.js';

const nonNull = (a) => (a || []).map((v) => (v === null ? NaN : v));

function modeFor(f, styleMode) {
  return styleMode === 'pts' ? 'markers' : styleMode === 'line' ? 'lines' : 'lines+markers';
}

export function yTitle(f) {
  const d = f.data[f.detector];
  const fit = f.fit[f.detector];
  if (fit?.plot_y_label && fit.plot_y) return displayOf(f).inv ? `${fit.plot_y_label} (inverted)` : fit.plot_y_label;
  const unit = f.info.unit ? ` [${f.info.unit}]` : '';
  const base = f.mode === 'contrast' ? 'Contrast' : `${MODE_LABEL[f.mode] || f.mode}${unit}`;
  return displayOf(f).inv ? `${base} (inverted)` : base;
}

const displayOf = (f) => {
  const d = f.display || {};
  return { inv: !!d.invert, xs: Number.isFinite(d.xscale) ? d.xscale : 1, xo: Number.isFinite(d.xoffset) ? d.xoffset : 0,
    cal: Number.isFinite(d.xscale) || Number.isFinite(d.xoffset) };
};

export function xTitle(f) {
  const t = friendlyAxis(f.info.x_label, f.info.x_unit);
  return displayOf(f).cal ? `${t} (calibrated)` : t;
}

// Returns { data, layout } for Plotly plus helpers the view needs.
export function figure1D(f, { forExport = false } = {}) {
  const st = baseStyle(f.style);
  const pal = PALETTES[st.palette] || PALETTES.lab;
  const L = f.layers;
  const det = f.detector;
  const d = f.data[det];
  if (!d || !d.proc) return null;
  const fit = f.fit[det];
  const traces = [];
  const showRes = L.res && fit && !L.fft;
  const bottom = showRes || (L.fft && !f.info.irregular);
  const domainMain = bottom ? [0.3, 1] : [0, 1];
  const mode = modeFor(f, f.styleMode);

  // display-only transforms (never touch stored data or fits): invert flips y, xscale/xoffset calibrate x
  const dsp = displayOf(f);
  const TX = (a) => (dsp.cal ? a.map((v) => v * dsp.xs + dsp.xo) : a);
  const TY = (a) => (dsp.inv ? a.map((v) => -v) : a);
  let xr = nonNull(d.proc.x), yr = nonNull(d.proc.y);
  if (fit?.plot_y) { xr = nonNull(fit.x); yr = nonNull(fit.plot_y); }
  const x = TX(xr), y = TY(yr);

  // uncertainty
  const sem = !fit?.plot_y && d.proc.sem ? nonNull(d.proc.sem) : null;
  if (L.band && sem) {
    if (f.errorMode === 'bars') {
      // drawn on the data trace below
    } else {
      const s = seriesStyle(st, 'band', { name: '±1 SEM', color: pal[0] });
      traces.push({ type: 'scatter', x: [...x, ...[...x].reverse()], y: [...y.map((v, i) => v + sem[i]), ...y.map((v, i) => v - sem[i]).reverse()],
        fill: 'toself', fillcolor: hexA(s.color, 0.16), line: { width: 0 }, name: s.name, hoverinfo: 'skip', showlegend: s.legend, visible: s.visible === false ? 'legendonly' : true, meta: { key: 'band' } });
    }
  }
  // iterations
  const its = d.iterations;
  if (its && (L.iter || L.meanSd) && !fit?.plot_y) {
    const rawX = TX(nonNull(d.raw.x));
    const sel = f.iterSel && f.iterSel.length ? f.iterSel : its.values.map((_, i) => i);
    if (L.iter) {
      const s = seriesStyle(st, 'iter', { name: `Iterations (${sel.length})`, color: '#8C95A4', width: 0.8, opacity: 0.45 });
      sel.forEach((k, j) => traces.push({ type: 'scatter', mode: 'lines', x: rawX, y: TY(nonNull(its.values[k])), line: { color: s.color, width: s.width }, opacity: s.opacity,
        name: s.name, legendgroup: 'iter', showlegend: j === 0 && s.legend, hovertemplate: `iteration ${k + 1}: %{y:.5g}<extra></extra>`, meta: { key: 'iter' }, visible: s.visible === false ? 'legendonly' : true }));
    }
    if (L.meanSd) {
      const n = rawX.length, mean = [], sd = [];
      for (let i = 0; i < n; i++) {
        const v = sel.map((k) => (dsp.inv ? -its.values[k][i] : its.values[k][i])).filter((q) => q !== null && Number.isFinite(q));
        const m = v.reduce((a, b) => a + b, 0) / (v.length || 1);
        mean.push(m); sd.push(v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)) : 0);
      }
      const s = seriesStyle(st, 'meanSd', { name: 'Iteration mean ± SD', color: '#7A4FD6' });
      traces.push({ type: 'scatter', x: [...rawX, ...[...rawX].reverse()], y: [...mean.map((v, i) => v + sd[i]), ...mean.map((v, i) => v - sd[i]).reverse()], fill: 'toself',
        fillcolor: hexA(s.color, 0.14), line: { width: 0 }, name: `${s.name} (band)`, showlegend: false, hoverinfo: 'skip', meta: { key: 'meanSd' } });
      traces.push({ type: 'scatter', mode: 'lines', x: rawX, y: mean, line: { color: s.color, width: 1.2 }, name: s.name, meta: { key: 'meanSd' }, showlegend: s.legend });
    }
  }
  // overlays (other files, same observable)
  S.overlays.filter((p) => p !== f.path && S.files[p]).forEach((p, i) => {
    const o = S.files[p];
    const od = o.data[o.info.detectors.includes(det) ? det : o.detector];
    if (!od?.proc) return;
    traces.push(scatterTrace(st, `ov:${p}`, { name: o.info.name.replace(/\.mat$/, ''), color: pal[(i + 2) % pal.length], mode }, TX(nonNull(od.proc.x)), TY(nonNull(od.proc.y))));
  });
  // second detector on the same plot
  if (f.compareDetectors && f.info.detectors.length > 1) {
    const other = f.info.detectors.find((k) => k !== det);
    const od = f.data[other];
    if (od?.proc) {
      const right = f.mode !== 'contrast';
      traces.push(scatterTrace(st, `det:${other}`, { name: detLabel(f, other), color: pal[2], mode }, TX(nonNull(od.proc.x)), TY(nonNull(od.proc.y)), right ? { yaxis: 'y2' } : {}));
      const ofit = f.fit[other];
      if (ofit && L.fit) traces.push(scatterTrace(st, `fit:${other}`, { name: `Fit · ${detLabel(f, other)}`, color: pal[3], mode: 'lines', width: 2 },
        TX(nonNull(ofit.dense?.x || ofit.x)), TY(nonNull(ofit.dense?.y || ofit.y_fit)), right ? { yaxis: 'y2' } : {}));
    }
  }
  // main data
  if (L.data) {
    const extra = {};
    if (L.band && sem && f.errorMode === 'bars') extra.error_y = { type: 'data', array: sem, visible: true, thickness: 1, width: 2, color: hexA(pal[0], 0.6) };
    const name = f.info.detectors.length > 1 ? `Data · ${detLabel(f, det)}` : (its ? `Data · mean of ${its.count} iterations` : 'Data');
    traces.push(scatterTrace(st, 'data', { name, color: pal[0], mode }, x, y, { customdata: x.map((_, i) => i), ...extra }));
  }
  if (L.smooth && d.proc.smooth && !fit?.plot_y) traces.push(scatterTrace(st, 'smooth', { name: `Smoothed (${f.proc.smooth} pt)`, color: '#222222', mode: 'lines', width: 1.6 }, x, TY(nonNull(d.proc.smooth))));
  // fit
  if (fit && L.fit) {
    const fx = TX(nonNull(fit.dense?.x || fit.x)), fy = TY(nonNull(fit.dense?.y || fit.y_fit));
    const fb = nonNull(fit.dense?.band);
    if (L.conf && fit.dense?.band && fb.length === fx.length && fb.every(Number.isFinite)) {
      const s = seriesStyle(st, 'conf', { name: 'Fit ±1σ', color: pal[1] });
      traces.push({ type: 'scatter', x: [...fx, ...[...fx].reverse()], y: [...fy.map((v, i) => v + fb[i]), ...fy.map((v, i) => v - fb[i]).reverse()],
        fill: 'toself', fillcolor: hexA(s.color, 0.18), line: { width: 0 }, name: s.name, hoverinfo: 'skip', showlegend: s.legend, visible: s.visible === false ? 'legendonly' : true, meta: { key: 'conf' } });
    }
    traces.push(scatterTrace(st, 'fit', { name: `Fit · ${fit.model}`, color: pal[1], mode: 'lines', width: Math.max(2, st.line + 0.4) }, fx, fy, { hovertemplate: 'fit %{x:.6g}, %{y:.6g}<extra></extra>' }));
    if (L.envelope && fit.envelope) {
      const s = seriesStyle(st, 'envelope', { name: 'Decay envelope', color: '#7A4FD6', dash: 'dash', width: 1.4 });
      traces.push({ type: 'scatter', mode: 'lines', x: TX(nonNull(fit.envelope.x)), y: TY(nonNull(fit.envelope.upper)), line: { color: s.color, dash: s.dash, width: s.width }, name: s.name, legendgroup: 'env', meta: { key: 'envelope' }, showlegend: s.legend });
      traces.push({ type: 'scatter', mode: 'lines', x: TX(nonNull(fit.envelope.x)), y: TY(nonNull(fit.envelope.lower)), line: { color: s.color, dash: s.dash, width: s.width }, name: s.name, legendgroup: 'env', showlegend: false, meta: { key: 'envelope' } });
    }
  }
  // signal + reference counts on the right axis
  if (L.sigref && !fit?.plot_y && f.mode === 'contrast' && !f.info.reference_only) {
    const rx = TX(nonNull(d.raw.x));
    traces.push(scatterTrace(st, 'signal', { name: 'Signal', color: '#0E8A7E', mode: 'lines', width: 1.1 }, rx, nonNull(d.raw.signal), { yaxis: 'y2' }));
    traces.push(scatterTrace(st, 'reference', { name: 'Reference', color: '#B5307A', mode: 'lines', width: 1.1 }, rx, nonNull(d.raw.reference), { yaxis: 'y2' }));
  }
  // pinned points
  if (f.pins.length) {
    traces.push({ type: 'scatter', mode: 'markers', x: f.pins.map((p) => p.x), y: f.pins.map((p) => p.y), name: 'Pinned', showlegend: false, hoverinfo: 'skip',
      marker: { size: 14, color: 'rgba(0,0,0,0)', line: { color: '#E0A400', width: 2 } }, meta: { key: 'pins' } });
  }
  // residuals
  if (showRes && L.data) {
    const rx = TX(nonNull(fit.x));
    traces.push({ type: 'scatter', mode: 'markers', x: rx, y: TY(nonNull(fit.residuals)), xaxis: 'x', yaxis: 'y3', name: 'Residuals', showlegend: false,
      marker: { size: Math.max(3, st.marker - 2), color: '#5B6474' }, hovertemplate: 'residual %{y:.4g}<extra></extra>', meta: { key: 'res' } });
  }
  // FFT of the processed data (regular sampling only)
  let fftInfo = null;
  if (L.fft && !f.info.irregular && x.length > 8) {
    fftInfo = fft(xr, yr, f.info.x_unit); // uncalibrated x keeps the frequency unit tied to x_unit
    traces.push({ type: 'scatter', mode: 'lines', x: fftInfo.f, y: fftInfo.mag, xaxis: 'x2', yaxis: 'y3', name: 'FFT', showlegend: false, line: { color: '#7A4FD6', width: 1.4 }, meta: { key: 'fft' } });
  }

  const deco = userDecorations(st);
  const annotations = [...deco.annotations];
  if (L.results && fit) { const a = resultsAnnotation(st, fit, f.info); if (a) annotations.push(a); }
  const shapes = [...deco.shapes];
  if (L.peaks && fit) fit.params.filter((p) => /^(x0|center)(_\d+)?$/.test(p.name)).forEach((p) =>
    shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: p.value, x1: p.value, y0: domainMain[0], y1: 1, line: { color: '#E06A1B', width: 1, dash: 'dot' } }));
  if (f.measure && f.measure.length === 2) {
    const [a, b] = f.measure;
    shapes.push({ type: 'line', x0: a.x, y0: a.y, x1: b.x, y1: b.y, line: { color: '#C0392B', width: 1.5, dash: 'dot' } });
  }
  const range = fit ? TX([fit.x?.[0], fit.x?.[fit.x.length - 1]]).sort((a, b) => a - b) : null;
  const hasRight = traces.some((t) => t.yaxis === 'y2');
  const layout = baseLayout(st, {
    xaxis: { ...axisLayout(st, 'x', xTitle(f), { range: f.xrange }), domain: [0, hasRight ? 0.92 : 1], anchor: 'y' },
    yaxis: { ...axisLayout(st, 'y', yTitle(f), { range: f.yrange }), domain: domainMain, ...(f.mode !== 'contrast' && !fit?.plot_y ? { exponentformat: 'SI' } : {}) },
    legend: legendLayout(st),
    showlegend: L.legend,
    annotations, shapes,
    uirevision: f.path + det + f.mode,
  });
  if (hasRight) layout.yaxis2 = { ...axisLayout(st, 'y2', f.compareDetectors && f.mode !== 'contrast' ? `${detLabel(f, f.info.detectors.find((k) => k !== det))}` : `Counts${f.info.unit ? ` [${f.info.unit}]` : ''}`), overlaying: 'y', side: 'right', anchor: 'x', exponentformat: 'SI' };
  if (bottom) {
    if (fftInfo) {
      layout.xaxis2 = { ...axisLayout(st, 'x2', `Frequency [${fftInfo.unit}]`), anchor: 'y3', domain: [0, 1] };
      layout.yaxis3 = { ...axisLayout(st, 'y3', '|FFT|'), domain: [0, 0.2], anchor: 'x2' };
      layout.xaxis.title.text = layout.xaxis.title.text; // keep
    } else {
      layout.yaxis3 = { ...axisLayout(st, 'y3', 'Resid.'), domain: [0, 0.2], anchor: 'x', zeroline: true, zerolinecolor: '#C9CED6' };
      layout.xaxis.anchor = 'y3';
      layout.yaxis.anchor = 'x';
    }
  }
  const xbreakBlocked = bottom ? 'residual panel or FFT' : hasRight ? 'right axis (second detector / signal+reference)' : '';
  applyInset(f, st, traces, layout, shapes, annotations, domainMain, hasRight ? 0.92 : 1);
  applyXBreak(f, st, traces, layout, shapes, annotations, domainMain, xbreakBlocked);
  if (forExport) layout.uirevision = undefined;
  return { data: traces, layout, range, xbreakBlocked, fftPeak: fftInfo ? { f: fftInfo.peak, unit: fftInfo.unit } : null };
}

// ---- presentation extras: inset zoom (xaxis4/yaxis4) and broken x axis (xaxis5) ----
const mainAx = (t) => (!t.xaxis || t.xaxis === 'x') && (!t.yaxis || t.yaxis === 'y');
const NO_INSET = new Set(['band', 'conf', 'meanSd', 'iter', 'pins', 'res', 'fft']);

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
  // Uniformly sampled assumption: use median step; zero-mean; magnitude spectrum.
  const n = y.length;
  const dx = (x[n - 1] - x[0]) / (n - 1);
  const m = y.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0) / n;
  const N = Math.min(2048, n);
  const { scale, unit } = fftUnit(xUnit);
  const f = [], mag = [];
  let best = -1, peak = null;
  for (let k = 1; k < N / 2; k++) {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) { const v = (Number.isFinite(y[i]) ? y[i] : m) - m; const ang = -2 * Math.PI * k * i / N; re += v * Math.cos(ang); im += v * Math.sin(ang); }
    f.push((k / (N * Math.abs(dx))) * scale);
    mag.push(Math.hypot(re, im) / n);
    if (mag[mag.length - 1] > best) { best = mag[mag.length - 1]; peak = f[f.length - 1]; }
  }
  return { f, mag, peak, unit };
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
