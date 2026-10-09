// Stacks workspace: z-stack of 2D maps, spot trajectory through z, and drift across sequential maps.
// Maps are shown at native resolution (no interpolation, no regridding) with equal axis scaling.
import { call } from './api.js';
import { S, emit, detLabel } from './state.js';
import { h, mount, select, field, num, fmt, fmtPm, toast, basename, dirname, tag } from './util.js';
import { copyText } from './export.js';

const UNITS = 'stage units (as recorded)';
const DET_COLOR = { detector1: '#1F6FB5', detector2: '#E07B00' };
const KEY_COLOR = { x: '#1F6FB5', y: '#C0392B' };
const OBSERVABLES = [['auto', 'Auto fluorescence (raw counts)'], ['value', 'Map value (selected mode)'], ['signal', 'Signal'], ['reference', 'Reference'], ['difference', 'Reference − signal']];

// Persistent across view switches.
const ST = {
  sel: [], det: 'detector1', source: 'auto', both: true,
  polarity: 'Auto', threshold: 3, maxJump: 3, gate: 3,
  stack: null, traj: null, drift: null, built: '', idx: 0, busy: '', peekTries: 0,
};
const els = {};

const peekOf = (p) => S.library.peeks[p] || S.files[p]?.info || null;
const is2d = (p) => peekOf(p)?.dim === 'scan2d';
const isXY = (p) => {
  const info = peekOf(p);
  if (!info) return false;
  const ax = (info.axes || []).map((a) => String(a).trim().toLowerCase().replace(/[^xyz]/g, ''));
  return ax.length === 2 ? ax.includes('x') && ax.includes('y') : info.tag === 'XY';
};
function candidates() {
  const lib = S.library;
  const all = [...new Set([...(lib.listing?.files || []).map((f) => f.path), ...lib.extra, ...Object.keys(S.files)])];
  return all.filter(is2d).sort((a, b) => basename(a).localeCompare(basename(b), undefined, { numeric: true }));
}
const selKey = () => JSON.stringify([ST.sel, ST.det, ST.source]);
const zKind = (src) => (!src ? 'none' : src.startsWith('metadata') ? 'metadata' : src.includes('assumed') ? 'filename (assumed)' : 'filename');

function zLabel() {
  const sl = ST.stack?.slices || [];
  const kinds = [...new Set(sl.map((s) => zKind(s.third_source)))];
  const ax = ST.stack?.third_axis || 'z';
  const from = kinds.length === 1 ? kinds[0] : kinds.join(' + ');
  return `${ax} (${from === 'metadata' ? 'from file metadata' : `from ${from}, not verified`}; ${UNITS})`;
}

// ------------------------------------------------------------------ actions
const run = async (label, fn) => {
  ST.busy = label; emit();
  try { await fn(); } catch (e) { /* toast shown by api */ } finally { ST.busy = ''; emit(); }
};
const baseOpts = () => ({ source: ST.source, polarity: ST.polarity, threshold: ST.threshold });

async function buildStack() {
  if (ST.sel.length < 1) { toast('Select at least one 2D map.'); return; }
  ST.stack = await call('stack_build', { paths: ST.sel, detector: ST.det, mode: 'contrast', opts: { source: ST.source, include_maps: true } }, { label: 'Building stack' });
  ST.built = selKey(); ST.idx = Math.min(ST.idx, Math.max(0, ST.stack.slices.length - 1)); ST.traj = null;
}

async function trackZ() {
  if (ST.sel.length < 2) { toast('Select at least two maps for a z trajectory.'); return; }
  if (!ST.stack || ST.built !== selKey()) await buildStack();
  const dets = [ST.det];
  const other = ST.det === 'detector1' ? 'detector2' : 'detector1';
  if (ST.both && ST.sel.every((p) => (peekOf(p)?.detectors || 1) > 1)) dets.push(other);
  ST.traj = await call('stack_trajectory', { paths: ST.sel, detectors: dets, mode: 'contrast', opts: { ...baseOpts(), max_jump_fwhm: ST.maxJump } }, { label: 'Tracking spot through z' });
}

async function trackDrift() {
  if (ST.sel.length < 2) { toast('Select at least two maps to measure drift.'); return; }
  ST.drift = await call('track_spots', { paths: ST.sel, detector: ST.det, mode: 'contrast', opts: { ...baseOpts(), gate_fwhm: ST.gate } }, { label: 'Tracking drift' });
}

// ------------------------------------------------------------------ tables / export
const keysOf = () => ST.traj?.plane_keys || ST.stack?.plane_keys || ST.drift?.plane_keys || ['x', 'y'];

function walkRows() {
  const out = [];
  for (const [d, r] of Object.entries(ST.traj?.detectors || {})) {
    for (const k of keysOf()) {
      const w = r.walk?.[k];
      if (w) out.push({ det: d, key: k, n: w.n, slope: w.slope, se: w.slope_se, r2: w.r_squared, units: `${k} per ${ST.traj.third_axis} (${UNITS})` });
    }
  }
  return out;
}
function offsetRows() {
  const o = ST.traj?.offset;
  if (!o) return [];
  return keysOf().map((k) => ({ key: k, ...(o.fit_vs_z?.[k] || {}) }));
}
function driftRows() {
  return keysOf().map((k) => ({ key: k, ...(ST.drift?.drift_rate?.[k] || {}) })).filter((r) => r.n !== undefined);
}
function sliceRows() {
  const out = [];
  for (const [d, r] of Object.entries(ST.traj?.detectors || {})) for (const p of r.points) out.push({ det: d, ...p });
  if (!out.length) for (const s of ST.stack?.slices || []) out.push({ det: s.detector, z: s.third_value, z_source: s.third_source, file_name: s.file_name, linked: null });
  return out;
}
function warnings() {
  return [...new Set([...(ST.stack?.warnings || []), ...(ST.traj?.warnings || []), ...(ST.drift?.warnings || [])])];
}

function csvText(sep = '\t') {
  const q = (v) => {
    const s = v === null || v === undefined ? '' : String(v);
    return sep === ',' && /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const row = (a) => a.map(q).join(sep);
  const out = [`# SmartFitter stack results; positions in ${UNITS}; z provenance per slice below`];
  const walk = walkRows();
  if (walk.length) {
    out.push('', '# Walk slope vs z (OLS; SE from scatter about the line)', row(['detector', 'axis', 'n', 'slope', 'slope_SE', 'R2', 'units']));
    walk.forEach((r) => out.push(row([r.det, r.key, r.n, r.slope, r.se, r.r2, r.units])));
  }
  const off = offsetRows();
  if (off.length) {
    out.push('', `# Detector offset vs z (${ST.traj.offset.definition})`, row(['axis', 'n', 'slope_per_z', 'slope_SE', 'intercept', 'R2']));
    off.forEach((r) => out.push(row([r.key, r.n, r.slope, r.slope_se, r.intercept, r.r_squared])));
  }
  const sl = sliceRows();
  if (sl.length) {
    out.push('', '# Per-slice results', row(['detector', 'file', 'z', 'z_source', 'linked', 'x', 'x_err', 'y', 'y_err', 'FWHM', 'amplitude', 'SNR', 'err_source']));
    const [k0, k1] = keysOf();
    sl.forEach((p) => out.push(row([p.det, p.file_name, p.z, p.z_source, p.linked, p.pos?.[k0], p.err?.[k0], p.pos?.[k1], p.err?.[k1], p.fwhm, p.amplitude, p.snr, p.err_source])));
  }
  const dr = driftRows();
  if (dr.length) {
    out.push('', '# Drift rate (OLS of cumulative common-mode displacement vs time)', row(['axis', 'n', 'rate_per_min', 'rate_SE', 'intercept', 'R2', 'units']));
    dr.forEach((r) => out.push(row([r.key, r.n, r.slope, r.slope_se, r.intercept, r.r_squared, r.units])));
    out.push('', '# Drift vs time', row(['file', 't_min', 'n_detected', 'n_matched', ...keysOf().map((k) => `drift_${k}`)]));
    (ST.drift.drift || []).forEach((d) => out.push(row([d.file_name, d.t_min, d.n_detected, d.n_matched, ...keysOf().map((k) => d.drift?.[k])])));
  }
  const w = warnings();
  if (w.length) { out.push('', '# Warnings'); w.forEach((x) => out.push(q(x))); }
  return out.join('\n');
}

async function saveCsv() {
  const p = await call('dialog', { kind: 'save', title: 'Save stack results', filename: 'stack_results.csv', directory: S.library.folder || '' });
  if (!p) return;
  await call('write_text', { path: p.endsWith('.csv') ? p : `${p}.csv`, text: csvText(',') });
  toast('Saved stack results', 'ok');
}

// ------------------------------------------------------------------ plots
const AX = { automargin: true, showline: true, mirror: true, ticks: 'outside', showgrid: true, gridcolor: '#ECEEF2', zeroline: false, exponentformat: 'none' };
const LAYOUT = (extra = {}) => ({ separators: '.', margin: { l: 72, r: 16, t: 14, b: 52 }, font: { family: 'Arial', size: 12, color: '#1D2330' }, showlegend: true, legend: { orientation: 'h', y: 1.12, x: 0 }, paper_bgcolor: '#fff', plot_bgcolor: '#fff', ...extra });
const CFG = { responsive: true, displayModeBar: false };
const plot = (name) => (els[name] = els[name] || h('div', { class: 'plot' }));

function drawSlice() {
  const sl = ST.stack?.slices?.[ST.idx];
  if (!sl || !sl.values) return;
  const [ax0, ax1] = sl.axes;
  const traces = [{ type: 'heatmap', x: sl.x, y: sl.y, z: sl.values, colorscale: 'Viridis', zsmooth: false, hoverongaps: false,
    colorbar: { thickness: 12, len: 0.9, title: { standoff: 10, text: 'map value', side: 'top' } }, hovertemplate: `${ax0} %{x:.6g}<br>${ax1} %{y:.6g}<br>%{z:.6g}<extra></extra>` }];
  const pt = ST.traj?.detectors?.[ST.det]?.points?.find((p) => p.file_name === sl.file_name);
  const keys = ST.stack.plane_keys;
  if (pt?.linked) {
    const kx = sl.axes.map((a) => (/^\s*[xyz]\s*$/i.test(a) ? a.trim().toLowerCase() : (a.toLowerCase().match(/(?:^|[^a-z])([xyz])(?:[^a-z]|$)/) || [])[1] || a));
    traces.push({ type: 'scatter', mode: 'markers', x: [pt.pos[kx[0]] ?? pt.pos[keys[0]]], y: [pt.pos[kx[1]] ?? pt.pos[keys[1]]], marker: { symbol: 'cross-thin', size: 18, line: { color: '#FFFFFF', width: 2.5 } },
      name: 'tracked spot', hovertemplate: `FWHM ${fmt(pt.fwhm, 3)}<extra>tracked spot</extra>`, showlegend: false });
  }
  const dx = sl.x_step, dy = sl.y_step;
  const hx = (Number.isFinite(dx) ? dx : 1) / 2, hy = (Number.isFinite(dy) ? dy : 1) / 2; // half a pixel: the axis frames whole pixels
  const layout = LAYOUT({ margin: { l: 72, r: 16, t: 14, b: 52 }, showlegend: false,
    xaxis: { ...AX, showgrid: false, title: { standoff: 10, text: `${ax0} (${UNITS})` }, constrain: 'domain', autorange: false, range: [Math.min(...sl.x) - hx, Math.max(...sl.x) + hx] },
    yaxis: { ...AX, showgrid: false, title: { standoff: 10, text: `${ax1} (${UNITS})` }, scaleanchor: 'x', scaleratio: 1, constrain: 'domain', autorange: false, range: [Math.min(...sl.y) - hy, Math.max(...sl.y) + hy] } });
  Plotly.react(plot('slice'), traces, layout, CFG);
  if (els.sliceInfo) {
    els.sliceInfo.textContent = `Slice ${ST.idx + 1}/${ST.stack.slices.length} · ${sl.file_name} · ${ST.stack.third_axis} = ${fmt(sl.third_value, 6)} [${zKind(sl.third_source)}] · ${sl.shape[1]} × ${sl.shape[0]} px · step ${fmt(dx, 4)} × ${fmt(dy, 4)} ${UNITS}`;
    els.sliceInfo.style.color = zKind(sl.third_source) === 'metadata' ? '' : 'var(--warn)';
  }
}

function drawTraj() {
  const T = ST.traj;
  if (!T) return;
  const [k0, k1] = keysOf();
  const traces = [];
  for (const [d, r] of Object.entries(T.detectors)) {
    const pts = r.points.filter((p) => p.linked);
    const c = DET_COLOR[d] || '#444';
    const name = detLabel({ labels: {} }, d);
    [k0, k1].forEach((k, i) => {
      const ya = i === 0 ? 'y' : 'y2';
      traces.push({ type: 'scatter', mode: 'markers', x: pts.map((p) => p.z), y: pts.map((p) => p.pos[k]), xaxis: 'x', yaxis: ya, name: `${name}`, legendgroup: d, showlegend: i === 0,
        marker: { color: c, size: 8 }, error_y: { type: 'data', array: pts.map((p) => p.err[k]), visible: true, color: c, thickness: 1.2, width: 4 },
        text: pts.map((p) => `${p.file_name}<br>z source: ${p.z_source}<br>err: ${p.err_source}`), hovertemplate: `${k} %{y:.6g} ± %{error_y.array:.2g}<br>z %{x:.6g}<br>%{text}<extra>${name}</extra>` });
      const w = r.walk?.[k];
      if (w && w.slope !== null && pts.length >= 2) {
        const zs = pts.map((p) => p.z), lo = Math.min(...zs), hi = Math.max(...zs);
        traces.push({ type: 'scatter', mode: 'lines', x: [lo, hi], y: [w.intercept + w.slope * lo, w.intercept + w.slope * hi], xaxis: 'x', yaxis: ya, line: { color: c, dash: 'dash', width: 1.4 }, legendgroup: d, showlegend: false, hoverinfo: 'skip' });
      }
    });
  }
  const layout = LAYOUT({ xaxis: { ...AX, title: { standoff: 10, text: zLabel() }, anchor: 'y2' },
    yaxis: { ...AX, domain: [0.56, 1], title: { standoff: 10, text: `${k0} centre (${UNITS})` } }, yaxis2: { ...AX, domain: [0, 0.44], anchor: 'x', title: { standoff: 10, text: `${k1} centre (${UNITS})` } } });
  Plotly.react(plot('traj'), traces, layout, CFG);
}

function drawShape() {
  const T = ST.traj;
  if (!T) return;
  const traces = [];
  for (const [d, r] of Object.entries(T.detectors)) {
    const pts = r.points.filter((p) => p.linked);
    const c = DET_COLOR[d] || '#444';
    const name = detLabel({ labels: {} }, d);
    traces.push({ type: 'scatter', mode: 'lines+markers', x: pts.map((p) => p.z), y: pts.map((p) => p.fwhm), yaxis: 'y', name, legendgroup: d, line: { color: c, width: 1.3 }, marker: { color: c, size: 7 } });
    traces.push({ type: 'scatter', mode: 'lines+markers', x: pts.map((p) => p.z), y: pts.map((p) => p.amplitude), yaxis: 'y2', name, legendgroup: d, showlegend: false, line: { color: c, width: 1.3 }, marker: { color: c, size: 7 } });
  }
  const layout = LAYOUT({ xaxis: { ...AX, title: { standoff: 10, text: zLabel() }, anchor: 'y2' },
    yaxis: { ...AX, domain: [0.56, 1], title: { standoff: 10, text: `FWHM (major, ${UNITS})` } }, yaxis2: { ...AX, domain: [0, 0.44], anchor: 'x', title: { standoff: 10, text: 'amplitude (map units)' } } });
  Plotly.react(plot('shape'), traces, layout, CFG);
}

function drawOffset() {
  const o = ST.traj?.offset;
  if (!o) return;
  const traces = [];
  for (const k of keysOf()) {
    const rows = o.rows;
    const c = KEY_COLOR[k] || '#444';
    traces.push({ type: 'scatter', mode: 'markers', x: rows.map((r) => r.z), y: rows.map((r) => r.offset[k]), name: `${k} offset`, marker: { color: c, size: 8 },
      error_y: { type: 'data', array: rows.map((r) => r.err[k]), visible: true, color: c, thickness: 1.2, width: 4 } });
    const f = o.fit_vs_z?.[k];
    if (f && f.slope !== null && rows.length >= 2) {
      const zs = rows.map((r) => r.z), lo = Math.min(...zs), hi = Math.max(...zs);
      traces.push({ type: 'scatter', mode: 'lines', x: [lo, hi], y: [f.intercept + f.slope * lo, f.intercept + f.slope * hi], line: { color: c, dash: 'dash', width: 1.4 }, showlegend: false, hoverinfo: 'skip' });
    }
  }
  Plotly.react(plot('offset'), traces, LAYOUT({ xaxis: { ...AX, title: { standoff: 10, text: zLabel() } }, yaxis: { ...AX, title: { standoff: 10, text: `offset ${o.definition}` } } }), CFG);
}

function drawDrift() {
  const D = ST.drift;
  if (!D) return;
  const good = (D.drift || []).filter((d) => d.drift);
  const traces = [];
  for (const k of keysOf()) {
    const c = KEY_COLOR[k] || '#444';
    traces.push({ type: 'scatter', mode: 'lines+markers', x: good.map((d) => d.t_min), y: good.map((d) => d.drift[k]), name: `${k} drift`, line: { color: c, width: 1 }, marker: { color: c, size: 8 },
      text: good.map((d) => `${d.file_name}<br>matched ${d.n_matched}/${d.n_detected}`), hovertemplate: `%{y:.6g}<br>t %{x:.3g} min<br>%{text}<extra>${k}</extra>` });
    const f = D.drift_rate?.[k];
    if (f && f.slope !== null && good.length >= 2) {
      const ts = good.map((d) => d.t_min), lo = Math.min(...ts), hi = Math.max(...ts);
      traces.push({ type: 'scatter', mode: 'lines', x: [lo, hi], y: [f.intercept + f.slope * lo, f.intercept + f.slope * hi], line: { color: c, dash: 'dash', width: 1.4 }, name: `${k} fit: ${fmtPm(f.slope, f.slope_se)} / min`, hoverinfo: 'skip' });
    }
  }
  Plotly.react(plot('drift'), traces, LAYOUT({ xaxis: { ...AX, title: { standoff: 10, text: 'time since first map (min; from filename MMDD_HHMM or file mtime)' } }, yaxis: { ...AX, title: { standoff: 10, text: `cumulative displacement (${UNITS})` } } }), CFG);
}

// ------------------------------------------------------------------ view
const card = (title, sub, el, min = 300) => h('section', { style: 'display:flex;flex-direction:column;gap:4px;min-width:0' },
  h('div', { class: 'row' }, h('h2', { style: 'margin:0;font-size:14px;font-weight:600' }, title), h('span', { class: 'note' }, sub)),
  h('div', { class: 'plotcard', style: `flex:none;height:${min}px;min-height:${min}px` }, el));

export function renderStack(root) {
  const cands = candidates();
  const sel = ST.sel.filter((p) => cands.includes(p) || is2d(p));
  ST.sel = sel;
  // File peeks arrive asynchronously after a folder opens; re-check a few times while some are missing.
  const missing = (S.library.listing?.files || []).filter((f) => !S.library.peeks[f.path]).length;
  if (missing && ST.peekTries < 12) { ST.peekTries++; setTimeout(() => { if (S.view === 'stack') emit(); }, 900); } else if (!missing) ST.peekTries = 0;

  const stale = ST.stack && ST.built !== selKey();
  const nDet = Math.max(1, ...sel.map((p) => peekOf(p)?.detectors || 1));
  const busy = !!ST.busy;

  const toggle = (p, on) => { ST.sel = on ? [...new Set([...ST.sel, p])] : ST.sel.filter((q) => q !== p); emit(); };
  const left = h('aside', { class: 'library', style: 'padding:12px;gap:10px' },
    h('div', { class: 'row' }, h('span', { class: 'lbl' }, `2D maps · ${sel.length} selected`), h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: () => { ST.sel = []; emit(); } }, 'Clear')),
    h('div', { class: 'row wrap', style: 'gap:6px' },
      h('button', { class: 'btn sm', title: 'Select every XY map in the library folder', onclick: () => { ST.sel = cands.filter(isXY); emit(); } }, 'Select all XY in folder'),
      h('button', { class: 'btn sm', onclick: () => { ST.sel = [...cands]; emit(); } }, 'All 2D')),
    h('div', { style: 'flex:1;min-height:80px;overflow:auto;display:flex;flex-direction:column;gap:2px' }, cands.length ? cands.map((p) => {
      const info = peekOf(p) || {};
      return h('label', { class: 'frow', style: 'cursor:default', title: p },
        h('input', { type: 'checkbox', checked: ST.sel.includes(p), onchange: (e) => toggle(p, e.target.checked) }), tag(info.tag),
        h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'fname ellip' }, basename(p).replace(/\.mat$/i, '')),
          h('div', { class: 'sub ellip' }, [info.axes ? info.axes.join('–') : '', info.grid ? `${info.grid[1]} × ${info.grid[0]}` : '', info.detectors > 1 ? '2 detectors' : ''].filter(Boolean).join(' · '))));
    }) : h('div', { class: 'note' }, S.library.listing ? 'No 2D maps found in the current library folder (file scan may still be running).' : 'Choose a data folder in Analyze first.')),
    h('div', { style: 'display:flex;flex-direction:column;gap:8px' },
      field('Detector', select(Array.from({ length: Math.max(2, nDet) }, (_, i) => [`detector${i + 1}`, `Detector ${i + 1}`]), ST.det, (v) => { ST.det = v; emit(); })),
      field('Observable', select(OBSERVABLES, ST.source, (v) => { ST.source = v; emit(); })),
      h('div', { class: 'g2' },
        field('Polarity', select(['Auto', 'Bright', 'Dark', 'Bright and dark'], ST.polarity, (v) => { ST.polarity = v; })),
        field('Threshold (σ)', h('input', { class: 'field', value: ST.threshold, onchange: (e) => { ST.threshold = num(e.target.value) ?? 3; } }))),
      h('div', { class: 'g2' },
        field('Max jump (FWHM)', h('input', { class: 'field', value: ST.maxJump, title: 'z trajectory: a spot may move at most this many FWHM between adjacent slices', onchange: (e) => { ST.maxJump = num(e.target.value) ?? 3; } })),
        field('Drift gate (FWHM)', h('input', { class: 'field', value: ST.gate, title: 'drift: match spots between consecutive maps within this many FWHM', onchange: (e) => { ST.gate = num(e.target.value) ?? 3; } }))),
      h('label', { class: 'ck', title: 'Track the same feature on both detectors (needs two-detector files) and report the offset' },
        h('input', { type: 'checkbox', checked: ST.both, onchange: (e) => { ST.both = e.target.checked; } }), 'Both detectors (offset vs z)')),
    h('div', { style: 'display:flex;flex-direction:column;gap:6px' },
      h('button', { class: 'btn block', disabled: busy || !sel.length, onclick: () => run('Building stack', buildStack) }, 'Build stack'),
      h('button', { class: 'btn primary block', disabled: busy || sel.length < 2, onclick: () => run('Tracking', trackZ) }, 'Track spot through Z'),
      h('button', { class: 'btn block', disabled: busy || sel.length < 2, onclick: () => run('Drift', trackDrift) }, 'Drift across maps'),
      busy ? h('div', { class: 'row busy' }, h('span', { class: 'spinner' }), ST.busy) : null),
    h('div', { class: 'note' }, `Maps stay at native resolution. Positions, FWHM and offsets are in ${UNITS}; the file does not store the unit.`));

  const nSl = ST.stack?.slices?.length || 0;
  ST.idx = Math.min(ST.idx, Math.max(0, nSl - 1));
  els.sliceInfo = h('div', { class: 'note', style: 'min-height:16px' });
  const slider = h('input', { type: 'range', min: 0, max: Math.max(0, nSl - 1), step: 1, value: ST.idx, disabled: nSl < 2, style: 'flex:1', 'aria-label': 'z slice',
    oninput: (e) => { ST.idx = Number(e.target.value); drawSlice(); } });

  const showTraj = !!ST.traj, showOff = !!ST.traj?.offset, showDrift = !!ST.drift;
  const main = h('main', { class: 'main' },
    h('div', { class: 'row wrap' }, h('h1', { style: 'margin:0;font-size:17px;font-weight:600;margin-right:auto' }, 'Stack slice'),
      stale ? h('span', { class: 'pill WARN' }, 'selection changed since build') : null),
    nSl ? h('div', { class: 'row' }, h('span', { class: 'lbl' }, 'Z slice'), slider) : null,
    els.sliceInfo,
    nSl ? h('section', { class: 'plotcard', style: 'flex:none;height:420px;min-height:420px' }, plot('slice'))
      : h('div', { class: 'card pad note' }, 'Select 2D maps on the left (for example “Select all XY in folder”) and press Build stack. Slices are ordered by their fixed third-axis value. Z comes from file metadata when present, otherwise from the file name (flagged).'),
    showTraj || showDrift ? h('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:14px' },
      showTraj ? card('Spot centre vs z', 'bars: position uncertainty; dashed: linear fit', plot('traj'), 420) : null,
      showTraj ? card('FWHM and amplitude vs z', 'major-axis FWHM; amplitude in map units', plot('shape'), 420) : null,
      showOff ? card('Detector offset vs z', ST.traj.offset.definition, plot('offset'), 340) : null,
      showDrift ? card('Drift across maps', 'common-mode displacement vs time; dashed: fitted rate', plot('drift'), 340) : null) : null);

  const walk = walkRows(), off = offsetRows(), dr = driftRows(), sl = sliceRows(), warn = warnings();
  const any = walk.length || dr.length || sl.length;
  const tbl = (head, rows) => h('table', { class: 'tbl', style: 'margin-top:6px' }, h('tr', {}, head.map((t) => h('th', {}, t))), rows);
  const zNon = (ST.stack?.slices || []).some((s) => zKind(s.third_source) !== 'metadata');
  const right = h('aside', { class: 'inspector' }, h('div', { class: 'insp-body' },
    h('div', { class: 'row' }, h('span', { class: 'lbl' }, 'Results'), h('span', { class: 'sp' }),
      h('button', { class: 'btn sm', disabled: !any, onclick: () => copyText(csvText('\t'), 'Stack results copied') }, 'Copy'),
      h('button', { class: 'btn sm', disabled: !any, onclick: () => saveCsv().catch(() => {}) }, 'Save CSV')),
    zNon ? h('div', { class: 'card pad note', style: 'color:var(--warn)' }, 'Some z values come from file names, not file metadata. They are labelled, unverified stage values.') : null,
    walk.length ? h('div', { class: 'card pad' }, h('div', { style: 'font-weight:600' }, 'Walk slope vs z'),
      tbl(['Det', 'Axis', 'Slope ± SE', 'n'], walk.map((r) => h('tr', { title: r.units }, h('td', { class: 'txt' }, r.det.replace('detector', 'D')), h('td', { class: 'txt' }, r.key), h('td', {}, fmtPm(r.slope, r.se)), h('td', {}, r.n)))),
      h('div', { class: 'note' }, `Slope = ${keysOf().join('/')} stage units per ${ST.traj.third_axis} unit. SE from scatter about the OLS line.`)) : null,
    off.length ? h('div', { class: 'card pad' }, h('div', { style: 'font-weight:600' }, 'Detector offset slope vs z'),
      tbl(['Axis', 'Slope ± SE', 'Offset at z=0', 'n'], off.map((r) => h('tr', {}, h('td', { class: 'txt' }, r.key), h('td', {}, fmtPm(r.slope, r.slope_se)), h('td', {}, fmt(r.intercept, 4)), h('td', {}, r.n)))),
      h('div', { class: 'note' }, `${ST.traj.offset.definition}`)) : null,
    dr.length ? h('div', { class: 'card pad' }, h('div', { style: 'font-weight:600' }, 'Drift rate'),
      tbl(['Axis', 'Rate ± SE (units/min)', 'n'], dr.map((r) => h('tr', { title: r.units }, h('td', { class: 'txt' }, r.key), h('td', {}, fmtPm(r.slope, r.slope_se)), h('td', {}, r.n)))),
      h('div', { class: 'note' }, `${UNITS} per minute. ${ST.drift.time_note || ''}`)) : null,
    sl.length ? h('div', { class: 'card pad' }, h('div', { style: 'font-weight:600' }, 'Slices and z provenance'),
      tbl(['File', 'z', 'z source', 'Linked'], sl.filter((r) => !ST.traj || r.det === ST.det).map((r) => {
        const kind = zKind(r.z_source);
        return h('tr', { class: 'click', onclick: () => { const i = (ST.stack?.slices || []).findIndex((s) => s.file_name === r.file_name); if (i >= 0) { ST.idx = i; emit(); } } },
          h('td', { class: 'txt ellip', style: 'max-width:110px', title: r.file_name }, r.file_name.replace(/\.mat$/i, '')), h('td', {}, fmt(r.z, 6)),
          h('td', { class: 'txt', style: kind === 'metadata' ? '' : 'color:var(--warn)', title: r.z_source }, kind), h('td', {}, r.linked === null ? '–' : r.linked ? 'yes' : 'no'));
      }))) : null,
    ST.stack?.skipped?.length ? h('div', { class: 'card pad' }, h('div', { style: 'font-weight:600' }, `Skipped (${ST.stack.skipped.length})`),
      ST.stack.skipped.map((s) => h('div', { class: 'note' }, `${basename(s.path)}: ${s.reason}`))) : null,
    warn.length ? h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:4px' }, h('div', { style: 'font-weight:600' }, `Warnings (${warn.length})`),
      warn.map((w) => h('div', { class: 'note', style: 'color:var(--warn)' }, w))) : null,
    any ? null : h('div', { class: 'note' }, 'Run Track spot through Z or Drift across maps to fill the results.')));

  mount(root, left, main, right);
  drawSlice(); drawTraj(); drawShape(); drawOffset(); drawDrift();
}
