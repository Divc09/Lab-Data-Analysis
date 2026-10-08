// Analyze workspace: 1D experiments, time traces and 2D maps, with their inspectors.
import { call } from './api.js';
import { S, emit, active, commit, loadData, detLabel, defaultFitSpec } from './state.js';
import { h, mount, icon, seg, select, input, check, field, fmt, fmtPm, fmtCounts, num, toast, clone, tag, basename, dirname, stem, contextMenu, linreg } from './util.js';
import { figure1D, figureTT, xTitle, yTitle } from './plot1d.js';
import { figureMaps, linecuts, CMAP_GRADIENT, visibleSpots, peakCell, displayMap, mapPanels, nearest, fitColorbars, lineLength } from './plot2d.js';
import { plotConfig, resultLines, baseStyle, MODE_LABEL, friendlyAxis } from './plotstyle.js';
import { renderImage, copyImage, copyText } from './export.js';
import { openSeriesEditor, editorPanel, applyPlotlyEdit } from './ploteditor.js';

let plotEl = null, plotBound = false, cutEl = null, plotPending = false;
const roomy = (el) => el.clientWidth >= 40 && el.clientHeight >= 40;
let readoutLive = null; // span in the readout bar that hover updates in place (null when something is pinned)
const overlayCache = {};

export function plotDiv() {
  if (!plotEl) {
    plotEl = h('div', { class: 'plot' });
    // Plotly emits SVG attributes of -Infinity (console errors) when laid out in a near-zero box, so wait until it has room.
    new ResizeObserver(() => {
      if (!roomy(plotEl)) return;
      if (plotPending) { plotPending = false; const f = active(); if (f) drawPlot(f); }
      else if (plotEl.data) Plotly.Plots.resize(plotEl);
    }).observe(plotEl);
  }
  return plotEl;
}

export const kindOf = (f) => (f.info.dim === 'scan2d' ? '2d' : f.info.type === 'TimeTrace' ? 'tt' : '1d');
const family = (f) => {
  const t = f.fitSpec.experiment || f.info.type;
  return { DEERDuration: 'Rabi', DEERFrequency: 'ODMR', DEERDecay: 'SpinEcho', DynamicDecoupling: 'SpinEcho', DEERPosition: 'LineScan' }[t] || t;
};

// ---------------------------------------------------------------- actions
export async function setMode(f, mode) {
  if (mode === f.mode) return;
  commit('observable');
  f.fitByMode = f.fitByMode || {};
  f.fitByMode[f.mode] = f.fit;
  f.mode = mode;
  f.fit = f.fitByMode[mode] || {};
  f.yrange = null;
  await loadData(f);
  emit();
}

export async function setProc(f, patch, label = 'processing') {
  commit(label);
  Object.assign(f.proc, patch);
  if (Object.keys(f.fit).length) f.fitStale = true;
  await loadData(f);
  emit();
}

export async function runFit(f, { robust = false } = {}) {
  if (!f.info.fit_allowed && kindOf(f) === '1d' && f.fitSpec.model !== 'GaussianPeaks' && f.fitSpec.model !== 'CustomModel') {
    toast('This scan type is plot-only. Choose Gaussian peaks or a custom model in the Fit tab to fit it.');
    return;
  }
  commit('fit');
  const spec = clone(f.fitSpec);
  spec.robust = robust;
  if (spec.model === 'CustomModel') spec.custom_spec = customSpec(spec);
  const dets = spec.apply_both || kindOf(f) === 'tt' ? f.info.detectors : [f.detector];
  f.fitting = true; emit('insp');
  try {
    for (const det of dets) {
      const res = await call('fit', { path: f.path, detector: det, mode: f.mode, proc: f.proc, spec }, { label: `Fitting ${detLabel(f, det)}` });
      f.fit[det] = res;
      f.fitHistory.unshift({ at: new Date().toLocaleTimeString(), det, mode: f.mode, spec, res });
      f.fitHistory = f.fitHistory.slice(0, 12);
    }
    f.fitStale = false;
    if (!f.layersTouched) { f.layers.fit = true; f.layers.results = kindOf(f) === '1d'; }
    const r = f.fit[f.detector];
    if (r) toast(`${r.model}: R² ${fmt(r.r2, 4)} · ${r.status}${r.elapsed_s ? ` · ${r.elapsed_s.toFixed(1)} s` : ''}`, r.status === 'PASS' ? 'ok' : '');
  } finally { f.fitting = false; emit(); }
}

function customSpec(spec) {
  const names = spec.custom_params.map((p) => p.name);
  return { name: 'CustomModel', expression: spec.custom_expr, param_names: names,
    lower: spec.custom_params.map((p) => p.min ?? -1e300), upper: spec.custom_params.map((p) => p.max ?? 1e300), seed: spec.custom_params.map((p) => p.seed ?? 1) };
}

const RESERVED = new Set(['x', 'abs', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'clip', 'pi', 'e', 'np']);
export function exprParams(expr, prev = []) {
  const names = [...new Set((expr.match(/[A-Za-z_]\w*/g) || []).filter((n) => !RESERVED.has(n)))];
  return names.map((n) => prev.find((p) => p.name === n) || { name: n, seed: 1, min: null, max: null });
}

function excludePins(f) {
  const d = f.data[f.detector];
  const idx = new Set(f.proc.excluded);
  f.pins.forEach((p) => (d.proc.groups[p.i] || []).forEach((r) => idx.add(r)));
  f.pins = [];
  setProc(f, { excluded: [...idx] }, 'exclude points');
}

export function copyFigure(f) {
  const fig = currentFigure(f);
  if (!fig) return;
  const st = baseStyle(f.style);
  renderImage(fig, { width_in: st.width_in, height_in: st.height_in, dpi: st.dpi }).then(copyImage);
}

export function currentFigure(f) {
  if (!f) return null;
  const k = kindOf(f);
  if (k === '1d') return figure1D(f, { forExport: true });
  if (k === 'tt') return figureTT(f);
  return figureMaps(f, overlayCache[f.path]);
}

export function resetView(f) {
  f.xrange = null; f.yrange = null;
  if (f.map) f.map.bounds = null;
  if (plotEl?.data) Plotly.relayout(plotEl, { 'xaxis.autorange': true, 'yaxis.autorange': true });
  emit();
}

// ---------------------------------------------------------------- main view
export function renderAnalyze(main, insp) {
  const f = active();
  if (!f) { mount(main, h('div', { class: 'empty' }, 'Open a file from the Library, or press Ctrl+O.')); mount(insp); return; }
  const k = kindOf(f);
  mount(main, header(f), k === '2d' ? mapToolbar(f) : toolbar(f), plotSection(f), k === '2d' ? null : readout(f));
  drawPlot(f);
  if (S.editor.open) { insp.className = 'drawer'; mount(insp, editorPanel(f)); }
  else { insp.className = 'inspector'; renderInspector(f, insp); }
}

function header(f) {
  const i = f.info;
  const meta = i.metadata || {};
  const bits = [];
  if (i.dim === 'scan2d') {
    const m = f.data[f.detector]?.map;
    if (m) bits.push(`${m.x.length} × ${m.y.length} · ${friendlyAxis(m.x_label).split(' [')[0]} ${fmt(Math.min(...m.x), 6)}–${fmt(Math.max(...m.x), 6)}, ${friendlyAxis(m.y_label).split(' [')[0]} ${fmt(Math.min(...m.y), 6)}–${fmt(Math.max(...m.y), 6)}`);
  } else {
    const x = f.data[f.detector]?.raw?.x || [];
    if (x.length) bits.push(`${x.length} points · ${fmt(x[0], 5)}–${fmt(x[x.length - 1], 5)}${i.type === 'TimeTrace' ? ' s' : ''}`);
  }
  if (meta.save_type === 'checkpoint' || meta.run_status === 'checkpoint') bits.push(`Checkpoint, ${meta.completed_iterations}/${meta.target_iterations} iterations`);
  else if (meta.target_iterations) bits.push(`${meta.completed_iterations}/${meta.target_iterations} iterations`);
  if (meta.rf_frequency_ghz) bits.push(`RF ${meta.rf_frequency_ghz} GHz`);
  if (i.irregular) bits.push(h('span', { style: 'color:var(--warn)' }, `Irregular sampling`));
  if (i.reference_only) bits.push('Reference only');
  if (i.unit) bits.push(i.unit);
  if (i.detectors.length > 1) bits.push(`${i.detectors.length} detectors`);
  const ctlr = h('div', { class: 'row wrap' });
  if (i.dim !== 'scan2d') {
    ctlr.append(h('label', { class: 'row' }, h('span', { class: 'lbl' }, 'Experiment'),
      select([['', `Auto · ${i.type}`], ...['Ramsey', 'Rabi', 'SpinEcho', 'DynamicDecoupling', 'T1', 'ODMR', 'DEERFrequency', 'DEERDuration', 'DEERDecay', 'DEERPosition', 'LineScan', 'TimeTrace'].map((t) => [t, t])],
        f.fitSpec.experiment, (v) => { commit('experiment'); f.fitSpec.experiment = v; emit(); }, { style: 'width:150px', title: 'Override the detected experiment type (changes the fit model family)' })));
  }
  if (i.detectors.length > 1) {
    ctlr.append(seg(i.detectors.map((d) => [d, detLabel(f, d)]), f.detector, (d) => { commit('detector'); f.detector = d; emit(); }));
  }
  ctlr.append(seg(['contrast', 'signal', 'reference', 'difference'].map((m) => [m, MODE_LABEL[m], i.observables.includes(m) ? '' : 'Not recorded in this file']), f.mode,
    (m) => setMode(f, m), { disabled: ['contrast', 'signal', 'reference', 'difference'].filter((m) => !i.observables.includes(m)) }));
  return h('div', { class: 'row wrap', style: 'justify-content:space-between;align-items:flex-start;gap:12px;flex:none' },
    h('div', { class: 'filehead', style: 'min-width:0' },
      h('h1', { title: f.path }, i.name),
      h('div', { class: 'metaline' }, tag(i.tag), bits.map((b, n) => [n ? h('span', { class: 'dim' }, '·') : null, b]))),
    ctlr);
}

function layerChip(f, key, label, sw, opts = {}) {
  return h('button', { class: `chip ${f.layers[key] ? 'on' : ''}`, disabled: opts.disabled, title: opts.title || '',
    onclick: () => { f.layers[key] = !f.layers[key]; f.layersTouched = true; emit(); } },
  sw ? h('span', { class: 'sw', style: `background:${sw};${opts.tall ? 'height:8px' : ''}` }) : null, label);
}

function toolbar(f) {
  const d = f.data[f.detector];
  const fit = f.fit[f.detector];
  const tt = kindOf(f) === 'tt';
  const chips = h('div', { class: 'row wrap', style: 'gap:6px' });
  if (tt) {
    chips.append(seg([['stacked', 'Stacked'], ['shared', 'Shared axis'], ['right', 'Right axis'], ['percent', '% of A'], ['single', 'Active only']].filter(([k]) => f.info.detectors.length > 1 || k === 'single' || k === 'percent'),
      f.info.detectors.length > 1 ? f.tt.layout : (f.tt.layout === 'percent' ? 'percent' : 'single'), (v) => { f.tt.layout = v; emit(); }));
    chips.append(layerChip(f, 'fit', 'Decay fit', '#F08A3C', { disabled: !fit }),
      h('button', { class: `chip ${f.tt.showRolling ? 'on' : ''}`, onclick: () => { f.tt.showRolling = !f.tt.showRolling; emit(); } }, h('span', { class: 'sw', style: 'background:#E6E8EC' }), `Rolling mean ${f.tt.rolling} s`),
      h('button', { class: `chip ${f.tt.showWindows ? 'on' : ''}`, onclick: () => { f.tt.showWindows = !f.tt.showWindows; emit(); } }, h('span', { class: 'sw', style: 'background:#9B7BEA;height:8px' }), 'Windows A / B'),
      h('button', { class: `chip ${f.tt.showRetries ? 'on' : ''}`, onclick: () => { f.tt.showRetries = !f.tt.showRetries; emit(); } }, `Retries ${d?.time?.retries?.length || 0}`),
      h('button', { class: `chip ${f.tt.showEvents ? 'on' : ''}`, onclick: () => { f.tt.showEvents = !f.tt.showEvents; emit(); } }, `Events ${d?.time?.events?.length || 0}`),
      layerChip(f, 'legend', 'Legend'));
  } else {
    const its = d?.iterations;
    chips.append(
      layerChip(f, 'data', 'Data', '#5A9BEA'), layerChip(f, 'fit', 'Fit', '#F08A3C', { disabled: !fit }),
      layerChip(f, 'band', f.errorMode === 'bars' ? '± SEM bars' : '± SEM', '#3D5E8C', { tall: true, disabled: !d?.proc?.sem, title: d?.proc?.sem ? 'Standard error of the mean across iterations' : 'No per-point uncertainty in this file' }),
      layerChip(f, 'res', 'Residuals', null, { disabled: !fit }), layerChip(f, 'smooth', 'Smoothed', '#222'),
      h('button', { class: `chip ${f.layers.iter ? 'on' : ''}`, disabled: !its, title: its ? 'Show individual iterations (choose which in Process)' : 'No per-iteration data',
        onclick: () => { f.layers.iter = !f.layers.iter; emit(); } }, 'Iterations ', h('span', { class: 'mono dim' }, its ? (f.iterSel?.length || its.count) : '–')),
      layerChip(f, 'meanSd', 'Mean ± SD', null, { disabled: !its }),
      layerChip(f, 'envelope', 'Envelope', '#7A4FD6', { disabled: !fit?.envelope, title: 'Rabi decay envelope (T2ρ)' }),
      layerChip(f, 'sigref', 'Signal + Ref (right axis)', null, { disabled: f.mode !== 'contrast' || f.info.reference_only }),
      layerChip(f, 'fft', 'FFT', null, { disabled: f.info.irregular }),
      layerChip(f, 'peaks', 'Peak markers', null, { disabled: !fit?.params?.some((p) => /^(x0|center)/.test(p.name)) }),
      f.info.detectors.length > 1 ? h('button', { class: `chip ${f.compareDetectors ? 'on' : ''}`, onclick: async () => { f.compareDetectors = !f.compareDetectors; emit(); } }, 'Both detectors') : null,
      layerChip(f, 'legend', 'Legend'), layerChip(f, 'results', 'Results box', null, { disabled: !fit }));
  }
  const tools = h('div', { class: 'row', style: 'gap:2px' },
    seg([['both', 'Line + points'], ['pts', 'Points'], ['line', 'Line']], f.styleMode, (v) => { f.styleMode = v; emit(); }),
    h('span', { class: 'sep' }),
    h('button', { class: 'ibtn', title: 'Reset view (Ctrl+Home)', onclick: () => resetView(f) }, icon('home')),
    h('button', { class: `ibtn ${f.tool === 'zoom' || !f.tool ? 'on' : ''}`, title: 'Box zoom (scroll also zooms)', onclick: () => setTool(f, 'zoom') }, icon('zoom')),
    h('button', { class: `ibtn ${f.tool === 'pan' ? 'on' : ''}`, title: 'Pan', onclick: () => setTool(f, 'pan') }, icon('pan')),
    tt ? h('button', { class: `btn sm ${f.ttPick === 'a' ? 'primary' : ''}`, title: 'Then drag a box on the plot to set window A', onclick: () => setWindowByDrag(f, 'a') }, 'Set A') : null,
    tt ? h('button', { class: `btn sm ${f.ttPick === 'b' ? 'primary' : ''}`, title: 'Then drag a box on the plot to set window B', onclick: () => setWindowByDrag(f, 'b') }, 'Set B') : null,
    tt ? null : h('button', { class: `ibtn ${f.tool === 'select' ? 'on' : ''}`, title: 'Box-select points to exclude from the fit', onclick: () => setTool(f, 'select') }, icon('edit')),
    h('button', { class: `ibtn ${f.tool === 'measure' ? 'on' : ''}`, title: 'Measure: click two points', onclick: () => { setTool(f, f.tool === 'measure' ? 'zoom' : 'measure'); f.measure = []; } }, icon('measure')),
    h('span', { class: 'sep' }),
    h('button', { class: 'btn', onclick: () => { S.editor.open = !S.editor.open; emit(); } }, icon('edit', 14), S.editor.open ? 'Close editor' : 'Edit plot'),
    h('button', { class: 'btn', title: 'Copy figure (Ctrl+Shift+C)', onclick: () => copyFigure(f) }, icon('copy', 14), 'Copy', h('span', { class: 'kbd' }, 'Ctrl Shift C')));
  return h('div', { class: 'toolbar', style: 'flex:none' }, chips, tools);
}

function setTool(f, tool) {
  f.tool = tool;
  if (plotEl?.data) Plotly.relayout(plotEl, { dragmode: f.ttPick ? 'select' : tool === 'pan' ? 'pan' : tool === 'select' ? 'select' : 'zoom' });
  emit();
}

function plotSection(f) {
  const card = h('section', { class: 'plotcard', 'aria-label': 'Plot' }, plotDiv());
  if (f.fitStale) card.append(h('div', { style: 'position:absolute;right:12px;bottom:10px;z-index:2', class: 'row' },
    h('span', { class: 'pill WARN' }, 'Fit is from earlier settings'), h('button', { class: 'btn sm', onclick: () => runFit(f) }, 'Refit')));
  if (f.data[f.detector]?.baseline_note) card.append(h('div', { style: 'position:absolute;left:12px;bottom:8px;z-index:2;color:#4A5262;font-size:11.5px' }, f.data[f.detector].baseline_note));
  if (kindOf(f) === '2d') {
    const cuts = linecuts(f);
    const wrap = h('div', { class: 'row', style: 'flex:1;min-height:380px;gap:10px;align-items:stretch' }, card);
    if (cuts) {
      cutEl = cutEl || h('div', { class: 'plot' });
      wrap.append(h('section', { class: 'plotcard', style: 'flex:0 0 32%;min-width:260px' }, cutEl));
    }
    return wrap;
  }
  return card;
}

function drawPlot(f) {
  const el = plotDiv();
  const k = kindOf(f);
  let fig;
  if (k === '1d') fig = figure1D(f);
  else if (k === 'tt') fig = figureTT(f);
  else fig = figureMaps(f, overlayCache[f.path]);
  if (!fig) return;
  fig.layout.dragmode = f.ttPick && k === 'tt' ? 'select' : f.tool === 'pan' ? 'pan' : f.tool === 'select' ? 'select' : 'zoom';
  requestAnimationFrame(() => {
    if (!roomy(el)) { plotPending = true; return; }
    Plotly.react(el, fig.data, fig.layout, plotConfig({ edits: { ...plotConfig().edits, titleText: !!f.style.title } }));
    bindPlot(el);
    if (k === '2d') { fitColorbars(el); drawCuts(f); }
  });
}

function drawCuts(f) {
  const cuts = linecuts(f);
  if (!cuts || !cutEl) return;
  const colors = ['#D63384', '#11998C', '#1F5FBF'];
  const data = [], st = baseStyle(f.style);
  const norm = f.map.cutNorm !== false && cuts.length > 1;
  const nz = (arr) => { if (!norm) return arr; const m = Math.max(...arr.filter(Number.isFinite)); return arr.map((v) => v / (m || 1)); };
  cuts.forEach((c, i) => {
    if (c.h) c.h = { ...c.h, y: nz(c.h.y) };
    if (c.v) c.v = { ...c.v, y: nz(c.v.y) };
    if (c.h) data.push({ type: 'scatter', mode: 'lines+markers', x: c.h.x, y: c.h.y, name: `${c.label} · horizontal`, line: { color: colors[i], width: 1.6 }, marker: { size: 4 } });
    if (c.v) data.push({ type: 'scatter', mode: 'lines+markers', x: c.v.x, y: c.v.y, name: `${c.label} · vertical`, line: { color: colors[i], width: 1.6, dash: 'dot' }, marker: { size: 4 }, xaxis: c.h ? 'x2' : 'x', yaxis: c.h ? 'y2' : 'y' });
  });
  const both = cuts[0].h && cuts[0].v;
  const layout = {
    margin: { l: 56, r: 12, t: 30, b: 46 }, font: { family: 'Arial', size: 11 }, showlegend: true, legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
    title: { text: (cuts[0].h || cuts[0].v).label, font: { size: 12 } },
    xaxis: { title: { text: cuts[0].xTitle ? cuts[0].xTitle : cuts[0].h ? friendlyAxis(f.map.swap ? f.data[f.detector].map.y_label : f.data[f.detector].map.x_label) : '' }, showline: true, mirror: true, ticks: 'outside', domain: [0, 1], anchor: 'y' },
    yaxis: { showline: true, mirror: true, ticks: 'outside', domain: both ? [0.58, 1] : [0, 1], exponentformat: 'SI', title: { text: norm ? 'Normalized to max' : (f.info.unit || '') } },
    separators: '.',
  };
  if (both) { layout.xaxis2 = { showline: true, mirror: true, ticks: 'outside', anchor: 'y2' }; layout.yaxis2 = { showline: true, mirror: true, ticks: 'outside', domain: [0, 0.4], exponentformat: 'SI' }; }
  Plotly.react(cutEl, data, layout, { displayModeBar: false, responsive: true });
}

function bindPlot(el) {
  if (plotBound) return;
  plotBound = true;
  el.on('plotly_click', (ev) => {
    const f = active();
    if (!f || !ev.points?.length) return;
    const p = ev.points[0];
    if (kindOf(f) === '2d') {
      if (p.data.type !== 'heatmap') return;
      if (f.map.linecut === 'line') {
        // Display coordinates (after swap); a third click starts a new line.
        const L = f.map.line;
        f.map.line = L && L.length === 1 ? [L[0], [p.x, p.y]] : [[p.x, p.y]];
        emit(); return;
      }
      f.map.cursor = [p.x, p.y];
      if (f.map.linecut === 'none') f.map.linecut = 'h';
      emit(); return;
    }
    if (f.tool === 'measure') {
      f.measure = [...(f.measure || []), { x: p.x, y: p.y }].slice(-2);
      emit(); return;
    }
    const key = p.data.meta?.key;
    if (key !== 'data' && !String(key || '').startsWith('tt:')) return;
    const pin = { x: p.x, y: p.y, i: p.pointIndex, key };
    const multi = ev.event.ctrlKey || ev.event.shiftKey || ev.event.metaKey;
    const exists = f.pins.findIndex((q) => q.i === pin.i && q.key === key);
    if (exists >= 0) f.pins.splice(exists, 1);
    else f.pins = multi ? [...f.pins, pin] : [pin];
    emit();
  });
  // Live hover readout: edit the readout text directly (no emit/re-render), throttled to one update per frame.
  let hoverEv = null, hoverRaf = 0;
  el.on('plotly_hover', (ev) => {
    hoverEv = ev;
    if (hoverRaf) return;
    hoverRaf = requestAnimationFrame(() => {
      hoverRaf = 0;
      const f = active(), p = hoverEv?.points?.[0];
      if (!f || !p || !readoutLive || !readoutLive.isConnected) return;
      let text = null;
      if (kindOf(f) === '2d') {
        if (p.data.type !== 'heatmap') return;
        const parts = [];
        for (const det of f.info.detectors) {
          const m = f.data[det]?.map;
          if (!m) continue;
          const d = displayMap(m, f.map.swap, f.map.source || 'value');
          parts.push(`${detLabel(f, det)} ${fmt(d.z[nearest(d.y, p.y)]?.[nearest(d.x, p.x)], 6)}${f.info.unit ? ' ' + f.info.unit : ''}`);
        }
        const d0 = displayMap(f.data[f.detector].map, f.map.swap);
        text = [`${friendlyAxis(d0.xl).split(' [')[0]} ${fmt(p.x, 6)}`, `${friendlyAxis(d0.yl).split(' [')[0]} ${fmt(p.y, 6)}`, ...parts].join(' · ');
      } else {
        const key = p.data.meta?.key;
        if (key !== 'data' && !String(key || '').startsWith('tt:')) return;
        text = `x ${fmt(p.x, 6)} · y ${fmt(p.y, 6)}`;
      }
      readoutLive.textContent = text;
      readoutLive.classList.add('mono');
    });
  });
  el.on('plotly_unhover', () => {
    hoverEv = null;
    if (readoutLive?.isConnected) { readoutLive.textContent = readoutLive.dataset.def; readoutLive.classList.remove('mono'); }
  });
  el.on('plotly_relayout', (ev) => {
    const f = active();
    if (!f) return;
    if (applyPlotlyEdit(f, ev)) return;
    if (kindOf(f) === '2d') {
      if (ev['xaxis.range[0]'] !== undefined) f.map.bounds = { x: [ev['xaxis.range[0]'], ev['xaxis.range[1]']], y: [ev['yaxis.range[0]'] ?? f.map.bounds?.y?.[0], ev['yaxis.range[1]'] ?? f.map.bounds?.y?.[1]] };
      if (ev['xaxis.autorange']) f.map.bounds = null;
      return;
    }
    if (ev['xaxis.range[0]'] !== undefined) f.xrange = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
    if (ev['yaxis.range[0]'] !== undefined) f.yrange = [ev['yaxis.range[0]'], ev['yaxis.range[1]']];
    if (ev['xaxis.autorange']) f.xrange = null;
    if (ev['yaxis.autorange']) f.yrange = null;
  });
  el.on('plotly_restyle', (ev) => { const f = active(); if (f) applyPlotlyEdit(f, ev, el, 'restyle'); });
  el.on('plotly_legendclick', (ev) => {
    const f = active(); if (!f) return true;
    const key = ev.data[ev.curveNumber]?.meta?.key;
    if (!key) return true;
    if (ev.event?.detail >= 2 || ev.event?.altKey) { openSeriesEditor(f, key, ev.event.clientX, ev.event.clientY); return false; }
    f.style.series = f.style.series || {};
    const cur = f.style.series[key]?.visible;
    f.style.series[key] = { ...(f.style.series[key] || {}), visible: cur === false ? true : false };
    emit();
    return false;
  });
  el.on('plotly_legenddoubleclick', (ev) => {
    const f = active(); if (!f) return false;
    const key = ev.data[ev.curveNumber]?.meta?.key;
    if (key) openSeriesEditor(f, key, ev.event?.clientX || 400, ev.event?.clientY || 200);
    return false;
  });
  el.on('plotly_selected', (ev) => {
    const f = active();
    if (f && kindOf(f) === 'tt') {
      const r = ev?.range?.x;
      if (!f.ttPick || !r) return;
      const k = f.tt.unit === 'min' ? 60 : 1;
      const w = [Math.min(r[0], r[1]) * k, Math.max(r[0], r[1]) * k];
      commit('window');
      f.tt.windows[f.ttPick] = w;
      toast(`Window ${f.ttPick.toUpperCase()} = ${fmt(w[0], 6)} to ${fmt(w[1], 6)} s`, 'ok');
      f.ttPick = null;
      if (plotEl?.data) Promise.resolve(Plotly.relayout(plotEl, { dragmode: f.tool === 'pan' ? 'pan' : 'zoom', selections: [] })).catch(() => {});
      emit();
      return;
    }
    if (!f || !ev?.points?.length || kindOf(f) !== '1d') return;
    const d = f.data[f.detector];
    const idx = new Set(f.proc.excluded);
    ev.points.filter((p) => p.data.meta?.key === 'data').forEach((p) => (d.proc.groups[p.pointIndex] || []).forEach((r) => idx.add(r)));
    toast(`Excluded ${ev.points.length} point(s). Undo with Ctrl+Z.`);
    setProc(f, { excluded: [...idx] }, 'exclude selection');
  });
  el.addEventListener('dblclick', (e) => {
    const f = active();
    if (!f) return;
    const t = e.target.closest('.scatterlayer .trace, .legendtext, .legendtoggle');
    if (t && e.target.closest('.legend')) return; // handled by legend events
  });
  el.addEventListener('contextmenu', (e) => {
    const f = active(); if (!f) return;
    e.preventDefault();
    contextMenu(e.clientX, e.clientY, [
      { label: 'Edit plot…', run: () => { S.editor.open = true; emit(); } },
      { label: 'Copy figure', kbd: 'Ctrl Shift C', run: () => copyFigure(f) },
      { label: 'Reset view', kbd: 'Ctrl Home', run: () => resetView(f) },
      f.pins.length ? { label: `Exclude ${f.pins.length} pinned point(s)`, run: () => excludePins(f) } : null,
      f.pins.length ? { label: 'Clear pinned points', kbd: 'Esc', run: () => { f.pins = []; emit(); } } : null,
      { label: 'Add to Figure composer', run: () => import('./figure.js').then((m) => m.addCurrentPlot()) },
    ]);
  });
}

// Tab-separated table of the processed data currently shown (active detector for 1D; every detector for time traces).
function dataTable(f) {
  const cell = (v) => (v === null || v === undefined || !Number.isFinite(Number(v)) ? '' : String(v));
  const cols = [], head = [];
  const tt = kindOf(f) === 'tt';
  const dets = tt ? f.info.detectors : [f.detector];
  const xt = tt ? 'Elapsed time [s]' : xTitle(f);
  for (const det of dets) {
    const d = f.data[det];
    if (!d?.proc) continue;
    const fit = f.fit[det];
    const tag = dets.length > 1 ? ` · ${detLabel(f, det)}` : '';
    const x = d.proc.x || [];
    cols.push(x); head.push(`x${tag} (${xt})`);
    cols.push(d.proc.y || []); head.push(`y${tag} (${tt ? `${detLabel(f, det)}${f.info.unit ? ` [${f.info.unit}]` : ''}` : yTitle(f)})`);
    if (d.proc.sem) { cols.push(d.proc.sem); head.push(`sem${tag}`); }
    if (d.proc.smooth) { cols.push(d.proc.smooth); head.push(`smoothed${tag}`); }
    if (fit?.x && fit.y_fit) {
      const same = fit.x.length === x.length && fit.x.every((v, i) => Math.abs(v - x[i]) <= 1e-9 * Math.max(1, Math.abs(v)));
      const fy = x.map((xv, i) => { const j = same ? i : fit.x.findIndex((v) => Math.abs(v - xv) <= 1e-9 * Math.max(1, Math.abs(v))); return j >= 0 ? fit.y_fit[j] : null; });
      cols.push(fy); head.push(`fit${tag}`);
      cols.push(x.map((_, i) => (fy[i] === null || d.proc.y?.[i] === null ? null : d.proc.y[i] - fy[i]))); head.push(`residual${tag}`);
    }
  }
  const n = Math.max(0, ...cols.map((c) => c.length));
  const lines = [head.join('\t')];
  for (let i = 0; i < n; i++) lines.push(cols.map((c) => cell(c[i])).join('\t'));
  return lines.join('\n');
}

function copyData(f) {
  if (!f.data[f.detector]?.proc) { toast('No processed data to copy yet.'); return; }
  const t = dataTable(f);
  copyText(t, `Data copied (${t.split('\n').length - 1} rows, tab-separated)`);
}

// Time traces: drag a box on the plot to set window A or B.
function setWindowByDrag(f, which) {
  f.ttPick = f.ttPick === which ? null : which;
  if (plotEl?.data) Plotly.relayout(plotEl, { dragmode: f.ttPick ? 'select' : (f.tool === 'pan' ? 'pan' : 'zoom') });
  if (f.ttPick) toast(`Drag a box on the plot to set window ${which.toUpperCase()}.`);
  emit();
}

function readout(f) {
  const d = f.data[f.detector];
  const fit = f.fit[f.detector];
  const row = h('div', { class: 'readout' });
  readoutLive = null;
  if (f.tool === 'measure') {
    const m = f.measure || [];
    if (m.length < 2) row.append(h('span', { class: 'muted' }, `Measure: click ${2 - m.length} point${m.length ? '' : 's'} on the plot.`));
    else row.append(h('span', { class: 'muted' }, 'Δ'), h('span', { class: 'mono' }, `Δx ${fmt(m[1].x - m[0].x, 6)}`), h('span', { class: 'mono' }, `Δy ${fmt(m[1].y - m[0].y, 6)}`),
      h('span', { class: 'mono dim' }, `slope ${fmt((m[1].y - m[0].y) / (m[1].x - m[0].x), 4)}`), h('button', { class: 'btn sm', onclick: () => { f.measure = []; emit(); } }, 'Clear'));
    return row;
  }
  if (!f.pins.length) {
    const hint = h('span', { class: 'dim' }, kindOf(f) === 'tt' ? 'Click a sample to inspect its timing. Set windows A/B in the Stats tab.' : 'Click a point to pin it · Ctrl-click adds more · Esc clears · double-click a legend entry to style it · right-click for more');
    hint.dataset.def = hint.textContent; readoutLive = hint;
    row.append(hint, h('span', { class: 'sp' }), h('button', { class: 'btn sm', title: 'Copy the processed data (and fit, residual) as a tab-separated table', onclick: () => copyData(f) }, 'Copy data'));
    return row;
  }
  const p = f.pins[f.pins.length - 1];
  row.append(h('span', { class: 'row', style: 'gap:6px' }, h('span', { style: 'width:9px;height:9px;border-radius:5px;border:2px solid #E0A400;display:inline-block' }), h('span', { class: 'muted' }, f.pins.length > 1 ? `${f.pins.length} pinned` : 'Pinned')),
    h('span', { class: 'mono' }, `x ${p.x}`), h('span', { class: 'mono' }, `y ${p.y}`));
  if (kindOf(f) === 'tt') {
    const det = String(p.key).split(':')[1] || f.detector;
    const tm = f.data[det]?.time;
    const retry = tm?.retries?.find((r) => r.index === p.i);
    const sched = tm?.scheduled?.[p.i], late = tm?.lateness?.[p.i];
    row.append(h('span', { class: 'mono dim' }, `sample ${p.i + 1} · scheduled ${Number.isFinite(Number(sched)) ? fmt(Number(sched), 5) + ' s' : 'n/a'} · lateness ${Number.isFinite(Number(late)) ? fmt(Number(late), 3) + ' s' : 'n/a'} · ${retry ? `${retry.count} retry` : 'no retry'}`));
  } else {
    const sem = d?.proc?.sem?.[p.i];
    if (sem) row.append(h('span', { class: 'mono dim' }, `± ${fmt(sem, 3)} SEM`));
    if (fit) { const j = fit.x.findIndex((v) => Math.abs(v - p.x) < 1e-9 * Math.max(1, Math.abs(v))); if (j >= 0) row.append(h('span', { class: 'mono dim' }, `fit ${fmt(fit.y_fit[j], 6)}`)); }
    if (f.pins.length > 1) { const q = f.pins[f.pins.length - 2]; row.append(h('span', { class: 'mono dim' }, `Δx ${fmt(p.x - q.x, 6)} · Δy ${fmt(p.y - q.y, 5)}`)); }
    row.append(h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: () => excludePins(f) }, `Exclude ${f.pins.length > 1 ? f.pins.length + ' points' : 'point'}`));
  }
  row.append(h('button', { class: 'btn sm', onclick: () => copyText(f.pins.map((q) => `${q.x}\t${q.y}`).join('\n'), 'Pinned values copied') }, 'Copy'),
    h('button', { class: 'btn sm', title: 'Copy the processed data table', onclick: () => copyData(f) }, 'Copy data'),
    h('button', { class: 'btn sm', onclick: () => { f.pins = []; emit(); } }, 'Clear'));
  return row;
}

// ---------------------------------------------------------------- 2D toolbar
function mapToolbar(f) {
  const M = f.map;
  const set = (patch) => { Object.assign(M, patch); emit(); };
  const cms = ['viridis', 'magma', 'cividis', 'parula', 'gray'];
  const layouts = [['single', 'Single'], ...(f.info.detectors.length > 1 ? [['side', 'Side by side']] : []), ['overlay', 'RGB overlay'], ['maps+overlay', 'Maps + overlay']];
  return h('div', { style: 'display:flex;flex-direction:column;gap:8px;flex:none' },
    h('div', { class: 'row wrap', style: 'justify-content:space-between' },
      seg(layouts, M.layout, async (v) => { M.prevLayout = M.layout; M.layout = v; if (v.includes('overlay')) await ensureOverlay(f); emit(); }),
      h('div', { class: 'row', style: 'gap:2px' },
        h('button', { class: 'ibtn', title: 'Reset view (Ctrl+Home)', onclick: () => resetView(f) }, icon('home')),
        h('button', { class: `ibtn ${f.tool !== 'pan' ? 'on' : ''}`, title: 'Box zoom', onclick: () => setTool(f, 'zoom') }, icon('zoom')),
        h('button', { class: `ibtn ${f.tool === 'pan' ? 'on' : ''}`, title: 'Pan', onclick: () => setTool(f, 'pan') }, icon('pan')),
        h('span', { class: 'sep' }),
        h('button', { class: 'btn', onclick: () => { S.editor.open = !S.editor.open; emit(); } }, icon('edit', 14), 'Edit plot'),
        h('button', { class: 'btn', onclick: () => copyFigure(f) }, icon('copy', 14), 'Copy', h('span', { class: 'kbd' }, 'Ctrl Shift C')))),
    h('div', { class: 'mapbar' },
      h('div', { class: 'row', style: 'gap:6px' }, h('span', { class: 'lbl' }, 'Colormap'),
        cms.map((c) => h('button', { class: `cm ${M.cmap === c ? 'on' : ''}`, title: c, 'aria-label': c, style: `background:linear-gradient(90deg,${CMAP_GRADIENT[c]})`, onclick: () => set({ cmap: c }) })),
        select(['viridis', 'parula', 'cividis', 'magma', 'inferno', 'plasma', 'turbo', 'coolwarm', 'RdBu', 'gray'], M.cmap, (v) => set({ cmap: v }), { style: 'width:96px;height:26px' })),
      h('div', { class: 'row', style: 'gap:6px' }, h('span', { class: 'lbl' }, 'Range'),
        input(M.plo, (v) => set({ plo: num(v) ?? 1, limits: 'robust' }), { style: 'width:52px;height:26px', title: 'Low percentile' }), h('span', { class: 'dim' }, 'to'),
        input(M.phi, (v) => set({ phi: num(v) ?? 99.5, limits: 'robust' }), { style: 'width:58px;height:26px', title: 'High percentile' }), h('span', { class: 'dim' }, '%')),
      seg([['independent', 'Independent'], ['shared', 'Shared']], M.scale, (v) => set({ scale: v })),
      seg([['physical', 'Physical pixels'], ['square', 'Square']], M.geometry, (v) => { set({ geometry: v }); call('settings_set', { patch: { pixel_geometry: v } }, { quiet: true }); }),
      h('button', { class: `btn ${M.swap ? '' : ''}`, onclick: () => { commit('swap axes'); M.swap = !M.swap; M.bounds = null; M.cursor = M.cursor ? [M.cursor[1], M.cursor[0]] : null; if (M.line) M.line = M.line.map((q) => [q[1], q[0]]); if (overlayCache[f.path]) ensureOverlay(f).then(emit); emit(); } }, `Swap ${swapLabel(f)}`),
      seg([['none', 'No cut'], ['h', 'Horizontal'], ['v', 'Vertical'], ['cross', 'Both'], ['line', 'Line']], M.linecut, (v) => { if (v !== 'none' && v !== 'line' && !M.cursor) M.cursor = peakCell(f, f.detector); set({ linecut: v }); }),
      h('button', { class: 'btn sm', onclick: () => { M.cursor = peakCell(f, f.detector); if (M.linecut === 'none') M.linecut = 'h'; emit(); } }, 'Cut at peak'),
      h('button', { class: 'btn sm', onclick: () => { M.cursor = peakCell(f, f.detector, true); if (M.linecut === 'none') M.linecut = 'h'; emit(); } }, 'Cut at dip')),
    mapReadout(f));
}

const swapLabel = (f) => { const m = f.data[f.detector]?.map; return m ? `${friendlyAxis(m.x_label).split(' [')[0]} / ${friendlyAxis(m.y_label).split(' [')[0]}` : 'axes'; };

function mapReadout(f) {
  const cuts = linecuts(f);
  const row = h('div', { class: 'readout' });
  readoutLive = null;
  if (f.map.linecut === 'line') {
    const L = f.map.line || [];
    const dd = displayMap(f.data[f.detector].map, f.map.swap);
    const nm = (l) => friendlyAxis(l).split(' [')[0];
    if (L.length < 2) row.append(h('span', { class: 'muted' }, L.length ? 'Click the second point on the map' : 'Click two points on the map'));
    else row.append(h('span', { class: 'muted' }, 'Line'), h('span', { class: 'mono' }, `length ${fmt(lineLength(L), 6)} (axis units)`),
      h('span', { class: 'mono dim' }, `(${nm(dd.xl)}, ${nm(dd.yl)}): (${fmt(L[0][0], 6)}, ${fmt(L[0][1], 6)}) to (${fmt(L[1][0], 6)}, ${fmt(L[1][1], 6)})`));
    row.append(h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: () => { f.map.cursor = null; f.map.line = null; f.map.linecut = 'none'; emit(); } }, 'Clear'));
    return row;
  }
  if (!f.map.cursor) {
    const hint = h('span', { class: 'dim' }, 'Click the map to pin a cell and draw linecuts · scroll to zoom · drag to box-zoom · Ctrl+Home resets');
    hint.dataset.def = hint.textContent; readoutLive = hint;
    row.append(hint); return row;
  }
  const m = f.data[f.detector].map;
  const d = displayMap(m, f.map.swap);
  row.append(h('span', { class: 'muted' }, 'Cursor'), h('span', { class: 'mono' }, `${friendlyAxis(d.xl).split(' [')[0]} ${f.map.cursor[0]}`), h('span', { class: 'mono' }, `${friendlyAxis(d.yl).split(' [')[0]} ${f.map.cursor[1]}`));
  (cuts || f.info.detectors.map((det) => ({ det, label: detLabel(f, det), value: (() => { const dd = displayMap(f.data[det].map, f.map.swap); return dd.z[nearest(dd.y, f.map.cursor[1])][nearest(dd.x, f.map.cursor[0])]; })() })))
    .forEach((c) => row.append(h('span', { class: 'mono' }, `${c.label} ${fmt(c.value, 6)}${f.info.unit ? ' ' + f.info.unit : ''}`)));
  row.append(h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: () => { f.map.cursor = null; f.map.line = null; f.map.linecut = 'none'; emit(); } }, 'Clear'));
  return row;
}

export async function ensureOverlay(f) {
  const M = f.map;
  let second = M.compareWith;
  if (!second && f.info.detectors.length > 1) second = { path: f.path, detector: f.info.detectors.find((d) => d !== f.detector) };
  if (!second) { toast('Add another 2D file of the same plane, then pick it in Compare.'); return; }
  try {
    overlayCache[f.path] = await call('map_overlay', { first: { path: f.path, detector: f.detector, mode: f.mode, source: M.source || 'value' }, second: { ...second, mode: f.mode, source: M.source || 'value' },
      opts: { shared: M.overlayScale === 'shared', background: M.background, palette: M.palette, swap: M.swap } }, { label: 'Building overlay' });
  } catch (e) { overlayCache[f.path] = null; }
}

// ---------------------------------------------------------------- inspector
const TABS = {
  '1d': [['fit', 'Fit'], ['proc', 'Process'], ['plot', 'Plot'], ['export', 'Export'], ['info', 'Info']],
  tt: [['stats', 'Stats'], ['fit', 'Fit'], ['proc', 'Process'], ['plot', 'Plot'], ['export', 'Export'], ['info', 'Info']],
  '2d': [['map', 'Map'], ['spots', 'Spots'], ['compare', 'Compare'], ['export', 'Export'], ['info', 'Info']],
};
// Alt+1..7 keep the original app's numbering: Model, Params, Process, Plot, Results, Map, Export.
export const ALT_TABS = { 1: 'fit', 2: 'fit', 3: 'proc', 4: 'plot', 5: 'fit', 6: 'map', 7: 'export', 8: 'info' };

export function renderInspector(f, insp) {
  const k = kindOf(f);
  const tabs = TABS[k];
  if (!tabs.some(([t]) => t === f.tab)) f.tab = tabs[0][0];
  const body = h('div', { class: 'insp-body' });
  const builders = { fit: fitTab, proc: procTab, plot: plotTab, export: exportTab, info: infoTab, stats: statsTab, map: mapTab, spots: spotsTab, compare: mapCompareTab };
  body.append(builders[f.tab](f));
  mount(insp, h('div', { class: 'tabs', role: 'tablist' }, tabs.map(([t, l]) => h('button', { class: f.tab === t ? 'on' : '', role: 'tab', onclick: () => { f.tab = t; emit('insp'); } }, l))), body);
}

// ---- Fit tab
const MODELS_BY_FAMILY = {
  Rabi: [['Auto compare', 'Auto compare (fit quality + BIC)'], ['Adaptive pulse-area (recommended)', 'Adaptive pulse area'], ['Phase-ramp', 'Phase ramp'], ['Long damped Rabi', 'Long damped'], ['Chirped cosine', 'Chirped cosine'], ['Constant-frequency damped cosine', 'Constant-frequency damped cosine']],
};
const MODEL_HELP = {
  'Auto compare': 'Fits every Rabi model and keeps the best by fit quality and BIC, rejecting pathological adaptive solutions.',
  'Adaptive pulse-area (recommended)': 'Fits an explicit delay, a stretched decay envelope and a smooth pulse-area calibration. chirp1/chirp2 describe rotation rate vs programmed duration, not carrier chirp.',
  'Phase-ramp': 'Exponential amplitude turn-on (τr) plus a stretched decay envelope.',
  'Long damped Rabi': 'Emphasizes the early calibration lobes of long, strongly decayed scans.',
  'Chirped cosine': 'Comparison model with a linear frequency chirp.',
  'Constant-frequency damped cosine': 'Classic damped cosine; useful as a reference.',
};

function fitTab(f) {
  const s = f.fitSpec;
  const fam = family(f);
  const fit = f.fit[f.detector];
  const set = (patch, label = 'fit settings') => { commit(label); Object.assign(s, patch); emit('insp'); };
  const out = h('div', { style: 'display:flex;flex-direction:column;gap:12px' });
  const generic = [['Auto', `Auto · ${fam}`], ['GaussianPeaks', 'Gaussian peaks'], ['CustomModel', 'Custom expression']];
  if (kindOf(f) === 'tt') generic.splice(0, 1, ['Auto', 'TimeTraceDecay · background + A·exp(−t/τ)']);
  out.append(field('Model', select(generic, s.model, (v) => set({ model: v }))));
  if (s.model === 'Auto') {
    if (fam === 'Rabi') {
      out.append(field('Rabi model', select(MODELS_BY_FAMILY.Rabi, s.rabi_mode, (v) => set({ rabi_mode: v }))), h('div', { class: 'note' }, MODEL_HELP[s.rabi_mode] || ''));
    } else if (fam === 'Ramsey') {
      out.append(field('Ramsey model', select(['Auto', 'Simple damped cosine', 'N14 hyperfine', 'Extended'], s.ramsey_model, (v) => set({ ramsey_model: v }))),
        h('div', { class: 'note' }, 'Auto fits all three and prefers models with R² ≥ 0.85, then fewer bound hits, then lower BIC.'));
    } else if (fam === 'ODMR') {
      out.append(h('div', { class: 'g2' }, field('Polarity', select(['Auto', 'Peak', 'Dip'], s.odmr_polarity, (v) => set({ odmr_polarity: v }))),
        field('Peaks', select([['auto', 'Auto'], ['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], s.odmr_auto_peaks ? 'auto' : String(s.odmr_peaks), (v) => set(v === 'auto' ? { odmr_auto_peaks: true } : { odmr_auto_peaks: false, odmr_peaks: Number(v), odmr_multi: Number(v) > 1 })))),
      check('Multi-Lorentzian when several peaks are found', s.odmr_multi, (v) => set({ odmr_multi: v })),
      h('div', { class: 'g2' }, field('Centers from', input(s.odmr_ranges[0]?.[0] ?? '', (v) => set({ odmr_ranges: [[num(v), s.odmr_ranges[0]?.[1] ?? null]] }), { placeholder: 'any' })),
        field('to', input(s.odmr_ranges[0]?.[1] ?? '', (v) => set({ odmr_ranges: [[s.odmr_ranges[0]?.[0] ?? null, num(v)]] }), { placeholder: 'any' }))));
    } else if (fam === 'T1') {
      out.append(field('Preprocess', seg([['Auto', 'Auto'], ['Direct T1', 'Direct'], ['Normalize then 1-data', 'Normalize, 1 − data']], s.t1_prep, (v) => set({ t1_prep: v }))),
        field('Shape', seg([['Regular exponential', 'Exponential'], ['Stretched exponential', 'Stretched']], s.t1_family, (v) => set({ t1_family: v }))));
    } else if (fam === 'SpinEcho') {
      out.append(h('div', { class: 'note' }, 'Stretched exponential y0 + A·exp(−(t/T2)ⁿ). Lock n in Seeds & bounds to force n = 1, 2 or 3.'));
    } else if (fam === 'LineScan' || fam === 'Scan2D') {
      out.append(h('div', { class: 'note' }, 'Line scans are plot-only under Auto. Choose Gaussian peaks to fit fluorescence peaks.'));
    }
  }
  if (s.model === 'GaussianPeaks') out.append(field('Number of peaks', select([1, 2, 3, 4, 5, 6, 7, 8], s.peaks, (v) => set({ peaks: Number(v) }))),
    h('div', { class: 'note' }, 'Positive Gaussians on a shared constant background. Centers, σ and FWHM are in axis units. A fitted component is not proof of a separate emitter.'));
  if (s.model === 'CustomModel') {
    const ta = h('textarea', { class: 'field', rows: 2 }, s.custom_expr);
    ta.addEventListener('change', () => set({ custom_expr: ta.value, custom_params: exprParams(ta.value, s.custom_params) }));
    if (!s.custom_params.length) s.custom_params = exprParams(s.custom_expr);
    out.append(field('Expression in x (numpy functions: exp, log, sqrt, sin, cos, abs, clip)', ta),
      h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, 'Param'), h('th', {}, 'Seed'), h('th', {}, 'Min'), h('th', {}, 'Max')),
        s.custom_params.map((p) => h('tr', {}, h('td', { class: 'txt' }, p.name), ['seed', 'min', 'max'].map((k) => h('td', {}, input(p[k] ?? '', (v) => { p[k] = num(v); set({}); }, { placeholder: k === 'seed' ? '1' : '–' })))))));
  }
  // range
  const fr = f.data[f.detector]?.proc?.x || [];
  out.append(h('div', { class: 'f' }, h('span', { class: 'lbl' }, `Fit range${f.info.type === 'TimeTrace' ? ' (s)' : ''}`),
    h('div', { class: 'row' }, input(s.range_min ?? '', (v) => set({ range_min: num(v) }), { placeholder: fr.length ? fmt(Math.min(...fr.filter(Number.isFinite)), 5) : 'min', style: 'width:90px' }), h('span', { class: 'dim' }, 'to'),
      input(s.range_max ?? '', (v) => set({ range_max: num(v) }), { placeholder: fr.length ? fmt(Math.max(...fr.filter(Number.isFinite)), 5) : 'max', style: 'width:90px' }),
      h('button', { class: 'btn sm', title: 'Use the current zoomed x range', onclick: () => set({ range_min: f.xrange?.[0] ?? null, range_max: f.xrange?.[1] ?? null }) }, 'Use view'),
      h('button', { class: 'btn sm', onclick: () => set({ range_min: null, range_max: null }) }, 'All'))));
  // options
  const opts = h('details', { class: 'card pad' }, h('summary', {}, `Options · multistart ${s.multistart}${fam === 'Rabi' ? ` · SEM weights ${s.sem_weights ? 'on' : 'off'}` : ''}${s.apply_both ? ' · both detectors' : ''}`),
    h('div', { style: 'display:flex;flex-direction:column;gap:8px;margin-top:8px' },
      h('div', { class: 'g2' }, field('Multistart', input(s.multistart, (v) => set({ multistart: Math.max(4, num(v) || 12) }))),
        fam === 'Rabi' ? field('Delay / dead time (ns)', input(s.dead_time_ns || '', (v) => set({ dead_time_ns: num(v) || 0 }), { placeholder: 'fit' })) : h('span')),
      fam === 'Rabi' ? [check('Weight by iteration SEM', s.sem_weights, (v) => set({ sem_weights: v })),
        check('Lock to π / π/2 times', s.pi_lock, (v) => set({ pi_lock: v })),
        s.pi_lock ? h('div', { class: 'g2' }, field('π (ns)', input(s.pi_ns, (v) => set({ pi_ns: num(v) }))), field('π/2 (ns)', input(s.pi2_ns, (v) => set({ pi2_ns: num(v) })))) : null] : null,
      check('Fit smoothed data', s.fit_smoothed, (v) => set({ fit_smoothed: v })),
      f.proc.steps.length ? check('Fit the transformed data', s.fit_transformed, (v) => set({ fit_transformed: v })) : null,
      f.info.detectors.length > 1 ? field('Apply to', seg([[false, 'Active detector'], [true, 'Both detectors']], s.apply_both, (v) => set({ apply_both: v }))) : null));
  out.append(h('div', { class: 'row' },
    h('button', { class: 'btn primary', style: 'flex:1;justify-content:center', disabled: f.fitting, onclick: () => runFit(f) }, f.fitting ? h('span', { class: 'spinner' }) : null, 'Run fit', h('span', { class: 'kbd' }, 'Ctrl F')),
    h('button', { class: 'btn', style: 'flex:1;justify-content:center', disabled: f.fitting, title: 'Fits subsets of the data and keeps the candidate that best describes the full trace', onclick: () => runFit(f, { robust: true }) }, 'Robust fit', h('span', { class: 'kbd' }, 'Ctrl Shift F'))));
  if (fit) out.append(resultCard(f, fit), paramCard(f, fit));
  if (f.fitHistory.length > 1) out.append(historyCard(f));
  return out;
}

function resultCard(f, fit) {
  const lines = resultLines(fit, f.info).filter((l) => l.key !== 'r2');
  const tiles = lines.slice(0, 6).map((l) => h('div', { class: 'metric' }, h('div', { class: 'lbl' }, l.label),
    h('div', { class: 'val' }, l.err ? fmtPm(l.value, l.err).split(' ± ')[0] : fmt(l.value, 5), l.unit ? h('small', {}, ' ' + l.unit) : null),
    l.err ? h('div', { class: 'sub' }, `± ${fmt(l.err, 2)}`) : (l.key === 'pi_time_ns' && fit.metrics.programmed_pi_time_ns ? h('div', { class: 'sub' }, `programmed ${fmt(fit.metrics.programmed_pi_time_ns, 4)} ns`) : l.key === 'pi_over_2_time_ns' && fit.metrics.programmed_pi_over_2_time_ns ? h('div', { class: 'sub' }, `programmed ${fmt(fit.metrics.programmed_pi_over_2_time_ns, 4)} ns`) : null)));
  const card = h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:10px' },
    h('div', { class: 'row' }, h('span', { style: 'font-weight:600' }, `Result${f.info.detectors.length > 1 ? ' · ' + detLabel(f, f.detector) : ''}`), h('span', { class: 'sp' }), h('span', { class: `pill ${fit.status}`, title: fit.reason }, fit.status)),
    h('div', { class: 'note' }, `${fit.model} · ${fit.selection || ''}`),
    tiles.length ? h('div', { class: 'g2' }, tiles) : null,
    h('div', { class: 'kv' }, h('span', { class: 'k' }, 'R²'), h('span', { class: 'v' }, fmt(fit.r2, 5)), h('span', { class: 'k' }, 'RMSE'), h('span', { class: 'v' }, fmt(fit.rmse, 4)),
      h('span', { class: 'k' }, 'Durbin–Watson'), h('span', { class: 'v' }, fmt(fit.dw, 3)), h('span', { class: 'k' }, 'AIC / BIC'), h('span', { class: 'v' }, `${fmt(fit.aic, 5)} / ${fmt(fit.bic, 5)}`),
      h('span', { class: 'k' }, 'Bound hits'), h('span', { class: 'v' }, String(fit.bound_hits))),
    fit.status !== 'PASS' ? h('div', { class: 'note', style: 'color:var(--warn)' }, fit.reason) : null,
    fit.candidates ? h('div', { class: 'note' }, 'Candidates: ' + fit.candidates.map((c) => `${c.model} R² ${fmt(c.r2, 3)}`).join(' · ')) : null,
    h('details', {}, h('summary', {}, 'All derived values'), h('div', { class: 'kv', style: 'margin-top:6px' }, Object.entries(fit.metrics || {}).flatMap(([k, v]) => [h('span', { class: 'k' }, k), h('span', { class: 'v' }, typeof v === 'number' ? fmt(v, 6) : String(v))])),
      h('pre', { class: 'note', style: 'white-space:pre-wrap;font-family:var(--mono);font-size:11px' }, fit.message)),
    h('div', { class: 'row' }, h('button', { class: 'btn', style: 'flex:1;justify-content:center', onclick: () => copyText(slidesTable(f, fit), 'Results copied as a table for slides') }, 'Copy for slides'),
      h('button', { class: 'btn', style: 'flex:1;justify-content:center', onclick: () => copyText(notebookText(f, fit), 'Results copied for your notebook') }, 'Copy to notebook')));
  return card;
}

function slidesTable(f, fit) {
  const rows = [['Quantity', 'Value', 'Unit']];
  resultLines(fit, f.info).forEach((l) => rows.push([l.label, l.err ? fmtPm(l.value, l.err) : fmt(l.value, 5), l.unit || '']));
  return rows.map((r) => r.join('\t')).join('\n');
}
function notebookText(f, fit) {
  const lines = [`${f.info.name}${f.info.detectors.length > 1 ? ` (${detLabel(f, f.detector)})` : ''}`, `${fit.model}, ${MODE_LABEL[f.mode]}, R² = ${fmt(fit.r2, 4)} (${fit.status})`];
  resultLines(fit, f.info).filter((l) => l.key !== 'r2').forEach((l) => lines.push(`  ${l.label} = ${l.err ? fmtPm(l.value, l.err) : fmt(l.value, 5)}${l.unit ? ' ' + l.unit : ''}`));
  if (f.fitSpec.range_min !== null || f.fitSpec.range_max !== null) lines.push(`  fit range ${f.fitSpec.range_min ?? 'start'} to ${f.fitSpec.range_max ?? 'end'}`);
  lines.push(`  ${f.path}`);
  return lines.join('\n');
}

function paramCard(f, fit) {
  const s = f.fitSpec;
  f.pv = f.pv || 'res';
  const card = h('div', { class: 'card pad' });
  card.append(h('div', { class: 'row', style: 'margin-bottom:4px' }, h('span', { style: 'font-weight:600' }, 'Parameters'), h('span', { class: 'sp' }),
    seg([['res', 'Result'], ['seed', 'Seeds & bounds']], f.pv, (v) => { f.pv = v; emit('insp'); })));
  if (f.pv === 'res') {
    card.append(h('table', { class: 'tbl' }, fit.params.map((p) => h('tr', { title: p.desc },
      h('td', { class: 'txt', style: 'color:var(--text2)' }, `${p.name}${p.unit && p.unit !== '-' ? ` (${p.unit})` : ''}`),
      h('td', {}, fmt(p.value, 6)), h('td', { style: 'color:var(--text3)' }, p.error ? `± ${fmt(p.error, 2)}` : 'n/a'),
      h('td', { style: 'width:14px;color:var(--warn)', title: p.bound_hit ? 'At a bound' : '' }, p.bound_hit ? '⚑' : '')))));
    if (fit.params.some((p) => !p.error)) card.append(h('div', { class: 'note', style: 'margin-top:6px' }, 'n/a: the covariance gives no usable standard error for this parameter (degenerate or at a bound).'));
  } else {
    s.params = s.params || {};
    card.append(h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, 'Param'), h('th', {}, 'Value'), h('th', {}, 'Min'), h('th', {}, 'Max'), h('th', {}, 'Lock')),
      fit.params.map((p) => {
        const e = s.params[p.name] || {};
        const upd = (patch) => { commit('parameter bounds'); s.params[p.name] = { ...e, ...patch }; emit('insp'); };
        return h('tr', {}, h('td', { class: 'txt' }, p.name), h('td', {}, input(e.value ?? '', (v) => upd({ value: num(v) }), { placeholder: fmt(p.value, 4) })),
          h('td', {}, input(e.min ?? '', (v) => upd({ min: num(v) }), { placeholder: '–' })), h('td', {}, input(e.max ?? '', (v) => upd({ max: num(v) }), { placeholder: '–' })),
          h('td', {}, h('input', { type: 'checkbox', checked: !!e.lock, 'aria-label': `Lock ${p.name}`, onchange: (ev) => upd({ lock: ev.target.checked, value: e.value ?? p.value }) })));
      })),
    h('div', { class: 'row', style: 'margin-top:8px' }, h('button', { class: 'btn sm', onclick: () => { commit('seeds'); fit.params.forEach((p) => { s.params[p.name] = { ...(s.params[p.name] || {}), value: p.value }; }); emit('insp'); } }, 'Values from current fit'),
      h('button', { class: 'btn sm', onclick: () => { commit('clear bounds'); s.params = {}; emit('insp'); } }, 'Clear'), h('button', { class: 'btn sm', onclick: () => savePreset(f) }, 'Save as preset')));
  }
  return card;
}

async function savePreset(f) {
  const name = prompt('Preset name', `${f.info.type} preset`);
  if (!name) return;
  const presets = { ...(S.settings.fit_presets || {}), [name]: { ...clone(f.fitSpec), type: f.info.type } };
  S.settings = await call('settings_set', { patch: { fit_presets: presets } });
  toast(`Saved preset “${name}”.`, 'ok');
}

function historyCard(f) {
  return h('details', { class: 'card pad' }, h('summary', {}, `Fit history (${f.fitHistory.length})`),
    h('table', { class: 'tbl', style: 'margin-top:6px' }, f.fitHistory.map((e, i) => h('tr', { class: 'click', title: 'Restore this fit', onclick: () => { commit('restore fit'); f.fit[e.det] = e.res; f.fitSpec = clone(e.spec); emit(); } },
      h('td', { class: 'txt' }, `${e.at} · ${e.res.model}${f.info.detectors.length > 1 ? ' · ' + detLabel(f, e.det) : ''}`), h('td', {}, `R² ${fmt(e.res.r2, 4)}`)))));
}

// ---- Process tab
function procTab(f) {
  const d = f.data[f.detector];
  const its = d?.iterations;
  const out = h('div', { style: 'display:flex;flex-direction:column;gap:14px' });
  if (its) {
    const sel = new Set(f.iterSel?.length ? f.iterSel : its.values.map((_, i) => i));
    out.append(h('div', { class: 'f' }, h('div', { class: 'row' }, h('span', { class: 'lbl' }, 'Iterations shown'), h('span', { class: 'sp' }), h('span', { class: 'mono dim' }, `${sel.size} / ${its.count}`)),
      h('div', { style: `display:grid;grid-template-columns:repeat(${Math.min(its.count, 20)},minmax(0,1fr));gap:2px` },
        its.values.map((_, i) => h('button', { title: `Iteration ${i + 1}`, style: `height:16px;border:0;border-radius:2px;cursor:pointer;background:${sel.has(i) ? '#3A6AA8' : '#2A2F3A'}`,
          onclick: () => { sel.has(i) ? sel.delete(i) : sel.add(i); f.iterSel = [...sel].sort((a, b) => a - b); f.layers.iter = true; emit(); } }))),
      h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => { f.iterSel = null; emit(); } }, 'All'), h('button', { class: 'btn sm', onclick: () => { f.iterSel = [its.count - 1]; f.layers.iter = true; emit(); } }, 'Latest'),
        h('span', { class: 'note' }, 'Iteration display only; the analysed trace is the file\'s mean of complete iterations.'))));
  }
  out.append(h('div', { class: 'g2' },
    field('Bin size', input(f.proc.bin, (v) => setProc(f, { bin: Math.max(1, Math.round(num(v) || 1)) }))),
    field(kindOf(f) === 'tt' ? 'Smoothing (points)' : 'Smoothing window', input(f.proc.smooth, (v) => { f.proc.smooth = Math.max(1, Math.round(num(v) || 1)); f.layers.smooth = f.proc.smooth > 1; setProc(f, {}); }))));
  out.append(h('div', { class: 'g2' }, field('Data from', input(f.proc.roi_min ?? '', (v) => setProc(f, { roi_min: num(v) }), { placeholder: 'start' })), field('to', input(f.proc.roi_max ?? '', (v) => setProc(f, { roi_max: num(v) }), { placeholder: 'end' }))));
  const nEx = f.proc.excluded.length + f.proc.exclusion_ranges.length;
  out.append(h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Excluded points'),
    nEx ? h('div', { class: 'row wrap' }, h('span', { class: 'mono' }, `${f.proc.excluded.length} point(s)`), h('button', { class: 'btn sm', onclick: () => setProc(f, { excluded: [], exclusion_ranges: [] }, 'restore points') }, 'Restore all'))
      : h('div', { class: 'note' }, 'None. Click a point and press Exclude, or use the box-select tool (pencil) to exclude several. Excluded points are dropped from fits but never from the file.')));
  // baseline file
  const others = Object.values(S.files).filter((o) => o.path !== f.path && o.info.dim !== 'scan2d');
  out.append(field('Subtract another file', select([['', 'None'], ...others.map((o) => [o.path, o.info.name])], f.proc.baseline?.path || '',
    (v) => setProc(f, { baseline: v ? { path: v, detector: f.detector } : null }, 'baseline file'))));
  // transform steps
  const KINDS = [['baseline', 'Baseline'], ['detrend', 'Detrend'], ['normalize', 'Normalize'], ['derivative', 'Derivative'], ['integral', 'Integral'], ['resample', 'Resample'], ['expression', 'Expression']];
  const stepRows = f.proc.steps.map((st, i) => {
    const upd = (patch) => { const steps = clone(f.proc.steps); Object.assign(steps[i], patch); setProc(f, { steps }, 'transform'); };
    let param = null;
    if (st.kind === 'baseline') param = select([['median', 'median'], ['mean', 'mean'], ['first', 'first point'], ['linear', 'linear fit']], st.params?.method || 'median', (v) => upd({ params: { method: v } }), { style: 'width:110px;height:26px' });
    if (st.kind === 'normalize') param = select([['minmax', 'min–max'], ['zscore', 'z-score'], ['area', 'area']], st.params?.method || 'minmax', (v) => upd({ params: { method: v } }), { style: 'width:110px;height:26px' });
    if (st.kind === 'resample') param = input(st.params?.count || '', (v) => upd({ params: { count: num(v) } }), { placeholder: 'points', style: 'width:80px;height:26px' });
    if (st.kind === 'expression') param = input(st.params?.expression || 'y', (v) => upd({ params: { expression: v } }), { style: 'height:26px', placeholder: 'e.g. y / max(y)' });
    return h('div', { class: 'card', style: 'padding:6px 8px;display:flex;align-items:center;gap:8px' },
      h('input', { type: 'checkbox', checked: st.enabled !== false, 'aria-label': 'Enable step', onchange: (e) => upd({ enabled: e.target.checked }) }),
      h('span', { style: 'min-width:80px' }, KINDS.find(([k]) => k === st.kind)?.[1]), param, h('span', { class: 'sp' }),
      h('button', { class: 'ibtn', title: 'Remove step', onclick: () => { const steps = clone(f.proc.steps); steps.splice(i, 1); setProc(f, { steps }, 'remove transform'); } }, icon('close', 14)));
  });
  out.append(h('div', { class: 'f' }, h('div', { class: 'row' }, h('span', { class: 'lbl' }, 'Transform steps · in order'), h('span', { class: 'sp' }),
    select([['', 'Add step…'], ...KINDS], '', (v) => { if (v) setProc(f, { steps: [...clone(f.proc.steps), { kind: v, params: {}, enabled: true }] }, 'add transform'); }, { style: 'width:120px;height:26px' })),
  stepRows.length ? stepRows : h('div', { class: 'note' }, 'Steps change only the analysed copy. Fits use the untransformed data unless "Fit the transformed data" is ticked in Fit › Options.')));
  out.append(h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Overlays'), h('div', { class: 'note' }, 'Tick other 1D files in the Library to overlay them. Overlays follow this file\'s detector and observable.'),
    S.overlays.filter((p) => p !== f.path).map((p) => h('div', { class: 'row' }, h('span', { class: 'ellip', style: 'flex:1' }, basename(p)), h('button', { class: 'btn sm', onclick: () => { S.overlays = S.overlays.filter((q) => q !== p); emit(); } }, 'Remove')))));
  return out;
}

// ---- Plot tab (quick settings; full control in the plot editor)
function plotTab(f) {
  const st = f.style;
  const set = (patch) => { commit('plot style'); Object.assign(st, patch); emit(); };
  const s = baseStyle(st);
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    field('Title', input(st.title || '', (v) => set({ title: v }), { placeholder: 'None (double-click the plot to edit)' })),
    h('div', { class: 'g2' }, field('X label', input(st.x?.title ?? xTitle(f), (v) => set({ x: { ...(st.x || {}), title: v } }))), field('Y label', input(st.y?.title ?? (kindOf(f) === 'tt' ? '' : yTitle(f)), (v) => set({ y: { ...(st.y || {}), title: v } })))),
    h('div', { class: 'g2' }, field('X range', input(st.x?.min != null ? `${st.x.min}, ${st.x.max}` : 'auto', (v) => { const p = v.split(/[,;\s]+/).map(num); set({ x: { ...(st.x || {}), min: p[0] ?? null, max: p[1] ?? null } }); })),
      field('Y range', input(st.y?.min != null ? `${st.y.min}, ${st.y.max}` : 'auto', (v) => { const p = v.split(/[,;\s]+/).map(num); set({ y: { ...(st.y || {}), min: p[0] ?? null, max: p[1] ?? null } }); }))),
    field('Style preset', seg([['presentation', 'Presentation'], ['paper', 'Paper'], ['dense', 'Dense']], s.preset, (v) => import('./ploteditor.js').then((m) => m.applyPreset(f, v)))),
    field('Legend', seg([['ul', 'Upper left'], ['ur', 'Upper right'], ['lr', 'Lower right'], ['outside', 'Outside'], ['hidden', 'Hidden']], f.layers.legend ? s.legend.pos : 'hidden',
      (v) => { commit('legend'); if (v === 'hidden') f.layers.legend = false; else { f.layers.legend = true; st.legend = { ...(st.legend || {}), pos: v, x: null, y: null }; } emit(); })),
    kindOf(f) === '1d' ? field('Results box', seg([['off', 'Off'], ['compact', 'Compact'], ['pm', 'With errors']], !f.layers.results ? 'off' : (s.results.format === 'pm' ? 'pm' : 'compact'),
      (v) => { commit('results box'); f.layers.results = v !== 'off'; st.results = { ...(st.results || {}), format: v === 'pm' ? 'pm' : 'sig4' }; emit(); })) : null,
    kindOf(f) === '1d' ? field('Uncertainty', seg([['band', 'Band'], ['bars', 'Error bars']], f.errorMode, (v) => { f.errorMode = v; f.layers.band = true; emit(); })) : null,
    h('div', { class: 'g3' }, field('Width (in)', input(s.width_in, (v) => set({ width_in: num(v) || 8.5 }))), field('Height (in)', input(s.height_in, (v) => set({ height_in: num(v) || 5.5 }))), field('DPI', input(s.dpi, (v) => set({ dpi: num(v) || 300 })))),
    h('div', { class: 'row wrap' }, check('Grid', s.grid, (v) => set({ grid: v })), check('Mirror ticks', s.mirror, (v) => set({ mirror: v })), kindOf(f) === '1d' ? check('Residual panel', f.layers.res, (v) => { f.layers.res = v; emit(); }) : null),
    h('button', { class: 'btn block', onclick: () => { S.editor.open = true; emit(); } }, 'Open full plot editor'),
    h('div', { class: 'note' }, 'Double-click the title or an axis label on the plot to retype it. Drag the legend or results box to move them. Double-click a legend entry to restyle that series.'));
}

// ---- Export tab
function exportTab(f) {
  const k = kindOf(f);
  const st = baseStyle(f.style);
  f.exp = f.exp || { png: true, pdf: false, svg: false, json: true, csv: k === '1d', origin: false, report: false, detectors_raw: false, time_trace: k === 'tt', spots_csv: k === '2d', map_csv: false, session: false };
  const e = f.exp;
  const c = (key, label, dis) => check(label, e[key], (v) => { e[key] = v; }, { disabled: dis });
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Figure (as shown)'), h('div', { class: 'row wrap' }, c('png', 'PNG'), c('svg', 'SVG (vector)'), c('pdf', 'PDF'))),
    h('div', { class: 'note' }, `${st.width_in} × ${st.height_in} in at ${st.dpi} dpi. Change size in Plot. PDF embeds the ${st.dpi}-dpi image; use SVG for editable vector output.`),
    h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Data files'),
      c('json', 'Results JSON (settings, fit, provenance)'),
      k === '1d' ? [c('csv', 'Fit parameters CSV'), c('origin', 'Origin bundle CSV (x, y, smooth, fit, residual)'), c('report', 'Report PNG (figure + all metrics)')] : null,
      k === 'tt' ? [c('time_trace', '__time_trace.csv (both detectors, indices, timing)'), c('csv', 'Fit parameters CSV')] : null,
      k === '2d' ? [c('spots_csv', '__spots.csv (full spot table)'), c('map_csv', '__map_comparison.csv (cells, values, RGB)', !overlayCache[f.path])] : null,
      f.info.detectors.length > 1 ? c('detectors_raw', '__detectors_raw.csv') : null,
      c('session', 'Analysis session (reopen everything later)')),
    field('File name', input(f.expStem || stem(f.path), (v) => { f.expStem = v; })),
    field('Folder', h('div', { class: 'row' }, input(f.expDir || `${dirname(f.path)}\\SmartFitter results`, (v) => { f.expDir = v; }), h('button', { class: 'btn sm', onclick: async () => { const p = await call('dialog', { kind: 'open_folder', directory: dirname(f.path) }); if (p) { f.expDir = p; emit('insp'); } } }, 'Browse'))),
    h('button', { class: 'btn primary block', onclick: () => saveResults(f) }, 'Save results', h('span', { class: 'kbd' }, 'Ctrl S')),
    k === '2d' ? null : h('button', { class: 'btn block', title: 'Tab-separated: x, y, sem, smoothed, fit, residual', onclick: () => copyData(f) }, 'Copy data'),
    h('button', { class: 'btn block', onclick: () => import('./figure.js').then((m) => m.addCurrentPlot()) }, 'Add to Figure composer'));
}

export async function saveResults(f) {
  if (!f) return;
  const k = kindOf(f);
  f.exp = f.exp || { png: true, json: true, csv: k === '1d', time_trace: k === 'tt', spots_csv: k === '2d' };
  const e = f.exp;
  const st = baseStyle(f.style);
  const fig = currentFigure(f);
  const images = {};
  const size = { width_in: st.width_in, height_in: st.height_in, dpi: st.dpi };
  if (e.png || e.pdf) { const png = await renderImage(fig, size); if (e.png) images.png = png; if (e.pdf) images.pdf_png = png; }
  if (e.svg || e.pdf) { const svg = await renderImage(fig, { ...size, format: 'svg' }); if (e.svg) images.svg = svg; if (e.pdf) images.pdf_svg = svg; }
  const fit = f.fit[f.detector];
  if (e.report && fit) images.report_png = await renderImage(reportFigure(f, fig, fit), { width_in: st.width_in + 3.2, height_in: st.height_in, dpi: st.dpi });
  const dir = f.expDir || `${dirname(f.path)}\\SmartFitter results`;
  const det = f.info.detectors.length > 1 ? `__${f.detector}` : '';
  const stemName = (f.expStem || stem(f.path)) + det;
  const spotsRows = k === '2d' ? f.info.detectors.flatMap((d) => visibleSpots(f, d).map((s) => ({ ...s, detector_label: detLabel(f, d) }))) : null;
  const M = f.map;
  const res = await call('export', { req: {
    output_dir: dir, stem: stemName, images, dpi: st.dpi, source: f.path, detector: f.detector, mode: f.mode, processing: f.proc, fit_spec: f.fitSpec,
    fit: fit ? { ...fit, raw: undefined, dense: undefined } : null, json: e.json, csv: e.csv && !!fit, origin: e.origin, detectors_raw: e.detectors_raw, time_trace: e.time_trace,
    spots: spotsRows, spots_csv: e.spots_csv && spotsRows?.length, plot_state: { style: f.style, layers: f.layers },
    map_comparison_csv: e.map_csv && overlayCache[f.path], map_comparison_request: k === '2d' && overlayCache[f.path] ? [{ path: f.path, detector: f.detector, mode: f.mode }, { ...(M.compareWith || { path: f.path, detector: f.info.detectors.find((x) => x !== f.detector) }), mode: f.mode }, { shared: M.overlayScale === 'shared', background: M.background, palette: M.palette }] : null,
  } }, { label: 'Saving results' });
  if (e.session) { const { saveSession } = await import('./app.js'); await saveSession(`${dir}\\${stemName}.sfsession.json`); }
  toast(`Saved ${res.written.length} file(s) to ${res.folder}`, 'ok');
  return res;
}

function reportFigure(f, fig, fit) {
  const lines = resultLines(fit, f.info).map((l) => `${l.label} = ${l.err ? fmtPm(l.value, l.err) : fmt(l.value, 5)}${l.unit ? ' ' + l.unit : ''}`);
  const params = fit.params.map((p) => `${p.name} = ${fmt(p.value, 5)}${p.error ? ' ± ' + fmt(p.error, 2) : ''}`);
  const txt = [`<b>${f.info.name}</b>`, `${fit.model} · ${MODE_LABEL[f.mode]}`, `Status ${fit.status}`, '', ...lines, '', '<b>Parameters</b>', ...params,
    '', `RMSE ${fmt(fit.rmse, 4)} · DW ${fmt(fit.dw, 3)}`].join('<br>');
  const layout = clone(fig.layout);
  layout.margin = { ...(layout.margin || {}), r: 320 };
  layout.annotations = [...(layout.annotations || []).filter((a) => a.name !== 'results'), { xref: 'paper', yref: 'paper', x: 1.03, y: 1, xanchor: 'left', yanchor: 'top', align: 'left', showarrow: false, text: txt, font: { family: 'Consolas, monospace', size: 11 } }];
  return { data: fig.data, layout };
}

// ---- Info tab
function infoTab(f) {
  const i = f.info;
  const meta = i.metadata || {};
  const rows = [['File', i.path], ['Type', i.type], ['Scan', i.dim], ['Axes', (i.axes || []).join(', ')], ['Detectors', i.detectors.map((d) => detLabel(f, d)).join(', ')], ['Units', i.unit || '–']];
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    h('div', { class: 'kv' }, rows.flatMap(([k, v]) => [h('span', { class: 'k' }, k), h('span', { class: 'v', style: 'font-size:11.5px' }, v)])),
    i.detectors.length > 1 ? h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Detector labels (this session, saved in exports)'),
      i.detectors.map((d) => input(detLabel(f, d), (v) => { f.labels[d] = v; emit(); }))) : null,
    Object.keys(i.detector_errors || {}).length ? h('div', { class: 'note', style: 'color:var(--bad)' }, Object.entries(i.detector_errors).map(([k, v]) => `${k}: ${v}`).join('\n')) : null,
    h('details', { open: true }, h('summary', {}, `File metadata (${Object.keys(meta).length})`),
      h('div', { class: 'kv', style: 'margin-top:6px' }, Object.entries(meta).flatMap(([k, v]) => [h('span', { class: 'k' }, k), h('span', { class: 'v', style: 'font-size:11px' }, String(v))]))),
    h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => call('reveal', { path: f.path }) }, 'Show in folder'), h('button', { class: 'btn sm', onclick: () => copyText(f.path, 'Path copied') }, 'Copy path')));
}

// ---- Time-trace stats tab
function statsTab(f) {
  const tt = f.tt;
  const d = f.data[f.detector];
  const x = d?.raw?.x || [];
  const tmax = x.length ? x[x.length - 1] : 0;
  if (!tt.windows.a && tmax) { tt.windows.a = [0, Math.min(60, tmax / 4)]; tt.windows.b = [Math.max(0, tmax - Math.min(60, tmax / 4)), tmax]; }
  if (!tt.stats || tt.statsKey !== JSON.stringify(tt.windows) + f.mode) {
    tt.statsKey = JSON.stringify(tt.windows) + f.mode;
    call('time_stats', { path: f.path, mode: f.mode, windows: tt.windows }, { quiet: true }).then((r) => { tt.stats = r; emit('insp'); });
  }
  const wIn = (key, j) => input(tt.windows[key] ? fmt(tt.windows[key][j], 6) : '', (v) => { commit('window'); tt.windows[key] = [...(tt.windows[key] || [0, 0])]; tt.windows[key][j] = num(v) ?? 0; emit(); }, { style: 'width:70px' });
  const dt = d?.time?.dt || {};
  const st = tt.stats || {};
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    h('div', { class: 'g2' }, h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Window A (s)'), h('div', { class: 'row' }, wIn('a', 0), '–', wIn('a', 1))),
      h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Window B (s)'), h('div', { class: 'row' }, wIn('b', 0), '–', wIn('b', 1)))),
    h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => { if (f.xrange) { commit('window'); tt.windows.a = [...f.xrange].map((v) => v * (tt.unit === 'min' ? 60 : 1)); emit(); } else toast('Zoom the plot to the range you want first.'); } }, 'A = current view'),
      h('button', { class: 'btn sm', onclick: () => { if (f.xrange) { commit('window'); tt.windows.b = [...f.xrange].map((v) => v * (tt.unit === 'min' ? 60 : 1)); emit(); } else toast('Zoom the plot to the range you want first.'); } }, 'B = current view')),
    h('div', { class: 'card pad' }, h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, ''), h('th', {}, 'Mean'), h('th', {}, 'SD'), h('th', {}, 'CV'), h('th', {}, 'A→B')),
      f.info.detectors.map((det) => { const r = st[det] || {}; return h('tr', {}, h('td', { class: 'txt' }, detLabel(f, det)), h('td', {}, fmtCounts(r.mean)), h('td', {}, fmtCounts(r.sd)), h('td', {}, r.cv != null ? (100 * r.cv).toFixed(1) + ' %' : '–'), h('td', {}, r.change != null ? (100 * r.change).toFixed(1) + ' %' : '–')); })),
    h('div', { class: 'note', style: 'margin-top:6px' }, 'Whole trace; A→B compares window means.'),
    h('button', { class: 'btn sm', style: 'margin-top:6px', onclick: () => copyText(['Detector\tMean\tSD\tCV\tA mean\tB mean\tA→B'].concat(f.info.detectors.map((det) => { const r = st[det] || {}; return [detLabel(f, det), r.mean, r.sd, r.cv, r.a_mean, r.b_mean, r.change].join('\t'); })).join('\n'), 'Stats copied') }, 'Copy table')),
    h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:6px' }, h('span', { style: 'font-weight:600' }, 'Sampling'),
      h('div', { class: 'kv' }, h('span', { class: 'k' }, 'Median Δt'), h('span', { class: 'v' }, `${fmt(dt.median, 4)} s`), h('span', { class: 'k' }, '5–95 % Δt'), h('span', { class: 'v' }, `${fmt(dt.p5, 4)}–${fmt(dt.p95, 4)} s`),
        h('span', { class: 'k' }, 'Longest gap'), h('span', { class: 'v' }, `${fmt(dt.max, 4)} s`), h('span', { class: 'k' }, 'Retries'), h('span', { class: 'v' }, String(d?.time?.retries?.length || 0))),
      h('div', { class: 'note' }, f.info.irregular ? 'FFT is off because sampling is uneven. The Lomb–Scargle periodogram handles uneven spacing.' : 'Sampling is regular.'),
      h('button', { class: 'btn', onclick: () => showPeriodogram(f) }, 'Show periodogram')),
    field('Time unit', seg([['s', 'Seconds'], ['min', 'Minutes']], tt.unit, (v) => { tt.unit = v; f.xrange = null; emit(); })));
}

async function showPeriodogram(f) {
  const { modal } = await import('./util.js');
  const div = h('div', { style: 'height:360px' });
  const m = modal(h('div', { style: 'padding:14px;display:flex;flex-direction:column;gap:8px' }, h('div', { style: 'font-weight:600' }, `Lomb–Scargle periodogram · ${f.info.name}`), div));
  m.box.style.width = 'min(900px,94vw)';
  const data = [];
  for (const [i, det] of f.info.detectors.entries()) {
    const r = await call('periodogram', { path: f.path, detector: det, mode: f.mode }, { label: 'Periodogram' });
    data.push({ type: 'scatter', mode: 'lines', x: r.freq_hz.map((v) => v * 1000), y: r.power, name: detLabel(f, det), line: { width: 1.2, color: ['#1F5FBF', '#0E8A7E'][i] } });
  }
  Plotly.newPlot(div, data, { margin: { l: 60, r: 10, t: 10, b: 50 }, xaxis: { title: { text: 'Frequency [mHz]' }, type: 'log', showline: true, mirror: true, ticks: 'outside' }, yaxis: { title: { text: 'Normalized power' }, showline: true, mirror: true, ticks: 'outside' }, font: { family: 'Arial', size: 12 } }, { displayModeBar: false, responsive: true });
}

// ---- 2D: Map tab
function mapTab(f) {
  const M = f.map;
  const set = (patch) => { Object.assign(M, patch); emit(); };
  const m = f.data[f.detector]?.map;
  const sources = [['value', `${MODE_LABEL[f.mode]} (observable)`], ...(m?.z?.signal ? [['signal', 'Signal counts']] : []), ...(m?.z?.reference ? [['reference', 'Reference counts']] : [])];
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    h('div', { class: 'g2' }, field('Colormap', select(['viridis', 'parula', 'cividis', 'magma', 'inferno', 'plasma', 'turbo', 'coolwarm', 'RdBu', 'gray'], M.cmap, (v) => { set({ cmap: v }); call('settings_set', { patch: { colormap: v } }, { quiet: true }); })),
      field('Color scale', seg([[false, 'Linear'], [true, 'Log']], M.log, (v) => set({ log: v })))),
    check('Reverse colormap', M.reverse, (v) => set({ reverse: v })),
    field('Map shows', select(sources, M.source || 'value', (v) => set({ source: v }))),
    field('Color limits', seg([['auto', 'Full range'], ['robust', 'Percentiles'], ['manual', 'Manual']], M.limits, (v) => set({ limits: v }))),
    M.limits === 'manual' ? h('div', { class: 'card pad', style: 'display:grid;grid-template-columns:auto minmax(0,1fr) minmax(0,1fr);gap:6px 8px;align-items:center' },
      h('span'), h('span', { class: 'lbl' }, 'Min'), h('span', { class: 'lbl' }, 'Max'),
      (M.scale === 'shared' ? ['shared'] : f.info.detectors).flatMap((det) => [h('span', {}, det === 'shared' ? 'All' : detLabel(f, det)),
        input(M.manual[det]?.[0] ?? '', (v) => { M.manual[det] = [num(v), M.manual[det]?.[1] ?? null]; emit(); }), input(M.manual[det]?.[1] ?? '', (v) => { M.manual[det] = [M.manual[det]?.[0] ?? null, num(v)]; emit(); })])) : null,
    h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'View bounds'),
      h('div', { style: 'display:grid;grid-template-columns:auto minmax(0,1fr) minmax(0,1fr);gap:6px 8px;align-items:center' },
        h('span', { class: 'muted' }, 'X'), input(M.bounds?.x?.[0] ?? '', (v) => set({ bounds: { ...(M.bounds || {}), x: [num(v), M.bounds?.x?.[1] ?? null] } }), { placeholder: 'auto' }), input(M.bounds?.x?.[1] ?? '', (v) => set({ bounds: { ...(M.bounds || {}), x: [M.bounds?.x?.[0] ?? null, num(v)] } }), { placeholder: 'auto' }),
        h('span', { class: 'muted' }, 'Y'), input(M.bounds?.y?.[0] ?? '', (v) => set({ bounds: { ...(M.bounds || {}), y: [num(v), M.bounds?.y?.[1] ?? null] } }), { placeholder: 'auto' }), input(M.bounds?.y?.[1] ?? '', (v) => set({ bounds: { ...(M.bounds || {}), y: [M.bounds?.y?.[0] ?? null, num(v)] } }), { placeholder: 'auto' }))),
    field('Pixels', seg([['physical', 'Physical size'], ['square', 'Square']], M.geometry, (v) => set({ geometry: v }))),
    h('div', { class: 'note' }, 'Maps are never stretched: physical keeps equal data units on both axes; square keeps every pixel square. Swapping axes keeps the geometry.'),
    h('div', { class: 'g2' }, field('Linecut', select([['none', 'None'], ['h', 'Horizontal'], ['v', 'Vertical'], ['cross', 'Both'], ['line', 'Line (two clicks)']], M.linecut, (v) => { if (v !== 'none' && v !== 'line' && !M.cursor) M.cursor = peakCell(f, f.detector); set({ linecut: v }); })),
      field('Linecut source', select(sources, M.cutSource || M.source || 'value', (v) => set({ cutSource: v })))),
    check('Link zoom across maps', M.link !== false, (v) => set({ link: v })),
    check('Normalize linecuts per detector', M.cutNorm !== false, (v) => set({ cutNorm: v })),
    h('button', { class: 'btn block', onclick: () => { S.editor.open = true; emit(); } }, 'Edit titles, axes, fonts'));
}

// ---- 2D: Spots tab
function spotsTab(f) {
  const o = f.spotOpts;
  const M = f.map;
  const set = (patch) => { Object.assign(o, patch); emit('insp'); };
  const run = async () => {
    commit('spot detection');
    const dets = o.scope === 'both' ? f.info.detectors : [f.detector];
    for (const det of dets) f.spots[det] = await call('spots', { path: f.path, detector: det, mode: f.mode, opts: o }, { label: `Detecting spots · ${detLabel(f, det)}` });
    f.spotRemoved = {}; f.spotSel = [];
    toast(dets.map((det) => `${detLabel(f, det)}: ${f.spots[det].spots.length} spot(s)`).join(' · '), 'ok');
    emit();
  };
  const rows = f.info.detectors.flatMap((det) => visibleSpots(f, det).map((s) => ({ ...s, det })));
  const removedAny = Object.values(f.spotRemoved).some((a) => a.length);
  const toggle = (key, e, r) => {
    const multi = e.ctrlKey || e.shiftKey || e.metaKey;
    const sel = new Set(multi ? f.spotSel : []); sel.has(key) ? sel.delete(key) : sel.add(key); f.spotSel = [...sel];
    if (!multi && sel.has(key) && r) centerOnSpot(f, r);
    emit();
  };
  const sortBy = (col) => { const s0 = f.spotSort; f.spotSort = s0 && s0.col === col ? { col, dir: -s0.dir } : { col, dir: 1 }; emit('insp'); };
  const SORT = { det: (r) => `${r.det}:${String(r.spot_id).padStart(6, '0')}`, x: (r) => r.center_x, y: (r) => r.center_y, fwhm: (r) => r.fwhm_major, snr: (r) => r.snr, r2: (r) => r.r_squared };
  if (f.spotSort && SORT[f.spotSort.col]) { const g = SORT[f.spotSort.col], dr = f.spotSort.dir; rows.sort((a, b) => { const A = g(a), B = g(b); return (A < B ? -1 : A > B ? 1 : 0) * dr; }); }
  const th = (col, label) => h('th', { style: 'cursor:pointer;user-select:none', title: 'Click to sort', onclick: () => sortBy(col) }, label, f.spotSort?.col === col ? (f.spotSort.dir > 0 ? ' ▲' : ' ▼') : '');
  const cmpSel = f.spotSel.map((k) => { const [det, id] = k.split(':'); return rows.find((r) => r.det === det && String(r.spot_id) === id); }).filter(Boolean);
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    h('div', { class: 'row wrap' }, h('button', { class: 'btn primary', onclick: run }, 'Detect and fit'), f.info.detectors.length > 1 ? seg([['both', 'Both detectors'], ['active', 'Active']], o.scope, (v) => set({ scope: v })) : null),
    h('div', { class: 'g2' }, field('Analyze', select([['auto', 'Auto fluorescence'], ['value', 'Current observable'], ['signal', 'Signal'], ['reference', 'Reference'], ['difference', 'Difference']], o.source, (v) => set({ source: v }))),
      field('Polarity', select(['Auto', 'Bright', 'Dark', 'Bright and dark'], o.polarity, (v) => set({ polarity: v })))),
    h('div', { class: 'g3' }, field('Threshold (σ)', input(o.threshold, (v) => set({ threshold: num(v) || 3 }))), field('Min pixels', input(o.min_pixels, (v) => set({ min_pixels: num(v) || 4 }))), field('Background', input(o.bg_window || 'auto', (v) => set({ bg_window: num(v) || 0 })))),
    h('div', { class: 'row wrap' }, check('Fitted shapes', M.showShapes, (v) => { M.showShapes = v; emit(); }), check('Centers', M.showCenters, (v) => { M.showCenters = v; emit(); }), check('Numbers', M.showNumbers, (v) => { M.showNumbers = v; emit(); })),
    h('div', { class: 'card pad' },
      h('div', { class: 'row' }, h('span', { style: 'font-weight:600' }, `${rows.length} spot${rows.length === 1 ? '' : 's'}`), h('span', { class: 'sp' }),
        h('button', { class: 'btn sm', disabled: !M.bounds, title: 'Reset the map view to the full map', onclick: () => { M.bounds = null; emit(); } }, 'Show all'),
        h('button', { class: 'btn sm', disabled: !rows.length, onclick: () => copyText(spotTable(f, rows), 'Spot table copied (tab-separated, X/Y ordered)') }, 'Copy table')),
      rows.length ? h('table', { class: 'tbl', style: 'margin-top:6px' }, h('tr', {}, th('det', 'Det.'), th('x', 'X'), th('y', 'Y'), th('fwhm', 'FWHM'), th('snr', 'SNR'), th('r2', 'R²')),
        rows.map((r) => { const key = `${r.det}:${r.spot_id}`; return h('tr', { class: `click ${f.spotSel.includes(key) ? 'sel' : ''}`, title: `contrast ${fmt(r.contrast_percent, 3)} % · axis ratio ${fmt(r.axis_ratio, 3)} · angle ${fmt(r.angle_deg, 3)}°`, onclick: (e) => toggle(key, e, r), onmouseenter: () => hoverSpot(f, key), onmouseleave: () => hoverSpot(f, null) },
          h('td', { class: 'txt' }, `${r.spot_id} · ${detLabel(f, r.det).replace('Detector ', 'D')}`), h('td', {}, fmt(r.center_x, 6)), h('td', {}, fmt(r.center_y, 6)), h('td', {}, `${fmt(r.fwhm_major, 3)}×${fmt(r.fwhm_minor, 3)}`), h('td', {}, fmt(r.snr, 3)), h('td', {}, fmt(r.r_squared, 3))); }))
        : h('div', { class: 'note', style: 'margin-top:6px' }, Object.keys(f.spots).length ? 'No spots above the threshold.' : 'Run Detect and fit to characterize fluorescent spots (rotated elliptical Gaussians on a local background).'),
      h('div', { class: 'row wrap', style: 'margin-top:8px' },
        h('button', { class: 'btn sm', disabled: !f.spotSel.length, onclick: () => { commit('remove spots'); f.spotSel.forEach((k) => { const [det, id] = k.split(':'); f.spotRemoved[det] = [...(f.spotRemoved[det] || []), Number(id)]; }); f.spotSel = []; emit(); } }, 'Remove selected'),
        h('button', { class: 'btn sm', disabled: !f.spotSel.length, onclick: () => { commit('keep spots'); for (const det of f.info.detectors) { const keep = f.spotSel.filter((k) => k.startsWith(det + ':')).map((k) => Number(k.split(':')[1])); f.spotRemoved[det] = (f.spots[det]?.spots || []).map((s) => s.spot_id).filter((id) => !keep.includes(id)); } f.spotSel = []; emit(); } }, 'Keep selected only'),
        h('button', { class: 'btn sm', disabled: !removedAny, onclick: () => { commit('restore spots'); f.spotRemoved = {}; emit(); } }, 'Restore removed'))),
    f.info.detectors.length > 1 ? spotCompareCard(f, cmpSel) : null);
}

// Centre the map view on a spot. bounds are display coordinates (axes swapped when f.map.swap).
function centerOnSpot(f, r) {
  const m = f.data[r.det]?.map || f.data[f.detector]?.map;
  if (!m) return;
  const step = (a) => (a.length > 1 ? Math.abs(a[a.length - 1] - a[0]) / (a.length - 1) : 1);
  const swap = !!f.map.swap;
  const dx = swap ? m.y : m.x, dy = swap ? m.x : m.y;
  const [cx, cy] = swap ? [r.center_y, r.center_x] : [r.center_x, r.center_y];
  const half = Math.max(3 * (r.fwhm_major || 0), 4 * Math.max(step(dx), step(dy)));
  const win = (c, a) => {
    const pad = step(a) / 2, L = Math.min(...a) - pad, H = Math.max(...a) + pad;
    const hw = Math.min(half, (H - L) / 2);
    let a0 = c - hw, a1 = c + hw;
    if (a0 < L) { a1 += L - a0; a0 = L; }
    if (a1 > H) { a0 -= a1 - H; a1 = H; }
    return [Math.max(a0, L), Math.min(a1, H)];
  };
  f.map.bounds = { x: win(cx, dx), y: win(cy, dy) };
}

function hoverSpot(f, key) {
  if (f.spotHover === key) return;
  f.spotHover = key;
  drawPlot(f);
}

function spotTable(f, rows) {
  const sorted = [...rows].sort((a, b) => a.center_x - b.center_x || a.center_y - b.center_y);
  return [['Spot', 'Detector', 'X', 'Y', 'FWHM major', 'FWHM minor', 'Angle (°)', 'SNR', 'R²'].join('\t'),
    ...sorted.map((r) => [r.spot_id, detLabel(f, r.det), fmt(r.center_x, 6), fmt(r.center_y, 6), fmt(r.fwhm_major, 3), fmt(r.fwhm_minor, 3), fmt(r.angle_deg, 3), fmt(r.snr, 3), fmt(r.r_squared, 3)].join('\t'))].join('\n');
}

function spotCompareCard(f, sel) {
  const pair = sel.length === 2 && sel[0].det !== sel[1].det ? sel : null;
  const auto = !pair && f.info.detectors.every((d) => visibleSpots(f, d).length === 1) ? f.info.detectors.map((d) => ({ ...visibleSpots(f, d)[0], det: d })) : null;
  const use = pair || auto;
  const card = h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' }, h('span', { style: 'font-weight:600' }, `${detLabel(f, f.info.detectors[0])} vs ${detLabel(f, f.info.detectors[1])}`));
  if (!use) { card.append(h('div', { class: 'note' }, 'Select one spot from each detector (Ctrl-click rows) to compare them.')); return card; }
  const key = use.map((s) => `${s.det}:${s.spot_id}`).join('|');
  if (f.spotCmpKey !== key) {
    f.spotCmpKey = key; f.spotCmp = null;
    call('compare_spots', { a: { ...use[0], path: f.path, detector: use[0].det, source: f.spotOpts.source }, b: { ...use[1], path: f.path, detector: use[1].det, source: f.spotOpts.source }, mode: f.mode }, { quiet: true })
      .then((r) => { f.spotCmp = r; emit('insp'); });
  }
  const c = f.spotCmp;
  if (!c) { card.append(h('span', { class: 'spinner' })); return card; }
  card.append(h('div', { class: 'kv' },
    h('span', { class: 'k' }, 'Center offset ΔX'), h('span', { class: 'v' }, fmt(c.dx, 4)), h('span', { class: 'k' }, 'Center offset ΔY'), h('span', { class: 'v' }, fmt(c.dy, 4)),
    h('span', { class: 'k' }, '|Δ|'), h('span', { class: 'v' }, `${fmt(c.distance, 4)}${c.distance_px ? ` (≈ ${fmt(c.distance_px, 2)} px)` : ''}`),
    h('span', { class: 'k' }, 'FWHM ratio (major, minor)'), h('span', { class: 'v' }, `${fmt(c.fwhm_major_ratio, 3)}, ${fmt(c.fwhm_minor_ratio, 3)}`),
    h('span', { class: 'k' }, 'Angle difference'), h('span', { class: 'v' }, `${fmt(c.angle_diff, 3)}°`), h('span', { class: 'k' }, 'Amplitude ratio'), h('span', { class: 'v' }, fmt(c.amplitude_ratio, 3)),
    h('span', { class: 'k' }, 'Local profile correlation'), h('span', { class: 'v' }, fmt(c.profile_correlation, 3))),
  h('div', { class: 'note' }, `Offsets in stage units from the fitted Gaussian centers (second minus first). ${auto ? 'Using the only spot on each detector.' : ''}`));
  return card;
}

// ---- 2D: Compare tab (RGB overlays)
function mapCompareTab(f) {
  const M = f.map;
  const candidates = [];
  if (f.info.detectors.length > 1) f.info.detectors.filter((d) => d !== f.detector).forEach((d) => candidates.push({ path: f.path, detector: d, label: `${detLabel(f, d)} (this file)` }));
  Object.values(S.files).filter((o) => o.path !== f.path && o.info.dim === 'scan2d').forEach((o) => o.info.detectors.forEach((d) => candidates.push({ path: o.path, detector: d, label: `${o.info.name} · ${detLabel(o, d)}`, axes: o.info.axes })));
  const cur = M.compareWith ? `${M.compareWith.path}|${M.compareWith.detector}` : (candidates[0] ? `${candidates[0].path}|${candidates[0].detector}` : '');
  const update = async (patch) => { Object.assign(M, patch); await ensureOverlay(f); if (!M.layout.includes('overlay')) { M.prevLayout = M.layout; M.layout = 'maps+overlay'; } emit(); };
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    field('Compare with', select(candidates.map((c) => [`${c.path}|${c.detector}`, c.label]), cur, (v) => { const [p, d] = v.split('|'); update({ compareWith: { path: p, detector: d } }); })),
    h('div', { class: 'note' }, 'Lists maps from loaded files. They must cover the same plane (XY, XZ, YZ), including opposite stored axis order. Add more files from the Library.'),
    field('Palette', seg(['Pink / teal', 'Green / magenta', 'Red / cyan'], M.palette, (v) => update({ palette: v }))),
    field('Normalization', seg([['independent', 'Independent'], ['shared', 'Shared']], M.overlayScale, (v) => update({ overlayScale: v }))),
    field('Background removed (percentile)', input(M.background, (v) => update({ background: Math.min(100, Math.max(0, num(v) ?? 10)) }))),
    h('div', { class: 'note' }, 'Different grids are combined on native pixel edges by copying each measured cell. No interpolation or extrapolation; uncovered area stays transparent.'),
    h('div', { class: 'row wrap' }, h('button', { class: 'btn primary', onclick: () => update({}) }, 'Show overlay'),
      h('button', { class: 'btn', disabled: !M.prevLayout, onclick: () => { M.layout = M.prevLayout || 'side'; M.prevLayout = null; emit(); } }, 'Restore previous view')));
}
