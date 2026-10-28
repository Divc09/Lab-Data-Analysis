import { finiteExtent } from './numeric.js';
// Analyze workspace: 1D experiments, time traces and 2D maps, with their inspectors.
import { call } from './api.js';
import { canonical, sourcePath } from './analysisidentity.js';
import { S, emit, active, commit, markDirty, loadData, detLabel, defaultFitSpec, syncOverlays, syncOverlay, removeOverlay, subscribe, recordResult, resultFreshness, currentResult, envelopeFor } from './state.js';
import { openOverlayPicker, browseOverlays } from './overlaypicker.js';
import { isNormalizedT1, t1Normalization, displaySpace, displayProvenance } from './fitdisplay.js';
import { h, mount, icon, seg, select, input, check, field, fmt, fmtPm, fmtCounts, num, toast, clone, tag, basename, dirname, stem, contextMenu, linreg } from './util.js';
import { figure1D, figureTT, xTitle, yTitle, fftAvailability } from './plot1d.js';
import { figureMaps, linecuts, CMAP_GRADIENT, visibleSpots, peakCell, displayMap, mapPanels, nearest, fitColorbars, lineLength, mapRelayoutView, rememberMapRanges, mapValueLabel } from './plot2d.js';
// Readout suffix for map values: contrast of the selected observable is dimensionless; raw sources keep their unit.
const mapUnitSuffix = (f) => ((f.map?.source || 'value') === 'value' && f.mode === 'contrast' ? '' : f.info.unit ? ` ${f.info.unit}` : '');
import { plotConfig, resultLines, baseStyle, MODE_LABEL, friendlyAxis } from './plotstyle.js';
import { renderImage, copyImage, copyText } from './export.js';
import { openSeriesEditor, editorPanel, applyPlotlyEdit, inlinePlotEditor, styleTools as styleToolsBox } from './ploteditor.js';
import * as sci from './science.js';
import { seriesPanel, comparisonRows, comparisonCSV, comparisonFiles, filePresentation, fileKeys, setFileVisible, fitSeries, fitAllSeries, fitFamily, fitPlan } from './seriescontrol.js';
import { xDisplay, displayValue, unitOptions, parameterUnit } from './unitdisplay.js';
import { iterationCard, iterationFitActions, iterationProvenance, iterationCSV, openCombineDialog, fitTargetLabel } from './iterations.js';

let legendTimer = null;
let lastMapHover = null; // last heatmap cell under the mouse (for the right-click menu)
let plotEl = null, plotBound = false, cutEl = null, plotPending = false, drawToken = 0;
const roomy = (el) => el.clientWidth >= 40 && el.clientHeight >= 40;
let fftEl = null; // readout span showing the FFT dominant frequency
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
const family = (f) => fitFamily(f);

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
  await syncOverlays(f);
  emit();
}

export async function setProc(f, patch, label = 'processing') {
  commit(label);
  Object.assign(f.proc, patch);
  if (Object.keys(f.fit).length) f.fitStale = true;
  f.iterationFitsStale = true;
  await loadData(f);
  emit();
}

export async function runFit(f, { robust = false, recordUndo = true } = {}) {
  if (!f.info.fit_allowed && kindOf(f) === '1d' && f.fitSpec.model !== 'GaussianPeaks' && f.fitSpec.model !== 'CustomModel') {
    toast('This scan type is plot-only. Choose Gaussian peaks or a custom model in the Fit tab to fit it.');
    return;
  }
  if (recordUndo) commit('fit');
  if (f.data[f.detector]?.iterations?.selected?.length === 0) { toast('Select at least one iteration before fitting.', 'err'); return; }
  if (f.fitting || f.iterationFitting || f.sensitivityFitting) return;
  const spec = prepareFitSpec(f), proc = clone(f.proc), mode = f.mode;
  const signature = JSON.stringify([mode, proc, spec]);
  spec.robust = robust;
  const dets = spec.apply_both || kindOf(f) === 'tt' ? [...f.info.detectors] : [f.detector];
  f.fitting = true; emit('insp');
  try {
    for (const det of dets) {
      const res = await call('fit', { path: f.path, detector: det, mode, proc, spec }, { label: `Fitting ${f.info.name} · ${detLabel(f, det)}`, retry: () => runFit(f, { robust, recordUndo: false }) });
      recordResult(f,res,{det,settings:spec});
      if (f.mode === mode) f.fit[det] = res;
      else { f.fitByMode ||= {}; (f.fitByMode[mode] ||= {})[det] = res; }
      f.fitHistory.unshift({ at: new Date().toLocaleTimeString(), det, mode, spec, res });
      f.fitHistory = f.fitHistory.slice(0, 12);
    }
    markDirty();
    f.fitStale = signature !== JSON.stringify([f.mode, f.proc, prepareFitSpec(f)]);
    if (!f.layersTouched) { f.layers.fit = true; f.layers.results = kindOf(f) === '1d'; }
    const r = f.fit[f.detector];
    if (r) toast(`${r.model}: R² ${fmt(r.r2, 4)} · ${r.status}${r.elapsed_s ? ` · ${r.elapsed_s.toFixed(1)} s` : ''}`, r.status === 'PASS' ? 'ok' : '');
  } finally { f.fitting = false; emit(); }
}

// Run fit / Ctrl+F: with compared series on the plot, fits all of them unless the Fit tab scope says "This file only".
export const fitScopeAll = (f) => kindOf(f) === '1d' && comparisonFiles(f).length > 1 && f.display.fitScope !== 'primary';
export function runFitScoped(f, opts = {}) { return fitScopeAll(f) ? fitAllSeries(f, opts) : runFit(f, opts); }

// ODMR "peak-pick only" (v1 parity): detected sample positions, no model fit, no uncertainties.
function peakPickCard(f) {
  const opt = (f.peakPickOpts ||= { prominence: 8, smoothed: !!f.fitSpec.fit_smoothed });
  const res = f.peakPick?.[f.detector];
  const run = async () => {
    const r = await call('odmr_pick_peaks', { path: f.path, detector: f.detector, mode: f.mode, proc: f.proc, polarity: f.fitSpec.odmr_polarity,
      ranges: (f.fitSpec.odmr_ranges || []).filter((q) => q && (q[0] !== null || q[1] !== null)), smoothed: opt.smoothed, prominence: opt.prominence / 100 }, { label: 'Picking peaks' });
    (f.peakPick ||= {})[f.detector] = { ...r, signature: JSON.stringify([f.mode, f.proc]) };
    f.layers.picked = true; emit();
  };
  const unit = f.info.x_unit ? ` ${f.info.x_unit}` : '';
  const stale = res && res.signature !== JSON.stringify([f.mode, f.proc]);
  return h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:6px' },
    h('strong', {}, 'Peak-pick only (no fit)'),
    h('div', { class: 'g2' }, field('Min prominence (% of range)', input(opt.prominence, (v) => { opt.prominence = Math.max(0.1, Math.min(100, num(v) || 8)); }, { numeric: true })),
      h('div', { style: 'align-self:end' }, check(`Use smoothed data${f.data[f.detector]?.proc?.smoothing_window > 1 ? ` (${f.data[f.detector].proc.smoothing_window} pt)` : ''}`, opt.smoothed, (v) => { opt.smoothed = v; }))),
    h('div', { class: 'row wrap' }, h('button', { class: 'btn', onclick: run }, res ? 'Pick again' : 'Pick peaks'),
      res ? h('button', { class: 'btn sm', onclick: () => copyText(res.peaks.map((p) => `${p.x}\t${p.y}`).join('\n'), 'Peak positions copied') }, 'Copy') : null,
      res ? check('Show on plot', f.layers.picked !== false, (v) => { f.layers.picked = v; emit(); }) : null),
    res ? h('div', { class: 'note' }, `${res.peaks.length} ${res.polarity === 'dip' ? 'dip' : 'peak'}${res.peaks.length === 1 ? '' : 's'} (${res.source}${stale ? ' · STALE: processing changed, pick again' : ''}): ${res.peaks.map((p) => fmt(p.x, 7) + unit).join(', ') || 'none above the prominence threshold'}`) : null,
    h('div', { class: 'note', title: 'Uses the Polarity and Centers range above (best peak per range). Positions are sample points (resolution = point spacing); raise the prominence or use smoothed data if noise is picked.' }, 'Uses Polarity + Centers range above · hover for details'));
}

export function prepareFitSpec(f) {
  const spec = clone(f.fitSpec);
  if (spec.model === 'CustomModel') spec.custom_spec = customSpec(spec);
  return spec;
}

function customSpec(spec) {
  const names = spec.custom_params.map((p) => p.name);
  return { name: 'CustomModel', expression: spec.custom_expr, param_names: names,
    lower: spec.custom_params.map((p) => p.min ?? -1e300), upper: spec.custom_params.map((p) => p.max ?? 1e300), seed: spec.custom_params.map((p) => p.seed ?? 1) };
}

const RESERVED = new Set(['x', 't', 'abs', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'clip', 'pi', 'e', 'np']);
export function exprParams(expr, prev = []) {
  // x and t both name the independent variable; the lookbehind skips exponents such as 1e3 and attributes such as np.exp
  const names = [...new Set((expr.match(/(?<![\w.])[A-Za-z_]\w*/g) || []).filter((n) => !RESERVED.has(n)))];
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
  return renderImage(fig, { width_in: st.width_in, height_in: st.height_in, dpi: st.dpi }).then(copyImage);
}

export function currentFigure(f) {
  if (!f) return null;
  const k = kindOf(f);
  if (k === '1d') return figure1D(f, { forExport: true });
  if (k === 'tt') return figureTT(f);
  return figureMaps(f, displayedOverlay(f));
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
  f.fitStale=Object.entries(f.fit || {}).some(([det,r])=>!resultFreshness(f,r,det).fresh);
  if(k==='2d' && f.map.layout.includes('overlay') && mapRequest(f) && !displayedOverlay(f) && !f.overlayLoading && f.overlayErrorKey!==canonical(mapRequest(f))) {
    f.overlayLoading=true;
    ensureOverlay(f).finally(()=>{f.overlayLoading=false;emit();});
  }
  mount(main, header(f), k === '2d' ? mapToolbar(f) : toolbar(f),
    sci.historicalToolNote(f),
    k === '1d' && displayProvenance(f) ? h('div', { class: 'note', 'aria-label': 'Display provenance' }, displayProvenance(f)) : null,
    k === '1d' && iterationNoteUseful(f) ? h('div', { class: 'note iteration-provenance', 'aria-label': 'Iteration provenance' }, iterationProvenance(f)) : null,
    // Plot first; per-file Series controls sit below it (like a docked properties panel) so the plot never gets pushed off-screen.
    plotSection(f), k === '2d' ? mapReadout(f) : readout(f), k === '1d' ? seriesPanel(f) : null);
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
    if (m) bits.push(`${m.x.length} × ${m.y.length} · ${friendlyAxis(m.x_label).split(' [')[0]} ${fmt(finiteExtent(m.x)[0], 6)}–${fmt(finiteExtent(m.x)[1], 6)}, ${friendlyAxis(m.y_label).split(' [')[0]} ${fmt(finiteExtent(m.y)[0], 6)}–${fmt(finiteExtent(m.y)[1], 6)}`);
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
    ctlr.append(select([['', `Auto · ${i.type}`], ...['Ramsey', 'Rabi', 'SpinEcho', 'DynamicDecoupling', 'T1', 'ODMR', 'DEERFrequency', 'DEERDuration', 'DEERDecay', 'DEERPosition', 'LineScan', 'TimeTrace'].map((t) => [t, t])],
      f.fitSpec.experiment, (v) => { commit('experiment'); f.fitSpec.experiment = v; emit(); }, { style: 'width:138px', title: 'Experiment type (override the detected type; changes the fit model family)', 'aria-label': 'Experiment type' }));
  }
  if (i.detectors.length > 1) {
    ctlr.append(seg(i.detectors.map((d) => [d, detLabel(f, d)]), f.detector, (d) => { commit('detector'); f.detector = d; emit(); }));
  }
  ctlr.append(seg(['contrast', 'signal', 'reference', 'difference'].map((m) => [m, MODE_LABEL[m], i.observables.includes(m) ? '' : 'Not recorded in this file']), f.mode,
    (m) => setMode(f, m), { disabled: ['contrast', 'signal', 'reference', 'difference'].filter((m) => !i.observables.includes(m)) }));
  // One compact bar: identity + metadata on the left (full text on hover), view controls on the right.
  const metaText = bits.map((b) => (typeof b === 'string' ? b : b.textContent)).join(' · ');
  return h('div', { class: 'filebar' },
    h('div', { class: 'filehead' }, tag(i.tag),
      h('h1', { title: f.path }, i.name.replace(/\.mat$/i, '')),
      h('div', { class: 'metaline', title: metaText }, bits.map((b, n) => [n ? h('span', { class: 'dim' }, '·') : null, b]))),
    ctlr);
}

function layerChip(f, key, label, sw, opts = {}) {
  return h('button', { class: `chip ${f.layers[key] ? 'on' : ''}`, disabled: opts.disabled, title: opts.title || '',
    onclick: () => { f.layers[key] = !f.layers[key]; f.layersTouched = true; emit(); } },
  sw ? h('span', { class: 'sw', style: `background:${sw};${opts.tall ? 'height:8px' : ''}` }) : null, label);
}

// The iteration provenance line is shown only when it says something: a subset, a pooled dataset, or a smoothed fit target.
function iterationNoteUseful(f) {
  const it = f.data[f.detector]?.iterations;
  if (!it || !iterationProvenance(f)) return false;
  return it.selected.length !== it.count || !!f.info.combined || f.fit[f.detector]?.provenance?.fit_target === 'smoothed';
}

// Less-used layers live in one menu so the chip row fits on one line; the button shows how many are on.
function moreLayers(f, fit) {
  const sampling = fftAvailability(f);
  const items = [
    ['meanSd', 'Iteration mean ± SD band', !f.data[f.detector]?.iterations], ['conf', 'Model mean confidence (±1σ)', !fit?.dense?.band], ['envelope', 'Rabi envelope (T2ρ)', !fit?.envelope], ['sigref', 'Signal + Ref panel', f.info.reference_only],
    ['fft', sampling.ok ? 'FFT panel' : `FFT panel — ${sampling.reason}`, !sampling.ok], ['peaks', 'Fitted peak markers', !fit?.params?.some((p) => /^(x0|center)/.test(p.name))],
    ['picked', 'Picked peaks (peak-pick only)', !f.peakPick?.[f.detector]], ...(f.info.detectors.length > 1 ? [['__both', 'Both detectors', false]] : []),
  ];
  const isOn = (k) => (k === '__both' ? !!f.compareDetectors : k === 'picked' ? f.layers.picked !== false && !!f.peakPick?.[f.detector] : !!f.layers[k]);
  const on = items.filter(([k, , off]) => !off && isOn(k)).length;
  return h('button', { class: `chip ${on ? 'on' : ''}`, title: 'Confidence band, Rabi envelope, Signal + Ref panel, FFT, peak markers, both detectors',
    onclick: (e) => {
      const r = e.currentTarget.getBoundingClientRect();
      contextMenu(r.left, r.bottom + 4, items.map(([k, label, off]) => ({ disabled: off, label: `${isOn(k) && !off ? '✓ ' : '   '}${label}${off ? ' (n/a)' : ''}`,
        run: () => { if (off) return; if (k === '__both') f.compareDetectors = !f.compareDetectors; else if (k === 'picked') f.layers.picked = !isOn('picked'); else { f.layers[k] = !f.layers[k]; f.layersTouched = true; } emit(); } })));
    } }, `More layers${on ? ` · ${on}` : ''} ▾`);
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
    // Only layers this file can show: a chip that could never turn on (no fit yet, no per-point SEM, a single
    // iteration) is left out instead of sitting greyed out, which keeps the row to one line.
    const its = d?.iterations;
    chips.append(
      layerChip(f, 'data', 'Data', '#5A9BEA'), fit ? layerChip(f, 'fit', 'Fit', '#F08A3C') : null,
      d?.proc?.sem ? layerChip(f, 'band', f.errorMode === 'bars' ? '± SEM bars' : '± SEM', '#3D5E8C', { tall: true, title: 'Standard error of the mean across iterations' }) : null,
      fit ? layerChip(f, 'res', 'Residuals') : null, layerChip(f, 'smooth', 'Smoothed', '#222'),
      its?.count > 1 ? h('button', { class: `chip ${f.layers.iter ? 'on' : ''}`, title: 'Show individual iterations (choose which in Process)',
        onclick: () => { f.layers.iter = !f.layers.iter; emit(); } }, 'Iterations ', h('span', { class: 'mono dim' }, its.selected.length)) : null,
      layerChip(f, 'legend', 'Legend'), fit ? layerChip(f, 'results', 'Results box') : null,
      moreLayers(f, fit));
  }
  const tools = h('div', { class: 'row wrap', style: 'gap:2px' },
    !tt ? h('button', { class: 'btn', title: 'Overlay other files on this plot (independent data/fit/style per file)', onclick: () => openOverlayPicker(f) }, `Compare…${S.overlays.length ? ` (${S.overlays.length})` : ''}`) : null,
    !tt ? h('button', { class: 'btn', title: 'Choose iterations, fit each iteration, or pool runs', onclick: (e) => {
      const r = e.currentTarget.getBoundingClientRect();
      contextMenu(r.left, r.bottom + 4, [
        { label: 'Choose iterations for mean + fit…', run: () => { f.tab = 'proc'; emit('insp'); } },
        d?.iterations ? { label: 'Fit each selected iteration', run: () => import('./iterations.js').then((m) => m.runIterationFits(f)) } : null,
        { label: 'Combine runs (pool iterations)…', run: () => openCombineDialog(f) }]);
    } }, `Iterations${d?.iterations ? ` ${d.iterations.selected.length}/${d.iterations.count}` : ''} ▾`) : null,
    !tt && isNormalizedT1(fit) ? t1DisplayControl(f) : null,
    !tt ? h('label', { class: 'row', style: 'gap:4px', title: 'X-axis display unit (display only; fits stay in native units)' }, h('span', { class: 'dim' }, 'X'),
      select(unitOptions(f.info.x_unit), f.display.xunit || 'auto', (xunit) => { commit('x display units'); f.display.xunit = xunit; emit(); }, { style: 'width:78px', 'aria-label': 'X unit' })) : null,
    tt ? seg([['both', 'Line + points'], ['pts', 'Points'], ['line', 'Line']], f.styleMode, (v) => { f.styleMode = v; emit(); })
      : select([['both', 'Line + points'], ['pts', 'Points'], ['line', 'Line']], f.styleMode, (v) => { f.styleMode = v; emit(); }, { style: 'width:108px', title: 'Data drawing style', 'aria-label': 'Data drawing style' }),
    h('span', { class: 'sep' }),
    h('button', { class: 'ibtn', title: 'Reset view (Ctrl+Home)', onclick: () => resetView(f) }, icon('home')),
    h('button', { class: `ibtn ${f.tool === 'zoom' || !f.tool ? 'on' : ''}`, title: 'Box zoom (scroll also zooms)', onclick: () => setTool(f, 'zoom') }, icon('zoom')),
    h('button', { class: `ibtn ${f.tool === 'pan' ? 'on' : ''}`, title: 'Pan', onclick: () => setTool(f, 'pan') }, icon('pan')),
    tt ? h('button', { class: `btn sm ${f.ttPick === 'a' ? 'primary' : ''}`, title: 'Then drag a box on the plot to set window A', onclick: () => setWindowByDrag(f, 'a') }, 'Set A') : null,
    tt ? h('button', { class: `btn sm ${f.ttPick === 'b' ? 'primary' : ''}`, title: 'Then drag a box on the plot to set window B', onclick: () => setWindowByDrag(f, 'b') }, 'Set B') : null,
    tt ? null : (f.tool === 'select'
      ? h('button', { class: 'btn sm primary', title: 'Drag a box around points to exclude them. Click again or press Esc to cancel.', onclick: () => setTool(f, 'zoom') }, icon('edit', 13), 'Drag to exclude · cancel')
      : h('button', { class: 'ibtn', title: 'Exclude points: drag a box around them (one-shot)', onclick: () => setTool(f, 'select') }, icon('edit'))),
    h('button', { class: `ibtn ${f.tool === 'measure' ? 'on' : ''}`, title: 'Measure: click two points', onclick: () => { setTool(f, f.tool === 'measure' ? 'zoom' : 'measure'); f.measure = []; } }, icon('measure')));
  // Figure actions sit at the right end of the layer row, which has room, so the tools fit on one line.
  const figure = h('div', { class: 'row', style: 'gap:2px;margin-left:auto' },
    h('button', { class: `btn ${S.editor.open ? 'on' : ''}`, title: 'Open the plot editor (series, legend, axes, text, style). Tip: double-click titles/legend on the plot to edit in place.', onclick: () => { S.editor.open = !S.editor.open; emit(); } }, icon('edit', 14), S.editor.open ? 'Close editor' : 'Edit plot'),
    !tt ? h('button', { class: 'btn sm', title: 'Jump to an editor section', 'aria-label': 'Editor sections', onclick: (e) => {
      const r = e.currentTarget.getBoundingClientRect();
      contextMenu(r.left, r.bottom + 4, [['series', 'Series'], ['legend', 'Legend'], ['axes', 'Axes & ticks'], ['text', 'Annotations & results'], ['inset', 'Inset & breaks'], ['style', 'Layout & style']]
        .map(([tab, label]) => ({ label, run: () => { S.editor.open = true; S.editor.tab = tab; emit(); } })));
    } }, '▾') : null,
    h('button', { class: 'ibtn', title: 'Copy figure to the clipboard (Ctrl+Shift+C)', 'aria-label': 'Copy figure', onclick: () => copyFigure(f) }, icon('copy', 15)));
  chips.style.flex = '1 1 100%';
  chips.append(figure);
  return h('div', { class: 'toolbar', style: 'flex:none' }, chips, tools);
}

function setTool(f, tool) {
  f.tool = tool;
  if (plotEl?.data) Plotly.relayout(plotEl, { dragmode: f.ttPick ? 'select' : tool === 'pan' ? 'pan' : tool === 'select' ? 'select' : 'zoom' });
  emit();
}

function plotSection(f) {
  const card = h('section', { class: 'plotcard', 'aria-label': 'Plot', 'data-drop': kindOf(f) === '1d' ? 'compare' : null }, plotDiv());
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
  else fig = figureMaps(f, displayedOverlay(f));
  if (!fig) return;
  if (fftEl?.isConnected) {
    const pk = fig.fftPeak, sampling = fftAvailability(f);
    fftEl.textContent = f.layers.fft && !sampling.ok ? `FFT unavailable: ${sampling.reason}`
      : f.layers.fft && pk && Number.isFinite(pk.f) ? `FFT peak ≈ ${fmt(pk.f, 5)} ${pk.unit || ''}` : '';
  }
  fig.layout.dragmode = f.ttPick && k === 'tt' ? 'select' : f.tool === 'pan' ? 'pan' : f.tool === 'select' ? 'select' : 'zoom';
  // Only the newest request draws. If the box is not laid out yet, retry each frame: the plot div is reused, so the
  // ResizeObserver alone never fires when it settles at its previous size (the next file then stayed undrawn/stale).
  const token = ++drawToken;
  let tries = 0;
  const attempt = () => {
    if (token !== drawToken) return;
    if (!roomy(el)) {
      if (++tries < 90) requestAnimationFrame(attempt); else plotPending = true;
      return;
    }
    plotPending = false;
    // The 1D double-click editor owns text; native single-click editing steals its second click.
    Plotly.react(el, fig.data, fig.layout, plotConfig({ edits: { ...plotConfig().edits, titleText: k !== '1d' && !!f.style.title, axisTitleText: k !== '1d', legendText: k !== '1d' } }));
    bindPlot(el);
    if (k === '2d') { fitColorbars(el).then(() => rememberMapRanges(el)); drawCuts(f); }
  };
  requestAnimationFrame(attempt);
}

function drawCuts(f) {
  const cuts = linecuts(f);
  if (!cuts || !cutEl) return;
  const colors = ['#D63384', '#11998C', '#1F5FBF'];
  const data = [], st = baseStyle(f.style);
  const norm = f.map.cutNorm !== false && cuts.length > 1;
  const nz = (arr) => { if (!norm) return arr; const m = finiteExtent(arr, [0, 0])[1]; return arr.map((v) => v / (m || 1)); };
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
    yaxis: { showline: true, mirror: true, ticks: 'outside', domain: both ? [0.58, 1] : [0, 1], exponentformat: 'SI', title: { text: norm ? 'Normalized to max' : mapValueLabel(f) } },
    separators: '.',
  };
  if (both) { layout.xaxis2 = { showline: true, mirror: true, ticks: 'outside', anchor: 'y2' }; layout.yaxis2 = { showline: true, mirror: true, ticks: 'outside', domain: [0, 0.4], exponentformat: 'SI' }; }
  Plotly.react(cutEl, data, layout, { displayModeBar: false, responsive: true });
}

function bindPlot(el) {
  if (plotBound) return;
  plotBound = true;
  inlinePlotEditor(el, active, () => clearTimeout(legendTimer));
  let mapClickTimer = null;
  const cancelMapClick = () => { if (mapClickTimer !== null) { clearTimeout(mapClickTimer); mapClickTimer = null; } };
  el.on('plotly_click', (ev) => {
    const f = active();
    if (!f || !ev.points?.length) return;
    const p = ev.points[0];
    if (kindOf(f) === '2d') {
      if (p.data.type !== 'heatmap') return;
      // Plotly emits a click before it recognizes a double-click reset. Defer the
      // cursor action so a reset cannot unexpectedly open the linecut panel.
      cancelMapClick();
      if (ev.event?.detail >= 2) return;
      const path = f.path, detector = f.detector, linecut = f.map.linecut, point = [p.x, p.y];
      mapClickTimer = setTimeout(() => {
        mapClickTimer = null;
        const current = active();
        if (!current || current.path !== path || current.detector !== detector || kindOf(current) !== '2d' || current.map.linecut !== linecut) return;
        if (linecut === 'line') {
          // Display coordinates (after swap); a third click starts a new line.
          const L = current.map.line;
          current.map.line = L && L.length === 1 ? [L[0], point] : [point];
        } else {
          current.map.cursor = point;
          if (linecut === 'none') current.map.linecut = 'h';
        }
        emit();
      }, 350);
      return;
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
          parts.push(`${detLabel(f, det)} ${fmt(d.z[nearest(d.y, p.y)]?.[nearest(d.x, p.x)], 6)}${mapUnitSuffix(f)}`);
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
  el.on('plotly_doubleclick', cancelMapClick);
  el.on('plotly_relayout', (ev) => {
    const f = active();
    if (!f) return;
    if (applyPlotlyEdit(f, ev)) return;
    if (kindOf(f) === '2d') {
      // Zoom/pan/reset on any panel -> one shared view (linked panels follow on re-render; one-axis zooms made uniform).
      const view = mapRelayoutView(el, ev);
      if (view) { f.map.bounds = view.reset ? null : { x: view.x, y: view.y }; emit(); }
      return;
    }
    const ranges = Object.keys(ev).some((key) => /^(x|y)axis\.(range|autorange)/.test(key));
    if (ranges) commit('plot range');
    if (ev['xaxis.range[0]'] !== undefined) f.xrange = [ev['xaxis.range[0]'], ev['xaxis.range[1]']].map((v) => f.style.x?.log ? 10 ** v : v);
    if (ev['yaxis.range[0]'] !== undefined) f.yrange = [ev['yaxis.range[0]'], ev['yaxis.range[1]']].map((v) => f.style.y?.log ? 10 ** v : v);
    if (ev['xaxis.autorange']) f.xrange = null;
    if (ev['yaxis.autorange']) f.yrange = null;
    // Keep saved manual bounds in step with direct zoom/reset, including exports.
    for (const ax of ['x', 'y']) if (f.style[ax] && Object.keys(ev).some((k) => k.startsWith(`${ax}axis.range`) || k === `${ax}axis.autorange`)) {
      if (f.style[ax].min != null || f.style[ax].max != null) { f.style[ax].min = f[`${ax}range`]?.[0] ?? null; f.style[ax].max = f[`${ax}range`]?.[1] ?? null; }
    }
  });
  el.on('plotly_restyle', (ev) => { const f = active(); if (f) applyPlotlyEdit(f, ev, el, 'restyle'); });
  el.on('plotly_legendclick', (ev) => {
    const f = active(); if (!f) return true;
    const key = ev.data[ev.curveNumber]?.meta?.key;
    if (!key) return true;
    clearTimeout(legendTimer);
    if (ev.event?.altKey) { openSeriesEditor(f, key, ev.event.clientX, ev.event.clientY); return false; }
    if (ev.event?.detail >= 2) return false;
    legendTimer = setTimeout(() => {
      commit('legend visibility'); f.style.series = f.style.series || {};
      const cur = f.style.series[key]?.visible;
      f.style.series[key] = { ...(f.style.series[key] || {}), visible: cur === false ? true : false };
      emit();
    }, 350);
    return false;
  });
  el.on('plotly_legenddoubleclick', (ev) => {
    clearTimeout(legendTimer);
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
    const picked = ev.points.filter((p) => p.data.meta?.key === 'data');
    picked.forEach((p) => (d.proc.groups[p.pointIndex] || []).forEach((r) => idx.add(r)));
    const interpolated = picked.some((p) => d.proc.derived?.[p.pointIndex]);
    toast(`Excluded ${ev.points.length} point(s)${interpolated ? '; interpolated points remove the measured points they were computed from' : ''}. Undo with Ctrl+Z.`);
    f.tool = 'zoom'; // one-shot: return to zoom after excluding
    setProc(f, { excluded: [...idx] }, 'exclude selection');
  });
  el.addEventListener('dblclick', (e) => {
    const f = active();
    if (!f) return;
    if (kindOf(f) === '2d') cancelMapClick();
    const t = e.target.closest('.scatterlayer .trace, .legendtext, .legendtoggle');
    if (t && e.target.closest('.legend')) return; // handled by legend events
  });
  el.addEventListener('mousedown', (e) => { if (e.button === 2) e.stopPropagation(); }, true);
  el.on('plotly_hover', (ev) => { const p = ev.points?.[0]; if (p?.data?.type === 'heatmap') lastMapHover = [p.x, p.y]; });
  el.addEventListener('contextmenu', (e) => {
    const f = active(); if (!f) return;
    e.preventDefault();
    e.stopPropagation();
    const map = kindOf(f) === '2d';
    const cutAt = (pos, mode) => { if (!pos) return; f.map.cursor = pos; f.map.linecut = mode; emit(); };
    contextMenu(e.clientX, e.clientY, [
      map && lastMapHover ? { label: 'Horizontal cut here', run: () => cutAt(lastMapHover, 'h') } : null,
      map && lastMapHover ? { label: 'Vertical cut here', run: () => cutAt(lastMapHover, 'v') } : null,
      map && lastMapHover ? { label: 'Both cuts here', run: () => cutAt(lastMapHover, 'cross') } : null,
      map ? { label: 'Cut at peak', run: () => cutAt(peakCell(f, f.detector), f.map.linecut === 'none' ? 'cross' : f.map.linecut) } : null,
      map ? { label: 'Cut at dip', run: () => cutAt(peakCell(f, f.detector, true), f.map.linecut === 'none' ? 'cross' : f.map.linecut) } : null,
      map && f.map.cursor ? { label: 'Clear cut', run: () => { f.map.cursor = null; f.map.linecut = 'none'; emit(); } } : null,
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
    const fit = currentResult(f,det);
    const tag = dets.length > 1 ? ` · ${detLabel(f, det)}` : '';
    const x = d.proc.x || [];
    cols.push(tt ? x : xDisplay(f, comparisonFiles(f)).transform(x)); head.push(`x${tag} (${xt})`);
    cols.push(d.proc.y || []); head.push(`y${tag} (${tt ? `${detLabel(f, det)}${f.info.unit ? ` [${f.info.unit}]` : ''}` : `${MODE_LABEL[f.mode]} [${f.mode === 'contrast' ? 'dimensionless' : f.info.unit || 'a.u.'}] · processed, not normalized · ${iterationProvenance(f)}`})`);
    if (d.proc.sem) { cols.push(d.proc.sem); head.push(`sem${tag}`); }
    if (d.proc.smooth) { cols.push(d.proc.smooth); head.push(`smoothed${tag} (${d.proc.smoothing_window ?? f.proc.smooth} pt)`); }
    if (fit?.x && fit.y_fit) {
      const same = fit.x.length === x.length && fit.x.every((v, i) => Math.abs(v - x[i]) <= 1e-9 * Math.max(1, Math.abs(v)));
      const fy = x.map((xv, i) => { const j = same ? i : fit.x.findIndex((v) => Math.abs(v - xv) <= 1e-9 * Math.max(1, Math.abs(v))); return j >= 0 ? fit.y_fit[j] : null; });
      const aligned = (values) => x.map((xv) => { const j = fit.x.findIndex((v) => Math.abs(v - xv) <= 1e-9 * Math.max(1, Math.abs(v))); return j >= 0 ? values?.[j] : null; });
      cols.push(fy); head.push(`fit${tag} (${fit.plot_y_label || f.mode} · ${fitTargetLabel(fit)})`);
      cols.push(aligned(fit.plot_y)); head.push(`fit_target_y${tag} (${fit.plot_y_label || f.mode})`);
      cols.push(aligned(fit.residuals)); head.push(`residual${tag} (stored fit-space target − fit)`);
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
  const fit = currentResult(f);
  const row = h('div', { class: 'readout' });
  readoutLive = null; fftEl = null;
  if (f.tool === 'measure') {
    const m = f.measure || [];
    if (m.length < 2) row.append(h('span', { class: 'muted' }, `Measure: click ${2 - m.length} point${m.length ? '' : 's'} on the plot.`));
    else row.append(h('span', { class: 'muted' }, 'Δ'), h('span', { class: 'mono' }, `Δx ${fmt(m[1].x - m[0].x, 6)}`), h('span', { class: 'mono' }, `Δy ${fmt(m[1].y - m[0].y, 6)}`),
      h('span', { class: 'mono dim' }, `slope ${fmt((m[1].y - m[0].y) / (m[1].x - m[0].x), 4)}`), h('button', { class: 'btn sm', onclick: () => { f.measure = []; emit(); } }, 'Clear'));
    return row;
  }
  if (!f.pins.length) {
    const hint = h('span', { class: 'dim hint-line', title: kindOf(f) === 'tt' ? '' : 'Click a point to pin it · Ctrl-click adds more · Esc clears · double-click a legend label, title or axis title to rename it · Alt-click a legend entry (or Series → Style & legend) to change its appearance · drag the legend/results box to move it · right-click for more' },
      kindOf(f) === 'tt' ? 'Click a sample to inspect its timing. Set windows A/B in the Stats tab.' : 'Click to pin a point · double-click any text to edit · right-click for more');
    hint.dataset.def = hint.textContent; readoutLive = hint;
    fftEl = f.layers.fft ? h('span', { class: 'mono' }) : null;
    row.append(hint, fftEl, h('span', { class: 'sp' }), h('button', { class: 'btn sm', title: 'Copy the processed data (and fit, residual) as a tab-separated table', onclick: () => copyData(f) }, 'Copy data'));
    return row;
  }
  const p = f.pins[f.pins.length - 1];
  if (f.layers.fft && kindOf(f) === '1d') { fftEl = h('span', { class: 'mono' }); row.append(fftEl); }
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
    if (fit) { const nativeX = xDisplay(f, comparisonFiles(f)).inverse(p.x); const j = fit.x.findIndex((v) => Math.abs(v - nativeX) < 1e-9 * Math.max(1, Math.abs(v))); if (j >= 0) row.append(h('span', { class: 'mono dim' }, `fit ${fmt(fit.y_fit[j], 6)}`)); }
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
  // Everyday controls on one row; the less-changed display settings collapse behind a summary of their current values.
  const summary = `${M.cmap} · ${M.plo}–${M.phi}% · ${M.scale} · ${M.geometry === 'square' ? 'square' : 'physical'} pixels${M.swap ? ' · swapped' : ''}`;
  return h('div', { style: 'display:flex;flex-direction:column;gap:6px;flex:none' },
    h('div', { class: 'row wrap', style: 'gap:6px' },
      seg(layouts, M.layout, async (v) => { M.prevLayout = M.layout; M.layout = v; if (v.includes('overlay')) await ensureOverlay(f); emit(); }),
      h('label', { class: 'row', style: 'gap:4px', title: 'Colormap' }, h('span', { class: 'cm on', style: `background:linear-gradient(90deg,${CMAP_GRADIENT[M.cmap] || CMAP_GRADIENT.viridis});width:34px` }),
        select(['viridis', 'parula', 'cividis', 'magma', 'inferno', 'plasma', 'turbo', 'coolwarm', 'RdBu', 'gray'], M.cmap, (v) => set({ cmap: v }), { style: 'width:92px;height:28px', 'aria-label': 'Colormap' })),
      seg([['none', 'No cut'], ['h', 'Horiz.'], ['v', 'Vert.'], ['cross', 'Both'], ['line', 'Line']], M.linecut, (v) => { if (v !== 'none' && v !== 'line' && !M.cursor) M.cursor = peakCell(f, f.detector); set({ linecut: v }); }),
      h('button', { class: 'btn sm', title: 'Put the cursor (and linecuts) on the brightest cell', onclick: () => { M.cursor = peakCell(f, f.detector); if (M.linecut === 'none') M.linecut = 'h'; emit(); } }, 'Cut at peak'),
      h('button', { class: 'btn sm', title: 'Put the cursor (and linecuts) on the darkest cell', onclick: () => { M.cursor = peakCell(f, f.detector, true); if (M.linecut === 'none') M.linecut = 'h'; emit(); } }, 'Cut at dip'),
      h('span', { class: 'sp' }),
      h('button', { class: 'ibtn', title: 'Reset view (Ctrl+Home)', onclick: () => resetView(f) }, icon('home')),
      h('button', { class: `ibtn ${f.tool !== 'pan' ? 'on' : ''}`, title: 'Box zoom', onclick: () => setTool(f, 'zoom') }, icon('zoom')),
      h('button', { class: `ibtn ${f.tool === 'pan' ? 'on' : ''}`, title: 'Pan', onclick: () => setTool(f, 'pan') }, icon('pan')),
      h('button', { class: 'btn', onclick: () => { S.editor.open = !S.editor.open; emit(); } }, icon('edit', 14), 'Edit plot'),
      h('button', { class: 'btn', title: 'Copy figure (Ctrl+Shift+C)', onclick: () => copyFigure(f) }, icon('copy', 14), 'Copy')),
    h('details', { class: 'map-display', open: !!M.displayOpen, ontoggle: (e) => { M.displayOpen = e.target.open; } },
      h('summary', { title: 'Colour range, shared colour scale, pixel geometry, axis swap' }, `Display · ${summary}`),
    h('div', { class: 'mapbar' },
      h('div', { class: 'row', style: 'gap:6px' }, h('span', { class: 'lbl' }, 'Colormap'),
        cms.map((c) => h('button', { class: `cm ${M.cmap === c ? 'on' : ''}`, title: c, 'aria-label': c, style: `background:linear-gradient(90deg,${CMAP_GRADIENT[c]})`, onclick: () => set({ cmap: c }) }))),
      h('div', { class: 'row', style: 'gap:6px' }, h('span', { class: 'lbl' }, 'Range'),
        input(M.plo, (v) => set({ plo: num(v) ?? 1, limits: 'robust' }), { numeric: true, style: 'width:52px;height:26px', title: 'Low percentile' }), h('span', { class: 'dim' }, 'to'),
        input(M.phi, (v) => set({ phi: num(v) ?? 99.5, limits: 'robust' }), { numeric: true, style: 'width:58px;height:26px', title: 'High percentile' }), h('span', { class: 'dim' }, '%')),
      seg([['independent', 'Independent'], ['shared', 'Shared']], M.scale, (v) => set({ scale: v })),
      seg([['physical', 'Physical pixels'], ['square', 'Square']], M.geometry, (v) => { set({ geometry: v }); call('settings_set', { patch: { pixel_geometry: v } }, { quiet: true }); }),
      h('button', { class: `btn ${M.swap ? '' : ''}`, onclick: () => { commit('swap axes'); M.swap = !M.swap; M.bounds = null; M.cursor = M.cursor ? [M.cursor[1], M.cursor[0]] : null; if (M.line) M.line = M.line.map((q) => [q[1], q[0]]); if (overlayCache[f.path]) ensureOverlay(f).then(emit); emit(); } }, `Swap ${swapLabel(f)}`))));
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
    const hint = h('span', { class: 'dim hint-line', title: 'Click the map to pin a cell and draw linecuts · scroll to zoom · drag to box-zoom · double-click or Ctrl+Home resets' }, 'Click to pin a cell + linecuts · drag to zoom · double-click resets');
    hint.dataset.def = hint.textContent; readoutLive = hint;
    row.append(hint); return row;
  }
  const m = f.data[f.detector].map;
  const d = displayMap(m, f.map.swap);
  row.append(h('span', { class: 'muted' }, 'Cursor'), h('span', { class: 'mono' }, `${friendlyAxis(d.xl).split(' [')[0]} ${f.map.cursor[0]}`), h('span', { class: 'mono' }, `${friendlyAxis(d.yl).split(' [')[0]} ${f.map.cursor[1]}`));
  (cuts || f.info.detectors.map((det) => ({ det, label: detLabel(f, det), value: (() => { const dd = displayMap(f.data[det].map, f.map.swap); return dd.z[nearest(dd.y, f.map.cursor[1])][nearest(dd.x, f.map.cursor[0])]; })() })))
    .forEach((c) => row.append(h('span', { class: 'mono' }, `${c.label} ${fmt(c.value, 6)}${mapUnitSuffix(f)}`)));
  row.append(h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: () => { f.map.cursor = null; f.map.line = null; f.map.linecut = 'none'; emit(); } }, 'Clear'));
  return row;
}

export function mapRequest(f) {
  const M = f.map;
  let second = M.compareWith;
  if (!second && f.info.detectors.length > 1) second = { path: f.path, detector: f.info.detectors.find((d) => d !== f.detector) };
  if (!second) return null;
  return {
    first: { path: f.path, detector: f.detector, mode: f.mode, source: M.source || 'value' },
    second: { ...second, mode: f.mode, source: M.source || 'value' },
    opts: { shared: M.overlayScale === 'shared', background: M.background, palette: M.palette, swap: M.swap },
  };
}
function displayedOverlay(f) {
  const overlay = overlayCache[f.path];
  if (!overlay || canonical(overlay.request) !== canonical(mapRequest(f))) return null;
  for (const env of overlay.analyses || []) {
    const owner = Object.values(S.files).find((o) => sourcePath(o.path) === env.descriptor.source);
    if (owner?.info.source_revision?.sha256 && owner.info.source_revision.sha256 !== env.source_revision?.sha256) return null;
  }
  return overlay;
}
export async function ensureOverlay(f) {
  const request = mapRequest(f);
  if (!request) {
    toast('Add another 2D file of the same plane, then pick it in Compare.');
    return;
  }
  delete f.overlayErrorKey;
  try {
    const result = await call('map_overlay', request, { label: 'Building overlay' });
    recordResult(f, result, { kind: 'map_overlay' });
    if (canonical(request) === canonical(mapRequest(f))) {
      overlayCache[f.path] = result;
      return result;
    }
  } catch (e) {
    overlayCache[f.path] = null;
    f.overlayErrorKey = canonical(request);
  }
}

// ---------------------------------------------------------------- inspector
const TABS = {
  '1d': [['fit', 'Fit'], ['proc', 'Process'], ['plot', 'Plot'], ['export', 'Export'], ['info', 'Info']],
  tt: [['stats', 'Stats'], ['fit', 'Fit'], ['proc', 'Process'], ['plot', 'Plot'], ['export', 'Export'], ['info', 'Info']],
  '2d': [['map', 'Map'], ['spots', 'Spots'], ['compare', 'Compare'], ['export', 'Export'], ['info', 'Info']],
};
// Alt+1..7 keep the original app's numbering: Model, Params, Process, Plot, Results, Map, Export.
export const ALT_TABS = { 1: 'fit', 2: 'fit', 3: 'proc', 4: 'plot', 5: 'fit', 6: 'map', 7: 'export', 8: 'info' };

const inspectorScroll = new WeakMap();
export function renderInspector(f, insp) {
  const previous = insp.querySelector('.insp-body');
  if (previous && previous.dataset.path === f.path) {
    const saved = inspectorScroll.get(f) || {};
    saved[previous.dataset.tab] = previous.scrollTop;
    inspectorScroll.set(f, saved);
  }
  const k = kindOf(f);
  const tabs = TABS[k];
  if (!tabs.some(([t]) => t === f.tab)) f.tab = tabs[0][0];
  const body = h('div', { class: 'insp-body' });
  body.dataset.path = f.path; body.dataset.tab = f.tab;
  body.addEventListener('scroll', () => { const saved = inspectorScroll.get(f) || {}; saved[body.dataset.tab] = body.scrollTop; inspectorScroll.set(f, saved); });
  const builders = { fit: fitTab, proc: procTab, plot: plotTab, export: exportTab, info: infoTab, stats: statsTab, map: mapTab, spots: spotsTab, compare: mapCompareTab };
  const pane = builders[f.tab](f);
  if (f.tab === 'stats') pane.append(sci.ttSciCard(f));
  body.append(pane);
  const keep = previous && previous.dataset.path === f.path && previous.dataset.tab === f.tab ? redrawState(previous) : null;
  mount(insp, resultHeader(f), h('div', { class: 'tabs', role: 'tablist' }, tabs.map(([t, l]) => h('button', { class: f.tab === t ? 'on' : '', role: 'tab', onclick: () => { f.tab = t; emit('insp'); } }, l))), body);
  body.scrollTop = inspectorScroll.get(f)?.[f.tab] || 0;
  if (keep) restoreRedrawState(body, keep);
}

// UX08: a redraw rebuilds the inspector; carry over which sections were open and which field had focus
// (located by its position among the pane's fields) so a background load or fit does not interrupt editing.
const FIELDS = 'input, select, textarea, button';
function redrawState(body) {
  const fields = [...body.querySelectorAll(FIELDS)], active = document.activeElement;
  const i = active && body.contains(active) ? fields.indexOf(active) : -1;
  return {
    open: new Set([...body.querySelectorAll('details[open] > summary')].map((s) => s.textContent)),
    closed: new Set([...body.querySelectorAll('details:not([open]) > summary')].map((s) => s.textContent)),
    focus: i >= 0 ? { i, count: fields.length, tag: active.tagName, value: active.value, start: active.selectionStart, end: active.selectionEnd } : null,
  };
}
function restoreRedrawState(body, keep) {
  body.querySelectorAll('details > summary').forEach((s) => {
    if (keep.open.has(s.textContent)) s.parentElement.open = true;
    else if (keep.closed.has(s.textContent)) s.parentElement.open = false;
  });
  const fields = [...body.querySelectorAll(FIELDS)], el = keep.focus && fields.length === keep.focus.count ? fields[keep.focus.i] : null;
  if (!el || el.tagName !== keep.focus.tag) return;
  el.focus({ preventScroll: true });
  // An uncommitted edit in a text field survives the redraw instead of snapping back to the stored value.
  if (el.tagName === 'INPUT' && el.value !== keep.focus.value && /^(text|search|)$/.test(el.type)) el.value = keep.focus.value;
  try { if (keep.focus.start != null) el.setSelectionRange(keep.focus.start, keep.focus.end); } catch { /* not a text input */ }
}

// UX06: what the shown result belongs to, always visible above the tabs.
function resultHeader(f) {
  const fit = f.fit?.[f.detector];
  const p = f.proc || {};
  const proc = [p.bin > 1 ? `bin ${p.bin}` : '', p.roi_min != null || p.roi_max != null ? 'ROI' : '', p.excluded?.length ? `${p.excluded.length} excluded` : '',
    p.steps?.length ? `${p.steps.length} transform${p.steps.length > 1 ? 's' : ''}` : '', p.iterations ? 'iteration subset' : ''].filter(Boolean);
  const identity = [detLabel(f, f.detector), MODE_LABEL[f.mode] || f.mode, proc.length ? proc.join(', ') : 'unprocessed'].join(' · ');
  if (!fit) return h('div', { class: 'result-head', role: 'status' }, h('div', { class: 'ellip', title: f.path }, h('b', {}, f.info?.name || f.path)), h('div', { class: 'dim ellip' }, `${identity} · not fitted`));
  const status = f.fitStale ? 'STALE' : fit.status;
  const key = (fit.params || []).filter((q) => /^(T1|T2|T2_star|T|tau|tau_s|f0|freq|center_1|x0|pi_time)/.test(q.name)).slice(0, 2)
    .map((q) => { const d = displayValue(q.value, q.error, parameterUnit(q, f.info), filePresentation(f, f)); return `${q.name} ${d.text}${d.unit && d.unit !== '-' ? ` ${d.unit}` : ''}`; });
  return h('div', { class: 'result-head', role: 'status', 'aria-label': 'Shown result' },
    h('div', { class: 'row', style: 'gap:6px' }, h('b', { class: 'ellip', style: 'flex:1', title: f.path }, f.info.name), h('span', { class: `pill ${status}`, title: f.fitStale ? 'Processing or fit settings changed since this fit; refit to update it' : fit.reason || '' }, status)),
    h('div', { class: 'dim ellip', title: identity }, `${identity} · ${fit.model}`),
    key.length ? h('div', { class: 'mono ellip' }, `${key.join(' · ')} · R² ${fmt(fit.r2, 4)}`) : h('div', { class: 'mono' }, `R² ${fmt(fit.r2, 4)}`));
}

function t1DisplayControl(f) {
  const fit = f.fit[f.detector], n = t1Normalization(fit);
  return seg([['original', 'Original units'], ['normalized', 'Normalized 1 − data']], displaySpace(f), (v) => {
    commit('T1 display units'); f.display = { ...f.display, t1: v }; f.yrange = null;
    if (f.style.y) { f.style.y.min = null; f.style.y.max = null; }
    emit();
  }, { disabled: n ? [] : ['original'] });
}

function overlaysCard(f) {
  const paths = S.overlays.filter((p) => p !== f.path);
  const rows = [f.path, ...paths].filter((p) => S.files[p]).map((p) => {
    const o = S.files[p], det = o.info.detectors.includes(f.detector) ? f.detector : o.detector;
    const r = currentResult(o,det), t1 = r?.params.find((q) => q.name === 'T1');
    return { o, det, r, t1 };
  });
  return h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' },
    h('div', { class: 'row wrap' }, h('span', { class: 'lbl' }, `Overlays (${paths.length})`), h('span', { class: 'sp' }),
      h('button', { class: 'btn sm', onclick: () => openOverlayPicker(f) }, 'Compare with…'), h('button', { class: 'btn sm', onclick: () => browseOverlays(f) }, 'Browse…')),
    h('div', { class: 'note', title: 'Same observable and detector where available; otherwise the file’s active detector. Each normalized file uses its own lo/span. Data-derived normalization is marked in the legend. Maps and time traces use the Compare workspace.' }, 'Same observable/detector; each file keeps its own normalization (hover for details).'),
    paths.map((p) => h('div', { class: 'row', title: p }, h('span', { class: 'ellip', style: 'flex:1' }, basename(p)),
      h('button', { class: 'btn sm', onclick: () => { commit('remove comparison'); removeOverlay(p); } }, 'Remove'))),
    paths.length ? check('Show compared fits (dashed, same colour)', f.display?.overlayFits !== false, (v) => { commit('compared fits visibility'); f.display = { ...f.display, overlayFits: v }; emit(); }) : null,
    paths.length ? h('div', { class: 'row wrap' }, h('button', { class: 'btn sm', onclick: () => { commit('clear comparisons'); S.overlays = []; emit(); } }, 'Clear all'),
      h('button', { class: 'btn sm', disabled: f.compareFitting, title: 'Copies the current fit settings and range. Each file keeps its own processing; existing fit results are replaced. Does not change the primary file.', onclick: async () => {
        commit('fit compared files'); f.compareFitting = true; emit('insp');
        try {
          for (const p of paths) {
            const o = S.files[p]; if (!o) continue;
            try {
              await fitSeries(f, o, { recordUndo: false });
            } catch (e) { toast(`Fit skipped for ${basename(p)}: ${e.message}`, 'err'); }
          }
        } finally { f.compareFitting = false; emit(); }
      } }, f.compareFitting ? 'Fitting…' : 'Fit compared files')) : null,
    rows.some((r) => r.t1) ? h('table', { class: 'tbl', 'aria-label': 'T1 comparison results' },
      h('tr', {}, h('th', {}, 'File / detector'), h('th', {}, 'T1 ± error'), h('th', {}, 'Unit')),
      rows.map(({ o, det, r, t1 }) => h('tr', { title: `${o.path}\n${r?.selection || 'Not fit'}${o.fitStale ? '\nFit is from earlier settings' : ''}` },
        h('td', { class: 'txt', style: 'max-width:145px;overflow-wrap:anywhere' }, `${o.info.name} · ${detLabel(o, det)}`),
        h('td', {}, t1 ? displayValue(t1.value, t1.error, t1.unit || o.info.x_unit || 'ns', filePresentation(f, o)).text : 'Not fit'), h('td', {}, t1 ? displayValue(t1.value, t1.error, t1.unit || o.info.x_unit || 'ns', filePresentation(f, o)).unit : '—')))) : null);
}

// ---- ODMR display controls (rendering lives in plot1d.js; stored in f.display = { invert, xscale, xoffset })
function displayControls(f) {
  f.display = f.display || { invert: false, xscale: 1, xoffset: 0 };
  const D = f.display;
  const upd = (patch) => { commit('display'); Object.assign(D, patch); emit(); };
  return h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Display'),
    seg([[false, 'Peaks'], [true, 'Dips (invert)']], !!D.invert, (v) => upd({ invert: v })),
    h('div', { class: 'g2' }, field('Axis scale (×)', input(D.xscale ?? 1, (v) => upd({ xscale: num(v) || 1 }), { numeric: true, title: 'Displayed x = scale · x + offset' })),
      field(`Axis offset (native ${f.info.x_unit || 'axis units'})`, input(D.xoffset ?? 0, (v) => upd({ xoffset: num(v) ?? 0 }), { numeric: true }))),
    h('div', { class: 'note' }, 'Display only: the fit and exported numbers stay in the file\'s own units.'));
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

export function openFitSettingsDialog(o) {
  const body = h('div', { style: 'overflow:auto;padding:14px' });
  const shade = h('div', { class: 'modal-back', role: 'dialog', 'aria-label': `Fit settings ${o.info.name}` });
  let off;
  const close = () => { off?.(); document.removeEventListener('keydown', esc, true); shade.remove(); };
  const esc = (e) => { if (e.key === 'Escape') close(); };
  shade.append(h('div', { class: 'modal' }, h('div', { class: 'row', style: 'padding:12px' }, h('strong', {}, `Own settings · ${o.info.name}`), h('span', { class: 'sp' }), h('button', { class: 'btn sm primary', onclick: close }, 'Done')), body));
  const draw = () => mount(body, fitTab(o));
  off = subscribe(draw); draw(); document.body.append(shade); document.addEventListener('keydown', esc, true);
}

function fitTab(f) {
  const s = f.fitSpec;
  const fam = family(f);
  const fit = f.fit[f.detector];
  const set = (patch, label = 'fit settings') => { commit(label); Object.assign(s, patch); f.fitStale = !!Object.keys(f.fit).length; f.iterationFitsStale = true; emit('insp'); };
  const out = h('div', { style: 'display:flex;flex-direction:column;gap:12px' });
  const generic = [['Auto', `Auto · ${fam}`], ['GaussianPeaks', 'Gaussian peaks'], ['CustomModel', 'Custom expression']];
  if (kindOf(f) === 'tt') generic.splice(0, 1, ['Auto', 'TimeTraceDecay · background + A·exp(−t/τ)']);
  out.append(field('Model', select(generic, s.model, (v) => set({ model: v }))));
  if (fam === 'ODMR' && kindOf(f) === '1d') out.append(displayControls(f));
  if (s.model === 'Auto') {
    if (fam === 'Rabi') {
      out.append(field('Rabi model', select(MODELS_BY_FAMILY.Rabi, s.rabi_mode, (v) => set({ rabi_mode: v }))), h('div', { class: 'note' }, MODEL_HELP[s.rabi_mode] || ''));
    } else if (fam === 'Ramsey') {
      out.append(field('Ramsey model', select(['Auto', 'Simple damped cosine', 'N14 hyperfine', 'Extended'], s.ramsey_model, (v) => set({ ramsey_model: v }))),
        h('div', { class: 'note' }, 'Auto fits all three and prefers models with R² ≥ 0.85, then fewer bound hits, then lower BIC.'));
    } else if (fam === 'ODMR') {
      out.append(h('div', { class: 'g2' }, field('Polarity', select(['Auto', 'Peak', 'Dip'], s.odmr_polarity, (v) => set({ odmr_polarity: v }))),
        field('Peaks', select([['auto', 'Auto'], ['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], s.odmr_auto_peaks ? 'auto' : String(s.odmr_peaks), (v) => set(v === 'auto' ? { odmr_auto_peaks: true } : { odmr_auto_peaks: false, odmr_peaks: Number(v), odmr_multi: Number(v) > 1 })))),
      s.odmr_auto_peaks ? check('Multi-Lorentzian when several peaks are found', s.odmr_multi, (v) => set({ odmr_multi: v })) : null,
      odmrRangeChips(f, s, set));
      out.append(peakPickCard(f));
    } else if (fam === 'T1') {
      out.append(field('Preprocess', seg([['Auto', 'Auto'], ['Direct T1', 'Direct'], ['Normalize then 1-data', 'Normalize, 1 − data']], s.t1_prep, (v) => set({ t1_prep: v }))),
        field('Shape', seg([['Regular exponential', 'Exponential'], ['Stretched exponential', 'Stretched']], s.t1_family, (v) => set({ t1_family: v }))));
    } else if (fam === 'SpinEcho') {
      out.append(h('div', { class: 'note' }, 'Stretched exponential y0 + A·exp(−(t/T2)ⁿ). Lock n in Seeds & bounds to force n = 1, 2 or 3.'));
    } else if (fam === 'LineScan' || fam === 'Scan2D') {
      out.append(h('div', { class: 'note' }, fam === 'LineScan' ? 'Auto fits one Gaussian peak on a constant background (as Batch does). Choose Gaussian peaks to fit several.' : '2D maps are not fitted here; use Spots to fit Gaussian spots.'));
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
        s.custom_params.map((p) => h('tr', {}, h('td', { class: 'txt' }, p.name), ['seed', 'min', 'max'].map((k) => h('td', {}, input(p[k] ?? '', (v) => { p[k] = num(v); set({}); }, { numeric: true, placeholder: k === 'seed' ? '1' : '–' })))))));
  }
  // range
  const fr = f.data[f.detector]?.proc?.x || [];
  const splitAt = out.children.length; // everything appended from here on is secondary (moved below the results)
  const rangeUnit = f.info.type === 'TimeTrace' ? 's' : f.info.x_unit || '';
  out.append(h('div', { class: 'f' }, h('span', { class: 'lbl' }, `Fit range (native ${f.info.type === 'TimeTrace' ? 's' : f.info.x_unit || 'axis units'})`),
    h('div', { class: 'row wrap', style: 'gap:4px' }, input(s.range_min ?? '', (v) => set({ range_min: num(v) }), { numeric: true, unit: rangeUnit, placeholder: fr.length ? fmt(finiteExtent(fr)[0], 5) : 'min', style: 'width:90px' }), h('span', { class: 'dim' }, 'to'),
      input(s.range_max ?? '', (v) => set({ range_max: num(v) }), { numeric: true, unit: rangeUnit, placeholder: fr.length ? fmt(finiteExtent(fr)[1], 5) : 'max', style: 'width:90px' }),
      h('button', { class: 'btn sm', title: 'Use the current zoomed x range', onclick: () => { const r = f.xrange?.map(f.info.type === 'TimeTrace' ? (v) => v * (f.tt.unit === 'min' ? 60 : 1) : xDisplay(f, comparisonFiles(f)).inverse).sort((a, b) => a - b); set({ range_min: r?.[0] ?? null, range_max: r?.[1] ?? null }); } }, 'Use view'),
      h('button', { class: 'btn sm', onclick: () => set({ range_min: null, range_max: null }) }, 'All'),
      h('button', { class: 'btn sm', disabled: !fr.length, title: 'Fit only the first 20 % of the x range', onclick: () => { const [lo, hi] = finiteExtent(fr); set({ range_min: null, range_max: lo + 0.2 * (hi - lo) }); } }, 'Head 20%'),
      h('button', { class: 'btn sm', title: 'Profile default: clear the range so the experiment profile chooses it', onclick: () => set({ range_min: null, range_max: null }) }, 'Profile'))));
  // options
  const opts = h('details', { class: 'card pad', open: !!f.display.fitOptions, ontoggle: (e) => { f.display.fitOptions = e.target.open; } }, h('summary', {}, `Options · multistart ${s.multistart}${fam === 'Rabi' ? ` · SEM weights ${s.sem_weights ? 'on' : 'off'}` : s.sem_weights_all ? ' · SEM weights on' : ''}${s.apply_both ? ' · both detectors' : ''}`),
    h('div', { style: 'display:flex;flex-direction:column;gap:8px;margin-top:8px' },
      h('div', { class: 'g2' }, field('Multistart', input(s.multistart, (v) => set({ multistart: Math.max(4, num(v) || 12) }), { numeric: true })),
        fam === 'Rabi' ? field('Delay / dead time (ns)', input(s.dead_time_ns || '', (v) => set({ dead_time_ns: num(v) || 0 }), { numeric: true, unit: 'ns', placeholder: 'fit' })) : h('span')),
      fam === 'Rabi' ? [check('Weight by iteration SEM', s.sem_weights, (v) => set({ sem_weights: v })),
        check('Lock to π / π/2 times', s.pi_lock, (v) => set({ pi_lock: v })),
        s.pi_lock ? h('div', { class: 'g2' }, field('π (ns)', input(s.pi_ns, (v) => set({ pi_ns: num(v) }), { numeric: true, unit: 'ns' })), field('π/2 (ns)', input(s.pi2_ns, (v) => set({ pi2_ns: num(v) }), { numeric: true, unit: 'ns' }))) : null] : null,
      sci.semAllCheck(f, set),
      f.proc.steps.length ? check('Fit the transformed data', s.fit_transformed, (v) => set({ fit_transformed: v })) : null,
      f.info.detectors.length > 1 ? field('Apply to', seg([[false, 'Active detector'], [true, 'Both detectors']], s.apply_both, (v) => set({ apply_both: v }))) : null));
  out.append(h('div', { class: 'card pad' }, check('Fit smoothed data', s.fit_smoothed, (v) => { if (v) f.layers.smooth = true; set({ fit_smoothed: v }); emit(); }),
    h('div', { class: 'note', title: `Window ${f.data[f.detector]?.proc?.smoothing_window || 1} points (requested ${f.proc.smooth}; change in Process). Default off. Smoothing correlates neighbouring samples, so ordinary fit uncertainties are underestimated; no corrected estimate is supplied, and SEM weighting is disabled for smoothed fits.` },
      s.fit_smoothed ? `${f.data[f.detector]?.proc?.smoothing_window || 1}-pt window · errors underestimated (hover for details)` : `${f.data[f.detector]?.proc?.smoothing_window || 1}-pt window (Process tab) · hover for caveats`)), opts, presetCard(f));
  const series = kindOf(f) === '1d' ? comparisonFiles(f) : [f], all = fitScopeAll(f), busy = f.fitting || f.compareFitting;
  if (series.length > 1) out.append(h('div', { class: 'card pad fit-scope', style: 'display:flex;flex-direction:column;gap:6px' },
    field('Run fit applies to', seg([['all', `All ${series.length} series`], ['primary', 'This file only']], all ? 'all' : 'primary', (v) => { f.display.fitScope = v; emit('insp'); })),
    all ? h('div', { class: 'note', style: 'white-space:pre-line' }, series.map((o) => { const p = fitPlan(f, o); return `${o === f ? '★ ' : ''}${o.info.name.replace(/\.mat$/i, '')}: ${p.model} (${p.source})`; }).join('\n')) : null));
  out.append(h('div', { class: 'row fit-run' },
    h('button', { class: 'btn primary', style: 'flex:1;justify-content:center', disabled: busy, onclick: () => runFitScoped(f) }, busy ? h('span', { class: 'spinner' }) : null, all ? `Run fit · ${series.length} files` : 'Run fit', h('span', { class: 'kbd' }, 'Ctrl F')),
    h('button', { class: 'btn', style: 'flex:1;justify-content:center', disabled: busy, title: 'Fits subsets of the data and keeps the candidate that best describes the full trace', onclick: () => runFitScoped(f, { robust: true }) }, 'Robust fit', h('span', { class: 'kbd' }, 'Ctrl Shift F'))));
  if (kindOf(f) === '1d') out.append(iterationFitActions(f));
  const primaryResults = fit ? [resultCard(f, fit), paramCard(f, fit)].filter(Boolean) : [];
  if (fit) out.append(...primaryResults, sci.bootstrapCard(f, fit, () => {
    const hs = f.fitHistory.find((e) => e.res === fit);
    const spec = clone(hs ? hs.spec : f.fitSpec); spec.robust = false;
    if (spec.model === 'CustomModel') spec.custom_spec = customSpec(spec);
    return spec;
  }));
  if (fam === 'ODMR' && kindOf(f) === '1d') out.append(sci.nvFieldCard(f, () => { const spec = clone(f.fitSpec); spec.robust = false; return spec; }));
  if (f.fitHistory.length > 1) out.append(historyCard(f));
  // Most-used first: model choice, then Run fit and its result; range/smoothing/options/presets/iterations follow.
  const anchor = out.children[splitAt], run = out.querySelector('.fit-run');
  const scope = out.querySelector('.fit-scope');
  if (anchor && run) { if (scope) out.insertBefore(scope, anchor); out.insertBefore(run, anchor); primaryResults.forEach((el) => el instanceof Node && out.insertBefore(el, anchor)); }
  return out;
}

function resultCard(f, fit) {
  const lines = resultLines(fit, f.info).filter((l) => l.key !== 'r2');
  const tiles = lines.slice(0, 6).map((l) => {
    const d = displayValue(l.value, l.err, l.unit, filePresentation(f, f)), parts = d.text.split(' ± ');
    return h('div', { class: 'metric' }, h('div', { class: 'lbl' }, l.label),
      h('div', { class: 'val' }, parts[0], d.unit ? h('small', {}, ' ' + d.unit) : null),
      d.error ? h('div', { class: 'sub' }, `± ${parts[1]} ${d.unit}`) : (l.key === 'pi_time_ns' && fit.metrics.programmed_pi_time_ns ? h('div', { class: 'sub' }, `programmed ${fmt(fit.metrics.programmed_pi_time_ns, 4)} ns`) : l.key === 'pi_over_2_time_ns' && fit.metrics.programmed_pi_over_2_time_ns ? h('div', { class: 'sub' }, `programmed ${fmt(fit.metrics.programmed_pi_over_2_time_ns, 4)} ns`) : null));
  });
  const card = h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:10px' },
    h('div', { class: 'row' }, h('span', { style: 'font-weight:600' }, `Result${f.info.detectors.length > 1 ? ' · ' + detLabel(f, f.detector) : ''}`), h('span', { class: 'sp' }), h('span', { class: `pill ${fit.status}`, title: fit.reason }, fit.status)),
    h('div', { class: 'note' }, `${fit.model} · ${fitTargetLabel(fit)} · ${fit.selection || ''}`),
    fit.provenance?.uncertainty_note ? h('div', { class: 'note' }, fit.provenance.uncertainty_note) : null,
    tiles.length ? h('div', { class: 'g2' }, tiles) : null,
    h('div', { class: 'kv' }, h('span', { class: 'k' }, 'R²'), h('span', { class: 'v' }, fmt(fit.r2, 5)), h('span', { class: 'k' }, 'RMSE'), h('span', { class: 'v' }, fmt(fit.rmse, 4)),
      h('span', { class: 'k' }, 'Durbin–Watson'), h('span', { class: 'v' }, fmt(fit.dw, 3)), h('span', { class: 'k' }, 'AIC / BIC'), h('span', { class: 'v' }, `${fmt(fit.aic, 5)} / ${fmt(fit.bic, 5)}`),
      h('span', { class: 'k' }, 'Bound hits'), h('span', { class: 'v' }, String(fit.bound_hits)), sci.chi2Rows(fit)),
    sci.chi2Note(fit),
    fit.status !== 'PASS' ? h('div', { class: 'note', style: 'color:var(--warn)' }, fit.reason) : null,
    identifiabilityCard(fit),
    peakReviewCard(fit, f),
    rabiCalibrationCard(fit),
    modelSelectionCard(fit),
    h('details', {}, h('summary', {}, 'All derived values'), h('div', { class: 'kv', style: 'margin-top:6px' }, Object.entries(fit.metrics || {}).flatMap(([k, v]) => [h('span', { class: 'k' }, k), h('span', { class: 'v' }, typeof v === 'number' ? fmt(v, 6) : String(v))])),
      h('pre', { class: 'note', style: 'white-space:pre-wrap;font-family:var(--mono);font-size:11px' }, fit.message)),
    h('div', { class: 'row' }, h('button', { class: 'btn', style: 'flex:1;justify-content:center', onclick: () => copyText(slidesTable(f, fit), 'Results copied as a table for slides') }, 'Copy for slides'),
      h('button', { class: 'btn', style: 'flex:1;justify-content:center', onclick: () => copyText(notebookText(f, fit), 'Results copied for your notebook') }, 'Copy to notebook')));
  return card;
}

function slidesTable(f, fit) {
  const rows = [['Quantity', 'Value', 'Unit'], ['Fit target', fitTargetLabel(fit), '']];
  if (fit.provenance?.uncertainty_note) rows.push(['Uncertainty warning', fit.provenance.uncertainty_note, '']);
  resultLines(fit, f.info).forEach((l) => { const d = displayValue(l.value, l.err, l.unit, filePresentation(f, f)); rows.push([l.label, d.text, d.unit]); });
  return rows.map((r) => r.join('\t')).join('\n');
}
function notebookText(f, fit) {
  const lines = [`${f.info.name}${f.info.detectors.length > 1 ? ` (${detLabel(f, f.detector)})` : ''}`, `${fit.model}, ${MODE_LABEL[f.mode]}, R² = ${fmt(fit.r2, 4)} (${fit.status})`, fitTargetLabel(fit), fit.provenance?.uncertainty_note || ''];
  resultLines(fit, f.info).filter((l) => l.key !== 'r2').forEach((l) => lines.push(`  ${l.label} = ${displayValue(l.value, l.err, l.unit, filePresentation(f, f)).withUnit}`));
  if (f.fitSpec.range_min !== null || f.fitSpec.range_max !== null) lines.push(`  fit range ${f.fitSpec.range_min ?? 'start'} to ${f.fitSpec.range_max ?? 'end'} ${f.info.type === 'TimeTrace' ? 's' : f.info.x_unit || 'axis units'} (native)`);
  lines.push(`  ${f.path}`);
  return lines.join('\n');
}

// AN11: ODMR centre ranges as removable chips; range n constrains fitted peak n (in order). Native x units.
function odmrRangeChips(f, s, set) {
  const ranges = (s.odmr_ranges || []).filter((r) => r && r[0] != null && r[1] != null);
  const unit = f.info.x_unit || '';
  const draft = (f.odmrRangeDraft ||= [null, null]);
  const add = () => {
    const [lo, hi] = draft;
    if (lo == null || hi == null) { toast('Enter both ends of the range first.'); return; }
    f.odmrRangeDraft = [null, null];
    set({ odmr_ranges: [...ranges, [Math.min(lo, hi), Math.max(lo, hi)]] });
  };
  return h('div', { class: 'f' }, h('span', { class: 'lbl' }, `Centre ranges (native ${unit || 'x units'}) · range n constrains peak n`),
    h('div', { class: 'chips-s', style: 'margin:2px 0 6px' }, ranges.length ? ranges.map((r, i) => h('span', { class: 'chip-s ok', title: `Peak ${i + 1} centre must lie in this range` },
      `${i + 1}: ${fmt(r[0], 6)}–${fmt(r[1], 6)} `, h('button', { class: 'ibtn', style: 'height:14px;width:14px;font-size:10px', 'aria-label': `Remove range ${i + 1}`, onclick: () => set({ odmr_ranges: ranges.filter((_, j) => j !== i) }) }, '×')))
      : h('span', { class: 'dim', style: 'font-size:11px' }, 'Any centre')),
    h('div', { class: 'row', style: 'gap:4px' },
      input(draft[0] ?? '', (v) => { draft[0] = num(v); }, { numeric: true, unit, placeholder: 'from', style: 'width:84px', 'aria-label': 'Range start' }),
      h('span', { class: 'dim' }, 'to'),
      input(draft[1] ?? '', (v) => { draft[1] = num(v); }, { numeric: true, unit, placeholder: 'to', style: 'width:84px', 'aria-label': 'Range end' }),
      h('button', { class: 'btn sm', onclick: add }, 'Add range')));
}

// AN11: per-peak values and local fit quality for ODMR fits.
function peakReviewCard(fit, f) {
  const rows = fit.peak_review;
  if (!rows?.length) return null;
  const unit = f.info.x_unit || '';
  const v = (x, e) => displayValue(x, e, unit, filePresentation(f, f));
  return h('details', { open: rows.some((r) => r.flags.length) }, h('summary', {}, `Peaks · ${rows.length}${rows.some((r) => r.flags.length) ? ' · check flagged' : ''}`),
    h('table', { class: 'tbl', style: 'margin-top:6px' }, h('tr', {}, ['#', 'Centre', 'FWHM', 'Amplitude', 'Local/overall RMS'].map((t) => h('th', {}, t))),
      rows.map((r, i) => h('tr', { title: r.flags.join('\n') || `${r.points} points within ±2 FWHM` }, h('td', {}, String(i + 1)), h('td', {}, v(r.center, r.center_err).withUnit),
        h('td', {}, v(r.fwhm, r.fwhm_err).withUnit), h('td', {}, fmt(r.amplitude, 3)), h('td', { style: r.flags.length ? 'color:var(--warn)' : '' }, r.rms_ratio == null ? '–' : `${fmt(r.rms_ratio, 2)}×${r.flags.length ? ' ⚠' : ''}`)))),
    ...rows.flatMap((r, i) => r.flags.map((t) => h('div', { class: 'note', style: 'color:var(--warn)' }, `Peak ${i + 1}: ${t}`))),
    h('div', { class: 'note' }, 'FWHM = 2 × the Lorentzian half width w. Local RMS uses residuals within ±2 FWHM of each centre.'));
}

// AN11: Rabi π / π/2 timing with definitions and the context needed to reuse the calibration later.
function rabiCalibrationCard(fit) {
  const c = fit.rabi_calibration;
  if (!c || !Object.keys(c.values || {}).length) return null;
  const ctx = c.context || {};
  const text = () => [
    `Rabi calibration · ${ctx.source} · ${ctx.detector || ''} · ${ctx.observable}`,
    ...Object.entries(c.values).map(([k, q]) => `${k} = ${fmt(q.value, 6)} ns${q.error ? ` ± ${fmt(q.error, 3)} ns (${q.error_kind})` : ''}  [${q.definition}]`),
    `Model ${ctx.model}; ${ctx.selection || ''}`, `Quality: ${c.quality.status}, R² ${fmt(c.quality.r2, 4)}, converged ${c.quality.converged}, bound hits ${c.quality.bound_hits}`,
    `Acquired ${ctx.acquired || 'unknown'} (${ctx.acquired_source}); fitted ${ctx.fitted_at}; scan ${fmt(ctx.scan_ns[0], 5)}–${fmt(ctx.scan_ns[1], 5)} ns, ${ctx.points} points, ${ctx.iterations ?? '?'} iterations`,
    ctx.rf_frequency_ghz ? `RF ${ctx.rf_frequency_ghz} GHz` : '', ctx.source_sha256 ? `Source SHA-256 ${ctx.source_sha256}` : '', ctx.path || ''].filter(Boolean).join('\n');
  return h('details', {}, h('summary', {}, 'Rabi calibration'),
    h('table', { class: 'tbl', style: 'margin-top:6px' }, Object.entries(c.values).map(([k, q]) => h('tr', { title: q.definition },
      h('td', { class: 'txt' }, k.replace(/_ns$/, '').replace(/_/g, ' ')), h('td', {}, `${fmt(q.value, 5)} ns`), h('td', { class: 'dim' }, q.error ? `± ${fmt(q.error, 2)}${q.error_kind === 'frequency only' ? ' (f only)' : ''}` : '')))),
    h('div', { class: 'note' }, `Acquired ${ctx.acquired || 'unknown'} (${ctx.acquired_source}) · ${ctx.model} · ${c.quality.status}. Hover a row for its definition; "f only" errors omit the pulse-area terms.`),
    h('button', { class: 'btn sm', onclick: () => copyText(text(), 'Rabi calibration copied') }, 'Copy calibration'));
}

// AN08: warnings about weakly determined parameters, always visible; the correlation matrix behind a disclosure.
function identifiabilityCard(fit) {
  const id = fit.identifiability;
  if (!id || (!id.warnings?.length && !id.correlation)) return null;
  const c = id.correlation;
  const cell = (r) => h('td', { style: `color:${Math.abs(r) >= 0.95 ? 'var(--warn)' : Math.abs(r) >= 0.8 ? 'var(--text)' : 'var(--text3)'}` }, fmt(r, 2));
  return h('div', {},
    ...(id.warnings || []).map((w) => h('div', { class: 'note', style: 'color:var(--warn)' }, `⚠ ${w}`)),
    c ? h('details', {}, h('summary', {}, 'Parameter correlations'),
      h('table', { class: 'tbl', style: 'margin-top:6px' }, h('tr', {}, h('th', {}, ''), c.names.map((n) => h('th', {}, n))),
        c.names.map((n, i) => h('tr', {}, h('td', { class: 'txt' }, n), c.matrix[i].map((r, j) => (j === i ? h('td', { class: 'dim' }, '1') : cell(r)))))),
      h('div', { class: 'note' }, 'From the fit covariance (a linear approximation, like the ± errors). |r| ≥ 0.95 means the data pin down a combination of the two parameters better than either alone.')) : null);
}

// AN03: every model Auto compared, with the scores its selection rules use.
function modelSelectionCard(fit) {
  const rows = fit.candidates;
  if (!rows?.length) return null;
  const fitted = rows.filter((r) => !r.skipped && Number.isFinite(r.bic));
  const best = fitted.length ? Math.min(...fitted.map((r) => r.bic)) : NaN;
  return h('details', { open: false }, h('summary', {}, `Model selection · ${rows.length} candidates`),
    h('div', { class: 'note', style: 'margin:6px 0' }, `Rule: ${fit.selection || 'see model notes'}`),
    h('table', { class: 'tbl' }, h('tr', {}, ['', 'Model', 'R²', 'ΔBIC', 'Params', 'Bound hits', 'Converged'].map((t) => h('th', {}, t))),
      rows.map((r) => r.skipped
        ? h('tr', { class: 'dim' }, h('td', {}, ''), h('td', { class: 'txt' }, r.role || r.model), h('td', { colspan: 5, class: 'txt' }, `skipped: ${r.skipped}`))
        : h('tr', {}, h('td', {}, r.chosen ? '✓' : ''), h('td', { class: 'txt', title: r.model }, r.role ? `${r.role} (${r.model})` : r.model),
          h('td', {}, fmt(r.r2, 4)), h('td', {}, Number.isFinite(r.bic) && Number.isFinite(best) ? fmt(r.bic - best, 3) : '–'),
          h('td', {}, r.free_params ?? '–'), h('td', {}, String(r.bound_hits ?? '–')), h('td', {}, r.converged === false ? 'no' : 'yes')))),
    h('div', { class: 'note' }, 'ΔBIC compares models on the same data (lower is better; above ~10 is strong evidence against). Auto can prefer a lower-R² model for a physical or complexity reason stated in the rule.'));
}

function paramCard(f, fit) {
  const s = f.fitSpec;
  f.pv = f.pv || 'res';
  const card = h('div', { class: 'card pad' });
  card.append(h('div', { class: 'row', style: 'margin-bottom:4px' }, h('span', { style: 'font-weight:600' }, 'Parameters'), h('span', { class: 'sp' }),
    seg([['res', 'Result'], ['seed', 'Seeds & bounds']], f.pv, (v) => { f.pv = v; emit('insp'); })));
  if (f.pv === 'res') {
    card.append(h('table', { class: 'tbl' }, fit.params.map((p) => {
      const d = displayValue(p.value, p.error, parameterUnit(p, f.info), filePresentation(f, f));
      return h('tr', { title: `${p.desc}\nNative: ${p.value} ${parameterUnit(p, f.info)}; seeds/bounds use native units.` },
        h('td', { class: 'txt', style: 'color:var(--text2)' }, `${p.name}${d.unit && d.unit !== '-' ? ` (${d.unit})` : ''}`),
        h('td', {}, d.text), h('td', { style: 'color:var(--text3)' }, p.locked ? 'locked' : p.error == null ? 'n/a' : ''),
        h('td', { style: 'width:14px;color:var(--warn)', title: p.bound_hit ? 'At a bound' : '' }, p.bound_hit ? '⚑' : ''));
    })));
    if (isNormalizedT1(fit)) card.append(h('div', { class: 'note', style: 'margin-top:6px' }, 'A and y0 remain in normalized 1 − data space. Time/frequency values above are display-scaled only; fitting, seeds and bounds remain in the labelled native units.'));
    if (fit.params.some((p) => !p.locked && p.error == null)) card.append(h('div', { class: 'note', style: 'margin-top:6px' }, 'n/a: the covariance gives no usable standard error for this parameter (degenerate or at a bound).'));
  } else {
    s.params = s.params || {};
    card.append(h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, 'Param'), h('th', {}, 'Value'), h('th', {}, 'Min'), h('th', {}, 'Max'), h('th', {}, 'Lock')),
      fit.params.map((p) => {
        const e = s.params[p.name] || {};
        const upd = (patch) => { commit('parameter bounds'); s.params_model = fit.model; s.params[p.name] = { ...e, ...patch }; emit('insp'); };
        return h('tr', {}, h('td', { class: 'txt' }, `${p.name}${parameterUnit(p, f.info) ? ` (${parameterUnit(p, f.info)})` : ''}`), h('td', {}, input(e.value ?? '', (v) => upd({ value: num(v) }), { numeric: true, unit: parameterUnit(p, f.info), placeholder: fmt(p.value, 4) })),
          h('td', {}, input(e.min ?? '', (v) => upd({ min: num(v) }), { numeric: true, unit: parameterUnit(p, f.info), placeholder: '–' })), h('td', {}, input(e.max ?? '', (v) => upd({ max: num(v) }), { numeric: true, unit: parameterUnit(p, f.info), placeholder: '–' })),
          h('td', {}, h('input', { type: 'checkbox', checked: !!e.lock, 'aria-label': `Lock ${p.name}`, onchange: (ev) => upd({ lock: ev.target.checked, value: e.value ?? p.value }) })));
      })),
    h('div', { class: 'row', style: 'margin-top:8px' }, h('button', { class: 'btn sm', onclick: () => { commit('seeds'); s.params_model = fit.model; fit.params.forEach((p) => { s.params[p.name] = { ...(s.params[p.name] || {}), value: p.value }; }); emit('insp'); } }, 'Values from current fit'),
      h('button', { class: 'btn sm', onclick: () => { commit('clear bounds'); s.params = {}; delete s.params_model; emit('insp'); } }, 'Clear'), h('button', { class: 'btn sm', onclick: () => savePreset(f) }, 'Save as preset')));
  }
  return card;
}

async function savePreset(f) {
  const name = prompt('Preset name', `${f.info.type} preset`);
  if (!name) return;
  const presets = { ...(S.settings.fit_presets || {}), [name]: { ...clone(f.fitSpec), type: f.info.type } };
  S.settings = await call('settings_set', { patch: { fit_presets: presets } });
  toast(`Saved preset “${name}”.`, 'ok'); emit('insp');
}

function presetCard(f) {
  const presets = S.settings.fit_presets || {};
  const names = Object.keys(presets).sort((a, b) => (presets[b].type === f.info.type) - (presets[a].type === f.info.type) || a.localeCompare(b));
  const keep = f.keepRange !== false;
  const apply = (spec) => { commit('apply preset'); const { type, ...rest } = clone(spec); const r = keep ? { range_min: f.fitSpec.range_min, range_max: f.fitSpec.range_max } : {}; f.fitSpec = { ...defaultFitSpec(f.info), ...rest, ...r }; emit(); };
  const del = async (name) => { const next = { ...presets }; delete next[name]; S.settings = await call('settings_set', { patch: { fit_presets: next } }); emit('insp'); toast(`Deleted preset “${name}”.`); };
  const picker = h('input', { type: 'file', accept: '.json,application/json', style: 'display:none' });
  picker.addEventListener('change', () => {
    const file = picker.files[0]; if (!file) return;
    const rd = new FileReader();
    rd.onload = () => {
      try {
        const j = JSON.parse(String(rd.result));
        const pn = j.param_names || [];
        if (!j.expression) throw new Error('no "expression" field');
        const lim = (v) => (v === undefined || v === null || Math.abs(v) >= 1e299 ? null : v);
        commit('import custom model');
        Object.assign(f.fitSpec, { model: 'CustomModel', custom_expr: j.expression,
          custom_params: pn.map((n, i) => ({ name: n, seed: j.seed?.[i] ?? 1, min: lim(j.lower?.[i]), max: lim(j.upper?.[i]) })) });
        toast(`Imported custom model${j.name ? ` “${j.name}”` : ''}.`, 'ok'); emit();
      } catch (e) { toast(`Could not import model JSON: ${e.message}`, 'err'); }
    };
    rd.readAsText(file); picker.value = '';
  });
  return h('details', { class: 'card pad' }, h('summary', {}, `Presets (${names.length})`),
    h('div', { style: 'display:flex;flex-direction:column;gap:6px;margin-top:8px' },
      check('Keep current fit range when applying', keep, (v) => { f.keepRange = v; emit('insp'); }),
      names.length ? names.map((n) => h('div', { class: 'row' }, h('span', { class: 'ellip', style: 'flex:1', title: presets[n].type || '' }, n, presets[n].type ? h('span', { class: 'dim' }, ` · ${presets[n].type}`) : null),
        h('button', { class: 'btn sm', onclick: () => apply(presets[n]) }, 'Apply'), h('button', { class: 'btn sm', onclick: () => del(n) }, 'Delete')))
        : h('div', { class: 'note' }, 'None yet. Use Seeds & bounds › Save as preset.'),
      h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => picker.click() }, 'Import custom model JSON…'), picker)));
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
  if (kindOf(f) === '1d') out.append(iterationCard(f), iterationFitActions(f),
    h('button', { class: 'btn block', onclick: () => openCombineDialog(f) }, 'Combine runs…'));
  out.append(h('div', { class: 'g2' },
    field('Bin size', input(f.proc.bin, (v) => setProc(f, { bin: Math.max(1, Math.round(num(v) || 1)) }), { numeric: true })),
    field(kindOf(f) === 'tt' ? 'Smoothing (points)' : 'Smoothing window', input(f.proc.smooth, (v) => { const smooth = Math.max(1, Math.round(num(v) || 1)); f.layers.smooth = smooth > 1; setProc(f, { smooth }); }, { numeric: true }))));
  out.append(h('div', { class: 'g2' }, field('Data from', input(f.proc.roi_min ?? '', (v) => setProc(f, { roi_min: num(v) }), { numeric: true, placeholder: 'start' })), field('to', input(f.proc.roi_max ?? '', (v) => setProc(f, { roi_max: num(v) }), { numeric: true, placeholder: 'end' }))));
  const nEx = f.proc.excluded.length + f.proc.exclusion_ranges.length;
  out.append(h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Excluded points'),
    nEx ? h('div', { class: 'row wrap' }, h('span', { class: 'mono' }, `${f.proc.excluded.length} point(s)`), h('button', { class: 'btn sm', onclick: () => setProc(f, { excluded: [], exclusion_ranges: [] }, 'restore points') }, 'Restore all'))
      : h('div', { class: 'note' }, 'None. Click a point and press Exclude, or use the box-select tool (pencil) to exclude several. Excluded points are dropped from fits but never from the file.')));
  const exr = f.proc.exclusion_ranges || [];
  const exIn = [h('input', { class: 'field', placeholder: 'x from', style: 'width:80px' }), h('input', { class: 'field', placeholder: 'to', style: 'width:80px' })];
  out.append(h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Exclude x range'),
    h('div', { class: 'row' }, exIn, h('button', { class: 'btn sm', onclick: () => { const a = num(exIn[0].value), b = num(exIn[1].value); if (a === null || b === null) { toast('Enter both ends of the range.'); return; } setProc(f, { exclusion_ranges: [...exr, [Math.min(a, b), Math.max(a, b)]] }, 'exclude range'); } }, 'Add')),
    exr.map((r, i) => h('div', { class: 'row' }, h('span', { class: 'mono', style: 'flex:1' }, `${fmt(r[0], 6)} to ${fmt(r[1], 6)}`),
      h('button', { class: 'ibtn', title: 'Remove range', onclick: () => setProc(f, { exclusion_ranges: exr.filter((_, j) => j !== i) }, 'remove exclusion range') }, icon('close', 14))))));
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
    if (st.kind === 'resample') param = input(st.params?.count || '', (v) => upd({ params: { count: num(v) } }), { numeric: true, placeholder: 'points', style: 'width:80px;height:26px' });
    if (st.kind === 'expression') param = input(st.params?.expression || 'y', (v) => upd({ params: { expression: v } }), { style: 'height:26px', placeholder: 'e.g. y / max(y)' });
    return h('div', { class: 'card', style: 'padding:6px 8px;display:flex;align-items:center;gap:8px' },
      h('input', { type: 'checkbox', checked: st.enabled !== false, 'aria-label': 'Enable step', onchange: (e) => upd({ enabled: e.target.checked }) }),
      h('span', { style: 'min-width:80px' }, KINDS.find(([k]) => k === st.kind)?.[1]), param, h('span', { class: 'sp' }),
      h('button', { class: 'ibtn', title: 'Remove step', onclick: () => { const steps = clone(f.proc.steps); steps.splice(i, 1); setProc(f, { steps }, 'remove transform'); } }, icon('close', 14)));
  });
  out.append(h('div', { class: 'f' }, h('div', { class: 'row' }, h('span', { class: 'lbl' }, 'Transform steps · in order'), h('span', { class: 'sp' }),
    select([['', 'Add step…'], ...KINDS], '', (v) => { if (v) setProc(f, { steps: [...clone(f.proc.steps), { kind: v, params: {}, enabled: true }] }, 'add transform'); }, { style: 'width:120px;height:26px' })),
  stepRows.length ? stepRows : h('div', { class: 'note' }, 'Steps change only the analysed copy. Fits use the untransformed data unless "Fit the transformed data" is ticked in Fit › Options.')));
  if (kindOf(f) === '1d') out.append(sci.deerCard(f));
  if (kindOf(f) === '1d') out.append(overlaysCard(f));
  return out;
}

// ---- Plot tab (quick settings; full control in the plot editor)
function plotTab(f) {
  const st = f.style;
  const set = (patch) => { commit('plot style'); Object.assign(st, patch); emit(); };
  const s = baseStyle(st);
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    kindOf(f) === '1d' ? overlaysCard(f) : null,
    isNormalizedT1(f.fit[f.detector]) ? field('T1 display units', t1DisplayControl(f)) : null,
    field('Title', input(st.title || '', (v) => set({ title: v }), { placeholder: 'None (double-click the plot to edit)' })),
    h('div', { class: 'g2' }, field('X label', input(st.x?.title ?? xTitle(f), (v) => set({ x: { ...(st.x || {}), title: v } }))), field('Y label', input(st.y?.title ?? (kindOf(f) === 'tt' ? '' : yTitle(f)), (v) => set({ y: { ...(st.y || {}), title: v } })))),
    h('div', { class: 'g2' }, field('X range', input(st.x?.min != null ? `${st.x.min}, ${st.x.max}` : 'auto', (v) => { const p = v.split(/[,;\s]+/).map(num); set({ x: { ...(st.x || {}), min: p[0] ?? null, max: p[1] ?? null } }); })),
      field('Y range', input(st.y?.min != null ? `${st.y.min}, ${st.y.max}` : 'auto', (v) => { const p = v.split(/[,;\s]+/).map(num); set({ y: { ...(st.y || {}), min: p[0] ?? null, max: p[1] ?? null } }); }))),
    field('Style preset', seg([['presentation', 'Presentation'], ['paper', 'Paper'], ['dense', 'Dense']], s.preset, (v) => import('./ploteditor.js').then((m) => m.applyPreset(f, v)))),
    styleToolsBox(f),
    field('Legend', seg([['ul', 'Upper left'], ['ur', 'Upper right'], ['lr', 'Lower right'], ['outside', 'Outside'], ['hidden', 'Hidden']], f.layers.legend ? s.legend.pos : 'hidden',
      (v) => { commit('legend'); if (v === 'hidden') f.layers.legend = false; else { f.layers.legend = true; st.legend = { ...(st.legend || {}), pos: v, x: null, y: null }; } emit(); })),
    kindOf(f) === '1d' ? field('Results box', seg([['off', 'Off'], ['compact', 'Compact'], ['pm', 'With errors']], !f.layers.results ? 'off' : (s.results.format === 'pm' ? 'pm' : 'compact'),
      (v) => { commit('results box'); f.layers.results = v !== 'off'; st.results = { ...(st.results || {}), format: v === 'pm' ? 'pm' : 'sig4' }; emit(); })) : null,
    kindOf(f) === '1d' ? displayControls(f) : null,
    kindOf(f) === '1d' ? field('Uncertainty', seg([['band', 'Band'], ['bars', 'Error bars']], f.errorMode, (v) => { f.errorMode = v; f.layers.band = true; emit(); })) : null,
    h('div', { class: 'g3' }, field('Width (in)', input(s.width_in, (v) => set({ width_in: num(v) || 8.5 }), { numeric: true })), field('Height (in)', input(s.height_in, (v) => set({ height_in: num(v) || 5.5 }), { numeric: true })), field('DPI', input(s.dpi, (v) => set({ dpi: num(v) || 300 }), { numeric: true }))),
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
      k === '1d' ? [c('csv', 'Fit parameters CSV'), c('origin', 'Origin bundle CSV (x, y, smooth, fit, residual)'), c('report', 'Report PNG (figure + all metrics)'), c('report_pdf', 'Report PDF (figure + all metrics)')] : null,
      k === '1d' && f.fit[f.detector] ? reportPicker(f) : null,
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
  if(k==='2d' && f.map.layout.includes('overlay') && !displayedOverlay(f)) {
    await ensureOverlay(f);
    if(!displayedOverlay(f))throw new Error('Map overlay is unavailable; rebuild it before exporting.');
  }
  f.exp = f.exp || { png: true, json: true, csv: k === '1d', time_trace: k === 'tt', spots_csv: k === '2d' };
  const e = f.exp;
  const st = baseStyle(f.style);
  const fig = currentFigure(f);
  const images = {};
  const size = { width_in: st.width_in, height_in: st.height_in, dpi: st.dpi };
  if (e.png || e.pdf) { const png = await renderImage(fig, size); if (e.png) images.png = png; if (e.pdf) images.pdf_png = png; }
  if (e.svg || e.pdf) { const svg = await renderImage(fig, { ...size, format: 'svg' }); if (e.svg) images.svg = svg; if (e.pdf) images.pdf_svg = svg; }
  const fit = currentResult(f);
  let reportImgs = null;
  if ((e.report || e.report_pdf) && fit) {
    const rs = { width_in: st.width_in + 3.2, height_in: st.height_in, dpi: st.dpi }, rf = reportFigure(f, fig, fit);
    const rpng = await renderImage(rf, rs);
    if (e.report) images.report_png = rpng;
    if (e.report_pdf) reportImgs = { pdf_png: rpng, pdf_svg: await renderImage(rf, { ...rs, format: 'svg' }) };
  }
  const dir = f.expDir || `${dirname(f.path)}\\SmartFitter results`;
  const det = f.info.detectors.length > 1 ? `__${f.detector}` : '';
  const stemName = (f.expStem || stem(f.path)) + det;
  const spotsRows = k === '2d' ? f.info.detectors.flatMap((d) => visibleSpots(f, d).map((s) => ({ ...s, detector_label: detLabel(f, d) }))) : null;
  const M = f.map;
  const res = await call('export', { req: {
    output_dir: dir, stem: stemName, images, dpi: st.dpi, source: f.path, detector: f.detector, mode: f.mode, processing: f.proc, fit_spec: f.fitSpec,
    fit: fit ? { ...fit, raw: undefined, dense: undefined } : null, json: e.json, csv: e.csv && !!fit, origin: e.origin, detectors_raw: e.detectors_raw, time_trace: e.time_trace,
    provenance: f.data[f.detector]?.provenance, iteration_fits: f.iterationFits, iteration_sensitivity: f.iterationSensitivity, combined_recipe: f.combinedRecipe,
    spots: spotsRows, spots_csv: e.spots_csv && spotsRows?.length,
    extra_csv: k === '1d' && e.csv ? { '_comparison.csv': comparisonCSV(f), ...(f.iterationFits?.[f.detector] ? { '_iterations.csv': iterationCSV(f) } : {}) } : undefined,
    analysis:envelopeFor(f), historical_results:Object.values(S.results).filter(r=>r.analysis.descriptor.source===sourcePath(f.path)),
    map_comparison: k==='2d' ? displayedOverlay(f) : null,
    plot_state: { map:k==='2d' ? clone(f.map) : undefined, style: f.style, layers: f.layers, sample: f.sample, display: f.display, xrange: f.xrange, yrange: f.yrange,
      comparison: k === '1d' ? comparisonRows(f) : undefined,
      series: k === '1d' ? comparisonFiles(f).map((o) => ({ path: o.path, mode: o.mode, detector: o.detector, processing: o.proc, fit_spec: o.fitSpec, fit: Object.fromEntries(Object.entries(o.fit).map(([det, r]) => [det, { ...r, raw: undefined }])) })) : undefined },
    map_comparison_csv: !!e.map_csv && !!displayedOverlay(f),
  } }, { label: `Saving ${f.info.name}`, retry: () => saveResults(f) });
  const reportRes = reportImgs ? await call('export', { req: { output_dir: dir, stem: `${stemName}_report`, images: reportImgs, dpi: st.dpi, json: false } }, { label: 'Saving report PDF' }) : null;
  [...(res.warnings || []), ...(reportRes?.warnings || [])].forEach((w) => toast(w, 'err', 9000));
  if (e.session) { const { saveSession } = await import('./app.js'); await saveSession(`${dir}\\${stemName}.sfsession.json`); }
  toast(`Saved ${res.written.length} file(s) to ${res.folder}`, 'ok');
  return res;
}

function reportKeyList(f, fit) {
  const base = resultLines(fit, f.info).map((l) => [l.key, l.label]);
  return [['model', 'Model + status'], ...base, ['rmse', 'RMSE / Durbin–Watson'], ['aicbic', 'AIC / BIC'], ['params', 'Parameters table']];
}
function reportPicker(f) {
  const fit = f.fit[f.detector];
  const all = reportKeyList(f, fit);
  const cur = new Set(f.reportKeys || all.map(([k]) => k));
  return h('details', { class: 'card pad' }, h('summary', {}, `Report contents (${all.filter(([k]) => cur.has(k)).length}/${all.length})`),
    h('div', { style: 'display:flex;flex-direction:column;gap:4px;margin-top:6px' },
      all.map(([k, lab]) => check(lab, cur.has(k), (v) => { v ? cur.add(k) : cur.delete(k); f.reportKeys = all.map(([q]) => q).filter((q) => cur.has(q)); emit('insp'); }))));
}

function reportFigure(f, fig, fit) {
  const has = (k) => !f.reportKeys || f.reportKeys.includes(k);
  const lines = resultLines(fit, f.info).filter((l) => has(l.key)).map((l) => `${l.label} = ${displayValue(l.value, l.err, l.unit, filePresentation(f, f)).withUnit}`);
  const params = fit.params.map((p) => `${p.name} = ${displayValue(p.value, p.error, parameterUnit(p, f.info), filePresentation(f, f)).withUnit}`);
  const sam = f.sample || {};
  const sampleLine = [sam.id, sam.condition, sam.power, sam.date].filter(Boolean).join(' · ');
  const txt = [`<b>${f.info.name}</b>`, sampleLine || null, fitTargetLabel(fit) || null, fit.provenance?.uncertainty_note ? 'Smoothed: ordinary errors underestimated; no correction.' : null, has('model') ? `${fit.model} · ${MODE_LABEL[f.mode]}` : null, has('model') ? `Status ${fit.status}` : null, '', ...lines,
    has('params') ? '<br><b>Parameters</b>' : null, ...(has('params') ? params : []),
    has('rmse') || has('aicbic') ? '' : null, has('rmse') ? `RMSE ${fmt(fit.rmse, 4)} · DW ${fmt(fit.dw, 3)}` : null, has('aicbic') ? `AIC ${fmt(fit.aic, 5)} · BIC ${fmt(fit.bic, 5)}` : null].filter((t) => t !== null).join('<br>');
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
  f.sample = f.sample || {};
  const sm = (key, label, props = {}) => field(label, input(f.sample[key] ?? '', (v) => { commit('sample'); f.sample[key] = v; emit('insp'); }, props));
  const notes = h('textarea', { class: 'field', rows: 3 }, f.sample.notes || '');
  notes.addEventListener('change', () => { f.sample.notes = notes.value; });
  notes.addEventListener('keydown', (e) => e.stopPropagation());
  return h('div', { style: 'display:flex;flex-direction:column;gap:12px' },
    h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Sample (saved with the session and in results JSON)'),
      h('div', { class: 'g2' }, sm('id', 'Sample ID'), sm('condition', 'Condition')),
      h('div', { class: 'g2' }, sm('power', 'Power'), sm('date', 'Date', { placeholder: 'YYYY-MM-DD' })),
      field('Notes', notes)),
    h('div', { class: 'kv' }, rows.flatMap(([k, v]) => [h('span', { class: 'k' }, k), h('span', { class: 'v', style: 'font-size:11.5px' }, v)])),
    i.detectors.length > 1 ? h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Detector labels (this session, saved in exports)'),
      i.detectors.map((d) => input(detLabel(f, d), (v) => { f.labels[d] = v; emit(); }))) : null,
    Object.keys(i.detector_errors || {}).length ? h('div', { class: 'note', style: 'color:var(--bad)' }, Object.entries(i.detector_errors).map(([k, v]) => `${k}: ${v}`).join('\n')) : null,
    matParamsCard(f),
    h('details', { open: true }, h('summary', {}, `File metadata (${Object.keys(meta).length})`),
      h('div', { class: 'kv', style: 'margin-top:6px' }, Object.entries(meta).flatMap(([k, v]) => [h('span', { class: 'k' }, k), h('span', { class: 'v', style: 'font-size:11px' }, String(v))]))),
    h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => call('reveal', { path: f.path }) }, 'Show in folder'), h('button', { class: 'btn sm', onclick: () => copyText(f.path, 'Path copied') }, 'Copy path')));
}

// Acquisition settings exactly as saved in the .mat (params / scanInfo / metadata); read on first open, no interpretation.
function matParamsCard(f) {
  const st = (f.matParams ||= { open: false, filter: '' });
  const load = async () => {
    st.loading = true; emit('insp');
    try { st.data = await call('mat_params', { path: f.path }, { label: 'Reading .mat parameters' }); st.error = null; }
    catch (e) { st.error = e.message; } finally { st.loading = false; emit('insp'); }
  };
  if (st.open && !st.data && !st.loading && !st.error) load();
  // Live filter: hides non-matching rows in place so typing never loses focus to a re-render.
  const applyFilter = (root) => {
    const q = st.filter.trim().toLowerCase();
    root.querySelectorAll('[data-param]').forEach((el) => { el.style.display = !q || el.dataset.param.includes(q) ? '' : 'none'; });
    root.querySelectorAll('details[data-section]').forEach((d) => { if (q) d.open = !!d.querySelector('[data-param]:not([style*="none"])'); });
  };
  const filter = h('input', { class: 'field', value: st.filter, placeholder: 'Filter (e.g. piTime, RF, duration)', 'aria-label': 'Filter .mat parameters',
    oninput: (e) => { st.filter = e.target.value; applyFilter(e.target.closest('.mat-params')); } });
  filter.addEventListener('keydown', (e) => e.stopPropagation());
  const card = h('details', { open: st.open, ontoggle: (e) => { if (e.target.open !== st.open) { st.open = e.target.open; emit('insp'); } } },
    h('summary', {}, 'Run parameters (saved in the .mat)'),
    !st.open ? null : st.loading ? h('div', { class: 'note' }, 'Reading…')
      : st.error ? h('div', { class: 'note', style: 'color:var(--bad)' }, st.error, ' ', h('button', { class: 'btn sm', onclick: () => { st.error = null; load(); } }, 'Retry'))
      : st.data ? h('div', { class: 'mat-params', style: 'display:flex;flex-direction:column;gap:6px;margin-top:6px' },
        h('div', { class: 'row' }, filter, h('button', { class: 'btn sm', title: 'Copy all parameters as tab-separated text',
          onclick: () => copyText(st.data.sections.flatMap((s) => s.rows.map(([k, v]) => `${k}\t${v}`)).join('\n'), 'Parameters copied') }, 'Copy')),
        st.data.sections.map((s) => h('details', { open: s.name !== 'Data', 'data-section': s.name }, h('summary', {}, `${s.name} (${s.rows.length})`),
          h('div', { class: 'kv', style: 'margin-top:4px' }, s.rows.flatMap(([k, v]) => {
            const key = `${k} ${v}`.toLowerCase();
            return [h('span', { class: 'k', 'data-param': key, style: 'overflow-wrap:anywhere' }, k.slice(s.name.length + 1) || k),
              h('span', { class: 'v', 'data-param': key, style: 'font-size:11px;overflow-wrap:anywhere' }, v)];
          })))),
        h('div', { class: 'note' }, 'Values as stored by the acquisition code (no unit conversion). MATLAB string/datetime objects can only be read in MATLAB.')) : null);
  const body = card.querySelector('.mat-params');
  if (body && st.filter) applyFilter(body);
  return card;
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
  const wIn = (key, j) => input(tt.windows[key] ? fmt(tt.windows[key][j], 6) : '', (v) => { commit('window'); tt.windows[key] = [...(tt.windows[key] || [0, 0])]; tt.windows[key][j] = num(v) ?? 0; emit(); }, { numeric: true, style: 'width:70px' });
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
        input(M.manual[det]?.[0] ?? '', (v) => { M.manual[det] = [num(v), M.manual[det]?.[1] ?? null]; emit(); }, { numeric: true }), input(M.manual[det]?.[1] ?? '', (v) => { M.manual[det] = [M.manual[det]?.[0] ?? null, num(v)]; emit(); }, { numeric: true })])) : null,
    h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'View bounds'),
      h('div', { style: 'display:grid;grid-template-columns:auto minmax(0,1fr) minmax(0,1fr);gap:6px 8px;align-items:center' },
        h('span', { class: 'muted' }, 'X'), input(M.bounds?.x?.[0] ?? '', (v) => set({ bounds: { ...(M.bounds || {}), x: [num(v), M.bounds?.x?.[1] ?? null] } }), { numeric: true, placeholder: 'auto' }), input(M.bounds?.x?.[1] ?? '', (v) => set({ bounds: { ...(M.bounds || {}), x: [M.bounds?.x?.[0] ?? null, num(v)] } }), { numeric: true, placeholder: 'auto' }),
        h('span', { class: 'muted' }, 'Y'), input(M.bounds?.y?.[0] ?? '', (v) => set({ bounds: { ...(M.bounds || {}), y: [num(v), M.bounds?.y?.[1] ?? null] } }), { numeric: true, placeholder: 'auto' }), input(M.bounds?.y?.[1] ?? '', (v) => set({ bounds: { ...(M.bounds || {}), y: [M.bounds?.y?.[0] ?? null, num(v)] } }), { numeric: true, placeholder: 'auto' }))),
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
    h('div', { class: 'g3' }, field('Max spots', input(o.max_spots ?? '', (v) => set({ max_spots: num(v) }), { numeric: true, placeholder: 'auto' })), field('Threshold (σ)', input(o.threshold, (v) => set({ threshold: num(v) || 3 }), { numeric: true })), field('Min pixels', input(o.min_pixels, (v) => set({ min_pixels: num(v) || 4 }), { numeric: true })), field('Background', input(o.bg_window || 'auto', (v) => set({ bg_window: num(v) || 0 }), { numeric: true }))),
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
    const [lo, hi] = finiteExtent(a); const pad = step(a) / 2, L = lo - pad, H = hi + pad;
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
    h('span', { class: 'k' }, '|Δ|'), h('span', { class: 'v', title: 'Pixels: Δx and Δy each divided by that axis\'s own pixel spacing' }, `${fmt(c.distance, 4)}${c.distance_px ? ` (≈ ${fmt(c.distance_px, 2)} px)` : ''}`),
    h('span', { class: 'k' }, 'FWHM ratio (major, minor)'), h('span', { class: 'v' }, `${fmt(c.fwhm_major_ratio, 3)}, ${fmt(c.fwhm_minor_ratio, 3)}`),
    h('span', { class: 'k' }, 'Angle difference'), h('span', { class: 'v' }, `${fmt(c.angle_diff, 3)}°`), h('span', { class: 'k' }, 'Amplitude ratio'), h('span', { class: 'v' }, fmt(c.amplitude_ratio, 3)),
    h('span', { class: 'k' }, 'Local profile correlation'), h('span', { class: 'v', title: c.profile_correlation_note || '' }, c.profile_correlation_note ? 'n/a' : fmt(c.profile_correlation, 3))),
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
    field('Background removed (percentile)', input(M.background, (v) => update({ background: Math.min(100, Math.max(0, num(v) ?? 10)) }), { numeric: true })),
    h('div', { class: 'note' }, 'Different grids are combined on native pixel edges by copying each measured cell. No interpolation or extrapolation; uncovered area stays transparent.'),
    h('div', { class: 'row wrap' }, h('button', { class: 'btn primary', onclick: () => update({}) }, 'Show overlay'),
      h('button', { class: 'btn', disabled: !M.prevLayout, onclick: () => { M.layout = M.prevLayout || 'side'; M.prevLayout = null; emit(); } }, 'Restore previous view')));
}
