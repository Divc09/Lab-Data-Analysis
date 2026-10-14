// Application state, file loading and undo/redo history.
import { call } from './api.js';
import { clone, toast } from './util.js';
import { unitScale } from './unitdisplay.js';

export const S = {
  version: '', desktop: false, settings: {},
  view: 'home',
  libCollapsed: false,
  library: { folder: null, listing: null, peeks: {}, filter: '', chip: 'all', extra: [], seen: {}, newSince: {} },
  files: {},
  primary: null,
  overlays: [], overlaySlots: {},
  compare: { set: [], norm: 'max', layout: 'overlaid', param: '', against: 'time', results: {}, trendSel: null },
  figure: { panels: [], size: 'slide', labels: 'a', preset: 'presentation', font: 'Arial', base: 14, dpi: 300, w: 13.33, h: 7.5, cols: 0, sharey: false },
  batch: { opts: null, job: null, status: null, rows: [], filter: 'all', text: '' },
  editor: { open: false, tab: 'series' },
};

const listeners = new Set();
export const subscribe = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
export function emit(what = 'all') { listeners.forEach((fn) => fn(what)); }

export const DEFAULT_LAYERS = {
  data: true, fit: true, band: false, res: true, smooth: false, iter: false, iterFits: false, meanSd: false, envelope: false,
  conf: false, sigref: false, fft: false, legend: true, results: false, peaks: true,
};

export function defaultFitSpec(info) {
  return {
    model: 'Auto', experiment: '', range_min: null, range_max: null, multistart: 12, robust: false,
    rabi_mode: 'Auto compare', sem_weights: true, pi_lock: false, pi_ns: 200, pi2_ns: 100, dead_time_ns: 0,
    odmr_polarity: 'Auto', odmr_multi: false, odmr_peaks: 2, odmr_auto_peaks: true, odmr_ranges: [],
    t1_prep: 'Auto', t1_family: 'Stretched exponential', ramsey_model: 'Auto', peaks: 1,
    custom_expr: 'y0 + A*exp(-x/tau)', custom_params: [], fit_smoothed: false, quality_gate: true, params: {},
    apply_both: false,
  };
}

function newFileState(path, info, settings) {
  const st = settings || {};
  const mode = info.reference_only ? 'reference' : (info.observables.includes('contrast') ? 'contrast' : info.observables[0] || 'contrast');
  return {
    path, info, mode,
    detector: info.detectors[0] || 'detector1',
    labels: { ...(st.detector_labels || {}), ...info.detector_labels },
    proc: { bin: 1, smooth: 5, iterations: null, roi_min: null, roi_max: null, excluded: [], exclusion_ranges: [], steps: [] },
    data: {}, fit: {}, fitHistory: [], iterationFits: {}, iterationView: {}, combinedRecipe: info.combined_recipe || null,
    display: { invert: false, xscale: 1, xoffset: 0, t1: 'original', overlayFits: true }, sample: {}, reportKeys: null,
    fitSpec: defaultFitSpec(info),
    layers: { ...DEFAULT_LAYERS, ...(info.type === 'TimeTrace' ? { res: false, legend: true } : {}) },
    styleMode: 'both', errorMode: 'band',
    compareDetectors: false,
    pins: [], tab: info.dim === 'scan2d' ? 'spots' : (info.type === 'TimeTrace' ? 'stats' : 'fit'),
    style: clone(st.plot_style || {}) || {},
    xrange: null, yrange: null,
    map: { cmap: st.colormap || 'viridis', reverse: false, log: false, limits: 'robust', plo: 1, phi: 99.5, manual: {}, scale: 'independent',
      geometry: st.pixel_geometry || 'physical', swap: st.swap_2d_axes !== false, layout: info.detectors.length > 1 ? 'side' : 'single',
      linecut: 'none', cutSource: 'value', cursor: null, bounds: null, compareWith: null, palette: 'Pink / teal', overlayScale: 'independent', background: 10,
      showShapes: true, showCenters: true, showNumbers: false, showSource: false },
    spotOpts: { source: 'auto', polarity: 'Auto', threshold: 3, min_pixels: 4, bg_window: 0, scope: 'both' },
    spots: {}, spotRemoved: {}, spotSel: [],
    tt: { layout: 'stacked', unit: 's', rolling: 10, showRolling: false, windows: { a: null, b: null }, showWindows: true, showRetries: true, showEvents: true, norm: 'none', stats: null, pgram: null },
  };
}

export async function openFile(path, { primary = true } = {}) {
  let f = S.files[path];
  if (!f) {
    const peek = await call('load', { path, mode: 'contrast' }, { label: 'Loading' });
    f = newFileState(path, peek.info, S.settings);
    S.files[path] = f;
    await loadData(f);
    if (!path.startsWith('combined:')) call('remember', { kind: 'recent_files', path }, { quiet: true }).catch(() => {});
  }
  if (primary) { S.primary = path; S.view = 'analyze'; await syncOverlays(f); }
  emit();
  return f;
}

// Fetch (or refetch) the processed data for every detector this view needs.
export async function loadData(f, { all = true } = {}) {
  const dets = all ? f.info.detectors : [f.detector];
  const results = await Promise.all(dets.map((d) => call('load', { path: f.path, detector: d, mode: f.mode, proc: f.proc }, { label: 'Loading' })));
  dets.forEach((d, i) => { f.data[d] = results[i]; });
  f.info = results[0].info;
}

export async function reloadFile(f) {
  await loadData(f);
  emit();
}

export async function syncOverlay(o, f) {
  if (o.info.dim === 'scan2d' || o.info.type === 'TimeTrace') throw new Error('Compare overlays require 1D files. Use the Compare workspace for maps/time traces.');
  if (!o.info.observables.includes(f.mode)) throw new Error(`${o.info.name} has no ${f.mode} observable.`);
  if (unitScale(o.info.x_unit || '', f.info.x_unit || '') === null) throw new Error(`${o.info.name} has incompatible x units (${o.info.x_unit || 'unitless'} vs ${f.info.x_unit || 'unitless'}).`);
  if (o.mode !== f.mode) {
    o.fitByMode = o.fitByMode || {};
    o.fitByMode[o.mode] = o.fit;
    o.mode = f.mode; o.fit = o.fitByMode[f.mode] || {};
    await loadData(o);
  }
  return o;
}
export async function addOverlay(path, f = active()) {
  if (!f || path === f.path) return;
  const o = await openFile(path, { primary: false });
  await syncOverlay(o, f);
  if (!S.overlays.includes(path)) S.overlays.push(path);
  if (S.overlaySlots[path] === undefined) S.overlaySlots[path] = Object.keys(S.overlaySlots).length;
  if (!S.library.extra.includes(path)) S.library.extra.push(path);
  emit();
  return o;
}
export function removeOverlay(path) { S.overlays = S.overlays.filter((p) => p !== path); emit(); }
export async function syncOverlays(f) {
  for (const p of S.overlays.filter((p) => p !== f.path && S.files[p])) {
    try { await syncOverlay(S.files[p], f); } catch (e) { toast(`Overlay skipped: ${e.message}`, 'err'); }
  }
}

export const active = () => (S.primary ? S.files[S.primary] : null);
export const detLabel = (f, d) => (f.labels && f.labels[d]) || (d === 'detector2' ? 'Detector 2' : 'Detector 1');

// ---- undo / redo: snapshots of the analysis-relevant parts of the active file ----
const undoStack = [], redoStack = [];
const SNAP_KEYS = ['mode', 'detector', 'proc', 'fitSpec', 'fit', 'fitByMode', 'fitStale', 'layers', 'layersTouched', 'styleMode', 'errorMode', 'style', 'map', 'spots', 'spotRemoved', 'tt', 'compareDetectors', 'pins', 'measure', 'display', 'xrange', 'yrange', 'sample', 'reportKeys', 'iterationFits', 'iterationView', 'iterationFitsStale'];
const fileSnap = (f) => clone(Object.fromEntries(SNAP_KEYS.map((k) => [k, f[k]])));
const snap = (f) => ({ path: f.path, state: fileSnap(f), states: Object.fromEntries(Object.values(S.files).map((o) => [o.path, fileSnap(o)])), overlays: [...S.overlays], overlaySlots: { ...S.overlaySlots } });

export function commit(label = '') {
  const f = active();
  if (!f) return;
  undoStack.push({ label, ...snap(f) });
  if (undoStack.length > 80) undoStack.shift();
  redoStack.length = 0;
}

async function restore(entry, into) {
  const f = S.files[entry.path];
  if (!f) return;
  into.push({ label: entry.label, ...snap(f) });
  for (const [path, state] of Object.entries(entry.states || { [entry.path]: entry.state })) {
    const o = S.files[path]; if (!o) continue;
    const needReload = state.mode !== o.mode || JSON.stringify(state.proc) !== JSON.stringify(o.proc);
    Object.assign(o, clone(state));
    if (needReload) await loadData(o);
  }
  if (entry.overlays) S.overlays = [...entry.overlays];
  if (entry.overlaySlots) S.overlaySlots = { ...entry.overlaySlots };
  S.primary = entry.path;
  await syncOverlays(f);
  emit();
}
export async function undo() { const e = undoStack.pop(); if (e) { await restore(e, redoStack); toast(`Undid ${e.label || 'change'}`); } }
export async function redo() { const e = redoStack.pop(); if (e) { await restore(e, undoStack); toast(`Redid ${e.label || 'change'}`); } }
export const canUndo = () => undoStack.length > 0;
export const canRedo = () => redoStack.length > 0;

// ---- session documents ----
export function sessionDoc() {
  const files = Object.values(S.files).map((f) => ({
    path: f.path, mode: f.mode, detector: f.detector, labels: f.labels, proc: f.proc, fitSpec: f.fitSpec,
    fit: Object.fromEntries(Object.entries(f.fit).map(([d, r]) => [d, { ...r, raw: undefined }])),
    layers: f.layers, styleMode: f.styleMode, errorMode: f.errorMode, style: f.style, map: f.map, spotOpts: f.spotOpts,
    spots: f.spots, spotRemoved: f.spotRemoved, tt: { ...f.tt, pgram: null }, compareDetectors: f.compareDetectors, pins: f.pins, measure: f.measure,
    xrange: f.xrange, yrange: f.yrange, tab: f.tab, display: f.display, sample: f.sample, reportKeys: f.reportKeys,
    fitByMode: f.fitByMode, fitStale: f.fitStale, layersTouched: f.layersTouched, exp: f.exp, expStem: f.expStem, expDir: f.expDir,
    iterationFits: f.iterationFits, iterationView: f.iterationView, iterationFitsStale: f.iterationFitsStale, combinedRecipe: f.combinedRecipe,
  }));
  // Compare fits may have been produced in its own workspace and need not exist in f.fit.
  // Keep the curves/metrics needed for the overlay and trend, omitting only the large raw fitter payload.
  const compareResults = Object.fromEntries(Object.entries(S.compare.results || {})
    .filter(([path]) => S.files[path] && S.compare.set.includes(path))
    .map(([path, result]) => [path, { ...result, raw: undefined }]));
  return { files: files.map((x) => x.path), state: files, primary: S.primary, overlays: S.overlays, overlaySlots: S.overlaySlots, compare: { ...S.compare, results: compareResults }, figure: S.figure, library: { folder: S.library.folder, extra: S.library.extra } };
}

export async function applySession(doc) {
  const moved = doc.moved || {};
  S.files = {};
  const remapSources = (value) => {
    if (!value || typeof value !== 'object') return;
    if (typeof value.path === 'string') value.path = moved[value.path] || value.path;
    Object.values(value).forEach(remapSources);
  };
  for (let saved of doc.state || []) {
    const changed = (doc.changed || []).includes(saved.path) || (saved.combinedRecipe?.runs || []).some((r) => (doc.changed || []).includes(r.path));
    saved = clone(saved); remapSources(saved);
    const path = moved[saved.path] || saved.path;
    try {
      if (saved.combinedRecipe) {
        const recipe = clone(saved.combinedRecipe);
        recipe.runs = recipe.runs.map((r) => ({ ...r, path: moved[r.path] || r.path }));
        saved = { ...saved, combinedRecipe: recipe };
        await call('combine_iterations', { ...recipe, mode: saved.mode }, { label: 'Rebuilding combined iterations' });
      }
      const f = await openFile(path, { primary: false });
      Object.assign(f, clone({ ...saved, path }));
      if (changed) { f.fitStale = true; f.iterationFitsStale = true; }
      f.style ||= {};
      if (f.style.fileOrder) f.style.fileOrder = f.style.fileOrder.map((p) => moved[p] || p);
      if (f.style.comparison) f.style.comparison = Object.fromEntries(Object.entries(f.style.comparison).map(([p, cfg]) => [moved[p] || p, cfg]));
      const remapKey = (k) => k.replace(/^(ovfit:|ov:)(.*)$/, (_, prefix, p) => prefix + (moved[p] || p));
      if (f.style.series) f.style.series = Object.fromEntries(Object.entries(f.style.series).map(([k, cfg]) => [remapKey(k), cfg]));
      if (f.style.seriesOrder) f.style.seriesOrder = f.style.seriesOrder.map(remapKey);
      f.data = {};
      await loadData(f);
    } catch (e) {
      toast(`Could not reopen ${saved.path}: ${e.message}`, 'err');
    }
  }
  S.primary = doc.primary && S.files[moved[doc.primary] || doc.primary] ? (moved[doc.primary] || doc.primary) : Object.keys(S.files)[0] || null;
  S.overlays = (doc.overlays || []).map((p) => moved[p] || p).filter((p) => !!S.files[p]);
  S.overlaySlots = Object.fromEntries(Object.entries(doc.overlaySlots || {}).map(([p, slot]) => [moved[p] || p, slot]));
  if (active()) await syncOverlays(active());
  const savedCompare = doc.compare || {};
  S.compare = { set: [], norm: 'max', layout: 'overlaid', param: '', against: 'time', results: {}, trendSel: null, ...clone(savedCompare) };
  for (const key of ['set', 'hidden', 'sel', 'trendSel']) if (Array.isArray(S.compare[key])) S.compare[key] = S.compare[key].map((p) => moved[p] || p);
  const changed = new Set(doc.changed || []);
  S.compare.results = Object.fromEntries(Object.entries(savedCompare.results || {})
    .map(([path, result]) => [moved[path] || path, result, path])
    .filter(([path, , oldPath]) => S.files[path] && !changed.has(oldPath) && !changed.has(path))
    .map(([path, result]) => [path, result]));
  // A pooled fit refers to the original source paths too; discard it when any input changed.
  if (S.compare.global?.res?.datasets?.some((d) => changed.has(d.path) || changed.has(moved[d.path] || d.path) || !S.files[moved[d.path] || d.path])) S.compare.global = null;
  else if (S.compare.global?.res?.datasets) {
    S.compare.global.res.datasets.forEach((d) => { d.path = moved[d.path] || d.path; });
    S.compare.global.key = S.compare.global.key?.split('|').map((p) => moved[p] || p).join('|');
  }
  Object.assign(S.figure, doc.figure || {});
  S.figure.panels = S.figure.panels.map((p) => ({ ...p, path: moved[p.path] || p.path }));
  if (doc.library?.folder) S.library.folder = doc.library.folder;
  S.library.extra = (doc.library?.extra || []).map((p) => moved[p] || p);
  S.view = S.primary ? 'analyze' : 'home';
  if ((doc.changed || []).length) toast(`${doc.changed.length} file(s) changed since the session was saved; showing current data.`);
  emit();
}
