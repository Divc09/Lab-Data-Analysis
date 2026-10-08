// Application state, file loading and undo/redo history.
import { call } from './api.js';
import { clone, toast } from './util.js';

export const S = {
  version: '', desktop: false, settings: {},
  view: 'home',
  libCollapsed: false,
  library: { folder: null, listing: null, peeks: {}, filter: '', chip: 'all', extra: [], seen: {}, newSince: {} },
  files: {},
  primary: null,
  overlays: [],
  compare: { set: [], norm: 'max', layout: 'overlaid', param: '', against: 'time', results: {}, trendSel: null },
  figure: { panels: [], size: 'slide', labels: 'a', preset: 'presentation', font: 'Arial', base: 14, dpi: 300, w: 13.33, h: 7.5, cols: 0, sharey: false },
  batch: { opts: null, job: null, status: null, rows: [], filter: 'all', text: '' },
  editor: { open: false, tab: 'series' },
};

const listeners = new Set();
export const subscribe = (fn) => listeners.add(fn);
export function emit(what = 'all') { listeners.forEach((fn) => fn(what)); }

export const DEFAULT_LAYERS = {
  data: true, fit: true, band: false, res: true, smooth: false, iter: false, meanSd: false, envelope: false,
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
    proc: { bin: 1, smooth: 5, roi_min: null, roi_max: null, excluded: [], exclusion_ranges: [], steps: [] },
    data: {}, fit: {}, fitHistory: [],
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
    call('remember', { kind: 'recent_files', path }, { quiet: true }).catch(() => {});
  }
  if (primary) { S.primary = path; S.view = 'analyze'; }
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

export const active = () => (S.primary ? S.files[S.primary] : null);
export const detLabel = (f, d) => (f.labels && f.labels[d]) || (d === 'detector2' ? 'Detector 2' : 'Detector 1');

// ---- undo / redo: snapshots of the analysis-relevant parts of the active file ----
const undoStack = [], redoStack = [];
const SNAP_KEYS = ['mode', 'detector', 'proc', 'fitSpec', 'fit', 'layers', 'styleMode', 'errorMode', 'style', 'map', 'spots', 'spotRemoved', 'tt', 'compareDetectors', 'pins'];
const snap = (f) => ({ path: f.path, state: clone(Object.fromEntries(SNAP_KEYS.map((k) => [k, f[k]]))) });

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
  const needReload = entry.state.mode !== f.mode || JSON.stringify(entry.state.proc) !== JSON.stringify(f.proc);
  Object.assign(f, clone(entry.state));
  S.primary = entry.path;
  if (needReload) await loadData(f);
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
    spots: f.spots, spotRemoved: f.spotRemoved, tt: { ...f.tt, pgram: null }, compareDetectors: f.compareDetectors, pins: f.pins,
    xrange: f.xrange, yrange: f.yrange, tab: f.tab,
  }));
  return { files: files.map((x) => x.path), state: files, primary: S.primary, overlays: S.overlays, compare: { ...S.compare, results: {} }, figure: S.figure, library: { folder: S.library.folder, extra: S.library.extra } };
}

export async function applySession(doc) {
  const moved = doc.moved || {};
  S.files = {};
  for (const saved of doc.state || []) {
    const path = moved[saved.path] || saved.path;
    try {
      const f = await openFile(path, { primary: false });
      Object.assign(f, clone({ ...saved, path }));
      f.data = {};
      await loadData(f);
    } catch (e) {
      toast(`Could not reopen ${saved.path}: ${e.message}`, 'err');
    }
  }
  S.primary = doc.primary && S.files[moved[doc.primary] || doc.primary] ? (moved[doc.primary] || doc.primary) : Object.keys(S.files)[0] || null;
  S.overlays = (doc.overlays || []).map((p) => moved[p] || p).filter((p) => S.files[p] || true);
  Object.assign(S.compare, doc.compare || {});
  Object.assign(S.figure, doc.figure || {});
  if (doc.library?.folder) S.library.folder = doc.library.folder;
  S.library.extra = doc.library?.extra || [];
  S.view = S.primary ? 'analyze' : 'home';
  if ((doc.changed || []).length) toast(`${doc.changed.length} file(s) changed since the session was saved; showing current data.`);
  emit();
}
