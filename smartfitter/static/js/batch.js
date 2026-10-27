// Batch workspace: wraps nvfit.pro_batch with live progress and a filterable results table.
import { call } from './api.js';
import { S, emit, openFile, loadData } from './state.js';
import { runFit } from './analyze.js';
import { h, mount, seg, select, input, field, check, toast, fmt, num, basename, clone } from './util.js';
import { copyText } from './export.js';

let timer = null;

function defaults() {
  const folder = S.library.folder || '';
  return { input_dir: folder, output_dir: folder ? `${folder}\\SmartFitter batch` : '', recursive: false, skip_checkpoints: false, detector: 'all', model: '', mode: 'contrast',
    bin: 1, smooth: 1, multistart: 12, robust: false, odmr_multi: false, dpi: 180, origin: true, detector_labels: { detector1: 'Detector 1', detector2: 'Detector 2' } };
}

async function poll() {
  const B = S.batch;
  if (!B.job) return;
  try {
    B.status = await call('batch_status', { job: B.job }, { quiet: true });
    if (B.status.state !== 'running') { clearInterval(timer); timer = null; B.rows = B.status.rows; toast(`Batch ${B.status.state}: ${B.rows.length} result row(s).`, B.status.state === 'finished' ? 'ok' : ''); }
    emit();
  } catch (e) { clearInterval(timer); timer = null; }
}

export function renderBatch(root) {
  const B = S.batch;
  B.opts = B.opts || defaults();
  const o = B.opts;
  const set = (patch) => { Object.assign(o, patch); emit(); };
  const running = B.status?.state === 'running' || !!B.rerunning;
  const browse = async (key, kind) => { const p = await call('dialog', { kind: 'open_folder', directory: o[key] || S.library.folder || '' }); if (p) set({ [key]: p, ...(key === 'input_dir' && !o.output_dir ? { output_dir: `${p}\\SmartFitter batch` } : {}) }); };
  const left = h('aside', { class: 'library', style: 'flex-basis:320px;width:320px;padding:14px;gap:12px;overflow:auto' },
    field('Input folder', h('div', { class: 'row' }, input(o.input_dir, (v) => set({ input_dir: v })), h('button', { class: 'btn sm', onclick: () => browse('input_dir') }, 'Browse'))),
    check('Include subfolders', o.recursive, (v) => set({ recursive: v })), check('Skip __checkpoint files', o.skip_checkpoints, (v) => set({ skip_checkpoints: v })),
    field('Output folder', h('div', { class: 'row' }, input(o.output_dir, (v) => set({ output_dir: v })), h('button', { class: 'btn sm', onclick: () => browse('output_dir') }, 'Browse'))),
    field('Detectors', seg([['all', 'Both'], ['detector1', 'Detector 1'], ['detector2', 'Detector 2']], o.detector, (v) => set({ detector: v }))),
    field('Model', select([['', 'Auto per experiment type'], 'Rabi', 'Ramsey', 'SpinEcho', 'DynamicDecoupling', 'T1', 'ODMR', 'DEERFrequency', 'DEERDuration', 'DEERDecay', 'DEERPosition', 'TimeTraceDecay'], o.model, (v) => set({ model: v }))),
    field('Observable', select([['contrast', 'Contrast'], ['signal', 'Signal'], ['reference', 'Reference'], ['difference', 'Difference']], o.mode, (v) => set({ mode: v }))),
    h('div', { class: 'g2' }, field('Bin size', input(o.bin, (v) => set({ bin: Math.max(1, num(v) || 1) }))), field('Smoothing', input(o.smooth, (v) => set({ smooth: Math.max(1, num(v) || 1) }))),
      field('Multistart', input(o.multistart, (v) => set({ multistart: Math.max(4, num(v) || 12) }))), field('Figure DPI', input(o.dpi, (v) => set({ dpi: num(v) || 180 }))),
      field('D1 label', input(o.detector_labels.detector1, (v) => set({ detector_labels: { ...o.detector_labels, detector1: v } }))), field('D2 label', input(o.detector_labels.detector2, (v) => set({ detector_labels: { ...o.detector_labels, detector2: v } })))),
    check('Robust fitting (subset comparison)', o.robust, (v) => set({ robust: v })), check('ODMR multi-peak', o.odmr_multi, (v) => set({ odmr_multi: v })), check('Write Origin bundles', o.origin, (v) => set({ origin: v })),
    h('div', { class: 'row' }, running ? h('button', { class: 'btn danger', style: 'flex:1;justify-content:center', onclick: () => call('batch_cancel', { job: B.job }) }, 'Cancel')
      : h('button', { class: 'btn primary', style: 'flex:1;justify-content:center', disabled: !o.input_dir || !o.output_dir, onclick: start }, 'Run batch'),
    h('button', { class: 'btn', onclick: async () => copyText(await call('batch_cli', { opts: o }), 'Command copied') }, 'Copy CLI command')),
    h('div', { class: 'note' }, 'Uses the same nvfit.pro_batch pipeline as the command line, so results match.'));
  const rows = B.rows || [];
  const statuses = ['PASS', 'WARN', 'FAIL'];
  const counts = Object.fromEntries(['all', ...statuses, 'other'].map((k) => [k, k === 'all' ? rows.length : k === 'other' ? rows.filter((r) => !statuses.includes(r.status)).length : rows.filter((r) => r.status === k).length]));
  const words = (B.text || '').toLowerCase().split(/\s+/).filter(Boolean);
  const shown = rows.filter((r) => (B.filter === 'all' || (B.filter === 'other' ? !statuses.includes(r.status) : r.status === B.filter)) && words.every((w) => Object.values(r).join(' ').toLowerCase().includes(w)));
  const st = B.status;
  const pathOf = (r) => `${B.ranInput || o.input_dir}\\${r.relative_path || r.file}`;
  const filterEl = input(B.text || '', (v) => { B.text = v; emit(); }, { placeholder: 'Filter rows', style: 'width:220px' });
  const main = h('main', { class: 'main' },
    h('div', { class: 'row wrap' }, h('h1', { style: 'margin:0;font-size:17px;font-weight:600' }, basename(o.input_dir) || 'Batch'),
      st ? h('span', { class: 'muted' }, `${st.done}/${st.total || '?'} · ${st.state}${st.error ? ': ' + st.error : ''}`) : h('span', { class: 'muted' }, 'Not run yet'),
      st ? h('div', { class: 'progress' }, h('div', { style: `width:${st.total ? (100 * st.done) / st.total : 0}%` })) : null),
    h('div', { class: 'row wrap', style: 'gap:6px' }, [['all', 'All'], ['PASS', 'Pass'], ['WARN', 'Check'], ['FAIL', 'Fail'], ['other', 'Plot only / other']].map(([k, l]) => h('button', { class: `chip ${B.filter === k ? 'on' : ''}`, onclick: () => { B.filter = k; emit(); } }, `${l} ${counts[k]}`)),
      h('span', { class: 'sp' }), filterEl),
    h('div', { class: 'card', style: 'flex:1;overflow:auto;min-height:200px' },
      rows.length ? h('table', { class: 'tbl bigtbl' }, h('tr', {}, ['File', 'Detector', 'Type', 'Status', 'Model', 'R²', 'Reason', ''].map((t) => h('th', {}, t))),
        shown.map((r) => h('tr', {},
          h('td', { class: 'txt', title: r.relative_path }, r.file), h('td', { class: 'txt' }, r.detector_label), h('td', { class: 'txt' }, r.experiment_type),
          h('td', { class: 'txt' }, h('span', { class: `pill ${r.status}` }, r.status)), h('td', { class: 'txt' }, r.model), h('td', { class: 'num' }, r.r2 ? fmt(Number(r.r2), 4) : ''),
          h('td', { class: 'txt ellip', style: 'max-width:340px', title: r.status_reason }, r.status_reason),
          h('td', {}, h('button', { class: 'btn sm', onclick: () => openFile(pathOf(r)).then((f) => { if (r.detector_id && f.info.detectors.includes(r.detector_id)) f.detector = r.detector_id; emit(); }) }, 'Open'), ' ', h('button', { class: 'btn sm', onclick: () => openFigure(r) }, 'Figure')))))
        : h('div', { class: 'empty', style: 'padding:40px' }, running ? 'Running…' : 'Results appear here as a table with one row per file and detector.')),
    st?.log?.length ? h('details', { class: 'card pad' }, h('summary', {}, `Log (${st.log.length} lines)`), h('pre', { class: 'mono', style: 'font-size:11px;max-height:200px;overflow:auto;white-space:pre-wrap' }, st.log.slice(-300).join('\n'))) : null,
    h('div', { class: 'row wrap' },
      h('button', { class: 'btn', disabled: !rows.some((r) => r.status === 'PASS' || r.status === 'WARN'), title: 'Open the passing rows in Compare and refit each file/detector with this batch run\'s settings (batch summaries hold metrics, not curves)', onclick: sendToCompare }, 'Send fitted rows to Compare'),
      h('button', { class: 'btn', disabled: !o.output_dir, onclick: () => call('open_path', { path: o.output_dir }) }, 'Open results folder'),
      h('button', { class: 'btn', disabled: !rows.length, onclick: () => call('open_path', { path: `${o.output_dir}\\batch_summary.csv` }) }, 'Open summary CSV'),
      h('button', { class: 'btn', disabled: !o.output_dir, onclick: async () => { B.rows = await call('batch_summary', { output_dir: o.output_dir }); B.ranInput = o.input_dir; if (B.ranOpts?.output_dir !== o.output_dir) B.ranOpts = null; emit(); } }, 'Load previous results'),
      h('button', { class: 'btn', disabled: running || !o.output_dir || !rows.some((r) => r.status === 'FAIL'), onclick: rerunFailed }, 'Re-run failed')));
  mount(root, left, main);

  // Batch summaries carry metrics (T1_ns, R², ...) but no curves, so Compare refits each row with the batch run's own
  // settings (observable, bin, smoothing, model, multistart, robust, ODMR multi-peak) through the normal workspace fit.
  async function sendToCompare() {
    const ok = rows.filter((x) => x.status === 'PASS' || x.status === 'WARN');
    const run = B.ranOpts || o;
    // Compare stores one result per file path, so a second detector row for the same
    // file would silently replace the first result and overstate the sent count.
    const seen = new Set();
    let fitted = 0, failed = 0, skippedDetectors = 0;
    for (const r of ok) {
      const p = pathOf(r);
      if (seen.has(p)) { skippedDetectors++; continue; }
      try {
        const f = await openFile(p, { primary: false });
        const mode = f.info.observables.includes(run.mode) ? run.mode : f.mode;
        if (f.mode !== mode || f.proc.bin !== run.bin || f.proc.smooth !== run.smooth) { f.mode = mode; Object.assign(f.proc, { bin: run.bin, smooth: run.smooth }); await loadData(f); }
        Object.assign(f.fitSpec, { model: run.model || 'Auto', multistart: run.multistart, odmr_multi: !!run.odmr_multi, apply_both: false });
        if (r.detector_id && f.info.detectors.includes(r.detector_id)) f.detector = r.detector_id;
        const previousFits = f.fitHistory.length;
        await runFit(f, { robust: !!run.robust, recordUndo: false });
        if (f.fitHistory.length > previousFits && f.fit[f.detector]) {
          seen.add(p);
          if (!S.compare.set.includes(p)) S.compare.set.push(p);
          S.compare.results[p] = f.fit[f.detector];
          fitted++;
        } else failed++;
      } catch (e) { failed += 1; }
    }
    toast(`Sent ${fitted} file(s) to Compare${failed ? `, ${failed} failed` : ''}${skippedDetectors ? `, ${skippedDetectors} additional detector row(s) skipped (Compare shows one detector per file)` : ''}.`, failed || skippedDetectors ? '' : 'ok');
    S.view = 'compare'; emit();
  }
}

async function start() {
  const B = S.batch;
  B.rows = [];
  B.ranInput = B.opts.input_dir;
  B.ranOpts = clone(B.opts);
  B.job = await call('batch_start', { opts: B.opts }, { label: 'Starting batch' });
  B.status = { state: 'running', done: 0, total: 0, log: [] };
  if (timer) clearInterval(timer);
  timer = setInterval(poll, 700);
  emit();
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const rowKey = (r) => r.relative_path || r.file;

// pro_batch writes <stem>_fit.png and records the stem as output_stem (de-duplicated when two sources
// would collide). Older summaries lack it: dual-detector files get a __<detector_id> suffix, and
// recursive runs flatten subfolders into the stem with "__" separators.
async function openFigure(r) {
  const dir = S.batch.opts.output_dir;
  const stem = rowKey(r).replace(/\.[^./\\]+$/, '').split(/[\\/]/).join('__');
  const guesses = r.detector_id ? [`${stem}__${r.detector_id}_fit.png`, `${stem}_fit.png`] : [`${stem}_fit.png`];
  const names = r.output_stem ? [`${r.output_stem}_fit.png`] : guesses;
  for (const name of names) {
    try { await call('open_path', { path: `${dir}\\${name}` }, { quiet: true }); return; } catch (e) { /* try the next candidate */ }
  }
  toast('Figure not found', 'err');
}

// Re-runs each failed file alone (single_file), sequentially, with merge_summary so the CSVs keep all rows.
async function rerunFailed() {
  const B = S.batch, o = B.opts, base = B.ranInput || o.input_dir;
  const failed = [...new Set((B.rows || []).filter((r) => r.status === 'FAIL').map(rowKey))];
  if (!failed.length) return;
  let cancelled = false;
  B.rerunning = true; emit();
  try {
    for (const key of failed) {
      B.job = await call('batch_start', { opts: { ...o, input_dir: base, output_dir: o.output_dir, single_file: `${base}\\${key}`, merge_summary: true } }, { label: 'Starting re-run' });
      B.status = { state: 'running', done: 0, total: 0, log: [] };
      emit();
      let st;
      do {
        await wait(700);
        st = await call('batch_status', { job: B.job }, { quiet: true });
        B.status = st; emit();
      } while (st.state === 'running');
      if (st.state === 'cancelled') { cancelled = true; break; }
    }
    B.rows = await call('batch_summary', { output_dir: o.output_dir }, { quiet: true }); emit();
    toast(cancelled ? 'Re-run cancelled.' : 'Re-run finished.', cancelled ? '' : 'ok');
  } finally {
    B.rerunning = false; emit();
  }
}
