// Batch workspace: wraps nvfit.pro_batch with live progress and a filterable results table.
import { call } from './api.js';
import { S, emit, openFile, loadData, addCompareItem } from './state.js';
import { h, mount, seg, select, input, field, check, toast, fmt, num, basename, clone } from './util.js';
import { copyText } from './export.js';
import { BATCH_STATUSES, comparableBatchRow, needsBatchRetry, batchTableText } from './batchstatus.js';

let timer = null;

export async function saveBatchSummary() {
  const path = await call('dialog', { kind: 'save', title: 'Export batch summary', filename: 'batch_summary.csv', directory: S.batch.opts?.output_dir || '' });
  if (!path) return;
  await call('write_text', { path: /\.csv$/i.test(path) ? path : `${path}.csv`, text: batchTableText(S.batch.rows) });
  toast('Saved batch summary', 'ok');
}
export const copyBatchSummary = () => copyText(batchTableText(S.batch.rows, '\t'), 'Batch summary copied');

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
    B.rows = B.status.rows || [];
    B.manifestPath = B.status.manifest_path || B.manifestPath;
    if (B.status.state !== 'running') {
      clearInterval(timer);
      timer = null;
      B.rows = B.status.rows;
      B.manifestPath = B.status.manifest_path;
      toast(
        `Batch ${B.status.state}: ${B.status.done}/${B.status.total} requested rows completed.`,
        B.status.state === 'finished' ? 'ok' : ''
      );
    }
    emit();
  } catch (e) {
    clearInterval(timer);
    timer = null;
  }
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
    field('Model', select([['', 'Auto per experiment type'], 'Rabi', 'Ramsey', 'SpinEcho', 'DynamicDecoupling', 'T1', 'ODMR', 'DEERFrequency', 'DEERDuration', 'DEERDecay', 'DEERPosition', 'TimeTraceDecay', ['GaussianPeaks', 'Gaussian peak']], o.model, (v) => set({ model: v }))),
    field('Observable', select([['contrast', 'Contrast'], ['signal', 'Signal'], ['reference', 'Reference'], ['difference', 'Difference']], o.mode, (v) => set({ mode: v }))),
    h('div', { class: 'g2' }, field('Bin size', input(o.bin, (v) => set({ bin: Math.max(1, num(v) || 1) }), { numeric: true })), field('Smoothing', input(o.smooth, (v) => set({ smooth: Math.max(1, num(v) || 1) }), { numeric: true })),
      field('Multistart', input(o.multistart, (v) => set({ multistart: Math.max(1, num(v) || 12) }), { numeric: true })), field('Figure DPI', input(o.dpi, (v) => set({ dpi: num(v) || 180 }), { numeric: true })),
      field('D1 label', input(o.detector_labels.detector1, (v) => set({ detector_labels: { ...o.detector_labels, detector1: v } }))), field('D2 label', input(o.detector_labels.detector2, (v) => set({ detector_labels: { ...o.detector_labels, detector2: v } })))),
    check('Robust fitting (subset comparison)', o.robust, (v) => set({ robust: v })), check('ODMR multi-peak', o.odmr_multi, (v) => set({ odmr_multi: v })), check('Write Origin bundles', o.origin, (v) => set({ origin: v })),
    h('div', { class: 'row' }, running ? h('button', { class: 'btn danger', style: 'flex:1;justify-content:center', onclick: () => call('batch_cancel', { job: B.job }) }, 'Cancel after row')
      : h('button', { class: 'btn primary', style: 'flex:1;justify-content:center', disabled: !o.input_dir || !o.output_dir, onclick: start }, 'Run batch'),
    h('button', { class: 'btn', onclick: async () => copyText(await call('batch_cli', { opts: o }), 'Command copied') }, 'Copy CLI command')),
    h('div', { class: 'note' }, 'Uses the same nvfit.pro_batch pipeline as the command line, so results match.'));
  const rows = B.rows || [];
  const statuses = BATCH_STATUSES;
  const counts = Object.fromEntries(['all', ...statuses, 'other'].map((k) => [k, k === 'all' ? rows.length : k === 'other' ? rows.filter((r) => !statuses.includes(r.status)).length : rows.filter((r) => r.status === k).length]));
  const words = (B.text || '').toLowerCase().split(/\s+/).filter(Boolean);
  const shown = rows.filter((r) => (B.filter === 'all' || (B.filter === 'other' ? !statuses.includes(r.status) : r.status === B.filter)) && words.every((w) => Object.values(r).join(' ').toLowerCase().includes(w)));
  const st = B.status;
  const pathOf = (r) => r.source_path || `${B.ranInput || o.input_dir}\\${r.relative_path || r.file}`;
  const filterEl = input(B.text || '', (v) => { B.text = v; emit(); }, { placeholder: 'Filter rows', style: 'width:220px' });
  const main = h('main', { class: 'main' },
    h('div', { class: 'row wrap' }, h('h1', { style: 'margin:0;font-size:17px;font-weight:600' }, basename(o.input_dir) || 'Batch'),
      st ? h('span', { class: 'muted' }, `${st.done}/${st.total || '?'} · ${st.state}${st.error ? ': ' + st.error : ''}`) : h('span', { class: 'muted' }, 'Not run yet'),
      st ? h('div', { class: 'progress' }, h('div', { style: `width:${st.total ? (100 * st.done) / st.total : 0}%` })) : null),
    h('div', { class: 'row wrap', style: 'gap:6px' }, [['all', 'All'], ['PASS', 'Fit pass'], ['WARN', 'Fit check'], ['FAIL', 'Fail'], ['PLOT_ONLY', 'Plot only'], ['UNCONVERGED', 'Unconverged'], ['FALLBACK', 'Fallback'], ['EXCLUDED', 'Excluded'], ['other', 'Other']].map(([k, l]) => h('button', { class: `chip ${B.filter === k ? 'on' : ''}`, onclick: () => { B.filter = k; emit(); } }, `${l} ${counts[k]}`)),
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
      h('button', { class: 'btn', disabled: !rows.some(comparableBatchRow), title: 'Import the saved curves and exact run settings for each converged fitted row', onclick: sendToCompare }, 'Send fitted rows to Compare'),
      h('button', { class: 'btn', disabled: !o.output_dir, onclick: () => call('open_path', { path: o.output_dir }) }, 'Open results folder'),
      h('button', { class: 'btn', disabled: !rows.length, onclick: () => call('open_path', { path: `${o.output_dir}\\batch_summary.csv` }) }, 'Open summary CSV'),
      h('button', { class: 'btn', disabled: !o.output_dir, onclick: async () => { B.rows=await call('batch_summary',{output_dir:o.output_dir});const run=await call('batch_run_record',{output_dir:o.output_dir});B.manifestPath=run?.manifest_path;B.ranInput=run?.options.input_dir || o.input_dir;B.ranOpts=run?.ui_options || null;B.status=run?{state:run.state,done:run.completed_count,total:run.requested_rows.length,unfinished_count:run.rows.filter(r=>r.state==='pending').length,discovery_complete:run.discovery_complete}:null;emit();} }, 'Load previous results'),
      h('button', { class: 'btn', disabled: running || !rows.some(r=>needsBatchRetry(r) && r.row_id && r.manifest_path), title: 'Retry these row IDs using the saved run’s sources and options', onclick: rerunFailed }, 'Re-run problem rows'),
      h('button',{class:'btn',disabled:running || !B.manifestPath || !(B.status?.unfinished_count || B.status?.discovery_complete===false),onclick:resumeUnfinished},'Resume unfinished rows')));
  mount(root, left, main);

  // Import the stored curves/settings, preserving each detector and every dirty Analyze control.
  async function sendToCompare() {
    const ok=rows.filter(comparableBatchRow);let imported=0,failed=0;
    for(const row of ok) {
      try {
        if(!row.manifest_path || !row.row_id)throw new Error('Legacy summary lacks a reproducible run record; start a new batch.');
        const item=await call('batch_import',{manifest_path:row.manifest_path,row_id:row.row_id},{label:'Importing stored batch fit'});
        const f=await openFile(item.path,{primary:false});
        addCompareItem(f,{detector:item.detector,mode:item.mode,proc:item.proc,spec:item.spec,result:item.fit,linked:false,data:item.data});
        imported++;
      } catch(e){failed++;}
    }
    S.compare.spec={model:'useOwn'};
    toast(`Imported ${imported} stored analysis item(s)${failed ? `; ${failed} unavailable or legacy rows` : ''}.`,failed?'':'ok');
    S.view='compare';emit();
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
  if(r.figure_path){await call('open_path',{path:r.figure_path});return;}
  const dir = S.batch.ranOpts?.output_dir || S.batch.opts.output_dir;
  const stem = rowKey(r).replace(/\.[^./\\]+$/, '').split(/[\\/]/).join('__');
  const guesses = r.detector_id ? [`${stem}__${r.detector_id}_fit.png`, `${stem}_fit.png`] : [`${stem}_fit.png`];
  const names = r.output_stem ? [`${r.output_stem}_fit.png`] : guesses;
  for (const name of names) {
    try { await call('open_path', { path: `${dir}\\${name}` }, { quiet: true }); return; } catch (e) { /* try the next candidate */ }
  }
  toast('Figure not found', 'err');
}

// Replays saved row IDs and frozen options; completed rows keep their archived outputs.
async function rerunFailed() {
  const B = S.batch,
    failed = (B.rows || []).filter((r) => needsBatchRetry(r) && r.row_id && r.manifest_path);
  if (!failed.length) return;
  B.rerunning = true;
  emit();
  try {
    const manifest = B.manifestPath || failed[0].manifest_path;
    B.job = await call(
      'batch_rerun',
      { manifest_path: manifest, rows: failed.map((r) => r.row_id) },
      { label: 'Rerunning saved batch rows' }
    );
    B.status = { state: 'running', done: 0, total: failed.length, log: [] };
    emit();
    do {
      await wait(700);
      B.status = await call('batch_status', { job: B.job }, { quiet: true });
      emit();
    } while (B.status.state === 'running');
    B.rows = B.status.rows;
    B.manifestPath = B.status.manifest_path;
    toast(
      `Re-run ${B.status.state}: ${B.status.done}/${B.status.total} requested rows completed.${B.status.error ? ' ' + B.status.error : ''}`,
      B.status.state === 'finished' ? 'ok' : ''
    );
  } finally {
    B.rerunning = false;
    emit();
  }
}

async function resumeUnfinished() {
  const B = S.batch;
  B.job = await call('batch_resume', { manifest_path: B.manifestPath }, { label: 'Resuming recorded batch' });
  B.status = { state: 'running', done: 0, total: 0, log: [] };
  if (timer) clearInterval(timer);
  timer = setInterval(poll, 700);
  emit();
}
