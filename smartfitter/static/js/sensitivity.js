// Leave-one-out diagnostics and explicit, recorded iteration exclusions.
import { call } from './api.js';
import { S, commit, emit, loadData } from './state.js';
import { h, mount, field, select, modal, toast, fmt, clone, check } from './util.js';
import { displayValue, displayUnit, parameterUnit } from './unitdisplay.js';
import { formatIterationRange, parseIterationRange } from './iterations.js';
import { baseStyle, baseLayout, axisLayout, plotConfig } from './plotstyle.js';

const specKey = (spec) => { const s = { ...spec }; delete s.custom_spec; return JSON.stringify(s); };
export function sensitivityStale(f, result = f.iterationSensitivity?.[f.detector]) {
  return !result || !!result.source_changed || result.detector !== f.detector || result.mode !== f.mode
    || JSON.stringify(result.applied_processing || result.processing) !== JSON.stringify(f.proc)
    || specKey(result.applied_spec || result.requested_spec) !== specKey(f.fitSpec);
}
export const sensitivityParameter = (result) => result.baseline.params.find((p) => /^(T1|T2|tau|f|omega)/i.test(p.name))?.name || result.baseline.params[0]?.name;
export function flaggedIterations(result, parameter, threshold = 10) {
  return sensitivityRows(result, parameter).slice(0, result.rows.length)
    .filter((r) => r.omitted.length === 1 && Number.isFinite(r.percent) && Math.abs(r.percent) >= threshold)
    .flatMap((r) => r.omitted);
}
export function sensitivityRows(result, parameter) {
  const baseline = result.baseline.params.find((p) => p.name === parameter);
  return [...result.rows, ...(result.group ? [result.group] : [])].map((row) => {
    const p = row.fit?.params?.find((p) => p.name === parameter);
    const comparable = row.comparable && p?.unit === baseline?.unit && Number.isFinite(p?.value) && Number.isFinite(baseline?.value);
    const delta = comparable ? p.value - baseline.value : null;
    return { ...row, parameter, baseline, parameter_fit: p, delta,
      percent: comparable && baseline.value !== 0 ? 100 * delta / Math.abs(baseline.value) : null };
  });
}
export function sensitivityCSV(f, result = f.iterationSensitivity?.[f.detector]) {
  if (!result) return '';
  const keys = ['omitted', 'source_iterations', 'retained', 'detector', 'observable', 'parameter', 'native_unit', 'baseline_value', 'baseline_error',
    'value', 'error', 'change', 'change_percent', 'r2', 'status', 'reason', 'stale', 'processing', 'fit_spec', 'exclusion_review'];
  const rows = result.baseline.params.flatMap((p) => sensitivityRows(result, p.name).map((r) => [
    formatIterationRange(r.omitted), JSON.stringify(r.sources), formatIterationRange(r.retained), result.detector, result.mode, p.name, p.unit,
    p.value, p.error, r.parameter_fit?.value, r.parameter_fit?.error, r.delta, r.percent, r.fit?.r2, r.status, r.reason,
    sensitivityStale(f, result), JSON.stringify(result.processing), JSON.stringify(result.fit_spec), JSON.stringify(result.review || null),
  ]));
  const cell = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
  return [keys.join(','), ...rows.map((r) => r.map(cell).join(','))].join('\r\n');
}
export function sensitivityFigure(f, result, parameter, { errorBars = true, revision = 0 } = {}) {
  const rows = sensitivityRows(result, parameter), base = result.baseline.params.find((p) => p.name === parameter);
  const unit = parameterUnit(base || {}, f.info), shown = displayUnit(base?.value, unit);
  const st = baseStyle({ title: `${parameter} after omission`, x: {}, y: {} });
  return { data: [{ type: 'scatter', mode: 'markers',
    x: rows.map((r, i) => `${i < result.rows.length ? 'Omit' : 'Group'} ${formatIterationRange(r.omitted)}`),
    y: rows.map((r) => r.comparable && Number.isFinite(r.parameter_fit?.value) ? r.parameter_fit.value * shown.scale : null),
    marker: { size: 9, color: '#2a78d6' }, error_y: { type: 'data',
      array: rows.map((r) => Number.isFinite(r.parameter_fit?.error) ? r.parameter_fit.error * shown.scale : null), visible: errorBars }, showlegend: false }],
    layout: baseLayout(st, { xaxis: { ...axisLayout(st, 'x', 'Omitted iteration(s)'), type: 'category' },
      yaxis: axisLayout(st, 'y', `${parameter} [${shown.unit || 'unitless'}]`),
      shapes: Number.isFinite(base?.value) ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: base.value * shown.scale,
        y1: base.value * shown.scale, line: { dash: 'dash', color: '#d56722' } }] : [],
      uirevision: `${parameter}:${errorBars}:${revision}`, margin: { l: 65, r: 20, b: 45, t: 35 }, showlegend: false }) };
}
function downloadCSV(f, result) {
  const a = h('a', { download: `${f.info.name.replace(/\.mat$/i, '')}_iteration_sensitivity.csv`,
    href: URL.createObjectURL(new Blob([sensitivityCSV(f, result)], { type: 'text/csv;charset=utf-8' })) });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export async function runSensitivity(f) {
  if (f.fitting || f.iterationFitting || f.sensitivityFitting) return;
  const { prepareFitSpec } = await import('./analyze.js');
  const args = { path: f.path, detector: f.detector, mode: f.mode, proc: clone(f.proc), spec: prepareFitSpec(f) };
  const sourceRevision = f.sourceRevision || 0;
  commit('iteration sensitivity'); f.sensitivityFitting = true; emit('insp');
  try {
    const result = await call('iteration_sensitivity', args, { label: 'Fitting full selection and each leave-one-out mean' });
    if (S.files[f.path] !== f) return;
    if ((f.sourceRevision || 0) !== sourceRevision) result.source_changed = true;
    (f.iterationSensitivity ||= {})[args.detector] = result;
    const previous = f.iterationView?.sensitivity || {};
    const parameter = result.baseline.params.some((p) => p.name === previous.parameter) ? previous.parameter : sensitivityParameter(result);
    const view = (f.iterationView ||= {}).sensitivity = { ...previous, threshold: previous.threshold ?? 10, parameter, omitted: [], reason: '' };
    view.omitted = flaggedIterations(result, parameter, view.threshold);
    result.flagged = [...view.omitted];
    f.sensitivityFitting = false;
    if (!sensitivityStale(f, result)) {
      if (view.omitted.length < result.selected.length) await applySensitivitySelection(f, result, view);
      else toast('Every iteration exceeded the threshold. No selection changed; raise the threshold or choose iterations to keep.', 'err');
    }
    emit();
    if (f.detector === args.detector) openSensitivity(f);
  } finally { f.sensitivityFitting = false; emit(); }
}
export function sensitivityActions(f) {
  const it = f.data[f.detector]?.iterations, result = f.iterationSensitivity?.[f.detector];
  if (!it) return null;
  return h('div', { class: 'sensitivity-actions' },
    h('div', { class: 'row wrap' },
      h('button', { class: 'btn', disabled: f.fitting || f.iterationFitting || f.sensitivityFitting || it.selected.length < 2,
        onclick: () => runSensitivity(f) }, f.sensitivityFitting ? 'Checking omissions…' : 'Check iteration sensitivity…'),
      result ? h('button', { class: 'btn', onclick: () => openSensitivity(f) }, 'Sensitivity results…') : null),
    field('Auto-exclude if parameter changes ≥ (%)', h('input', { class: 'field', type: 'number', min: 0, value: f.iterationView?.sensitivity?.threshold ?? 10,
      'aria-label': 'Auto exclusion threshold percent', oninput: (e) => { const v = Number(e.target.value); ((f.iterationView ||= {}).sensitivity ||= {}).threshold = Number.isFinite(v) && v >= 0 ? v : 10; } })),
    h('div', { class: 'note' }, `Fits the full selection and each omission, then automatically deselects flagged iterations and refits the main plot. Default parameter: T1 / T2 / timescale. Restore iterations in Process → Iterations used for mean + fit.${result && sensitivityStale(f) ? ' Saved sensitivity is STALE; rerun for current data/settings.' : ''}`));
}

export async function applySensitivitySelection(f, result, view) {
  if (f.fitting || f.iterationFitting || f.sensitivityFitting || sensitivityStale(f, result)) throw new Error('Rerun sensitivity for the current detector, observable and settings first.');
  const omitted = [...new Set(view.omitted || [])].sort((a, b) => a - b);
  const retained = result.selected.filter((i) => !omitted.includes(i));
  if (!retained.length || omitted.some((i) => !result.selected.includes(i))) throw new Error('Retain at least one baseline iteration.');
  const { runFit } = await import('./analyze.js');
  commit('exclude reviewed iterations and refit');
  const review = { baseline_selection: result.selected, omitted, retained, parameter: view.parameter, threshold_percent: view.threshold,
    sources: omitted.map((i) => result.sources[result.selected.indexOf(i)]), reason: view.reason?.trim() || `Sensitivity exclusion: |${view.parameter} change| ≥ ${view.threshold}% (user-adjustable rule).`, at: new Date().toISOString() };
  result.review = review;
  f.proc.iterations = retained;
  f.proc.iteration_review = clone(review);
  // Retain Auto choices used for the comparison in the final fit as well.
  Object.assign(f.fitSpec, result.locked);
  result.applied_processing = clone(f.proc);
  result.applied_spec = clone(f.fitSpec);
  if (Object.keys(f.fit).length) f.fitStale = true;
  f.iterationFitsStale = true;
  await loadData(f);
  await runFit(f, { recordUndo: false });
}

export function openSensitivity(f) {
  const result = f.iterationSensitivity?.[f.detector]; if (!result) return;
  const names = result.baseline.params.map((p) => p.name);
  const view = ((f.iterationView ||= {}).sensitivity ||= { threshold: 10, omitted: [], reason: '' });
  if (!names.includes(view.parameter)) view.parameter = sensitivityParameter(result);
  view.omitted = result.selected.filter((i) => !(f.data[f.detector]?.iterations?.selected || result.selected).includes(i));
  let busy = false, closed = false;
  const chart = h('div', { class: 'iteration-chart', 'aria-label': 'Omission sensitivity plot' });
  const body = h('div', { class: 'iteration-results' });
  const updateSelection = async (omitted) => {
    if (busy) return;
    const old = [...view.omitted]; busy = true; draw();
    try {
      await applySensitivitySelection(f, result, { ...view, omitted });
      view.omitted = [...omitted];
    } catch (e) { view.omitted = old; toast(e.message, 'err'); }
    finally { busy = false; if (!closed) draw(); emit(); }
  };
  const testGroup = async () => {
    if (busy || !view.omitted.length) return;
    busy = true; f.sensitivityFitting = true; draw();
    try {
      const out = await call('iteration_sensitivity', { path: f.path, detector: result.detector, mode: result.mode,
        proc: clone(result.processing), spec: clone(result.fit_spec), omit_groups: [view.omitted] }, { label: 'Testing combined omission' });
      commit('test combined iteration omission'); result.group = out.rows[0];
    } finally { busy = false; f.sensitivityFitting = false; if (!closed) draw(); emit('insp'); }
  };
  const draw = () => {
    if (closed) return;
    const stale = sensitivityStale(f, result), rows = sensitivityRows(result, view.parameter);
    const base = result.baseline.params.find((p) => p.name === view.parameter);
    const unit = parameterUnit(base || {}, f.info);
    const show = (p) => displayValue(p?.value, p?.error, unit, { precision: 3 }).withUnit;
    const percent = (v) => Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${fmt(v, 4)}%` : 'n/a';
    const range = h('input', { class: 'field', value: formatIterationRange(view.omitted), placeholder: '2,5-7', 'aria-label': 'Iterations to exclude' });
    const parse = () => {
      try {
        const omitted = parseIterationRange(range.value, f.data[f.detector]?.iterations?.count || Math.max(...result.selected) + 1);
        if (omitted.some((i) => !result.selected.includes(i))) throw new Error('Choose only iterations in the baseline selection.');
        updateSelection(omitted);
      } catch (e) { toast(e.message, 'err'); }
    };
    range.addEventListener('keydown', (e) => { if (e.key === 'Enter') parse(); });
    const threshold = h('input', { class: 'field', type: 'number', min: 0, step: 1, value: view.threshold, 'aria-label': 'Sensitivity threshold percent',
      disabled: busy || stale, oninput: () => { const v = Number(threshold.value); view.threshold = Number.isFinite(v) && v >= 0 ? v : 10; } });
    mount(body,
      h('div', { class: 'row sensitivity-heading' }, h('strong', { class: 'ellip', title: f.path }, `Iteration sensitivity · ${f.info.name} · ${result.detector}`),
        h('span', { class: 'sp' }), h('button', { class: 'btn', onclick: () => m.close() }, 'Close sensitivity')),
      h('div', { class: 'note' }, `Baseline: ${formatIterationRange(result.selected)} · ${result.baseline.model} · ${result.mode}. Main plot now uses ${formatIterationRange(f.data[f.detector]?.iterations?.selected || []) || 'none'}.${stale ? ' STALE settings: rerun before changing selection here.' : ''}`),
      h('div', { class: 'row wrap' }, field('Parameter', select(names, view.parameter, (p) => { view.parameter = p; draw(); })),
        field('Auto-exclude |change| ≥ (%)', threshold), h('button', { class: 'btn', disabled: busy || stale,
          onclick: () => updateSelection(flaggedIterations(result, view.parameter, view.threshold)) }, 'Apply threshold & refit'),
        check('Error bars', view.errorBars !== false, (v) => { view.errorBars = v; view.revision = (view.revision || 0) + 1; draw(); }),
        h('button', { class: 'btn', onclick: () => { view.revision = (view.revision || 0) + 1; draw(); } }, 'Rescale plot')),
      h('div', { class: 'note' }, `Full-selection ${view.parameter}: ${show(base)}. Dashed line = baseline. Checked rows are excluded from the MAIN plot and fit. Recheck iterations in Process to add them back.`),
      chart,
      h('div', { class: 'iteration-table-wrap' }, h('table', { class: 'tbl', 'aria-label': 'Iteration omission results' },
        h('thead', {}, h('tr', {}, ['Excluded', 'Iteration / source', view.parameter, 'Change', 'Change %', 'R²', 'Quality'].map((v) => h('th', {}, v)))),
        h('tbody', {}, rows.map((r, i) => h('tr', { title: r.reason },
          h('td', {}, i < result.rows.length && r.omitted.length === 1 ? h('input', { type: 'checkbox', checked: view.omitted.includes(r.omitted[0]),
            disabled: busy || stale, 'aria-label': `Exclude iteration ${r.omitted[0] + 1}`, onchange: (e) => updateSelection(e.target.checked
              ? [...view.omitted, r.omitted[0]] : view.omitted.filter((n) => n !== r.omitted[0])) }) : 'Group'),
          h('td', { class: 'sensitivity-source', title: r.sources.map((s) => `${s.path} #${s.iteration}`).join('\n') },
            i < result.rows.length ? `${formatIterationRange(r.omitted)} · acquisition #${r.sources[0]?.iteration}` : `Group ${formatIterationRange(r.omitted)} (${r.sources.length} traces)`),
          h('td', {}, show(r.parameter_fit)), h('td', {}, Number.isFinite(r.delta) ? displayValue(r.delta, null, unit, { precision: 3 }).withUnit : 'n/a'),
          h('td', {}, percent(r.percent)), h('td', {}, fmt(r.fit?.r2, 4)), h('td', {}, `${r.status || ''}${!r.comparable ? ' · incomparable' : Number.isFinite(r.percent) && Math.abs(r.percent) >= view.threshold ? ' · flagged' : ''}`)))))),
      h('div', { class: 'row wrap' }, field('Excluded iterations', range), h('button', { class: 'btn', disabled: busy || stale, onclick: parse }, 'Exclude listed & refit'),
        h('button', { class: 'btn', disabled: busy || stale || !view.omitted.length, onclick: testGroup }, busy ? 'Updating…' : 'Test combined omission'),
        h('button', { class: 'btn', disabled: busy || stale, onclick: () => updateSelection([]) }, 'Restore baseline iterations'),
        h('button', { class: 'btn', onclick: () => downloadCSV(f, result) }, 'Export sensitivity CSV')),
      h('details', {}, h('summary', {}, 'Method, sources and exclusion record'),
        h('div', { class: 'note' }, result.note),
        h('div', { class: 'note' }, 'Exclusions update the main selection and refit immediately. Test combined omission adds a comparison row for the already excluded group. Full source names are available on hover and in the CSV. Flags use the selected parameter’s percentage change; they are a sensitivity rule, not a statistical outlier test. % is undefined for a zero baseline.'),
        h('input', { class: 'field', value: view.reason || result.review?.reason || '', 'aria-label': 'Iteration exclusion reason', placeholder: 'Optional exclusion note',
          oninput: (e) => { view.reason = e.target.value; } })));
    const fig = sensitivityFigure(f, result, view.parameter, { errorBars: view.errorBars !== false, revision: view.revision || 0 });
    Plotly.react(chart, fig.data, fig.layout, plotConfig({ edits: {} }));
  };
  const m = modal(body, { className: 'iteration-results-modal sensitivity-modal', onClose: () => { closed = true; Plotly.purge(chart); emit('insp'); } });
  m.box.parentElement.classList.add('sensitivity-back');
  draw();
}
