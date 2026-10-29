import { maxAbs } from './numeric.js';
// Iteration controls, independent fits and source-preserving run pooling.
import { call } from './api.js';
import { S, emit, commit, loadData, openFile, active, syncOverlay } from './state.js';
import { h, mount, check, input, field, select, modal, toast, fmt, clone, basename, dirname } from './util.js';
import { displayValue, parameterUnit, displayUnit, unitOptions, unitScale } from './unitdisplay.js';
import { filePresentation } from './seriescontrol.js';
import { baseStyle, baseLayout, axisLayout, plotConfig } from './plotstyle.js';
import { sensitivityActions } from './sensitivity.js';
import { ITER_SCALES, DEFAULT_ITER_SCALE } from './plot2d.js';

export function parseIterationRange(text, count) {
  if (!text.trim()) return [];
  const selected = new Set();
  for (const part of text.split(',')) {
    const m = part.trim().match(/^(\d+)(?:\s*-\s*(\d+))?$/);
    if (!m) throw new Error('Use iteration numbers/ranges such as 1-5,8.');
    const lo = Number(m[1]), hi = Number(m[2] || m[1]);
    if (lo < 1 || hi < lo || hi > count) throw new Error(`Choose iteration numbers from 1 to ${count}.`);
    for (let n = lo; n <= hi; n++) selected.add(n - 1);
  }
  return [...selected].sort((a, b) => a - b);
}
export function formatIterationRange(indices) {
  const a = [...new Set(indices)].sort((x, y) => x - y), ranges = [];
  for (let i = 0; i < a.length; i++) {
    const start = a[i]; let end = start;
    while (a[i + 1] === end + 1) end = a[++i];
    ranges.push(start === end ? `${start + 1}` : `${start + 1}-${end + 1}`);
  }
  return ranges.join(',');
}
export const selectedIterations = (f) => f.data[f.detector]?.iterations?.selected ?? [];
export function fitTargetLabel(fit) {
  const p = fit?.provenance;
  if (!p) return '';
  return `${p.fit_target === 'smoothed' ? `SMOOTHED ${p.smoothing_window} pt` : 'Raw mean'}${p.iteration_count !== undefined && p.iteration_sources?.length ? ` · iterations ${formatIterationRange((p.selected_iterations || []).map((i) => i - 1))}` : ''}`;
}
export function iterationProvenance(f) {
  const d = f.data[f.detector], it = d?.iterations;
  if (!it) return '';
  const fit = f.fit[f.detector], target = fitTargetLabel(fit);
  return `${f.info.combined ? 'Pooled' : 'Selected'} mean: ${it.selected.length}/${it.count} iterations [${formatIterationRange(it.selected) || 'none'}] · SEM = sample SD/√n (n ≥ 2).${target ? ` Fit: ${target}.` : ''}${f.info.combined ? ` Grid: ${f.info.combined.grid}; ${f.info.combined.runs.length} runs; no interpolation.` : ''}`;
}
export async function setIterationSelection(f, selection) {
  if (f.fitting || f.iterationFitting || f.sensitivityFitting) { toast('Wait for the current fit before changing iterations.'); return; }
  commit('iteration selection');
  f.proc.iterations = selection;
  delete f.proc.iteration_review;
  if (Object.keys(f.fit).length) f.fitStale = true;
  f.iterationFitsStale = true;
  await loadData(f); emit();
}

function selectionControls(count, selection, onChange, prefix = 'Iteration', current = () => selection) {
  const all = Array.from({ length: count }, (_, i) => i);
  const sel = selection ?? all;
  const chosen = new Set(sel);
  const latest = () => current() ?? all;
  const range = h('input', { class: 'field', value: formatIterationRange(sel), placeholder: '1-5,8', 'aria-label': `${prefix} range` });
  const apply = () => { try { onChange(parseIterationRange(range.value, count)); } catch (e) { toast(e.message, 'err'); } };
  range.addEventListener('keydown', (e) => { if (e.key === 'Enter') apply(); });
  return h('div', { class: 'iteration-selection' },
    h('div', { class: 'row wrap' },
      h('button', { class: 'btn sm', onclick: () => onChange(null) }, 'Select all'),
      h('button', { class: 'btn sm', onclick: () => onChange([]) }, 'Select none'),
      h('button', { class: 'btn sm', onclick: () => onChange(all.filter((i) => !latest().includes(i))) }, 'Invert'),
      h('button', { class: 'btn sm', onclick: () => onChange([count - 1]) }, 'Latest')),
    h('div', { class: 'row' }, range, h('button', { class: 'btn sm', onclick: apply }, 'Apply range')),
    h('div', { class: 'iteration-checks', role: 'group', 'aria-label': `${prefix} checkboxes` }, Array.from({ length: count }, (_, i) =>
      check(String(i + 1), chosen.has(i), (v) => onChange(v ? [...new Set([...latest(), i])].sort((a, b) => a - b) : latest().filter((n) => n !== i)), { 'aria-label': `${prefix} ${i + 1}` }))));
}
export function iterationCard(f) {
  const it = f.data[f.detector]?.iterations;
  if (!it) return h('div', { class: 'note' }, 'No complete per-iteration matrix is available for this observable. The stored mean is used; no iterations are invented.');
  return h('details', { class: 'card pad iteration-card', open: f.iterationView?.selectionOpen !== false, ontoggle: (e) => { (f.iterationView ||= {}).selectionOpen = e.target.open; } },
    h('summary', {}, `Iterations used for mean + fit · ${it.selected.length}/${it.count}`),
    selectionControls(it.count, f.proc.iterations ?? null, (s) => setIterationSelection(f, s), 'Iteration', () => f.proc.iterations),
    h('div', { class: 'note' }, 'List numbers identify the available complete traces. Incomplete/failed acquisitions are excluded by the loader. None means no data, not all.'),
    h('div', { class: 'row wrap' }, check('Show iteration traces', f.layers.iter, (v) => { commit('iteration traces'); f.layers.iter = v; emit(); }),
      check('Mean ± SD', f.layers.meanSd, (v) => { commit('iteration spread'); f.layers.meanSd = v; emit(); }),
      h('label', { class: 'row', style: 'gap:4px', title: 'Colour iteration traces and individual fits by iteration number (early → late)' }, 'Colour',
        select([...Object.keys(ITER_SCALES), 'gray'], f.iterationView?.cmap === 'gray' || ITER_SCALES[f.iterationView?.cmap] ? f.iterationView.cmap : DEFAULT_ITER_SCALE,
          (v) => { commit('iteration colours'); (f.iterationView ||= {}).cmap = v; emit(); }, { style: 'width:92px;height:28px', 'aria-label': 'Iteration colour scale' }))),
    h('details', {}, h('summary', {}, 'Exclude-outlier help / source iterations'),
      h('div', { class: 'note' }, 'Sensitivity checks automatically uncheck iterations whose omission changes the selected parameter by at least the configured percentage, and refit the main plot. Recheck any iteration here to restore it to the mean; run a fit again to update the fit after a manual selection change. The sensitivity rule and excluded sources are recorded with the fit. Low R² or a large change alone does not prove an outlier.'),
      h('div', { class: 'iteration-source-list' }, it.sources.map((s, i) => h('div', { title: s.path }, `${i + 1} → ${basename(s.path)} · acquisition iteration ${s.iteration}`)))));
}
export async function runIterationFits(f) {
  if (f.iterationFitting || f.fitting || f.sensitivityFitting) return;
  commit('fit individual iterations');
  const { prepareFitSpec } = await import('./analyze.js');
  const spec = prepareFitSpec(f), proc = clone(f.proc), det = f.detector, mode = f.mode;
  const signature = JSON.stringify([mode, proc, spec]);
  f.iterationFitting = true; emit('insp');
  try {
    const result = await call('fit_iterations', { path: f.path, detector: det, mode, proc, spec }, { label: 'Fitting selected iterations' });
    (f.iterationFits ||= {})[det] = result;
    f.iterationFitsStale = signature !== JSON.stringify([f.mode, f.proc, prepareFitSpec(f)]);
    f.iterationView ||= {}; f.iterationView.parameter ||= result.rows.find((r) => r.fit)?.fit.params.find((p) => /^(T1|T2|tau|f|omega)/i.test(p.name))?.name || result.rows.find((r) => r.fit)?.fit.params[0]?.name;
    emit(); openIterationResults(f);
  } finally { f.iterationFitting = false; emit(); }
}
export function iterationFitActions(f) {
  const it = f.data[f.detector]?.iterations, result = f.iterationFits?.[f.detector];
  if (!it) return null;
  return h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' },
    h('strong', {}, `Individual iterations · ${it.selected.length} selected`),
    h('div', { class: 'row wrap' }, h('button', { class: 'btn primary', disabled: f.fitting || f.iterationFitting || f.sensitivityFitting || !it.selected.length, onclick: () => runIterationFits(f) }, f.iterationFitting ? 'Fitting iterations…' : 'Fit each selected iteration'),
      result ? h('button', { class: 'btn', onclick: () => openIterationResults(f) }, 'Iteration results…') : null),
    result ? check('Overlay individual fit curves', f.layers.iterFits, (v) => { commit('iteration fit overlays'); f.layers.iterFits = v; emit(); }) : null,
    result && f.iterationFitsStale ? h('div', { class: 'note' }, 'Individual fits are from earlier processing/settings. Refit to update.') : null,
    h('div', { class: 'note' }, 'Uses the current model, range, seeds/bounds and smoothing setting. Single iterations have no across-iteration SEM.'),
    sensitivityActions(f));
}
export function iterationRows(f, result = f.iterationFits?.[f.detector]) {
  const cfg = filePresentation(f, f);
  return (result?.rows || []).flatMap((r) => (r.fit?.params?.length ? r.fit.params : [{}]).map((p) => {
    const unit = parameterUnit(p, f.info), d = displayValue(p.value, p.error, unit, cfg);
    return { iteration: r.iteration, source: r.source?.path, source_iteration: r.source?.iteration, detector: result.detector, observable: result.mode,
      parameter: p.name || '', value: d.value, error: d.error, unit: d.unit, formatted: d.text, native_value: p.value, native_error: p.error, native_unit: unit,
      r2: r.fit?.r2, status: r.status, reason: r.reason, fit_target: r.fit?.provenance?.fit_target, smoothing_window: r.fit?.provenance?.smoothing_window,
      stale: !!f.iterationFitsStale };
  }));
}
export function iterationCSV(f) {
  const keys = ['iteration', 'source', 'source_iteration', 'detector', 'observable', 'parameter', 'value', 'error', 'unit', 'native_value', 'native_error', 'native_unit', 'r2', 'status', 'reason', 'fit_target', 'smoothing_window', 'stale'];
  const cell = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
  return [keys.join(','), ...iterationRows(f).map((r) => keys.map((k) => cell(r[k])).join(','))].join('\r\n');
}
export function downloadIterationCSV(f) {
  const a = h('a', { download: `${f.info.name.replace(/\.mat$/i, '')}_iterations.csv`, href: URL.createObjectURL(new Blob([iterationCSV(f)], { type: 'text/csv;charset=utf-8' })) });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
export function iterationFigure(f) {
  const result = f.iterationFits?.[f.detector], parameter = f.iterationView?.parameter;
  const rows = (result?.rows || []).filter((r) => r.fit?.params.some((p) => p.name === parameter));
  const unit = parameterUnit(rows[0]?.fit.params.find((p) => p.name === parameter) || {}, f.info);
  const max = maxAbs(rows.map((r) => r.fit.params.find((p) => p.name === parameter).value));
  const chosen = f.iterationView?.unit || 'auto', shownUnit = chosen === 'auto' ? displayUnit(max, unit).unit : chosen === 'native' ? unit : chosen;
  const scale = unitScale(unit, shownUnit) ?? 1;
  const st = baseStyle({ ...f.style, title: `${parameter || 'Parameter'} vs iteration · ${result?.mode || f.mode}`, x: {}, y: {} });
  const p = (r) => r.fit.params.find((v) => v.name === parameter);
  return { data: [{ type: 'scatter', mode: 'lines+markers', x: rows.map((r) => r.iteration), y: rows.map((r) => p(r).value * scale),
    error_y: { type: 'data', array: rows.map((r) => Number.isFinite(p(r).error) ? p(r).error * scale : NaN), visible: true, thickness: 1, width: 3 },
    line: { color: '#2a78d6', width: 2 }, marker: { color: '#2a78d6', size: 9, symbol: rows.map((r) => r.status === 'PASS' ? 'circle' : r.status === 'WARN' ? 'diamond' : 'x'), line: { color: '#FFFFFF', width: 2 } },
    customdata: rows.map((r) => `${basename(r.source?.path || '')} · acquisition ${r.source?.iteration} · ${r.status} · R² ${fmt(r.fit.r2, 4)} · ${fitTargetLabel(r.fit)}`),
    hovertemplate: `Iteration %{x}<br>${parameter} = %{y:.6g} ${shownUnit}<br>%{customdata}<extra></extra>`, showlegend: false }],
    layout: baseLayout(st, { xaxis: { ...axisLayout(st, 'x', 'Iteration (available-trace list number)'), dtick: 1 }, yaxis: axisLayout(st, 'y', `${parameter || 'Parameter'}${shownUnit ? ` [${shownUnit}]` : ''}`),
      showlegend: false, hovermode: 'closest', margin: { l: 80, r: 24, b: 65, t: 55 } }) };
}
export function openIterationResults(f) {
  const result = f.iterationFits?.[f.detector]; if (!result) return;
  const names = [...new Set(result.rows.flatMap((r) => (r.fit?.params || []).map((p) => p.name)))];
  f.iterationView ||= {}; if (!names.includes(f.iterationView.parameter)) f.iterationView.parameter = names.find((n) => /^(T1|T2|tau|f|omega)/i.test(n)) || names[0];
  const chart = h('div', { class: 'iteration-chart', 'aria-label': 'Parameter versus iteration plot' });
  const controls = h('div', { class: 'row wrap' });
  const draw = () => {
    const p = result.rows.find((r) => r.fit?.params.some((p) => p.name === f.iterationView.parameter))?.fit.params.find((p) => p.name === f.iterationView.parameter);
    mount(controls, field('Parameter', select(names, f.iterationView.parameter, (parameter) => { f.iterationView.parameter = parameter; f.iterationView.unit = 'auto'; draw(); })),
      field('Trend unit', select(unitOptions(parameterUnit(p || {}, f.info)), f.iterationView.unit || 'auto', (unit) => { f.iterationView.unit = unit; draw(); })));
    const fig = iterationFigure(f); Plotly.react(chart, fig.data, fig.layout, { ...plotConfig(), displayModeBar: true });
  };
  const m = modal(h('div', { class: 'iteration-results' },
    h('div', { class: 'row' }, h('strong', {}, `Individual fits · ${f.info.name} · ${result.detector}`), h('span', { class: 'sp' }), h('button', { class: 'btn', onclick: () => m.close() }, 'Close iteration results')),
    h('div', { class: 'note' }, `${result.rows.length} selected fits · ${fitTargetLabel(result.rows.find((r) => r.fit)?.fit)}${f.iterationFitsStale ? ' · STALE settings' : ''}. ${result.note}`),
    result.fit_spec.fit_smoothed ? h('div', { class: 'note' }, 'SMOOTHED fits: correlated samples underestimate ordinary fit uncertainties. No corrected error is supplied.') : null,
    controls, chart,
    h('div', { class: 'row wrap' }, h('button', { class: 'btn', onclick: () => downloadIterationCSV(f) }, 'Export iteration CSV'),
      check('Overlay individual fit curves', f.layers.iterFits, (v) => { commit('iteration fit overlays'); f.layers.iterFits = v; emit(); })),
    h('div', { class: 'iteration-table-wrap' }, h('table', { class: 'tbl', 'aria-label': 'Per-iteration results' },
      h('thead', {}, h('tr', {}, ['Iteration / source', ...names, 'R²', 'Quality'].map((v) => h('th', {}, v)))),
      h('tbody', {}, result.rows.map((r) => h('tr', { title: r.reason }, h('td', { class: 'txt' }, `${r.iteration} · ${basename(r.source?.path || '')} #${r.source?.iteration}`),
        names.map((name) => { const p = r.fit?.params.find((p) => p.name === name); return h('td', {}, p ? displayValue(p.value, p.error, parameterUnit(p, f.info), filePresentation(f, f)).withUnit : '—'); }),
        h('td', {}, fmt(r.fit?.r2, 4)), h('td', {}, `${r.status}${r.fit ? '' : ` · ${r.reason}`}`))))))),
    { className: 'iteration-results-modal', onClose: () => { Plotly.purge(chart); emit('insp'); } });
  requestAnimationFrame(draw);
}

export async function openCombineDialog(f = active()) {
  if (!f) return;
  // Start from what is open: this file, or the source runs (and their iteration choices) of a combined dataset,
  // so adding one more run to a pooled dataset only needs one more click.
  const seeds = f.combinedRecipe?.runs?.length ? clone(f.combinedRecipe.runs)
    : f.path.startsWith('combined:') ? [] : [{ path: f.path, iterations: clone(f.proc.iterations ?? null) }];
  const pinned = new Set(seeds.map((r) => r.path));
  const candidates = new Map(), chosen = new Map();
  let busy = false;
  const list = h('div', { class: 'combine-list' }), message = h('div', { class: 'note', role: 'status' });
  const name = h('input', { class: 'field', value: f.combinedRecipe?.name || `Combined ${f.info.type} iterations`, 'aria-label': 'Combined dataset name' });
  const grid = select([['identical', 'Identical x grids (required by default)'], ['intersect', 'Exact intersection (drop unmatched x; no interpolation)']], f.combinedRecipe?.grid || 'identical', () => {}, { 'aria-label': 'X grid compatibility' });
  const search = h('input', { class: 'field', placeholder: 'Search run files', 'aria-label': 'Search runs to combine' });
  const choose = async (path, enabled, iterations) => {
    if (!enabled) { chosen.delete(path); draw(); return; }
    try {
      const o = await openFile(path, { primary: false });
      await syncOverlay(o, f);
      if (!o.data[o.detector]?.iterations) throw new Error('No complete iteration matrix.');
      chosen.set(path, { path, iterations: clone(iterations !== undefined ? iterations : o.proc.iterations ?? null) }); message.textContent = ''; draw();
    } catch (e) { message.textContent = `${basename(path)}: ${e.message}`; draw(); }
  };
  const when = (t) => (Number.isFinite(t) ? new Date(t * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '');
  const draw = () => {
    const q = search.value.toLowerCase();
    // Current runs first, then newest acquisition first (file time, then name) — the same order as the library.
    const rows = [...candidates].filter(([path]) => path.toLowerCase().includes(q)).sort(([a, x], [b, y]) =>
      (pinned.has(b) - pinned.has(a)) || ((y.mtime ?? 0) - (x.mtime ?? 0)) || (x.name || '').localeCompare(y.name || ''));
    const names = new Map(); for (const [, i] of candidates) names.set(i.name, (names.get(i.name) || 0) + 1);
    mount(list, rows.map(([path, info]) => {
      const run = chosen.get(path), o = S.files[path], it = o?.data[o.detector]?.iterations, label = info.name || basename(path);
      const meta = [pinned.has(path) ? (f.combinedRecipe ? 'in this combined dataset' : 'current file') : '', when(info.mtime),
        info.iterations && !run ? `${info.iterations} iterations acquired` : '', names.get(info.name) > 1 ? basename(dirname(path)) : ''].filter(Boolean).join(' · ');
      return h('div', { class: 'card pad', title: path }, check(label, !!run, (v) => choose(path, v), { disabled: busy, 'aria-label': `Combine ${label}` }),
        h('div', { class: 'note', style: 'margin:2px 0 0 24px' }, meta),
        run && it ? h('details', {}, h('summary', {}, `${run.iterations?.length ?? it.count}/${it.count} iterations · choose for this run`),
          selectionControls(it.count, run.iterations, (iterations) => { run.iterations = iterations; draw(); }, `${info.name} iteration`)) : null);
    }));
    const n = chosen.size;
    create.disabled = busy || n < 2;
    create.textContent = n >= 2 ? `Combine ${n} runs` : 'Create combined dataset';
    hint.textContent = n >= 2 ? '' : n === 1 ? 'Tick at least one more run to pool with it.' : 'Tick two or more runs to pool.';
  };
  const add = async (paths) => {
    paths = paths.filter((p) => !p.startsWith('combined:') && !candidates.has(p));
    if (!paths.length) return;
    const infos = await call('peek', { paths }, { label: 'Reading run candidates' });
    infos.filter((i) => !i.error && i.type === f.info.type && i.dim !== 'scan2d').forEach((i) => candidates.set(i.path, i)); draw();
  };
  const hint = h('span', { class: 'note', style: 'margin:0' });
  const create = h('button', { class: 'btn primary', onclick: async () => {
    if (busy || chosen.size < 2) return;
    busy = true; create.disabled = true;
    try {
      const res = await call('combine_iterations', { runs: [...chosen.values()], grid: grid.value, name: name.value, mode: f.mode }, { label: 'Checking and pooling iterations' });
      commit('combine runs'); const combined = await openFile(res.path); combined.combinedRecipe = res.recipe;
      combined.fitSpec = clone(f.fitSpec); combined.mode = f.mode; await loadData(combined);
      combined.expDir = f.expDir || `${dirname(f.combinedRecipe?.runs?.[0]?.path || f.path)}\\SmartFitter results`;
      combined.expStem = name.value.replace(/[^\w .-]/g, '_');
      // Keep individual runs in Job A's independent Series controls.
      S.overlays = [...chosen.keys()];
      S.overlays.forEach((p) => { if (S.overlaySlots[p] === undefined) S.overlaySlots[p] = Object.keys(S.overlaySlots).length; });
      if (!S.library.extra.includes(res.path)) S.library.extra.push(res.path);
      m.close(); emit();
    } catch (e) { message.textContent = e.message; }
    finally { busy = false; create.disabled = false; draw(); }
  } }, 'Create combined dataset');
  const m = modal(h('div', { class: 'combine-dialog' },
    h('div', { class: 'row' }, h('strong', {}, `Combine ${f.info.type} iterations`), h('span', { class: 'sp' }), h('button', { class: 'btn', onclick: () => m.close() }, 'Cancel')),
    h('div', { class: 'note' }, 'Pooled mean ± SEM across selected iterations, not a mean of run means. Source processing is not copied; the combined dataset starts unbinned. Detector layout, observable and native units must match.'),
    field('Name', name), field('X grid', grid), search, list,
    h('div', { class: 'note' }, 'A checkpoint repeats the first iterations of its final file: runs sharing identical iterations are refused rather than counted twice, so pick the final file (or deselect the shared iterations). SEM assumes independent iterations. Acquisition files are never modified.'),
    message, h('div', { class: 'row wrap' }, h('button', { class: 'btn', onclick: async () => {
      const paths = await call('dialog', { kind: 'open_files', title: 'Choose runs to combine', directory: S.library.folder || dirname(f.path) }); if (paths?.length) await add(paths);
    } }, 'Browse runs…'), h('span', { class: 'sp' }), hint, create)), { className: 'combine-modal' });
  search.addEventListener('input', draw);
  // Preselect before reading the (possibly large) library folder, so the current runs are ticked at once.
  await add(seeds.map((r) => r.path));
  for (const r of seeds) await choose(r.path, true, r.iterations);
  await add([...new Set([...Object.keys(S.files), ...(S.library.listing?.files || []).map((r) => r.path)])]);
}
