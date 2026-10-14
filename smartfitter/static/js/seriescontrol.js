// Main Analyze comparison controls; presentation belongs to the primary plot, not the source files.
import { S, emit, commit, detLabel, syncOverlay, removeOverlay } from './state.js';
import { h, field, input, select, check, clone, toast } from './util.js';
import { PALETTES, baseStyle, resultLines } from './plotstyle.js';
import { displayValue, unitOptions, unitScale, parameterUnit } from './unitdisplay.js';
import { openOverlayPicker } from './overlaypicker.js';
import { copyText } from './export.js';

export const fileKeys = (f, path) => path === f.path ? ['data', 'fit'] : [`ov:${path}`, `ovfit:${path}`];
export const seriesDetector = (f, o) => o.info.detectors.includes(f.detector) ? f.detector : o.detector;
export function comparisonFiles(f) {
  const paths = [f.path, ...S.overlays.filter((p) => p !== f.path && S.files[p])];
  const order = f.style.fileOrder || [], fallback = new Map(paths.map((p, i) => [p, i]));
  return paths.sort((a, b) => (order.includes(a) ? order.indexOf(a) : order.length + fallback.get(a)) - (order.includes(b) ? order.indexOf(b) : order.length + fallback.get(b))).map((p) => p === f.path ? f : S.files[p]);
}
export function filePresentation(f, o) {
  const st = baseStyle(f.style), primary = o.path === f.path;
  const slot = primary ? 0 : (S.overlaySlots[o.path] ?? (S.overlaySlots[o.path] = Object.keys(S.overlaySlots).length));
  const pal = PALETTES[st.palette] || PALETTES.lab;
  const colors = st.palette === 'lab' ? ['#1BAF7A', pal[1], ...pal.slice(3)] : pal.slice(1);
  const [dk, fk] = fileKeys(f, o.path), ds = st.series[dk] || {}, fs = st.series[fk] || {};
  const cfg = st.comparison?.[o.path] || {};
  return { label: o.info.name.replace(/\.mat$/i, ''), color: primary ? pal[0] : (colors[slot] || '#6B7380'),
    marker: primary ? 'circle' : ['diamond', 'square', 'triangle-up', 'cross', 'x', 'triangle-down'][slot % 6],
    mode: f.styleMode === 'pts' ? 'markers' : f.styleMode === 'line' ? 'lines' : 'lines+markers',
    dash: primary ? 'solid' : ['solid', 'dot', 'dashdot', 'longdash'][Math.floor(slot / colors.length) % 4], fitDash: 'dash',
    ...cfg, data: (primary ? f.layers.data : true) && ds.visible !== false, fit: (primary ? f.layers.fit : f.display?.overlayFits !== false) && fs.visible !== false,
    legend: ds.legend !== false, result: cfg.result !== false, precision: cfg.precision || 2, units: cfg.units || 'auto',
  };
}
export function legendLabel(f, o) {
  const cfg = filePresentation(f, o), fit = o.fit[seriesDetector(f, o)];
  const lines = resultLines(fit, o.info);
  const keys = cfg.params ?? lines.filter((l) => !l.defaultOff && l.key !== 'r2').slice(0, o.info.type === 'ODMR' ? 2 : 1).map((l) => l.key);
  const results = cfg.result ? keys.map((k) => lines.find((l) => l.key === k)).filter(Boolean).map((l) => `${l.label} = ${displayValue(l.value, l.err, l.unit, cfg).withUnit}`).join('; ') : '';
  const target = fit?.provenance?.fit_target === 'smoothed' ? ` [SMOOTHED ${fit.provenance.smoothing_window} pt]` : '';
  return (cfg.template || '{label}{results}').replaceAll('{label}', cfg.label).replaceAll('{results}', results ? ` — ${results}` : '').replaceAll('{model}', fit?.model || '').replaceAll('{detector}', detLabel(o, seriesDetector(f, o))) + target;
}
export function updateFilePresentation(f, o, patch) {
  commit('series presentation');
  f.style.comparison = f.style.comparison || {};
  f.style.comparison[o.path] = { ...(f.style.comparison[o.path] || {}), ...patch };
  f.style.series ||= {};
  const [dk, fk] = fileKeys(f, o.path);
  if (patch.label !== undefined || patch.template !== undefined || patch.params !== undefined || patch.result !== undefined || patch.units !== undefined || patch.precision !== undefined) {
    for (const key of [dk, fk]) if (f.style.series[key]) delete f.style.series[key].name;
  }
  for (const key of [dk, fk]) {
    if (patch.color) f.style.series[key] = { ...(f.style.series[key] || {}), color: patch.color };
  }
  for (const key of ['mode', 'marker', 'dash']) if (patch[key]) f.style.series[dk] = { ...(f.style.series[dk] || {}), [key]: patch[key] };
  if (patch.fitDash) f.style.series[fk] = { ...(f.style.series[fk] || {}), dash: patch.fitDash };
  emit();
}
export function setFileVisible(f, o, which, value) {
  commit(`${which} visibility`);
  const key = fileKeys(f, o.path)[which === 'data' ? 0 : 1];
  f.style.series = f.style.series || {};
  f.style.series[key] = { ...(f.style.series[key] || {}), visible: value };
  if (o === f) f.layers[which] = value;
  if (which === 'fit' && o !== f && value) f.display.overlayFits = true;
  emit();
}
export async function fitSeries(f, o, { recordUndo = true } = {}) {
  if (recordUndo) commit(`fit ${o.info.name}`);
  if (o !== f) {
    await syncOverlay(o, f);
    o.detector = seriesDetector(f, o);
    if (filePresentation(f, o).fitSettings !== 'own') {
      if (unitScale(f.info.x_unit || '', o.info.x_unit || '') !== 1) throw new Error('Native x units differ. Choose this file’s own fit settings; seeds and bounds stay in native units.');
      o.fitSpec = clone(f.fitSpec);
    }
  }
  const { runFit } = await import('./analyze.js');
  await runFit(o, { recordUndo: false, robust: !!o.fitSpec.robust });
  const key = fileKeys(f, o.path)[1];
  f.style.series = f.style.series || {};
  f.style.series[key] = { ...(f.style.series[key] || {}), visible: true };
  if (o !== f) f.display.overlayFits = true;
  emit();
}
export async function fitAllSeries(f) {
  commit('fit all compared files'); f.compareFitting = true; emit();
  try {
    for (const o of comparisonFiles(f)) {
      try { await fitSeries(f, o, { recordUndo: false }); }
      catch (e) { toast(`Fit failed for ${o.info.name}: ${e.message}`, 'err'); }
    }
  } finally { f.compareFitting = false; emit(); }
}
// v1 "Make primary": swap the primary with a compared file, keeping the comparison set and every per-file presentation
// (labels, colours, visibility, order). Series keys differ for the primary (data/fit) and comparisons (ov:/ovfit:).
export async function makePrimary(f, o) {
  if (o === f) return;
  commit('make primary');
  const style = clone(f.style), series = style.series || {}, swap = {};
  for (const [k, v] of Object.entries(series)) {
    if (k === 'data') swap[`ov:${f.path}`] = v;
    else if (k === 'fit') swap[`ovfit:${f.path}`] = v;
    else if (k === `ov:${o.path}`) swap.data = v;
    else if (k === `ovfit:${o.path}`) swap.fit = v;
    else if (k.startsWith('ov:') || k.startsWith('ovfit:')) swap[k] = v;
  }
  o.style = { ...o.style, comparison: style.comparison, fileOrder: style.fileOrder, series: { ...(o.style.series || {}), ...swap } };
  S.overlays = [f.path, ...S.overlays.filter((p) => p !== o.path && p !== f.path)];
  const { openFile } = await import('./state.js');
  await openFile(o.path, { primary: true });
  toast(`${o.info.name} is now the primary file; ${f.info.name} is a comparison.`);
}
export function moveFile(f, o, direction) {
  const order = comparisonFiles(f).map((x) => x.path), i = order.indexOf(o.path), j = i + direction;
  if (j < 0 || j >= order.length) return;
  commit('series order'); [order[i], order[j]] = [order[j], order[i]]; f.style.fileOrder = order; delete f.style.seriesOrder; emit();
}

export function seriesOptions(f, o) {
  const cfg = filePresentation(f, o), set = (patch) => updateFilePresentation(f, o, patch);
  const fit = o.fit[seriesDetector(f, o)], lines = resultLines(fit, o.info);
  const selected = cfg.params ?? lines.filter((l) => !l.defaultOff && l.key !== 'r2').slice(0, o.info.type === 'ODMR' ? 2 : 1).map((l) => l.key);
  const resultUnits = [['auto', 'Auto'], ['native', 'Native (per parameter)'], ...new Map((lines.length ? lines : [{ unit: o.info.x_unit }]).flatMap((l) => unitOptions(l.unit)).filter(([u]) => u !== 'auto' && u !== 'native').map((v) => [v[0], v])).values()];
  return h('div', { class: 'series-options' },
    h('div', { class: 'g2' }, field('Data style', select([['lines+markers', 'Line + points'], ['markers', 'Points'], ['lines', 'Line']], cfg.mode, (mode) => set({ mode }))),
      field('Marker', select(['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x'], cfg.marker, (marker) => set({ marker })))),
    h('div', { class: 'g2' }, field('Data line', select(['solid', 'dash', 'dot', 'dashdot', 'longdash'], cfg.dash, (dash) => set({ dash }))),
      field('Fit line', select(['solid', 'dash', 'dot', 'dashdot', 'longdash'], cfg.fitDash, (fitDash) => set({ fitDash })))),
    o !== f ? h('div', { class: 'g2' },
      field('Y axis', select([['left', 'Left (shared)'], ['right', 'Right (own scale)']], cfg.axis || 'left', (axis) => set({ axis }), { title: 'Right: this series gets an independent right-hand y axis' })),
      field('Uncertainty', select([['none', 'None'], ['band', '±1 SEM band'], ['bars', '±1 SEM bars']], cfg.err || 'none', (err) => set({ err }), { title: 'Per-series uncertainty display (this file\'s own SEM)' }))) : null,
    o !== f && cfg.axis === 'right' ? field('Right axis title', input(f.style.yRight?.title || '', (v) => { commit('right axis title'); f.style.yRight = { ...(f.style.yRight || {}), title: v || undefined }; emit(); }, { placeholder: 'auto (or double-click it on the plot)' })) : null,
    field('Fit settings', select([['current', 'Current primary settings'], ['own', 'This file’s own settings']], cfg.fitSettings || 'current', (fitSettings) => set({ fitSettings }))),
    o !== f ? h('button', { class: 'btn sm', onclick: () => { set({ fitSettings: 'own' }); import('./analyze.js').then((m) => m.openFitSettingsDialog(o)); } }, 'Edit this file’s fit settings…') : null,
    h('div', { class: 'row wrap' }, check('In legend', cfg.legend, (v) => { commit('legend visibility'); const key = fileKeys(f, o.path)[0]; f.style.series ||= {}; f.style.series[key] = { ...(f.style.series[key] || {}), legend: v }; emit(); }),
      check('Fit results in legend', cfg.result, (result) => set({ result }))),
    field('Legend template', input(cfg.template || '{label}{results}', (template) => set({ template }), { title: '{label}, {results}, {model}, {detector}' })),
    h('div', { class: 'g2' }, field('Precision (significant error digits)', input(cfg.precision, (v) => set({ precision: Math.max(1, Math.min(8, Number(v) || 2)) }))),
      field('Result units', select(resultUnits, cfg.units, (units) => set({ units })))),
    lines.length ? field('Legend parameters', h('div', { class: 'row wrap' }, lines.map((l) => check(l.label, selected.includes(l.key), (v) => set({ params: v ? [...selected, l.key] : selected.filter((k) => k !== l.key) }))))) : h('div', { class: 'note' }, 'Fit first to select result parameters. Display overrides never change fit values.'));
}

export function comparisonRows(f) {
  return comparisonFiles(f).flatMap((o) => {
    const det = seriesDetector(f, o), fit = o.fit[det], cfg = filePresentation(f, o);
    const params = fit?.params || [];
    return params.length ? params.map((p) => {
      const unit = parameterUnit(p, o.info), d = displayValue(p.value, p.error, unit, cfg);
      return { file: o.info.name, path: o.path, label: cfg.label, detector: detLabel(o, det), model: fit.model, parameter: p.name,
        value: d.value, error: d.error, unit: d.unit, formatted: d.text, native_value: p.value, native_error: p.error, native_unit: unit, status: fit.status, stale: !!o.fitStale,
        fit_target: fit.provenance?.fit_target, smoothing_window: fit.provenance?.smoothing_window, iterations: (fit.provenance?.selected_iterations || []).join(';'), sources: JSON.stringify(fit.provenance?.iteration_sources || []) };
    }) : [{ file: o.info.name, path: o.path, label: cfg.label, detector: detLabel(o, det), status: 'Not fit' }];
  });
}
export function comparisonCSV(f, delimiter = ',') {
  const keys = ['file', 'label', 'detector', 'model', 'parameter', 'value', 'error', 'unit', 'formatted', 'native_value', 'native_error', 'native_unit', 'status', 'stale', 'path', 'fit_target', 'smoothing_window', 'iterations', 'sources'];
  const cell = (v) => { const s = String(v ?? ''); return /["\n\r,\t]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s; };
  return [keys.join(delimiter), ...comparisonRows(f).map((r) => keys.map((k) => cell(r[k])).join(delimiter))].join('\r\n');
}
export function downloadComparison(f) {
  const a = h('a', { download: `${f.info.name.replace(/\.mat$/i, '')}_comparison.csv`, href: URL.createObjectURL(new Blob([comparisonCSV(f)], { type: 'text/csv;charset=utf-8' })) });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
export function seriesPanel(f) {
  const files = comparisonFiles(f);
  // Collapsed while only the primary is shown (opens automatically once comparisons are added); the user's choice sticks.
  const open = f.display?.seriesPanel ?? files.length > 1;
  return h('details', { class: 'series-panel', open, 'aria-label': 'Series panel', ontoggle: (e) => { if (e.target.open !== open) f.display.seriesPanel = e.target.open; } },
    h('summary', {}, files.length > 1 ? `Series (${files.length}) · data / fit / style / legend` : 'Series · style & legend for this file · add comparisons'),
    h('div', { class: 'row wrap', style: 'padding:6px 0' },
      h('button', { class: 'btn sm', onclick: () => openOverlayPicker(f) }, 'Add comparison…'),
      h('button', { class: 'btn sm primary', disabled: f.compareFitting || files.some((o) => o.fitting), onclick: () => fitAllSeries(f) }, f.compareFitting ? 'Fitting…' : 'Fit all compared files'),
      h('button', { class: 'btn sm', onclick: () => downloadComparison(f) }, 'Export table CSV'),
      h('button', { class: 'btn sm', onclick: () => copyText(comparisonCSV(f, '\t'), 'Comparison results copied') }, 'Copy table')),
    h('div', { class: 'series-rows' }, files.map((o, i) => {
      const cfg = filePresentation(f, o), fit = o.fit[seriesDetector(f, o)];
      // Drag the ⋮⋮ grip onto another row to reorder (same as ↑/↓; legend and draw order follow).
      return h('div', { class: 'file-series', 'data-path': o.path,
        ondragover: (e) => { if (e.dataTransfer.types.includes('application/x-sf-series')) { e.preventDefault(); e.currentTarget.classList.add('drop-row'); } },
        ondragleave: (e) => e.currentTarget.classList.remove('drop-row'),
        ondrop: (e) => {
          const from = e.dataTransfer.getData('application/x-sf-series');
          e.currentTarget.classList.remove('drop-row');
          if (!from || from === o.path) return;
          e.preventDefault(); e.stopPropagation();
          const order = comparisonFiles(f).map((x) => x.path).filter((p) => p !== from);
          order.splice(order.indexOf(o.path), 0, from);
          commit('series order'); f.style.fileOrder = order; delete f.style.seriesOrder; emit();
        } },
        h('div', { class: 'file-series-row' },
          h('span', { class: 'drag-grip', draggable: 'true', title: 'Drag to reorder', 'aria-label': `Reorder ${o.info.name}`,
            ondragstart: (e) => { e.dataTransfer.setData('application/x-sf-series', o.path); e.dataTransfer.effectAllowed = 'move'; } }, '⋮⋮'),
          h('input', { type: 'color', value: cfg.color, 'aria-label': `Colour ${o.info.name}`, onchange: (e) => updateFilePresentation(f, o, { color: e.target.value }) }),
          input(cfg.label, (label) => updateFilePresentation(f, o, { label }), { 'aria-label': `Legend label ${o.info.name}`, title: `${o.path}${o === f ? ' · primary' : ''}` }),
          check('Data', cfg.data, (v) => setFileVisible(f, o, 'data', v), { 'aria-label': `Data ${o.info.name}` }),
          check('Fit', cfg.fit, (v) => setFileVisible(f, o, 'fit', v), { disabled: !fit, 'aria-label': `Fit ${o.info.name}` }),
          h('button', { class: 'btn sm', disabled: o.fitting || f.compareFitting, onclick: () => fitSeries(f, o) }, o.fitting ? 'Fitting…' : fit ? 'Refit' : 'Fit'),
          h('button', { class: 'btn sm', disabled: i === 0, title: 'Move up (legend/draw order)', 'aria-label': `Move up ${o.info.name}`, onclick: () => moveFile(f, o, -1) }, '↑'),
          h('button', { class: 'btn sm', disabled: i === files.length - 1, title: 'Move down (legend/draw order)', 'aria-label': `Move down ${o.info.name}`, onclick: () => moveFile(f, o, 1) }, '↓'),
          o !== f ? h('button', { class: 'btn sm', title: 'Make primary: this file gets the inspector and fit settings; the current primary becomes a comparison', 'aria-label': `Make primary ${o.info.name}`, onclick: () => makePrimary(f, o) }, '★') : null,
          h('button', { class: 'btn sm', title: o === f ? 'The primary stays loaded; hide its Data and Fit to remove it from the plot.' : 'Remove comparison', 'aria-label': `Remove ${o.info.name}`, disabled: o === f, onclick: () => { commit('remove comparison'); removeOverlay(o.path); } }, '✕')),
        h('details', { class: 'series-detail', open: !!cfg.optionsOpen, ontoggle: (e) => { f.style.comparison ||= {}; f.style.comparison[o.path] = { ...(f.style.comparison[o.path] || {}), optionsOpen: e.target.open }; } }, h('summary', {}, `${o === f ? 'Primary · ' : ''}${fit ? fit.status + (o.fitStale ? ' · stale' : '') : 'Not fit'} · Style & legend…`), seriesOptions(f, o)));
    })),
    files.some((o) => o.fit[seriesDetector(f, o)]) ? h('details', { class: 'comparison-table', open: !!f.display.tableOpen, ontoggle: (e) => { f.display.tableOpen = e.target.open; } }, h('summary', {}, 'Comparison results · all parameters ± errors'),
      h('div', { style: 'overflow:auto;max-height:230px' }, h('table', { class: 'tbl', 'aria-label': 'Comparison results' },
        h('thead', {}, h('tr', {}, ['File / detector', 'Parameter', 'Value ± error', 'Unit', 'Status'].map((x) => h('th', {}, x)))),
        h('tbody', {}, comparisonRows(f).map((r) => h('tr', {}, h('td', { class: 'txt' }, `${r.label} · ${r.detector}`), h('td', {}, r.parameter || '—'), h('td', {}, r.formatted || 'Not fit'), h('td', {}, r.unit || '—'), h('td', {}, `${r.status}${r.stale ? ' · stale' : ''}`))))))) : null);
}
