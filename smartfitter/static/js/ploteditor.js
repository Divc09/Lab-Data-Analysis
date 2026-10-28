// Plot editor: in-place series popover, full editor drawer, and syncing of on-plot edits.
import { call } from './api.js';
import { S, emit, commit } from './state.js';
import { h, mount, icon, seg, select, input, check, field, fmt, num, toast, clone } from './util.js';
import { PALETTES, PRESETS, baseStyle, seriesStyle, resultLines } from './plotstyle.js';
import { currentFigure, plotDiv, kindOf } from './analyze.js';
import { seriesOptions, filePresentation, updateFilePresentation, comparisonFiles } from './seriescontrol.js';
import { unitOptions, xDisplay, displayValue } from './unitdisplay.js';

// Colour field: a picker for quick choices beside the text box, which still takes any CSS colour
// (e.g. a translucent rgba background the picker cannot express).
function colorInput(value, onCommit) {
  const hex = /^#[0-9a-f]{6}$/i.test(value || '') ? value : '#ffffff';
  return h('div', { class: 'row', style: 'gap:4px' },
    h('input', { type: 'color', value: hex, title: 'Pick a colour', style: 'width:28px;height:26px;padding:0;border:none;background:none', onchange: (e) => onCommit(e.target.value) }),
    input(value, onCommit, { style: 'flex:1;min-width:0' }));
}

const DASH = [['solid', '—'], ['dash', '– –'], ['dot', '···'], ['dashdot', '–·']];
const MARKERS = [['circle', '●'], ['square', '■'], ['diamond', '◆'], ['triangle-up', '▲'], ['x', '✕']];

function seriesList(f) {
  const fig = currentFigure(f);
  const seen = new Map();
  for (const t of fig?.data || []) {
    const key = t.meta?.key;
    if (!key || seen.has(key) || key === 'pins' || key === 'res' || key === 'fft') continue;
    seen.set(key, { key, name: t.name, color: t.line?.color || t.marker?.color || t.fillcolor || '#888', type: t.mode || t.type });
  }
  return [...seen.values()];
}

function updSeries(f, key, patch, label = 'series style') {
  commit(label);
  f.style.series = f.style.series || {};
  f.style.series[key] = { ...(f.style.series[key] || {}), ...patch };
  emit();
}

export function openSeriesEditor(f, key, x, y) {
  document.querySelectorAll('.popover').forEach((p) => p.remove());
  const path = key === 'data' || key === 'fit' ? f.path : /^ov(?:fit)?:/.test(key) ? key.replace(/^ov(?:fit)?:/, '') : null;
  const o = path && S.files[path];
  const draw = () => {
    const def = seriesList(f).find((s) => s.key === key) || { name: key, color: '#1F5FBF' };
    const s = { ...seriesStyle(baseStyle(f.style), key, { name: def.name, color: def.color }), ...(f.style.series?.[key] || {}) };
    const pal = PALETTES[baseStyle(f.style).palette] || PALETTES.lab;
    const set = (patch) => { updSeries(f, key, patch); draw(); };
    const mode = s.mode || def.type || 'lines+markers';
    const hasLine = !String(mode).startsWith('markers'), hasMarker = String(mode).includes('markers');
    mount(pop,
      h('div', { class: 'row' }, h('span', { style: 'font-weight:600' }, `Series · ${s.name}`), h('span', { class: 'sp' }), h('button', { class: 'ibtn', 'aria-label': 'Close', onclick: () => pop.remove() }, icon('close', 14))),
      field('Legend label', input(s.name, (v) => set({ name: v }))),
      h('div', { class: 'f' }, h('span', { class: 'lbl' }, 'Color'), h('div', { class: 'swatches' },
        pal.map((c) => h('button', { class: `swatch ${s.color === c ? 'on' : ''}`, style: `background:${c}`, 'aria-label': c, onclick: () => set({ color: c }) })),
        h('input', { type: 'color', value: /^#[0-9a-f]{6}$/i.test(s.color) ? s.color : '#1f5fbf', oninput: (e) => set({ color: e.target.value }) }))),
      h('div', { class: 'g2' },
        field('Line', seg([...DASH, ['none', 'None']], hasLine ? s.dash : 'none', (v) => set(v === 'none' ? { mode: 'markers' } : { dash: v, mode: hasMarker ? 'lines+markers' : 'lines' }))),
        field('Marker', seg([...MARKERS.slice(0, 3), ['none', 'None']], hasMarker ? s.marker : 'none', (v) => set(v === 'none' ? { mode: 'lines' } : { marker: v, mode: hasLine ? 'lines+markers' : 'markers' })))),
      h('div', { class: 'g3' }, field('Width', input(s.width, (v) => set({ width: num(v) ?? 1.5 }), { numeric: true })), field('Marker size', input(s.size, (v) => set({ size: num(v) ?? 6 }), { numeric: true })), field('Opacity', input(s.opacity, (v) => set({ opacity: Math.min(1, Math.max(0.05, num(v) ?? 1)) }), { numeric: true }))),
      h('div', { class: 'row wrap' }, check('On plot', s.visible !== false, (v) => set({ visible: v })), check('In legend', s.legend !== false, (v) => set({ legend: v })),
        h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: () => { commit('reset series'); delete f.style.series[key]; emit(); draw(); } }, 'Reset')));
    if (o) pop.append(h('details', {}, h('summary', {}, 'File-level legend template & fit settings'), seriesOptions(f, o)));
  };
  const pop = h('div', { class: 'popover', role: 'dialog' });
  pop.style.left = `${Math.min(x, innerWidth - 330)}px`;
  pop.style.top = `${Math.min(y, innerHeight - 380)}px`;
  document.body.append(pop);
  draw();
  const esc = (e) => { if (e.key === 'Escape') { pop.remove(); document.removeEventListener('keydown', esc, true); } };
  document.addEventListener('keydown', esc, true);
  setTimeout(() => document.addEventListener('mousedown', function off(e) { if (!pop.contains(e.target) && !e.target.closest('.legend')) { pop.remove(); document.removeEventListener('mousedown', off); } }), 0);
}

// Sync edits made directly on the plot (titles, legend text, dragged legend/annotations/shapes).
export function applyPlotlyEdit(f, ev, el, kind = 'relayout') {
  f.style = f.style || {};
  const st = f.style;
  const before = clone(st);
  const finish = (handled) => {
    if (handled && JSON.stringify(before) !== JSON.stringify(f.style)) {
      const after = f.style; f.style = before; commit('direct plot edit'); f.style = after;
      setTimeout(() => emit('insp'), 0);
    }
    return handled;
  };
  let handled = false;
  if (kind === 'restyle') {
    const [patch, idx] = ev;
    if (patch?.name && idx) {
      const t = el.data[idx[0]];
      const key = t?.meta?.key;
      if (key) { st.series = st.series || {}; st.series[key] = { ...(st.series[key] || {}), name: Array.isArray(patch.name) ? patch.name[0] : patch.name }; }
    }
    return finish(!!patch?.name);
  }
  const lay = (el || plotDiv()).layout || {};
  for (const [k, v] of Object.entries(ev)) {
    let m;
    if (k === 'title.text') { st.title = v; handled = true; }
    else if ((m = k.match(/^(x|y)axis(\d?)\.title\.text$/))) { const ax = m[1] === 'x' ? 'x' : m[2] === '2' ? 'y2' : m[2] === '9' ? 'yRight' : 'y'; st[ax] = { ...(st[ax] || {}), title: v }; handled = true; }
    else if (k === 'legend.x' || k === 'legend.y') {
      // Plotly reports the dragged position for the anchor active during the drag (e.g. 'right' for the upper-right
      // preset); keep that anchor so the redraw lands exactly where the legend was dropped.
      const fl = plotDiv()?._fullLayout?.legend || {};
      st.legend = { ...(st.legend || {}), [k.slice(7)]: v, xanchor: fl.xanchor || st.legend?.xanchor, yanchor: fl.yanchor || st.legend?.yanchor };
      handled = true;
    }
    else if ((m = k.match(/^annotations\[(\d+)\]\.(x|y|ax|ay)$/))) {
      const a = lay.annotations?.[Number(m[1])];
      if (a?.name === 'results') { st.results = { ...(st.results || {}), [m[2]]: v }; }
      else if (a?.name?.startsWith('u')) { const d = st.shapes[Number(a.name.slice(1))]; if (d) d[m[2]] = v; }
      handled = true;
    } else if ((m = k.match(/^shapes\[(\d+)\]\.(x0|x1|y0|y1)$/))) {
      const sh = lay.shapes?.[Number(m[1])];
      if (sh?.name?.startsWith('u')) {
        const d = st.shapes[Number(sh.name.slice(1))];
        if (d?.kind === 'vline' && m[2] === 'x0') d.x = v;
        if (d?.kind === 'hline' && m[2] === 'y0') d.y = v;
        if (d?.kind === 'region' && (m[2] === 'x0' || m[2] === 'x1')) d[m[2]] = v;
      }
      handled = true;
    }
  }
  return finish(handled);
}

// Double-click edits stay anchored to the element; Escape cancels, Enter/blur applies once.
export function inlinePlotEditor(el, getFile, beforeEdit = () => {}) {
  const start = (e) => {
    const target = e.target.closest('.xtitle,.ytitle,.y2title,.y9title,.gtitle,.legendtext') || e.target.closest('.legend .traces')?.querySelector('.legendtext');
    const f = getFile(); if (!target || !f) return;
    if (document.querySelector('.plot-inline-edit')) return;
    beforeEdit(); e.preventDefault(); e.stopImmediatePropagation();
    document.querySelectorAll('.plot-inline-edit').forEach((n) => n.remove());
    let value = target.textContent, apply;
    if (target.matches('.legendtext')) {
      const trace = target.closest('.traces')?.__data__?.[0]?.trace;
      const key = trace?.meta?.key;
      const path = key === 'data' || key === 'fit' ? f.path : key?.replace(/^ov(?:fit)?:/, '');
      const o = S.files[path];
      if (o) { value = filePresentation(f, o).label; apply = (label) => updateFilePresentation(f, o, { label }); }
      else if (key) apply = (name) => updSeries(f, key, { name });
    } else {
      const key = target.matches('.gtitle') ? 'title.text' : target.matches('.xtitle') ? 'xaxis.title.text' : target.matches('.y2title') ? 'yaxis2.title.text' : target.matches('.y9title') ? 'yaxis9.title.text' : 'yaxis.title.text';
      apply = (text) => { applyPlotlyEdit(f, { [key]: text }, el); emit(); };
    }
    if (!apply) return;
    const rect = target.getBoundingClientRect();
    const edit = h('input', { class: 'field plot-inline-edit', value, 'aria-label': 'Edit plot text', style: `position:fixed;z-index:65;left:${Math.max(8, Math.min(rect.left, innerWidth - 320))}px;top:${Math.max(8, rect.top - 4)}px;width:${Math.min(520, Math.max(300, rect.width + 30))}px` });
    let done = false;
    const finish = (save) => { if (done) return; done = true; if (save && edit.value !== value) apply(edit.value); edit.remove(); };
    edit.addEventListener('blur', () => finish(true));
    edit.addEventListener('keydown', (ev) => { ev.stopPropagation(); if (ev.key === 'Enter') finish(true); if (ev.key === 'Escape') finish(false); });
    document.body.append(edit); edit.focus(); edit.select();
  };
  el.addEventListener('dblclick', start, true);
  // Plotly's draggable legend uses a dragcover and synthesizes click(detail=2), not dblclick.
  el.addEventListener('click', (e) => { if (e.detail >= 2) start(e); }, true);
}

export function applyPreset(f, name) {
  commit('style preset');
  const p = PRESETS[name];
  Object.assign(f.style, { preset: name, font: p.font, base: p.base, width_in: p.width_in, height_in: p.height_in, line: p.line, marker: p.marker });
  emit();
}

// ------------------------------------------------------------ portable styles (copy/paste, apply to all, templates)
// Only the "look" travels; plot-specific content (title, axis titles/limits/tick positions, annotations, series names,
// comparison set) stays with its plot.
const AXIS_LOCAL = ['min', 'max', 'title', 'ticks', 'tick0', 'dtick'];
export function portableStyle(style) {
  const s = clone(style || {});
  for (const k of ['title', 'series', 'shapes', 'comparison', 'fileOrder', 'inset', 'xbreak', 'legendFix']) delete s[k];
  for (const ax of ['x', 'y', 'y2']) if (s[ax]) for (const k of AXIS_LOCAL) delete s[ax][k];
  if (s.legend) { delete s.legend.x; delete s.legend.y; delete s.legend.xanchor; delete s.legend.yanchor; }
  if (s.results) { delete s.results.x; delete s.results.y; }
  return s;
}
export function applyStyle(files, style, label = 'paste style') {
  if (!style || !files.length) return;
  commit(label);
  const port = portableStyle(style);
  for (const f of files) {
    const keep = {};
    for (const ax of ['x', 'y', 'y2']) keep[ax] = Object.fromEntries(AXIS_LOCAL.filter((k) => f.style[ax]?.[k] !== undefined).map((k) => [k, f.style[ax][k]]));
    f.style = { ...f.style, ...clone(port), legend: { ...(f.style.legend || {}), ...clone(port.legend || {}) }, results: { ...(f.style.results || {}), ...clone(port.results || {}) } };
    for (const ax of ['x', 'y', 'y2']) f.style[ax] = { ...(port[ax] || {}), ...keep[ax] };
  }
  emit();
}
let styleClipboard = null;
export function styleTools(f) {
  const templates = S.settings.style_templates || {};
  const nameEl = h('input', { class: 'field', placeholder: 'Template name', 'aria-label': 'Style template name', style: 'width:150px' });
  const save = async () => {
    const name = nameEl.value.trim();
    if (!name) { toast('Type a template name first.'); nameEl.focus(); return; }
    const next = { ...templates, [name]: portableStyle(f.style) };
    S.settings = await call('settings_set', { patch: { style_templates: next } }, { quiet: true });
    toast(`Saved style template "${name}".`, 'ok'); emit();
  };
  const remove = async (name) => { const next = { ...templates }; delete next[name]; S.settings = await call('settings_set', { patch: { style_templates: next } }, { quiet: true }); emit(); };
  const sameKind = () => Object.values(S.files).filter((o) => o !== f && kindOf(o) === kindOf(f));
  return h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:6px' },
    h('div', { class: 'row wrap' },
      h('button', { class: 'btn sm', title: 'Copy fonts, sizes, palette, grid, ticks, legend/results layout and axis formatting (not titles, limits or annotations)', onclick: () => { styleClipboard = portableStyle(f.style); toast('Plot style copied.'); emit(); } }, 'Copy style'),
      h('button', { class: 'btn sm', disabled: !styleClipboard, onclick: () => applyStyle([f], styleClipboard) }, 'Paste style'),
      h('button', { class: 'btn sm', disabled: !sameKind().length, title: 'Apply this plot\'s look to every other open plot of the same kind', onclick: () => { applyStyle(sameKind(), f.style, 'apply style to all'); toast(`Style applied to ${sameKind().length} open plot(s).`, 'ok'); } }, `Apply to all open (${sameKind().length})`)),
    h('div', { class: 'row wrap' },
      select([['', Object.keys(templates).length ? 'Apply template…' : 'No saved templates'], ...Object.keys(templates).map((n) => [n, n])], '', (n) => n && applyStyle([f], templates[n], `template ${n}`)),
      nameEl, h('button', { class: 'btn sm', onclick: save }, 'Save as template'),
      Object.keys(templates).length ? select([['', 'Delete template…'], ...Object.keys(templates).map((n) => [n, n])], '', (n) => n && remove(n)) : null));
}

// ------------------------------------------------------------ editor drawer
export function editorPanel(f) {
  const tab = S.editor.tab;
  const tabs = [['series', 'Series'], ['legend', 'Legend'], ['axes', 'Axes & ticks'], ['text', 'Text & results'], ...(kindOf(f) === '1d' ? [['inset', 'Inset & breaks']] : []), ['style', 'Layout & style']];
  const body = h('div', { class: 'insp-body' });
  body.append(({ series: seriesTab, legend: legendTab, axes: axesTab, text: textTab, inset: insetTab, style: styleTab }[tab] || seriesTab)(f));
  return h('frag', {},
    h('div', { class: 'row', style: 'padding:12px 12px 0' }, h('span', { style: 'font-weight:600' }, 'Plot editor'), h('span', { class: 'sp' }),
      h('button', { class: 'btn sm', title: 'Undo (Ctrl+Z)', onclick: () => import('./state.js').then((m) => m.undo()) }, icon('undo', 13)),
      h('button', { class: 'btn sm primary', onclick: () => { S.editor.open = false; emit(); } }, 'Done')),
    h('div', { class: 'row wrap', style: 'padding:8px 12px 0;gap:4px' }, h('span', { class: 'lbl' }, 'Add'),
      ['Text', 'Arrow', 'V line', 'H line', 'Region', 'Fit value', ...(kindOf(f) === '1d' ? ['Inset'] : [])].map((k) => h('button', { class: 'btn sm', onclick: (e) => addDecoration(f, k, e) }, k))),
    h('div', { class: 'etabs', role: 'tablist' }, tabs.map(([k, l]) => h('button', { class: tab === k ? 'on' : '', onclick: () => { S.editor.tab = k; emit('insp'); } }, l))),
    body,
    h('div', { class: 'row wrap', style: 'padding:10px 12px;border-top:1px solid var(--line)' },
      h('button', { class: 'btn sm', onclick: async () => { S.settings = await call('settings_set', { patch: { plot_style: { ...clone(f.style), shapes: [], inset: undefined, xbreak: undefined, title: '', x: {}, y: {}, y2: {} } } }); toast('Saved as your default style for new files.', 'ok'); } }, 'Use as default for new files'),
      h('button', { class: 'btn sm', title: 'Replace this plot\'s look with your saved default (undo with Ctrl+Z)', onclick: () => { commit('reset style'); f.style = clone(S.settings.plot_style || {}); emit(); } }, 'Reset to my default'),
      h('button', { class: 'btn sm', title: `Apply this look to every other open ${f.info.type} file`, onclick: () => applyToType(f) }, `Apply to open ${f.info.type} files`)));
}

// Same semantics as the style tools: only the portable look travels (no titles, limits, annotations, series names or
// comparison setup, which belong to their own plot).
function applyToType(f) {
  const targets = Object.values(S.files).filter((o) => o !== f && o.info.type === f.info.type);
  applyStyle(targets, f.style, 'apply style to same type');
  toast(`Applied this look to ${targets.length} other ${f.info.type} file(s).`, 'ok');
}

function addDecoration(f, kind, ev) {
  const el = plotDiv();
  const physicalRange = (r, log) => log ? r.map((v) => 10 ** v) : r;
  const xa = el.layout?.xaxis?.range ? physicalRange(el.layout.xaxis.range, f.style.x?.log) : f.xrange || [0, 1];
  const ya = el.layout?.yaxis?.range ? physicalRange(el.layout.yaxis.range, f.style.y?.log) : f.yrange || [0, 1];
  const cx = (xa[0] + xa[1]) / 2, cy = (ya[0] + ya[1]) / 2;
  f.style.shapes = f.style.shapes || [];
  if (kind === 'Inset') {
    commit('add inset');
    const third = (xa[1] - xa[0]) / 3;
    const zoomed = f.xrange || el.layout?.xaxis?.autorange === false;
    f.style.inset = { pos: 'ur', size: 0.35, y: null, ...(f.style.inset || {}), enabled: true, x0: zoomed ? xa[0] : xa[0] + third, x1: zoomed ? xa[1] : xa[0] + 2 * third };
    S.editor.tab = 'inset';
    emit();
    toast('Inset added. Edit its range and corner in Inset & breaks.');
    return;
  }
  const pinX = f.pins?.[0]?.x;
  if (kind === 'Fit value') {
    const fit = f.fit[f.detector];
    if (!fit) { toast('Run a fit first.'); return; }
    const m = fit.metrics || {};
    const opts = [['programmed_pi_time_ns', 'π pulse (programmed)'], ['programmed_pi_over_2_time_ns', 'π/2 pulse (programmed)'], ['first_peak_ns', 'First peak'], ['delay_ns', 'Delay']]
      .filter(([k]) => typeof m[k] === 'number').map(([k, l]) => ({ label: `${l} · ${displayValue(m[k], null, 'ns').withUnit}`, x: xDisplay(f, comparisonFiles(f)).transform([m[k]], 'ns')?.[0], text: `${l} ${displayValue(m[k], null, 'ns').withUnit}` }));
    fit.params.filter((p) => /^(x0|center)/.test(p.name)).forEach((p) => opts.push({ label: `${p.name} · ${displayValue(p.value, p.error, p.unit).withUnit}`, x: xDisplay(f, comparisonFiles(f)).transform([p.value], p.unit || f.info.x_unit)?.[0], text: `${p.name} = ${displayValue(p.value, p.error, p.unit).withUnit}` }));
    if (!opts.length) { toast('No x-axis fit values for this model.'); return; }
    import('./util.js').then(({ contextMenu }) => contextMenu(ev.clientX, ev.clientY, opts.map((o) => ({ label: o.label, run: () => { commit('add fit line'); f.style.shapes.push({ kind: 'vline', x: o.x, text: o.text, color: '#7A4FD6' }); emit(); } }))));
    return;
  }
  commit('add to plot');
  const add = {
    Text: { kind: 'text', x: cx, y: cy, text: 'Text' }, Arrow: { kind: 'arrow', x: cx, y: cy, text: 'Note', ax: -50, ay: -40 },
    'V line': { kind: 'vline', x: pinX ?? cx, text: '' }, 'H line': { kind: 'hline', y: f.pins?.[0]?.y ?? cy, text: '' },
    Region: { kind: 'region', x0: xa[0] + (xa[1] - xa[0]) / 3, x1: xa[0] + 2 * (xa[1] - xa[0]) / 3, text: '' },
  }[kind];
  f.style.shapes.push({ color: '#7A4FD6', ...add });
  S.editor.tab = 'text';
  emit();
  toast('Added. Drag it on the plot, or edit it in Text & results.');
}

function seriesTab(f) {
  const st = baseStyle(f.style);
  const list = seriesList(f);
  const cols = 'grid-template-columns:26px minmax(0,1fr) 40px 48px auto';
  return h('div', { style: 'display:flex;flex-direction:column;gap:10px' },
    h('div', {},
      h('div', { class: 'srow srow-head', style: cols }, h('span', {}, ''), h('span', {}, 'Label'), h('span', {}, 'Plot'), h('span', {}, 'Legend'), h('span', {}, 'Order')),
      list.map((s) => {
      const o = f.style.series?.[s.key] || {};
      return h('div', { class: 'srow', style: cols },
        h('button', { class: 'swatch', style: `background:${o.color || s.color};width:22px;height:18px`, title: 'Colour, line, marker and width for this series', 'aria-label': `Style ${s.name}`, onclick: (e) => openSeriesEditor(f, s.key, e.clientX - 330, e.clientY) }),
        input(o.name ?? s.name, (v) => updSeries(f, s.key, { name: v }), { style: 'height:26px', 'aria-label': `Label for ${s.name}` }),
        h('input', { type: 'checkbox', checked: o.visible !== false, title: 'Draw on the plot', 'aria-label': `Show ${s.name} on plot`, onchange: (e) => updSeries(f, s.key, { visible: e.target.checked }) }),
        h('input', { type: 'checkbox', checked: o.legend !== false, title: 'List in the legend', 'aria-label': `Show ${s.name} in legend`, onchange: (e) => updSeries(f, s.key, { legend: e.target.checked }) }),
        h('div', { class: 'row', style: 'gap:2px' }, [-1, 1].map((direction) => h('button', { class: 'btn sm', title: direction < 0 ? 'Move up' : 'Move down', onclick: () => {
          const order = list.map((t) => t.key), i = order.indexOf(s.key), j = i + direction;
          if (j < 0 || j >= order.length) return;
          commit('trace order'); [order[i], order[j]] = [order[j], order[i]]; f.style.seriesOrder = order; emit();
        } }, direction < 0 ? '↑' : '↓'))));
    }),
      h('div', { class: 'note' }, 'Click a colour swatch for line, marker and width options.')),
    field('Colour palette', select([['lab', 'Lab default (blue, orange, teal, magenta)'], ['okabe', 'Colour-blind safe (Okabe–Ito)'], ['gray', 'Grayscale']], st.palette, (v) => { commit('palette'); f.style.palette = v; emit(); })),
    kindOf(f) === '1d' ? field('Uncertainty display', seg([['band', 'Band'], ['bars', 'Error bars']], f.errorMode, (v) => { f.errorMode = v; f.layers.band = true; emit(); })) : null,
    h('div', { class: 'g2' }, field('Default line width', input(st.line, (v) => { commit('line width'); f.style.line = num(v) ?? 1.6; emit(); }, { numeric: true })), field('Default marker size', input(st.marker, (v) => { commit('marker size'); f.style.marker = num(v) ?? 6; emit(); }, { numeric: true }))));
}

function legendTab(f) {
  const st = baseStyle(f.style);
  const L = st.legend;
  const set = (patch) => { commit('legend'); f.style.legend = { ...(f.style.legend || {}), ...patch }; emit(); };
  const custom = L.x != null;
  return h('div', { style: 'display:flex;flex-direction:column;gap:10px' },
    check('Show legend', f.layers.legend, (v) => { commit('legend'); f.layers.legend = v; emit(); }),
    // Common choices first; exact anchors only when needed (dragging the legend on the plot sets them too).
    field('Position', seg([['ul', 'Upper left'], ['ur', 'Upper right'], ['ll', 'Lower left'], ['lr', 'Lower right'], ['outside', 'Outside'], ['below', 'Below']], custom ? '' : L.pos, (v) => set({ pos: v, x: null, y: null }))),
    h('div', { class: 'g2' }, field('Columns', input(L.cols || 1, (v) => set({ cols: Math.max(1, num(v) || 1) }), { numeric: true })), field('Font size (px)', input(L.size || '', (v) => set({ size: num(v) }), { numeric: true, placeholder: 'auto' }))),
    field('Legend title', input(L.title || '', (title) => set({ title }), { placeholder: 'none' })),
    h('details', { open: custom || !!f.display.editorDetails?.legendBox, ontoggle: (e) => { f.display.editorDetails ||= {}; f.display.editorDetails.legendBox = e.target.open; } },
      h('summary', {}, custom ? 'Box and exact position (custom position set)' : 'Box and exact position'),
      h('div', { style: 'display:flex;flex-direction:column;gap:8px;margin-top:8px' },
        h('div', { class: 'row wrap' }, check('Frame', L.frame !== false, (v) => set({ frame: v })),
          field('Border width (px)', input(L.borderwidth ?? 1, (v) => set({ borderwidth: Math.max(0, num(v) ?? 1) }), { numeric: true, style: 'width:60px' }))),
        field('Text colour', colorInput(L.color || '#1D2330', (color) => set({ color }))),
        field('Background', colorInput(L.bg || 'rgba(255,255,255,0.92)', (bg) => set({ bg }))),
        field('Border colour', colorInput(L.edge || '#D9DDE4', (edge) => set({ edge }))),
        h('div', { class: 'g2' }, field('Anchor x (0–1 of plot)', input(L.x ?? '', (v) => set({ x: num(v), y: L.y ?? 0.99 }), { numeric: true, placeholder: 'preset' })),
          field('Anchor y (0–1 of plot)', input(L.y ?? '', (v) => set({ x: L.x ?? 0.01, y: num(v) }), { numeric: true, placeholder: 'preset' }))),
        h('div', { class: 'note' }, 'Dragging the legend on the plot sets the anchor; choosing a Position above clears it.'))),
    h('div', { class: 'note' }, 'Double-click legend text on the plot to rename it; Alt-click opens full trace styling. Per-file result values, precision and units are in Series → Style & legend.'));
}

export function parseTickTable(text) {
  return text.split(/\r?\n/).filter((line) => line.trim()).map((line, i) => {
    const match = line.match(/^\s*([^,\t;]+)[,\t;](.*)$/);
    if (!match || !match[1].trim() || !Number.isFinite(Number(match[1])) || !match[2].trim()) throw new Error(`Tick row ${i + 1}: use position, label`);
    return { value: Number(match[1]), text: match[2].trim() };
  });
}

function axisCard(f, ax, label) {
  const a = f.style[ax] || {};
  const set = (patch) => { commit('axis'); f.style[ax] = { ...(f.style[ax] || {}), ...patch }; if ('min' in patch || 'max' in patch) { if (ax === 'x') f.xrange = null; if (ax === 'y') f.yrange = null; } emit(); };
  const ticks = h('textarea', { class: 'field', rows: 4, 'aria-label': `${label} explicit tick table`, placeholder: '0, zero\n1.5, label' }, (a.ticks || []).map((t) => `${t.value}, ${t.text}`).join('\n'));
  ticks.addEventListener('keydown', (e) => e.stopPropagation());
  ticks.addEventListener('change', () => { try { set({ ticks: parseTickTable(ticks.value) }); } catch (e) { toast(e.message, 'err'); } });
  return h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' }, h('span', { style: 'font-weight:600' }, label),
    field('Label', input(a.title ?? '', (v) => set({ title: v === '' ? null : v }), { placeholder: 'automatic' })),
    h('div', { class: 'g3' }, field('Min', input(a.min ?? '', (v) => set({ min: num(v) }), { numeric: true, placeholder: 'auto' })), field('Max', input(a.max ?? '', (v) => set({ max: num(v) }), { numeric: true, placeholder: 'auto' })),
      field('Scale', select([['', 'Linear'], ['log', 'Log']], a.log ? 'log' : '', (v) => set({ log: v === 'log' })))),
    h('div', { class: 'g3' }, field('Major every', input(a.dtick ?? '', (v) => set({ dtick: num(v) }), { numeric: true, placeholder: 'auto' })), field('Minor per major', input(a.minor ?? '', (v) => set({ minor: num(v) }), { numeric: true, placeholder: 'off' })),
      field('Format', select([['auto', 'Auto'], ['fixed1', '0.0'], ['fixed2', '0.00'], ['fixed3', '0.000'], ['sci', 'Scientific'], ['eng', 'Engineering (k, M)'], ['percent', 'Percent']], a.format || 'auto', (v) => set({ format: v })))),
    field('First major tick', input(a.tick0 ?? '', (v) => set({ tick0: num(v) }), { numeric: true, placeholder: 'auto' })),
    h('details', { open: !!f.display.editorDetails?.[`ticks:${ax}`], ontoggle: (e) => { f.display.editorDetails ||= {}; f.display.editorDetails[`ticks:${ax}`] = e.target.open; } }, h('summary', {}, 'Explicit tick table…'), h('div', { class: 'note' }, 'One position, label per row, in displayed axis units. For log axes use positive positions; major spacing is in decades. Empty restores automatic ticks.'), ticks,
      h('button', { class: 'btn sm', onclick: () => set({ ticks: [] }) }, 'Clear tick table')));
}

function axesTab(f) {
  const st = baseStyle(f.style);
  return h('div', { style: 'display:flex;flex-direction:column;gap:10px' },
    field('Title', input(f.style.title || '', (v) => { commit('title'); f.style.title = v; emit(); }, { placeholder: 'None' })),
    kindOf(f) === '1d' ? field('X display unit', select(unitOptions(f.info.x_unit), f.display.xunit || 'auto', (xunit) => { commit('x display units'); f.display.xunit = xunit; emit(); })) : null,
    axisCard(f, 'x', 'X axis'), axisCard(f, 'y', 'Y axis'),
    f.layers.sigref || f.compareDetectors ? axisCard(f, 'y2', 'Counts / detector Y axis') : h('div', { class: 'note' }, 'A counts axis appears automatically when Signal + Reference counts or a second detector in counts are shown.'),
    // Compared series moved to the right (Series panel → Y axis: Right) share this independent axis (yaxis9).
    Object.entries(f.style.comparison || {}).some(([p, c]) => c?.axis === 'right' && S.overlays.includes(p))
      ? axisCard(f, 'yRight', 'Comparison right Y axis') : h('div', { class: 'note' }, 'Tip: put a compared file on its own right-hand axis via Series → Style & legend → Y axis: Right.'),
    field('Ticks', seg([['outside', 'Outside'], ['inside', 'Inside'], ['none', 'None']], st.ticks, (v) => { commit('ticks'); f.style.ticks = v; emit(); })),
    h('div', { class: 'row wrap' }, check('Mirror ticks on top/right', st.mirror, (v) => { commit('mirror'); f.style.mirror = v; emit(); }), check('Grid', st.grid, (v) => { commit('grid'); f.style.grid = v; emit(); })));
}

function insetTab(f) {
  const ins = f.style.inset || {};
  const xb = f.style.xbreak || {};
  const blocked = currentFigure(f)?.xbreakBlocked;
  const setI = (patch) => { commit('inset'); f.style.inset = { pos: 'ur', size: 0.35, ...(f.style.inset || {}), ...patch }; emit(); };
  const setB = (patch) => { commit('x break'); f.style.xbreak = { ...(f.style.xbreak || {}), ...patch }; emit(); };
  const yAuto = !ins.y;
  return h('div', { style: 'display:flex;flex-direction:column;gap:10px' },
    h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' }, h('span', { style: 'font-weight:600' }, 'Inset zoom'),
      check('Show inset', !!ins.enabled, (v) => setI({ enabled: v })),
      h('div', { class: 'g2' }, field('Zoom from x', input(ins.x0 ?? '', (v) => setI({ x0: num(v) }), { numeric: true })), field('Zoom to x', input(ins.x1 ?? '', (v) => setI({ x1: num(v) }), { numeric: true }))),
      field('Corner', seg([['ul', 'Upper left'], ['ur', 'Upper right'], ['ll', 'Lower left'], ['lr', 'Lower right']], ins.pos || 'ur', (v) => setI({ pos: v }))),
      field('Size (fraction of plot width)', input(ins.size ?? 0.35, (v) => setI({ size: Math.min(0.7, Math.max(0.1, num(v) ?? 0.35)) }), { numeric: true })),
      check('Automatic y range', yAuto, (v) => setI({ y: v ? null : { y0: 0, y1: 1 } })),
      yAuto ? null : h('div', { class: 'g2' }, field('y min', input(ins.y.y0, (v) => setI({ y: { ...ins.y, y0: num(v) } }), { numeric: true })), field('y max', input(ins.y.y1, (v) => setI({ y: { ...ins.y, y1: num(v) } }), { numeric: true }))),
      h('div', { class: 'note' }, 'Magnifies the data and fit between the two x values; a dashed box marks the region on the main plot. Use Add > Inset to start from the current zoom.')),
    h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' }, h('span', { style: 'font-weight:600' }, 'Broken x axis'),
      check('Hide an x interval', !!xb.enabled, (v) => setB({ enabled: v }), { disabled: !!blocked }),
      h('div', { class: 'g2' }, field('Gap starts at x', input(xb.from ?? '', (v) => setB({ from: num(v) }), { numeric: true })), field('Gap ends at x', input(xb.to ?? '', (v) => setB({ to: num(v) }), { numeric: true }))),
      h('div', { class: 'note' }, blocked ? `Broken axis is switched off while the ${blocked} is shown.` : 'Two x panels share the y axis; useful for long Rabi or T1 scans. Needs from < to inside the data range.')));
}

function textTab(f) {
  const st = baseStyle(f.style);
  const fit = f.fit[f.detector];
  const lines = fit ? resultLines(fit, f.info) : [];
  const cfg = st.results;
  const setLine = (key, patch) => { commit('results line'); f.style.results = { ...(f.style.results || {}), lines: { ...(cfg.lines || {}), [key]: { ...(cfg.lines?.[key] || {}), ...patch } } }; emit(); };
  const out = h('div', { style: 'display:flex;flex-direction:column;gap:10px' });
  const setCfg = (patch) => { commit('annotation configuration'); f.style.results = { ...(f.style.results || {}), ...patch }; emit(); };
  const all = [...lines, ...(cfg.static || []).map((l) => ({ ...l, static: true }))];
  const order = cfg.order || [];
  const fallback = new Map(all.map((l, i) => [l.key, i]));
  const rank = (l) => order.includes(l.key) ? order.indexOf(l.key) : order.length + fallback.get(l.key);
  all.sort((a, b) => rank(a) - rank(b));
  out.append(check('Show results box on the plot', f.layers.results, (v) => { commit('results visibility'); f.layers.results = v; emit(); }),
    h('div', { class: 'g2' }, field('Box position', select([['ul', 'Upper left'], ['ur', 'Upper right'], ['ll', 'Lower left'], ['lr', 'Lower right']], cfg.pos, (v) => { commit('results position'); f.style.results = { ...(f.style.results || {}), pos: v, x: null, y: null }; emit(); })),
      field('Number format', select([['pm', 'Value ± error'], ['sig4', '4 significant']], cfg.format, (v) => { commit('format'); f.style.results = { ...(f.style.results || {}), format: v }; emit(); }))));
  if (all.length) out.append(h('div', { class: 'card', style: 'padding:4px 10px' }, all.map((l, i) => {
    const line = cfg.lines?.[l.key] || {};
    const move = (direction) => { const keys = all.map((r) => r.key), j = i + direction; if (j < 0 || j >= keys.length) return; [keys[i], keys[j]] = [keys[j], keys[i]]; setCfg({ order: keys }); };
    return h('details', { open: !!f.display.editorDetails?.[l.key], ontoggle: (e) => { f.display.editorDetails ||= {}; f.display.editorDetails[l.key] = e.target.open; } }, h('summary', {}, `${line.label ?? l.label} · format / units / order`),
      field('Label', input(line.label ?? l.label, (label) => setLine(l.key, { label }))),
      h('div', { class: 'row wrap' }, check('Show on plot', line.plot ?? !l.defaultOff, (plot) => setLine(l.key, { plot })),
        h('button', { class: 'btn sm', disabled: i === 0, onclick: () => move(-1) }, '↑'), h('button', { class: 'btn sm', disabled: i === all.length - 1, onclick: () => move(1) }, '↓'),
        l.static ? h('button', { class: 'btn sm', onclick: () => setCfg({ static: cfg.static.filter((r) => r.key !== l.key) }) }, 'Remove static row') : null),
      l.static ? field('Static text', input(line.text ?? l.text, (text) => setLine(l.key, { text }))) : h('frag', {},
        h('div', { class: 'g2' }, field('Units', select(unitOptions(l.unit), line.units || 'auto', (units) => setLine(l.key, { units }))),
          field('Precision', input(line.precision || 2, (v) => setLine(l.key, { precision: Math.max(1, Math.min(8, Number(v) || 2)) })))),
        field('Format', select([['pm', 'Value ± error'], ['sig4', '4 significant'], ['value', 'Value only']], line.format || cfg.format || 'pm', (format) => setLine(l.key, { format }))),
        field('Line template', input(line.template || '{label} = {value} {unit}', (template) => setLine(l.key, { template }), { title: '{label}, {value}, {unit}' }))));
  })));
  else out.append(h('div', { class: 'note' }, 'Fit values appear here after a fit; custom static rows work without fitting.'));
  out.append(h('button', { class: 'btn sm', onclick: () => setCfg({ static: [...(cfg.static || []), { key: `static:${Date.now()}`, label: 'Custom row', text: 'Note' }] }) }, 'Add static annotation row'),
    h('details', { open: !!f.display.editorDetails?.resultsBox, ontoggle: (e) => { f.display.editorDetails ||= {}; f.display.editorDetails.resultsBox = e.target.open; } },
      h('summary', {}, 'Box appearance'),
      h('div', { style: 'display:flex;flex-direction:column;gap:8px;margin-top:8px' },
        h('div', { class: 'g2' }, field('Font size (px)', input(cfg.size || '', (v) => setCfg({ size: num(v) }), { numeric: true, placeholder: 'auto' })), field('Alignment', select(['left', 'center', 'right'], cfg.align || 'left', (align) => setCfg({ align })))),
        field('Text colour', colorInput(cfg.color || '#1D2330', (color) => setCfg({ color }))),
        field('Background', colorInput(cfg.bg || 'rgba(255,255,255,0.92)', (bg) => setCfg({ bg }))),
        field('Border colour', colorInput(cfg.edge || '#D9DDE4', (edge) => setCfg({ edge }))),
        h('div', { class: 'g2' }, field('Padding (px)', input(cfg.padding ?? 6, (v) => setCfg({ padding: num(v) ?? 6 }), { numeric: true })), check('Frame', cfg.frame !== false, (frame) => setCfg({ frame }))))));
  out.append(h('span', { class: 'lbl', style: 'margin-top:6px' }, 'Added items'));
  const shapes = f.style.shapes || [];
  if (!shapes.length) out.append(h('div', { class: 'note' }, 'Use Add above for text, arrows, reference lines, shaded regions and fit-value lines.'));
  shapes.forEach((d, i) => {
    const upd = (patch) => { commit('edit item'); Object.assign(d, patch); emit(); };
    const pos = d.kind === 'vline' ? [field('x', input(d.x, (v) => upd({ x: num(v) }), { numeric: true }))] : d.kind === 'hline' ? [field('y', input(d.y, (v) => upd({ y: num(v) }), { numeric: true }))]
      : d.kind === 'region' ? [field('from x', input(d.x0, (v) => upd({ x0: num(v) }), { numeric: true })), field('to x', input(d.x1, (v) => upd({ x1: num(v) }), { numeric: true }))]
        : [field('x', input(d.x, (v) => upd({ x: num(v) }), { numeric: true })), field('y', input(d.y, (v) => upd({ y: num(v) }), { numeric: true }))];
    out.append(h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:6px' },
      h('div', { class: 'row' }, h('span', { class: 'lbl' }, { vline: 'Vertical line', hline: 'Horizontal line', region: 'Shaded region', text: 'Text', arrow: 'Arrow' }[d.kind]), h('span', { class: 'sp' }),
        h('input', { type: 'color', value: d.color || '#7a4fd6', oninput: (e) => upd({ color: e.target.value }) }),
        h('button', { class: 'ibtn', title: 'Delete', onclick: () => { commit('delete item'); shapes.splice(i, 1); emit(); } }, icon('trash', 14))),
      field('Text', input(d.text || '', (v) => upd({ text: v }))), h('div', { class: 'g2' }, pos)));
  });
  return out;
}

function styleTab(f) {
  const st = baseStyle(f.style);
  const set = (patch, label = 'style') => { commit(label); Object.assign(f.style, patch); emit(); };
  return h('div', { style: 'display:flex;flex-direction:column;gap:10px' },
    field('Preset', seg([['presentation', 'Presentation'], ['paper', 'Paper'], ['dense', 'Dense']], st.preset, (v) => applyPreset(f, v))),
    h('div', { class: 'g2' }, field('Export width (in)', input(st.width_in, (v) => set({ width_in: num(v) || 8.5 }), { numeric: true })), field('Export height (in)', input(st.height_in, (v) => set({ height_in: num(v) || 5.5 }), { numeric: true })),
      field('Font', select(['Arial', 'Helvetica', 'Times New Roman', 'Calibri', 'Segoe UI', 'Courier New'], st.font, (v) => set({ font: v }))), field('Base size (px)', input(st.base, (v) => set({ base: num(v) || 14 }), { numeric: true }))),
    field('DPI', input(st.dpi, (v) => set({ dpi: num(v) || 300 }), { numeric: true })),
    field('Background', seg([['white', 'White'], ['transparent', 'Transparent']], st.bg, (v) => set({ bg: v }))),
    kindOf(f) === '1d' ? check('Residual panel', f.layers.res, (v) => { f.layers.res = v; emit(); }) : null,
    h('div', { class: 'note' }, 'Grid, box frame (mirrored ticks) and tick direction are in Axes & ticks.'),
    h('span', { class: 'lbl' }, 'Reuse this look'), styleTools(f),
    h('div', { class: 'note' }, 'Edits apply to this file. Copy/paste, templates and the "same type" and default buttons carry only the look (fonts, sizes, palette, grid, ticks, legend/results layout, axis formatting), never titles, limits or annotations.'));
}
