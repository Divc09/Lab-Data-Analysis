// Plot editor: in-place series popover, full editor drawer, and syncing of on-plot edits.
import { call } from './api.js';
import { S, emit, commit } from './state.js';
import { h, mount, icon, seg, select, input, check, field, fmt, num, toast, clone } from './util.js';
import { PALETTES, PRESETS, baseStyle, seriesStyle, resultLines } from './plotstyle.js';
import { currentFigure, plotDiv, kindOf } from './analyze.js';

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
      h('div', { class: 'g3' }, field('Width', input(s.width, (v) => set({ width: num(v) ?? 1.5 }))), field('Marker size', input(s.size, (v) => set({ size: num(v) ?? 6 }))), field('Opacity', input(s.opacity, (v) => set({ opacity: Math.min(1, Math.max(0.05, num(v) ?? 1)) })))),
      h('div', { class: 'row wrap' }, check('On plot', s.visible !== false, (v) => set({ visible: v })), check('In legend', s.legend !== false, (v) => set({ legend: v })),
        h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: () => { commit('reset series'); delete f.style.series[key]; emit(); draw(); } }, 'Reset')));
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
  let handled = false;
  if (kind === 'restyle') {
    const [patch, idx] = ev;
    if (patch?.name && idx) {
      const t = el.data[idx[0]];
      const key = t?.meta?.key;
      if (key) { st.series = st.series || {}; st.series[key] = { ...(st.series[key] || {}), name: patch.name[0] }; }
    }
    return true;
  }
  const lay = plotDiv().layout || {};
  for (const [k, v] of Object.entries(ev)) {
    let m;
    if (k === 'title.text') { st.title = v; handled = true; }
    else if ((m = k.match(/^(x|y)axis(\d?)\.title\.text$/))) { const ax = m[1] === 'x' ? 'x' : (m[2] === '2' ? 'y2' : 'y'); st[ax] = { ...(st[ax] || {}), title: v }; handled = true; }
    else if (k === 'legend.x' || k === 'legend.y') { st.legend = { ...(st.legend || {}), [k.slice(7)]: v }; handled = true; }
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
  if (handled) setTimeout(() => emit('insp'), 0);
  return handled;
}

export function applyPreset(f, name) {
  commit('style preset');
  const p = PRESETS[name];
  Object.assign(f.style, { preset: name, font: p.font, base: p.base, width_in: p.width_in, height_in: p.height_in, line: p.line, marker: p.marker });
  emit();
}

// ------------------------------------------------------------ editor drawer
export function editorPanel(f) {
  const tab = S.editor.tab;
  const tabs = [['series', 'Series'], ['legend', 'Legend'], ['axes', 'Axes & ticks'], ['text', 'Text & results'], ['style', 'Layout & style']];
  const body = h('div', { class: 'insp-body' });
  body.append({ series: seriesTab, legend: legendTab, axes: axesTab, text: textTab, style: styleTab }[tab](f));
  return h('frag', {},
    h('div', { class: 'row', style: 'padding:12px 12px 0' }, h('span', { style: 'font-weight:600' }, 'Plot editor'), h('span', { class: 'sp' }),
      h('button', { class: 'btn sm', title: 'Undo (Ctrl+Z)', onclick: () => import('./state.js').then((m) => m.undo()) }, icon('undo', 13)),
      h('button', { class: 'btn sm primary', onclick: () => { S.editor.open = false; emit(); } }, 'Done')),
    h('div', { class: 'row wrap', style: 'padding:8px 12px 0;gap:4px' }, h('span', { class: 'lbl' }, 'Add'),
      ['Text', 'Arrow', 'V line', 'H line', 'Region', 'Fit value'].map((k) => h('button', { class: 'btn sm', onclick: (e) => addDecoration(f, k, e) }, k))),
    h('div', { class: 'etabs', role: 'tablist' }, tabs.map(([k, l]) => h('button', { class: tab === k ? 'on' : '', onclick: () => { S.editor.tab = k; emit('insp'); } }, l))),
    body,
    h('div', { class: 'row wrap', style: 'padding:10px 12px;border-top:1px solid var(--line)' },
      h('button', { class: 'btn sm', onclick: async () => { S.settings = await call('settings_set', { patch: { plot_style: { ...clone(f.style), shapes: [], title: '', x: {}, y: {}, y2: {} } } }); toast('Saved as your default style for new files.', 'ok'); } }, 'Save as my default'),
      h('button', { class: 'btn sm', onclick: () => { commit('reset style'); f.style = clone(S.settings.plot_style || {}); emit(); } }, 'Reset to default'),
      h('button', { class: 'btn sm', title: 'Apply this presentation to every open file of the same type', onclick: () => applyToType(f) }, 'Apply to same type')));
}

function applyToType(f) {
  commit('apply style');
  let n = 0;
  for (const o of Object.values(S.files)) if (o !== f && o.info.type === f.info.type) { o.style = { ...clone(f.style), title: o.style.title || '' }; n++; }
  toast(`Applied to ${n} other ${f.info.type} file(s).`, 'ok');
  emit();
}

function addDecoration(f, kind, ev) {
  const el = plotDiv();
  const xa = el.layout?.xaxis?.range || f.xrange || [0, 1];
  const ya = el.layout?.yaxis?.range || f.yrange || [0, 1];
  const cx = (xa[0] + xa[1]) / 2, cy = (ya[0] + ya[1]) / 2;
  f.style.shapes = f.style.shapes || [];
  const pinX = f.pins?.[0]?.x;
  if (kind === 'Fit value') {
    const fit = f.fit[f.detector];
    if (!fit) { toast('Run a fit first.'); return; }
    const m = fit.metrics || {};
    const opts = [['programmed_pi_time_ns', 'π pulse (programmed)'], ['programmed_pi_over_2_time_ns', 'π/2 pulse (programmed)'], ['first_peak_ns', 'First peak'], ['delay_ns', 'Delay']]
      .filter(([k]) => typeof m[k] === 'number').map(([k, l]) => ({ label: `${l} · ${fmt(m[k], 5)}`, x: m[k], text: `${l} ${fmt(m[k], 4)}` }));
    fit.params.filter((p) => /^(x0|center)/.test(p.name)).forEach((p) => opts.push({ label: `${p.name} · ${fmt(p.value, 6)}`, x: p.value, text: `${p.name} = ${fmt(p.value, 6)}` }));
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
  return h('div', { style: 'display:flex;flex-direction:column;gap:10px' },
    h('div', { class: 'note' }, 'Colour · label · on plot · in legend. Click the colour for line, marker and width options.'),
    h('div', {}, list.map((s) => {
      const o = f.style.series?.[s.key] || {};
      return h('div', { class: 'srow' },
        h('button', { class: 'swatch', style: `background:${o.color || s.color};width:22px;height:18px`, title: 'Style this series', onclick: (e) => openSeriesEditor(f, s.key, e.clientX - 330, e.clientY) }),
        input(o.name ?? s.name, (v) => updSeries(f, s.key, { name: v }), { style: 'height:26px' }),
        h('input', { type: 'checkbox', checked: o.visible !== false, title: 'On plot', onchange: (e) => updSeries(f, s.key, { visible: e.target.checked }) }),
        h('input', { type: 'checkbox', checked: o.legend !== false, title: 'In legend', onchange: (e) => updSeries(f, s.key, { legend: e.target.checked }) }));
    })),
    field('Colour palette', select([['lab', 'Lab default (blue, orange, teal, magenta)'], ['okabe', 'Colour-blind safe (Okabe–Ito)'], ['gray', 'Grayscale']], st.palette, (v) => { commit('palette'); f.style.palette = v; emit(); })),
    kindOf(f) === '1d' ? field('Uncertainty display', seg([['band', 'Band'], ['bars', 'Error bars']], f.errorMode, (v) => { f.errorMode = v; f.layers.band = true; emit(); })) : null,
    h('div', { class: 'g2' }, field('Default line width', input(st.line, (v) => { commit('line width'); f.style.line = num(v) ?? 1.6; emit(); })), field('Default marker size', input(st.marker, (v) => { commit('marker size'); f.style.marker = num(v) ?? 6; emit(); }))));
}

function legendTab(f) {
  const st = baseStyle(f.style);
  const L = st.legend;
  const set = (patch) => { commit('legend'); f.style.legend = { ...(f.style.legend || {}), ...patch }; emit(); };
  return h('div', { style: 'display:flex;flex-direction:column;gap:10px' },
    check('Show legend', f.layers.legend, (v) => { f.layers.legend = v; emit(); }),
    field('Position', seg([['ul', 'Upper left'], ['ur', 'Upper right'], ['ll', 'Lower left'], ['lr', 'Lower right'], ['outside', 'Outside'], ['below', 'Below']], L.x != null ? '' : L.pos, (v) => set({ pos: v, x: null, y: null }))),
    h('div', { class: 'note' }, 'Or drag the legend on the plot; the position is remembered.'),
    h('div', { class: 'g2' }, field('Columns', input(L.cols || 1, (v) => set({ cols: Math.max(1, num(v) || 1) }))), field('Font size (px)', input(L.size || '', (v) => set({ size: num(v) }), { placeholder: 'auto' }))),
    check('Frame', L.frame !== false, (v) => set({ frame: v })));
}

function axisCard(f, ax, label) {
  const a = f.style[ax] || {};
  const set = (patch) => { commit('axis'); f.style[ax] = { ...(f.style[ax] || {}), ...patch }; emit(); };
  return h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' }, h('span', { style: 'font-weight:600' }, label),
    field('Label', input(a.title ?? '', (v) => set({ title: v === '' ? null : v }), { placeholder: 'automatic' })),
    h('div', { class: 'g3' }, field('Min', input(a.min ?? '', (v) => set({ min: num(v) }), { placeholder: 'auto' })), field('Max', input(a.max ?? '', (v) => set({ max: num(v) }), { placeholder: 'auto' })),
      field('Scale', select([['', 'Linear'], ['log', 'Log']], a.log ? 'log' : '', (v) => set({ log: v === 'log' })))),
    h('div', { class: 'g3' }, field('Major every', input(a.dtick ?? '', (v) => set({ dtick: num(v) }), { placeholder: 'auto' })), field('Minor per major', input(a.minor ?? '', (v) => set({ minor: num(v) }), { placeholder: 'off' })),
      field('Format', select([['auto', 'Auto'], ['fixed1', '0.0'], ['fixed2', '0.00'], ['fixed3', '0.000'], ['sci', 'Scientific'], ['eng', 'Engineering (k, M)'], ['percent', 'Percent']], a.format || 'auto', (v) => set({ format: v })))));
}

function axesTab(f) {
  const st = baseStyle(f.style);
  return h('div', { style: 'display:flex;flex-direction:column;gap:10px' },
    field('Title', input(f.style.title || '', (v) => { commit('title'); f.style.title = v; emit(); }, { placeholder: 'None' })),
    axisCard(f, 'x', 'X axis'), axisCard(f, 'y', 'Y axis'),
    f.layers.sigref || f.compareDetectors ? axisCard(f, 'y2', 'Right Y axis') : h('div', { class: 'note' }, 'A right-hand axis appears automatically when Signal + Reference counts or a second detector in counts are shown.'),
    field('Ticks', seg([['outside', 'Outside'], ['inside', 'Inside'], ['none', 'None']], st.ticks, (v) => { commit('ticks'); f.style.ticks = v; emit(); })),
    h('div', { class: 'row wrap' }, check('Mirror ticks on top/right', st.mirror, (v) => { commit('mirror'); f.style.mirror = v; emit(); }), check('Grid', st.grid, (v) => { commit('grid'); f.style.grid = v; emit(); })));
}

function textTab(f) {
  const st = baseStyle(f.style);
  const fit = f.fit[f.detector];
  const lines = fit ? resultLines(fit, f.info) : [];
  const cfg = st.results;
  const setLine = (key, patch) => { commit('results line'); f.style.results = { ...(f.style.results || {}), lines: { ...(cfg.lines || {}), [key]: { ...(cfg.lines?.[key] || {}), ...patch } } }; emit(); };
  const out = h('div', { style: 'display:flex;flex-direction:column;gap:10px' });
  out.append(check('Show results box on the plot', f.layers.results, (v) => { f.layers.results = v; emit(); }, { disabled: !fit }));
  if (lines.length) out.append(h('div', { class: 'card', style: 'padding:4px 10px' }, lines.map((l) => h('div', { class: 'srow', style: 'grid-template-columns:minmax(0,1fr) 22px' },
    input(cfg.lines?.[l.key]?.label ?? l.label, (v) => setLine(l.key, { label: v }), { style: 'height:26px', title: `${l.label} = ${fmt(l.value, 5)} ${l.unit}` }),
    h('input', { type: 'checkbox', checked: cfg.lines?.[l.key]?.plot ?? !l.defaultOff, title: 'Show on plot', onchange: (e) => setLine(l.key, { plot: e.target.checked }) })))));
  else out.append(h('div', { class: 'note' }, 'Fit values appear here after a fit and update after every refit.'));
  out.append(h('div', { class: 'g2' }, field('Number format', select([['pm', 'Value ± error'], ['sig4', '4 significant']], cfg.format, (v) => { commit('format'); f.style.results = { ...(f.style.results || {}), format: v }; emit(); })),
    field('Box position', select([['ul', 'Upper left'], ['ur', 'Upper right'], ['ll', 'Lower left'], ['lr', 'Lower right']], cfg.pos, (v) => { commit('results position'); f.style.results = { ...(f.style.results || {}), pos: v, x: null, y: null }; emit(); }))));
  out.append(h('span', { class: 'lbl', style: 'margin-top:6px' }, 'Added items'));
  const shapes = f.style.shapes || [];
  if (!shapes.length) out.append(h('div', { class: 'note' }, 'Use Add above for text, arrows, reference lines, shaded regions and fit-value lines.'));
  shapes.forEach((d, i) => {
    const upd = (patch) => { commit('edit item'); Object.assign(d, patch); emit(); };
    const pos = d.kind === 'vline' ? [field('x', input(d.x, (v) => upd({ x: num(v) })))] : d.kind === 'hline' ? [field('y', input(d.y, (v) => upd({ y: num(v) })))]
      : d.kind === 'region' ? [field('from x', input(d.x0, (v) => upd({ x0: num(v) }))), field('to x', input(d.x1, (v) => upd({ x1: num(v) })))]
        : [field('x', input(d.x, (v) => upd({ x: num(v) }))), field('y', input(d.y, (v) => upd({ y: num(v) })))];
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
    h('div', { class: 'g2' }, field('Export width (in)', input(st.width_in, (v) => set({ width_in: num(v) || 8.5 }))), field('Export height (in)', input(st.height_in, (v) => set({ height_in: num(v) || 5.5 }))),
      field('Font', select(['Arial', 'Helvetica', 'Times New Roman', 'Calibri', 'Segoe UI', 'Courier New'], st.font, (v) => set({ font: v }))), field('Base size (px)', input(st.base, (v) => set({ base: num(v) || 14 })))),
    field('DPI', input(st.dpi, (v) => set({ dpi: num(v) || 300 }))),
    field('Background', seg([['white', 'White'], ['transparent', 'Transparent']], st.bg, (v) => set({ bg: v }))),
    h('div', { class: 'row wrap' }, check('Grid', st.grid, (v) => set({ grid: v })), check('Box frame', st.mirror, (v) => set({ mirror: v })),
      kindOf(f) === '1d' ? check('Residual panel', f.layers.res, (v) => { f.layers.res = v; emit(); }) : null),
    h('div', { class: 'note' }, 'Edits apply to this file. Use "Apply to same type" or "Save as my default" below to reuse them.'));
}
