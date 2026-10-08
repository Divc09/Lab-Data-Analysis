// Compare workspace: overlay several files and track fitted values across them.
import { call } from './api.js';
import { S, emit, openFile, detLabel } from './state.js';
import { h, mount, seg, select, input, field, fmt, fmtPm, toast, modal, basename, tag, num, linreg, fmtTime, clone } from './util.js';
import { PALETTES, baseStyle, friendlyAxis, MODE_LABEL } from './plotstyle.js';
import { copyText, renderImage, copyImage } from './export.js';
import { resultLines } from './plotstyle.js';

let overlayEl, trendEl, alignNoteEl;
const C = () => S.compare;

function colorFor(i) { return PALETTES.lab[i % PALETTES.lab.length]; }

// Offset/scale so that (y - off) / scale is the normalized curve; the same transform is applied to fits.
function normParams(x, y, how) {
  const fin = y.filter(Number.isFinite);
  if (!fin.length || how === 'none') return { off: 0, scale: 1 };
  if (how === 'max') return { off: 0, scale: Math.max(...fin.map(Math.abs)) || 1 };
  if (how === 'minmax') { const lo = Math.min(...fin), hi = Math.max(...fin); return { off: lo, scale: (hi - lo) || 1 }; }
  let a = 0;
  for (let i = 1; i < x.length; i++) if (Number.isFinite(y[i]) && Number.isFinite(y[i - 1])) a += Math.abs(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]));
  return { off: 0, scale: a || 1 };
}

// x-offset (added to x) that aligns file `p`; null when the chosen alignment is unavailable for it.
function alignShift(p) {
  const c = C(), how = c.align || 'none';
  if (how === 'none') return 0;
  if (how === 'peak') {
    const d = S.files[p]?.data[S.files[p].detector];
    if (!d?.proc) return null;
    let best = -Infinity, bx = null;
    d.proc.y.forEach((v, i) => { const a = v === null || !Number.isFinite(v) ? NaN : (c.norm === 'max' ? Math.abs(v) : v); if (a > best && Number.isFinite(d.proc.x[i])) { best = a; bx = d.proc.x[i]; } });
    return bx === null ? null : -bx;
  }
  const r = c.results[p];
  if (!r) return null;
  const par = (r.params || []).find((q) => ['center_1', 'x0', 'x0_1'].includes(q.name) && Number.isFinite(q.value));
  const v = par ? par.value : r.metrics?.center_freq_units;
  return Number.isFinite(v) ? -v : null;
}

function trendValues(path) {
  const r = C().results[path];
  if (!r) return {};
  const out = {};
  for (const p of r.params) { out[p.name] = { v: p.value, e: p.error }; if (p.name.startsWith('sigma_')) out[`FWHM${p.name.slice(5)}`] = { v: p.value * 2.354820045, e: p.error ? p.error * 2.354820045 : null }; }
  for (const [k, v] of Object.entries(r.metrics || {})) if (typeof v === 'number' && !k.endsWith('_err')) out[k] = { v, e: r.metrics[`${k}_err`] ?? null };
  out.r2 = { v: r.r2, e: null };
  return out;
}

function xValue(path, i) {
  const c = C();
  const info = S.library.peeks[path] || {};
  if (c.against === 'order') return i + 1;
  if (c.against === 'name') {
    const pref = c.namePrefix || '';
    const re = pref ? new RegExp(`${pref.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(-?\\d+(?:[._]\\d+)?)`) : /(-?\d+(?:[._]\d+)?)(?!.*\d)/;
    const m = basename(path).replace(/\.mat$/i, '').match(re);
    return m ? Number(m[1].replace('_', '.')) : NaN;
  }
  if (c.against === 'meta') { const v = Number((S.files[path]?.info.metadata || {})[c.metaKey]); return Number.isFinite(v) ? v : NaN; }
  return (info.mtime || 0) * 1000; // acquisition (file save) time
}

export function renderCompare(root) {
  const c = C();
  c.hidden = c.hidden || [];
  c.spec = c.spec || { model: 'Auto', peaks: 1 };
  c.sel = c.sel || null;
  c.align = c.align || 'none';
  alignNoteEl = h('div', { class: 'note' });
  const set = c.set.filter((p) => S.files[p]);
  overlayEl = overlayEl || h('div', { class: 'plot' });
  trendEl = trendEl || h('div', { class: 'plot' });
  const fitted = set.filter((p) => c.results[p]);
  const keys = [...new Set(fitted.flatMap((p) => Object.keys(trendValues(p))))];
  if (!c.param || !keys.includes(c.param)) c.param = keys.find((k) => /center_1|pi_time_ns$|T2_ns|T1_ns|T2_star_ns|center_freq|tau_s/.test(k)) || keys[0] || '';
  const left = h('aside', { class: 'library', style: 'padding:12px;gap:10px' },
    h('div', { class: 'row' }, h('span', { class: 'lbl' }, `Compare set · ${set.length} files`), h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: addFromLibrary }, 'Add files')),
    h('div', { class: 'note' }, 'Tick to draw in the overlay; every fitted file feeds the trend. Files keep their own detector and observable.'),
    h('div', { style: 'flex:1;overflow:auto;display:flex;flex-direction:column;gap:2px' }, set.map((p, i) => {
      const f = S.files[p], r = c.results[p];
      const tv = trendValues(p)[c.param];
      const sh = c.align === 'none' ? null : alignShift(p);
      const shTxt = sh === null ? '' : ` · shift ${sh < 0 ? '−' : '+'}${fmt(Math.abs(sh), 6)}`;
      return h('label', { class: 'frow', style: 'cursor:default' },
        h('input', { type: 'checkbox', checked: !c.hidden.includes(p), onchange: (e) => { c.hidden = e.target.checked ? c.hidden.filter((q) => q !== p) : [...c.hidden, p]; emit(); } }),
        h('span', { style: `width:12px;height:12px;border-radius:3px;flex:none;background:${c.hidden.includes(p) ? '#3A4250' : colorFor(i)}` }),
        h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'fname ellip', title: p }, basename(p).replace(/\.mat$/i, '')),
          h('div', { class: 'sub ellip' }, r ? `${c.param}: ${tv ? fmtPm(tv.v, tv.e) : '–'} · R² ${fmt(r.r2, 3)}${r.status !== 'PASS' ? ' · ' + r.status : ''}${shTxt}` : `${detLabel(f, f.detector)} · ${MODE_LABEL[f.mode]}${shTxt}`)),
        h('button', { class: 'ibtn', title: 'Remove', onclick: (e) => { e.preventDefault(); c.set = c.set.filter((q) => q !== p); emit(); } }, '×'));
    })),
    set.length ? null : h('div', { class: 'note' }, 'Add files here, or right-click files in the Library and choose "Add to Compare set". Batch results can also be sent here.'));

  const main = h('main', { class: 'main' },
    h('div', { class: 'row wrap' }, h('h1', { style: 'margin:0;font-size:17px;font-weight:600;margin-right:auto' }, 'Overlay'),
      h('span', { class: 'lbl' }, 'Normalize'), seg([['none', 'None'], ['max', 'To max'], ['minmax', 'Min–max'], ['area', 'Area']], c.norm, (v) => { c.norm = v; emit(); }),
      h('span', { class: 'lbl' }, 'Align'), seg([['none', 'None'], ['peak', 'Peak'], ['fit', 'Fit center']], c.align, (v) => { c.align = v; emit(); }),
      h('span', { class: 'lbl' }, 'Layout'), seg([['overlaid', 'Overlaid'], ['waterfall', 'Waterfall'], ['grid', 'Grid']], c.layout, (v) => { c.layout = v; emit(); }),
      h('button', { class: 'btn', onclick: () => renderImage({ data: overlayEl.data, layout: overlayEl.layout }, { width_in: 8.5, height_in: 5, dpi: 300 }).then(copyImage) }, 'Copy')),
    h('section', { class: 'plotcard', style: 'flex:1.2;min-height:300px' }, overlayEl),
    alignNoteEl,
    h('div', { class: 'row wrap' }, h('h2', { style: 'margin:0;font-size:15px;font-weight:600;margin-right:auto' }, 'Trend across files'),
      h('span', { class: 'lbl' }, 'Plot'), select(keys.length ? keys : [['', 'Fit files first']], c.param, (v) => { c.param = v; c.sel = null; emit(); }, { style: 'width:190px' }),
      h('span', { class: 'lbl' }, 'Against'), select([['time', 'Acquisition time'], ['order', 'File order'], ['name', 'Number in file name'], ['meta', 'Saved parameter']], c.against, (v) => { c.against = v; emit(); }, { style: 'width:170px' }),
      c.against === 'name' ? input(c.namePrefix || '', (v) => { c.namePrefix = v; emit(); }, { placeholder: 'prefix, e.g. Z', style: 'width:110px' }) : null,
      c.against === 'meta' ? select(Object.keys(S.files[set[0]]?.info.metadata || {}), c.metaKey, (v) => { c.metaKey = v; emit(); }, { style: 'width:160px' }) : null,
      h('button', { class: 'btn', onclick: () => renderImage({ data: trendEl.data, layout: trendEl.layout }, { width_in: 6, height_in: 3.6, dpi: 300 }).then(copyImage) }, 'Copy')),
    h('section', { class: 'plotcard', style: 'flex:1;min-height:240px' }, trendEl));

  const pts = fitted.map((p, i) => ({ p, x: xValue(p, set.indexOf(p)), ...(trendValues(p)[c.param] || {}) })).filter((q) => Number.isFinite(q.x) && Number.isFinite(q.v));
  const sel = c.sel ? pts.filter((q) => c.sel.includes(q.p)) : pts;
  const reg = sel.length >= 2 ? linreg(sel.map((q) => q.x), sel.map((q) => q.v)) : null;
  const perUnit = c.against === 'time' ? 60000 : 1;
  const right = h('aside', { class: 'inspector' }, h('div', { class: 'insp-body' },
    h('span', { class: 'lbl' }, 'Fitting the set'),
    h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' },
      field('Model', select([['Auto', 'Auto (each file\'s type)'], ['GaussianPeaks', 'Gaussian peaks'], ['useOwn', 'Each file\'s own Fit settings']], c.spec.model, (v) => { c.spec.model = v; emit(); })),
      c.spec.model === 'GaussianPeaks' ? field('Peaks', select([1, 2, 3, 4], c.spec.peaks, (v) => { c.spec.peaks = Number(v); emit(); })) : null,
      h('div', { class: 'g2' }, field('Range from', input(c.spec.range_min ?? '', (v) => { c.spec.range_min = num(v); }, { placeholder: 'all' })), field('to', input(c.spec.range_max ?? '', (v) => { c.spec.range_max = num(v); }, { placeholder: 'all' }))),
      h('div', { class: 'note' }, 'The same model, range and processing are applied to every file; each keeps its own result.'),
      h('button', { class: 'btn primary block', disabled: !set.length || c.fitting, onclick: fitAll }, c.fitting ? `Fitting ${c.fitting}` : `Fit all ${set.length} files`)),
    h('div', { class: 'card pad' }, h('div', { class: 'row' }, h('span', { style: 'font-weight:600' }, 'Trend table'), h('span', { class: 'sp' }), h('button', { class: 'btn sm', disabled: !pts.length, onclick: () => copyText(trendTable(pts), 'Trend table copied') }, 'Copy'), h('button', { class: 'btn sm', disabled: !pts.length, onclick: () => saveCsv(pts) }, 'Save CSV')),
      pts.length ? h('table', { class: 'tbl', style: 'margin-top:6px' }, h('tr', {}, h('th', {}, 'File'), h('th', {}, c.param), h('th', {}, '±')),
        pts.map((q) => h('tr', { class: `click ${c.sel?.includes(q.p) ? 'sel' : ''}`, onclick: () => toggleSel(q.p, pts) }, h('td', { class: 'txt ellip', style: 'max-width:140px', title: q.p }, c.against === 'time' ? fmtTime(q.x / 1000) : basename(q.p).slice(0, 18)), h('td', {}, fmt(q.v, 6)), h('td', {}, q.e ? fmt(q.e, 2) : '–'))))
        : h('div', { class: 'note' }, 'Run Fit all to fill the table.')),
    reg ? h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:6px' }, h('span', { style: 'font-weight:600' }, `Linear trend · ${sel.length} points`),
      h('div', { class: 'mono', style: 'font-size:18px' }, `${fmt(reg.slope * perUnit, 4)}`, h('small', { class: 'muted', style: 'font-size:12px' }, ` ${c.param} per ${c.against === 'time' ? 'min' : 'unit'}`)),
      reg.slope_err ? h('div', { class: 'note' }, `± ${fmt(reg.slope_err * perUnit, 2)} (standard error, unweighted)`) : null,
      h('div', { class: 'note' }, 'Click rows (or points) to choose which files define the trend.')) : null));
  mount(root, left, main, right);
  drawOverlay(set);
  drawTrend(pts, reg);
}

function toggleSel(p, pts) {
  const c = C();
  const cur = new Set(c.sel || pts.map((q) => q.p));
  cur.has(p) ? cur.delete(p) : cur.add(p);
  c.sel = [...cur];
  emit();
}

function trendTable(pts, sep = '\t') {
  const c = C();
  const esc = (v) => (sep === ',' && /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
  return [['File', c.against === 'time' ? 'Time' : 'x', c.param, 'error'].join(sep), ...pts.map((q) => [basename(q.p), c.against === 'time' ? new Date(q.x).toLocaleString() : q.x, q.v, q.e ?? ''].map(esc).join(sep))].join('\n');
}

async function saveCsv(pts) {
  const p = await call('dialog', { kind: 'save', title: 'Save trend table', filename: 'trend.csv' });
  if (!p) return;
  await call('write_text', { path: p.endsWith('.csv') ? p : p + '.csv', text: trendTable(pts, ',') });
  toast('Saved trend table', 'ok');
}

function drawOverlay(set) {
  const c = C();
  const vis = set.filter((p) => !c.hidden.includes(p));
  const data = [];
  const layout = { separators: '.', margin: { l: 64, r: 16, t: 14, b: 50 }, font: { family: 'Arial', size: 12, color: '#1D2330' }, showlegend: true, legend: { x: 0.99, xanchor: 'right', y: 0.99, bgcolor: 'rgba(255,255,255,.9)', bordercolor: '#D9DDE4', borderwidth: 1 }, hovermode: 'closest' };
  let xLabel = '';
  const unaligned = [];
  const n = vis.length;
  const cols = Math.ceil(Math.sqrt(n)) || 1, rows = Math.ceil(n / cols) || 1;
  vis.forEach((p) => {
    const i = set.indexOf(p);
    const f = S.files[p];
    const d = f.data[f.detector];
    if (!d?.proc) return;
    const xBase = friendlyAxis(f.info.x_label);
    xLabel = c.align === 'peak' ? `Δ${xBase} from peak` : c.align === 'fit' ? `Δ${xBase} from fit center` : xBase;
    let sh = alignShift(p);
    if (sh === null) { unaligned.push(basename(p).replace(/\.mat$/i, '')); sh = 0; }
    const x = d.proc.x.map((v) => (v === null ? NaN : v + sh));
    const yRaw = d.proc.y.map((v) => (v === null ? NaN : v));
    const np_ = normParams(x, yRaw, c.norm);
    let y = yRaw.map((v) => (v - np_.off) / np_.scale);
    const k = vis.indexOf(p);
    if (c.layout === 'waterfall') y = y.map((v) => v + k * (c.norm === 'none' ? 0 : 1.1));
    const ax = c.layout === 'grid' && k ? String(k + 1) : '';
    data.push({ type: 'scatter', mode: 'lines+markers', x, y, name: basename(p).replace(/\.mat$/i, ''), line: { color: colorFor(i), width: 1.4 }, marker: { size: 4, color: colorFor(i) }, xaxis: `x${ax}`, yaxis: `y${ax}` });
    const r = c.results[p];
    if (r && (r.dense || r.y_fit)) {
      let fy = (r.dense?.y || r.y_fit).map((v) => (v === null ? NaN : v));
      fy = fy.map((v) => (v - np_.off) / np_.scale);
      if (c.layout === 'waterfall') fy = fy.map((v) => v + k * (c.norm === 'none' ? 0 : 1.1));
      data.push({ type: 'scatter', mode: 'lines', x: (r.dense?.x || r.x).map((v) => (v === null ? NaN : v + sh)), y: fy, name: 'fit', showlegend: false, line: { color: colorFor(i), width: 2, dash: 'dot' }, xaxis: `x${ax}`, yaxis: `y${ax}` });
    }
    if (c.layout === 'grid') {
      const col = k % cols, row = Math.floor(k / cols);
      const gx = 0.06, gy = 0.1;
      layout[`xaxis${ax}`] = { domain: [col / cols + (col ? gx / 2 : 0), (col + 1) / cols - gx / 2], anchor: `y${ax}`, showline: true, mirror: true, ticks: 'outside', title: { text: row === rows - 1 ? xLabel : '' } };
      layout[`yaxis${ax}`] = { domain: [1 - (row + 1) / rows + gy / 2, 1 - row / rows - (row ? gy / 2 : 0)], anchor: `x${ax}`, showline: true, mirror: true, ticks: 'outside' };
    }
  });
  if (c.layout !== 'grid') {
    layout.xaxis = { title: { text: xLabel }, exponentformat: 'none', showline: true, mirror: true, ticks: 'outside', showgrid: true, gridcolor: '#ECEEF2', zeroline: false };
    layout.yaxis = { title: { text: c.norm === 'none' ? 'Signal' : `Normalized (${c.norm})${c.layout === 'waterfall' ? ', offset' : ''}` }, showline: true, mirror: true, ticks: 'outside', showgrid: true, gridcolor: '#ECEEF2', zeroline: false, exponentformat: 'SI' };
  } else layout.showlegend = false;
  alignNoteEl.textContent = unaligned.length ? `not aligned: ${unaligned.join(', ')}` : '';
  Plotly.react(overlayEl, data, layout, { responsive: true, displayModeBar: false, scrollZoom: true, edits: { axisTitleText: true, legendPosition: true } });
}

function passed(p) { return C().results[p]?.status === 'PASS'; }

function drawTrend(pts, reg) {
  const c = C();
  const isTime = c.against === 'time';
  const xs = pts.map((q) => (isTime ? new Date(q.x) : q.x));
  const sel = c.sel || pts.map((q) => q.p);
  const data = [{ type: 'scatter', mode: 'markers', x: xs, y: pts.map((q) => q.v), error_y: { type: 'data', array: pts.map((q) => q.e || 0), visible: true, color: '#1D2330', thickness: 1.2, width: 4 },
    marker: { size: 10, color: pts.map((q) => (sel.includes(q.p) ? '#7A4FD6' : '#FFFFFF')), line: { color: pts.map((q) => (passed(q.p) ? '#1D2330' : '#C0392B')), width: pts.map((q) => (passed(q.p) ? 1 : 2)) } }, text: pts.map((q) => `${basename(q.p)} · ${C().results[q.p]?.status ?? 'no status'}`), hovertemplate: '%{text}<br>%{y:.6g}<extra></extra>', customdata: pts.map((q) => q.p), name: c.param }];
  if (reg) {
    const sx = pts.filter((q) => sel.includes(q.p)).map((q) => q.x);
    const lo = Math.min(...sx), hi = Math.max(...sx);
    data.push({ type: 'scatter', mode: 'lines', x: isTime ? [new Date(lo), new Date(hi)] : [lo, hi], y: [reg.intercept + reg.slope * lo, reg.intercept + reg.slope * hi], line: { color: '#7A4FD6', dash: 'dash', width: 1.4 }, name: 'linear trend', hoverinfo: 'skip' });
  }
  const layout = { separators: '.', margin: { l: 70, r: 16, t: 10, b: 50 }, font: { family: 'Arial', size: 12 }, showlegend: false,
    xaxis: { title: { text: { time: 'Acquisition time', order: 'File order', name: `Value in file name${c.namePrefix ? ` (${c.namePrefix})` : ''}`, meta: c.metaKey || 'Saved parameter' }[c.against] }, showline: true, mirror: true, ticks: 'outside', showgrid: true, gridcolor: '#ECEEF2', type: isTime ? 'date' : 'linear' },
    yaxis: { title: { text: c.param }, showline: true, mirror: true, ticks: 'outside', showgrid: true, gridcolor: '#ECEEF2', exponentformat: 'none' } };
  Plotly.react(trendEl, data, layout, { responsive: true, displayModeBar: false, edits: { axisTitleText: true } });
  if (!trendEl._bound) {
    trendEl._bound = true;
    trendEl.on('plotly_click', (ev) => { const p = ev.points?.[0]?.customdata; if (p) toggleSel(p, pts.length ? pts : []); });
  }
}

async function fitAll() {
  const c = C();
  const set = c.set.filter((p) => S.files[p]);
  for (const [i, p] of set.entries()) {
    const f = S.files[p];
    c.fitting = `${i + 1}/${set.length}`; emit();
    const spec = c.spec.model === 'useOwn' ? clone(f.fitSpec) : { ...clone(f.fitSpec), model: c.spec.model, peaks: c.spec.peaks, range_min: c.spec.range_min ?? null, range_max: c.spec.range_max ?? null };
    if (f.info.type === 'LineScan' && spec.model === 'Auto') spec.model = 'GaussianPeaks';
    try { c.results[p] = await call('fit', { path: p, detector: f.detector, mode: f.mode, proc: f.proc, spec }, { label: `Fitting ${basename(p)}` }); }
    catch (e) { /* toast shown */ }
  }
  c.fitting = null;
  emit();
}

async function addFromLibrary() {
  const lib = S.library;
  const all = [...new Set([...Object.keys(S.files), ...lib.extra, ...(lib.listing?.files || []).map((f) => f.path)])].filter((p) => (lib.peeks[p] || S.files[p]?.info || {}).dim !== 'scan2d');
  const chosen = new Set();
  const list = h('div', { style: 'overflow:auto;max-height:52vh;padding:6px' }, all.map((p) => {
    const info = lib.peeks[p] || S.files[p]?.info || {};
    return h('label', { class: 'frow' }, h('input', { type: 'checkbox', checked: C().set.includes(p), onchange: (e) => (e.target.checked ? chosen.add(p) : chosen.delete(p)) }), tag(info.tag), h('span', { class: 'ellip' }, basename(p)));
  }));
  const m = modal(h('div', { style: 'display:flex;flex-direction:column' }, h('div', { style: 'padding:12px 14px;font-weight:600' }, 'Add files to the Compare set'), list,
    h('div', { class: 'row', style: 'padding:10px 14px;border-top:1px solid var(--line)' }, h('span', { class: 'note' }, '1D files from the Library and open files.'), h('span', { class: 'sp' }),
      h('button', { class: 'btn primary', onclick: async () => { m.close(); for (const p of chosen) { await openFile(p, { primary: false }); if (!C().set.includes(p)) C().set.push(p); } emit(); } }, 'Add'))));
}
