// Figure composer: multi-panel figures at true physical size (WYSIWYG), exported as SVG/PNG/PDF.
import { call } from './api.js';
import { S, emit, active, sessionDoc } from './state.js';
import { h, mount, icon, seg, select, input, field, check, toast, num, basename, dirname, stem, clone } from './util.js';
import { currentFigure, kindOf } from './analyze.js';
import { composeSvg, svgToPng, copyImage, renderImage } from './export.js';

const SIZES = {
  slide: { w: 13.33, h: 7.5, label: 'Slide 16:9' },
  two: { w: 7.0, h: 2.8, label: 'Paper, two-column' },
  one: { w: 3.4, h: null, label: 'Paper, one-column' },
  custom: { w: null, h: null, label: 'Custom' },
};
const FONT_PT = { presentation: 14, paper: 8, dense: 9 };
let pageEl, pageInner;
let nextId = 1;

// Each panel keeps the comparison set it was added with (S.overlays is global and later belongs to other plots).
const withOverlays = (overlays, fn) => { const saved = S.overlays; S.overlays = overlays; try { return fn(); } finally { S.overlays = saved; } };
export async function addCurrentPlot({ quiet = false, overlays } = {}) {
  const f = active();
  if (!f) { toast('Open a plot first.'); return; }
  const own = Array.isArray(overlays) ? overlays : S.overlays.filter((q) => q !== f.path);
  const fig = withOverlays(own, () => currentFigure(f));
  const thumb = await Plotly.toImage({ data: fig.data, layout: { ...fig.layout, width: 320, height: 210 } }, { format: 'png', width: 320, height: 210 });
  S.figure.panels.push({ id: nextId++, path: f.path, det: f.detector, kind: kindOf(f), title: '', thumb, overlays: own, snapshot: clone({ data: fig.data, layout: fig.layout }) });
  if (!quiet) toast(`Added to Figure composer (${S.figure.panels.length} panel${S.figure.panels.length > 1 ? 's' : ''}).`, 'ok');
  emit();
}

// One panel per open file, each with its own current fit/style (the primary is restored afterwards).
async function addOpenPlots() {
  const prev = S.primary;
  try {
    for (const path of Object.keys(S.files)) { S.primary = path; await addCurrentPlot({ quiet: true, overlays: path === prev ? S.overlays.filter((q) => q !== path) : [] }); }
    toast(`Added ${Object.keys(S.files).length} panels (${S.figure.panels.length} in the figure).`, 'ok');
  } finally { S.primary = prev; emit(); }
}

function panelFigure(p, fontPx) {
  // Linked panels follow the current analysis of their file; otherwise use the snapshot.
  const f = S.files[p.path];
  let fig = p.snapshot;
  if (f && f.data[p.det]) {
    const saved = f.detector; f.detector = p.det;
    try { fig = withOverlays(p.overlays ?? S.overlays.filter((q) => q !== p.path), () => currentFigure(f)) || fig; } finally { f.detector = saved; }
  }
  fig = clone(fig);
  const L = fig.layout;
  L.font = { ...(L.font || {}), family: S.figure.font, size: fontPx };
  L.margin = { l: Math.round(fontPx * 4.2), r: Math.round(fontPx * 1.2), t: Math.round(fontPx * (p.title ? 2.4 : 1.2)), b: Math.round(fontPx * 3.4) };
  if (p.title) L.title = { text: p.title, font: { size: Math.round(fontPx * 1.1) }, x: 0.5 };
  for (const k of Object.keys(L)) if (/^(x|y)axis\d*$/.test(k)) { L[k] = { ...L[k], tickfont: { size: Math.round(fontPx * 0.9) } }; }
  if (L.legend) L.legend = { ...L.legend, font: { size: Math.round(fontPx * 0.85) } };
  for (const t of fig.data) if (t.type === 'heatmap' && t.colorbar) t.colorbar = { ...t.colorbar, tickfont: { size: Math.round(fontPx * 0.8) }, title: { ...(t.colorbar.title || {}), font: { size: Math.round(fontPx * 0.8) } } };
  return fig;
}

function geometry() {
  const F = S.figure;
  const n = Math.max(1, F.panels.length);
  const size = SIZES[F.size] || SIZES.slide;
  const cols = F.cols || (F.size === 'one' ? 1 : Math.min(n, F.size === 'slide' ? (n === 4 ? 2 : 3) : 3));
  const rows = Math.ceil(n / cols);
  const W = F.size === 'custom' ? F.w : size.w;
  const H = F.size === 'custom' ? F.h : (size.h ?? 2.6 * rows);
  return { W, H, cols, rows, Wpx: Math.round(W * 96), Hpx: Math.round(H * 96) };
}

function cells(g) {
  const pad = 8;
  const cw = (g.Wpx - pad * (g.cols + 1)) / g.cols, ch = (g.Hpx - pad * (g.rows + 1)) / g.rows;
  return S.figure.panels.map((p, i) => ({ p, x: pad + (i % g.cols) * (cw + pad), y: pad + Math.floor(i / g.cols) * (ch + pad), w: cw, h: ch }));
}

const label = (i) => ({ a: 'abcdefghij'[i], '(a)': `(${'abcdefghij'[i]})`, A: 'ABCDEFGHIJ'[i], off: '' })[S.figure.labels] || '';

export function renderFigure(root) {
  const F = S.figure;
  const g = geometry();
  const fontPx = Math.round(((F.base || FONT_PT[F.preset] || 9) * 96) / 72);
  pageInner = h('div', { style: `position:relative;width:${g.Wpx}px;height:${g.Hpx}px;background:#fff;transform-origin:top left` });
  pageEl = h('div', { class: 'figpage', style: 'position:relative;overflow:hidden' }, pageInner);
  const preview = h('main', { class: 'main', style: 'align-items:center;justify-content:center;background:#0B0D10' },
    h('div', { class: 'mono dim', style: 'font-size:11.5px' }, `${(SIZES[F.size] || {}).label} · ${g.W} × ${g.H.toFixed(2)} in · ${F.dpi} dpi · ${F.font} ${F.base || FONT_PT[F.preset]} pt`),
    pageEl,
    h('div', { class: 'note' }, F.panels.length ? 'What you see is what is exported: same layout, fonts and sizes. Panels follow their file\'s current fit and plot edits.' : 'Add panels with "Add the current plot", from the plot\'s right-click menu, or the Export tab.'));
  const list = h('div', { style: 'display:flex;flex-direction:column;gap:6px' });
  let dragIdx = null;
  F.panels.forEach((p, i) => {
    const row = h('div', { class: 'prow', draggable: 'true' },
      h('span', { class: 'mono dim', style: 'width:22px' }, label(i) || i + 1),
      h('img', { class: 'thumb', src: p.thumb, alt: '' }),
      h('div', { style: 'min-width:0;flex:1' }, input(p.title, (v) => { p.title = v; emit(); }, { placeholder: basename(p.path).replace(/\.mat$/i, ''), style: 'height:26px' }),
        h('div', { class: 'dim ellip', style: 'font-size:11px;margin-top:2px' }, S.files[p.path] ? 'linked' : 'snapshot')),
      h('button', { class: 'ibtn', title: 'Remove panel', onclick: () => { F.panels.splice(i, 1); emit(); } }, icon('trash', 14)));
    row.addEventListener('dragstart', (e) => { dragIdx = i; row.classList.add('drag'); e.dataTransfer.setData('application/x-sf-series', String(i)); });
    row.addEventListener('dragend', () => row.classList.remove('drag'));
    row.addEventListener('dragover', (e) => e.preventDefault());
    row.addEventListener('drop', (e) => { e.preventDefault(); e.stopPropagation(); if (dragIdx === null || dragIdx === i) return; const [m] = F.panels.splice(dragIdx, 1); F.panels.splice(i, 0, m); dragIdx = null; emit(); });
    list.append(row);
  });
  const left = h('aside', { class: 'library', style: 'padding:12px;gap:10px', 'data-drop': 'figure' }, h('span', { class: 'lbl' }, 'Panels · drag to reorder · drop files to add'), list,
    h('button', { class: 'btn block', disabled: !active(), onclick: addCurrentPlot }, 'Add the current plot'),
    Object.keys(S.files).length > 1 ? h('button', { class: 'btn block', onclick: addOpenPlots }, `Add all ${Object.keys(S.files).length} open plots`) : null,
    F.panels.length ? h('button', { class: 'btn block', onclick: () => { F.panels = []; emit(); } }, 'Clear all') : null);
  const set = (patch) => { Object.assign(F, patch); emit(); };
  const right = h('aside', { class: 'inspector' }, h('div', { class: 'insp-body' },
    field('Size', seg([['slide', 'Slide 16:9'], ['two', 'Paper 2-col'], ['one', 'Paper 1-col'], ['custom', 'Custom']], F.size, (v) => set({ size: v, preset: v === 'slide' ? 'presentation' : 'paper', base: v === 'slide' ? 14 : 8 }))),
    F.size === 'custom' ? h('div', { class: 'g2' }, field('Width (in)', input(F.w, (v) => set({ w: num(v) || 7 }), { numeric: true })), field('Height (in)', input(F.h, (v) => set({ h: num(v) || 4 }), { numeric: true }))) : null,
    h('div', { class: 'g3' }, field('Columns', select([[0, 'Auto'], [1, '1'], [2, '2'], [3, '3'], [4, '4']], F.cols, (v) => set({ cols: Number(v) }))), field('DPI', input(F.dpi, (v) => set({ dpi: num(v) || 300 }), { numeric: true })),
      field('Base (pt)', input(F.base || FONT_PT[F.preset], (v) => set({ base: num(v) || 9 }), { numeric: true }))),
    field('Panel labels', seg([['a', 'a'], ['(a)', '(a)'], ['A', 'A'], ['off', 'Off']], F.labels, (v) => set({ labels: v }))),
    field('Font', select(['Arial', 'Helvetica', 'Times New Roman', 'Calibri'], F.font, (v) => set({ font: v }))),
    h('div', { class: 'card pad note' }, 'Every panel keeps its source file, detector, fit and processing; the exported JSON records them so the figure can be rebuilt.'),
    h('button', { class: 'btn primary block', disabled: !F.panels.length, onclick: () => exportFigure('copy') }, 'Copy figure', h('span', { class: 'kbd' }, 'PNG')),
    h('div', { class: 'row' }, h('button', { class: 'btn', style: 'flex:1;justify-content:center', disabled: !F.panels.length, onclick: () => exportFigure('svg') }, 'Save SVG'),
      h('button', { class: 'btn', style: 'flex:1;justify-content:center', disabled: !F.panels.length, onclick: () => exportFigure('png') }, 'Save PNG'),
      h('button', { class: 'btn', style: 'flex:1;justify-content:center', disabled: !F.panels.length, onclick: () => exportFigure('pdf') }, 'Save PDF'))));
  mount(root, left, preview, right);
  // fit the true-size page into the preview area
  requestAnimationFrame(async () => {
    const avail = preview.getBoundingClientRect();
    const scale = Math.min(1, (avail.width - 40) / g.Wpx, (avail.height - 80) / g.Hpx);
    pageEl.style.width = `${g.Wpx * scale}px`; pageEl.style.height = `${g.Hpx * scale}px`;
    pageInner.style.transform = `scale(${scale})`;
    for (const [i, c] of cells(g).entries()) {
      const div = h('div', { style: `position:absolute;left:${c.x}px;top:${c.y}px;width:${c.w}px;height:${c.h}px` });
      pageInner.append(div);
      if (label(i)) pageInner.append(h('div', { style: `position:absolute;left:${c.x + 2}px;top:${c.y}px;font:bold ${Math.round(fontPx * 1.15)}px ${F.font};color:#111;z-index:2` }, label(i)));
      const fig = panelFigure(c.p, fontPx);
      Plotly.newPlot(div, fig.data, { ...fig.layout, width: c.w, height: c.h, uirevision: undefined }, { staticPlot: true }).then(() => import('./plot2d.js').then((m) => m.fitColorbars(div)));
    }
  });
}

export async function exportFigure(kind) {
  const F = S.figure;
  const g = geometry();
  const fontPx = Math.round(((F.base || FONT_PT[F.preset] || 9) * 96) / 72);
  const parts = [];
  for (const [i, c] of cells(g).entries()) {
    const fig = panelFigure(c.p, fontPx);
    const svg = await renderImage(fig, { width_in: c.w / 96, height_in: c.h / 96, format: 'svg' });
    parts.push({ svg, x: c.x, y: c.y, w: c.w, h: c.h, label: label(i), labelSize: Math.round(fontPx * 1.15) });
  }
  const svg = composeSvg(parts, g.Wpx, g.Hpx);
  if (kind === 'svg') return save(svg, 'svg');
  const png = await svgToPng(svg, g.Wpx, g.Hpx, F.dpi / 96);
  if (kind === 'copy') return copyImage(png);
  if (kind === 'png') return save(png, 'png');
  if (kind === 'pdf') return save(png, 'pdf', svg);
}

async function save(data, ext, vectorSvg) {
  const first = S.figure.panels[0];
  const p = await call('dialog', { kind: 'save', title: 'Save figure', filename: `figure.${ext}`, directory: first ? dirname(first.path) : '' });
  if (!p) return;
  const target = p.toLowerCase().endsWith(`.${ext}`) ? p : `${p}.${ext}`;
  if (ext === 'svg') await call('write_text', { path: target, text: data });
  else if (ext === 'png') await call('write_data_url', { path: target, data });
  else (await call('export', { req: { output_dir: dirname(target), stem: stem(target).replace(/_fit$/, ''), pdf_filename: basename(target), images: { pdf_png: data, pdf_svg: vectorSvg }, dpi: S.figure.dpi, json: false } })).warnings?.forEach((w) => toast(w, 'err', 9000));
  // The sidecar is a full analysis session (files, processing, fits, styles, figure panels + snapshots), so opening it
  // (Open session, or opening the .json from the Library) rebuilds the figure exactly; `figure_export` is a readable summary.
  const summary = { size: geometry(), labels: S.figure.labels, panels: S.figure.panels.map((p) => ({ path: p.path, detector: p.det, title: p.title, fit: S.files[p.path]?.fit?.[p.det] ? { model: S.files[p.path].fit[p.det].model, r2: S.files[p.path].fit[p.det].r2 } : null })) };
  await call('save_session', { path: target.replace(/\.[^.]+$/, '') + '.figure.json', doc: { ...sessionDoc(), figure_export: summary } }, { quiet: true });
  toast(`Saved ${basename(target)}`, 'ok');
}
