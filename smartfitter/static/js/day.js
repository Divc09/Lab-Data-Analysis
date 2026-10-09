// Day workspace: everything acquired in the current library folder, grouped by experiment type in
// acquisition (file modification) order, with batch fitting and a summary table.
import { call } from './api.js';
import { S, emit, openFile, detLabel } from './state.js';
import { h, mount, fmt, fmtPm, fmtDate, fmtTime, toast, basename, tag, TAG_COLORS } from './util.js';
import { resultLines } from './plotstyle.js';
import { runFit, kindOf } from './analyze.js';
import { addCurrentPlot } from './figure.js';
import { copyText } from './export.js';

const D = { fitting: '', adding: false, onlyFitted: false };
let stripEl;

function dayFiles() {
  const lst = S.library.listing;
  if (!lst) return [];
  return [...lst.files].sort((a, b) => a.mtime - b.mtime).map((m) => ({ ...m, info: S.library.peeks[m.path] || S.files[m.path]?.info || null }));
}

// Main fitted quantity for a file (first non-R² result line of its current detector's fit).
function summaryOf(row) {
  const f = S.files[row.path];
  const info = row.info || {};
  const out = { time: row.mtime, path: row.path, name: basename(row.path), type: info.type || '…', tag: info.tag, qty: '', value: null, err: null, unit: '', r2: null, status: '', lines: [] };
  if (info.error) { out.status = 'unreadable'; return out; }
  if (info.dim === 'scan2d') { out.status = '2D map (not fitted here)'; return out; }
  const fit = f?.fit?.[f.detector];
  if (!fit) { out.status = f && (f.info.fit_allowed === false || f.info.type === 'LineScan') && f.fitSpec.model !== 'GaussianPeaks' ? 'plot-only' : 'not fitted'; return out; }
  const lines = resultLines(fit, f.info).filter((l) => l.key !== 'r2');
  out.lines = lines.filter((l) => !l.defaultOff).slice(0, 4);
  const main = lines.find((l) => !l.defaultOff) || lines[0];
  if (main) Object.assign(out, { qty: main.label, value: main.value, err: main.err ?? null, unit: main.unit || '' });
  out.r2 = fit.r2; out.status = fit.status || 'fit';
  return out;
}

const qtyText = (s) => (s.qty ? `${s.qty} ${s.value === null ? '–' : fmtPm(s.value, s.err)}${s.unit ? ` ${s.unit}` : ''}` : '');

function tableText(rows, sep = '\t') {
  const q = (v) => {
    const s = v === null || v === undefined ? '' : String(v);
    return sep === ',' && /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const head = ['time', 'file', 'type', 'quantity', 'value', 'error', 'unit', 'R2', 'status'];
  return [head.join(sep), ...rows.map((s) => [new Date(s.time * 1000).toLocaleString(), s.name, s.type, s.qty, s.value, s.err, s.unit, s.r2, s.status].map(q).join(sep))].join('\n');
}

async function saveCsv(rows) {
  const p = await call('dialog', { kind: 'save', title: 'Save day summary', filename: `day_summary_${basename(S.library.folder || 'folder')}.csv`, directory: S.library.folder || '' });
  if (!p) return;
  await call('write_text', { path: p.endsWith('.csv') ? p : `${p}.csv`, text: tableText(rows, ',') });
  toast('Saved day summary', 'ok');
}

// ------------------------------------------------------------------ actions
const fittable = (r) => {
  const i = r.info;
  if (!i || i.error || i.dim === 'scan2d' || i.type === 'Unreadable') return false;
  return true;
};

async function fitAll(files) {
  const todo = files.filter(fittable);
  let ok = 0, skipped = 0, failed = 0;
  const prev = S.primary;
  for (const [i, r] of todo.entries()) {
    D.fitting = `${i + 1}/${todo.length}`; emit();
    try {
      const f = await openFile(r.path, { primary: false });
      if (kindOf(f) === '2d' || ((f.info.fit_allowed === false || f.info.type === 'LineScan') && kindOf(f) === '1d' && f.fitSpec.model !== 'GaussianPeaks' && f.fitSpec.model !== 'CustomModel')) { skipped++; continue; }
      S.primary = r.path; // runFit records its undo step against the primary file
      await runFit(f);
      ok++;
    } catch (e) { failed++; }
  }
  S.primary = prev;
  D.fitting = '';
  toast(`Fit all: ${ok} fitted, ${skipped} plot-only skipped, ${failed} failed.`, failed ? '' : 'ok');
  emit();
}

async function addAllPlots(rows) {
  const prev = S.primary;
  D.adding = true; emit();
  let n = 0;
  try {
    for (const s of rows) {
      if (!s.value && s.value !== 0) continue;
      S.primary = s.path;
      await addCurrentPlot();
      n++;
    }
  } finally { S.primary = prev; D.adding = false; emit(); }
  if (!n) toast('No fitted files to add.');
}

// ------------------------------------------------------------------ view
export function renderDay(root) {
  const files = dayFiles();
  const sums = files.map(summaryOf);
  const lst = S.library.listing;
  const peeked = files.filter((r) => r.info).length;
  const fitted = sums.filter((s) => s.status !== '' && s.value !== null);
  const nFit = files.filter(fittable).length;

  const groups = [];
  sums.forEach((s) => {
    let g = groups.find((x) => x.type === s.type);
    if (!g) { g = { type: s.type, tag: s.tag, rows: [] }; groups.push(g); }
    g.rows.push(s);
  });

  const left = h('aside', { class: 'library', style: 'padding:12px;gap:10px' },
    h('span', { class: 'lbl' }, 'Day'),
    lst ? h('div', { class: 'card pad' }, h('div', { style: 'font-weight:600' }, lst.name), h('div', { class: 'note', title: lst.path }, lst.path),
      h('div', { class: 'note' }, `${files.length} .mat files · ${groups.length} experiment types${peeked < files.length ? ` · scanning ${peeked}/${files.length}` : ''}`),
      files.length ? h('div', { class: 'note' }, `${fmtDate(files[0].mtime)} ${fmtTime(files[0].mtime)} – ${fmtDate(files[files.length - 1].mtime)} ${fmtTime(files[files.length - 1].mtime)} (file modification time)`) : null)
      : h('div', { class: 'note' }, 'Open a data folder (Analyze view, Library) to see its timeline.'),
    h('button', { class: 'btn primary block', disabled: !nFit || !!D.fitting, onclick: () => fitAll(files) }, D.fitting ? `Fitting ${D.fitting}` : `Fit all fittable files (${nFit})`),
    h('div', { class: 'note' }, 'Each file is opened in the background and fitted with its current Fit settings (Auto model for its experiment type). 2D maps and plot-only scans are skipped.'),
    h('button', { class: 'btn block', disabled: !fitted.length || D.adding, onclick: () => addAllPlots(fitted) }, D.adding ? 'Adding…' : `Add all fitted plots to Figure composer (${fitted.length})`),
    h('label', { class: 'ck' }, h('input', { type: 'checkbox', checked: D.onlyFitted, onchange: (e) => { D.onlyFitted = e.target.checked; emit(); } }), 'Show fitted files only'),
    h('div', { class: 'note' }, 'Order is by file modification time, which is the save time (for checkpointed runs, the last checkpoint), not the acquisition start.'));

  stripEl = stripEl || h('div', { class: 'plot' });
  const main = h('main', { class: 'main' },
    h('div', { class: 'row wrap' }, h('h1', { style: 'margin:0;font-size:17px;font-weight:600;margin-right:auto' }, 'Timeline by experiment type'),
      h('span', { class: 'note' }, 'click a row to open it in Analyze')),
    files.length ? h('section', { class: 'plotcard', style: `flex:none;height:${Math.max(150, 46 + 34 * groups.length)}px;min-height:150px` }, stripEl) : null,
    groups.length ? groups.map((g) => {
      const rows = g.rows.filter((s) => !D.onlyFitted || s.value !== null);
      if (!rows.length) return null;
      const [bg, fg] = TAG_COLORS[g.tag] || ['#2A2F3A', '#C9D1DD'];
      return h('section', { class: 'card', style: 'padding:10px 12px;display:flex;flex-direction:column;gap:4px;flex:none' },
        h('div', { class: 'row' }, tag(g.tag), h('span', { style: 'font-weight:600' }, g.type), h('span', { class: 'note' }, `${rows.length} file${rows.length === 1 ? '' : 's'} · ${fmtTime(g.rows[0].time)} – ${fmtTime(g.rows[g.rows.length - 1].time)}`)),
        rows.map((s) => h('div', { class: 'frow', style: `border-left:3px solid ${fg}`, title: s.path, onclick: () => openFile(s.path).catch(() => {}) },
          h('span', { class: 'mono dim', style: 'width:48px;flex:none' }, fmtTime(s.time)),
          h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'fname ellip' }, s.name.replace(/\.mat$/i, '')),
            h('div', { class: 'sub' }, s.lines.length ? s.lines.map((l) => `${l.label} ${l.value === null ? '–' : fmtPm(l.value, l.err)}${l.unit ? ` ${l.unit}` : ''}`).join(' · ') : s.status)),
          s.r2 !== null ? h('span', { class: 'mono dim' }, `R² ${fmt(s.r2, 3)}`) : null,
          s.value !== null ? h('span', { class: `pill ${s.status}` }, s.status) : null)));
    }) : h('div', { class: 'card pad note' }, 'No files yet. Open a data folder from the Analyze view; this view lists what that folder contains.'));

  const rows = sums.filter((s) => !D.onlyFitted || s.value !== null);
  const right = h('aside', { class: 'inspector' }, h('div', { class: 'insp-body' },
    h('div', { class: 'row' }, h('span', { class: 'lbl' }, 'Summary'), h('span', { class: 'sp' }),
      h('button', { class: 'btn sm', disabled: !rows.length, onclick: () => copyText(tableText(rows), 'Day summary copied') }, 'Copy'),
      h('button', { class: 'btn sm', disabled: !rows.length, onclick: () => saveCsv(rows).catch(() => {}) }, 'Save CSV')),
    rows.length ? h('table', { class: 'tbl' }, h('tr', {}, ['Time', 'File', 'Type', 'Quantity', 'R²', 'Status'].map((t) => h('th', {}, t))),
      rows.map((s) => h('tr', { class: 'click', title: s.path, onclick: () => openFile(s.path).catch(() => {}) },
        h('td', {}, fmtTime(s.time)), h('td', { class: 'txt ellip', style: 'max-width:96px' }, s.name.replace(/\.mat$/i, '')), h('td', { class: 'txt' }, s.tag || s.type),
        h('td', { class: 'txt' }, qtyText(s)), h('td', {}, s.r2 === null ? '' : fmt(s.r2, 3)), h('td', { class: 'txt' }, s.status))))
      : h('div', { class: 'note' }, 'Nothing to summarize yet.')));

  mount(root, left, main, right);
  drawStrip(sums);
}

function drawStrip(sums) {
  if (!sums.length || !stripEl) return;
  const types = [...new Set(sums.map((s) => s.type))];
  const trace = { type: 'scatter', mode: 'markers', x: sums.map((s) => new Date(s.time * 1000)), y: sums.map((s) => s.type), customdata: sums.map((s) => s.path),
    text: sums.map((s) => `${s.name}<br>${s.status}${s.qty ? '<br>' + qtyText(s) : ''}`), hovertemplate: '%{text}<br>%{x|%H:%M:%S}<extra></extra>',
    marker: { size: 11, color: sums.map((s) => (s.value !== null ? '#2E8B57' : '#FFFFFF')), line: { color: sums.map((s) => (TAG_COLORS[s.tag] || ['', '#555'])[1]), width: 2 } } };
  const layout = { separators: '.', margin: { l: 120, r: 16, t: 8, b: 40 }, font: { family: 'Arial', size: 12, color: '#1D2330' }, showlegend: false, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    xaxis: { automargin: true, title: { text: 'file modification time (green fill = fitted)' }, showline: true, mirror: true, ticks: 'outside', gridcolor: '#ECEEF2' },
    yaxis: { categoryorder: 'array', categoryarray: types.slice().reverse(), showline: true, mirror: true, gridcolor: '#ECEEF2', automargin: true } };
  Plotly.react(stripEl, [trace], layout, { responsive: true, displayModeBar: false });
  setTimeout(() => stripEl.isConnected && Plotly.Plots.resize(stripEl), 60);
  if (!stripEl._bound) {
    stripEl._bound = true;
    stripEl.on('plotly_click', (ev) => { const p = ev.points?.[0]?.customdata; if (p) openFile(p).catch(() => {}); });
  }
}
