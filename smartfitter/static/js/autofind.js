import { finiteExtent } from './numeric.js';
// AutoFind: saved search evidence in native absolute stage coordinates (um).
import { call } from './api.js';
import { S, emit, openFile } from './state.js';
import { h, mount, select, fmt, toast } from './util.js';

const COLORS = { coarse: '#757575', confirm: '#757575', refine: '#eb6834', z: '#2a78d6', fine: '#1baf7a', patch: '#1baf7a', repeat: '#2a78d6', other: '#757575' };
const DASH = { coarse: 'solid', confirm: 'dot', refine: 'solid', fine: 'solid', patch: 'dash', repeat: 'dashdot', other: 'dot' };
const RAMP = [[0, '#cde2fb'], [0.35, '#6da7ec'], [0.7, '#2a78d6'], [1, '#0d366b']];
const PALETTE = ['#2a78d6', '#eb6834', '#1baf7a'];
const MODES = [['metric', 'AutoFind metric (detector choice from log)'], ['det1', 'Raw det1'], ['det2', 'Raw det2'], ['ratio', 'autoFindDetect line-normalized ratio']];
// 'live' = autoFindLivePlot's log10(max(v,1)); it blanks everything below 1, i.e. the background of the dimensionless metrics.
const SCALES = [['live', 'Scale: log10 max(v,1) (live)'], ['log', 'Scale: log10 v (v ≤ 0 blank)'], ['linear', 'Scale: linear (raw values)']];
const SCALE_LABEL = { live: 'log10 max(value, 1)', log: 'log10 value', linear: 'value (linear)' };
const ST = { root: '', runs: [], source: '', run: null, view: 'map', mode: 'metric', scale: 'live', contours: false, selected: [], busy: '', errors: [],
  pursuit: null, pursuitData: [], refine: '', fine: '', ztag: '', exportDir: '', exportResult: null, progress: '', epoch: 0, compare: [], colors: new Map() };
const cache = new Map();
const plots = new Map();
const note = (text) => h('div', { class: 'hint', style: 'white-space:normal;margin:8px 0;line-height:1.5' }, text);
const button = (text, fn, disabled = false) => h('button', { class: 'btn sm', disabled, onclick: () => action(text, fn) }, text);
const textInput = (value, onChange, placeholder) => h('input', { class: 'field', value, placeholder, style: 'min-width:0;flex:1', onchange: (e) => onChange(e.target.value) });
const table = (heads, rows) => h('div', { style: 'overflow:auto;max-height:360px' }, h('table', { class: 'table', style: 'width:100%;font-size:11px' },
  h('thead', {}, h('tr', {}, heads.map((v) => h('th', {}, v)))), h('tbody', {}, rows.map((r) => h('tr', {}, r.map((v) => h('td', { style: 'padding:5px;white-space:normal' }, v)))))));
const numArray = (v) => Array.isArray(v) ? v : v === null || v === undefined ? [] : [v];
const timeScale = (seconds) => seconds >= 7200 ? [3600, 'h'] : seconds >= 120 ? [60, 'min'] : [1, 's'];
const logValues = (m) => m.values.map((row) => row.map((v) => v === null || !Number.isFinite(v) ? null
  : ST.scale === 'linear' ? v : ST.scale === 'log' ? (v > 0 ? Math.log10(v) : null) : Math.log10(Math.max(v, 1))));
const scans = () => ST.run?.scans || [];
const key = (source, mode) => JSON.stringify([source, mode]);
const mapsOf = (source) => cache.get(key(source, ST.mode))?.maps || [];

// Navigation (views, runs, metric) pre-empts a running load: the loaders check ST.epoch and stop, so clicks made
// while 60+ rasters stream in are not silently dropped. Only export waits for an idle workspace.
let busyToken = 0, runRequest = 0, discoveryRequest = 0;
async function action(label, fn, { preempt = true } = {}) {
  if (ST.busy && !preempt) { toast(`Wait: ${ST.busy} is still running.`); return; }
  const token = ++busyToken;
  ST.busy = label; emit();
  try { await fn(); } catch (e) { toast(e.message, 'err'); } finally { if (token === busyToken) { ST.busy = ''; emit(); } }
}
function trimCache() {
  let count = [...cache.values()].reduce((n, v) => n + v.maps.length, 0);
  for (const [k, v] of cache) {
    if (cache.size <= 6 && count <= 160) break;
    if (k === key(ST.source, ST.mode)) continue;
    count -= v.maps.length; cache.delete(k);
  }
}
async function discover() {
  if (!ST.root.trim()) { toast('Enter a run folder or a parent folder.'); return; }
  ST.epoch++; runRequest++;
  const request = ++discoveryRequest;
  const d = await call('autofind_discover', { path: ST.root });
  if (request !== discoveryRequest) return;
  ST.root = d.root; ST.runs = d.runs; ST.errors = d.errors.map((e) => JSON.stringify(e));
  ST.selected = ST.selected.filter((p) => ST.runs.some((r) => r.source === p));
  if (!d.runs.length) { ST.run = null; ST.source = ''; }
  ST.progress = `Indexed ${d.runs.length} runs in ${fmt(d.load_seconds)} s; rasters load only when needed.`;
  if (d.runs.length && !d.runs.some((r) => r.source === ST.source)) await chooseRun(d.runs[0].source);
}
// Re-index the folder (new runs, scan counts, durations of a run still being written), then reload the open run.
async function refresh() {
  cache.clear();
  const keep = ST.source;
  if (ST.root.trim()) await discover();
  if (keep && ST.runs.some((r) => r.source === keep)) await chooseRun(keep);
}
async function browseRoot() {
  const p = await call('dialog', { kind: 'open_folder', title: 'Choose AutoFind run or parent', directory: ST.root || S.library.folder || '' });
  if (p) { ST.root = p; await discover(); }
}
async function chooseRun(source) {
  ST.epoch++;
  const request = ++runRequest;
  const r = await call('autofind_run', { source });
  if (request !== runRequest) return;
  // A view/metric switch cancels raster work, not the selected run itself.
  ST.source = source; ST.run = r; ST.pursuit = null; ST.pursuitData = [];
  emit();
  if (ST.view === 'pursuit' && r.pursuits.length) await choosePursuit(r.pursuits[0].id);
  else if (ST.view === 'map') await loadMaps(source, ST.epoch);
}
async function loadMaps(source, epoch = ST.epoch) {
  const mode = ST.mode, k = key(source, mode);
  let entry = cache.get(k);
  if (!entry) { entry = { maps: [], offset: 0, total: Infinity, errors: [] }; cache.set(k, entry); }
  while (entry.offset < entry.total && epoch === ST.epoch) {
    const offset = entry.offset;
    const d = await call('autofind_maps', { source, mode, offset, limit: 12 });
    if (epoch !== ST.epoch || cache.get(k) !== entry) return;
    if (entry.offset !== offset) continue; // Another loader already appended this batch.
    entry.maps.push(...d.maps); entry.offset = d.next_offset; entry.total = d.total;
    entry.errors.push(...d.errors); trimCache();
    ST.progress = `${entry.offset} / ${entry.total} coarse rasters loaded (native pixels).`; emit();
    // Yield so progressive rendering/input is not blocked by a long run.
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
}
async function changeMode(mode) {
  ST.mode = mode; const epoch = ++ST.epoch;
  ST.pursuitData = []; emit();
  if (ST.view === 'pursuit' && ST.pursuit) await choosePursuit(ST.pursuit.id);
  else if (ST.view === 'compare') await buildCompare(epoch);
  else if (ST.run) await loadMaps(ST.source, epoch);
}
async function changeView(view) {
  ST.view = view; const epoch = ++ST.epoch; emit();
  if (view === 'map' && ST.run) await loadMaps(ST.source, epoch);
  if (view === 'compare') await buildCompare(epoch);
  if (view === 'pursuit' && ST.run?.pursuits.length && !ST.pursuit) await choosePursuit(ST.run.pursuits[0].id);
}
async function openScan(source, id) {
  const r = source === ST.source ? ST.run : ST.compare.find((r) => r.source === source);
  const scan = r?.scans.find((s) => s.id === String(id));
  if (!scan?.path || scan.evidence_unavailable) { toast('Missing/overwritten scan evidence; no substitute is opened.'); return; }
  await openFile(scan.path, { primary: true });
}
async function choosePursuit(id) {
  const epoch = ++ST.epoch, source = ST.source;
  const p = await call('autofind_pursuit', { source, pursuit_id: id });
  const data = [];
  for (const s of p.scans) {
    if (epoch !== ST.epoch) return;
    try { data.push(await call('autofind_scan', { source, scan_id: s.id, mode: s.type === 'z' && ST.mode === 'ratio' ? 'metric' : ST.mode })); }
    catch (e) { data.push({ scan: s, unavailable: e.message }); }
  }
  if (epoch !== ST.epoch) return;
  ST.pursuit = p; ST.pursuitData = data;
  ST.refine = data.filter((m) => m.scan.kind === 'refine').at(-1)?.scan.id || '';
  ST.fine = data.filter((m) => ['fine', 'repeat'].includes(m.scan.kind)).at(-1)?.scan.id || '';
  ST.ztag = zTags()[0] || ''; emit();
}
async function buildCompare(epoch = ++ST.epoch) {
  if (!ST.selected.length) { ST.compare = []; return; }
  const d = await call('autofind_compare', { sources: ST.selected });
  if (epoch !== ST.epoch) return;
  ST.compare = d.runs;
  const visible = ST.compare.slice(0, 3).map((r) => r.source);
  for (const source of ST.colors.keys()) if (!visible.includes(source)) ST.colors.delete(source);
  for (const source of visible) if (!ST.colors.has(source)) ST.colors.set(source, [0, 1, 2].find((i) => ![...ST.colors.values()].includes(i)));
  for (const r of ST.compare.slice(0, 3)) { await loadMaps(r.source, epoch); if (epoch !== ST.epoch) return; }
}
async function exportSet() {
  const sources = ST.view === 'compare' ? ST.selected : [ST.source];
  if (!sources.length || !ST.exportDir) { toast('Choose run(s) and an export directory outside the acquisition folder.'); return; }
  ST.exportResult = await call('autofind_export', { sources, directory: ST.exportDir, mode: ST.mode, contours: ST.contours, scale: ST.scale }, { label: 'Writing white 300-DPI figures' });
  toast(`Exported ${ST.exportResult.pngs.length} PNGs; ${ST.exportResult.errors.length} errors. See export.json for provenance.`);
}

export const canExportAutoFind = () => !!ST.source && !ST.busy;
export const exportAutoFind = () => action('Export figure set', async () => {
  if (!ST.exportDir) {
    const dir = await call('dialog', { kind: 'open_folder', title: 'Choose figure export directory', directory: '' });
    if (!dir) return;
    ST.exportDir = dir;
  }
  await exportSet();
}, { preempt: false });

function layout(title, extra = {}) {
  return { paper_bgcolor: 'white', plot_bgcolor: 'white', font: { family: 'Arial, sans-serif', size: 11, color: '#111111' },
    title: { text: title, font: { size: 13 } }, margin: { l: 72, r: 30, b: 65, t: 55 }, hovermode: 'closest',
    xaxis: { title: { text: 'x (um)' }, showgrid: false, zeroline: false, exponentformat: 'none', tickformat: '.6~g', constrain: 'domain' },
    yaxis: { title: { text: 'y (um)' }, showgrid: false, zeroline: false, scaleanchor: 'x', scaleratio: 1, constrain: 'domain', tickformat: '.6~g' },
    legend: { orientation: 'h', y: -0.18, font: { size: 10 } }, ...extra };
}
function plot(name, data, lay, height = 520) {
  let el = plots.get(name);
  if (!el) { el = h('div', { style: `width:100%;height:${height}px;min-width:0;background:white`, 'aria-label': name, 'data-autofind-plot': name }); plots.set(name, el); }
  el.style.height = `${height}px`;
  const version = el._autofindVersion = (el._autofindVersion || 0) + 1;
  requestAnimationFrame(() => {
    const draw = async () => {
      if (!el.isConnected || S.view !== 'autofind' || version !== el._autofindVersion) return;
      el._autofindRendering = true;
      try {
        await Plotly.react(el, data, lay, { responsive: true, displaylogo: false, toImageButtonOptions: { format: 'png', scale: 3 } });
        el.removeAllListeners('plotly_click');
        el.on('plotly_click', (e) => {
          const p = e.points?.[0];
          if (p?.data?.meta?.scanId !== undefined) action('Open scan', () => openScan(p.data.meta.source, p.data.meta.scanId));
          else if (p?.data?.type === 'bar' && p?.customdata?.[2] !== undefined) action('Open scan', () => openScan(ST.source, p.customdata[2]));
        });
      } catch (e) { toast(`AutoFind plot: ${e.message}`, 'err'); console.error(e); }
      finally { el._autofindRendering = false; }
    };
    el._autofindPending = (el._autofindPending || Promise.resolve()).then(draw);
  });
  return el;
}
function heatmap(m, source, opts = {}) {
  return { type: 'heatmap', x: m.x, y: m.y, z: logValues(m), zsmooth: false, hoverongaps: false,
    coloraxis: 'coloraxis', meta: { source, scanId: m.scan.id, mode: m.mode }, name: m.scan.label,
    hovertemplate: `${m.scan.label}<br>${m.axes[0]} %{x:.6g} um<br>${m.axes[1]} %{y:.6g} um<br>${SCALE_LABEL[ST.scale]} %{z:.4g}<br>Scan completion ${fmt(m.scan.t)} s<extra></extra>`, ...opts };
}
function range(maps) {
  const v = maps.flatMap((m) => m.values ? logValues(m).flat().filter((x) => x !== null) : []).sort((a, b) => a - b);
  const lo = v.length ? v[Math.max(0, Math.round(0.02 * v.length) - 1)] : 0;
  const hi = v.length ? v[Math.min(v.length - 1, Math.max(1, Math.round(0.998 * v.length) - 1))] + 1e-6 : 1;
  return { cmin: lo, cmax: Math.max(hi, lo + 1e-6), colorscale: RAMP };
}
function contourTraces(m, run, showlegend = true) {
  if (!ST.contours || ST.mode !== 'ratio') return [];
  return ['weakRatio', 'strongRatio'].flatMap((k) => {
    const v = run.params[k];
    if (typeof v !== 'number' || m.x.length < 2 || m.y.length < 2) return [];
    return [{ type: 'contour', x: m.x, y: m.y, z: m.values, contours: { start: v, end: v, size: 1, coloring: 'none', showlabels: true },
      showscale: false, line: { color: '#111111', width: k === 'strongRatio' ? 2 : 1 }, hoverinfo: 'skip', name: `${k} ${v}x`, legendgroup: k, showlegend }];
  });
}
function footprints(run) {
  const present = new Set();
  return run.scans.filter((s) => s.xLim && s.yLim).map((s) => {
    const [x0, x1] = finiteExtent(s.xLim), [y0, y1] = finiteExtent(s.yLim);
    const showlegend = !present.has(s.kind); present.add(s.kind);
    return { type: 'scatter', mode: 'lines', x: [x0, x1, x1, x0, x0], y: [y0, y0, y1, y1, y0],
      line: { color: COLORS[s.kind], width: 1, dash: DASH[s.kind] || 'dot' }, name: `${s.kind} footprint`, showlegend,
      meta: { source: run.source, scanId: s.id }, hovertemplate: `${s.label}<br>${s.kind}<br>Completion ${fmt(s.t)} s<br>Click: Analyze<extra></extra>` };
  });
}
function markerTraces(run, maps) {
  const out = [], candidates = maps.flatMap((m) => m.candidates || []);
  if (candidates.length) out.push({ type: 'scatter', mode: 'markers', x: candidates.map((c) => c.pos1), y: candidates.map((c) => c.pos2),
    marker: { symbol: 'circle-open', size: 9, color: PALETTE[0], line: { width: 1.5 } }, name: 'Candidate (recomputed)',
    text: candidates.map((c) => `Ratio ${fmt(c.ratio)}x; ${c.provenance}`), hovertemplate: '%{text}<br>Position %{x:.6g}, %{y:.6g} um<extra></extra>' });
  const rejected = run.pursuits.filter((p) => !p.particle && p.verdict !== 'unavailable').map((p) => {
    const s = run.scans.find((s) => p.scan_ids.includes(s.id) && s.xLim && s.yLim);
    return s ? { x: (s.xLim[0] + s.xLim.at(-1)) / 2, y: (s.yLim[0] + s.yLim.at(-1)) / 2, p } : null;
  }).filter(Boolean);
  if (rejected.length) out.push({ type: 'scatter', mode: 'markers', x: rejected.map((r) => r.x), y: rejected.map((r) => r.y),
    marker: { symbol: 'x', size: 10, color: '#d03b3b' }, name: 'Rejected (pursuit window centre)', text: rejected.map((r) => r.p.reason), hovertemplate: '%{text}<extra></extra>' });
  for (const definitive of [false, true]) {
    const p = run.particles.filter((p) => Boolean(p.definitive) === definitive);
    if (p.length) out.push({ type: 'scatter', mode: 'markers', x: p.map((p) => p.position[0]), y: p.map((p) => p.position[1]),
      marker: { symbol: definitive ? 'star' : 'diamond', size: definitive ? 14 : 10, color: definitive ? '#0ca30c' : '#fab219', line: { color: 'white', width: 2 } },
      name: definitive ? 'Definitive particle (logged)' : 'Not definitive (logged)', text: p.map((p) => `Logged ratio ${fmt(p.ratio)}x; z ${fmt(p.position[2])} um`), hovertemplate: '%{text}<extra></extra>' });
  }
  for (const [field, name, symbol] of [['start', 'Start', 'square'], ['final_position', 'Final', 'cross']]) {
    const p = run[field];
    const label = field === 'final_position' && run.final_position_source !== 'L.finalPosition' ? 'Last saved stage (final not logged)' : name;
    if (p?.length >= 2 && p.slice(0, 2).every((v) => typeof v === 'number' && Number.isFinite(v))) out.push({ type: 'scatter', mode: 'markers', x: [p[0]], y: [p[1]],
      marker: { symbol, size: 11, color: '#111111', line: { color: 'white', width: 2 } }, name: label, hovertemplate: `${label}<br>%{x:.6g}, %{y:.6g} um<extra></extra>` });
  }
  return out;
}
function mapView() {
  const run = ST.run, maps = mapsOf(ST.source), good = maps.filter((m) => m.values);
  const visits = run.visits;
  const path = { type: 'scatter', mode: 'lines+markers', x: visits.map((v) => v.x), y: visits.map((v) => v.y), marker: { size: 6, color: '#52514e' }, line: { width: 1, color: '#52514e' }, name: 'Coarse visit order', text: visits.map((v, i) => `Visit ${i + 1}; tile ${v.tile ?? 'unknown'}; ${fmt(v.t)} s`), hovertemplate: '%{text}<extra></extra>' };
  const coloraxis = { ...range(good), colorbar: { title: { text: SCALE_LABEL[ST.scale] }, thickness: 15 } };
  return h('div', {}, note(good[0]?.metric_label || 'No available coarse raster. Footprints and timing still show logged evidence.'),
    plot('stitched search map', [...good.flatMap((m, i) => [heatmap(m, run.source), ...contourTraces(m, run, i === 0)]), ...footprints(run), path, ...markerTraces(run, maps)],
      layout(`${run.run_id} — native search map`, { coloraxis, uirevision: `${ST.source}:${ST.mode}`, xaxis: { ...layout('').xaxis, title: { text: `${good[0]?.axes[0] || 'x'} (um)` } }, yaxis: { ...layout('').yaxis, title: { text: `${good[0]?.axes[1] || 'y'} (um)` } } }), 640),
    note('Native pixels; equal physical axis scale; no interpolation or registration. Grey line is visit order. Rejected markers are pursuit-window centres, not decoded candidate centroids. Definitive status is the historical log verdict, not a new scientific validation.'),
    ...[...new Set(good.flatMap((m) => m.notes))].map(note),
    ...cache.get(key(ST.source, ST.mode))?.errors.map((e) => note(`Raster error ${e.scan_id}: ${e.error}`)) || [], scanTable(run));
}
function scanTable(run, list = run.scans) {
  return h('details', { open: true }, h('summary', {}, `Scan list (${list.length}) — click to Analyze`), table(['Scan / file', 'Activity', 'Completion (s)', 'Interval (s)', 'Evidence'], list.map((s) => [
    button(s.label || s.file_name, () => openScan(run.source, s.id), !s.path || s.evidence_unavailable), s.kind, fmt(s.t), fmt(s.duration_s),
    s.evidence_unavailable ? 'overwritten / noData; suppressed' : !s.path ? 'missing' : s.logged ? 'logged; basename resolved' : 'unlogged / inferred',
  ])));
}
function timelineView() {
  const r = ST.run, [scale, unit] = timeScale(r.duration_s);
  const kinds = Object.keys(COLORS).filter((k) => r.scans.some((s) => s.kind === k));
  const bands = kinds.map((kind) => {
    const ss = r.scans.filter((s) => s.kind === kind);
    return { type: 'bar', orientation: 'h', y: ss.map(() => kind), x: ss.map((s) => s.duration_s / scale), base: ss.map((s) => s.start_t / scale), width: 0.6,
      marker: { color: COLORS[kind], line: { width: 0 } }, name: kind, showlegend: false,
      customdata: ss.map((s) => [s.label, s.duration_source, s.id]),
      hovertemplate: '%{customdata[0]}<br>%{x:.5g} ' + unit + '<br>%{customdata[1]}<br>Click scan to Analyze<extra></extra>' };
  });
  const short = r.scans.filter((s) => s.duration_s < r.duration_s / 1000);
  const ticks = short.map((s) => ({ type: 'scatter', mode: 'markers', x: [s.end_t / scale], y: [s.kind], marker: { symbol: 'line-ns', size: 8, color: COLORS[s.kind], line: { width: 1 } },
    name: 'Short scan completion tick', showlegend: false, meta: { source: r.source, scanId: s.id }, hovertemplate: `${s.label}<br>Completion ${fmt(s.end_t)} s<br>Interval ${fmt(s.duration_s)} s<extra></extra>` }));
  const events = { type: 'scatter', mode: 'markers', x: r.events.map((e) => e.t / scale), y: r.events.map(() => 'events'), marker: { symbol: 'diamond', size: 8, color: '#111111' }, name: 'Events', text: r.events.map((e) => `${e.type}: ${e.message}`), hovertemplate: '%{text}<br>%{x:.5g} ' + unit + '<extra></extra>' };
  // Each bar point has its own scan ID (unlike whole-raster traces).
  const el = plot('search timeline', [...bands, ...ticks, events], layout(`${r.run_id}: scan completion intervals / events`, { barmode: 'overlay', xaxis: { title: { text: `Elapsed time (${unit})` }, range: [0, Math.max(1, r.duration_s / scale)], showgrid: true, gridcolor: '#e1e0d9' }, yaxis: { type: 'category', categoryorder: 'array', categoryarray: [...kinds, 'events'], automargin: true } }), 420);
  const active = Object.entries(r.activity_seconds).filter(([k, v]) => v > 0);
  return h('div', {}, el, note('L.scans.t is scan COMPLETION. L.diag.seconds gives acquisition duration if present; otherwise bands include settling, decisions and overhead. Unassigned time is retained, not charged to acquisition. Very short scans use completion ticks without stretching their intervals.'),
    plot('time by activity', [{ type: 'bar', x: active.map(([k]) => k), y: active.map(([k, v]) => v / scale), marker: { color: PALETTE[0] }, text: active.map(([k, v]) => `${fmt(v / scale)} ${unit}`), textposition: 'auto', hovertemplate: '%{x}: %{y:.5g} ' + unit + '<extra></extra>' }],
      layout('Time-per-activity summary', { xaxis: { title: { text: 'Activity' } }, yaxis: { title: { text: `Time (${unit})` }, gridcolor: '#e1e0d9' } }), 280),
    table(['Elapsed (s)', 'Decision / event', 'Message'], r.events.map((e) => [fmt(e.t), e.type, e.message])), scanTable(r));
}
function zTags() { return [...new Set(ST.pursuitData.filter((m) => m.scan.type === 'z').map((m) => m.scan.label.split('_Z')[1]?.split('_')[0] || m.scan.label))]; }
function zLabel(m) { return m.scan.label.split('_Z')[1]?.split('_')[0] || m.scan.label; }
function xyPanel(kind, id, assign) {
  const options = ST.pursuitData.filter((m) => kind === 'refine' ? m.scan.kind === 'refine' : ['fine', 'repeat', 'confirm', 'patch'].includes(m.scan.kind));
  const m = options.find((m) => m.scan.id === id) || options.at(-1);
  return h('div', { style: 'flex:1;min-width:300px' }, select(options.map((m) => [m.scan.id, m.scan.label]), m?.scan.id || '', (v) => { assign(v); emit(); }),
    !m?.values ? note(m?.unavailable || `No ${kind} scan`) : plot(`${kind} pursuit map`, [heatmap(m, ST.source), ...contourTraces(m, ST.run)],
      layout(`${m.scan.label}`, { coloraxis: { ...range([m]), colorbar: { title: { text: SCALE_LABEL[ST.scale] }, thickness: 10 } }, xaxis: { ...layout('').xaxis, title: { text: `${m.axes[0]} (um)` } }, yaxis: { ...layout('').yaxis, title: { text: `${m.axes[1]} (um)` } } }), 350),
    note(m?.metric_label || ''), m?.scan ? button('Open scan in Analyze', () => openScan(ST.source, m.scan.id), !m.scan.path || m.scan.evidence_unavailable) : null);
}
function pursuitView() {
  const p = ST.pursuit;
  const chooser = select(ST.run.pursuits.map((g) => [g.id, `${g.id}: ${g.verdict}`]), p?.id || '', (v) => action('Loading pursuit', () => choosePursuit(v)));
  if (!p) return h('div', {}, chooser, note('No pursued candidate available. Recomputed coarse candidates are shown on the map.'));
  const z = ST.pursuitData.filter((m) => m.scan.type === 'z' && zLabel(m) === ST.ztag && m.values);
  const zseries = z.map((m) => ({ type: 'scatter', mode: 'lines+markers', x: m.z, y: m.values, name: m.scan.label, meta: { source: ST.source, scanId: m.scan.id }, marker: { size: 7, symbol: m.scan.label.includes('_rev') ? 'square' : 'circle' },
    line: { width: 2, dash: m.scan.label.includes('_rev') ? 'dash' : m.scan.label.includes('fine') ? 'dot' : 'solid', color: m.scan.label.includes('_rev') ? PALETTE[1] : PALETTE[0] }, hovertemplate: `${m.scan.label}<br>z %{x:.6g} um<br>%{y:.6g}<extra></extra>` }));
  return h('div', {}, chooser, h('h3', {}, `${p.id} — ${p.verdict}`), note(p.reason),
    note(p.bleaching === null ? 'No particle-level bleaching flag saved. Logged z-pass bleaching messages, if present, are shown below; no new bleaching diagnosis is inferred.' : `Particle bleaching flag in log: ${p.bleaching ? 'true' : 'false'}.`),
    h('div', { style: 'display:flex;gap:12px;flex-wrap:wrap' },
      xyPanel('refine', ST.refine, (v) => { ST.refine = v; }),
      h('div', { style: 'flex:1;min-width:300px' }, select(zTags().map((t) => [t, `Z search ${t}: all passes`]), ST.ztag, (v) => { ST.ztag = v; emit(); }),
        zseries.length ? plot('pursuit z passes', zseries, layout('Forward / reverse z (no sign flip)', { xaxis: { title: { text: 'z (um); lower z = stage higher' }, exponentformat: 'none', hoverformat: '.7g' }, yaxis: { title: { text: 'AutoFind metric (see label below)' }, gridcolor: '#e1e0d9' }, hovermode: 'x unified', margin: { l: 62, r: 15, b: 135, t: 55 }, legend: { orientation: 'h', y: -0.5, font: { size: 9 } } }), 420) : note('No z scan was recorded for this candidate (the pursuit ended before a z search).'), zseries.length ? note(z[0]?.metric_label || '') : null),
      xyPanel('fine / repeat', ST.fine, (v) => { ST.fine = v; })),
    plot('pursuit ratio history', [{ type: 'scatter', mode: 'lines+markers', x: p.ratios.map((r) => r.t), y: p.ratios.map((r) => r.ratio), marker: { color: PALETTE[0], size: 8 }, line: { color: PALETTE[0], width: 2 }, text: p.ratios.map((r) => r.source), hovertemplate: '%{text}<br>%{x:.5g} s<br>ratio %{y:.5g}x<extra></extra>' }],
      layout('Blob contrast versus time (rounded message values if exact history absent)', { xaxis: { title: { text: 'Elapsed time (s)' } }, yaxis: { title: { text: 'Blob contrast (dimensionless)' }, gridcolor: '#e1e0d9' }, hovermode: 'x unified' }), 310),
    (p.bleaching_events || []).length ? plot('logged z-pass bleaching events', [{ type: 'scatter', mode: 'markers', x: p.bleaching_events.map((e) => e.t), y: p.bleaching_events.map(() => 'logged z-pass bleaching'), marker: { color: PALETTE[1], symbol: 'triangle-down', size: 10 }, text: p.bleaching_events.map((e) => e.message), hovertemplate: '%{text}<br>%{x:.5g} s<extra></extra>' }],
      layout('Bleaching evidence: event times only (not blob contrast)', { xaxis: { title: { text: 'Elapsed time (s)' } }, yaxis: { type: 'category', automargin: true } }), 190) : null,
    note(p.ratio_note), table(['Elapsed (s)', 'Blob contrast', 'Source'], p.ratios.map((r) => [fmt(r.t), fmt(r.ratio), r.source])),
    table(['Elapsed (s)', 'Event', 'Verdict / evidence'], p.events.map((e) => [fmt(e.t), e.type, e.message])), scanTable(ST.run, p.scans));
}
function compareView() {
  const runs = ST.compare, subset = runs.slice(0, 3), traces = [], axes = {};
  const allMaps = subset.flatMap((r) => mapsOf(r.source)).filter((m) => m.values);
  const xs = allMaps.flatMap((m) => m.x_edges), ys = allMaps.flatMap((m) => m.y_edges);
  for (const r of subset) { xs.push(...r.visits.map((v) => v.x)); ys.push(...r.visits.map((v) => v.y)); }
  const bounds = (v) => v.length ? finiteExtent(v, [0, 1]) : [0, 1];
  subset.forEach((r) => {
    const i = ST.colors.get(r.source) || 0, maps = mapsOf(r.source).filter((m) => m.values), axis = `coloraxis${i === 0 ? '' : i + 1}`;
    axes[axis] = { ...range(maps), colorscale: [[0, 'white'], [1, PALETTE[i]]], showscale: false };
    traces.push(...maps.map((m) => heatmap(m, r.source, { coloraxis: axis, opacity: 0.45 })));
    traces.push({ type: 'scatter', mode: 'lines+markers', x: r.visits.map((v) => v.x), y: r.visits.map((v) => v.y), name: r.run_id,
      marker: { color: PALETTE[i], size: 7, symbol: ['circle', 'square', 'triangle-up'][i] }, line: { color: PALETTE[i], width: 2, dash: ['solid', 'dash', 'dashdot'][i] }, hovertemplate: `${r.run_id}<br>%{x:.6g}, %{y:.6g} um<extra></extra>` });
  });
  return h('div', {}, note('Check runs in the left list, then Build comparison. Overlay shows the first three selected runs, in shared absolute coordinates with independent colour ranges. Transparent rasters compare geometry, not calibrated brightness. Select fewer runs for unambiguous overlay; exports facet larger selections in groups of three.'),
    button('Build comparison', () => buildCompare(), !ST.selected.length),
    allMaps.length ? plot('multi-run comparison', traces, layout('Shared absolute stage coordinates — no alignment or interpolation', { ...axes,
      xaxis: { ...layout('').xaxis, range: bounds(xs), autorange: false }, yaxis: { ...layout('').yaxis, range: bounds(ys), autorange: false } }), 600) : note('No available overlay rasters loaded; build comparison or select a run with acquired data.'),
    table(['Run', 'Duration (s)', 'Tiles', 'Scans', 'Candidate detections*', 'Pursued', 'Particles / definitive', 'Stop reason', 'Key parameters'], runs.map((r) => {
      const s = r.summary, p = r.params;
      return [button(r.run_id, () => chooseRun(r.source)), fmt(s.duration_s), s.tiles, s.scans, s.candidate_detections ?? 'unavailable', s.pursued, `${s.particles} / ${s.definitive}`, s.stop_reason,
        `detectors ${numArray(p.detectors).join(',') || '?'} / ${p.detectorCombine || '?'}; coarse ${p.coarseStep ?? '?'} um; tile ${p.tileSize ?? '?'} um; weak ${p.weakRatio ?? '?'}x; strong ${p.strongRatio ?? '?'}x; maxTiles ${p.maxTiles ?? '?'}`];
    })), note('*Summed event candidate detections, not necessarily unique candidates. Coordinate-frame equality is assumed from recorded stage axes; no physical registration is inferred.'),
    ...runs.map((r) => h('details', {}, h('summary', {}, `${r.run_id}: all logged parameters / warnings`), h('pre', { style: 'white-space:pre-wrap' }, JSON.stringify(r.params, null, 2)), ...r.warnings.map(note))));
}

export function renderAutoFind(root) {
  if (!ST.root) ST.root = S.library.folder || '';
  const left = h('aside', { style: 'width:265px;min-width:230px;border-right:1px solid var(--line);overflow:auto;padding:12px' }, h('h3', {}, 'AutoFind runs'),
    h('div', { class: 'row' }, textInput(ST.root, (v) => { ST.root = v; }, 'Run folder or parent')),
    h('div', { class: 'row', style: 'margin-top:7px;gap:5px' }, button('Browse', browseRoot), button('Discover', discover), button('Refresh', refresh)),
    note(ST.busy || ST.progress), ...ST.errors.map(note),
    ...ST.runs.map((r) => h('div', { style: 'margin:8px 0;padding:7px;border:1px solid var(--line);border-radius:5px' },
      h('div', { class: 'row' }, h('input', { type: 'checkbox', checked: ST.selected.includes(r.source), 'aria-label': `Compare ${r.run_id}`, onchange: (e) => {
        if (e.target.checked && ST.selected.length >= 24) { e.target.checked = false; toast('Select up to 24 runs.'); return; }
        ST.selected = e.target.checked ? [...ST.selected, r.source] : ST.selected.filter((p) => p !== r.source); emit();
      } }), button(r.run_id, () => chooseRun(r.source))), note(`${r.name}\n${r.scans} scans · ${r.tiles} tiles · ${fmt(r.duration_s)} s${r.has_log ? '' : ' · no log'}`))),
    note('Each log is a separate run, including logs sharing one folder. Indexing does not load raster data.'));
  const main = h('main', { style: 'flex:1;min-width:0;overflow:auto;padding:14px' },
    h('div', { class: 'row', style: 'flex-wrap:wrap;gap:8px' }, ...[['map', 'Stitched map'], ['timeline', 'Timeline'], ['pursuit', 'Candidate pursuit'], ['compare', 'Multi-run comparison']].map(([k, label]) =>
      h('button', { class: `btn sm ${ST.view === k ? 'primary' : ''}`, onclick: () => action(label, () => changeView(k)) }, label)),
      select(MODES, ST.mode, (v) => action('Metric', () => changeMode(v)), { style: 'width:220px;height:28px', title: 'Map metric', 'aria-label': 'Map metric' }),
      select(SCALES, ST.scale, (v) => { ST.scale = v; emit(); }, { style: 'width:230px;height:28px', title: 'Colour scale: log10 or linear (raw values)', 'aria-label': 'Colour scale' }),
      h('label', { class: 'row', style: 'gap:4px', title: ST.mode === 'ratio' ? 'Draw strongRatio / weakRatio contours' : 'Contours need the line-normalized ratio metric' }, h('input', { type: 'checkbox', checked: ST.contours, disabled: ST.mode !== 'ratio', onchange: (e) => { ST.contours = e.target.checked; emit(); } }), 'contours')),
    // Provenance and caveats stay one click away instead of pushing the plot down.
    (() => {
      const notes = ['Replotted saved evidence; positions um, times s (or labelled min/h), detector units read from perSecond. Default colour is log10 max(value,1) like autoFindLivePlot; for dimensionless metrics (ratios near 1) that blanks the background, so use log10 or linear to see it. No new particle/bleaching verdict is inferred.', ...(ST.run?.warnings || [])];
      return h('details', { class: 'af-notes', open: !!ST.notesOpen, ontoggle: (e) => { ST.notesOpen = e.target.open; } },
        h('summary', { class: 'dim', style: 'cursor:pointer;margin:6px 0' }, `Provenance & caveats (${notes.length})`), ...notes.map(note));
    })(),
    ST.run ? (ST.view === 'map' ? mapView() : ST.view === 'timeline' ? timelineView() : ST.view === 'pursuit' ? pursuitView() : compareView()) : note('Open a run folder or its parent, Discover, then select a run.'),
    h('hr'), h('h3', {}, 'Export figure set'),
    h('div', { class: 'row', style: 'gap:6px;flex-wrap:wrap' }, textInput(ST.exportDir, (v) => { ST.exportDir = v; }, 'Output folder (outside acquisition data)'),
      button('Browse output', async () => { const p = await call('dialog', { kind: 'open_folder', title: 'Choose figure export directory', directory: ST.exportDir || '' }); if (p) ST.exportDir = p; }), h('button', { class: 'btn sm', disabled: !ST.source, onclick: () => action('Export figure set', exportSet, { preempt: false }) }, 'Export figure set')),
    note('White backgrounds, 300 DPI, unique new directory; includes maps, timeline, pursuit pages, comparison, CSV tables and provenance manifest.'),
    ST.exportResult ? note(`${ST.exportResult.pngs.length} PNGs written to ${ST.exportResult.directory}\nErrors: ${JSON.stringify(ST.exportResult.errors)}`) : null);
  mount(root, h('div', { style: 'display:flex;flex:1;min-width:0;min-height:0' }, left, main));
}
