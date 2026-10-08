// Application shell: top bar, library, routing, keyboard shortcuts, drag & drop, folder watching.
import { call, onBusy } from './api.js';
import { S, emit, subscribe, active, openFile, reloadFile, undo, redo, canUndo, canRedo, sessionDoc, applySession, detLabel } from './state.js';
import { h, mount, icon, toast, modal, contextMenu, fmt, fmtTime, fmtDate, basename, dirname, tag, debounce } from './util.js';
import { renderAnalyze, renderInspector, runFit, saveResults, copyFigure, resetView, ALT_TABS, kindOf } from './analyze.js';
import { renderCompare } from './compare.js';
import { renderFigure } from './figure.js';
import { renderBatch } from './batch.js';

const app = document.getElementById('app');
const els = {};

// ------------------------------------------------------------------ boot
async function boot() {
  const info = await call('app_info');
  S.version = info.version; S.desktop = info.desktop; S.settings = info.settings;
  S.home = info.home;
  S.library.folder = S.settings.recent_folders?.[0] || null;
  buildShell();
  if (S.library.folder) await openFolder(S.library.folder, { remember: false }).catch(() => { S.library.folder = null; });
  for (const p of info.pending_open || []) await openPath(p);
  render();
  setInterval(pollPending, 1500);
  setInterval(watchTick, 3000);
}

function buildShell() {
  els.top = h('header', { class: 'topbar' });
  els.lib = h('aside', { class: 'library' });
  els.main = h('main', { class: 'main' });
  els.insp = h('aside', { class: 'inspector' });
  els.inspHandle = h('div', { class: 'resize-handle insp-resize hidden', title: 'Drag to resize · double-click to reset', onmousedown: (e) => startResize(e, 'inspector'), ondblclick: () => setPanelWidth('inspector', 350, true) });
  els.full = h('div', { style: 'flex:1;display:flex;min-width:0;min-height:0' });
  els.status = h('footer', { class: 'status' });
  mount(app, els.top, h('div', { class: 'body' }, els.lib, els.main, els.inspHandle, els.insp, els.full), els.status);
  onBusy((n, label) => renderStatus(n, label));
  subscribe((what) => (what === 'insp' ? renderInspectorOnly() : render()));
}

let renderQueued = false;
function render() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderTop();
    renderLibrary();
    const v = S.view;
    const analyze = v === 'analyze' || v === 'home';
    els.main.classList.toggle('hidden', !analyze);
    els.insp.classList.toggle('hidden', !(v === 'analyze' && active()));
    els.full.classList.toggle('hidden', analyze);
    if (v === 'home' || (v === 'analyze' && !active())) renderHome(els.main);
    else if (v === 'analyze') renderAnalyze(els.main, els.insp);
    else if (v === 'compare') renderCompare(els.full);
    else if (v === 'figure') renderFigure(els.full);
    else if (v === 'batch') renderBatch(els.full);
    applyPanelWidths();
    renderStatus(0, '');
  });
}

function renderInspectorOnly() {
  const f = active();
  if (S.view === 'analyze' && f) {
    if (S.editor.open) render();
    else renderInspector(f, els.insp);
  } else render();
}

// Shared list for the Start-screen card and the "?" overlay.
const SHORTCUTS = [['Open files', 'Ctrl O'], ['Add files', 'Ctrl Shift O'], ['Search files / commands', 'Ctrl K'], ['Command palette', 'Ctrl Shift P'], ['Run fit · robust fit', 'Ctrl F · Ctrl Shift F'], ['Save results · session', 'Ctrl S · Ctrl Shift S'],
  ['Open session', 'Ctrl Alt O'], ['Undo · redo', 'Ctrl Z · Ctrl Y'], ['Copy figure', 'Ctrl Shift C'], ['Reset plot view', 'Ctrl Home'], ['Inspector tabs', 'Alt 1–7'], ['Next · previous library file', 'Alt Down · Alt Up'], ['Clear selection / search', 'Esc'], ['This shortcut list', '?']];

// ------------------------------------------------------------------ top bar
function renderTop() {
  const nav = [['analyze', 'Analyze'], ['compare', 'Compare'], ['figure', 'Figure'], ['batch', 'Batch']];
  mount(els.top,
    h('div', { class: 'brand', title: 'Start screen', onclick: () => { S.view = 'home'; emit(); } }, icon('logo', 22, '#5AA9F0'), 'SmartFitter'),
    h('nav', { class: 'nav', 'aria-label': 'Workspaces' }, nav.map(([k, l]) => h('button', { class: S.view === k || (k === 'analyze' && S.view === 'home' && active()) ? 'on' : '', onclick: () => { S.view = k; emit(); } }, l))),
    h('button', { class: 'btn search-btn', onclick: openPalette }, h('span', { class: 'row' }, icon('search', 14), 'Search files and commands'), h('span', { class: 'kbd' }, 'Ctrl K')),
    h('div', { class: 'sp' }),
    h('button', { class: 'ibtn', title: 'Keyboard shortcuts (?)', 'aria-label': 'Keyboard shortcuts', onclick: openShortcuts }, '?'),
    h('button', { class: 'ibtn', title: 'Undo (Ctrl+Z)', disabled: !canUndo(), onclick: undo }, icon('undo')),
    h('button', { class: 'ibtn', title: 'Redo (Ctrl+Y)', disabled: !canRedo(), onclick: redo }, icon('redo')),
    h('button', { class: 'btn', onclick: openFilesDialog }, icon('open', 14), 'Open', h('span', { class: 'kbd' }, 'Ctrl O')),
    h('button', { class: 'btn primary', disabled: !active(), onclick: () => saveResults(active()) }, 'Save results', h('span', { class: 'kbd' }, 'Ctrl S')));
}

function renderStatus(n, label) {
  const f = active();
  const busy = n > 0 ? h('span', { class: 'row busy' }, h('span', { class: 'spinner' }), label || 'Working') : null;
  mount(els.status,
    h('span', {}, `${Object.keys(S.files).length} file${Object.keys(S.files).length === 1 ? '' : 's'} open${f ? ` · ${detLabel(f, f.detector)} · ${f.mode}` : ''}`),
    S.library.watching ? h('span', { class: 'row', style: 'gap:5px;color:var(--ok)' }, h('span', { class: 'watch-dot' }), 'Watching folder') : null,
    busy,
    h('span', { class: 'sp' }),
    h('span', {}, 'Alt 1–7 inspector · Ctrl+K search · Ctrl+F fit · Ctrl+Shift+C copy figure'),
    h('span', { class: 'mono' }, `v${S.version}`));
}

// ------------------------------------------------------------------ library
export async function openFolder(path, { remember = true } = {}) {
  const listing = await call('list_dir', { path }, { label: 'Reading folder' });
  S.library.folder = listing.path;
  S.library.listing = listing;
  const seen = S.library.seen[listing.path] || {};
  if (!S.library.seen[listing.path]) listing.files.forEach((f) => { seen[f.path] = f.mtime; });
  S.library.seen[listing.path] = seen;
  if (remember) { S.settings = await call('remember', { kind: 'recent_folders', path: listing.path }); call('settings_set', { patch: { last_dir: listing.path } }, { quiet: true }); }
  requestPeeks(listing.files.map((f) => f.path));
  emit();
}

const peekQueue = new Set();
const flushPeeks = debounce(async () => {
  const paths = [...peekQueue].slice(0, 40);
  paths.forEach((p) => peekQueue.delete(p));
  if (!paths.length) return;
  try {
    const res = await call('peek', { paths }, { quiet: true, label: 'Scanning files' });
    res.forEach((r) => { S.library.peeks[r.path] = r; });
    renderLibrary();
  } catch (e) { /* keep going */ }
  if (peekQueue.size) flushPeeks();
}, 30);
function requestPeeks(paths) { paths.filter((p) => !S.library.peeks[p]).forEach((p) => peekQueue.add(p)); flushPeeks(); }

function fileMatches(p, info) {
  const lib = S.library;
  const chip = lib.chip;
  if (chip === '1d' && !(info.dim === 'time' || info.dim === 'scan1d') ) return false;
  if (chip === '1d' && info.type === 'TimeTrace') return false;
  if (chip === '2d' && info.dim !== 'scan2d') return false;
  if (chip === 'tt' && info.type !== 'TimeTrace') return false;
  if (chip === 'fitted' && !(S.files[p] && Object.keys(S.files[p].fit).length)) return false;
  const words = lib.filter.toLowerCase().split(/\s+/).filter(Boolean);
  const hay = `${basename(p)} ${dirname(p)} ${info.type || ''} ${info.tag || ''} ${info.dim || ''} ${info.detectors > 1 ? 'dual detector2' : ''}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}

function fileRow(p, meta) {
  const info = S.library.peeks[p] || {};
  const f = S.files[p];
  const isPrimary = S.primary === p && S.view === 'analyze';
  const prim = active();
  const canOverlay = prim && prim.path !== p && kindOf(prim) === '1d' && info.dim && info.dim !== 'scan2d' && info.type !== 'TimeTrace';
  const newer = S.library.newSince[p];
  const sub = [];
  if (meta?.mtime) sub.push(fmtTime(meta.mtime));
  if (info.grid) sub.push(`${info.grid[1]} × ${info.grid[0]}`);
  else if (info.points) sub.push(`${info.points} pts`);
  if (info.detectors > 1) sub.push('2 detectors');
  if (info.iterations && info.iterations !== '/') sub.push(`${info.iterations} it`);
  if (f && f.fit[f.detector]) sub.push(`fit R² ${fmt(f.fit[f.detector].r2, 3)}`);
  if (info.error) sub.push('unreadable');
  return h('div', {
    class: `frow ${isPrimary ? 'sel primary' : ''} ${newer ? 'newfile' : ''}`, title: info.error ? `${p}\n${info.error}` : p, 'data-path': p,
    onclick: (e) => { if (e.target.type === 'checkbox') return; openPath(p); },
    oncontextmenu: (e) => { e.preventDefault(); fileMenu(e, p); },
  },
  canOverlay ? h('input', { type: 'checkbox', checked: S.overlays.includes(p), title: 'Overlay on the current plot', 'aria-label': `Overlay ${basename(p)}`,
    onchange: async (e) => { if (e.target.checked) { await openFile(p, { primary: false }); S.overlays = [...S.overlays, p]; } else S.overlays = S.overlays.filter((q) => q !== p); emit(); } }) : null,
  tag(info.tag),
  h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'fname ellip' }, basename(p).replace(/\.mat$/i, '')),
    h('div', { class: 'sub ellip' }, isPrimary ? `Primary · ${sub.join(' · ')}` : sub.join(' · '))));
}

function fileMenu(e, p) {
  contextMenu(e.clientX, e.clientY, [
    { label: 'Open', run: () => openPath(p) },
    { label: 'Add to Compare set', run: async () => { await openFile(p, { primary: false }); if (!S.compare.set.includes(p)) S.compare.set.push(p); toast('Added to Compare.'); emit(); } },
    active() && active().path !== p ? { label: 'Subtract from current file (baseline)', run: async () => { await openFile(p, { primary: false }); const f = active(); const { setProc } = await import('./analyze.js'); setProc(f, { baseline: { path: p, detector: f.detector } }, 'baseline file'); } } : null,
    { label: 'Show in folder', run: () => call('reveal', { path: p }) },
    { label: 'Copy path', run: () => navigator.clipboard.writeText(p).then(() => toast('Path copied')) },
    S.files[p] ? { label: 'Close file', run: () => closeFile(p) } : null,
  ]);
}

function closeFile(p) {
  delete S.files[p];
  S.overlays = S.overlays.filter((q) => q !== p);
  S.compare.set = S.compare.set.filter((q) => q !== p);
  S.library.extra = S.library.extra.filter((q) => q !== p);
  if (S.primary === p) S.primary = Object.keys(S.files)[0] || null;
  emit();
}

function renderLibrary() {
  const lib = S.library;
  if (S.libCollapsed || !['analyze', 'home'].includes(S.view)) {
    els.lib.className = 'library rail';
    mount(els.lib, h('button', { class: 'ibtn', title: 'Show library', 'aria-label': 'Show library', onclick: () => { S.libCollapsed = false; if (!['analyze', 'home'].includes(S.view)) S.view = 'analyze'; emit(); } }, icon('folder', 18)),
      Object.keys(S.files).slice(0, 14).map((p) => h('button', { class: 'ibtn', title: basename(p), style: S.primary === p ? 'background:var(--sel)' : '', onclick: () => openPath(p) }, tag((lib.peeks[p] || S.files[p]?.info || {}).tag))));
    return;
  }
  els.lib.className = 'library';
  const listing = lib.listing;
  const chips = [['all', 'All'], ['1d', '1D'], ['2d', '2D'], ['tt', 'Time trace'], ['fitted', 'Fitted']];
  const filterEl = h('input', { class: 'field', id: 'lib-filter', placeholder: 'Filter: name, type, detector  (Ctrl K)', value: lib.filter, 'aria-label': 'Filter files' });
  filterEl.addEventListener('input', debounce(() => { lib.filter = filterEl.value; renderLibList(); }, 120));
  filterEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { const first = els.libList.querySelector('.frow'); if (first) first.click(); }
    if (e.key === 'Escape') { lib.filter = ''; filterEl.value = ''; renderLibList(); filterEl.blur(); }
    e.stopPropagation();
  });
  els.libList = h('div', { class: 'lib-list' });
  mount(els.lib,
    h('div', { class: 'lib-head' },
      h('div', { class: 'row' }, h('span', { class: 'lbl' }, 'Library'), h('span', { class: 'sp' }),
        h('button', { class: 'ibtn', title: 'Add files (Ctrl+Shift+O)', onclick: () => openFilesDialog(true) }, icon('plus', 15)),
        h('button', { class: 'ibtn', title: 'Choose folder', onclick: chooseFolder }, icon('folder', 15)),
        h('button', { class: 'ibtn', title: 'Collapse library', onclick: () => { S.libCollapsed = true; emit(); } }, icon('chevL', 15))),
      listing ? h('div', { class: 'folder-pill', title: `${listing.path}\nClick for recent folders`, onclick: folderMenu },
        icon('folder', 14, '#A2AAB7'),
        h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'ellip' }, listing.name),
          h('div', { style: 'font-size:11px;color:var(--text3)' }, lib.watching ? h('span', { style: 'color:var(--ok)' }, '● Watching · new files appear live') : `${listing.files.length} .mat file${listing.files.length === 1 ? '' : 's'}`)),
        h('button', { class: `btn sm ${lib.watching ? 'primary' : ''}`, title: 'Watch this folder for new and updated files', onclick: (e) => { e.stopPropagation(); lib.watching = !lib.watching; toast(lib.watching ? 'Watching for new files and checkpoint updates.' : 'Stopped watching.'); emit(); } }, lib.watching ? 'Watching' : 'Watch'))
        : h('button', { class: 'btn block', onclick: chooseFolder }, 'Choose a data folder'),
      filterEl,
      h('div', { class: 'row wrap', style: 'gap:6px' }, chips.map(([k, l]) => h('button', { class: `chip sm ${lib.chip === k ? 'on' : ''}`, onclick: () => { lib.chip = k; renderLibrary(); } }, l)))),
    els.libList,
    h('div', { style: 'border-top:1px solid var(--line);padding:8px 12px;display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;font-size:12px;color:var(--text2)' },
      h('button', { class: 'btn sm', title: 'Save session (Ctrl+Shift+S)', onclick: () => saveSessionAs() }, 'Save session'), h('button', { class: 'btn sm', onclick: openSessionDialog }, 'Open session'), h('span', { class: 'sp' }),
      h('label', { class: 'ck', style: 'font-size:11px', title: 'Re-run the fit automatically when a watched open file is updated' },
        h('input', { type: 'checkbox', checked: S.settings.auto_refit !== false, onchange: (e) => { S.settings.auto_refit = e.target.checked; call('settings_set', { patch: { auto_refit: e.target.checked } }, { quiet: true }).catch(() => {}); } }),
        'Refit on update')),
    h('div', { class: 'resize-handle lib-resize', title: 'Drag to resize · double-click to reset', onmousedown: (e) => startResize(e, 'library'), ondblclick: () => setPanelWidth('library', 280, true) }));
  applyPanelWidths();
  renderLibList();
}

// ------------------------------------------------------------------ resizable panels
const PANEL_LIMITS = { library: [200, 480, 280], inspector: [300, 560, 350] };
function panelWidth(name) {
  const [lo, hi, def] = PANEL_LIMITS[name];
  const w = Number(S.settings.panel_widths?.[name]);
  return Number.isFinite(w) && w > 0 ? Math.min(hi, Math.max(lo, w)) : def;
}
function applyPanelWidths() {
  const lib = els.lib, insp = els.insp;
  if (lib && !lib.classList.contains('rail')) { const w = `${panelWidth('library')}px`; lib.style.width = w; lib.style.flexBasis = w; }
  else if (lib) { lib.style.width = ''; lib.style.flexBasis = ''; }
  if (insp) {
    const w = `${panelWidth('inspector')}px`; insp.style.width = w; insp.style.flexBasis = w;
    els.inspHandle?.classList.toggle('hidden', insp.classList.contains('hidden'));
  }
}
function setPanelWidth(name, w, save) {
  const [lo, hi] = PANEL_LIMITS[name];
  w = Math.round(Math.min(hi, Math.max(lo, w)));
  S.settings.panel_widths = { ...(S.settings.panel_widths || {}), [name]: w };
  applyPanelWidths();
  if (save) call('settings_set', { patch: { panel_widths: S.settings.panel_widths } }, { quiet: true }).catch(() => {});
}
function startResize(e, name) {
  e.preventDefault();
  const startX = e.clientX, start = panelWidth(name), sign = name === 'library' ? 1 : -1;
  document.body.classList.add('resizing');
  const move = (ev) => setPanelWidth(name, start + sign * (ev.clientX - startX), false);
  const up = () => {
    document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up);
    document.body.classList.remove('resizing');
    setPanelWidth(name, panelWidth(name), true);
  };
  document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
}

function folderMenu(e) {
  const cur = S.library.listing?.path;
  const items = (S.settings.recent_folders || []).slice(0, 8).map((p) => ({ label: `${p === cur ? '✓ ' : ''}${basename(p) || p}  —  ${p}`, run: () => openFolder(p) }));
  contextMenu(e.clientX, e.clientY, [
    ...items,
    { label: 'Choose folder…', run: chooseFolder },
    S.library.listing?.parent ? { label: 'Parent folder', run: () => openFolder(S.library.listing.parent) } : null,
  ]);
}

function renderLibList() {
  const lib = S.library;
  const list = els.libList;
  if (!list) return;
  const listing = lib.listing;
  const out = [];
  const openPaths = Object.keys(S.files).filter((p) => !listing || dirname(p).toLowerCase() !== listing.path.toLowerCase());
  const extra = [...new Set([...openPaths, ...lib.extra])].filter((p) => fileMatches(p, lib.peeks[p] || S.files[p]?.info || {}));
  if (extra.length) {
    out.push(h('div', { class: 'group-lbl lbl' }, 'Open elsewhere'));
    extra.forEach((p) => { if (!lib.peeks[p]) requestPeeks([p]); out.push(fileRow(p, null)); });
  }
  if (listing) {
    if (listing.parent || listing.folders.length) {
      out.push(h('div', { class: 'group-lbl' }, h('span', { class: 'lbl' }, 'Folders')));
      if (listing.parent) out.push(h('div', { class: 'subfolder', onclick: () => openFolder(listing.parent) }, icon('up', 13), '..'));
      listing.folders.forEach((d) => out.push(h('div', { class: 'subfolder', title: d.path, onclick: () => openFolder(d.path) }, icon('folder', 13), h('span', { class: 'ellip' }, d.name))));
    }
    const files = [...listing.files].sort((a, b) => b.mtime - a.mtime).filter((f) => fileMatches(f.path, lib.peeks[f.path] || {}));
    let lastDay = '';
    out.push(h('div', { class: 'group-lbl' }, h('span', { class: 'lbl' }, `${listing.name}`), h('span', { class: 'dim', style: 'font-size:11px' }, `${files.length}/${listing.files.length}`)));
    files.forEach((f) => {
      const day = fmtDate(f.mtime);
      if (day !== lastDay) { out.push(h('div', { class: 'lbl', style: 'padding:8px 8px 2px;font-size:10px' }, day)); lastDay = day; }
      out.push(fileRow(f.path, f));
    });
    if (!listing.files.length) out.push(h('div', { class: 'note', style: 'padding:8px' }, 'No .mat files in this folder. Open a subfolder above.'));
  }
  if (!out.length) out.push(h('div', { class: 'note', style: 'padding:10px' }, 'Choose a data folder, drop .mat files on the window, or press Ctrl+O.'));
  mount(list, out);
}

// ------------------------------------------------------------------ opening
export async function openPath(p) {
  try {
    if (/\.sfsession\.json$|\.json$/i.test(p)) return loadSession(p);
    if (S.library.listing && dirname(p).toLowerCase() !== S.library.listing.path.toLowerCase() && !S.library.extra.includes(p)) S.library.extra.push(p);
    delete S.library.newSince[p];
    await openFile(p);
    requestPeeks([p]);
  } catch (e) { /* toast shown by api */ }
}

async function openFilesDialog(add = false) {
  const paths = await call('dialog', { kind: 'open_files', title: add ? 'Add data files' : 'Open data files' });
  if (!paths?.length) return;
  for (const [i, p] of paths.entries()) {
    if (i === 0 && !add) await openPath(p);
    else { await openFile(p, { primary: !add && i === 0 }); if (!S.library.extra.includes(p)) S.library.extra.push(p); requestPeeks([p]); }
  }
  if (add && !S.primary) await openPath(paths[0]);
  call('settings_set', { patch: { last_dir: dirname(paths[0]) } }, { quiet: true });
  emit();
}

async function chooseFolder() {
  const p = await call('dialog', { kind: 'open_folder', title: 'Choose a data folder', directory: S.library.folder || '' });
  if (p) openFolder(p);
}

// ------------------------------------------------------------------ sessions
export async function saveSession(path) { await call('save_session', { path, doc: sessionDoc() }, { label: 'Saving session' }); toast(`Session saved: ${basename(path)}`, 'ok'); }
async function saveSessionAs() {
  const f = active();
  const p = await call('dialog', { kind: 'save', title: 'Save analysis session', filename: `${f ? basename(f.path).replace(/\.mat$/i, '') : 'analysis'}.sfsession.json`, directory: f ? dirname(f.path) : '' });
  if (p) saveSession(p.endsWith('.json') ? p : `${p}.sfsession.json`);
}
async function openSessionDialog() {
  const p = await call('dialog', { kind: 'open_files', title: 'Open analysis session', filters: ['SmartFitter session (*.sfsession.json;*.json)'] });
  if (p?.length) loadSession(p[0]);
}
async function loadSession(p) { const doc = await call('load_session', { path: p }, { label: 'Opening session' }); await applySession(doc); if (S.library.folder) openFolder(S.library.folder, { remember: false }); }

// ------------------------------------------------------------------ home
function renderHome(main) {
  const st = S.settings;
  const recentFolders = (st.recent_folders || []).slice(0, 8);
  const recentFiles = (st.recent_files || []).slice(0, 8);
  mount(main, h('div', { style: 'width:100%;max-width:1180px;margin:0 auto;padding:20px 12px;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;align-content:start' },
    h('section', { style: 'grid-column:1/-1' }, h('h1', { style: 'margin:0;font-size:26px;font-weight:600' }, 'Open data'),
      h('p', { class: 'muted', style: 'margin:6px 0 0' }, 'MATLAB .mat files from the acquisition code: snapshots, checkpoints, legacy savedData, time traces, 1D and 2D scans.')),
    h('section', { class: 'card dropzone', style: 'grid-column:1/-1' }, icon('up', 32, '#7DBBF5'), h('div', { style: 'font-size:16px' }, 'Drop .mat files or a folder anywhere on this window'),
      h('div', { class: 'row wrap', style: 'justify-content:center' }, h('button', { class: 'btn primary', onclick: () => openFilesDialog() }, 'Open files', h('span', { class: 'kbd' }, 'Ctrl O')),
        h('button', { class: 'btn', onclick: chooseFolder }, 'Open folder'), h('button', { class: 'btn', onclick: openSessionDialog }, 'Open session', h('span', { class: 'kbd' }, 'Ctrl Alt O')))),
    h('section', { class: 'card', style: 'padding:14px;display:flex;flex-direction:column;gap:2px' }, h('div', { class: 'lbl', style: 'padding:0 10px 6px' }, 'Recent folders'),
      recentFolders.length ? recentFolders.map((p) => h('div', { class: 'frow', title: p, onclick: () => { openFolder(p); S.view = 'analyze'; S.libCollapsed = false; emit(); } }, icon('folder', 14, '#A2AAB7'), h('span', { class: 'ellip', style: 'flex:1' }, basename(p)), h('span', { class: 'dim ellip', style: 'font-size:11px;max-width:45%' }, dirname(p))))
        : h('div', { class: 'note', style: 'padding:0 10px' }, 'Folders you open appear here.')),
    h('section', { class: 'card', style: 'padding:14px;display:flex;flex-direction:column;gap:2px' }, h('div', { class: 'lbl', style: 'padding:0 10px 6px' }, 'Recent files and sessions'),
      recentFiles.length ? recentFiles.map((p) => h('div', { class: 'frow', title: p, onclick: () => openPath(p) }, h('span', { class: 'ellip', style: 'flex:1' }, basename(p)), h('span', { class: 'dim ellip', style: 'font-size:11px;max-width:45%' }, basename(dirname(p)))))
        : h('div', { class: 'note', style: 'padding:0 10px' }, 'Nothing yet.')),
    h('section', { class: 'card', style: 'padding:14px 18px;display:flex;flex-direction:column;gap:10px' }, h('div', { class: 'lbl' }, 'Watch a folder while acquiring'),
      h('p', { class: 'muted', style: 'margin:0;line-height:1.5' }, 'Open the folder MATLAB saves into and press Watch in the Library. New files appear as they are written, and the open plot refreshes when its checkpoint file updates.'),
      h('div', { class: 'row wrap' }, h('button', { class: 'btn', onclick: async () => { const p = 'D:\\BacklundLabResearch\\Code\\Backlund-Lab-Code-Base\\Saved Data'; await openFolder(p).then(() => { S.library.watching = true; S.view = 'analyze'; emit(); }).catch(() => {}); } }, 'Watch Saved Data'), h('button', { class: 'btn', onclick: async () => { await chooseFolder(); S.library.watching = true; S.view = 'analyze'; emit(); } }, 'Choose folder'))),
    h('section', { class: 'card', style: 'padding:14px 18px;display:flex;flex-direction:column;gap:10px' }, h('div', { class: 'lbl' }, 'Keyboard shortcuts'),
      h('div', { class: 'kv' }, SHORTCUTS.flatMap(([a, b]) => [h('span', { class: 'k' }, a), h('span', {}, h('span', { class: 'kbd' }, b))]))),
  ));
  mount(els.insp);
}

// ------------------------------------------------------------------ command palette
function commands() {
  const f = active();
  return [
    ['Open files…', 'Ctrl O', () => openFilesDialog()], ['Add files…', 'Ctrl Shift O', () => openFilesDialog(true)], ['Choose data folder…', '', chooseFolder],
    ['Run fit', 'Ctrl F', () => f && runFit(f)], ['Robust fit', 'Ctrl Shift F', () => f && runFit(f, { robust: true })],
    ['Save results', 'Ctrl S', () => f && saveResults(f)], ['Save session…', 'Ctrl Shift S', saveSessionAs], ['Open session…', 'Ctrl Alt O', openSessionDialog],
    ['Copy figure', 'Ctrl Shift C', () => f && copyFigure(f)], ['Reset plot view', 'Ctrl Home', () => f && resetView(f)], ['Edit plot', '', () => { S.editor.open = true; emit(); }],
    ['Go to Analyze', '', () => { S.view = 'analyze'; emit(); }], ['Go to Compare', '', () => { S.view = 'compare'; emit(); }], ['Go to Figure composer', '', () => { S.view = 'figure'; emit(); }], ['Go to Batch', '', () => { S.view = 'batch'; emit(); }],
    ['Add current plot to Figure', '', () => import('./figure.js').then((m) => m.addCurrentPlot())],
    ['Toggle library', '', () => { S.libCollapsed = !S.libCollapsed; emit(); }],
    ['Watch current folder', '', () => { S.library.watching = !S.library.watching; emit(); }],
    ['Show file in folder', '', () => f && call('reveal', { path: f.path })],
    ['Next file in library', 'Alt Down', () => stepLibrary(1)], ['Previous file in library', 'Alt Up', () => stepLibrary(-1)],
    ['Undo', 'Ctrl Z', undo], ['Redo', 'Ctrl Y', redo],
    ['Keyboard shortcuts', '?', openShortcuts],
  ].map(([label, kbd, run]) => ({ label, kbd, run, kind: 'cmd' }));
}

function openPalette() {
  const inputEl = h('input', { class: 'field', placeholder: 'Type a file name or a command…', 'aria-label': 'Search' });
  const items = h('div', { class: 'items' });
  let act = 0, current = [];
  const files = () => {
    const lib = S.library;
    const all = new Set([...Object.keys(S.files), ...lib.extra, ...(lib.listing?.files || []).map((f) => f.path), ...(S.settings.recent_files || [])]);
    return [...all].map((p) => ({ label: basename(p), sub: dirname(p), path: p, kind: 'file', info: lib.peeks[p] || S.files[p]?.info || {} }));
  };
  const draw = () => {
    const q = inputEl.value.toLowerCase().split(/\s+/).filter(Boolean);
    const match = (s) => q.every((w) => s.toLowerCase().includes(w));
    current = [...files().filter((f) => match(`${f.label} ${f.sub} ${f.info.type || ''} ${f.info.tag || ''}`)).slice(0, 30), ...commands().filter((c) => match(c.label))].slice(0, 60);
    act = Math.min(act, Math.max(0, current.length - 1));
    mount(items, current.map((it, i) => h('div', { class: `item ${i === act ? 'act' : ''}`, onmouseenter: () => { act = i; draw(); }, onclick: () => choose(it) },
      it.kind === 'file' ? tag(it.info.tag) : icon('chevR', 14), h('div', { style: 'flex:1;min-width:0' }, h('div', { class: 'ellip' }, it.label), it.sub ? h('div', { class: 'dim ellip', style: 'font-size:11px' }, it.sub) : null),
      it.kbd ? h('span', { class: 'kbd' }, it.kbd) : null)));
    items.querySelector('.act')?.scrollIntoView({ block: 'nearest' });
  };
  const m = modal(h('div', { class: 'palette', style: 'display:flex;flex-direction:column;min-height:0' }, inputEl, items));
  const choose = (it) => { m.close(); it.kind === 'file' ? openPath(it.path) : it.run(); };
  inputEl.addEventListener('input', () => { act = 0; draw(); });
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { act = Math.min(act + 1, current.length - 1); draw(); e.preventDefault(); }
    if (e.key === 'ArrowUp') { act = Math.max(act - 1, 0); draw(); e.preventDefault(); }
    if (e.key === 'Enter' && current[act]) choose(current[act]);
    e.stopPropagation();
  });
  draw();
  inputEl.focus();
}

// ------------------------------------------------------------------ shortcuts overlay
function openShortcuts() {
  if (document.querySelector('.modal.shortcuts')) return;
  const half = Math.ceil(SHORTCUTS.length / 2);
  const cols = [SHORTCUTS.slice(0, half), SHORTCUTS.slice(half)];
  modal(h('div', { style: 'padding:18px 20px;display:flex;flex-direction:column;gap:14px' },
    h('div', { class: 'row', style: 'justify-content:space-between' }, h('div', { class: 'lbl' }, 'Keyboard shortcuts'), h('span', { class: 'dim', style: 'font-size:11px' }, 'Esc to close')),
    h('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:10px 28px' },
      cols.map((col) => h('div', { class: 'kv' }, col.flatMap(([a, b]) => [h('span', { class: 'k' }, a), h('span', {}, h('span', { class: 'kbd' }, b))]))))),
    { className: 'shortcuts' });
}

// ------------------------------------------------------------------ keyboard
// Open the next/previous file of the visible (filtered) library list, relative to the primary file.
function stepLibrary(dir) {
  const rows = [...(els.libList?.querySelectorAll('.frow') || [])];
  const paths = rows.map((r) => r.getAttribute('data-path')).filter(Boolean);
  if (!paths.length) return;
  const i = paths.indexOf(S.primary);
  const next = i < 0 ? (dir > 0 ? 0 : paths.length - 1) : i + dir;
  if (next < 0 || next >= paths.length) return;
  openPath(paths[next]).then(() => els.libList?.querySelector(`.frow[data-path="${CSS.escape(paths[next])}"]`)?.scrollIntoView({ block: 'nearest' }));
}

document.addEventListener('keydown', (e) => {
  const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
  const k = e.key.toLowerCase();
  const f = active();
  const ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && k === 'k') { e.preventDefault(); if (S.view === 'analyze' && !S.libCollapsed) document.getElementById('lib-filter')?.focus(); else openPalette(); return; }
  if (ctrl && e.shiftKey && k === 'p') { e.preventDefault(); openPalette(); return; }
  if (ctrl && e.altKey && k === 'o') { e.preventDefault(); openSessionDialog(); return; }
  if (ctrl && e.shiftKey && k === 'o') { e.preventDefault(); openFilesDialog(true); return; }
  if (ctrl && k === 'o') { e.preventDefault(); openFilesDialog(); return; }
  if (ctrl && e.shiftKey && k === 's') { e.preventDefault(); saveSessionAs(); return; }
  if (ctrl && k === 's') { e.preventDefault(); if (f) saveResults(f); return; }
  if (ctrl && e.shiftKey && k === 'c') { e.preventDefault(); if (f) copyFigure(f); return; }
  if (ctrl && e.shiftKey && k === 'f') { e.preventDefault(); if (f) runFit(f, { robust: true }); return; }
  if (ctrl && k === 'f') { e.preventDefault(); if (f) runFit(f); return; }
  if (ctrl && k === 'home') { e.preventDefault(); if (f) resetView(f); return; }
  if (typing) return;
  if (e.key === '?' && !ctrl && !e.altKey) { e.preventDefault(); openShortcuts(); return; }
  if (e.altKey && !ctrl && !e.shiftKey && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) { e.preventDefault(); stepLibrary(e.key === 'ArrowDown' ? 1 : -1); return; }
  if (ctrl && k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
  if (ctrl && k === 'y') { e.preventDefault(); redo(); return; }
  if (e.altKey && /^[1-8]$/.test(e.key) && f) {
    e.preventDefault();
    const want = ALT_TABS[e.key];
    const map2d = { fit: 'map', proc: 'map', plot: 'map', map: 'map', export: 'export', info: 'info' };
    const mapTT = { fit: 'fit', proc: 'proc', plot: 'plot', map: 'stats', export: 'export', info: 'info' };
    S.editor.open = false;
    f.tab = kindOf(f) === '2d' ? (e.key === '6' ? 'spots' : map2d[want]) : kindOf(f) === 'tt' ? mapTT[want] : (want === 'map' ? 'fit' : want);
    S.view = 'analyze';
    emit();
    return;
  }
  if (e.key === 'Escape') {
    document.querySelectorAll('.popover,.ctx').forEach((p) => p.remove());
    if (f?.pins?.length) { f.pins = []; emit(); }
    else if (f?.measure?.length) { f.measure = []; emit(); }
    else if (S.editor.open) { S.editor.open = false; emit(); }
  }
});

// ------------------------------------------------------------------ drag & drop
let dragDepth = 0;
const dropOverlay = h('div', { class: 'drop-overlay hidden' }, 'Drop .mat files or a folder to open');
document.body.append(dropOverlay);
window.addEventListener('dragenter', (e) => { if (e.dataTransfer?.types?.includes('Files')) { dragDepth++; dropOverlay.classList.remove('hidden'); } });
window.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) dropOverlay.classList.add('hidden'); });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0; dropOverlay.classList.add('hidden');
  const files = [...(e.dataTransfer?.files || [])];
  const paths = [];
  for (const file of files) {
    const full = file.pywebviewFullPath || file.path;
    if (full) paths.push(full);
    else if (/\.mat$/i.test(file.name)) {
      const data = await new Promise((res) => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(file); });
      paths.push(await call('save_upload', { name: file.name, data }, { label: 'Receiving file' }));
    }
  }
  if (!paths.length) return;
  for (const [i, p] of paths.entries()) {
    if (!/\.\w+$/.test(p)) { await openFolder(p); S.view = 'analyze'; continue; }
    if (i === 0) await openPath(p);
    else { await openFile(p, { primary: false }); if (!S.library.extra.includes(p)) S.library.extra.push(p); requestPeeks([p]); }
  }
  emit();
});

// ------------------------------------------------------------------ background polling
async function pollPending() {
  try {
    const items = await call('take_pending', {}, { quiet: true });
    for (const p of items || []) await openPath(p);
  } catch (e) { /* server busy */ }
}

async function watchTick() {
  const lib = S.library;
  const paths = Object.keys(S.files);
  try {
    if (lib.watching && lib.folder) {
      const listing = await call('list_dir', { path: lib.folder }, { quiet: true });
      const seen = lib.seen[listing.path] || {};
      const fresh = listing.files.filter((f) => !(f.path in seen));
      fresh.forEach((f) => { seen[f.path] = f.mtime; lib.newSince[f.path] = true; delete lib.peeks[f.path]; });
      listing.files.forEach((f) => { if (seen[f.path] !== f.mtime) { seen[f.path] = f.mtime; delete lib.peeks[f.path]; } });
      lib.seen[listing.path] = seen;
      lib.listing = listing;
      if (fresh.length) { toast(`New file${fresh.length > 1 ? 's' : ''}: ${fresh.map((f) => f.name).join(', ')}`); requestPeeks(fresh.map((f) => f.path)); renderLibList(); }
    }
    if (lib.watching && paths.length) {
      const st = await call('stat', { paths }, { quiet: true });
      for (const p of paths) {
        const f = S.files[p];
        if (st[p] && f.mtime && st[p] !== f.mtime) {
          const hadFit = !!f.fit[f.detector];
          f.mtime = st[p];
          await reloadFile(f);
          toast(`Updated: ${basename(p)} (new checkpoint)`);
          if (hadFit && S.settings.auto_refit !== false && !f.fitting) { try { await runFit(f); } catch (e) { /* toast shown by api */ } }
        }
        if (st[p]) f.mtime = st[p];
      }
    }
  } catch (e) { /* folder temporarily unavailable */ }
}

boot();
