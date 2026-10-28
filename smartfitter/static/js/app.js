// Application shell: top bar, library, routing, keyboard shortcuts, drag & drop, folder watching.
import { call, onBusy } from './api.js';
import { S, emit, subscribe, active, openFile, reloadFile, undo, redo, canUndo, canRedo, sessionDoc, applySession, detLabel, addOverlay, removeOverlay, addCompareItem, compareFile, isDirty, markClean } from './state.js';
import { h, mount, icon, toast, modal, contextMenu, fmt, fmtTime, fmtDate, basename, dirname, tag, debounce } from './util.js';
import { renderAnalyze, renderInspector, runFit, runFitScoped, saveResults, copyFigure, resetView, ALT_TABS, kindOf } from './analyze.js';
import { renderCompare, hasCompareTrend, hasCompareOverlay, saveCompareTrend, copyCompareOverlay } from './compare.js';
import { renderFigure, exportFigure } from './figure.js';
import { renderBatch, saveBatchSummary, copyBatchSummary } from './batch.js';
import { renderDay, hasDaySummary, saveDaySummary, copyDaySummary } from './day.js';
import { renderStack, hasStackResults, saveStackResults, copyStackResults } from './stack.js';
import { renderAutoFind, canExportAutoFind, exportAutoFind } from './autofind.js';
import { workspaceActions } from './workspaceactions.js';

const app = document.getElementById('app');
const els = {};

// ------------------------------------------------------------------ boot
async function boot() {
  const info = await call('app_info');
  S.version = info.version; S.desktop = info.desktop; S.settings = info.settings;
  S.home = info.home;
  Object.assign(S.library, { sort: 'date', type: 'all', ...S.settings.library_nav });
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
  els.inspHandle = h('div', { class: 'resize-handle insp-resize hidden', title: 'Drag to resize · double-click to reset', onmousedown: (e) => startResize(e, 'inspector'), ondblclick: () => setPanelWidth('inspector', window.innerWidth < 1500 ? 300 : 350, true) });
  els.full = h('div', { style: 'flex:1;display:flex;min-width:0;min-height:0' });
  els.status = h('footer', { class: 'status' });
  mount(app, els.top, h('div', { class: 'body' }, els.lib, els.main, els.inspHandle, els.insp, els.full), els.status);
  onBusy((n, label) => renderStatus(n, label));
  subscribe((what) => (what === 'insp' ? renderInspectorOnly() : render()));
  // Closing the window with unsaved work asks first (where the host honours beforeunload).
  window.addEventListener('beforeunload', (e) => { if (isDirty()) { e.preventDefault(); e.returnValue = ''; } });
  subscribe(() => { document.title = `${isDirty() ? '● ' : ''}SmartFitter`; });
}

let renderQueued = false;
function render() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
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
    else if (v === 'day') renderDay(els.full);
    else if (v === 'stack') renderStack(els.full);
    else if (v === 'autofind') renderAutoFind(els.full);
    renderTop();
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
const SHORTCUTS = [['Open files', 'Ctrl O'], ['Add files', 'Ctrl Shift O'], ['Search files / commands', 'Ctrl K'], ['Command palette', 'Ctrl Shift P'], ['Run fit · robust fit', 'Ctrl F · Ctrl Shift F'], ['Save/export workspace · save session', 'Ctrl S · Ctrl Shift S'],
  ['Open session', 'Ctrl Alt O'], ['Undo · redo', 'Ctrl Z · Ctrl Y'], ['Copy workspace figure/table', 'Ctrl Shift C'], ['Reset plot view', 'Ctrl Home'], ['Inspector tabs', 'Alt 1–7'], ['Next · previous library file', 'Alt Down · Alt Up'], ['Clear selection / search', 'Esc'], ['This shortcut list', '?']];

function currentWorkspaceActions() {
  const view = S.view;
  return workspaceActions(S.view, {
    analysis: !!active(), figure: !!S.figure.panels.length,
    trend: view === 'compare' && hasCompareTrend(), overlay: view === 'compare' && hasCompareOverlay(),
    day: view === 'day' && hasDaySummary(), stack: view === 'stack' && hasStackResults(),
    batch: !!S.batch.rows.length, autofind: view === 'autofind' && canExportAutoFind(),
    session: !!Object.keys(S.files).length || !!S.figure.panels.length,
  }, {
    analysis_save: () => saveResults(active()), analysis_copy: () => copyFigure(active()),
    figure_save: () => {
      const rect = els.top.querySelector('.workspace-save')?.getBoundingClientRect();
      contextMenu(rect?.left || 20, rect?.bottom || 50, ['svg', 'png', 'pdf'].map((kind) => ({
        label: `Export figure as ${kind.toUpperCase()}…`, run: () => exportFigure(kind).catch((e) => toast(e.message, 'err')),
      })));
    },
    figure_copy: () => exportFigure('copy'), trend_save: saveCompareTrend, overlay_copy: copyCompareOverlay,
    day_save: saveDaySummary, day_copy: copyDaySummary, stack_save: saveStackResults, stack_copy: copyStackResults,
    batch_save: saveBatchSummary, batch_copy: copyBatchSummary, autofind_save: exportAutoFind, session_save: saveSessionAs,
  });
}

async function performWorkspaceAction(kind) {
  const action = currentWorkspaceActions()[kind];
  if (!action?.enabled) { toast(action ? `No results available to ${action.label.toLowerCase()}.` : 'Copy is unavailable in this workspace.'); return; }
  try { await action.run(); } catch (e) { toast(e.message, 'err'); }
}

// ------------------------------------------------------------------ top bar
function renderTop() {
  const nav = [['analyze', 'Analyze'], ['compare', 'Compare'], ['figure', 'Figure'], ['day', 'Day'], ['stack', 'Stacks'], ['autofind', 'AutoFind'], ['batch', 'Batch']];
  const save = currentWorkspaceActions().save;
  mount(els.top,
    h('div', { class: 'brand', title: isDirty() ? 'Start screen · unsaved changes since the last session save (Ctrl+Shift+S)' : 'Start screen', onclick: () => { S.view = 'home'; emit(); } }, icon('logo', 20, '#5AA9F0'), h('span', { class: 'brand-name' }, 'SmartFitter'),
      isDirty() ? h('span', { class: 'dirty-dot', 'aria-label': 'Unsaved changes' }, '●') : null),
    h('nav', { class: 'nav', 'aria-label': 'Workspaces' }, nav.map(([k, l]) => h('button', { class: S.view === k || (k === 'analyze' && S.view === 'home' && active()) ? 'on' : '', onclick: () => { S.view = k; emit(); } }, l))),
    h('div', { class: 'sp' }),
    h('button', { class: 'search-btn', title: 'Search files and commands (Ctrl+K, Ctrl+Shift+P)', onclick: openPalette }, icon('search', 13), h('span', {}, 'Search'), h('span', { class: 'kbd' }, 'Ctrl K')),
    h('div', { class: 'tgroup' },
      h('button', { class: 'ibtn', title: 'Undo (Ctrl+Z)', 'aria-label': 'Undo', disabled: !canUndo(), onclick: undo }, icon('undo', 15)),
      h('button', { class: 'ibtn', title: 'Redo (Ctrl+Y)', 'aria-label': 'Redo', disabled: !canRedo(), onclick: redo }, icon('redo', 15)),
      h('button', { class: 'ibtn', title: 'Keyboard shortcuts (?)', 'aria-label': 'Keyboard shortcuts', onclick: openShortcuts }, '?')),
    h('button', { class: 'btn', title: 'Open files (Ctrl+O) · add files (Ctrl+Shift+O)', onclick: openFilesDialog }, icon('open', 14), 'Open'),
    h('button', { class: 'btn primary workspace-save', title: `${save.title} (Ctrl+S)`, disabled: !save.enabled, onclick: () => performWorkspaceAction('save') }, icon('save', 14), save.label));
}

function renderStatus(n, label) {
  const f = active();
  const busy = n > 0 ? h('span', { class: 'row busy' }, h('span', { class: 'spinner' }), label || 'Working') : null;
  mount(els.status,
    h('span', {}, `${Object.keys(S.files).length} file${Object.keys(S.files).length === 1 ? '' : 's'} open${f ? ` · ${detLabel(f, f.detector)} · ${f.mode}` : ''}`),
    S.library.watching ? h('span', { class: 'row', style: 'gap:5px;color:var(--ok)' }, h('span', { class: 'watch-dot' }), 'Watching folder') : null,
    busy,
    h('span', { class: 'sp' }),
    h('span', { title: 'Press ? for all shortcuts' }, '? shortcuts'),
    h('span', { class: 'mono' }, `v${S.version}`));
}

// ------------------------------------------------------------------ library
export async function openFolder(path, { remember = true } = {}) {
  const listing = await call('list_dir', { path }, { label: 'Reading folder' });
  const old = S.library.folder;
  if (old !== listing.path) {
    if (old) libScrolls[old] = libScroll;
    libScroll = libScrolls[listing.path] || 0;
    libSelected.clear(); libAnchor = null; libFocus = null;
  }
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
  if (lib.type && lib.type !== 'all' && info.type !== lib.type) return false;
  const words = lib.filter.toLowerCase().split(/\s+/).filter(Boolean);
  const meta = lib.listing?.files.find((r) => r.path === p);
  const date = meta ? new Date(meta.mtime * 1000).toLocaleDateString('sv-SE') : '';
  const hay = `${basename(p)} ${dirname(p)} ${info.type || ''} ${info.tag || ''} ${info.dim || ''} ${date} ${meta ? fmtDate(meta.mtime) : ''} ${(Array.isArray(info.detectors) ? info.detectors.length : info.detectors) > 1 ? 'dual detector2' : ''}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}

function fileRow(p, meta) {
  const info = S.files[p]?.info || S.library.peeks[p] || {};
  const f = S.files[p];
  const isPrimary = S.primary === p && S.view === 'analyze';
  const prim = active();
  const canOverlay = prim && prim.path !== p && kindOf(prim) === '1d' && info.dim && info.dim !== 'scan2d' && info.type !== 'TimeTrace';
  const newer = S.library.newSince[p];
  const sub = [];
  if (meta?.mtime) sub.push(fmtTime(meta.mtime));
  if (info.grid) sub.push(`${info.grid[1]} × ${info.grid[0]}`);
  else if (info.points) sub.push(`${info.points} pts`);
  if ((Array.isArray(info.detectors) ? info.detectors.length : info.detectors) > 1) sub.push('2 detectors');
  if (info.iterations && info.iterations !== '/') sub.push(`${info.iterations} it`);
  if (f && f.fit[f.detector]) sub.push(`fit R² ${fmt(f.fit[f.detector].r2, 3)}`);
  if (info.error) sub.push('unreadable');
  return h('div', {
    class: `frow ${isPrimary ? 'sel primary' : ''} ${libSelected.has(p) ? 'multi-sel' : ''} ${newer ? 'newfile' : ''}`, title: info.error ? `${p}\n${info.error}` : `${p}\nClick to open · Ctrl/Shift-click to select for comparison`, 'data-path': p, tabindex: 0,
    onfocus: () => { libFocus = p; },
    // Drag onto the plot (compare), the Compare set or the Figure composer; a multi-selection drags together.
    draggable: 'true',
    ondragstart: (e) => {
      const paths = libSelected.has(p) && libSelected.size > 1 ? [...libSelected] : [p];
      e.dataTransfer.setData(SF_PATHS, JSON.stringify(paths));
      e.dataTransfer.setData('text/plain', paths.join('\n'));
      e.dataTransfer.effectAllowed = 'copy';
    },
    onkeydown: libraryKeys,
    onclick: (e) => {
      if (e.target.closest('label,button,input')) return;
      libFocus = p;
      if (e.ctrlKey || e.metaKey || e.shiftKey) { selectLibraryFile(p, e); return; }
      libSelected.clear(); libAnchor = p; openPath(p);
    },
    oncontextmenu: (e) => { e.preventDefault(); fileMenu(e, p); },
  },
  tag(info.tag),
  h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'fname ellip' }, basename(p).replace(/\.mat$/i, '')),
    h('div', { class: 'sub ellip' }, [isPrimary ? 'Primary' : f ? 'Open' : '', S.overlays.includes(p) ? 'Compared' : '', newer ? 'New' : '', ...sub].filter(Boolean).join(' · '))),
  canOverlay ? h('label', { class: 'lib-compare', title: 'Compare on the current Analyze plot without replacing its primary file', onclick: (e) => e.stopPropagation() },
    h('input', { type: 'checkbox', checked: S.overlays.includes(p), 'aria-label': `Overlay ${basename(p)}`,
      onchange: async (e) => {
        try { if (e.target.checked) await addOverlay(p); else removeOverlay(p); }
        catch (err) { e.target.checked = false; toast(`Cannot compare: ${err.message}`, 'err'); }
      } }), 'Compare') : null);
}

function fileMenu(e, p) {
  contextMenu(e.clientX, e.clientY, [
    { label: 'Open', run: () => openPath(p) },
    { label: 'Add to Compare set', run: async () => { const f=await openFile(p,{primary:false}); addCompareItem(f); toast('Added to Compare.'); emit(); } },
    active() && active().path !== p ? { label: 'Subtract from current file (baseline)', run: async () => { await openFile(p, { primary: false }); const f = active(); const { setProc } = await import('./analyze.js'); setProc(f, { baseline: { path: p, detector: f.detector } }, 'baseline file'); } } : null,
    { label: 'Show in folder', run: () => call('reveal', { path: p }) },
    { label: 'Copy path', run: () => navigator.clipboard.writeText(p).then(() => toast('Path copied')) },
    S.files[p] ? { label: 'Close file', run: () => closeFile(p) } : null,
    Object.keys(S.files).length > 1 ? { label: 'Close all open files', run: closeAllFiles } : null,
  ]);
}

// Unsaved-work guard: run `go` directly when nothing would be lost, otherwise confirm first.
function confirmDiscard(question, go) {
  if (!isDirty()) return go();
  const m = modal(h('div', { style: 'padding:18px 20px;display:flex;flex-direction:column;gap:14px;max-width:420px' },
    h('div', {}, `${question} You have unsaved changes since the last session save; they will be lost unless you save a session (Ctrl+Shift+S).`),
    h('div', { class: 'row', style: 'justify-content:flex-end;gap:8px' },
      h('button', { class: 'btn', onclick: () => m.close() }, 'Cancel'),
      h('button', { class: 'btn', onclick: async () => { m.close(); if (await saveSessionAs()) go(); } }, 'Save session…'),
      h('button', { class: 'btn primary', onclick: () => { m.close(); go(); } }, 'Discard and continue'))));
}

function closeFile(p) {
  const f = S.files[p];
  const unsaved = f && (Object.keys(f.fit).length || Object.keys(f.iterationFits || {}).length || f.pins?.length);
  if (unsaved && isDirty()) return confirmDiscard(`Close ${f.info.name}? Its fits and analysis settings will be discarded.`, () => closeFileNow(p));
  closeFileNow(p);
}
function closeFileNow(p) {
  delete S.files[p];
  S.overlays = S.overlays.filter((q) => q !== p);
  S.compare.set = S.compare.set.filter((q) => (S.compare.items?.[q]?.path || q) !== p);
  S.library.extra = S.library.extra.filter((q) => q !== p);
  if (S.primary === p) S.primary = Object.keys(S.files)[0] || null;
  emit();
}

function closeAllFiles() {
  const n = Object.keys(S.files).length;
  if (!n) return;
  const fitted = Object.values(S.files).filter((f) => Object.keys(f.fit).length || S.compare.set.some(id=>(S.compare.items?.[id]?.path || id)===f.path && S.compare.results[id])
    || Object.keys(f.iterationFits || {}).length || Object.keys(f.iterationSensitivity || {}).length).length;
  const go = () => {
    S.files = {}; S.overlays = []; S.compare.set = []; S.compare.results = {}; S.compare.items = {}; S.compare.global = null; S.library.extra = []; S.primary = null;
    markClean(); emit(); toast(`Closed ${n} file${n === 1 ? '' : 's'}.`);
  };
  if (!fitted) return confirmDiscard(`Close all ${n} open file${n === 1 ? '' : 's'}?`, go);
  const m = modal(h('div', { style: 'padding:18px 20px;display:flex;flex-direction:column;gap:14px;max-width:420px' },
    h('div', {}, `Close all ${n} open file${n === 1 ? '' : 's'}? ${fitted} ${fitted === 1 ? 'has' : 'have'} fit results that will be discarded unless you saved them (Ctrl+S) or saved a session (Ctrl+Shift+S).`),
    h('div', { class: 'row', style: 'justify-content:flex-end;gap:8px' },
      h('button', { class: 'btn', onclick: () => m.close() }, 'Cancel'),
      h('button', { class: 'btn primary', onclick: () => { m.close(); go(); } }, 'Close all'))));
}

// Subfolder list: drag its bottom edge to resize (CSS resize: vertical); the height is remembered.
function resizableFolderList(rows) {
  const el = h('div', { class: 'lib-folders-list', style: S.library.foldersHeight ? `height:${S.library.foldersHeight}px` : '' }, rows);
  // The native resize grip sets an inline height; store it when the drag ends.
  el.addEventListener('mouseup', () => {
    const px = parseFloat(el.style.height);
    if (px > 0 && px !== S.library.foldersHeight) { S.library.foldersHeight = Math.round(px); saveLibraryPrefs(); }
  });
  return el;
}

// The library list is rebuilt on every render; remember its scroll so clicking a file doesn't jump to the top.
let libScroll = 0;
const libScrolls = {}, libSelected = new Set();
let libAnchor = null, libFocus = null;
const saveLibraryPrefs = debounce(() => {
  const { filter, chip, sort, type, foldersHeight } = S.library;
  S.settings.library_nav = { filter, chip, sort, type, foldersHeight };
  call('settings_set', { patch: { library_nav: S.settings.library_nav } }, { quiet: true }).catch(() => {});
}, 200);

function selectLibraryFile(p, e) {
  const paths = [...els.libList.querySelectorAll('.frow')].map((r) => r.dataset.path);
  if (e.shiftKey && paths.includes(libAnchor)) {
    const a = paths.indexOf(libAnchor), b = paths.indexOf(p);
    if (!e.ctrlKey && !e.metaKey) libSelected.clear();
    paths.slice(Math.min(a, b), Math.max(a, b) + 1).forEach((q) => libSelected.add(q));
  } else { libSelected.has(p) ? libSelected.delete(p) : libSelected.add(p); libAnchor = p; }
  renderLibrary();
}
function libraryKeys(e) {
  if (e.altKey || e.ctrlKey || e.metaKey || e.target.type === 'checkbox') return;
  const rows = [...els.libList.querySelectorAll('.frow')];
  const i = rows.findIndex((r) => r.dataset.path === libFocus);
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    const r = rows[Math.max(0, Math.min(rows.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)))];
    if (r) { r.focus(); r.scrollIntoView({ block: 'nearest' }); }
  } else if (e.key === 'Enter' && libFocus) openPath(libFocus);
  else if (e.key === 'Backspace' && S.library.listing?.parent) openFolder(S.library.listing.parent);
  else if (e.key === 'Escape') { libSelected.clear(); renderLibrary(); }
  else return;
  e.preventDefault(); e.stopPropagation();
}
function breadcrumbs(path) {
  const p = path.replace(/\//g, '\\');
  const root = p.match(/^(?:[A-Za-z]:\\|\\\\[^\\]+\\[^\\]+\\?)/)?.[0] || '';
  const parts = p.slice(root.length).split('\\').filter(Boolean);
  let current = root;
  return h('div', { class: 'breadcrumbs', title: path, 'aria-label': 'Folder path' },
    root ? h('button', { onclick: () => openFolder(root), title: root }, root.replace(/\\$/, '')) : null,
    parts.map((name) => {
      current = `${current.replace(/\\$/, '')}\\${name}`;
      const target = current;
      return [h('span', { class: 'dim' }, '›'), h('button', { title: target, onclick: () => openFolder(target) }, name)];
    }));
}

function renderLibrary() {
  const focused = els.lib.contains(document.activeElement) ? document.activeElement : null;
  const focusId = focused?.id, focusPath = focused?.closest('.frow')?.dataset.path;
  const cursor = focused?.tagName === 'INPUT' && focused.type === 'text' ? [focused.selectionStart, focused.selectionEnd] : null;
  const folderTop = els.lib.querySelector('.lib-folders-list')?.scrollTop || 0;
  const lib = S.library;
  if (S.libCollapsed || !['analyze', 'home'].includes(S.view)) {
    els.lib.className = 'library rail';
    mount(els.lib, h('button', { class: 'ibtn', title: 'Show library', 'aria-label': 'Show library', onclick: () => { S.libCollapsed = false; if (!['analyze', 'home'].includes(S.view)) S.view = 'analyze'; emit(); } }, icon('folder', 18)),
      Object.keys(S.files).slice(0, 14).map((p) => h('button', { class: 'ibtn', title: `${basename(p)}
Click to open · drag onto the Compare set, Figure panels or the plot`, style: S.primary === p ? 'background:var(--sel)' : '', draggable: 'true', ondragstart: (e) => { e.dataTransfer.setData(SF_PATHS, JSON.stringify([p])); e.dataTransfer.effectAllowed = 'copy'; }, onclick: () => openPath(p) }, tag((lib.peeks[p] || S.files[p]?.info || {}).tag), h('span', { class: 'rail-id' }, railId(p)))));
    return;
  }
  els.lib.className = 'library';
  const listing = lib.listing;
  const chips = [['all', 'All'], ['1d', '1D'], ['2d', '2D'], ['tt', 'Trace'], ['fitted', 'Fitted']];
  const filterEl = h('input', { class: 'field', type: 'text', id: 'lib-filter', placeholder: 'Search name, type, YYYY-MM-DD (Ctrl K)', value: lib.filter, 'aria-label': 'Filter files' });
  const refreshFilter = debounce(renderLibList, 120);
  filterEl.addEventListener('input', () => { lib.filter = filterEl.value; saveLibraryPrefs(); refreshFilter(); });
  filterEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { const first = els.libList.querySelector('.frow'); if (first) first.click(); }
    if (e.key === 'Escape') { lib.filter = ''; filterEl.value = ''; saveLibraryPrefs(); renderLibList(); filterEl.blur(); }
    if (!e.altKey && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      const rows = els.libList.querySelectorAll('.frow');
      const row = e.key === 'ArrowDown' ? rows[0] : rows[rows.length - 1];
      row?.focus(); row?.scrollIntoView({ block: 'nearest' }); e.preventDefault();
    }
    e.stopPropagation();
  });
  els.libList = h('div', { class: 'lib-list' });
  els.libList.addEventListener('scroll', () => { if (els.libList.clientHeight > 0) { libScroll = els.libList.scrollTop; if (lib.folder) libScrolls[lib.folder] = libScroll; } });
  const types = [...new Set([...Object.values(lib.peeks), ...Object.values(S.files).map((f) => f.info)].map((i) => i.type).filter(Boolean))].sort();
  if (lib.type !== 'all' && lib.type && !types.includes(lib.type)) types.push(lib.type);
  mount(els.lib,
    h('div', { class: 'lib-head' },
      h('div', { class: 'row', style: 'gap:3px' }, h('span', { class: 'lbl' }, 'Library'), h('span', { class: 'sp' }),
        h('button', { class: 'btn sm', title: 'Parent folder (Backspace when the file list has focus)', disabled: !listing?.parent, onclick: () => openFolder(listing.parent) }, icon('up', 12), 'Up'),
        h('button', { class: 'btn sm', title: 'Recent and favorite folders', onclick: folderMenu }, 'Recent'),
        h('button', { class: 'ibtn', title: 'Add files (Ctrl+Shift+O)', onclick: () => openFilesDialog(true) }, icon('plus', 15)),
        h('button', { class: 'ibtn', title: 'Choose folder', onclick: chooseFolder }, icon('folder', 15)),
        h('button', { class: 'ibtn', title: 'Collapse library', onclick: () => { S.libCollapsed = true; emit(); } }, icon('chevL', 15))),
      listing ? breadcrumbs(listing.path) : null,
      // Folder tools on one slim line (the breadcrumb above already names the folder).
      listing ? h('div', { class: 'row lib-foldertools' },
        h('span', { class: 'dim ellip', style: 'flex:1;font-size:11.5px', title: listing.path }, lib.watching ? h('span', { style: 'color:var(--ok)' }, '● watching · new files appear live') : `${listing.files.length} .mat file${listing.files.length === 1 ? '' : 's'}`),
        h('button', { class: 'ibtn', title: isFavorite(listing.path) ? 'Remove from favorite folders' : 'Add to favorite folders', 'aria-label': 'Toggle favorite folder', style: isFavorite(listing.path) ? 'color:#F5C451' : '', onclick: (e) => { e.stopPropagation(); toggleFavorite(listing.path); } }, isFavorite(listing.path) ? '★' : '☆'),
        h('button', { class: `btn sm ${lib.watching ? 'primary' : ''}`, title: 'Watch this folder for new and updated files', onclick: (e) => { e.stopPropagation(); lib.watching = !lib.watching; toast(lib.watching ? 'Watching for new files and checkpoint updates.' : 'Stopped watching.'); emit(); } }, lib.watching ? 'Watching' : 'Watch'))
        : h('button', { class: 'btn block', onclick: chooseFolder }, 'Choose a data folder'),
      filterEl,
      h('div', { class: 'g2' },
        h('select', { class: 'field', id: 'lib-type', title: 'Filter by experiment type', 'aria-label': 'Filter experiment type', onchange: (e) => { lib.type = e.target.value; saveLibraryPrefs(); renderLibList(); } },
          [['all', 'All types'], ...types.map((t) => [t, t])].map(([v, text]) => h('option', { value: v, selected: lib.type === v }, text))),
        h('select', { class: 'field', id: 'lib-sort', title: 'Sort / group files', 'aria-label': 'Sort files', onchange: (e) => { lib.sort = e.target.value; saveLibraryPrefs(); renderLibList(); } },
          [['date', 'Newest first'], ['name', 'Name A–Z'], ['type', 'By type']].map(([v, text]) => h('option', { value: v, selected: lib.sort === v }, text)))),
      h('div', { class: 'row wrap', style: 'gap:6px' }, chips.map(([k, l]) => h('button', { class: `chip sm ${lib.chip === k ? 'on' : ''}`, onclick: () => { lib.chip = k; saveLibraryPrefs(); renderLibrary(); } }, l)),
        h('span', { class: 'sp' }),
        Object.keys(S.files).length ? h('button', { class: 'btn sm', title: 'Close every open file (nothing on disk is deleted)', onclick: closeAllFiles }, `Close all (${Object.keys(S.files).length})`) : null)),
    listing?.folders.length ? h('details', { class: 'lib-folders', open: lib.foldersOpen !== false, ontoggle: (e) => { lib.foldersOpen = e.target.open; } },
      h('summary', { class: 'lbl' }, `Subfolders (${listing.folders.length})`),
      resizableFolderList(listing.folders.map((d) => h('button', { class: 'subfolder', title: d.path, onclick: () => openFolder(d.path) }, icon('folder', 13), h('span', { class: 'ellip' }, d.name))))) : null,
    // Selection bar only when something is selected (Ctrl/Shift-click hint lives in the row tooltips).
    libSelected.size ? h('div', { class: 'lib-selection row wrap' },
      h('span', { class: 'note', style: 'flex:1' }, `${libSelected.size} selected`),
      libSelected.size ? h('button', { class: 'btn sm', disabled: !active() || kindOf(active()) !== '1d', onclick: async () => {
        for (const p of libSelected) { try { await addOverlay(p); } catch (e) { toast(`Cannot compare: ${e.message}`, 'err'); } }
      } }, 'Compare selected') : null,
      h('button', { class: 'ibtn', title: 'Clear selected files', onclick: () => { libSelected.clear(); renderLibrary(); } }, '✕')) : null,
    els.libList,
    h('div', { style: 'border-top:1px solid var(--line);padding:8px 12px;display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;font-size:12px;color:var(--text2)' },
      h('button', { class: 'btn sm', title: 'Save session (Ctrl+Shift+S)', onclick: () => saveSessionAs() }, 'Save session'), h('button', { class: 'btn sm', onclick: openSessionDialog }, 'Open session'), h('span', { class: 'sp' }),
      h('label', { class: 'ck', style: 'font-size:11px', title: 'Re-run the fit automatically when a watched open file is updated' },
        h('input', { type: 'checkbox', checked: S.settings.auto_refit !== false, onchange: (e) => { S.settings.auto_refit = e.target.checked; call('settings_set', { patch: { auto_refit: e.target.checked } }, { quiet: true }).catch(() => {}); } }),
        'Refit on update')),
    h('div', { class: 'resize-handle lib-resize', title: 'Drag to resize · double-click to reset', onmousedown: (e) => startResize(e, 'library'), ondblclick: () => setPanelWidth('library', window.innerWidth < 1500 ? 280 : 340, true) }));
  applyPanelWidths();
  renderLibList();
  const crumbs = els.lib.querySelector('.breadcrumbs');
  if (crumbs) crumbs.scrollLeft = crumbs.scrollWidth;
  const folderList = els.lib.querySelector('.lib-folders-list');
  if (folderList) folderList.scrollTop = folderTop;
  const restoreFocus = focusId ? document.getElementById(focusId) : focusPath ? els.libList.querySelector(`.frow[data-path="${CSS.escape(focusPath)}"]`) : null;
  if (restoreFocus) { restoreFocus.focus({ preventScroll: true }); if (cursor) restoreFocus.setSelectionRange(...cursor); }
}

// ------------------------------------------------------------------ resizable panels
const PANEL_LIMITS = { library: [280, 600, 340], inspector: [300, 560, 350] };
function panelWidth(name) {
  const [lo, hi, def] = PANEL_LIMITS[name];
  const w = Number(S.settings.panel_widths?.[name]);
  // Unsaved widths follow the screen: laptop-size windows give the plot the room (a dragged width is always kept).
  return Number.isFinite(w) && w > 0 ? Math.min(hi, Math.max(lo, w)) : (window.innerWidth < 1500 ? lo : def);
}
// Unsaved (responsive) panel widths follow the window as it is resized, not only on the next unrelated render.
window.addEventListener('resize', debounce(() => applyPanelWidths(), 120));
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

function favoriteFolders() { return Array.isArray(S.settings.favorite_folders) ? S.settings.favorite_folders : []; }
function isFavorite(p) { return !!p && favoriteFolders().includes(p); }
function toggleFavorite(p) {
  if (!p) return;
  const favs = favoriteFolders();
  S.settings.favorite_folders = favs.includes(p) ? favs.filter((x) => x !== p) : [...favs, p];
  call('settings_set', { patch: { favorite_folders: S.settings.favorite_folders } }, { quiet: true }).catch(() => {});
  toast(isFavorite(p) ? 'Added to favorite folders.' : 'Removed from favorite folders.');
  emit();
}

function folderMenu(e) {
  const cur = S.library.listing?.path;
  const favItems = favoriteFolders().map((p) => ({ label: `★ ${p === cur ? '✓ ' : ''}${basename(p) || p}  —  ${p}`, run: () => openFolder(p) }));
  const items = (S.settings.recent_folders || []).slice(0, 8).map((p) => ({ label: `${p === cur ? '✓ ' : ''}${basename(p) || p}  —  ${p}`, run: () => openFolder(p) }));
  contextMenu(e.clientX, e.clientY, [
    ...favItems,
    ...items,
    cur ? { label: isFavorite(cur) ? '☆ Remove current folder from favorites' : '★ Add current folder to favorites', run: () => toggleFavorite(cur) } : null,
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
  const listed = new Set((listing?.files || []).map((f) => f.path));
  const extra = [...new Set([...openPaths, ...lib.extra])].filter((p) => !listed.has(p) && fileMatches(p, lib.peeks[p] || S.files[p]?.info || {}));
  if (listing) {
    const nameSort = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true });
    const files = [...listing.files].sort((a, b) => lib.sort === 'name' ? nameSort(a, b) : lib.sort === 'type'
      ? (lib.peeks[a.path]?.type || '').localeCompare(lib.peeks[b.path]?.type || '') || nameSort(a, b) : b.mtime - a.mtime || nameSort(a, b))
      .filter((f) => fileMatches(f.path, S.files[f.path]?.info || lib.peeks[f.path] || {}));
    let lastDay = '';
    out.push(h('div', { class: 'group-lbl' }, h('span', { class: 'lbl' }, `${listing.name}`), h('span', { class: 'dim', style: 'font-size:11px' }, `${files.length}/${listing.files.length}`)));
    files.forEach((f) => {
      const group = lib.sort === 'type' ? lib.peeks[f.path]?.type || 'Scanning…' : lib.sort === 'name' ? '' : fmtDate(f.mtime);
      if (group && group !== lastDay) { out.push(h('div', { class: 'lbl', style: 'padding:8px 8px 2px;font-size:10px' }, group)); lastDay = group; }
      out.push(fileRow(f.path, f));
    });
    if (!listing.files.length) out.push(h('div', { class: 'note', style: 'padding:8px' }, 'No .mat files in this folder. Open a subfolder above.'));
  }
  if (extra.length) {
    out.push(h('div', { class: 'group-lbl lbl' }, `Open elsewhere (${extra.length})`));
    extra.forEach((p) => { if (!lib.peeks[p]) requestPeeks([p]); out.push(fileRow(p, null)); });
  }
  if (!out.length) out.push(h('div', { class: 'note', style: 'padding:10px' }, 'Choose a data folder, drop .mat files on the window, or press Ctrl+O.'));
  const top = libScroll, focusPath = list.contains(document.activeElement) ? document.activeElement.closest('.frow')?.dataset.path : null;
  mount(list, out);
  list.scrollTop = top;
  if (focusPath) list.querySelector(`.frow[data-path="${CSS.escape(focusPath)}"]`)?.focus({ preventScroll: true });
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
export async function saveSession(path) { await call('save_session', { path, doc: sessionDoc() }, { label: 'Saving session' }); markClean(); toast(`Session saved: ${basename(path)}`, 'ok'); return path; }
async function saveSessionAs() {
  const f = active();
  const p = await call('dialog', { kind: 'save', title: 'Save analysis session', filename: `${f ? f.info.name.replace(/\.mat$/i, '').replace(/[<>:"/\\|?*]/g, '_') : 'analysis'}.sfsession.json`, directory: f ? dirname(f.combinedRecipe?.runs?.[0]?.path || f.path) : '' });
  if (p) return saveSession(p.endsWith('.json') ? p : `${p}.sfsession.json`);
}
async function openSessionDialog() {
  const p = await call('dialog', { kind: 'open_files', title: 'Open analysis session', filters: ['SmartFitter session (*.sfsession.json;*.json)'] });
  if (p?.length) loadSession(p[0]);
}
function loadSession(p) { confirmDiscard('Open another session?', () => loadSessionNow(p)); }
async function loadSessionNow(p) { const doc = await call('load_session', { path: p }, { label: 'Opening session' }); await applySession(doc); if (S.library.folder) openFolder(S.library.folder, { remember: false }); }

// ------------------------------------------------------------------ home
function renderHome(main) {
  const st = S.settings;
  const recentFolders = (st.recent_folders || []).slice(0, 8);
  const recentFiles = (st.recent_files || []).slice(0, 8);
  const favFolders = favoriteFolders();
  mount(main, h('div', { style: 'width:100%;max-width:1180px;margin:0 auto;padding:20px 12px;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;align-content:start' },
    h('section', { style: 'grid-column:1/-1' }, h('h1', { style: 'margin:0;font-size:26px;font-weight:600' }, 'Open data'),
      h('p', { class: 'muted', style: 'margin:6px 0 0' }, 'MATLAB .mat files from the acquisition code: snapshots, checkpoints, legacy savedData, time traces, 1D and 2D scans.')),
    h('section', { class: 'card dropzone', style: 'grid-column:1/-1' }, icon('up', 32, '#7DBBF5'), h('div', { style: 'font-size:16px' }, 'Drop .mat files or a folder anywhere on this window'),
      h('div', { class: 'row wrap', style: 'justify-content:center' }, h('button', { class: 'btn primary', onclick: () => openFilesDialog() }, 'Open files', h('span', { class: 'kbd' }, 'Ctrl O')),
        h('button', { class: 'btn', onclick: chooseFolder }, 'Open folder'), h('button', { class: 'btn', onclick: openSessionDialog }, 'Open session', h('span', { class: 'kbd' }, 'Ctrl Alt O')))),
    h('section', { class: 'card', style: 'padding:14px;display:flex;flex-direction:column;gap:2px' }, h('div', { class: 'lbl', style: 'padding:0 10px 6px' }, 'Favorite folders'),
      favFolders.length ? favFolders.map((p) => h('div', { class: 'frow', title: p, onclick: () => { openFolder(p); S.view = 'analyze'; S.libCollapsed = false; emit(); } }, h('span', { style: 'color:#F5C451' }, '★'), h('span', { class: 'ellip', style: 'flex:1' }, basename(p)), h('span', { class: 'dim ellip', style: 'font-size:11px;max-width:45%' }, dirname(p)),
        h('button', { class: 'ibtn', title: 'Remove from favorites', onclick: (e) => { e.stopPropagation(); toggleFavorite(p); } }, '✕')))
        : h('div', { class: 'note', style: 'padding:0 10px' }, 'Star a folder in the Library to pin it here.')),
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
  const actions = currentWorkspaceActions();
  return [
    ['Open files…', 'Ctrl O', () => openFilesDialog()], ['Add files…', 'Ctrl Shift O', () => openFilesDialog(true)], ['Choose data folder…', '', chooseFolder],
    ['Run fit', 'Ctrl F', () => f && runFitScoped(f)], ['Robust fit', 'Ctrl Shift F', () => f && runFitScoped(f, { robust: true })],
    [actions.save.label, 'Ctrl S', () => performWorkspaceAction('save')], ['Save session…', 'Ctrl Shift S', saveSessionAs], ['Open session…', 'Ctrl Alt O', openSessionDialog],
    ...(actions.copy ? [[actions.copy.label, 'Ctrl Shift C', () => performWorkspaceAction('copy')]] : []), ['Reset plot view', 'Ctrl Home', () => f && resetView(f)], ['Edit plot', '', () => { S.editor.open = true; emit(); }],
    ['Go to Analyze', '', () => { S.view = 'analyze'; emit(); }], ['Go to Compare', '', () => { S.view = 'compare'; emit(); }], ['Go to Figure composer', '', () => { S.view = 'figure'; emit(); }], ['Go to Batch', '', () => { S.view = 'batch'; emit(); }],
    ['Go to Day view', '', () => { S.view = 'day'; emit(); }], ['Go to Stacks', '', () => { S.view = 'stack'; emit(); }], ['Go to AutoFind', '', () => { S.view = 'autofind'; emit(); }],
    ['Add current plot to Figure', '', () => import('./figure.js').then((m) => m.addCurrentPlot())],
    ['Close all open files', '', closeAllFiles],
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
  if (ctrl && k === 's') { e.preventDefault(); performWorkspaceAction('save'); return; }
  if (ctrl && e.shiftKey && k === 'c') { e.preventDefault(); performWorkspaceAction('copy'); return; }
  if (ctrl && e.shiftKey && k === 'f') { e.preventDefault(); if (f) runFitScoped(f, { robust: true }); return; }
  if (ctrl && k === 'f') { e.preventDefault(); if (f) runFitScoped(f); return; }
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
    if (f?.tool === 'select') { f.tool = 'zoom'; emit(); }
    else if (f?.pins?.length) { f.pins = []; emit(); }
    else if (f?.measure?.length) { f.measure = []; emit(); }
    else if (S.editor.open) { S.editor.open = false; emit(); }
  }
});

// ------------------------------------------------------------------ drag & drop
let dragDepth = 0;
const dropOverlay = h('div', { class: 'drop-overlay hidden' }, 'Drop .mat files or a folder to open');
document.body.append(dropOverlay);
// Drops are routed by what is under the cursor ([data-drop] zones): the Analyze plot compares, the Compare set adds,
// the Figure panel list adds panels; anywhere else opens. Library rows drag their path(s) as SF_PATHS.
export const SF_PATHS = 'application/x-sf-paths';
const DROP_LABEL = { compare: 'Drop to compare on this plot', compareset: 'Drop to add to the Compare set', figure: 'Drop to add figure panels' };
let dropHot = null;
const dropZone = (e) => document.elementFromPoint(e.clientX, e.clientY)?.closest?.('[data-drop]') || null;
function markDrop(zone) {
  if (dropHot && dropHot !== zone) dropHot.classList.remove('drop-hot');
  dropHot = zone;
  if (zone) { zone.classList.add('drop-hot'); zone.dataset.dropLabel = DROP_LABEL[zone.dataset.drop] || 'Drop here'; }
}
const isInternal = (e) => e.dataTransfer?.types?.includes(SF_PATHS);
window.addEventListener('dragenter', (e) => { if (!isInternal(e) && e.dataTransfer?.types?.includes('Files')) { dragDepth++; dropOverlay.classList.remove('hidden'); } });
window.addEventListener('dragleave', (e) => {
  if (!isInternal(e)) { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) dropOverlay.classList.add('hidden'); }
  if (!e.relatedTarget) markDrop(null);
});
window.addEventListener('dragover', (e) => {
  if (e.dataTransfer?.types?.includes('application/x-sf-series')) return; // Series rows reorder themselves
  e.preventDefault();
  const zone = dropZone(e);
  markDrop(zone);
  if (!isInternal(e)) dropOverlay.textContent = zone ? DROP_LABEL[zone.dataset.drop] : 'Drop .mat files or a folder to open';
});
window.addEventListener('dragend', () => { markDrop(null); dragDepth = 0; dropOverlay.classList.add('hidden'); });
window.addEventListener('drop', async (e) => {
  if (e.dataTransfer?.types?.includes('application/x-sf-series')) return;
  e.preventDefault();
  const zone = dropZone(e)?.dataset.drop || null;
  markDrop(null);
  dragDepth = 0; dropOverlay.classList.add('hidden');
  const paths = [];
  if (isInternal(e)) {
    try { paths.push(...JSON.parse(e.dataTransfer.getData(SF_PATHS))); } catch (err) { /* malformed drag */ }
  } else {
    const dropped = [...(e.dataTransfer?.files || [])];
    // Desktop window: real paths come from the Python-side drop listener (Api.drop_paths), not temporary copies.
    const real = S.desktop && dropped.length ? await call('drop_paths', { names: dropped.map((x) => x.name) }, { quiet: true }).catch(() => []) : [];
    for (const [k, file] of dropped.entries()) {
      const full = real?.[k] || file.pywebviewFullPath || file.path;
      if (full) paths.push(full);
      else if (/\.mat$/i.test(file.name)) {
        const data = await new Promise((res) => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(file); });
        paths.push(await call('save_upload', { name: file.name, data }, { label: 'Receiving file' }));
      }
    }
  }
  if (!paths.length) return;
  // A folder is anything that is not a data/session file (folder names may contain dots).
  const isData = (p) => /\.(mat|json)$/i.test(p);
  for (const p of paths.filter((q) => !isData(q))) {
    try { await openFolder(p); S.view = 'analyze'; } catch (err) { toast(`Cannot open ${basename(p)}: ${err.message}`, 'err'); }
  }
  const files = paths.filter(isData);
  if (!files.length) return emit();
  try {
    if (zone === 'compare' && active()) {
      for (const p of files) await addOverlay(p);
      toast(`Comparing ${files.length} file${files.length === 1 ? '' : 's'} on this plot.`, 'ok');
    } else if (zone === 'compareset') {
      for (const p of files) { const f=await openFile(p,{primary:false}); addCompareItem(f); }
    } else if (zone === 'figure') {
      const prev = S.primary, { addCurrentPlot } = await import('./figure.js');
      try { for (const p of files) { await openFile(p, { primary: false }); S.primary = p; await addCurrentPlot({ quiet: true, overlays: [] }); } }
      finally { S.primary = prev; }
      toast(`Added ${files.length} figure panel${files.length === 1 ? '' : 's'}.`, 'ok');
    } else {
      for (const [i, p] of files.entries()) {
        if (i === 0) await openPath(p);
        else { await openFile(p, { primary: false }); if (!S.library.extra.includes(p)) S.library.extra.push(p); requestPeeks([p]); }
      }
    }
  } catch (err) { /* toast shown by api */ }
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

// Short visible identity for rail badges: the distinguishing tail of the file name (e.g. a time stamp), since
// several open files of one type would otherwise be identical badges.
function railId(p) {
  const name = basename(p).replace(/\.mat$/i, '').replace(/__?checkpoint\d*$/i, '·cp');
  const parts = name.split(/[_\s-]+/).filter(Boolean);
  const tail = parts[parts.length - 1] || name;
  return (tail.length > 7 ? tail.slice(-7) : tail);
}
