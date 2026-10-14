// Quick 1D comparison picker shared by Analyze and the navigator.
import { call } from './api.js';
import { S, emit, addOverlay, removeOverlay, commit } from './state.js';
import { h, mount, modal, basename, dirname, toast } from './util.js';

export async function browseOverlays(f) {
  const paths = await call('dialog', { kind: 'open_files', title: 'Compare against data files', directory: S.library.folder || dirname(f.path) });
  if (paths?.length) commit('add comparisons');
  for (const p of paths || []) {
    try { await addOverlay(p, f); } catch (e) { toast(`Cannot compare: ${e.message}`, 'err'); }
  }
  if (paths?.length) {
    const infos = await call('peek', { paths }, { quiet: true });
    infos.forEach((i) => { S.library.peeks[i.path] = i; }); emit();
  }
}

export async function openOverlayPicker(f) {
  const paths = [...new Set([...Object.keys(S.files), ...S.library.extra, ...(S.library.listing?.files || []).map((r) => r.path)])].filter((p) => p !== f.path);
  const missing = paths.filter((p) => !S.library.peeks[p] && !S.files[p]);
  if (missing.length) {
    const infos = await call('peek', { paths: missing }, { label: 'Reading compare candidates' });
    infos.forEach((i) => { S.library.peeks[i.path] = i; });
  }
  let allTypes = false, busy = false;
  const search = h('input', { class: 'field', placeholder: 'Search name, folder or experiment type', 'aria-label': 'Search comparison files' });
  const list = h('div', { class: 'compare-picker-list' });
  const count = h('span', { class: 'note' });
  const candidates = () => [...new Set([...paths, ...S.overlays, ...Object.keys(S.files)])].filter((p) => p !== f.path).map((p) => ({ p, info: S.files[p]?.info || S.library.peeks[p] || {} }));
  const draw = () => {
    const q = search.value.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = candidates().filter(({ p, info }) => info.dim !== 'scan2d' && info.type !== 'TimeTrace' && !info.error &&
      (allTypes || info.type === f.info.type) && q.every((w) => `${p} ${info.type || ''}`.toLowerCase().includes(w)))
      .sort((a, b) => basename(a.p).localeCompare(basename(b.p), undefined, { numeric: true }));
    const top = list.scrollTop;
    mount(list, rows.length ? rows.map(({ p, info }) => h('label', { class: 'compare-candidate', title: p },
      h('input', { type: 'checkbox', checked: S.overlays.includes(p), disabled: busy, 'aria-label': `Compare ${basename(p)}`,
        onchange: async (e) => {
          commit('comparison selection'); busy = true;
          list.querySelectorAll('input').forEach((el) => { el.disabled = true; });
          try { if (e.target.checked) await addOverlay(p, f); else removeOverlay(p); }
          catch (err) { toast(`Cannot compare: ${err.message}`, 'err'); }
          finally { busy = false; draw(); }
        } }),
      h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'ellip' }, basename(p)), h('div', { class: 'note ellip' }, `${info.type || 'Unknown'} · ${dirname(p)}`)),
      S.files[p] ? h('span', { class: 'note' }, 'Open') : null)) : h('div', { class: 'note', style: 'padding:12px' }, 'No matching 1D files. Show all experiment types or Browse…'));
    list.scrollTop = top;
    count.textContent = `${rows.length} matching · ${S.overlays.filter((p) => p !== f.path).length} compared`;
  };
  const m = modal(h('div', { class: 'compare-picker' },
    h('div', { class: 'row' }, h('strong', {}, 'Compare with…'), h('span', { class: 'sp' }), h('button', { class: 'btn', onclick: () => m.close() }, 'Done')),
    h('div', { class: 'note' }, `Overlays on the current Analyze plot. Default: ${f.info.type} files. Selecting a file does not replace the primary file.`),
    search,
    h('div', { class: 'row wrap' }, h('label', { class: 'ck' }, h('input', { type: 'checkbox', onchange: (e) => { allTypes = e.target.checked; draw(); } }), 'Show all experiment types'), h('span', { class: 'sp' }), count),
    list,
    h('div', { class: 'row' }, h('button', { class: 'btn', onclick: async () => { await browseOverlays(f); draw(); } }, 'Browse…'),
      h('button', { class: 'btn', onclick: () => { commit('clear comparisons'); S.overlays = []; emit(); draw(); } }, 'Clear all'))));
  search.addEventListener('input', draw);
  draw(); search.focus();
}
