// Small DOM + formatting helpers shared by every view.

// Views pass optional children (cond ? node : null) and arrays to append(); make the
// native Element.append skip null/false and flatten arrays instead of printing "null".
const nativeAppend = Element.prototype.append;
Element.prototype.append = function append(...nodes) {
  nativeAppend.apply(this, nodes.flat(Infinity).filter((n) => n !== null && n !== undefined && n !== false));
};

export function h(tag, props = {}, ...children) {
  const el = tag === 'frag' ? document.createDocumentFragment() : document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'style') el.style.cssText = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'value') el.value = v;
    else if (k === 'checked') el.checked = !!v;
    else if (k === 'disabled') el.disabled = !!v;
    else if (k === 'selected') el.selected = !!v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function mount(target, ...children) {
  target.replaceChildren();
  append(target, children);
}

export const $ = (sel, root = document) => root.querySelector(sel);

// SVG icons (inline stroke icons, currentColor)
const ICONS = {
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
  folder: '<path d="M3 7h6l2 2h10v10H3z"/>',
  open: '<path d="M3 7h6l2 2h10v10H3z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 010 12h-3"/>',
  redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H10a6 6 0 000 12h3"/>',
  home: '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/>',
  zoom: '<circle cx="10" cy="10" r="6"/><path d="M20 20l-5.5-5.5M10 7v6M7 10h6"/>',
  pan: '<path d="M12 2v20M2 12h20M12 2l-3 3M12 2l3 3M12 22l-3-3M12 22l3-3M2 12l3-3M2 12l3 3M22 12l-3-3M22 12l-3 3"/>',
  measure: '<path d="M3 17L17 3l4 4L7 21z"/><path d="M7 13l2 2M10 10l2 2M13 7l2 2"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 00-1-1H5a1 1 0 00-1 1v10a1 1 0 001 1h3"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M14 6l4 4"/>',
  up: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  chevL: '<path d="M15 18l-6-6 6-6"/>',
  chevR: '<path d="M9 18l6-6-6-6"/>',
  eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  refresh: '<path d="M20 11a8 8 0 10-2.3 5.7"/><path d="M20 4v7h-7"/>',
  save: '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v6h8V3M8 21v-7h8v7"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
  logo: '<rect x="2.5" y="2.5" width="19" height="19" rx="5"/><path d="M5.5 14c1.5-6 3-6 4.5 0s3 6 4.5 0 3-6 4-1"/>',
  drag: '<path d="M9 5h.01M15 5h.01M9 12h.01M15 12h.01M9 19h.01M15 19h.01"/>',
};
export function icon(name, size = 16, stroke = 'currentColor') {
  const span = document.createElement('span');
  span.style.display = 'inline-flex';
  span.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ''}</svg>`;
  return span;
}

// ---- formatting ----
export function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (typeof v !== 'number') return String(v);
  if (!Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(Math.max(1, digits - 2)).replace('e', 'e');
  return Number(v.toPrecision(digits)).toString();
}
export function fmtFixed(v, d = 2) { return v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(d); }
// Value with uncertainty, rounded to the uncertainty's first two significant digits.
export function fmtPm(v, e) {
  if (v === null || v === undefined) return '–';
  if (!e || !Number.isFinite(e) || e <= 0) return fmt(v, 5);
  const p = Math.floor(Math.log10(e)) - 1;
  if (Math.abs(v) >= 1e6 || Math.abs(v) < 1e-3 || p < -8) return `${v.toExponential(3)} ± ${e.toExponential(1)}`;
  const d = Math.max(0, -p);
  return `${v.toFixed(d)} ± ${e.toFixed(d)}`;
}
export function fmtCounts(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(1) + 'k';
  return Number(v.toPrecision(4)).toString();
}
export function fmtTime(sec) {
  const d = new Date(sec * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
export function fmtDate(sec) {
  const d = new Date(sec * 1000);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}
export const basename = (p) => (p || '').split(/[\\/]/).pop();
export const dirname = (p) => { const parts = (p || '').split(/[\\/]/); parts.pop(); return parts.join('\\'); };
export const stem = (p) => basename(p).replace(/\.[^.]+$/, '');
export const clone = (o) => (o === undefined ? undefined : JSON.parse(JSON.stringify(o)));
export const debounce = (fn, ms = 200) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export const num = (v) => { if (v === '' || v === null || v === undefined) return null; const n = Number(String(v).replace(/[^\d.eE+\-−]/g, '').replace('−', '-')); return Number.isFinite(n) ? n : null; };

// ---- feedback ----
export function toast(msg, kind = '', ms = 3800) {
  let box = document.querySelector('.toasts');
  if (!box) { box = h('div', { class: 'toasts' }); document.body.appendChild(box); }
  const t = h('div', { class: `toast ${kind}` }, msg);
  box.appendChild(t);
  setTimeout(() => t.remove(), kind === 'err' ? Math.max(ms, 8000) : ms);
}

export function modal(content, { onClose, className = '' } = {}) {
  const back = h('div', { class: 'modal-back' });
  const box = h('div', { class: `modal ${className}` });
  box.append(content);
  back.append(box);
  const close = () => { back.remove(); document.removeEventListener('keydown', esc, true); onClose && onClose(); };
  const esc = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } };
  back.addEventListener('mousedown', (e) => { if (e.target === back) close(); });
  document.addEventListener('keydown', esc, true);
  document.body.append(back);
  return { close, box };
}

export function contextMenu(x, y, items) {
  document.querySelectorAll('.ctx').forEach((m) => m.remove());
  const menu = h('div', { class: 'ctx', style: { left: `${x}px`, top: `${y}px` } },
    items.filter(Boolean).map((it) => h('button', { onclick: () => { menu.remove(); it.run(); } }, h('span', {}, it.label), it.kbd ? h('span', { class: 'kbd' }, it.kbd) : null)));
  document.body.append(menu);
  const r = menu.getBoundingClientRect();
  if (r.right > innerWidth) menu.style.left = `${innerWidth - r.width - 8}px`;
  if (r.bottom > innerHeight) menu.style.top = `${innerHeight - r.height - 8}px`;
  setTimeout(() => document.addEventListener('mousedown', function off(e) { if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener('mousedown', off); } }), 0);
}

// segmented control
export function seg(options, value, onChange, { disabled = [] } = {}) {
  return h('div', { class: 'seg', role: 'group' }, options.map((o) => {
    const [val, label, title] = Array.isArray(o) ? o : [o, o];
    return h('button', { class: val === value ? 'on' : '', title, disabled: disabled.includes(val), onclick: () => onChange(val) }, label);
  }));
}

export function select(options, value, onChange, props = {}) {
  return h('select', { class: 'field', ...props, onchange: (e) => onChange(e.target.value) },
    options.map((o) => { const [v, l] = Array.isArray(o) ? o : [o, o]; return h('option', { value: v, selected: String(v) === String(value) }, l); }));
}

// text/number input committing on Enter or blur (keeps focus stable during re-renders)
export function input(value, onCommit, props = {}) {
  const el = h('input', { class: 'field', value: value ?? '', ...props });
  const commit = () => { if (el.value !== String(value ?? '')) onCommit(el.value); };
  el.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); commit(); el.blur(); } e.stopPropagation(); });
  el.addEventListener('blur', commit);
  return el;
}

export function check(label, value, onChange, props = {}) {
  return h('label', { class: 'ck', title: props.title }, h('input', { type: 'checkbox', checked: value, disabled: props.disabled, onchange: (e) => onChange(e.target.checked) }), label);
}

export function field(label, control, props = {}) {
  return h('label', { class: 'f', ...props }, h('span', { class: 'lbl' }, label), control);
}

export const TAG_COLORS = {
  RABI: ['#3A2A18', '#FFC78A'], ODMR: ['#182A3C', '#8EC5FF'], RAMSEY: ['#26213A', '#C4B5FF'], ECHO: ['#1F2E22', '#9FE0AC'],
  DD: ['#1F2E22', '#9FE0AC'], T1: ['#2F2A1A', '#E8D38A'], DEER: ['#2A1E2E', '#E3A6F0'], Z: ['#1E3330', '#8FE3CF'],
  XY: ['#2A2240', '#C9B6FF'], TT: ['#33261E', '#F2B48C'], '?': ['#3A1B1B', '#FF8A80'],
};
export function tag(t) {
  const [bg, fg] = TAG_COLORS[t] || ['#2A2F3A', '#C9D1DD'];
  return h('span', { class: 'tag', style: { background: bg, color: fg } }, t || '…');
}

export function download(name, dataUrlOrText, mime = 'text/plain') {
  const a = document.createElement('a');
  a.href = dataUrlOrText.startsWith('data:') ? dataUrlOrText : URL.createObjectURL(new Blob([dataUrlOrText], { type: mime }));
  a.download = name;
  a.click();
}

export function linreg(xs, ys) {
  const n = xs.length; if (n < 2) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const slope = sxx ? sxy / sxx : 0; const icpt = my - slope * mx;
  let ss = 0; for (let i = 0; i < n; i++) ss += (ys[i] - (icpt + slope * xs[i])) ** 2;
  const se = n > 2 && sxx ? Math.sqrt(ss / (n - 2) / sxx) : null;
  return { slope, intercept: icpt, slope_err: se };
}
