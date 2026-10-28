import { finiteExtent } from './numeric.js';
// 2D map figures: per-detector maps with colorbars, RGB overlay panel, spot shapes, cursor.
import { detLabel } from './state.js';
import { baseStyle, baseLayout, friendlyAxis } from './plotstyle.js';

export const CMAPS = {
  viridis: 'Viridis', cividis: 'Cividis', magma: 'Magma', inferno: 'Inferno', plasma: 'Plasma', turbo: 'Turbo', RdBu: 'RdBu', gray: 'Greys',
  parula: [[0, '#352a87'], [0.111, '#1265a8'], [0.222, '#148ab4'], [0.333, '#12a7a2'], [0.444, '#3fbc73'], [0.556, '#86c94b'], [0.667, '#c6d63c'], [0.778, '#f1df3a'], [0.889, '#f9c932'], [1, '#f9a721']],
  coolwarm: [[0, '#3b4cc0'], [0.25, '#8db0fe'], [0.5, '#dddddd'], [0.75, '#f49a7b'], [1, '#b40426']],
};
export const CMAP_GRADIENT = {
  viridis: '#440154,#3b528b,#21918c,#5ec962,#fde725', magma: '#000004,#51127c,#b73779,#fc8961,#fcfdbf', cividis: '#00224e,#434e6c,#7c7b78,#bcaf6f,#fee838',
  inferno: '#000004,#56106e,#bb3754,#f98e09,#fcffa4', plasma: '#0d0887,#7e03a8,#cc4778,#f89540,#f0f921', turbo: '#30123b,#4686fb,#1ae4b6,#faba39,#7a0403',
  parula: '#352a87,#148ab4,#3fbc73,#c6d63c,#f9a721', coolwarm: '#3b4cc0,#8db0fe,#dddddd,#f49a7b,#b40426', RdBu: '#67001f,#f4a582,#f7f7f7,#92c5de,#053061', gray: '#000000,#ffffff',
};
// Plotly's Greys runs white->black; flip so "gray" means black->white like Matplotlib.
const NEEDS_FLIP = { gray: true, RdBu: true };

function percentile(values, p) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const idx = (p / 100) * (v.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

// Colorbar/hover label for what the map actually shows: raw signal/reference sources are recorded counts; the
// 'value' source is the selected observable (contrast is dimensionless, not counts/s).
export function mapValueLabel(f) {
  const src = f.map?.source || 'value', unit = f.info.unit || '';
  if (src === 'signal' || src === 'reference') return `${src === 'signal' ? 'Signal' : 'Reference'}${unit ? ` [${unit}]` : ''}`;
  if (f.mode === 'contrast') return 'Contrast [dimensionless]';
  const name = { signal: 'Signal', reference: 'Reference', difference: 'Reference − signal' }[f.mode] || f.data[f.detector]?.map?.label || 'Value';
  return `${name}${unit ? ` [${unit}]` : ''}`;
}

// Display arrays honoring axis swap. Returns {x, y, z, xl, yl}
export function displayMap(m, swap, source = 'value') {
  const z0 = m.z[source] || m.z.value;
  const z = z0.map((r) => r.map((v) => (v === null ? NaN : v)));
  if (!swap) return { x: m.x, y: m.y, z, xl: m.x_label, yl: m.y_label };
  const zt = m.x.map((_, c) => m.y.map((__, r) => z[r][c]));
  return { x: m.y, y: m.x, z: zt, xl: m.y_label, yl: m.x_label };
}

export function mapLimits(f, det, z) {
  const M = f.map;
  const flat = z.flat();
  if (M.limits === 'manual' && M.manual[det] && M.manual[det][0] !== null) return M.manual[det];
  if (M.limits === 'auto') return finiteExtent(flat, [0, 1]);
  return [percentile(flat, M.plo), percentile(flat, M.phi)];
}

function ellipsePath(s, swap) {
  const a = s.fwhm_major / 2, b = s.fwhm_minor / 2, th = (s.angle_deg * Math.PI) / 180;
  const pts = [];
  for (let k = 0; k <= 48; k++) {
    const t = (2 * Math.PI * k) / 48;
    const px = s.center_x + a * Math.cos(t) * Math.cos(th) - b * Math.sin(t) * Math.sin(th);
    const py = s.center_y + a * Math.cos(t) * Math.sin(th) + b * Math.sin(t) * Math.cos(th);
    pts.push(swap ? [py, px] : [px, py]);
  }
  return 'M' + pts.map((p) => `${p[0]},${p[1]}`).join('L') + 'Z';
}

export function visibleSpots(f, det) {
  const res = f.spots[det];
  if (!res) return [];
  const removed = new Set(f.spotRemoved[det] || []);
  return res.spots.filter((s) => !removed.has(s.spot_id));
}

// panels: [{kind:'map', det}, {kind:'overlay'}]
export function mapPanels(f) {
  const lay = f.map.layout;
  const dets = f.info.detectors;
  if (lay === 'overlay') return [{ kind: 'overlay' }];
  if (lay === 'maps+overlay') return [...dets.map((d) => ({ kind: 'map', det: d })), { kind: 'overlay' }];
  if (lay === 'side' && dets.length > 1) return dets.map((d) => ({ kind: 'map', det: d }));
  return [{ kind: 'map', det: f.detector }];
}

export function figureMaps(f, overlay) {
  const st = baseStyle(f.style);
  const M = f.map;
  const panels = mapPanels(f);
  const n = panels.length;
  const gap = 0.06;
  const w = (1 - gap * (n - 1)) / n;
  const traces = [], shapes = [], annotations = [], images = [];
  const layout = baseLayout(st, { showlegend: false, margin: { l: 64, r: 16, t: 34, b: 52 }, uirevision: f.path + M.layout + M.swap, separators: '.' });
  let shared = null;
  if (M.scale === 'shared') {
    const all = f.info.detectors.flatMap((d) => (f.data[d]?.map ? displayMap(f.data[d].map, M.swap, M.source).z.flat() : []));
    shared = M.limits === 'manual' && M.manual.shared ? M.manual.shared : [percentile(all, M.plo), percentile(all, M.phi)];
  }
  panels.forEach((p, i) => {
    const ax = i === 0 ? '' : String(i + 1);
    const x0 = i * (w + gap);
    const dom = p.kind === 'map' ? [x0, x0 + w * 0.84] : [x0 + (i ? w * 0.08 : 0), x0 + w];
    let disp, title;
    if (p.kind === 'map') {
      const m = f.data[p.det]?.map;
      if (!m) return;
      disp = displayMap(m, M.swap, M.source || 'value');
      let z = disp.z;
      const [lo, hi] = shared || mapLimits(f, p.det, z);
      let zmin = lo, zmax = hi, tickvals, ticktext;
      if (M.log) {
        z = z.map((r) => r.map((v) => (v > 0 ? Math.log10(v) : NaN)));
        zmin = Math.log10(Math.max(lo, 1e-12)); zmax = Math.log10(Math.max(hi, 1e-12));
        tickvals = []; ticktext = [];
        for (let e = Math.ceil(zmin); e <= Math.floor(zmax); e++) { tickvals.push(e); ticktext.push(`1e${e}`); }
      }
      const cs = CMAPS[M.cmap] || 'Viridis';
      title = detLabel(f, p.det);
      traces.push({
        type: 'heatmap', x: disp.x, y: disp.y, z, zmin, zmax, colorscale: cs, reversescale: !!(M.reverse ^ !!NEEDS_FLIP[M.cmap]),
        xaxis: `x${ax}`, yaxis: `y${ax}`, zsmooth: false, hoverongaps: false,
        colorbar: { x: dom[1] + 0.012, len: 0.9, thickness: 12, outlinewidth: 1, outlinecolor: '#2A2F3A', tickfont: { size: Math.round(st.base * 0.75) },
          title: { text: mapValueLabel(f), side: 'top', font: { size: Math.round(st.base * 0.72) } }, tickvals, ticktext, exponentformat: 'SI' },
        hovertemplate: `${disp.xl} %{x:.6g}<br>${disp.yl} %{y:.6g}<br>%{z:.6g} · ${mapValueLabel(f)}<extra>${title}</extra>`, meta: { det: p.det },
      });
      if (M.showShapes || M.showCenters || M.showNumbers) {
        for (const s of visibleSpots(f, p.det)) {
          const sel = f.spotSel.includes(`${p.det}:${s.spot_id}`);
          const hov = f.spotHover === `${p.det}:${s.spot_id}`;
          const color = sel ? '#E0A400' : '#FFFFFF';
          if (M.showShapes) shapes.push({ type: 'path', path: ellipsePath(s, M.swap), xref: `x${ax}`, yref: `y${ax}`, line: { color: hov ? '#FFD400' : color, width: hov ? 4 : sel ? 2.2 : 1.4 } });
          const [cx, cy] = M.swap ? [s.center_y, s.center_x] : [s.center_x, s.center_y];
          if (M.showCenters) traces.push({ type: 'scatter', mode: 'markers', x: [cx], y: [cy], xaxis: `x${ax}`, yaxis: `y${ax}`, marker: { symbol: 'cross-thin', size: 12, line: { color, width: 2 } }, hoverinfo: 'skip', showlegend: false });
          if (M.showNumbers) annotations.push({ x: cx, y: cy, xref: `x${ax}`, yref: `y${ax}`, text: String(s.spot_id), showarrow: false, xshift: 10, yshift: 10, font: { color, size: 12 } });
        }
      }
    } else {
      if (!overlay) { annotations.push({ xref: 'paper', yref: 'paper', x: (dom[0] + dom[1]) / 2, y: 0.5, text: 'Choose a second map in Compare', showarrow: false }); return; }
      const swap = M.swap;
      const xe = swap ? overlay.y_edges : overlay.x_edges, ye = swap ? overlay.x_edges : overlay.y_edges;
      disp = { x: [xe[0], xe[xe.length - 1]], y: [ye[0], ye[ye.length - 1]], xl: swap ? overlay.y_label : overlay.x_label, yl: swap ? overlay.x_label : overlay.y_label };
      images.push({ source: overlay.png, xref: `x${ax}`, yref: `y${ax}`, x: xe[0], y: ye[ye.length - 1], sizex: xe[xe.length - 1] - xe[0], sizey: ye[ye.length - 1] - ye[0], sizing: 'stretch', layer: 'below' });
      traces.push({ type: 'scatter', x: [xe[0], xe[xe.length - 1]], y: [ye[0], ye[ye.length - 1]], mode: 'markers', marker: { opacity: 0 }, xaxis: `x${ax}`, yaxis: `y${ax}`, hoverinfo: 'skip', showlegend: false });
      const [c1, c2] = { 'Pink / teal': ['#FF0099', '#00CC99'], 'Green / magenta': ['#00C000', '#E000E0'], 'Red / cyan': ['#E00000', '#00C8C8'] }[M.palette] || ['#FF0099', '#00CC99'];
      const others = M.compareWith && M.compareWith.path !== f.path ? 'other file' : detLabel(f, f.info.detectors.find((d) => d !== f.detector) || f.detector);
      title = `<span style="color:${c1}">■</span> ${detLabel(f, f.detector)}  <span style="color:${c2}">■</span> ${others}`;
      annotations.push({ xref: `x${ax} domain`, yref: `y${ax} domain`, x: 0.5, y: -0.22, yanchor: "top", showarrow: false, font: { size: Math.round(st.base * 0.72), color: "#4A5262" }, text: `${M.overlayScale} scale · ${M.background}% background removed` });
    }
    const xs = disp.x, ys = disp.y;
    const dx = xs.length > 1 ? Math.abs(xs[1] - xs[0]) : 1, dy = ys.length > 1 ? Math.abs(ys[1] - ys[0]) : 1;
    // Never stretch: equal data units (physical) or equal pixel sizes (square).
    const ratio = M.geometry === 'square' ? dx / dy : 1;
    // Zoom linking goes through M.bounds (set from any panel's relayout), not Plotly `matches`: matching axes merges
    // every panel into one equal-scale constraint group, and a one-axis zoom then squeezes a panel's domain to a sliver.
    // Each panel keeps its own scaleanchor. Unlinked panels: the stored view applies to the first panel only.
    const own = M.bounds && (i === 0 || M.link !== false) ? M.bounds : null;
    const bounds = own || (p.kind === 'overlay' ? { x: [xs[0], xs[xs.length - 1]], y: [ys[0], ys[ys.length - 1]] } : {});
    const okRange = (r) => Array.isArray(r) && r.length === 2 && r.every(Number.isFinite) && r[0] !== r[1];
    layout[`xaxis${ax}`] = { domain: dom, anchor: `y${ax}`, title: { text: friendlyAxis(disp.xl), standoff: 6 }, showline: true, mirror: true, linecolor: '#2A2F3A', ticks: 'outside', tickangle: 0, nticks: 4, showgrid: false, zeroline: false, constrain: 'domain',
      range: okRange(bounds.x) ? bounds.x : undefined, autorange: !okRange(bounds.x) };
    // explicit full domain: otherwise uirevision keeps a domain the equal-scale constraint shrank on an earlier view
    layout[`yaxis${ax}`] = { domain: [0, 1], anchor: `x${ax}`, scaleanchor: `x${ax}`, scaleratio: ratio, title: { text: i === 0 ? friendlyAxis(disp.yl) : '', standoff: 6 }, showline: true, mirror: true, linecolor: '#2A2F3A', ticks: 'outside', showgrid: false, zeroline: false, constrain: 'domain',
      range: okRange(bounds.y) ? bounds.y : undefined, autorange: !okRange(bounds.y),
      // linked panels share y values: label them once (otherwise panel 2's ticks collide with panel 1's colorbar)
      ...(i > 0 && M.link !== false && panels[0].kind === p.kind ? { showticklabels: false } : {}) };
    annotations.push({ xref: `x${ax} domain`, yref: `y${ax} domain`, x: 0.5, y: 1.02, yanchor: 'bottom', text: `<b>${title}</b>`, showarrow: false, font: { size: Math.round(st.base * 0.9) } });
    if (M.cursor && p.kind === 'map') {
      const [cx, cy] = M.cursor;
      if (M.linecut === 'h' || M.linecut === 'cross') shapes.push({ type: 'line', xref: `x${ax}`, yref: `y${ax}`, x0: xs[0], x1: xs[xs.length - 1], y0: cy, y1: cy, line: { color: '#E0A400', width: 1.2, dash: 'dot' } });
      if (M.linecut === 'v' || M.linecut === 'cross') shapes.push({ type: 'line', xref: `x${ax}`, yref: `y${ax}`, x0: cx, x1: cx, y0: ys[0], y1: ys[ys.length - 1], line: { color: '#E0A400', width: 1.2, dash: 'dot' } });
      traces.push({ type: 'scatter', mode: 'markers', x: [cx], y: [cy], xaxis: `x${ax}`, yaxis: `y${ax}`, marker: { symbol: 'circle-open', size: 12, line: { color: '#E0A400', width: 2 } }, hoverinfo: 'skip', showlegend: false });
    }
    if (M.linecut === 'line' && M.line?.length) {
      if (M.line.length >= 2) shapes.push({ type: 'line', xref: `x${ax}`, yref: `y${ax}`, x0: M.line[0][0], y0: M.line[0][1], x1: M.line[1][0], y1: M.line[1][1], line: { color: '#E0A400', width: 1.6, dash: 'dot' } });
      traces.push({ type: 'scatter', mode: 'markers', x: M.line.map((q) => q[0]), y: M.line.map((q) => q[1]), xaxis: `x${ax}`, yaxis: `y${ax}`, marker: { symbol: 'circle-open', size: 10, line: { color: '#E0A400', width: 2 } }, hoverinfo: 'skip', showlegend: false });
    }
  });
  layout.shapes = shapes; layout.annotations = annotations; layout.images = images;
  return { data: traces, layout, panels };
}

// Linecut data through the cursor for every displayed detector.
export function linecuts(f) {
  const M = f.map;
  if (M.linecut === 'line') return lineProfiles(f);
  if (!M.cursor || M.linecut === 'none') return null;
  const out = [];
  for (const det of f.info.detectors) {
    const m = f.data[det]?.map;
    if (!m) continue;
    const d = displayMap(m, M.swap, M.cutSource || M.source || 'value');
    const ix = nearest(d.x, M.cursor[0]), iy = nearest(d.y, M.cursor[1]);
    const h = M.linecut !== 'v' ? { x: d.x, y: d.z[iy], label: `${friendlyAxis(d.xl)} at ${friendlyAxis(d.yl).split(' [')[0]} = ${fmtc(d.y[iy])}` } : null;
    const v = M.linecut === 'v' || M.linecut === 'cross' ? { x: d.y, y: d.z.map((r) => r[ix]), label: `${friendlyAxis(d.yl)} at ${friendlyAxis(d.xl).split(' [')[0]} = ${fmtc(d.x[ix])}` } : null;
    out.push({ det, label: detLabel(f, det), h, v, value: d.z[iy]?.[ix], at: [d.x[ix], d.y[iy]] });
  }
  return out;
}
// Length of the display-coordinate segment, in axis units (no rescaling between x and y).
export const lineLength = (line) => Math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]);

// Arbitrary-line profile: N = 2*max(nx, ny) samples along the segment; each sample takes the measured value of
// the cell containing it (nearest cell centre). No interpolation; samples outside the map extent are dropped.
function lineProfiles(f) {
  const M = f.map;
  if (!M.line || M.line.length < 2) return null;
  const [[x0, y0], [x1, y1]] = M.line;
  const len = lineLength(M.line);
  const out = [];
  for (const det of f.info.detectors) {
    const m = f.data[det]?.map;
    if (!m) continue;
    const d = displayMap(m, M.swap, M.cutSource || M.source || 'value');
    const N = 2 * Math.max(d.x.length, d.y.length);
    // Map extent = cell centres +/- half a cell at each edge.
    const ext = (a) => { const [lo, hi] = finiteExtent(a), hw = a.length > 1 ? (hi - lo) / (a.length - 1) / 2 : Infinity; return [lo - hw, hi + hw]; };
    const [exl, exh] = ext(d.x), [eyl, eyh] = ext(d.y);
    const xs = [], ys = [];
    for (let k = 0; k < N; k++) {
      const t = N > 1 ? k / (N - 1) : 0;
      const px = x0 + (x1 - x0) * t, py = y0 + (y1 - y0) * t;
      if (px < exl || px > exh || py < eyl || py > eyh) continue;
      xs.push(t * len);
      ys.push(d.z[nearest(d.y, py)][nearest(d.x, px)]);
    }
    const unit = (l) => (/\[(.*)\]/.exec(friendlyAxis(l)) || [])[1] || '';
    const ux = unit(d.xl), uy = unit(d.yl);
    const axisLabel = ux && ux === uy ? `Distance along line [${ux}]` : 'Distance along line (axis units; x and y units differ)';
    out.push({ det, label: detLabel(f, det), h: { x: xs, y: ys, label: `Line profile: (${fmtc(x0)}, ${fmtc(y0)}) to (${fmtc(x1)}, ${fmtc(y1)})` }, v: null, line: true, xTitle: axisLabel });
  }
  return out;
}
const fmtc = (v) => Number(v.toPrecision(7)).toString();
export const nearest = (arr, v) => { let b = 0, bd = Infinity; arr.forEach((a, i) => { const dd = Math.abs(a - v); if (dd < bd) { bd = dd; b = i; } }); return b; };

export function peakCell(f, det, dip = false) {
  const m = f.data[det]?.map;
  const d = displayMap(m, f.map.swap, f.map.source || 'value');
  let best = dip ? Infinity : -Infinity, at = null;
  d.z.forEach((r, iy) => r.forEach((v, ix) => { if (Number.isFinite(v) && (dip ? v < best : v > best)) { best = v; at = [d.x[ix], d.y[iy]]; } }));
  return at;
}

// Remember each map panel's drawn ranges; mapRelayoutView needs the pre-zoom view of a one-axis zoom.
export function rememberMapRanges(gd) {
  const fl = gd._fullLayout;
  if (!fl) return;
  const out = {};
  for (const k of Object.keys(fl)) {
    const m = /^xaxis(\d*)$/.exec(k);
    if (m && fl[`yaxis${m[1]}`]) out[m[1]] = { x: [...fl[k].range], y: [...fl[`yaxis${m[1]}`].range] };
  }
  gd._mapRanges = out;
}

// Map a plotly_relayout event from ANY map panel to a new shared view: {reset:true}, {x, y}, or null (not a zoom).
// Equal-scale maps cannot zoom one axis: Plotly would keep the other range and squeeze the panel's domain to a sliver
// (the "collapsed Detector 1" bug). A one-axis zoom becomes a uniform zoom by the same factor about the other axis's centre.
export function mapRelayoutView(gd, ev) {
  let panel = null, reset = false;
  const r = { x: [], y: [] };
  for (const [k, v] of Object.entries(ev || {})) {
    const m = /^([xy])axis(\d*)\.(range|range\[0\]|range\[1\]|autorange)$/.exec(k);
    if (!m) continue;
    panel = panel ?? m[2];
    if (m[2] !== panel) continue;
    if (m[3] === 'autorange') { if (v) reset = true; continue; }
    if (m[3] === 'range') r[m[1]] = [...v];
    else r[m[1]][m[3] === 'range[0]' ? 0 : 1] = v;
  }
  if (reset) return { reset: true };
  if (panel === null) return null;
  const ok = (a) => a.length === 2 && a.every(Number.isFinite) && a[0] !== a[1];
  const prev = gd._mapRanges?.[panel];
  if (ok(r.x) && ok(r.y)) return { x: r.x, y: r.y };
  if (!prev) return null;
  const scaled = (newR, oldR, other) => {
    const f = Math.abs(newR[1] - newR[0]) / Math.abs(oldR[1] - oldR[0]);
    const c = (other[0] + other[1]) / 2, h = ((other[1] - other[0]) * f) / 2;
    return [c - h, c + h];
  };
  if (ok(r.x)) return { x: r.x, y: scaled(r.x, prev.x, prev.y) };
  if (ok(r.y)) return { x: scaled(r.y, prev.y, prev.x), y: r.y };
  return null;
}

// Colorbars follow the drawn height of their (aspect-constrained) map.
export function fitColorbars(gd) {
  const fl = gd._fullLayout;
  if (!fl) return Promise.resolve();
  const idx = [], ys = [], lens = [];
  gd.data.forEach((t, i) => {
    if (t.type !== 'heatmap') return;
    const ax = fl[`yaxis${(t.yaxis || 'y').slice(1)}`];
    if (!ax?.domain) return;
    idx.push(i); ys.push((ax.domain[0] + ax.domain[1]) / 2); lens.push(ax.domain[1] - ax.domain[0]);
  });
  if (!idx.length) return Promise.resolve();
  return Plotly.restyle(gd, { 'colorbar.y': ys, 'colorbar.len': lens, 'colorbar.yanchor': idx.map(() => 'middle') }, idx);
}
