// Plot presentation model shared by every plot: defaults, user overrides, Plotly mapping.
import { fmt, fmtPm } from './util.js';
import { displayValue, parameterUnit, prettyUnit } from './unitdisplay.js';

export const PALETTES = {
  lab: ['#1F5FBF', '#E06A1B', '#0E8A7E', '#B5307A', '#7A4FD6', '#6B7380', '#C9A227', '#111111'],
  okabe: ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000'],
  gray: ['#111111', '#555555', '#888888', '#AAAAAA', '#333333', '#777777', '#999999', '#CCCCCC'],
  tableau: ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'],
  tol: ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB', '#111111'],
  dark2: ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666'],
  set1: ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999'],
};
// Line-trace scales for overlaid iterations: two-colour ramps, early light -> late dark, both ends readable on white.
export const ITER_SCALES = {
  blues: '#a6c8e6,#08306b',
  reds: '#f4a9a0,#7f0000',
  greens: '#a8dba0,#00441b',
  purples: '#cbc3e3,#3f007d',
  'blue→red': '#2166ac,#b2182b',
};
// Ordered scales for comparison series, sampled first file -> last file; pale ends trimmed so lines stay readable on white.
export const SERIES_SCALES = {
  viridis: '#440154,#3b528b,#21918c,#5ec962,#a5db36', plasma: '#0d0887,#7e03a8,#cc4778,#f89540', magma: '#000004,#51127c,#b73779,#fc8961',
  inferno: '#000004,#56106e,#bb3754,#f98e09', cividis: '#00224e,#434e6c,#7c7b78,#bcaf6f', turbo: '#30123b,#4686fb,#1ae4b6,#faba39,#7a0403',
  coolwarm: '#3b4cc0,#8db0fe,#f49a7b,#b40426', ...ITER_SCALES,
};
// Colour at t in [0, 1] interpolated linearly between comma-separated hex stops.
export function rampColor(stopList, t) {
  const stops = stopList.split(',').map((c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)));
  const u = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(u)), w = u - i;
  // Hex, not rgb(): colour inputs and hexA() only accept #rrggbb.
  return `#${stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * w).toString(16).padStart(2, '0')).join('')}`;
}
// Discrete colours for any palette name; a scale is sampled at 8 evenly spaced points.
export function paletteColors(name) {
  if (PALETTES[name]) return PALETTES[name];
  if (SERIES_SCALES[name]) return Array.from({ length: 8 }, (_, i) => rampColor(SERIES_SCALES[name], i / 7));
  return PALETTES.lab;
}
export const PALETTE_OPTIONS = [
  ['lab', 'Lab default (blue, orange, teal, magenta)'], ['okabe', 'Colour-blind safe (Okabe–Ito)'], ['tol', 'Colour-blind safe (Tol bright)'],
  ['tableau', 'Tableau 10'], ['dark2', 'Dark2'], ['set1', 'Set1'], ['gray', 'Grayscale'],
  ...Object.keys(SERIES_SCALES).map((k) => [k, `Ordered scale · ${k} (first → last file)`]),
];

export const PRESETS = {
  presentation: { font: 'Arial', base: 14, width_in: 8.5, height_in: 5.5, line: 1.6, marker: 6 },
  paper: { font: 'Arial', base: 9, width_in: 3.4, height_in: 2.6, line: 1.0, marker: 4 },
  dense: { font: 'Arial', base: 11, width_in: 7.0, height_in: 4.0, line: 1.0, marker: 3 },
};

export function baseStyle(s = {}) {
  return {
    preset: 'presentation', font: 'Arial', base: 14, width_in: 8.5, height_in: 5.5, dpi: 300, title: '',
    grid: true, mirror: true, ticks: 'outside', bg: 'white', palette: 'lab', line: 1.6, marker: 6,
    legend: { pos: 'ur', frame: true, size: null, cols: 1, x: null, y: null },
    x: {}, y: {}, y2: {}, series: {},
    results: { pos: 'ul', format: 'pm', lines: {}, x: null, y: null, mode: 'compact' },
    shapes: [],
    ...s,
    legendFix: undefined,
  };
}

// Friendly axis names (same mapping as the original app).
const AXIS_NAMES = {
  duration: 'Pulse duration [ns]', 'pulse duration (ns)': 'Pulse duration [ns]', frequency: 'RF frequency [GHz]',
  'rf frequency': 'RF frequency [GHz]', 'stage x': 'Stage X [um]', 'stage y': 'Stage Y [um]', 'stage z': 'Stage Z [um]',
};
// With a resolved unit (core.describe info.x_unit / axis_units) the label is "<name> [<unit>]"; without one the map above applies.
// The acquisition names the swept time "duration"/"Pulse duration" for every pulsed sequence; only Rabi sweeps a pulse
// length. Name it by experiment (the loaded label stays in the file and exports).
const DURATION_NAMES = { T1: 'Wait time τ', SpinEcho: 'Evolution time τ', DynamicDecoupling: 'Evolution time τ', DEERDecay: 'Evolution time τ', Ramsey: 'Free precession τ' };
export const friendlyAxis = (label, unit, type) => {
  const key = (label || '').trim().toLowerCase();
  if (DURATION_NAMES[type] && /^(pulse )?duration( \(ns\))?$/.test(key)) label = DURATION_NAMES[type] + (unit ? '' : ' [ns]');
  const pretty = (s) => s.replace(/\[([^\]]*)\]/, (m, u) => `[${prettyUnit(u)}]`);
  if (!unit) return pretty((DURATION_NAMES[type] && label.startsWith(DURATION_NAMES[type])) ? label : AXIS_NAMES[key] || label || 'X');
  const mapped = DURATION_NAMES[type] && label.startsWith(DURATION_NAMES[type]) ? null : AXIS_NAMES[key];
  const base = (mapped || label || 'X').replace(/\s*[\[(][^\])]*[\])]\s*$/, '').trim() || 'X';
  return `${base} [${prettyUnit(unit)}]`;
};

export const MODE_LABEL = { contrast: 'Contrast', signal: 'Signal', reference: 'Reference', difference: 'Difference' };

const TICKFORMAT = { auto: '', fixed1: '.1f', fixed2: '.2f', fixed3: '.3f', sci: '.2e', eng: '~s', percent: '.0%' };

export function axisLayout(st, ax, autoTitle, { range, type } = {}) {
  const a = st[ax] || {};
  const lay = {
    title: { text: a.title ?? autoTitle, standoff: 8 },
    showline: true, linecolor: '#1D2330', linewidth: 1, mirror: st.mirror ? (ax === 'y2' ? false : 'ticks') : false,
    ticks: st.ticks === 'none' ? '' : st.ticks, ticklen: 5, tickcolor: '#1D2330',
    showgrid: !!st.grid && ax !== 'y2', gridcolor: '#ECEEF2', zeroline: false,
    type: a.log ? 'log' : (type || 'linear'), automargin: true,
    tickformat: TICKFORMAT[a.format || 'auto'] || '',
    exponentformat: 'power', separatethousands: false,
  };
  if (a.dtick) lay.dtick = a.dtick;
  if (a.ticks?.length) { lay.tickmode = 'array'; lay.tickvals = a.ticks.map((t) => t.value); lay.ticktext = a.ticks.map((t) => t.text); }
  else if (a.tick0 != null) lay.tick0 = a.tick0;
  if (a.minor) lay.minor = { ticks: st.ticks === 'none' ? '' : st.ticks, ticklen: 3, nticks: Number(a.minor) + 1, showgrid: false };
  const lo = a.min ?? range?.[0], hi = a.max ?? range?.[1];
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo && (!a.log || lo > 0)) { lay.range = a.log ? [Math.log10(lo), Math.log10(hi)] : [lo, hi]; lay.autorange = false; }
  else lay.autorange = true;
  return lay;
}

export function legendLayout(st) {
  const L = st.legend || {};
  const pos = {
    ur: { x: 0.99, y: 0.99, xanchor: 'right', yanchor: 'top' }, ul: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top' },
    lr: { x: 0.99, y: 0.01, xanchor: 'right', yanchor: 'bottom' }, ll: { x: 0.01, y: 0.01, xanchor: 'left', yanchor: 'bottom' },
    outside: { x: 1.02, y: 1, xanchor: 'left', yanchor: 'top' }, below: { x: 0.5, y: -0.18, xanchor: 'center', yanchor: 'top', orientation: 'h' },
  }[L.pos || 'ur'] || {};
  const lay = {
    ...pos, bgcolor: L.bg || 'rgba(255,255,255,0.92)', bordercolor: L.edge || '#D9DDE4', borderwidth: L.frame === false ? 0 : (L.borderwidth ?? 1),
    title: L.title ? { text: L.title } : undefined,
    font: { size: L.size || Math.round(st.base * 0.82), color: L.color || '#1D2330' }, itemclick: 'toggle', itemdoubleclick: false,
  };
  if (L.x !== null && L.x !== undefined) { lay.x = L.x; lay.y = L.y; lay.xanchor = L.xanchor || 'left'; lay.yanchor = L.yanchor || 'top'; }
  if (L.cols > 1) { lay.orientation = 'h'; lay.entrywidthmode = 'fraction'; lay.entrywidth = 1 / L.cols; }
  return lay;
}

export function baseLayout(st, extra = {}) {
  return {
    paper_bgcolor: st.bg === 'transparent' ? 'rgba(0,0,0,0)' : '#FFFFFF',
    plot_bgcolor: st.bg === 'transparent' ? 'rgba(0,0,0,0)' : '#FFFFFF',
    font: { family: `${st.font}, Helvetica, sans-serif`, size: st.base, color: '#1D2330' },
    title: st.title ? { text: st.title, x: 0.5, xanchor: 'center', font: { size: Math.round(st.base * 1.2) } } : undefined,
    margin: { l: 70, r: 24, t: st.title ? 50 : 18, b: 58 },
    hovermode: 'closest', hoverlabel: { font: { family: 'IBM Plex Mono, Consolas, monospace', size: 12 } },
    dragmode: 'zoom', showlegend: true, separators: '.',
    ...extra,
  };
}

// Series styling: defaults per key, then user overrides from style.series[key].
export function seriesStyle(st, key, def) {
  const o = (st.series || {})[key] || {};
  return {
    name: o.name ?? def.name, color: o.color ?? def.color, dash: o.dash ?? def.dash ?? 'solid',
    width: o.width ?? def.width ?? st.line, marker: o.marker ?? def.marker ?? 'circle', size: o.size ?? def.size ?? st.marker,
    opacity: o.opacity ?? def.opacity ?? 1, visible: o.visible ?? def.visible ?? true, legend: o.legend ?? def.legend ?? true,
    mode: o.mode ?? def.mode,
  };
}

export function scatterTrace(st, key, def, x, y, extra = {}) {
  const s = seriesStyle(st, key, def);
  const mode = s.mode || 'lines+markers';
  return {
    type: (x?.length || 0) > 4000 ? 'scattergl' : 'scatter', x, y, name: s.name, mode,
    line: { color: s.color, width: s.width, dash: s.dash }, marker: { color: s.color, size: s.size, symbol: s.marker },
    opacity: s.opacity, visible: s.visible === false ? 'legendonly' : true, showlegend: s.legend, meta: { key },
    hovertemplate: '%{x:.6g}, %{y:.6g}<extra>' + (s.name || '').replace(/</g, '') + '</extra>',
    ...extra,
  };
}

// ---- Results box ----
// Shown on the plot by default; the rest can be ticked in the plot editor.
const DEFAULT_ON = new Set(['pi_time_ns', 'pi_over_2_time_ns', 'rabi_freq_MHz', 'T2_star_ns', 'detuning_MHz', 'T2_ns', 'T2_dd_ns', 'T1_ns', 'center_freq_units', 'fwhm_units']);
const METRIC_LINES = {
  Rabi: [['pi_time_ns', 'π', 'ns'], ['pi_over_2_time_ns', 'π/2', 'ns'], ['rabi_freq_MHz', 'f_Rabi', 'MHz'], ['delay_ns', 'delay', 'ns'], ['programmed_pi_time_ns', 'π (programmed)', 'ns'], ['T2rho_ns', 'T2ρ', 'ns']],
  Ramsey: [['T2_star_ns', 'T2*', 'ns'], ['detuning_MHz', 'detuning', 'MHz'], ['stretch_n', 'n', '']],
  SpinEcho: [['T2_ns', 'T2', 'ns'], ['stretch_n', 'n', '']],
  DynamicDecoupling: [['T2_dd_ns', 'T2,DD', 'ns'], ['stretch_n', 'n', '']],
  T1: [['T1_ns', 'T1', 'ns'], ['stretch_n', 'n', '']],
  ODMR: [['center_freq_units', 'f0', ''], ['fwhm_units', 'FWHM', ''], ['num_peaks', 'peaks', ''], ['center_mean', 'mean center', '']],
  TimeTrace: [],
};

export function resultLines(fit, info) {
  if (!fit) return [];
  const lines = [];
  const fam = { DEERDuration: 'Rabi', DEERFrequency: 'ODMR', DEERDecay: 'SpinEcho' }[fit.experiment] || fit.experiment;
  const m = fit.metrics || {};
  for (const [k, lab, unit] of METRIC_LINES[fam] || []) {
    if (typeof m[k] === 'number') lines.push({ key: k, label: lab, value: m[k], err: m[`${k}_err`] ?? m[k.replace('_ns', '_ns_nominal') + '_err'], unit, defaultOff: !DEFAULT_ON.has(k) });
  }
  if (fam === 'ODMR') for (const l of lines) if (['center_freq_units', 'fwhm_units', 'center_mean'].includes(l.key)) l.unit = info.x_unit || 'GHz';
  if (fit.model.startsWith('T1')) {
    const t1 = fit.params.find((p) => p.name === 'T1');
    const line = lines.find((l) => l.key === 'T1_ns');
    if (t1 && line) Object.assign(line, { value: t1.value, err: t1.error, unit: t1.unit || info.x_unit || 'ns' });
    for (const p of fit.params.filter((p) => p.name === 'A' || p.name === 'y0')) lines.push({ key: `p:${p.name}`, label: `${p.name}${fit.extras?.t1_prep_mode === 'normalized_1_minus_data' ? ' (norm.)' : ''}`, value: p.value, err: p.error, unit: p.unit, defaultOff: true });
  }
  if (fit.model === 'GaussianPeaks' || fit.model === 'TimeTraceDecay' || !lines.length) {
    for (const p of fit.params) {
      let label = p.symbol || p.name, value = p.value, err = p.error, unit = p.unit === '-' ? '' : parameterUnit(p, info);
      if (p.name.startsWith('sigma_')) { label = `FWHM${p.name.slice(5)}`; value = p.value * 2.354820045; err = p.error ? p.error * 2.354820045 : null; }
      if (p.name.startsWith('amplitude_')) continue;
      lines.push({ key: `p:${p.name}`, label, value, err, unit });
    }
  }
  // Derived metrics often omit errors. Reuse a matching fitted parameter (and exact linear factor), never invent covariance.
  const aliases = { T2_ns: ['T2', 1], T2_dd_ns: ['T2', 1], T2_star_ns: [fit.params.some((p) => p.name === 'T2_star') ? 'T2_star' : 'tau', 1], stretch_n: ['n', 1], center_freq_units: ['x0', 1], fwhm_units: ['w', 2], detuning_MHz: ['f', 1000] };
  for (const l of lines) {
    const alias = aliases[l.key], p = alias && fit.params.find((p) => p.name === alias[0]);
    if (p && l.err == null) l.err = p.error == null ? null : p.error * alias[1];
  }
  if (fam === 'ODMR') for (const p of fit.params.filter((p) => /^(x0|w)_\d+$/.test(p.name))) {
    const width = p.name.startsWith('w_');
    lines.push({ key: `p:${p.name}`, label: width ? `FWHM ${p.name.slice(2)}` : `f0 ${p.name.slice(3)}`, value: p.value * (width ? 2 : 1), err: p.error == null ? null : p.error * (width ? 2 : 1), unit: info.x_unit || 'GHz', defaultOff: false });
  }
  lines.push({ key: 'r2', label: 'R²', value: fit.r2, unit: '', defaultOff: false });
  // A standard error many orders below the value comes from a degenerate covariance; do not show it.
  for (const l of lines) if (l.err && Math.abs(l.err) < 1e-6 * Math.abs(l.value)) l.err = null;
  return lines;
}

export function resultsAnnotation(st, fit, info, color = '#1D2330') {
  const lines = resultLines(fit, info);
  const cfg = st.results || {};
  const all = [...lines, ...(cfg.static || []).map((l) => ({ ...l, static: true }))];
  const order = cfg.order || [];
  const shown = all.filter((l) => (cfg.lines?.[l.key]?.plot ?? !l.defaultOff)).sort((a, b) => (order.includes(a.key) ? order.indexOf(a.key) : order.length + all.indexOf(a)) - (order.includes(b.key) ? order.indexOf(b.key) : order.length + all.indexOf(b)));
  if (!shown.length) return null;
  const txt = shown.map((l) => {
    const line = cfg.lines?.[l.key] || {}, label = line.label ?? l.label;
    if (l.static) return line.text ?? l.text ?? label;
    const d = displayValue(l.value, l.err, l.unit, { format: line.format || cfg.format || 'pm', precision: line.precision || cfg.precision || 2, units: line.units || 'auto' });
    return (line.template || '{label} = {value} {unit}').replaceAll('{label}', label).replaceAll('{value}', d.text).replaceAll('{unit}', d.unit).trim();
  }).join('<br>');
  const pos = { ul: [0.01, 0.99, 'left', 'top'], ur: [0.99, 0.99, 'right', 'top'], ll: [0.01, 0.01, 'left', 'bottom'], lr: [0.99, 0.01, 'right', 'bottom'] }[cfg.pos || 'ul'];
  return {
    xref: 'paper', yref: 'paper', x: cfg.x ?? pos[0], y: cfg.y ?? pos[1], xanchor: cfg.x != null ? 'left' : pos[2], yanchor: cfg.y != null ? 'top' : pos[3],
    text: txt, showarrow: false, align: cfg.align || 'left', bgcolor: cfg.bg || 'rgba(255,255,255,0.92)', bordercolor: cfg.edge || '#D9DDE4', borderwidth: cfg.frame === false ? 0 : 1, borderpad: cfg.padding ?? 6,
    font: { family: 'IBM Plex Mono, Consolas, monospace', size: cfg.size || Math.round(st.base * 0.8), color: cfg.color || color }, name: 'results',
  };
}

// User-added shapes/annotations (Plot editor "Add to plot").
export function userDecorations(st) {
  const shapes = [], annotations = [];
  (st.shapes || []).forEach((d, i) => {
    const color = d.color || '#7A4FD6';
    if (d.kind === 'vline') {
      shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: d.x, x1: d.x, y0: 0, y1: 1, line: { color, width: d.width || 1.4, dash: d.dash || 'dash' }, name: `u${i}` });
      if (d.text) annotations.push({ x: d.x, xref: 'x', y: 0.02, yref: 'paper', text: d.text, showarrow: false, xanchor: 'left', xshift: 4, font: { color, size: Math.round(st.base * 0.8) }, name: `u${i}` });
    } else if (d.kind === 'hline') {
      shapes.push({ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: d.y, y1: d.y, line: { color, width: d.width || 1.4, dash: d.dash || 'dash' }, name: `u${i}` });
      if (d.text) annotations.push({ x: 0.99, xref: 'paper', y: d.y, yref: 'y', text: d.text, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color, size: Math.round(st.base * 0.8) }, name: `u${i}` });
    } else if (d.kind === 'region') {
      shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: d.x0, x1: d.x1, y0: 0, y1: 1, fillcolor: color, opacity: d.opacity ?? 0.12, line: { width: 0 }, layer: 'below', name: `u${i}` });
      if (d.text) annotations.push({ x: (d.x0 + d.x1) / 2, xref: 'x', y: 0.98, yref: 'paper', text: d.text, showarrow: false, yanchor: 'top', font: { color, size: Math.round(st.base * 0.8) }, name: `u${i}` });
    } else if (d.kind === 'text' || d.kind === 'arrow') {
      annotations.push({ x: d.x, y: d.y, xref: 'x', yref: 'y', text: d.text || 'Text', showarrow: d.kind === 'arrow', ax: d.ax ?? -40, ay: d.ay ?? -40,
        arrowhead: 2, arrowcolor: color, font: { color: d.kind === 'arrow' ? color : (d.color || '#1D2330'), size: d.size || st.base }, name: `u${i}` });
    }
  });
  return { shapes, annotations };
}

export function plotConfig(extra = {}) {
  return {
    responsive: true, displaylogo: false, scrollZoom: true, doubleClick: 'reset+autosize',
    displayModeBar: false,
    edits: { titleText: true, axisTitleText: true, legendText: true, legendPosition: true, annotationPosition: true, annotationTail: true, shapePosition: true, annotationText: false },
    ...extra,
  };
}
