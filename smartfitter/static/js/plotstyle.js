// Plot presentation model shared by every plot: defaults, user overrides, Plotly mapping.
import { fmt, fmtPm } from './util.js';

export const PALETTES = {
  lab: ['#1F5FBF', '#E06A1B', '#0E8A7E', '#B5307A', '#7A4FD6', '#6B7380', '#C9A227', '#111111'],
  okabe: ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000'],
  gray: ['#111111', '#555555', '#888888', '#AAAAAA', '#333333', '#777777', '#999999', '#CCCCCC'],
};

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
export const friendlyAxis = (label) => AXIS_NAMES[(label || '').trim().toLowerCase()] || label || 'X';

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
  if (a.minor) lay.minor = { ticks: st.ticks === 'none' ? '' : st.ticks, ticklen: 3, nticks: Number(a.minor) + 1, showgrid: false };
  const lo = a.min ?? range?.[0], hi = a.max ?? range?.[1];
  if (lo !== null && lo !== undefined && hi !== null && hi !== undefined) { lay.range = a.log ? [Math.log10(lo), Math.log10(hi)] : [lo, hi]; lay.autorange = false; }
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
    ...pos, bgcolor: 'rgba(255,255,255,0.92)', bordercolor: L.frame === false ? 'rgba(0,0,0,0)' : '#D9DDE4', borderwidth: 1,
    font: { size: L.size || Math.round(st.base * 0.82) }, itemclick: 'toggle', itemdoubleclick: false,
  };
  if (L.x !== null && L.x !== undefined) { lay.x = L.x; lay.y = L.y; lay.xanchor = 'left'; lay.yanchor = 'top'; }
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
  if (fit.model === 'GaussianPeaks' || fit.model === 'TimeTraceDecay' || !lines.length) {
    for (const p of fit.params) {
      let label = p.symbol || p.name, value = p.value, err = p.error, unit = p.unit === 'axis units' ? '' : (p.unit === '-' ? '' : p.unit);
      if (p.name.startsWith('sigma_')) { label = `FWHM${p.name.slice(5)}`; value = p.value * 2.354820045; err = p.error ? p.error * 2.354820045 : null; }
      if (p.name.startsWith('amplitude_')) continue;
      lines.push({ key: `p:${p.name}`, label, value, err, unit });
    }
  }
  lines.push({ key: 'r2', label: 'R²', value: fit.r2, unit: '', defaultOff: false });
  // A standard error many orders below the value comes from a degenerate covariance; do not show it.
  for (const l of lines) if (l.err && Math.abs(l.err) < 1e-6 * Math.abs(l.value)) l.err = null;
  return lines;
}

export function resultsAnnotation(st, fit, info, color = '#1D2330') {
  const lines = resultLines(fit, info);
  const cfg = st.results || {};
  const shown = lines.filter((l) => (cfg.lines?.[l.key]?.plot ?? !l.defaultOff));
  if (!shown.length) return null;
  const txt = shown.map((l) => {
    const label = cfg.lines?.[l.key]?.label ?? l.label;
    const val = cfg.format === 'pm' && l.err ? fmtPm(l.value, l.err) : fmt(l.value, 4);
    return `${label} = ${val}${l.unit ? ' ' + l.unit : ''}`;
  }).join('<br>');
  const pos = { ul: [0.01, 0.99, 'left', 'top'], ur: [0.99, 0.99, 'right', 'top'], ll: [0.01, 0.01, 'left', 'bottom'], lr: [0.99, 0.01, 'right', 'bottom'] }[cfg.pos || 'ul'];
  return {
    xref: 'paper', yref: 'paper', x: cfg.x ?? pos[0], y: cfg.y ?? pos[1], xanchor: cfg.x != null ? 'left' : pos[2], yanchor: cfg.y != null ? 'top' : pos[3],
    text: txt, showarrow: false, align: 'left', bgcolor: 'rgba(255,255,255,0.92)', bordercolor: '#D9DDE4', borderwidth: 1, borderpad: 6,
    font: { family: 'IBM Plex Mono, Consolas, monospace', size: Math.round(st.base * 0.8), color }, name: 'results',
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
