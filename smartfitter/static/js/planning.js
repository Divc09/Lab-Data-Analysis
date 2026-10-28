// Measurement planning beside a fit: convergence with iterations, next scan points, ODMR magnetic sensitivity.
import { call } from './api.js';
import { S, emit } from './state.js';
import { h, mount, field, select, modal, toast, fmt, clone, check, input, num } from './util.js';
import { displayValue, parameterUnit } from './unitdisplay.js';
import { baseStyle, baseLayout, axisLayout, plotConfig } from './plotstyle.js';
import { copyText } from './export.js';
import { prepareFitSpec } from './analyze.js';

const plan = (f) => (f.plan ||= { conv: {}, points: {}, view: { target: 5, count: 10, auto: false } });
const requestKey = (f, spec) => JSON.stringify({ d: f.detector, m: f.mode, p: f.proc, s: { ...spec, custom_spec: undefined }, r: f.sourceRevision || 0 });
const stale = (f, r, spec) => !r || r.key !== requestKey(f, spec);

export function fmtDuration(s) {
  if (!Number.isFinite(s) || s < 0) return '–';
  if (s < 90) return `${fmt(s, 2)} s`;
  if (s < 90 * 60) return `${fmt(s / 60, 2)} min`;
  if (s < 48 * 3600) return `${fmt(s / 3600, 2)} h`;
  return `${fmt(s / 86400, 2)} d`;
}
export function fmtTesla(t) {
  if (!Number.isFinite(t)) return '–';
  const a = Math.abs(t);
  const [scale, unit] = a >= 1e-3 ? [1e3, 'mT'] : a >= 1e-6 ? [1e6, 'µT'] : a >= 1e-9 ? [1e9, 'nT'] : [1e12, 'pT'];
  return `${fmt(t * scale, 3)} ${unit}`;
}

// Positions and offsets (resonance centres, baselines, phases) need an absolute target: their error relative to the
// value depends on where zero is (2.87 GHz, or a baseline near 0), not on how well they are measured.
// Case-sensitive: lowercase c is the Rabi offset, uppercase C the ODMR contrast amplitude.
export const isLocation = (name) => /^(x0(_\d+)?|center(_\d+)?|y0|c|phi\d*|t0|delay|background|offset|baseline)$/.test(name);

// Iterations needed to reach the target, assuming error ∝ 1/√N: a percentage for scale parameters,
// an absolute ± (native unit) for location parameters.
export function iterationsFor(row, n, view) {
  const loc = isLocation(row.name), t = loc ? view.abs?.[row.name] : view.target;
  const ratio = loc ? row.error / t : row.relative / (t / 100);
  if (!(t > 0) || !Number.isFinite(ratio)) return null;
  return Math.max(n, Math.ceil(n * ratio ** 2));
}

// analyze.js imports this module too; the cycle is safe because prepareFitSpec is only called at run time.
const specFor = (f) => prepareFitSpec(f);

export async function runConvergence(f, { quiet = false } = {}) {
  const p = plan(f);
  if (p.busy) return;
  const spec = specFor(f), det = f.detector;
  const key = requestKey(f, spec);
  p.busy = 'conv'; emit('insp');
  try {
    const result = await call('fit_convergence', { path: f.path, detector: det, mode: f.mode, proc: clone(f.proc), spec, target_percent: p.view.target },
      { quiet, label: 'Fitting cumulative iteration means' });
    if (S.files[f.path] !== f) return;
    result.key = key;
    p.conv[det] = result;
  } catch (e) { if (!quiet) toast(e.message, 'err'); }
  finally { p.busy = ''; emit('insp'); }
}

export async function runPoints(f) {
  const p = plan(f);
  if (p.busy) return;
  const spec = specFor(f), det = f.detector;
  const key = requestKey(f, spec);
  p.busy = 'points'; emit('insp');
  try {
    const result = await call('suggest_next_points', { path: f.path, detector: det, mode: f.mode, proc: clone(f.proc), spec,
      parameter: p.view.param || null, count: p.view.count || 10, x_max: p.view.xmax ?? null }, { label: 'Ranking candidate scan points' });
    if (S.files[f.path] !== f) return;
    result.key = key;
    p.points[det] = result;
    p.view.param = result.parameter;
  } catch (e) { toast(e.message, 'err'); }
  finally { p.busy = ''; emit('insp'); }
}

function convergenceSection(f, spec) {
  const p = plan(f), v = p.view, r = p.conv[f.detector], it = f.data[f.detector]?.iterations;
  const enough = (it?.selected?.length || 0) >= 3;
  const old = r && stale(f, r, spec);
  const perIter = Number.isFinite(v.minutesPerIteration) ? v.minutesPerIteration * 60 : r?.timing?.per_iteration_s;
  const n = r?.selected?.length;
  const head = h('div', { class: 'row wrap', style: 'gap:6px;align-items:flex-end' },
    field('Target ± (%)', input(v.target, (x) => { const t = num(x); if (t > 0) { v.target = t; emit('insp'); } }, { numeric: true, style: 'width:64px', 'aria-label': 'Target relative error percent' })),
    field('Time / iteration (min)', input(Number.isFinite(v.minutesPerIteration) ? v.minutesPerIteration : '', (x) => { v.minutesPerIteration = num(x) ?? undefined; emit('insp'); },
      { numeric: true, style: 'width:80px', placeholder: r?.timing?.per_iteration_s ? fmt(r.timing.per_iteration_s / 60, 3) : 'unknown', 'aria-label': 'Minutes per iteration' })),
    h('button', { class: 'btn sm', disabled: !!p.busy || !enough || f.fitting, onclick: () => runConvergence(f) }, p.busy === 'conv' ? 'Fitting…' : r ? 'Rerun' : 'Check convergence'),
    r ? h('button', { class: 'btn sm', onclick: () => openConvergencePlot(f) }, 'Plot…') : null);
  const auto = check('Rerun when the file updates (live acquisition)', !!v.auto, (x) => { v.auto = x; emit('insp'); },
    { title: 'When folder watching reloads a new checkpoint of this file, rerun the convergence check after the refit.' });
  if (!enough) return h('div', {}, h('div', { class: 'lbl' }, 'When is it done?'), h('div', { class: 'note' }, 'Needs at least 3 selected iterations of a 1D scan.'));
  const rows = r ? r.summary : [];
  return h('div', { style: 'display:flex;flex-direction:column;gap:6px' },
    h('div', { class: 'lbl' }, 'When is it done?'),
    head, auto,
    r ? h('div', { class: 'note', style: old ? 'color:var(--warn)' : '' }, old ? 'STALE: settings or file changed since this check; rerun.'
      : `${n} iterations fitted in ${r.rows.length} cumulative steps · ${r.model}. ${r.acquisition?.target ? `Run target: ${r.acquisition.target} iterations.` : ''}`) : null,
    r ? h('table', { class: 'tbl' }, h('tr', {}, ['Parameter', 'Now', '±', 'Target ±', 'Needs', 'Time left'].map((t) => h('th', {}, t))),
      rows.map((row) => {
        const need = iterationsFor(row, n, v), more = need == null ? null : need - n;
        const warn = !!row.note && Number.isFinite(row.exponent) && row.exponent > -0.25;
        const unit = parameterUnit({ unit: row.unit }, f.info), loc = isLocation(row.name);
        const target = loc
          ? input(v.abs?.[row.name] ?? '', (x) => { (v.abs ||= {})[row.name] = num(x) ?? undefined; emit('insp'); },
            { numeric: true, unit, style: 'width:76px', placeholder: `set (${unit || 'native'})`, 'aria-label': `Target error for ${row.name} in ${unit || 'native units'}` })
          : `${fmt(v.target, 3)}%`;
        return h('tr', { title: [loc ? `${row.name} is a position/offset: set an absolute target in ${unit || 'its native unit'}.` : '', row.note,
          Number.isFinite(row.exponent) ? `Measured error ∝ N^${fmt(row.exponent, 2)} (1/√N is −0.5)` : ''].filter(Boolean).join('\n') },
          h('td', { class: 'txt' }, row.name), h('td', {}, displayValue(row.value, null, unit, { precision: loc ? 7 : 4 }).withUnit),
          h('td', {}, loc ? displayValue(row.error, null, unit, { precision: 3 }).withUnit : Number.isFinite(row.relative) ? `${fmt(100 * row.relative, 3)}%` : '–'),
          h('td', {}, target),
          h('td', { style: warn ? 'color:var(--warn)' : '', title: need == null ? '' : `${need} iterations in total` },
            need == null ? '–' : `${more === 0 ? 'done' : `+${more}`}${warn ? ' ⚠' : ''}`),
          h('td', {}, more == null || !Number.isFinite(perIter) ? '–' : more === 0 ? '–' : fmtDuration(more * perIter)));
      })) : null,
    r ? h('div', { class: 'note' }, `Target ± (%) applies to scale parameters (lifetimes, widths, amplitudes, frequencies); centres and offsets take an absolute target in their own unit. Time per iteration: ${Number.isFinite(v.minutesPerIteration) ? 'entered' : r.timing?.per_iteration_s ? r.timing.source : 'unknown — enter it'}. ⚠ = error is not falling like 1/√N on these data (hover for details); the projection is then unreliable. Projections assume the value stays put and the model is right.`) : null);
}

function pointsSection(f, fit, spec) {
  const p = plan(f), v = p.view, r = p.points[f.detector];
  const free = (fit.params || []).filter((q) => !q.locked).map((q) => q.name);
  if (!free.length || f.info.dim === 'scan2d' || f.info.type === 'TimeTrace') return null;
  const unit = f.info.x_unit || '';
  const old = r && stale(f, r, spec);
  const param = free.includes(v.param) ? v.param : (r?.parameter && free.includes(r.parameter) ? r.parameter : free.find((n) => /^(T1|T2|T2_star|tau|f|f1|x0|x0_1)$/.test(n)) || free[0]);
  const list = r ? r.points.flatMap((q) => Array(q.repeats).fill(q.x)) : [];
  const pUnit = parameterUnit(fit.params.find((q) => q.name === r?.parameter) || {}, f.info);
  const sigma = (s) => displayValue(s, null, pUnit, { precision: 3 }).withUnit;
  return h('div', { style: 'display:flex;flex-direction:column;gap:6px' },
    h('div', { class: 'lbl' }, 'Where to measure next'),
    h('div', { class: 'row wrap', style: 'gap:6px;align-items:flex-end' },
      field('Improve', select(free, param, (x) => { v.param = x; emit('insp'); })),
      field('Points', input(v.count, (x) => { const c = Math.round(num(x)); if (c >= 1 && c <= 500) v.count = c; }, { numeric: true, style: 'width:56px', 'aria-label': 'Number of points to suggest' })),
      field(`Up to (${unit || 'x'})`, input(v.xmax ?? '', (x) => { v.xmax = num(x); }, { numeric: true, unit, style: 'width:90px', placeholder: 'auto', 'aria-label': 'Largest candidate x' })),
      h('button', { class: 'btn sm', disabled: !!p.busy || f.fitting, onclick: () => { v.param = param; runPoints(f); } }, p.busy === 'points' ? 'Ranking…' : 'Suggest points')),
    r ? h('div', { class: 'note', style: old ? 'color:var(--warn)' : '' }, old ? 'STALE: settings or file changed; rerun.' : '') : null,
    r ? h('div', { class: 'chips-s' }, r.points.map((q) => h('span', { class: 'chip-s ok', title: `${q.repeats} measurement(s) at ${q.x} ${unit}` }, `${fmt(q.x, 5)}${q.repeats > 1 ? ` ×${q.repeats}` : ''}`))) : null,
    r ? h('div', { class: 'kv' },
      h('span', { class: 'k' }, `σ(${r.parameter}) now`), h('span', { class: 'v' }, sigma(r.error_now)),
      h('span', { class: 'k' }, `with these ${r.count}`), h('span', { class: 'v' }, `${sigma(r.error_after)} (${fmt(100 * (1 - r.error_after / r.error_now), 3)}% smaller)`),
      h('span', { class: 'k' }, `${r.count} spread like the scan`), h('span', { class: 'v' }, `${sigma(r.error_uniform)} (${fmt(100 * (1 - r.error_uniform / r.error_now), 3)}% smaller)`),
      h('span', { class: 'k' }, 'Worth'), h('span', { class: 'v' }, r.equivalent_uniform_points == null ? '–' : `≈ ${r.equivalent_uniform_points} evenly spread points`)) : null,
    r ? h('div', { class: 'row', style: 'gap:6px' },
      h('button', { class: 'btn sm', onclick: () => copyText(list.join(', '), `Copied ${list.length} ${unit} values`) }, `Copy list (${unit || 'native'})`),
      h('span', { class: 'note', style: 'margin:0' }, `Searched ${fmt(r.range[0], 5)}–${fmt(r.range[1], 5)} ${unit}.`)) : null,
    r ? h('details', {}, h('summary', {}, 'Assumptions'), h('div', { class: 'note' }, `${r.note} σ is the model-based standard error; the fit reported ± ${sigma(r.fit_error)} (it may include SEM weights or other settings this estimate leaves out).`)) : null);
}

function sensitivitySection(f, fit) {
  const s = fit.odmr_sensitivity;
  if (!s) return null;
  if (s.error) return h('div', {}, h('div', { class: 'lbl' }, 'Magnetic sensitivity'), h('div', { class: 'note' }, s.error));
  const v = plan(f).view;
  const t = Number.isFinite(v.secondsPerPoint) ? v.secondsPerPoint : s.timing?.per_point_s;
  const eta = Number.isFinite(t) && Number.isFinite(s.delta_b_per_point_t) ? s.delta_b_per_point_t * Math.sqrt(t) : NaN;
  const shot = s.shot_noise?.eta_t_per_rthz;
  const lines = () => [
    `ODMR sensitivity · ${f.info.name} · ${f.detector} · ${s.mode}`,
    `Steepest slope at ${fmt(s.bias_frequency, 7)} ${s.x_unit}: ${fmt(s.max_slope_per_hz, 4)} per Hz; residual noise ${fmt(s.noise_rms, 4)}`,
    `δB per point (all iterations averaged) = ${fmtTesla(s.delta_b_per_point_t)}`,
    Number.isFinite(eta) ? `η = ${fmtTesla(eta)}/√Hz with ${fmtDuration(t)} per point (${Number.isFinite(v.secondsPerPoint) ? 'entered' : s.timing?.source})` : '',
    Number.isFinite(shot) ? `Shot-noise limit ${fmtTesla(shot)}/√Hz (C = ${fmt(s.shot_noise.contrast, 3)}, FWHM = ${fmt(s.shot_noise.fwhm_hz / 1e6, 4)} MHz, R = ${fmt(s.shot_noise.rate_per_s, 4)} /s)` : '',
    f.path].filter(Boolean).join('\n');
  return h('div', { style: 'display:flex;flex-direction:column;gap:6px' },
    h('div', { class: 'lbl' }, 'Magnetic sensitivity'),
    h('div', { class: 'kv' },
      h('span', { class: 'k' }, 'Bias at'), h('span', { class: 'v' }, `${fmt(s.bias_frequency, 7)} ${s.x_unit}`),
      h('span', { class: 'k' }, 'δB per point'), h('span', { class: 'v', title: 'Noise of one averaged point ÷ (γe × steepest slope)' }, fmtTesla(s.delta_b_per_point_t)),
      h('span', { class: 'k' }, 'η (measured)'), h('span', { class: 'v', style: 'font-weight:600' }, Number.isFinite(eta) ? `${fmtTesla(eta)}/√Hz` : 'enter time per point'),
      h('span', { class: 'k' }, 'η (shot-noise limit)'), h('span', { class: 'v', title: s.shot_noise?.note || s.rate_source }, Number.isFinite(shot) ? `${fmtTesla(shot)}/√Hz` : 'unavailable'),
      ...(Number.isFinite(eta) && Number.isFinite(shot) ? [h('span', { class: 'k' }, 'Measured / limit'), h('span', { class: 'v' }, `${fmt(eta / shot, 3)}×`)] : [])),
    h('div', { class: 'row wrap', style: 'gap:6px;align-items:flex-end' },
      field('Time per point (s, all iterations)', input(Number.isFinite(v.secondsPerPoint) ? v.secondsPerPoint : '', (x) => { v.secondsPerPoint = num(x) ?? undefined; emit('insp'); },
        { numeric: true, style: 'width:90px', placeholder: Number.isFinite(s.timing?.per_point_s) ? fmt(s.timing.per_point_s, 4) : 'unknown', 'aria-label': 'Integration time per frequency point in seconds' })),
      h('button', { class: 'btn sm', onclick: () => copyText(lines(), 'Sensitivity copied') }, 'Copy')),
    h('details', {}, h('summary', {}, 'Method'), h('div', { class: 'note' }, `${s.note} Time per point: ${Number.isFinite(v.secondsPerPoint) ? 'entered' : s.timing?.source || 'unknown'}; wall-clock time includes overhead, so η is conservative.`),
      s.shot_noise ? h('div', { class: 'note' }, s.shot_noise.note) : h('div', { class: 'note' }, `Shot-noise limit: ${s.rate_source}.`)));
}

export function planningCard(f, fit) {
  if (!fit || fit.success === false || f.info.dim === 'scan2d') return null;
  const spec = specFor(f);
  const parts = [convergenceSection(f, spec), pointsSection(f, fit, spec), sensitivitySection(f, fit)].filter(Boolean);
  if (!parts.length) return null;
  const v = plan(f).view;
  return h('details', { open: !!v.open, ontoggle: (e) => { v.open = e.target.open; } }, h('summary', {}, 'Plan the measurement'),
    h('div', { style: 'display:flex;flex-direction:column;gap:12px;margin-top:8px' }, parts));
}

// Error against iterations on log-log axes: relative % for scale parameters, absolute (native unit) for locations.
export function convergenceFigure(f, r, parameter, view) {
  const rows = r.rows.filter((x) => x.comparable);
  const pick = (row) => row.params.find((q) => q.name === parameter);
  const loc = isLocation(parameter), unit = pick(rows[rows.length - 1])?.unit || '';
  const metric = (q) => !q || !Number.isFinite(q.error) ? null : loc ? q.error : q.value ? 100 * q.error / Math.abs(q.value) : null;
  const target = loc ? view.abs?.[parameter] : view.target;
  const last = rows[rows.length - 1], now = metric(pick(last));
  const need = Number.isFinite(now) && target > 0 ? Math.max(last.iterations, Math.ceil(last.iterations * (now / target) ** 2)) : null;
  const projX = Number.isFinite(now) ? [last.iterations, Math.max(need || 0, last.iterations * 1.5)] : [];
  const st = baseStyle({ title: `${parameter}: ${loc ? 'error' : 'relative error'} vs iterations`, x: {}, y: {} });
  const targetText = loc ? `${fmt(target, 3)} ${unit}` : `${fmt(target, 3)}%`;
  return { data: [
    { type: 'scatter', mode: 'lines+markers', name: 'measured', x: rows.map((x) => x.iterations), y: rows.map((x) => metric(pick(x))), marker: { size: 8, color: '#2a78d6' } },
    { type: 'scatter', mode: 'lines', name: '1/√N projection', x: projX, y: projX.map((m) => now * Math.sqrt(last.iterations / m)), line: { dash: 'dash', color: '#d56722' } },
  ], layout: baseLayout(st, { xaxis: { ...axisLayout(st, 'x', 'Iterations averaged'), type: 'log' },
    yaxis: { ...axisLayout(st, 'y', loc ? `± ${parameter} [${unit || 'native'}]` : 'Error / |value| [%]'), type: 'log' },
    shapes: target > 0 ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: target, y1: target, line: { dash: 'dot', color: '#3a9d5d' } }] : [],
    annotations: target > 0 ? [{ xref: 'paper', x: 1, y: Math.log10(target), xanchor: 'right', yanchor: 'bottom', text: `target ${targetText}${need ? ` → ${need} iterations` : ''}`, showarrow: false, font: { size: 11, color: '#3a9d5d' } }] : [],
    margin: { l: 65, r: 20, b: 45, t: 35 }, legend: { x: 0, y: 0, xanchor: 'left', yanchor: 'bottom' } }) };
}

export function valueFigure(f, r, parameter) {
  const rows = r.rows.filter((x) => x.comparable);
  const pick = (row) => row.params.find((q) => q.name === parameter);
  const st = baseStyle({ title: `${parameter} vs iterations averaged`, x: {}, y: {} });
  const unit = parameterUnit({ unit: pick(rows[rows.length - 1])?.unit }, f.info);
  return { data: [{ type: 'scatter', mode: 'markers', x: rows.map((x) => x.iterations), y: rows.map((x) => pick(x)?.value ?? null),
    error_y: { type: 'data', array: rows.map((x) => pick(x)?.error ?? null) }, marker: { size: 8, color: '#2a78d6' }, showlegend: false }],
  layout: baseLayout(st, { xaxis: { ...axisLayout(st, 'x', 'Iterations averaged'), type: 'log' }, yaxis: axisLayout(st, 'y', `${parameter} [${unit || 'native'}]`),
    margin: { l: 65, r: 20, b: 45, t: 35 }, showlegend: false }) };
}

export function openConvergencePlot(f) {
  const p = plan(f), r = p.conv[f.detector];
  if (!r) return;
  const names = r.summary.map((x) => x.name);
  let parameter = names.includes(p.view.plotParam) ? p.view.plotParam : names.find((n) => /^(T1|T2|T2_star|tau|f|x0)$/.test(n)) || names[0];
  const a = h('div', { class: 'iteration-chart' }), b = h('div', { class: 'iteration-chart' }), body = h('div', { class: 'iteration-results' });
  const draw = () => {
    mount(body, h('div', { class: 'row' }, h('strong', {}, `Convergence · ${f.info.name} · ${r.detector}`), h('span', { class: 'sp' }),
      field('Parameter', select(names, parameter, (x) => { parameter = p.view.plotParam = x; draw(); })), h('button', { class: 'btn', onclick: () => m.close() }, 'Close')),
    h('div', { class: 'note' }, r.note), a, b);
    const fa = convergenceFigure(f, r, parameter, p.view), fb = valueFigure(f, r, parameter);
    Plotly.react(a, fa.data, fa.layout, plotConfig({ edits: {} }));
    Plotly.react(b, fb.data, fb.layout, plotConfig({ edits: {} }));
  };
  const m = modal(body, { className: 'iteration-results-modal', onClose: () => { Plotly.purge(a); Plotly.purge(b); } });
  draw();
}
