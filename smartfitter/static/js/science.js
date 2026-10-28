// Scientific helpers for the Analyze inspector: reduced chi-squared, bootstrap errors, NV field from ODMR lines,
// DEER reference normalisation + resonance helper, and time-trace step / ratio / drift tools.
// Everything here states its method and assumptions next to the numbers. State lives on f._sci / f.boot / f.deerRatio.
import { call } from './api.js';
import { S, emit, detLabel, envelopeFor, recordResult, resultFreshness } from './state.js';
import { h, input, check, field, select, seg, fmt, fmtPm, num } from './util.js';

const sci = (f) => (f._sci = f._sci || {});
export function historicalToolNote(f) {
  const records = [
    ...Object.values(f.boot || {}).map((b) => b.res),
    ...Object.values(f.peakPick || {}),
    f.deerRatio?.res,
    f._sci?.nv?.res,
    f._sci?.tt?.steps,
    f._sci?.tt?.ratio,
    f._sci?.tt?.drift,
  ].filter(Boolean);
  const stale = records.filter((r) => !resultFreshness(f, r, r.analysis?.descriptor.detector || f.detector).fresh);
  return stale.length
    ? warn(`${stale.length} scientific result(s) are historical or unverified. Rerun their tools for this analysis.`)
    : null;
}
const spin = () => h('span', { class: 'spinner' });
const card = (...kids) => h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:8px' }, ...kids);
const title = (t, right) =>
  h('div', { class: 'row' }, h('span', { style: 'font-weight:600' }, t), h('span', { class: 'sp' }), right || null);
const note = (t, style = '') => h('div', { class: 'note', style }, t);
const warn = (t) => note(t, 'color:var(--warn)');
const th = (...c) =>
  h(
    'tr',
    {},
    c.map((x) => h('th', {}, x))
  );
const td = (...c) =>
  h(
    'tr',
    {},
    c.map((x, i) => h('td', i === 0 ? { class: 'txt' } : {}, x))
  );

// Small Plotly plot inside an inspector card (drawn after the card is attached).
function miniPlot(height, data, layout) {
  const div = h('div', { style: `height:${height}px;background:#fff;border-radius:4px` });
  setTimeout(() => {
    if (!div.isConnected || typeof Plotly === 'undefined') return;
    Plotly.newPlot(
      div,
      data,
      {
        margin: { l: 56, r: 10, t: 8, b: 44 },
        font: { family: 'Arial', size: 11, color: '#1D2330' },
        paper_bgcolor: '#fff',
        plot_bgcolor: '#fff',
        showlegend: false,
        ...layout,
        xaxis: { showline: true, mirror: true, ticks: 'outside', automargin: true, ...(layout.xaxis || {}) },
        yaxis: { showline: true, mirror: true, ticks: 'outside', automargin: true, exponentformat: 'SI', ...(layout.yaxis || {}) },
      },
      { displayModeBar: false, responsive: true }
    );
  }, 40);
  return div;
}

// Run an async action with a busy flag on f._sci[key]; re-render the inspector before/after.
async function run(f, key, fn) {
  const s = sci(f);
  if (s[key + 'Busy']) return;
  s[key + 'Busy'] = true;
  emit('insp');
  const inputs =
    key === 'nv'
      ? { fit_D: s.nv?.fitD, fit_E: s.nv?.fitE, text: s.nv?.text }
      : key === 'deerRes'
        ? { B: s.deer?.B, fGHz: s.deer?.fGHz }
        : key === 'boot'
          ? { n_boot: s.nBoot ?? 200, parentSignature: f.fit[f.detector]?.analysis?.signature }
          : {};
  const analysis = envelopeFor(f, f.detector, { tool: key, inputs }, key);
  try {
    await fn();
    const bind = (r) => {
      if (r && typeof r === 'object') {
        r.analysis ||= analysis;
        recordResult(f, r, { kind: key });
      }
    };
    if (key === 'boot') bind(f.boot?.[analysis.descriptor.detector]?.res);
    else if (key === 'deer') bind(f.deerRatio?.res);
    else if (key === 'nv') bind(s.nv?.res);
    else if (key === 'deerRes') bind(s.deer?.res);
    else if (s.tt) bind(s.tt[key]);
  } catch (e) {
    s[key + 'Err'] = e.message;
  } finally {
    s[key + 'Busy'] = false;
    emit('insp');
  }
}

// ------------------------------------------------------------------ 1. SEM weighting + reduced chi-squared
export function semAllCheck(f, set) {
  return check('Weight all models by iteration SEM', !!f.fitSpec.sem_weights_all, (v) => set({ sem_weights_all: v }),
    { title: 'Weighted least squares with the per-point SEM of the iteration mean (all models except Rabi, which has its own switch above). Needs per-iteration data in the file; also enables reduced χ².' });
}

// kv rows for the result card (empty when the fit has no reduced chi-squared).
export function chi2Rows(fit) {
  const c = fit?.extras?.reduced_chi2;
  if (c === undefined || c === null || !Number.isFinite(c)) return [];
  return [h('span', { class: 'k', title: '≈ 1 good; ≫ 1 the model misses structure or the SEM underestimates the noise; ≪ 1 the SEM overestimates the noise or the fit is over-parameterised.' }, 'reduced χ²'),
    h('span', { class: 'v' }, fmt(c, 4))];
}
export function chi2Note(fit) {
  const c = fit?.extras?.reduced_chi2;
  if (c === undefined || c === null || !Number.isFinite(c)) return null;
  return note(`reduced χ² = ${fmt(c, 3)}: ≈ 1 is good; ≫ 1 means the model misses structure or the iteration SEM underestimates the noise.`
    + (c > 3 ? ' Analytic ± errors are probably optimistic here.' : ''));
}

// ------------------------------------------------------------------ 2. Bootstrap
export function bootstrapCard(f, fit, getSpec) {
  const s = sci(f);
  const det = f.detector;
  f.boot = f.boot || {};
  const b = f.boot[det];
  const busy = !!s.bootBusy;
  const n = s.nBoot ?? 200;
  const go = () => run(f, 'boot', async () => {
    s.bootErr = null;
    const spec = getSpec();
    const res = await call('bootstrap', { path: f.path, detector: det, mode: f.mode, proc: f.proc, spec, n_boot: Math.max(10, Math.round(s.nBoot ?? 200)) }, { label: 'Bootstrap' });
    f.boot[det] = { res, parentSignature:fit.analysis?.signature, n_boot: res.n_boot };
  });
  const out = card(title('Bootstrap errors', null),
    h('div', { class: 'row' }, field('Resamples', input(n, (v) => { s.nBoot = Math.max(10, Math.round(num(v) || 200)); emit('insp'); }, { numeric: true, style: 'width:70px' })),
      h('button', { class: 'btn', style: 'flex:1;justify-content:center;margin-top:16px', disabled: busy || f.fitting, onclick: go }, busy ? spin() : null, busy ? 'Running…' : 'Bootstrap errors')),
    note('Refits resampled data (multistart ≤ 4 for speed). Can take a minute or more at 200 resamples; try 30 first.'));
  if (s.bootErr) out.append(note(s.bootErr, 'color:var(--bad)'));
  if (b) {
    const r = b.res;
    const units = Object.fromEntries(fit.params.map((p) => [p.name, p.unit && p.unit !== '-' ? p.unit : '']));
    const method = r.method === 'iterations' ? 'iteration resampling (whole iterations drawn with replacement, then averaged)' : 'residual resampling (fit residuals redrawn; the per-point SEM stays fixed)';
    if (b.parentSignature !== fit.analysis?.signature || !resultFreshness(f,r,det).fresh) out.append(warn('Historical bootstrap result; rerun after changing the analysis.'));
    out.append(h('table', { class: 'tbl' }, th('Parameter', 'Value', 'Analytic ±', 'Bootstrap σ', '16–84 %'),
      Object.entries(r.params || {}).map(([name, p]) => td(`${name}${units[name] ? ` (${units[name]})` : ''}`, fmt(p.value, 5), p.analytic_error != null ? fmt(p.analytic_error, 2) : 'n/a',
        Number.isFinite(p.std) ? fmt(p.std, 2) : 'n/a', Number.isFinite(p.p16) ? `${fmt(p.p16, 4)} … ${fmt(p.p84, 4)}` : 'n/a'))));
    out.append(note(`Method: ${method}. Successful refits: ${r.n_success} / ${r.n_boot}. Model: ${r.model}.`));
    if (r.n_success < 0.8 * r.n_boot) out.append(warn('Many refits failed; bootstrap errors are unreliable.'));
    out.append(note('Bootstrap σ is the standard deviation of refit values (the 16–84 % interval is the percentile range). Both include only noise that resampling can reproduce; neither covers systematic model error.'));
  }
  return out;
}

// ------------------------------------------------------------------ 3. NV field from ODMR lines
const f5 = (v) => (Number.isFinite(v) ? v.toFixed(5) : '–');
const pm = (v, e, unit = '') => `${e != null && Number.isFinite(e) ? fmtPm(v, e) : fmt(v, 5)}${unit ? ' ' + unit : ''}`;

export function nvFieldCard(f, getSpec) {
  const s = sci(f);
  s.nv = s.nv || { fitD: false, fitE: false, text: '' };
  const nv = s.nv;
  const busy = !!s.nvBusy;
  const xu = String(f.info.x_unit || '').trim();
  const unitKey = { ghz: 'GHz', mhz: 'MHz', hz: 'Hz' }[xu.toLowerCase()] || null;
  const fromFit = () => run(f, 'nv', async () => {
    s.nvErr = null;
    const spec = { ...getSpec(), fit_D: nv.fitD, fit_E: nv.fitE, odmr_multi: true };
    if (unitKey) spec.x_unit = unitKey;
    const r = await call('nv_field_from_fit', { path: f.path, detector: f.detector, mode: f.mode, proc: f.proc, spec }, { label: 'NV field' });
    nv.res = {...r.field,analysis:r.analysis}; nv.source = 'fit'; nv.centres = r.centres_ghz; nv.centreErr = r.centre_errors_ghz;
  });
  const manual = () => run(f, 'nv', async () => {
    s.nvErr = null;
    const freqs = nv.text.split(/[,;\s]+/).map((t) => Number(t)).filter((v) => Number.isFinite(v));
    if (!freqs.length) throw new Error('Enter resonance frequencies in GHz, e.g. 2.80, 2.94.');
    nv.res = await call('nv_field', { freqs_ghz: freqs, fit_D: nv.fitD, fit_E: nv.fitE }, { label: 'NV field' });
    nv.source = 'manual'; nv.centres = freqs; nv.centreErr = null;
  });
  const txt = h('input', { class: 'field', value: nv.text, placeholder: 'e.g. 2.80, 2.94', style: 'flex:1' });
  txt.addEventListener('input', () => { nv.text = txt.value; });
  txt.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') manual(); });
  const out = card(title('NV magnetic field (from ODMR lines)'),
    note('Spin-1 NV ground state, γe = 28.024 GHz/T. D = 2.870 GHz and E = 0 are assumed unless fitted below. 13C/14N hyperfine structure is ignored. Lines come from the multi-Lorentzian fit; they are resonance frequencies in GHz.'),
    h('div', { class: 'row wrap' }, check('Fit D', nv.fitD, (v) => { nv.fitD = v; emit('insp'); }, { title: 'Needs at least as many lines as fitted parameters (ensemble: 4+ lines).' }),
      check('Fit E', nv.fitE, (v) => { nv.fitE = v; emit('insp'); }, { title: 'Strain splitting in each NV frame; ensemble fits only, indicative.' })),
    h('button', { class: 'btn', style: 'justify-content:center', disabled: busy, onclick: fromFit, title: 'Runs the ODMR multi-peak fit with your current Fit settings, then converts the fitted centres.' }, busy ? spin() : null, 'From fitted centres'),
    unitKey ? null : warn(`The file's x unit is "${xu || 'unknown'}"; the unit will be inferred from the size of the centres (reported in the notes).`),
    h('div', { class: 'row' }, txt, h('button', { class: 'btn', disabled: busy, onclick: manual }, 'Compute from lines (GHz)')));
  if (s.nvErr) out.append(note(s.nvErr, 'color:var(--bad)'));
  const r = nv.res;
  if (!r) return out;
  const estimateOnly = r.mode === 'quick' || r.mode === 'single' || !r.success || (r.dof ?? 0) <= 0;
  out.append(h('div', { class: 'note', style: 'border-top:1px solid #222731;padding-top:8px' }, `Result (${nv.source === 'fit' ? 'fitted centres' : 'manual lines'}) · mode: ${
    { quick: 'quick linear estimate (1 line)', single: 'single NV orientation (2 lines)', ensemble: 'ensemble of NV orientations' }[r.mode] || r.mode}`));
  if (r.mode === 'quick') out.append(warn('Estimate only: one line cannot determine the field. This is the small-field |B∥| ≈ (f − D)/γe assuming the stated D.'));
  else if (r.mode === 'single') out.append(warn('One NV orientation (two lines): only |B| and the angle to that NV axis are determined, not the direction in the crystal. Treat as an on-axis estimate if B⊥ is small.'));
  if (!r.success) out.append(warn('The solver did not reach an exact solution; lines may be inconsistent with this model (check D, E and which lines belong together).'));
  const kv = [['|B|', `${pm(r.B_mT, r.B_mT_err, 'mT')}`], ['|B|', `${pm(r.B_gauss, r.B_mT_err != null ? r.B_mT_err * 10 : null, 'G')}`]];
  if (r.mode === 'single') {
    if (r.angle_to_axis_deg != null) kv.push(['angle to NV axis', pm(r.angle_to_axis_deg, r.angle_to_axis_deg_err, '°')]);
    if (r.B_par_T != null) kv.push(['B∥ / B⊥', `${fmt(r.B_par_T * 1e3, 5)} / ${fmt((r.B_perp_T ?? 0) * 1e3, 5)} mT`]);
    if (r.on_axis_estimate) kv.push(['on-axis estimate B∥', `${fmt(r.on_axis_estimate.B_par_T * 1e3, 5)} mT (B⊥ = 0)`]);
  } else if (r.mode === 'ensemble') {
    kv.push(['θ (from crystal z)', pm(r.theta_deg, r.theta_deg_err, '°')], ['φ (from crystal x toward y)', pm(r.phi_deg, r.phi_deg_err, '°')]);
  }
  kv.push(['D used', `${f5(r.D_ghz)} GHz ${nv.fitD && r.mode === 'ensemble' ? '(fitted)' : '(assumed)'}`], ['E used', `${f5(r.E_ghz)} GHz ${nv.fitE && r.mode === 'ensemble' ? '(fitted)' : '(assumed)'}`]);
  if (r.mode !== 'quick') kv.push(['rms residual', `${fmt(r.rms_mhz, 3)} MHz`]);
  out.append(h('div', { class: 'kv' }, kv.flatMap(([k, v]) => [h('span', { class: 'k' }, k), h('span', { class: 'v' }, v)])));
  if ((r.lines_measured_ghz || []).length) {
    const pred = r.lines_predicted_ghz || [];
    out.append(h('table', { class: 'tbl' }, th('Line', 'Measured (GHz)', 'Predicted (GHz)', 'Δ (MHz)'),
      r.lines_measured_ghz.map((m, i) => td(String(i + 1), f5(m), pred[i] != null ? f5(pred[i]) : '–', r.residuals_ghz?.[i] != null ? fmt(r.residuals_ghz[i] * 1e3, 3) : '–'))));
  }
  if (r.errors_basis) out.append(/NOT/.test(r.errors_basis) ? warn(`Error basis: ${r.errors_basis}`) : note(`Error basis: ${r.errors_basis}`));
  if (estimateOnly && r.mode === 'ensemble') out.append(warn('No spare degrees of freedom: the error bars are not a statistical test of the model.'));
  if (r.mode === 'ensemble') out.append(note('Degeneracy: B and −B, and (for E = 0) permutations/sign flips of the crystal axes, give identical spectra. Angles are reported in the fundamental domain |Bx| ≥ |By| ≥ |Bz|, so the crystal direction is not unique from lines alone.'));
  if ((r.alternatives || []).length) out.append(note(`${r.alternatives.length} other solution(s) also fit; see Notes in the details below.`));
  if ((r.notes || []).length) out.append(h('details', {}, h('summary', {}, `Notes (${r.notes.length})`), r.notes.map((t) => note(t, 'margin-top:4px'))));
  return out;
}

// ------------------------------------------------------------------ 4. DEER reference + resonance helper
export function deerCard(f) {
  const s = sci(f);
  s.deer = s.deer || { B: '', fGHz: '' };
  const others = Object.values(S.files).filter((o) => o.path !== f.path && o.info.dim !== 'scan2d' && o.info.type !== 'TimeTrace');
  const isDeer = String(f.info.type || '').startsWith('DEER');
  const busy = !!s.deerBusy;
  const pick = (path) => {
    if (!path) { f.deerRatio = null; emit('insp'); return; }
    run(f, 'deer', async () => {
      s.deerErr = null; f.deerRatio = null;
      const res = await call('deer_ratio', { path: f.path, detector: f.detector, mode: f.mode, ref_path: path, ref_detector: f.detector }, { label: 'DEER ratio' });
      f.deerRatio = { refPath: path, res, det: res.analysis?.descriptor.detector || f.detector };
    });
  };
  const dr = f.deerRatio;
  const body = h('div', { style: 'display:flex;flex-direction:column;gap:8px;margin-top:8px' },
    field('Divide by reference echo', select([['', 'None'], ...others.map((o) => [o.path, o.info.name])], dr?.refPath || '', pick)),
    others.length ? null : note('Open the reference file (same x axis, no pulses on the DEER frequency) so that it appears in the Library list.'),
    busy ? h('div', { class: 'row' }, spin(), note('Computing…')) : null,
    s.deerErr ? note(s.deerErr, 'color:var(--bad)') : null);
  if (dr) {
    const r = dr.res;
    const xl = `${f.info.x_label || 'x'}${f.info.x_unit ? ` [${f.info.x_unit}]` : ''}`;
    const data = [{ type: 'scatter', mode: 'lines+markers', x: r.x, y: r.ratio, marker: { size: 4, color: '#1F5FBF' }, line: { width: 1, color: '#1F5FBF' },
      error_y: r.sem ? { type: 'data', array: r.sem, visible: true, thickness: 1, width: 0, color: 'rgba(31,95,191,0.45)' } : undefined }];
    body.append(miniPlot(220, data, { xaxis: { title: { text: xl } }, yaxis: { title: { text: 'DEER / reference (dimensionless)' } } }),
      note(`Ratio y_DEER / y_ref per point (${dr.res.x.length} points). Error bars: first-order propagation of both SEMs, independent errors. ${r.interpolated ? 'Reference was interpolated onto the DEER grid.' : 'Reference on the identical grid.'}${r.n_dropped ? ` ${r.n_dropped} point(s) dropped.` : ''}`),
      r.note ? note(r.note) : null,
      note('This ratio is shown here only; it is not applied to the fit or the main plot.'),
      h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => { f.deerRatio = null; emit('insp'); } }, 'Clear ratio')));
  }
  // resonance helper
  const res = s.deer.res;
  const rc = () => run(f, 'deerRes', async () => {
    const B = num(s.deer.B), fg = num(s.deer.fGHz);
    if (B === null && fg === null) throw new Error('Enter a field (mT) or a frequency (GHz).');
    s.deer.res = await call('deer_resonance', { B_T: B === null ? null : B / 1000, f_GHz: fg }, { label: 'DEER resonance' });
  });
  body.append(h('div', { class: 'f', style: 'border-top:1px solid #222731;padding-top:8px' }, h('span', { class: 'lbl' }, 'Resonance helper (free electron spin, g ≈ 2.0023)'),
    h('div', { class: 'g2' }, field('B (mT)', input(s.deer.B, (v) => { s.deer.B = v; emit('insp'); })), field('f (GHz)', input(s.deer.fGHz, (v) => { s.deer.fGHz = v; emit('insp'); }))),
    h('button', { class: 'btn sm', onclick: rc }, 'Compute'),
    s.deerResErr ? note(s.deerResErr, 'color:var(--bad)') : null,
    res ? h('div', { class: 'kv' }, res.f_GHz != null ? [h('span', { class: 'k' }, `f at B = ${fmt(res.B_T * 1e3, 5)} mT`), h('span', { class: 'v' }, `${fmt(res.f_GHz, 6)} GHz`)] : null,
      res.B_T_from_f != null ? [h('span', { class: 'k' }, `B at f = ${fmt(res.f_in_GHz, 6)} GHz`), h('span', { class: 'v' }, `${fmt(res.B_T_from_f * 1e3, 6)} mT`)] : null,
      h('span', { class: 'k' }, 'g · μB/h'), h('span', { class: 'v' }, `${fmt(res.g, 5)} · ${fmt(res.mu_B_over_h_GHz_per_T, 6)} GHz/T`)) : null,
    note('f = g·μB·B/h for a bare electron spin. Hyperfine, g-shifts and the NV transition itself are not included; compare with a measured DEER dip as a sanity check, not as proof of the species.')));
  return h('details', { class: 'card pad', open: isDeer || !!dr },
    h('summary', {}, `Reference file${isDeer ? ' · DEER normalisation' : ' (DEER)'}`), body);
}

// ------------------------------------------------------------------ 5. Time-trace tools
const fmtSec = (v) => `${fmt(v, 5)} s`;

function syncStepMarks(f) {
  const scale = f.tt.unit === 'min' ? 1 / 60 : 1;
  (f.style.shapes || []).forEach((d) => { if (d.stepT !== undefined) d.x = d.stepT * scale; });
}

function markSteps(f, steps) {
  f.style.shapes = (f.style.shapes || []).filter((d) => d.stepT === undefined);
  const scale = f.tt.unit === 'min' ? 1 / 60 : 1;
  steps.forEach((st) => f.style.shapes.push({ kind: 'vline', x: st.t_s * scale, color: '#C0392B', text: 'step', stepT: st.t_s }));
  emit();
}

export function ttSciCard(f) {
  const s = sci(f);
  const t = (s.tt = s.tt || { sigma: 4, minSeg: 5, minPct: 2, driftWin: 'whole' });
  syncStepMarks(f);
  const unit = f.info.unit || 'signal units';
  const out = h('div', { class: 'card pad', style: 'display:flex;flex-direction:column;gap:10px' }, title('Time-trace analysis'));

  // --- steps
  const stepsBusy = !!s.stepsBusy;
  const detect = () => run(f, 'steps', async () => {
    s.stepsErr = null;
    t.steps = await call('tt_steps', { path: f.path, detector: f.detector, mode: f.mode, opts: { min_size_sigma: t.sigma, min_segment: Math.max(2, Math.round(t.minSeg)) } }, { label: 'Detect steps' });
    markSteps(f, filteredSteps(t));
  });
  const nMarked = (f.style.shapes || []).filter((d) => d.stepT !== undefined).length;
  const steps = h('div', { class: 'f' }, h('span', { class: 'lbl' }, `Step detection · ${detLabel(f, f.detector)}`),
    h('div', { class: 'g2' }, field('Min size (σ of noise)', input(t.sigma, (v) => { t.sigma = Math.max(1, num(v) || 4); emit('insp'); }, { numeric: true })),
      field('Min segment (points)', input(t.minSeg, (v) => { t.minSeg = Math.max(2, num(v) || 5); emit('insp'); }, { numeric: true }))),
    field('Min step size (% of level)', input(t.minPct, (v) => { t.minPct = Math.max(0, num(v) ?? 2); if (t.steps) markSteps(f, filteredSteps(t)); emit('insp'); }, { numeric: true, title: 'Client-side filter: hides steps smaller than this fraction of the level before the step.' })),
    h('div', { class: 'row' }, h('button', { class: 'btn', style: 'flex:1;justify-content:center', disabled: stepsBusy, onclick: detect }, stepsBusy ? spin() : null, 'Detect steps'),
      h('button', { class: 'btn', disabled: !nMarked, onclick: () => { f.style.shapes = (f.style.shapes || []).filter((d) => d.stepT === undefined); emit(); } }, 'Clear step marks')),
    s.stepsErr ? note(s.stepsErr, 'color:var(--bad)') : null);
  if (t.steps) {
    const all = t.steps.steps || [];
    const shown = filteredSteps(t);
    steps.append(h('table', { class: 'tbl' }, th('t (s)', `before (${unit})`, `after (${unit})`, 'size', 'size/σ', 'signif.', '% level'),
      shown.map((st) => td(fmt(st.t_s, 5), fmt(st.level_before, 5), fmt(st.level_after, 5), fmt(st.size, 3), fmt(st.size_sigma, 3), fmt(st.significance, 3), `${fmt(100 * Math.abs(st.size / st.level_before), 3)}`))));
    steps.append(note(`${shown.length} of ${all.length} detected steps shown (≥ ${t.minPct} % of level); σ (noise) = ${fmt(t.steps.sigma, 4)} ${unit} from the point-to-point differences, N = ${t.steps.n}. Marked on the main plot in red.`));
    if (!all.length) steps.append(note('No steps found at this threshold.'));
    steps.append(warn('Small steps in a drifting trace may be drift, not events. Check the sign pattern: a staircase of same-sign steps is drift.'));
    steps.append(note('Binary segmentation of a piecewise-constant mean with a white-noise σ; correlated noise makes steps look more significant than they are.'));
  }
  out.append(steps);

  // --- ratio D2/D1
  if (f.info.detectors.length > 1) {
    const rb = !!s.ratioBusy;
    const rgo = () => run(f, 'ratio', async () => { s.ratioErr = null; t.ratio = await call('tt_ratio', { path: f.path, mode: f.mode }, { label: 'Detector ratio' }); });
    const ratio = h('div', { class: 'f', style: 'border-top:1px solid #222731;padding-top:8px' }, h('span', { class: 'lbl' }, 'Detector ratio D2 / D1'),
      h('button', { class: 'btn', disabled: rb, onclick: rgo }, rb ? spin() : null, 'Ratio D2/D1'),
      s.ratioErr ? note(s.ratioErr, 'color:var(--bad)') : null);
    if (t.ratio) {
      const r = t.ratio;
      const fin = r.ratio.map((v, i) => [v, r.sigma?.[i]]).filter(([v]) => Number.isFinite(v));
      const mean = fin.reduce((a, [v]) => a + v, 0) / (fin.length || 1);
      const lo = r.ratio.map((v, i) => v - (r.sigma?.[i] ?? 0)), hi = r.ratio.map((v, i) => v + (r.sigma?.[i] ?? 0));
      const tx = r.t_s;
      ratio.append(miniPlot(200, [
        { type: 'scatter', mode: 'lines', x: tx, y: hi, line: { width: 0 }, hoverinfo: 'skip' },
        { type: 'scatter', mode: 'lines', x: tx, y: lo, line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(31,95,191,0.25)', hoverinfo: 'skip' },
        { type: 'scatter', mode: 'lines', x: tx, y: r.ratio, line: { width: 1, color: '#1F5FBF' } },
        { type: 'scatter', mode: 'lines', x: [tx[0], tx[tx.length - 1]], y: [mean, mean], line: { width: 1.2, color: '#C0392B', dash: 'dash' } },
      ], { xaxis: { title: { text: 'Elapsed time [s]' } }, yaxis: { title: { text: `${r.definition || 'D2/D1'} (dimensionless)` } } }));
      ratio.append(note(`Mean (unweighted) = ${fmt(mean, 5)} (red dashed); band = ±1σ per point. ${r.definition}; ${r.n_matched} matched time stamps (${r.n_exact} exact, ${r.n_nearest} nearest within ${fmt(r.tolerance_s, 3)} s); ${r.n_unmatched_1 + r.n_unmatched_2} unmatched dropped. σ source: ${r.sigma_source}. First-order propagation, independent detector errors.`));
    }
    out.append(ratio);
  }

  // --- drift
  const wa = f.tt.windows.a;
  const db = !!s.driftBusy;
  const dgo = () => run(f, 'drift', async () => {
    s.driftErr = null;
    const window = t.driftWin === 'a' && wa ? wa : null;
    t.drift = { ...(await call('tt_drift', { path: f.path, detector: f.detector, mode: f.mode, window }, { label: 'Drift rate' })), win: window ? 'window A' : 'whole trace' };
  });
  const drift = h('div', { class: 'f', style: 'border-top:1px solid #222731;padding-top:8px' }, h('span', { class: 'lbl' }, `Drift rate · ${detLabel(f, f.detector)}`),
    h('div', { class: 'row' }, seg([['whole', 'Whole trace'], ['a', 'Window A']], t.driftWin, (v) => { t.driftWin = v; emit('insp'); }),
      h('button', { class: 'btn', style: 'flex:1;justify-content:center', disabled: db, onclick: dgo }, db ? spin() : null, 'Drift rate')),
    s.driftErr ? note(s.driftErr, 'color:var(--bad)') : null);
  if (t.drift) {
    const d = t.drift;
    drift.append(h('div', { class: 'kv' },
      h('span', { class: 'k' }, `slope (${unit}/s)`), h('span', { class: 'v' }, fmtPm(d.slope_per_s, d.slope_se_per_s)),
      h('span', { class: 'k' }, 'relative drift (% of mean per min)'), h('span', { class: 'v' }, fmtPm(d.percent_per_min, d.percent_per_min_se)),
      h('span', { class: 'k' }, 'mean over window'), h('span', { class: 'v' }, `${fmt(d.window_mean, 5)} ${unit}`),
      h('span', { class: 'k' }, 'slope / SE'), h('span', { class: 'v' }, fmt(d.z, 3)),
      h('span', { class: 'k' }, 'window'), h('span', { class: 'v' }, `${d.win}: ${fmt(d.t_start_s, 5)}–${fmt(d.t_end_s, 5)} s, N = ${d.n}`)));
    drift.append(note('Ordinary least-squares line; ± is the standard error from the residual scatter (N − 2 dof). It assumes uncorrelated residuals, so it is optimistic for a trace with slow fluctuations. % per min neglects the uncertainty of the mean.'));
  }
  out.append(drift);
  return out;
}

function filteredSteps(t) {
  const all = t.steps?.steps || [];
  return all.filter((st) => !(t.minPct > 0) || !Number.isFinite(st.level_before) || st.level_before === 0 || (100 * Math.abs(st.size / st.level_before)) >= t.minPct);
}
