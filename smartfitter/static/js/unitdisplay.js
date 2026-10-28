// Display-only units. Values and uncertainties always share the same conversion.
const GROUPS = {
  time: { ps: 1e-12, ns: 1e-9, us: 1e-6, ms: 1e-3, s: 1 },
  frequency: { Hz: 1, kHz: 1e3, MHz: 1e6, GHz: 1e9 },
  length: { nm:1e-9, um:1e-6, mm:1e-3, m:1 },
};
export const parameterUnit = (p, info) => p.unit === 'axis units' ? (info.x_unit || p.unit) : p.unit;
export const canonicalUnit = (u) => String(u || '').trim().replace(/[µμ�]/g, 'u');
export function unitGroup(unit) {
  const u = canonicalUnit(unit);
  return Object.keys(GROUPS).find((g) => Object.hasOwn(GROUPS[g], u)) || null;
}
export function unitOptions(unit) {
  const g = unitGroup(unit);
  return [['auto', 'Auto'], ['native', `Native (${unit || 'unitless'})`], ...(g ? Object.keys(GROUPS[g]).map((u) => [u, u]) : [])];
}
export function unitScale(from, to) {
  const g = unitGroup(from);
  if (!g || unitGroup(to) !== g) return canonicalUnit(from) === canonicalUnit(to) ? 1 : null;
  return GROUPS[g][canonicalUnit(from)] / GROUPS[g][canonicalUnit(to)];
}
export function displayUnit(value, native, override = 'auto') {
  const g = unitGroup(native), u = canonicalUnit(native);
  if (!g) return { unit: native || '', scale: 1 };
  if (override === 'native') return { unit: native, scale: 1 };
  if (override && override !== 'auto') {
    const scale = unitScale(native, override);
    if (scale !== null) return { unit: canonicalUnit(override), scale };
  }
  const base = Math.abs(value) * GROUPS[g][u];
  if (!Number.isFinite(base) || base === 0) return { unit: native, scale: 1 };
  const choices = Object.entries(GROUPS[g]);
  const choice = [...choices].reverse().find(([, factor]) => base >= factor) || choices[0];
  return { unit: choice[0], scale: GROUPS[g][u] / choice[1] };
}
export function displayValue(value, error, unit, { units = 'auto', precision = 2, format = 'pm' } = {}) {
  const d = displayUnit(value, unit, units), v = Number.isFinite(value) ? value * d.scale : NaN;
  const e = Number.isFinite(error) && error > 0 ? error * d.scale : null;
  const digits = Math.max(1, Math.min(8, Number(precision) || 2));
  let text = '—';
  if (Number.isFinite(v)) {
    if (format === 'pm' && e) {
      // Round both to the uncertainty's last significant decimal, including carry at 9.99… .
      const rounded = Number(e.toPrecision(digits));
      const places = digits - 1 - Math.floor(Math.log10(rounded));
      if (places >= 0 && places <= 10 && Math.abs(v) < 1e12) text = `${v.toFixed(places)} ± ${e.toFixed(places)}`;
      else if (places < 0 && Math.max(Math.abs(v), rounded) < 1e5) {
        // e.g. 156.2 ± 745.2 -> "160 ± 750": round both to the error's last significant digit, no exponent for small numbers
        const q = 10 ** -places;
        text = `${(Math.round(v / q) * q).toFixed(0)} ± ${(Math.round(rounded / q) * q).toFixed(0)}`;
      } else {
        const power = Math.floor(Math.log10(Math.max(Math.abs(v), e)));
        const decimals = Math.max(0, Math.min(10, places + power));
        text = `(${(v / 10 ** power).toFixed(decimals)} ± ${(e / 10 ** power).toFixed(decimals)}) × 10^${power}`;
      }
    } else text = Number(v.toPrecision(format === 'sig4' || format === 'pm' ? Math.max(4, digits) : digits)).toString();
  }
  return { value: v, error: e, unit: d.unit, scale: d.scale, text, withUnit: `${text}${d.unit ? ` ${prettyUnit(d.unit)}` : ''}` };
}
// Display-only unit spelling (canonical "us" stays the lookup key): us -> µs, um -> µm, and the loader's mis-decoded
// micro sign (U+FFFD, e.g. "�m" from MATLAB stage units) -> µ.
export function prettyUnit(u) {
  return String(u ?? '').replace(/�/g, 'µ').replace(/^u(s|m)\b/, 'µ$1');
}
// Presentation coordinates (zoom, ticks, decorations) follow a unit change too.
export function resolveXAxis(f, files = [f]) {
  const d = xDisplay(f, files), old = f.display?.resolvedXUnit ?? (f.info.x_unit || '');
  const ratio = unitScale(old, d.unit);
  if (ratio !== null && ratio !== 1) {
    const scale = (obj, keys) => { if (obj) for (const key of keys) if (Number.isFinite(obj[key])) obj[key] *= ratio; };
    const axis = f.style?.x;
    scale(axis, ['min', 'max']);
    if (!axis?.log) scale(axis, ['dtick', 'tick0']); // Log spacing/origin are decades, not physical coordinates.
    axis?.ticks?.forEach((t) => scale(t, ['value']));
    scale(f.style?.inset, ['x0', 'x1']); scale(f.style?.xbreak, ['from', 'to']);
    f.style?.shapes?.forEach((s) => scale(s, ['x', 'x0', 'x1']));
    f.pins?.forEach((p) => scale(p, ['x'])); f.measure?.forEach((p) => scale(p, ['x']));
    if (f.xrange) f.xrange = f.xrange.map((v) => v * ratio);
  }
  f.display = { ...f.display, resolvedXUnit: d.unit };
  return d;
}

export function xDisplay(f, files = [f]) {
  const native = f.info.x_unit || '', d = f.display || {};
  const xs = Number.isFinite(d.xscale) ? d.xscale : 1, xo = Number.isFinite(d.xoffset) ? d.xoffset : 0;
  let magnitude = 0;
  for (const o of files) {
    const scale = unitScale(o.info.x_unit || '', native);
    if (scale === null) continue;
    const det = o.info.detectors?.includes(f.detector) ? f.detector : o.detector;
    for (const v of o.data[det]?.proc?.x || []) if (Number.isFinite(v)) magnitude = Math.max(magnitude, Math.abs(v * scale * xs + xo));
  }
  const selected = displayUnit(magnitude, native, d.xunit || 'auto');
  return { ...selected, calibrated: xs !== 1 || xo !== 0,
    transform: (values, sourceUnit = native) => {
      const scale = unitScale(sourceUnit, native);
      return scale === null ? null : values.map((v) => (v * scale * xs + xo) * selected.scale);
    },
    inverse: (v) => (v / selected.scale - xo) / xs,
  };
}
