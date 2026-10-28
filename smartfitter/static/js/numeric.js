import { unitScale } from './unitdisplay.js';
// Iterate over data rather than passing each sample as a function argument.
// Missing/nonfinite values must not become zero-valued samples.
export function finiteExtent(values, fallback = [NaN, NaN]) {
  let lo = Infinity, hi = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  return lo === Infinity ? [...fallback] : [lo, hi];
}

// Parse a typed number. A unit suffix is converted into the field's native unit when the two are
// compatible ("1 ms" -> 1e6 in a ns field); anything else is an error, never a silently stripped number.
// Returns { value } on success (value null for an empty field), optionally with a conversion note,
// or { error } with a message for the user.
export function parseNumber(text, unit = '') {
  const t = String(text ?? '').trim().replace(/−/g, '-');
  if (!t) return { value: null };
  const m = /^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]*)$/.exec(t);
  if (!m) return { error: `"${t}" is not a number` };
  const value = Number(m[1]), suffix = m[2];
  if (!suffix) return { value };
  if (!unit) return { error: `Units cannot be converted here; enter a plain number` };
  const scale = unitScale(suffix, unit);
  if (scale === null) return { error: `"${suffix}" is not compatible with this field's unit (${unit})` };
  const converted = Number((value * scale).toPrecision(12)); // drop binary-scaling noise (2.5 us -> 2500 ns, not 2499.9999...)
  return { value: converted, note: `${value} ${suffix} = ${converted} ${unit}` };
}

export function maxAbs(values) {
  let max = 0;
  for (const value of values) if (Number.isFinite(value)) max = Math.max(max, Math.abs(value));
  return max;
}
