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

export function maxAbs(values) {
  let max = 0;
  for (const value of values) if (Number.isFinite(value)) max = Math.max(max, Math.abs(value));
  return max;
}
