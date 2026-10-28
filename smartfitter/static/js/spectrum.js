// FFT requires the actual processed grid to be uniform, including after exclusions/binning.
export function uniformSampling(x = [], y = []) {
  if (x.length !== y.length || x.length < 9) return { ok: false, reason: 'at least 9 paired samples required' };
  if (!x.every(Number.isFinite) || !y.every(Number.isFinite)) return { ok: false, reason: 'finite x and y samples required' };
  const dx = (x[x.length - 1] - x[0]) / (x.length - 1);
  if (!Number.isFinite(dx) || dx === 0) return { ok: false, reason: 'distinct, uniformly spaced x values required' };
  for (let i = 1; i < x.length; i++) {
    const step = x[i] - x[i - 1];
    const tol = Math.abs(dx) * 1e-6 + Math.max(Math.abs(x[i]), Math.abs(x[i - 1])) * Number.EPSILON * 8;
    if (step * dx <= 0 || Math.abs(step - dx) > tol) return { ok: false, reason: 'processed x spacing is irregular' };
  }
  return { ok: true, dx };
}

// Zero-pad to a power of two; keep the actual sample spacing and sample count.
// Iterative radix-2 FFT avoids both quadratic work and the former 2048-sample aliasing.
export function magnitudeSpectrum(x, y) {
  const sampling = uniformSampling(x, y);
  if (!sampling.ok) return null;
  const n = y.length;
  let N = 1;
  while (N < n) N *= 2;
  const re = new Float64Array(N), im = new Float64Array(N);
  const mean = y.reduce((sum, v) => sum + v / n, 0);
  for (let i = 0; i < n; i++) re[i] = y[i] - mean;
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; }
  }
  for (let size = 2; size <= N; size *= 2) {
    const angle = -2 * Math.PI / size, wr = Math.cos(angle), wi = Math.sin(angle);
    for (let start = 0; start < N; start += size) {
      let ur = 1, ui = 0;
      for (let j = 0; j < size / 2; j++) {
        const a = start + j, b = a + size / 2;
        const vr = re[b] * ur - im[b] * ui, vi = re[b] * ui + im[b] * ur;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
        [ur, ui] = [ur * wr - ui * wi, ur * wi + ui * wr];
      }
    }
  }
  const f = [], mag = [];
  let best = 0, peak = null;
  for (let k = 1; k <= N / 2; k++) {
    const freq = k / (N * Math.abs(sampling.dx)), value = Math.hypot(re[k], im[k]) / n;
    f.push(freq); mag.push(value);
    if (value > best) { best = value; peak = freq; }
  }
  return { f, mag, peak };
}
