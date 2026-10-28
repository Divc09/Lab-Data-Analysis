// Canonical numeric encoding is IEEE-754, matching nvfit.analysis_identity.
export function canonical(value) {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return 'null';
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value || 0, false);
    return 'n:' + [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
  }
  if (typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  return (
    '{' +
    Object.keys(value)
      .filter((k) => value[k] !== undefined)
      .sort()
      .map((k) => JSON.stringify(k) + ':' + canonical(value[k]))
      .join(',') +
    '}'
  );
}
export const sourcePath = (path) => {
  const p = String(path).replace(/\\/g, '/');
  return /^[A-Za-z]:/.test(p) ? p.toLowerCase() : p;
};
export const analysisDescriptor = (path, detector, mode, processing = {}) => ({
  source: sourcePath(path),
  detector,
  observable: mode,
  processing,
});
export const analysisId = (descriptor) => 'analysis:' + canonical(descriptor);
export function fitSettings(settings = {}) {
  const out = { ...settings };
  delete out.apply_both;
  if (out.model === 'CustomModel' && (out.custom_params?.length || !out.custom_spec)) {
    const params = out.custom_params || [];
    out.custom_spec = {
      name: 'CustomModel',
      expression: out.custom_expr,
      param_names: params.map((p) => p.name),
      lower: params.map((p) => p.min ?? -1e300),
      upper: params.map((p) => p.max ?? 1e300),
      seed: params.map((p) => p.seed ?? 1),
    };
  }
  return out;
}
export function scientificRevision(value) {
  if (Array.isArray(value)) return value.map(scientificRevision);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.entries(value)
        .filter(([k]) => k !== 'mtime_ns')
        .map(([k, v]) => [k, scientificRevision(v)])
    );
  return value;
}
export function resultEnvelope(descriptor, revision, settings = {}, dependencies = {}, kind = 'fit') {
  return {
    version: 1,
    id: analysisId(descriptor),
    kind,
    descriptor,
    source_revision: revision,
    dependencies,
    settings,
    signature: canonical({
      analysis: descriptor,
      revision: scientificRevision(revision),
      dependencies: scientificRevision(dependencies),
      settings,
      kind,
    }),
  };
}
export const sameAnalysis = (a, b) => !!a && !!b && a.signature === b.signature;

export function remapReferences(value, moved, parent = '') {
  const mapping = Object.fromEntries(Object.entries(moved || {}).map(([a, b]) => [sourcePath(a), b]));
  const path = (v) => (typeof v === 'string' ? mapping[sourcePath(v)] || v : v);
  const signatures = new Map();
  const walk = (v, key = '') => {
    if (!v || typeof v !== 'object') return ['path', 'source', 'refPath', 'ref_path', 'source_path'].includes(key) ? path(v) : v;
    if (Array.isArray(v))
      return v.map((x) => (typeof x === 'string' && ['paths', 'files', 'overlays', 'sel'].includes(key) ? path(x) : walk(x)));
    const envelope = v.descriptor && v.signature && v.kind;
    const valid =
      envelope &&
      v.version === 1 &&
      resultEnvelope(v.descriptor, v.source_revision, v.settings, v.dependencies, v.kind).signature === v.signature;
    const out = Object.fromEntries(
      Object.entries(v).map(([k, x]) => [['dependencies', 'sources'].includes(key) ? path(k) : k, walk(x, k)])
    );
    if (envelope) {
      out.descriptor.source = sourcePath(out.descriptor.source);
      if (out.descriptor.source === 'aggregate')
        out.source_revision = Object.fromEntries(Object.entries(out.source_revision || {}).map(([k, x]) => [sourcePath(path(k)), x]));
      out.dependencies = Object.fromEntries(Object.entries(out.dependencies || {}).map(([k, x]) => [sourcePath(k), x]));
      const env = resultEnvelope(out.descriptor, out.source_revision, out.settings, out.dependencies, out.kind);
      Object.assign(out, env, { unverified: !valid || !!v.unverified });
      signatures.set(v.signature, env.signature);
    }
    return out;
  };
  const out = walk(value, parent);
  const parents = (v) => {
    if (v && typeof v === 'object')
      for (const [k, x] of Object.entries(v)) {
        if (k === 'parentSignature' && signatures.has(x)) v[k] = signatures.get(x);
        else parents(x);
      }
  };
  parents(out);
  return out;
}

export function persistentValue(value) {
  if (Array.isArray(value)) return value.map(persistentValue);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([k, v]) => !/(Busy|Err)$/.test(k) && !['busy', 'fitRef', 'raw'].includes(k) && typeof v !== 'function')
      .map(([k, v]) => [k, persistentValue(v)])
  );
}
