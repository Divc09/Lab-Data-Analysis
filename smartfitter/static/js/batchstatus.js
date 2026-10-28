export const BATCH_STATUSES = ['PASS', 'WARN', 'FAIL', 'PLOT_ONLY', 'UNCONVERGED', 'FALLBACK', 'EXCLUDED'];
export const needsBatchRetry = (row) => ['FAIL', 'UNCONVERGED', 'FALLBACK'].includes(row.status);

// Older CSVs may label a plot-only, fallback or unsuccessful fit as PASS.
export function comparableBatchRow(row) {
  return ['PASS', 'WARN'].includes(row.status) && !!row.model && row.model.toLowerCase() !== 'none'
    && !['false', '0'].includes(String(row.success).toLowerCase())
    && !String(row.model_note || '').startsWith('fallback_after_error:');
}

export function batchTableText(rows, sep = ',') {
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const quote = (v) => {
    const text = v == null ? '' : String(v);
    return /["\r\n]/.test(text) || text.includes(sep) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [keys.map(quote).join(sep), ...rows.map((r) => keys.map((k) => quote(r[k])).join(sep))].join('\n');
}
