// JSON API client. Every call shows in the status bar while it runs.
import { toast } from './util.js';

let busy = 0;
const listeners = new Set();
export const onBusy = (fn) => listeners.add(fn);
const notify = (label) => listeners.forEach((fn) => fn(busy, label));

export async function call(name, args = {}, { quiet = false, label = '' } = {}) {
  busy++; notify(label || name);
  try {
    const res = await fetch(`/api/${name}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(args) });
    const body = await res.json();
    if (!body.ok) {
      const err = new Error(body.error || 'Request failed');
      err.trace = body.trace;
      throw err;
    }
    return body.result;
  } catch (err) {
    if (!quiet) toast(`${label || name}: ${err.message}`, 'err');
    console.error(name, err, err.trace || '');
    throw err;
  } finally {
    busy--; notify('');
  }
}
