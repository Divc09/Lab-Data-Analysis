// JSON API client. Every call shows in the status bar while it runs.
import { reportError } from './util.js';

const sessionToken = (() => {
  if (typeof window === 'undefined') return '';
  const supplied = new URLSearchParams(window.location.hash.slice(1)).get('token') || '';
  try {
    const key = `smartfitter-token:${window.location.port}`;
    if (supplied) {
      window.sessionStorage.setItem(key, supplied);
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    }
    return supplied || window.sessionStorage.getItem(key) || '';
  } catch {
    return supplied;
  }
})();

let busy = 0;
const listeners = new Set();
export const onBusy = (fn) => listeners.add(fn);
const notify = (label) => listeners.forEach((fn) => fn(busy, label));

// A failed call is reported in the persistent error tray unless quiet; `retry` redoes the caller's whole action.
export async function call(name, args = {}, { quiet = false, label = '', retry = null } = {}) {
  busy++;
  notify(label || name);
  try {
    const res = await fetch(`/api/${name}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-SmartFitter-Token': sessionToken },
      body: JSON.stringify(args),
    });
    const body = await res.json();
    if (!body.ok) {
      const err = new Error(body.error || 'Request failed');
      err.trace = body.trace;
      throw err;
    }
    return body.result;
  } catch (err) {
    if (!quiet) reportError({ title: label || name, message: err.message, trace: err.trace || '', context: { call: name, path: args.path, detector: args.detector, mode: args.mode }, retry });
    console.error(name, err, err.trace || '');
    throw err;
  } finally {
    busy--;
    notify('');
  }
}
