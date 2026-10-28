// JSON API client. Every call shows in the status bar while it runs.
import { toast } from './util.js';

const sessionToken = (() => {
  if(typeof window==='undefined')return '';
  const supplied=new URLSearchParams(window.location.hash.slice(1)).get('token') || '';
  try {
    const key=`smartfitter-token:${window.location.port}`;
    if(supplied){window.sessionStorage.setItem(key,supplied);window.history.replaceState(null,'',window.location.pathname+window.location.search);}
    return supplied || window.sessionStorage.getItem(key) || '';
  } catch {return supplied;}
})();

let busy = 0;
const listeners = new Set();
export const onBusy = (fn) => listeners.add(fn);
const notify = (label) => listeners.forEach((fn) => fn(busy, label));

export async function call(name, args = {}, { quiet = false, label = '' } = {}) {
  busy++; notify(label || name);
  try {
    const res = await fetch(`/api/${name}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-SmartFitter-Token':sessionToken }, body: JSON.stringify(args) });
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
