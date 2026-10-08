// Figure rendering for export/clipboard: the same Plotly figure, rendered at the
// configured physical size so the exported image matches what is on screen.
import { call } from './api.js';
import { toast } from './util.js';

export async function renderImage(fig, { width_in = 8.5, height_in = 5.5, dpi = 300, format = 'png' } = {}) {
  const div = document.createElement('div');
  div.style.cssText = 'position:fixed;left:-10000px;top:0;';
  document.body.append(div);
  const w = Math.round(width_in * 96), hgt = Math.round(height_in * 96);
  try {
    const layout = { ...fig.layout, width: w, height: hgt, uirevision: undefined };
    await Plotly.newPlot(div, fig.data, layout, { staticPlot: true });
    if (fig.data.some((t) => t.type === 'heatmap')) await (await import('./plot2d.js')).fitColorbars(div);
    const out = await Plotly.toImage(div, { format, width: w, height: hgt, scale: format === 'svg' ? 1 : dpi / 96 });
    return format === 'svg' ? decodeURIComponent(out.replace(/^data:image\/svg\+xml,/, '')) : out;
  } finally {
    Plotly.purge(div);
    div.remove();
  }
}

export async function copyImage(dataUrl) {
  try {
    const blob = await (await fetch(dataUrl)).blob();
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
  } catch (e) {
    await call('clipboard_image', { data: dataUrl }, { label: 'Copying figure' });
  }
  toast('Figure copied. Paste into slides or documents.', 'ok');
}

export async function copyText(text, msg = 'Copied') {
  try { await navigator.clipboard.writeText(text); } catch (e) { await call('clipboard_text', { text }); }
  toast(msg, 'ok');
}

// Compose several rendered SVG panels into one vector SVG (Figure composer).
export function composeSvg(panels, W, H) {
  const parts = panels.map((p) => {
    const inner = p.svg.replace(/^<\?xml[^>]*>/, '').replace(/<svg /, `<svg x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" `);
    const label = p.label ? `<text x="${p.x + 4}" y="${p.y + 16}" font-family="Arial" font-weight="bold" font-size="${p.labelSize || 16}">${p.label}</text>` : '';
    return inner + label;
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="100%" height="100%" fill="white"/>${parts.join('')}</svg>`;
}

export async function svgToPng(svg, W, H, scale) {
  const img = new Image();
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
  const c = document.createElement('canvas');
  c.width = Math.round(W * scale); c.height = Math.round(H * scale);
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(img, 0, 0, c.width, c.height);
  URL.revokeObjectURL(url);
  return c.toDataURL('image/png');
}
