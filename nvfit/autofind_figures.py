"""White-background 300-DPI figures from saved AutoFind evidence, without a GUI."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import textwrap
import threading

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from .autofind import number, vector
from .app_state import json_safe

COLORS = {'coarse': '#757575', 'confirm': '#757575', 'refine': '#eb6834', 'z': '#2a78d6',
          'fine': '#1baf7a', 'patch': '#1baf7a', 'repeat': '#2a78d6', 'other': '#757575'}
DASH = {'coarse': '-', 'confirm': ':', 'refine': '-', 'fine': '-', 'patch': '--', 'repeat': '-.', 'other': ':'}
CMAP = LinearSegmentedColormap.from_list('autofind_blue', ['#cde2fb', '#6da7ec', '#2a78d6', '#0d366b'])
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a']
_EXPORT_LOCK = threading.Lock()


def _safe(value):
    value = json_safe(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, list):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    return value


def _figure(width=10, height=7):
    fig = Figure(figsize=(width, height), facecolor='white', layout='constrained')
    FigureCanvasAgg(fig)
    return fig


# Colour transforms. 'live' is autoFindLivePlot's log10(max(v,1)); it hides everything below 1, which for the
# dimensionless reconstructed metrics (ratios near 1) blanks the background, so 'log' and 'linear' are offered too.
SCALES = {'live': 'log10 max(value, 1)', 'log': 'log10 value (values <= 0 blank)', 'linear': 'value (linear)'}
_scale = 'live'   # set per export under _EXPORT_LOCK


def _log(m):
    v = np.asarray(m['values'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        if _scale == 'linear':
            return np.where(np.isfinite(v), v, np.nan)
        if _scale == 'log':
            return np.where(np.isfinite(v) & (v > 0), np.log10(v), np.nan)
        return np.where(np.isfinite(v), np.log10(np.maximum(v, 1)), np.nan)


def _norm(maps):
    finite = [a[np.isfinite(a)] for m in maps if 'values' in m for a in [_log(m)]]
    v = np.concatenate(finite) if finite else np.array([])
    if not v.size:
        return Normalize(0, 1)
    # MATLAB live view's rounded 2nd / 99.8th percentile order statistics.
    v.sort()
    n = len(v)
    lo = v[max(0, int(np.floor(0.02 * n + 0.5)) - 1)]
    hi = v[min(n - 1, max(1, int(np.floor(0.998 * n + 0.5)) - 1))] + 1e-6
    return Normalize(lo, max(hi, lo + 1e-6))


def _raster(ax, m, norm=None, contours=False, params=None, alpha=1):
    if 'values' not in m:
        ax.text(0.5, 0.5, m.get('unavailable', 'Unavailable'), transform=ax.transAxes, ha='center', wrap=True)
        return None
    artist = ax.pcolormesh(m['x_edges'], m['y_edges'], _log(m), cmap=CMAP, norm=norm or _norm([m]), shading='flat', rasterized=True, alpha=alpha)
    if contours and m['mode'] == 'ratio':
        r = np.asarray(m['values'], float)
        v = r[np.isfinite(r)]
        if v.size and len(m['x']) > 1 and len(m['y']) > 1:
            for key, style in (('weakRatio', '--'), ('strongRatio', '-')):
                level = number((params or {}).get(key))
                if level is not None and v.min() < level < v.max():
                    cs = ax.contour(m['x'], m['y'], r, levels=[level], colors=['#111111'], linewidths=0.8, linestyles=style)
                    ax.clabel(cs, fmt={level: f'{key} {level:g}x'}, fontsize=7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'{m["axes"][0]} (um)')
    ax.set_ylabel(f'{m["axes"][1]} (um)')
    ax.ticklabel_format(useOffset=False, style='plain')
    return artist


def _markers(ax, run, maps):
    candidates = [c for m in maps for c in m.get('candidates', [])]
    if candidates:
        ax.scatter([c['pos1'] for c in candidates], [c['pos2'] for c in candidates], marker='o', s=26,
                   facecolors='none', edgecolors='#2a78d6', linewidths=0.8, label='Candidate (recomputed)')
    for g in run['pursuits']:
        if g['particle'] or g['verdict'] == 'unavailable':
            continue
        seq = [s for s in run['scans'] if s['id'] in g['scan_ids'] and s.get('xLim') is not None]
        if seq:
            s = seq[0]
            ax.scatter(np.mean(s['xLim']), np.mean(s['yLim']), marker='x', s=38, color='#d03b3b', linewidths=1,
                       label='Rejected pursuit (window centre)' if not any(t.get_label().startswith('Rejected') for t in ax.collections) else None)
    for definitive, symbol, name in ((False, 'D', 'Particle: not definitive'), (True, '*', 'Particle: definitive (logged)')):
        parts = [p for p in run['particles'] if bool(number(p.get('definitive'), 0)) == definitive]
        if parts:
            ax.scatter([vector(p['position'])[0] for p in parts], [vector(p['position'])[1] for p in parts],
                       marker=symbol, s=110 if definitive else 45, c='#0ca30c' if definitive else '#fab219', edgecolors='white', linewidths=1.2, label=name, zorder=10)
    for key, symbol, name in (('start', 's', 'Start'), ('final_position', 'P', 'Final position')):
        v = run[key]
        if key == 'final_position' and run['final_position_source'] != 'L.finalPosition':
            name = 'Last saved stage (final not logged)'
        if len(v) >= 2 and all(x is not None and np.isfinite(x) for x in v[:2]):
            ax.scatter(v[0], v[1], marker=symbol, s=55, c='#111111', edgecolors='white', linewidths=1.2, label=name, zorder=11)


def _save(fig, path):
    fig.savefig(path, dpi=300, facecolor='white', metadata={'Software': 'SmartFitter AutoFind; replotted saved evidence'})
    fig.clear()
    return str(path)


def search_map(run, maps, path, contours=False):
    fig = _figure(11, 8)
    ax = fig.subplots()
    norm, artist = _norm(maps), None
    for m in maps:
        if 'values' in m:
            artist = _raster(ax, m, norm, contours, run['params'])
    kinds = set()
    for s in run['scans']:
        if s.get('xLim') is None or s.get('yLim') is None:
            continue
        x, y = sorted(vector(s['xLim'])), sorted(vector(s['yLim']))
        kind = s['kind']
        kinds.add(kind)
        ax.add_patch(Rectangle((x[0], y[0]), x[-1] - x[0], y[-1] - y[0], fill=False, edgecolor=COLORS.get(kind, '#757575'), linewidth=0.6, linestyle=DASH.get(kind, ':')))
    visits = run['visits']
    if visits:
        ax.plot([v['x'] for v in visits], [v['y'] for v in visits], color='#52514e', linewidth=0.7, marker='.', markersize=3, label='Coarse visit path', zorder=8)
        for v in [visits[0], visits[-1]]:
            ax.annotate(f'T{v["tile"]} ({"first" if v is visits[0] else "last"})', (v['x'], v['y']), xytext=(4, 5), textcoords='offset points', fontsize=8)
    _markers(ax, run, maps)
    if not any('values' in m for m in maps):
        ax.text(0.5, 0.9, 'Raster unavailable: missing / overwritten evidence.\nFootprint is logged geometry, not a replacement raster.',
                transform=ax.transAxes, ha='center', va='top', fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=COLORS[k], linestyle=DASH.get(k, ':'), linewidth=1, label=f'{k} footprint') for k in sorted(kinds)]
    if handles:
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 0.72), fontsize=8, frameon=False)
    if artist:
        metric = next(m['metric_label'] for m in maps if 'values' in m)
        cb = fig.colorbar(artist, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(SCALES[_scale] + '\n' + '\n'.join(textwrap.wrap(metric, 42)), fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (um)' if not maps else f'{next((m["axes"][0] for m in maps if "axes" in m), "x")} (um)')
    ax.set_ylabel('y (um)' if not maps else f'{next((m["axes"][1] for m in maps if "axes" in m), "y")} (um)')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_title(f'{run["name"]} / {run["run_id"]}\nStitched search: native physical pixels, no interpolation', fontsize=12)
    fig.supxlabel('Replotted saved evidence. Candidate detections/reference fits recomputed with current MATLAB formulas.\n' + ('No log: decisions unavailable.' if not run['has_log'] else 'Historical multidetector reference state may differ; opaque tables reconstructed.'), fontsize=8)
    return _save(fig, path)


def _time_scale(seconds):
    return (3600, 'h') if seconds >= 7200 else (60, 'min') if seconds >= 120 else (1, 's')


def timeline(run, path):
    fig = _figure(11, 7)
    ax, summary = fig.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1.8]})
    scale, unit = _time_scale(run['duration_s'])
    kinds = [k for k in COLORS if any(s['kind'] == k for s in run['scans'])]
    for i, kind in enumerate(kinds):
        for s in run['scans']:
            if s['kind'] == kind:
                ax.broken_barh([(s['start_t'] / scale, s['duration_s'] / scale)], (i - 0.28, 0.56), facecolors=COLORS[kind], edgecolors='none', linewidth=0)
                if s['duration_s'] < run['duration_s'] / 1000:
                    ax.plot(s['end_t'] / scale, i, marker='|', color=COLORS[kind], markersize=6, markeredgewidth=0.7)
    ax.scatter([e['t'] / scale for e in run['events']], [len(kinds)] * len(run['events']), marker='d', s=15, color='#111111', label='Decision / event')
    ax.set_yticks(range(len(kinds) + 1), kinds + ['events'])
    ax.set_xlabel(f'Elapsed time ({unit})')
    ax.set_xlim(0, max(run['duration_s'] / scale, 1))
    ax.set_title(f'{run["run_id"]}: search activity and decisions\n{run["stop_reason"]}', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    active = [(k, v) for k, v in run['activity_seconds'].items() if v > 0]
    summary.barh([k for k, v in active], [v / scale for k, v in active], color='#2a78d6', height=0.55)
    for i, (k, v) in enumerate(active):
        summary.text(v / scale, i, f' {v / scale:.2f} {unit}', va='center', fontsize=8)
    summary.set_xlim(0, max([v / scale for k, v in active] + [1]) * 1.22)
    summary.set_xlabel(f'Time by activity ({unit})')
    for a in (ax, summary):
        a.grid(axis='x', color='#e1e0d9', linewidth=0.5)
        a.set_axisbelow(True)
    fig.supxlabel('Replotted saved evidence. Bands use L.diag.seconds when available; otherwise completion intervals\ninclude settling, decisions and overhead (not isolated acquisition time). Short scans: completion ticks; events in events.csv.', fontsize=8)
    return _save(fig, path)


def pursuit_figures(store, run, group, directory, mode):
    seq = [s for s in run['scans'] if s['id'] in group['scan_ids']]
    out = []
    xy = []
    for kind in ('refine', 'fine'):
        options = [s for s in seq if s['kind'] == kind]
        if not options and kind == 'fine':
            options = [s for s in seq if s['kind'] == 'repeat']
        if options:
            xy.append(options[-1])
    if not xy:
        xy = [s for s in seq if s['type'] == 'xy'][-1:]
    maps = [store.scan(run['source'], s['id'], mode) for s in xy]
    fig = _figure(12, 7.6)
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 1.7])
    for i, m in enumerate(maps):
        ax = fig.add_subplot(grid[0, 0 if m['scan']['kind'] == 'refine' else 2])
        artist = _raster(ax, m)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=8)
        if artist:
            b = m.get('blob')
            if b and not b.get('noData'):
                ax.scatter(*b['center'], marker='+', s=90, color='#111111', linewidths=1.2, label='Recomputed blob centre')
                ax.legend(loc='upper right', fontsize=6)
            fig.colorbar(artist, ax=ax, shrink=0.72).set_label(SCALES[_scale], fontsize=8)
        ax.set_title(m['scan']['label'], fontsize=9)
    zax = fig.add_subplot(grid[0, 1])
    zscans = [s for s in seq if s['type'] == 'z']
    # One z-search per figure panel; avoid joining unrelated extensions into a fabricated continuous trace.
    ztags = list(dict.fromkeys(s['label'].split('_Z', 1)[-1].split('_', 1)[0] for s in zscans))
    first = ztags[0] if ztags else None
    for s in zscans:
        if s['label'].split('_Z', 1)[-1].split('_', 1)[0] != first:
            continue
        m = store.scan(run['source'], s['id'], 'metric' if mode == 'ratio' else mode)
        if 'values' not in m:
            continue
        reverse = '_rev' in s['label']
        zax.plot(m['z'], m['values'], 'o--' if reverse else 'o-', color=PALETTE[1] if reverse else PALETTE[0], linewidth=1, markersize=2.5, label=s['label'])
    if zscans:
        zax.legend(fontsize=6, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.24))
    else:
        zax.text(0.5, 0.5, 'No z scan in this pursuit', transform=zax.transAxes, ha='center')
    zax.set_xlabel('z (um); lower z = stage higher')
    zax.set_ylabel('\n'.join(textwrap.wrap(next((store.scan(run['source'], s['id'], 'metric' if mode == 'ratio' else mode).get('metric_label', '') for s in zscans), 'No data'), 36)), fontsize=8)
    zax.set_title('First z search: forward / reverse passes', fontsize=10)
    zax.ticklabel_format(useOffset=False, style='plain')
    zax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    zax.tick_params(labelsize=8)
    rax = fig.add_subplot(grid[1, :])
    pts = group['ratios']
    if pts:
        rax.plot([p['t'] for p in pts], [p['ratio'] for p in pts], 'o-', color=PALETTE[0], markersize=4, linewidth=1, label='Blob contrast (logged / rounded messages)')
    else:
        rax.text(0.5, 0.5, 'No logged ratio history', transform=rax.transAxes, ha='center')
    rax.set_xlabel('Elapsed time (s)')
    rax.set_ylabel('Blob contrast (dimensionless)')
    rax.set_title('Ratio history (message values may be rounded)', fontsize=9)
    bleaching = group.get('bleaching_events', [])
    if bleaching:
        rax.scatter([e['t'] for e in bleaching], [0.95] * len(bleaching), transform=rax.get_xaxis_transform(),
                    marker='v', s=25, color=PALETTE[1], label='Logged z-pass bleaching events (top rug, not contrast)')
        rax.legend(loc='lower left', fontsize=7)
    reason = '\n'.join(textwrap.wrap(group['reason'], 140))
    fig.suptitle(f'{run["run_id"]} / {group["id"]} — {group["verdict"]}\n{reason}', fontsize=10)
    fig.supxlabel('Replotted saved evidence. XY panels: last refine / fine (or repeat fallback); blob recomputed.\n' + (maps[0].get('metric_label', '') if maps else '') + '\nAll scans, passes, messages and provenance are retained in the manifest/CSV and interactive workspace.', fontsize=8)
    out.append(_save(fig, directory / f'pursuit_{group["id"]}.png'))
    # Later z-searches get separate pages with every pass and extension overlaid.
    for tag in ztags[1:]:
        fig = _figure(9, 5.5)
        ax = fig.subplots()
        for s in zscans:
            if s['label'].split('_Z', 1)[-1].split('_', 1)[0] != tag:
                continue
            m = store.scan(run['source'], s['id'], 'metric' if mode == 'ratio' else mode)
            if 'values' not in m:
                continue
            rev = '_rev' in s['label']
            ax.plot(m['z'], m['values'], 'o--' if rev else 'o-', color=PALETTE[1] if rev else PALETTE[0], markersize=3, linewidth=1, label=s['label'])
            ax.set_ylabel(m['metric_label'], fontsize=9)
        ax.set_xlabel('z (um); lower z = stage higher')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.legend(fontsize=7, ncol=2)
        ax.set_title(f'{run["run_id"]} / {group["id"]}: Z{tag}\nForward / reverse, extensions and fine pass (saved evidence)', fontsize=11)
        out.append(_save(fig, directory / f'pursuit_{group["id"]}_Z{tag}.png'))
    return out


def comparison(runs, maps_by_run, path):
    fig = _figure(11, 8)
    ax = fig.subplots()
    # Independent native-pixel scales; comparisons are geometric, not a brightness calibration.
    for i, (run, maps) in enumerate(zip(runs, maps_by_run)):
        color = PALETTE[i]
        cmap = LinearSegmentedColormap.from_list('run_' + str(i), ['white', color])
        for m in maps:
            if 'values' in m:
                ax.pcolormesh(m['x_edges'], m['y_edges'], _log(m), cmap=cmap, norm=_norm(maps), shading='flat', alpha=0.45, rasterized=True)
        visits = run['visits']
        if visits:
            ax.plot([v['x'] for v in visits], [v['y'] for v in visits], color=color, linestyle=['-', '--', '-.'][i], marker=['o', 's', '^'][i], markersize=3, linewidth=1, label=run['run_id'])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Multi-run overlay in shared absolute stage coordinates\nNative pixels, independent per-run colour ranges; no alignment or interpolation', fontsize=11)
    fig.supxlabel('Replotted saved evidence. Transparent layers show geometry, not calibrated relative brightness.\nOnly runs in the same stage coordinate frame should be compared; no physical registration is inferred.', fontsize=9)
    return _save(fig, path)


def export_figure_set(store, sources, directory, mode='metric', *, contours=False, scale='live'):
    """Create a fresh export directory, never overwrite source data or earlier exports."""
    global _scale
    if scale not in SCALES:
        raise ValueError(f'Colour scale must be one of {", ".join(SCALES)}.')
    directory = Path(directory).resolve()
    runs = [store.run(source) for source in dict.fromkeys(sources)]
    if not runs:
        raise ValueError('No AutoFind runs selected')
    if mode not in ('metric', 'det1', 'det2', 'ratio'):
        raise ValueError('Unknown metric')
    if any(directory == Path(r['folder']) or Path(r['folder']) in directory.parents for r in runs):
        raise ValueError('Export outside the source run folders; acquisition evidence is read-only.')
    with _EXPORT_LOCK:
        _scale = scale
        root = directory / ('AutoFind_figures_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True, exist_ok=False)
        files, maps_by_run, errors = [], [], []
        for run in runs:
            dest = root / run['run_id']
            dest.mkdir()
            maps = []
            for s in run['scans']:
                if s['kind'] != 'coarse':
                    continue
                try:
                    maps.append(store.scan(run['source'], s['id'], mode))
                except Exception as exc:
                    errors.append({'source': run['source'], 'scan_id': s['id'], 'error': str(exc)})
            maps_by_run.append(maps)
            files.append(search_map(run, maps, dest / 'search_map.png', contours))
            files.append(timeline(run, dest / 'timeline.png'))
            for g in run['pursuits']:
                try:
                    files.extend(pursuit_figures(store, run, g, dest, mode))
                except Exception as exc:
                    errors.append({'source': run['source'], 'pursuit': g['id'], 'error': str(exc)})
            (dest / 'manifest.json').write_text(json.dumps(_safe(store.payload(run['source'])), indent=2), encoding='utf-8')
            with (dest / 'events.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['elapsed_s', 'type', 'message'])
                w.writerows((e['t'], e.get('type'), e.get('message')) for e in run['events'])
            with (dest / 'scans.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['scan_id', 'label', 'kind', 'path', 'start_s', 'end_s', 'duration_source', 'logged'])
                w.writerows((s['id'], s['label'], s['kind'], s['path'], s['start_t'], s['end_t'], s['duration_source'], s['logged']) for s in run['scans'])
        for first in range(0, len(runs), 3):
            group = runs[first:first + 3]
            if len(group) > 1:
                files.append(comparison(group, maps_by_run[first:first + 3], root / f'comparison_{first // 3 + 1}.png'))
        summaries = [store.summary(r) for r in runs]
        with (root / 'runs.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['run_id', 'source', 'duration_s', 'tiles', 'scans', 'candidate_detections_may_repeat', 'pursued', 'particles', 'definitive', 'stop_reason', 'detectors', 'detectorCombine', 'weakRatio', 'strongRatio'])
            w.writerows([r[k] for k in ('run_id', 'source', 'duration_s', 'tiles', 'scans', 'candidate_detections', 'pursued', 'particles', 'definitive', 'stop_reason')] + [str(r['params'].get(k, 'unknown')) for k in ('detectors', 'detectorCombine', 'weakRatio', 'strongRatio')] for r in summaries)
        report = {'directory': str(root), 'pngs': files, 'dpi': 300, 'mode': mode, 'colour_scale': SCALES[scale], 'errors': errors,
                  'provenance': 'replotted saved evidence, not new acquisition or simulation', 'runs': summaries}
        (root / 'export.json').write_text(json.dumps(_safe(report), indent=2), encoding='utf-8')
        return report
