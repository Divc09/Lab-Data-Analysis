"""Read-only, GUI-independent AutoFind run browsing and native-pixel analysis.

Raster rows follow the first loaded axis (MATLAB's slow x); columns follow the
second. Positions are absolute stage micrometers. No regridding is performed.
Opaque MATLAB objects are reported, not interpreted as numeric arrays.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path, PureWindowsPath
import re
import threading
import time
import warnings
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve, label, maximum_filter

from .map_comparison import axis_edges

EPS = np.finfo(float).eps
KINDS = ('coarse', 'confirm', 'refine', 'z', 'fine', 'patch', 'repeat', 'other')
EXCLUDED = {'.venv', '__pycache__', 'out', 'artifacts', 'site', 'tmp', '.git'}


def _plain(value):
    if type(value).__name__ == 'MatlabOpaque':
        rec = value.ravel()[0]
        cls = rec['s2']
        return {'opaque': cls.decode(errors='replace') if isinstance(cls, bytes) else str(cls)}
    if hasattr(value, '_fieldnames'):
        return {k: _plain(getattr(value, k)) for k in value._fieldnames}
    if isinstance(value, np.ndarray) and value.dtype == object:
        if all(hasattr(x, '_fieldnames') for x in value.ravel()):
            return [_plain(x) for x in value.ravel()]
        out = np.empty(value.shape, dtype=object)
        for i in np.ndindex(value.shape):
            out[i] = _plain(value[i])
        return out
    return value


def is_v73(path: str | Path) -> bool:
    """True for MATLAB v7.3 (HDF5) files, which scipy's loadmat cannot read."""
    with Path(path).open('rb') as f:
        header = f.read(128)
    return b'MATLAB 7.3' in header or header.startswith(b'\x89HDF')


def read_mat(path: str | Path, names: list[str] | None = None) -> dict:
    """read_mat, retried: Box Drive placeholders can fail an open/read (e.g. Errno 22) while they download or re-sync."""
    for delay in (0.25, 0.5, 1.0, 2.0):
        try:
            return _read_mat(path, names)
        except OSError:
            time.sleep(delay)
    return _read_mat(path, names)


def _read_mat(path: str | Path, names: list[str] | None = None) -> dict:
    """Load v5 structs/cells; v7.3 structs/cells use HDF5 references when available."""
    path = Path(path)
    if not is_v73(path):
        raw = loadmat(path, struct_as_record=False, squeeze_me=True, variable_names=names)
        return {k: _plain(v) for k, v in raw.items() if not k.startswith('__')}
    import h5py

    with h5py.File(path, 'r') as f:
        def decode(obj):
            cls = obj.attrs.get('MATLAB_class', b'')
            if isinstance(cls, bytes):
                cls = cls.decode()
            # MATLAB class objects (datetime, string, table...) are references into MCOS storage, not data.
            if cls in ('table', 'datetime', 'function_handle', 'string', 'duration', 'categorical') or 'MATLAB_object_decode' in obj.attrs:
                return {'opaque': cls}
            if obj.attrs.get('MATLAB_empty', 0):
                # MATLAB stores an empty value as its dimensions; the data is not those numbers.
                return '' if cls == 'char' else np.empty(0, dtype=object if cls in ('cell', 'struct') else float)
            if isinstance(obj, h5py.Group):
                fields = {k: decode(v) for k, v in obj.items() if not k.startswith('#')}
                # HDF5 lists members alphabetically; MATLAB_fields keeps the struct's own field order.
                order = [b''.join(np.asarray(n).ravel()).decode(errors='replace') for n in obj.attrs.get('MATLAB_fields', [])]
                if order:
                    fields = {k: fields[k] for k in order if k in fields} | {k: v for k, v in fields.items() if k not in order}
                if cls == 'struct':
                    refs = [v for v in obj.values() if isinstance(v, h5py.Dataset) and h5py.check_dtype(ref=v.dtype) and 'MATLAB_class' not in v.attrs]
                    count = int(np.prod(refs[0].shape)) if refs else 1
                    if count > 1:
                        return [{k: np.asarray(v, dtype=object).ravel(order='F')[i] for k, v in fields.items()} for i in range(count)]
                return fields
            a = obj[()]
            if h5py.check_dtype(ref=obj.dtype):
                a = a.T
                out = np.empty(a.shape, dtype=object)
                for i in np.ndindex(a.shape):
                    out[i] = decode(f[a[i]]) if a[i] else None
                if out.size == 1:
                    return out.ravel()[0]
                return np.squeeze(out)
            if cls == 'char':
                return ''.join(chr(int(x)) for x in a.T.ravel() if x)
            out = np.squeeze(a.T)
            return out[()] if out.ndim == 0 else out
        return {k: decode(v) for k, v in f.items() if not k.startswith('#') and (names is None or k in names)}


def records(value) -> list[dict]:
    if isinstance(value, dict):
        return [] if 'opaque' in value else [value]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [x for x in np.asarray(value, dtype=object).ravel() if isinstance(x, dict) and 'opaque' not in x]
    return []


def vector(value) -> list:
    if value is None:
        return []
    return np.asarray(value).ravel().tolist()


def number(value, default=None):
    try:
        a = np.asarray(value, dtype=float).ravel()
        return float(a[0]) if a.size and np.isfinite(a[0]) else default
    except (TypeError, ValueError):
        return default


def _timestamp(value):
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None
    return number(value) if not isinstance(value, dict) else None


def _position(meta, key):
    pos = meta.get(key, {})
    if not isinstance(pos, dict):
        return {}
    axes, values = vector(pos.get('axes')), vector(pos.get('micrometers'))
    return {str(k).lower(): number(v) for k, v in zip(axes, values)}


def scan_kind(label_text: str, scan_type: str = '') -> str:
    text = label_text.lower()
    if scan_type == 'z' or re.search(r'(?:_z\d|trendz)', text):
        return 'z'
    for word, kind in (('coarse', 'coarse'), ('confirm', 'confirm'), ('repeat', 'repeat'),
                       ('refine', 'refine'), ('patch', 'patch'), ('fine', 'fine')):
        if word in text:
            return kind
    return 'other'


def line_ratio(a: np.ndarray) -> dict:
    """autoFindDetect.m lines 15–30: two dim=1/2 median passes, then MAD noise.

    MATLAB dim=1 is the column median (NumPy axis=0). Zeros/negative values
    participate in the passes and are masked only afterwards, exactly as in MATLAB.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError('Line normalization requires a raster, not a z trace.')
    valid = np.isfinite(a) & (a > 0)
    bg = np.full(a.shape, np.median(a[valid]) if valid.any() else np.nan)
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        r = a / bg
        for _ in range(2):
            for dim in (0, 1):
                m = np.nanmedian(r, axis=dim, keepdims=True)
                m[~np.isfinite(m) | (m <= 0)] = 1
                bg = bg * m
                r = a / bg
    r[~valid] = np.nan
    v = r[np.isfinite(r)]
    noise = max(1.4826 * np.median(np.abs(v - np.median(v))), EPS) if v.size else EPS
    return {'ratio': r, 'background': bg, 'noise': float(noise)}


def detect_candidates(a, x, y, sigma=8, min_ratio=2, max_candidates=20) -> list[dict]:
    info = line_ratio(a)
    r, noise = info['ratio'], info['noise']
    rp = np.where(np.isfinite(r), r, -np.inf)
    maxima = (rp >= max(1 + sigma * noise, min_ratio)) & (rp >= maximum_filter(rp, size=3, mode='constant', cval=-np.inf))
    # MATLAB find() visits column-major; stable ties preserve that order.
    ij = np.argwhere(maxima.T)[:, ::-1]
    out = []
    for i, j in ij:
        v = r[i, j]
        regions, _ = label(rp >= 1 + 0.5 * (v - 1), np.ones((3, 3)))
        out.append({'pos1': float(x[i]), 'pos2': float(y[j]), 'ratio': float(v),
                    'sigma': float((v - 1) / noise), 'value': float(a[i, j]),
                    'nPix': int(np.count_nonzero(regions == regions[i, j]))})
    return sorted(out, key=lambda c: -c['ratio'])[:max_candidates]


def detector_levels(a, b, unmix=False) -> np.ndarray:
    """autoFindParticle.levelsOf: pairwise medians and five 3-MAD clipped LS fits."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    ref = np.array([np.median(a), np.median(b), 0.0]) if a.size else np.array([np.nan, np.nan, 0.0])
    if not unmix or a.size < 10:
        return ref
    w = np.ones(a.size, bool)
    cf = None
    for _ in range(5):
        if w.sum() < 5 or np.ptp(b[w]) <= 0:
            break
        cf = np.linalg.lstsq(np.column_stack((b[w], np.ones(w.sum()))), a[w], rcond=None)[0]
        residual = a - np.column_stack((b, np.ones(b.size))) @ cf
        med = np.median(residual[w])
        sg = 1.4826 * np.median(np.abs(residual[w] - med))
        w = np.abs(residual - med) <= 3 * max(sg, EPS)
    if cf is not None:
        ref[2] = max(cf[0], 0)
    return ref


def detector_metric(a, b, params, ref=None):
    dets = vector(params.get('detectors', 1))
    combine = str(params.get('detectorCombine', 'max')).lower()
    if len(dets) < 2:
        return np.asarray(b if dets == [2] else a, float)
    if combine == 'sum':
        return a + b
    if ref is None or not np.all(np.isfinite(ref)):
        ref = detector_levels(a, b, combine == 'unmix')
    r2 = b / max(ref[1], EPS)
    r1 = np.maximum(a - ref[2] * (b - ref[1]), 0) / max(ref[0], EPS) if combine == 'unmix' else a / max(ref[0], EPS)
    bad = np.isnan(a) | np.isnan(b)
    r1[bad], r2[bad] = np.nan, np.nan
    return (r1 + r2) / 2 if combine == 'mean' else np.fmax(r1, r2)


def blob(a, x, y) -> dict:
    """autoFindBlob's replicate-padded 0.7-pixel Gaussian and half-height region."""
    a = np.asarray(a, float)
    valid = np.isfinite(a)
    if valid.mean() < 0.25:
        return {'noData': True}
    a0 = np.where(valid, a, np.median(a[valid]))
    g = np.exp(-np.arange(-2, 3, dtype=float) ** 2 / (2 * 0.7 ** 2))
    g /= g.sum()
    s = convolve(a0, np.outer(g, g), mode='nearest')
    bg = np.median(a0)
    for _ in range(3):
        mask = convolve((s >= bg + 0.5 * (s.max() - bg)).astype(int), np.ones((3, 3)), mode='constant') > 0
        outside = ~mask & valid
        if outside.sum() >= max(8, 0.15 * a.size):
            bg = np.median(a0[outside])
    i, j = np.unravel_index(np.argmax(s.T), s.T.shape)[::-1]
    regions, _ = label(s >= bg + 0.5 * (s.max() - bg), np.ones((3, 3)))
    region = regions == regions[i, j]
    w = np.maximum(a0 - bg, 0) * region
    xx, yy = np.meshgrid(x, y, indexing='ij')
    centre = [float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum())] if w.sum() else [float(x[i]), float(y[j])]
    dx = np.median(np.abs(np.diff(x))) if len(x) > 1 else 0
    dy = np.median(np.abs(np.diff(y))) if len(y) > 1 else 0
    extent = [float(np.ptp(np.asarray(x)[region.any(axis=1)]) + dx), float(np.ptp(np.asarray(y)[region.any(axis=0)]) + dy)]
    outside = a0[(convolve(region.astype(int), np.ones((3, 3)), mode='constant') == 0) & valid]
    if outside.size < 5:
        outside = a0[valid]
    noise = max(1.4826 * np.median(np.abs(outside - np.median(outside))), EPS)
    return {'center': centre, 'extent': extent, 'peak': float(s.max()), 'background': float(bg),
            'ratio': float(s.max() / max(bg, EPS)), 'sigma': float((s.max() - bg) / noise),
            'nPix': int(region.sum()), 'touches': [bool(region[0].any()), bool(region[-1].any()), bool(region[:, 0].any()), bool(region[:, -1].any())],
            'noData': False, 'provenance': 'recomputed with current autoFindBlob formula'}


def _scan_axes(raw):
    axes = []
    for s in records(raw.get('scanInfo')):
        name = str(s.get('parameter', s.get('type', 'axis')))
        n = int(number(s.get('nSteps', s.get('nsteps')), 0))
        bounds = vector(s.get('bounds'))
        if not bounds:
            bounds = [number(s.get('starts'), 0), number(s.get('ends'), 0)]
        if n < 1 or len(bounds) < 2:
            raise ValueError('Invalid Stage_Scan axis bounds/nSteps')
        # The acquisition adapter used linspace(bounds,n), including reverse passes.
        axes.append((name, np.linspace(float(bounds[0]), float(bounds[-1]), n)))
    if not axes:
        raise ValueError('No Stage_Scan scanInfo axes')
    return axes


def _values(data, field, shape):
    v = data.get(field)
    out = np.full(shape, np.nan)
    if v is None:
        return out
    cells = np.asarray(v, dtype=object)
    if cells.size == int(np.prod(shape)):
        for i, cell in enumerate(cells.ravel()):
            out.flat[i] = number(cell, np.nan)
    elif int(np.prod(shape)) == 1:
        out.flat[0] = number(v, np.nan)
    else:
        raise ValueError(f'{field}: cell shape {cells.shape} does not match scan {shape}')
    it = np.asarray(data.get('iteration', np.zeros(shape)), float)
    if it.size != out.size:
        raise ValueError('iteration mask does not match scan geometry')
    out[it.reshape(shape) < 1] = np.nan
    return out


def _basename(path):
    return PureWindowsPath(str(path)).name


def _scan_files(folder):
    files = {p.name.casefold(): p for p in folder.glob('*.mat') if not p.name.lower().endswith('_log.mat')}
    out = []
    for p in files.values():
        stem = re.sub(r'_(checkpoint|fallback)$', '', p.stem, flags=re.I)
        if stem != p.stem and (stem + '.mat').casefold() in files:
            continue
        if '_AutoFind_' in p.name or 'autofind' in folder.name.lower():
            out.append(p)
    return sorted(out, key=lambda p: (p.stat().st_mtime, p.name))


def _logs(folder):
    # RH_*_log.mat are autoFindRehone logs (no L struct); the Library shows those runs, not AutoFind.
    return sorted(p for p in folder.glob('*_log.mat') if not p.name.upper().startswith('RH_'))


def _stamp(path):
    s = path.stat()
    return (s.st_mtime_ns, s.st_size)


class AutoFindStore:
    """Bounded, mtime-aware caches; run indexing never loads logged raster data."""

    def __init__(self):
        self._lock = threading.RLock()
        self._runs = OrderedDict()
        self._headers = OrderedDict()
        self._rasters = OrderedDict()
        self.read_counts = defaultdict(int)

    def _cached(self, cache, key, stamp, make, limit):
        with self._lock:
            hit = cache.get(key)
            if hit and hit[0] == stamp:
                cache.move_to_end(key)
                return hit[1]
            value = make()
            cache[key] = (stamp, value)
            cache.move_to_end(key)
            while len(cache) > limit:
                cache.popitem(last=False)
            return value

    def header(self, path):
        path = Path(path)
        def load():
            self.read_counts['headers'] += 1
            raw = read_mat(path, ['scanInfo', 'params', 'metadata'])
            axes = _scan_axes(raw)
            meta, params = raw.get('metadata', {}), raw.get('params', {})
            start = _position(meta, 'stagePositionAtStart')
            end = _position(meta, 'stagePositionAtSave')
            began, finished = _timestamp(meta.get('runStartTime')), _timestamp(meta.get('savedAt'))
            text = path.stem.split('_AutoFind_', 1)[-1]
            text = re.sub(r'_(checkpoint|fallback)$', '', text)
            kind = scan_kind(text, 'z' if len(axes) == 1 and axes[0][0].lower() in ('z', 'stage z') else '')
            out = {'label': text, 'kind': kind, 'type': 'xy' if len(axes) == 2 else 'z',
                   'axes': [a[0] for a in axes], 'shape': [len(a[1]) for a in axes],
                   'unit': 'counts/s' if number(params.get('perSecond')) == 1 else 'counts' if number(params.get('perSecond')) == 0 else 'stored units (unknown)',
                   'started': began, 'finished': finished or path.stat().st_mtime,
                   'time_source': 'scan metadata' if finished is not None else 'file mtime (save time; acquisition duration unknown)',
                   'start_position': start, 'end_position': end, 'z': start.get('z'),
                   'xy': [start.get('x'), start.get('y')], 'step': number(np.abs(np.diff(axes[0][1])), 1),
                   'save_status': meta.get('runStatus', ''), 'params': {k: params.get(k) for k in ('perSecond', 'detectorMode', 'sequenceTimePerDataPoint')}}
            if len(axes) == 2:
                out['xLim'], out['yLim'] = [[float(a[1][0]), float(a[1][-1])] for a in axes]
            else:
                out['zLim'] = [float(axes[0][1][0]), float(axes[0][1][-1])]
            return out
        return self._cached(self._headers, str(path), _stamp(path), load, 1024)

    def raster(self, path):
        path = Path(path)
        def load():
            self.read_counts['rasters'] += 1
            raw = read_mat(path, ['Data', 'data', 'scanInfo', 'params'])
            axes = _scan_axes(raw)
            shape = tuple(len(a[1]) for a in axes)
            data = raw.get('Data', raw.get('data', {}))
            return {'axes': axes, 'det1': _values(data, 'values', shape),
                    'det2': _values(data, 'detector2Values', shape)}
        return self._cached(self._rasters, str(path), _stamp(path), load, 96)

    def discover(self, root):
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f'Not a run folder: {root}')
        t0 = time.perf_counter()
        folders, queue = [], [root]
        while queue:
            folder = queue.pop()
            logs = _logs(folder)
            scans = _scan_files(folder)
            if logs or scans:
                folders.append((folder, logs, scans))
            queue.extend(p for p in folder.iterdir() if p.is_dir() and p.name not in EXCLUDED and not p.name.startswith('.'))
            if len(queue) + len(folders) > 2000:
                raise ValueError('Choose a smaller AutoFind data folder (over 2000 folders).')
        runs, errors = [], []
        for folder, logs, scans in sorted(folders):
            for source in logs or [folder]:
                try:
                    run = self.run(str(source))
                    runs.append(self.summary(run))
                except Exception as exc:
                    errors.append({'path': str(source), 'error': str(exc)})
        return {'root': str(root), 'runs': runs, 'errors': errors, 'load_seconds': time.perf_counter() - t0}

    def run(self, source):
        source = Path(source).resolve()
        if not source.exists():
            raise ValueError(f'AutoFind run does not exist: {source}')
        if source.is_file() and not source.name.lower().endswith('_log.mat'):
            raise ValueError('Choose an AutoFind *_log.mat or run folder.')
        folder = source if source.is_dir() else source.parent
        if source.is_dir():
            logs = _logs(source)
            if len(logs) == 1:
                source = logs[0]
            elif logs:
                raise ValueError('Several logs in this folder; choose a specific log/run.')
        files = _scan_files(folder)
        logs = _logs(folder)
        stamp =(tuple((p.name, *_stamp(p)) for p in files + logs), source.name)
        return self._cached(self._runs, str(source), stamp, lambda: self._load_run(source, folder, files, logs), 24)

    def _load_run(self, source, folder, files, logs):
        t0 = time.perf_counter()
        l = read_mat(source, ['L']).get('L', {}) if source.is_file() else {}
        if source.is_file() and (not isinstance(l, dict) or not l):
            raise ValueError('AutoFind log has no L struct')
        has_log = bool(l)
        warn = []
        if has_log:
            warn.append('MATLAB object tables/datetimes are opaque. Visit geometry is reconstructed from coarse scans; candidate markers are recomputed detections or rejected pursuit-window centres, explicitly labelled by provenance.')
        else:
            warn.append('no log: decisions unavailable; detector settings unknown, default metric is raw det1. Timeline intervals use scan metadata/save timestamps, not measured acquisition durations.')
        byname = {p.name.casefold(): p for p in files}
        scans = []
        for i, entry in enumerate(records(l.get('scans'))):
            s = dict(entry)
            name = _basename(s.get('file', ''))
            p = byname.get(name.casefold())
            fallback = False
            if p is None:
                p = byname.get((Path(name).stem + '_fallback.mat').casefold()) or byname.get((Path(name).stem + '_checkpoint.mat').casefold())
                fallback = p is not None
            s.update({'id': str(i), 'path': str(p) if p else None, 'file_name': name,
                      'kind': scan_kind(str(s.get('label', name)), str(s.get('type', ''))),
                      'time_source': 'L.scans.t (scan completion, s)', 'logged': True,
                      'fallback': fallback, 'shared_file': False})
            s['t'] = number(s.get('t'), 0)
            if p is None:
                warn.append(f'Missing scan: {name}')
            scans.append(s)
        claimed = {s['path'] for s in scans}
        extra = [p for p in files if str(p) not in claimed]
        if len(logs) > 1:
            other_names = defaultdict(int)
            for log in logs:
                other = l if log == source else read_mat(log, ['L']).get('L', {})
                for s in records(other.get('scans')):
                    other_names[_basename(s.get('file', '')).casefold()] += 1
            for s in scans:
                s['shared_file'] = other_names[s['file_name'].casefold()] > 1
            if any(s['shared_file'] for s in scans):
                warn.append('A scan basename is shared by multiple logs; the file may have been overwritten. Earlier noData evidence is suppressed rather than replaced with a later raster.')
            # Do not attach another run's files by proximity or folder membership.
            extra = []
        events = [dict(e, t=number(e.get('t'), 0), id=str(i)) for i, e in enumerate(records(l.get('events')))]
        last_log_t = max([s['t'] for s in scans] + [0])
        for p in extra:
            try:
                hdr = self.header(p)
                s = {k: v for k, v in hdr.items() if k != 'params'}
                if s['kind'] == 'other' and s['type'] == 'xy':
                    s['kind'] = 'coarse'
                    s['kind_source'] = 'unlabelled Stage_Scan raster; search activity inferred'
                s.update({'id': str(len(scans)), 'path': str(p), 'file_name': p.name, 'logged': False, 'shared_file': False})
                scans.append(s)
            except Exception as exc:
                warn.append(f'Skipped {p.name}: {exc}')
        unlogged = [s for s in scans if not s['logged']]
        if unlogged:
            times = [s['finished'] for s in unlogged]
            base = min(times)
            if has_log and claimed:
                log_files = [Path(p) for p in claimed if p]
                if log_files:
                    base = min(p.stat().st_mtime - s['t'] for s in scans if s['logged'] and s['path'] for p in [Path(s['path'])])
            else:
                began = [s.get('started') for s in unlogged if s.get('started') is not None]
                base = min(began) if began else base
            for s in unlogged:
                s['t'] = max(0, s['finished'] - base)
            if has_log:
                warn.append(f'{len(unlogged)} unlogged saved scan(s), including interrupted/checkpoint evidence; times estimated from file mtime relative to logged saves.')
        scans.sort(key=lambda s: s['t'])
        prev = 0.0
        diag = {str(d.get('scan')): d for d in records(l.get('diag'))}
        for s in scans:
            end = float(s['t'])
            d = diag.get(Path(s['file_name']).stem, {})
            secs = number(d.get('seconds'))
            start_t = max(prev, end - secs) if secs is not None else prev
            s.update({'start_t': start_t, 'end_t': end, 'duration_s': max(0, end - start_t),
                      'duration_source': 'L.diag.seconds' if secs is not None else 'completion interval (includes settling/decisions/overhead)',
                      'tile': int(re.search(r'T(\d+)', s.get('label', '')).group(1)) if re.search(r'T(\d+)', s.get('label', '')) else None})
            prev = max(prev, end)
            # First October 12 log says noData; its basename was reused by the next run.
            s['evidence_unavailable'] = s['shared_file'] and any(e.get('type') == 'noData' and str(s.get('label')) in str(e.get('message')) for e in events)
        params = l.get('p', {}) if has_log else {}
        particles = records(l.get('particles'))
        pursuits = self._pursuits(scans, events, particles)
        visits = []
        for s in scans:
            if s['kind'] == 'coarse' and s.get('xLim') is not None:
                visits.append({'scan_id': s['id'], 'tile': s['tile'], 'x': np.mean(s['xLim']), 'y': np.mean(s['yLim']),
                               'xLim': s['xLim'], 'yLim': s['yLim'], 't': s['t'], 'provenance': 'L.scans / Stage_Scan bounds, not L.visited navLevel'})
        duration = number(l.get('elapsed'), max([s['t'] for s in scans] + [e['t'] for e in events] + [0]))
        duration = max(duration, max([s['t'] for s in scans] + [0]))
        activity = {kind: sum(s['duration_s'] for s in scans if s['kind'] == kind) for kind in KINDS}
        activity['unassigned'] = max(0, duration - sum(activity.values()))
        counts = sum(int(m.group(1)) for e in events for m in [re.search(r'(\d+) candidate\(s\)', str(e.get('message', '')))] if m)
        run = {'source': str(source), 'folder': str(folder), 'run_id': str(l.get('runId', folder.name)), 'name': folder.name,
               'has_log': has_log, 'params': params, 'scans': scans, 'events': events, 'particles': particles,
               'pursuits': pursuits, 'visits': visits, 'duration_s': duration, 'duration_source': 'L.elapsed' if 'elapsed' in l else 'last available scan/event; incomplete or reconstructed',
               'activity_seconds': activity, 'candidate_detections': counts if has_log else None,
               'start': vector(l.get('start')), 'final_position': vector(l.get('finalPosition')),
               'z_limits': vector(l.get('zLimits')), 'stop_reason': str(l.get('stopReason', 'unfinished log' if has_log else 'no log: decisions unavailable')),
               'simulated': bool(number(l.get('simulated'), 0)), 'error': str(l.get('error', '')),
               'warnings': warn, 'loaded_seconds': time.perf_counter() - t0}
        run['final_position_source'] = 'L.finalPosition' if run['final_position'] else 'unavailable'
        if not has_log and scans:
            pos = scans[0].get('start_position', {})
            run['start'] = [pos.get(k) for k in ('x', 'y', 'z')]
        if not run['final_position'] and scans:
            last = next((s for s in reversed(scans) if s['path'] and not s['evidence_unavailable']), None)
            try:
                pos = self.header(last['path']).get('end_position', {}) if last else {}
                if pos and all(pos.get(k) is not None for k in ('x', 'y')):
                    run['final_position'] = [pos.get(k) for k in ('x', 'y', 'z')]
                    run['final_position_source'] = 'last scan metadata.stagePositionAtSave; final position not logged'
                    warn.append('No L.finalPosition: marker uses the last scan’s recorded stagePositionAtSave, not a verified post-run stage position.')
            except Exception as exc:
                warn.append(f'Last saved stage position unavailable: {exc}')
        return run

    def _pursuits(self, scans, events, particles):
        groups = OrderedDict()
        for s in scans:
            m = re.search(r'T\d+C\d+', str(s.get('label', '')))
            if m:
                tag = m.group(0)
                groups.setdefault(tag, {'id': tag, 'scan_ids': [], 'events': [], 'particle': None})['scan_ids'].append(s['id'])
        for tag, g in groups.items():
            seq = [s for s in scans if s['id'] in g['scan_ids']]
            lo, hi = seq[0]['start_t'], seq[-1]['end_t']
            g['events'] = [e for e in events if re.search(rf'\b{re.escape(tag)}(?:_|\b)', str(e.get('message', ''))) and lo <= e['t'] <= hi + 30]
            # confirm events lack a candidate tag: link by the logged scan completion interval.
            g['events'] += [e for e in events if e.get('type') == 'confirm' and any(s['kind'] == 'confirm' and s['end_t'] <= e['t'] <= s['end_t'] + 2 for s in seq)]
            g['events'] = sorted({e['id']: e for e in g['events']}.values(), key=lambda e: e['t'])
            for part in particles:
                names = {_basename(p).casefold() for p in vector(part.get('files'))}
                if any(s['file_name'].casefold() in names for s in seq):
                    g['particle'] = part
                    break
            verdicts = [e for e in g['events'] if e.get('type') in ('reject', 'lost', 'repeat', 'confirm', 'noData')]
            g['reason'] = str(verdicts[-1].get('message')) if verdicts else 'No final verdict recorded for this pursuit.'
            part = g['particle']
            g['verdict'] = 'definitive (logged)' if part and bool(number(part.get('definitive'), 0)) else 'not definitive (logged)' if part else 'rejected / not repeated' if verdicts else 'unavailable'
            g['ratios'] = []
            for e in sorted(g['events'], key=lambda e: e['t']):
                if e.get('type') in ('refine', 'reject', 'fine', 'repeat'):
                    m = re.search(r'(\d+(?:\.\d+)?)x', str(e.get('message', '')))
                    if m:
                        g['ratios'].append({'t': e['t'], 'ratio': float(m.group(1)), 'source': 'rounded decision message', 'scan_id': min(seq, key=lambda s: abs(s['t'] - e['t']))['id']})
            logged_ratios = vector(part.get('ratios')) if part else []
            if logged_ratios:
                names = {_basename(p).casefold() for p in vector(part.get('files'))}
                xy_steps = [s for s in seq if s['type'] == 'xy' and s['file_name'].casefold() in names]
                if len(xy_steps) == len(logged_ratios):
                    g['ratios'] = [{'t': s['t'], 'ratio': number(r), 'source': 'L.particles.ratios; time from matching particle file', 'scan_id': s['id']} for s, r in zip(xy_steps, logged_ratios)]
            g['bleaching'] = bool(number(part.get('bleaching'), 0)) if part and 'bleaching' in part else None
            g['bleaching_events'] = [{'t': e['t'], 'message': e.get('message', ''), 'source': 'logged z-pass bleaching decision'}
                                     for e in g['events'] if e.get('type') == 'bleach']
            g['time_range'] = [lo, hi]
            if sum(e.get('type') == 'repeat' for e in g['events']) > 1:
                g['reason'] += ' Candidate tag was reused; all attempts are shown together.'
        return list(groups.values())

    def summary(self, run):
        return {k: run[k] for k in ('source', 'folder', 'name', 'run_id', 'has_log', 'duration_s', 'duration_source',
                                   'stop_reason', 'params', 'candidate_detections', 'loaded_seconds', 'simulated')} | {
            'tiles': len({v['tile'] for v in run['visits'] if v['tile'] is not None}) or len(run['visits']),
            'scans': len(run['scans']), 'pursued': len(run['pursuits']), 'particles': len(run['particles']),
            'definitive': sum(bool(number(p.get('definitive'), 0)) for p in run['particles']), 'warnings': run['warnings']}

    def payload(self, source):
        run = self.run(source)
        return {k: v for k, v in run.items() if k not in ('events', 'params')} | {'params': run['params'],
            'events': [{k: v for k, v in e.items() if k != 'data'} for e in run['events']], 'summary': self.summary(run)}

    def scan(self, source, scan_id, mode='metric'):
        if mode not in ('metric', 'det1', 'det2', 'ratio'):
            raise ValueError('Metric must be metric, det1, det2, or ratio.')
        run = self.run(source)
        s = next((s for s in run['scans'] if s['id'] == str(scan_id)), None)
        if s is None:
            raise ValueError('Unknown scan ID')
        if not s['path'] or s['evidence_unavailable']:
            return {'scan': s, 'unavailable': 'Missing or overwritten scan evidence; not substituted with another run.'}
        data = self.raster(s['path'])
        hdr = self.header(s['path'])
        a, b = data['det1'], data['det2']
        params = run['params']
        ref = None
        combine = str(params.get('detectorCombine', 'max')).lower()
        both = len(vector(params.get('detectors', 1))) >= 2
        if both and combine != 'sum':
            coarse = [q for q in run['scans'] if q['kind'] == 'coarse' and q['t'] <= s['t'] and q['path'] and not q['evidence_unavailable']]
            if coarse:
                raw_ref = self.raster(coarse[-1]['path'])
                ref = detector_levels(raw_ref['det1'], raw_ref['det2'], combine == 'unmix')
        metric = detector_metric(a, b, params, ref)
        if mode == 'det1':
            value, metric_label = a, f'Raw det1 ({hdr["unit"]})'
        elif mode == 'det2':
            value, metric_label = b, f'Raw det2 ({hdr["unit"]})'
        else:
            value = metric
            dets = vector(params.get('detectors', 1))
            metric_label = f'AutoFind det{2 if dets == [2] else 1} ({hdr["unit"]})' if not both else f'AutoFind {combine} ({hdr["unit"]})' if combine == 'sum' else f'Reconstructed AutoFind {combine} (dimensionless; coarse reference)'
            if not run['has_log']:
                metric_label = f'Raw det1 ({hdr["unit"]}; no log, detector choice unknown)'
        axes = data['axes']
        if mode == 'ratio':
            if len(axes) != 2:
                raise ValueError('Line-normalized ratio applies to XY rasters, not z scans.')
            value = line_ratio(metric)['ratio']
            metric_label = 'Line-normalized ratio (dimensionless; autoFindDetect of AutoFind metric)'
        notes = []
        if both and combine != 'sum' and mode in ('metric', 'ratio'):
            notes.append('Formula ported from current MATLAB source. The in-memory detRef/early-stop partial-line state was not logged; final latest coarse-file medians/unmix fit reconstruct that reference, not guaranteed identical to the historical live metric.')
        if s['shared_file']:
            notes.append('File basename shared by logs; earlier noData evidence is unavailable. This raster is the file currently on disk.')
        if not np.isfinite(value).any():
            notes.append('No acquired finite detector values; unmeasured pixels remain blank.')
        unit = 'dimensionless' if mode == 'ratio' or mode == 'metric' and both and combine != 'sum' else hdr['unit']
        out = {'scan': s, 'axes': [ax[0] for ax in axes], 'metric_label': metric_label, 'unit': unit, 'stored_unit': hdr['unit'],
               'values': value, 'notes': notes, 'detector_reference': ref, 'mode': mode}
        if len(axes) == 2:
            x, y = axes[0][1], axes[1][1]
            out.update({'x': x, 'y': y, 'x_edges': axis_edges(x), 'y_edges': axis_edges(y), 'values': value.T,
                        'ratio': line_ratio(metric)['ratio'].T, 'blob': blob(metric, x, y) if s['kind'] != 'coarse' else None})
            candidates = []
            if s['kind'] in ('coarse', 'patch'):
                threshold = number(params.get('weakRatio'), 2)
                sigma = number(params.get('weakSigma'), 5)
                if both and combine not in ('sum', 'mean'):
                    ref_use = ref if ref is not None and np.all(np.isfinite(ref)) else detector_levels(a, b, combine == 'unmix')
                    r1 = np.maximum(a - ref_use[2] * (b - ref_use[1]), 0) / max(ref_use[0], EPS) if combine == 'unmix' else a / max(ref_use[0], EPS)
                    r2 = b / max(ref_use[1], EPS)
                    r1[np.isnan(a) | np.isnan(b)] = np.nan
                    r2[np.isnan(a) | np.isnan(b)] = np.nan
                    candidates = [dict(c, det=1) for c in detect_candidates(r1, x, y, sigma, threshold)]
                    step = np.median(np.abs(np.diff(x))) if len(x) > 1 else np.nan
                    if not np.isfinite(step):
                        step = np.median(np.abs(np.diff(y))) if len(y) > 1 else np.nan
                    # detectIn: each det2 hit is compared with the det1 list only; unmatched det2 hits are appended after.
                    unmatched = []
                    for c in detect_candidates(r2, x, y, sigma, threshold):
                        c = dict(c, det=2)
                        distances = [np.hypot(c['pos1'] - q['pos1'], c['pos2'] - q['pos2']) for q in candidates]
                        if distances and min(distances) <= 1.5 * step:
                            j = int(np.argmin(distances))
                            if c['ratio'] > candidates[j]['ratio']:
                                candidates[j] = c
                        else:
                            unmatched.append(c)
                    candidates = sorted(candidates + unmatched, key=lambda c: -c['ratio'])
                else:
                    candidates = detect_candidates(metric, x, y, sigma, threshold)
                for c in candidates:
                    c.update({'provenance': 'recomputed detection (current MATLAB formula; not decoded L.events.data)', 'scan_id': s['id']})
            out['candidates'] = candidates
        else:
            out['z'] = axes[0][1]
        return out

    def map_batch(self, source, mode='metric', offset=0, limit=12):
        run = self.run(source)
        coarse = [s for s in run['scans'] if s['kind'] == 'coarse']
        offset, limit = max(0, int(offset)), min(24, max(1, int(limit)))
        maps, errors = [], []
        for s in coarse[offset:offset + limit]:
            try:
                maps.append(self.scan(source, s['id'], mode))
            except Exception as exc:
                errors.append({'scan_id': s['id'], 'error': str(exc)})
        return {'maps': maps, 'errors': errors, 'total': len(coarse), 'next_offset': min(offset + limit, len(coarse))}

    def pursuit(self, source, pursuit_id):
        run = self.run(source)
        g = next((g for g in run['pursuits'] if g['id'] == pursuit_id), None)
        if g is None:
            raise ValueError('Unknown candidate pursuit')
        return dict(g, scans=[s for s in run['scans'] if s['id'] in g['scan_ids']],
                    ratio_note='Particle ratios, when present, are logged blob contrasts, not line-normalized candidate ratios. Message ratios are rounded; no bleaching verdict is inferred from this plot.')
