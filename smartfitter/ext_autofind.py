"""Lazy AutoFind endpoints; parsing and plotting live in the GUI-independent nvfit modules."""
from __future__ import annotations

from nvfit.autofind import AutoFindStore

from .core import clean


def _store(ws):
    with ws._lock:
        if not hasattr(ws, '_autofind'):
            ws._autofind = AutoFindStore()
        return ws._autofind


def autofind_discover(ws, path):
    return clean(_store(ws).discover(path))


def autofind_run(ws, source):
    return clean(_store(ws).payload(source))


def autofind_maps(ws, source, mode='metric', offset=0, limit=12):
    return clean(_store(ws).map_batch(source, mode, offset, limit))


def autofind_scan(ws, source, scan_id, mode='metric'):
    return clean(_store(ws).scan(source, scan_id, mode))


def autofind_pursuit(ws, source, pursuit_id):
    return clean(_store(ws).pursuit(source, pursuit_id))


def autofind_compare(ws, sources):
    if not sources or len(sources) > 24:
        raise ValueError('Select 1–24 runs for comparison.')
    store = _store(ws)
    return clean({'runs': [store.payload(p) for p in dict.fromkeys(sources)]})


def autofind_export(ws, sources, directory, mode='metric', contours=False, scale='live'):
    from nvfit.autofind_figures import export_figure_set

    if not sources or len(sources) > 24:
        raise ValueError('Select 1–24 runs to export.')
    return clean(export_figure_set(_store(ws), sources, directory, mode, contours=contours, scale=scale))


ENDPOINTS = {name: fn for name, fn in list(globals().items()) if name.startswith('autofind_')}
