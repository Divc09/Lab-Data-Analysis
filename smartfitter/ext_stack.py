"""SmartFitter endpoints for 3D scan stacks and sequential-map spot tracking.

Maps are used at native resolution (no regridding). See nvfit/scan_stack.py.
"""
from __future__ import annotations

from nvfit import scan_stack as ss

from .core import _map_arrays, clean


def _loader(ws, source: str):
    def load(path, detector, mode):
        return _map_arrays(ws, path, detector, mode, source)
    return load


def _opts_source(opts: dict | None) -> str:
    return str((opts or {}).get("source", "auto"))


def stack_build(ws, paths, detector="detector1", mode="contrast", opts=None):
    src = _opts_source(opts)
    stack = ss.build_stack(paths, detector, mode, source=src, loader=_loader(ws, src))
    return clean(stack.to_dict(include_maps=bool((opts or {}).get("include_maps", False))))


def stack_trajectory(ws, paths, detectors=None, mode="contrast", opts=None):
    opts = opts or {}
    dets = [detectors] if isinstance(detectors, str) else list(detectors or ["detector1"])
    src = _opts_source(opts)
    stack = ss.build_stack(paths, dets[0], mode, source=src, loader=_loader(ws, src))
    res = ss.stack_spot_trajectory(stack, dets, opts, max_jump_fwhm=float(opts.get("max_jump_fwhm", 3.0)))
    res["stack"] = stack.to_dict()
    return clean(res)


def track_spots(ws, paths, detector="detector1", mode="contrast", opts=None):
    opts = opts or {}
    src = _opts_source(opts)
    gate = opts.get("gate")
    res = ss.track_spots(paths, detector, mode, opts, gate_fwhm=float(opts.get("gate_fwhm", 3.0)),
                         gate=float(gate) if gate not in (None, "") else None,
                         source=src, loader=_loader(ws, src))
    return clean(res)


ENDPOINTS = {"stack_build": stack_build, "stack_trajectory": stack_trajectory, "track_spots": track_spots}
