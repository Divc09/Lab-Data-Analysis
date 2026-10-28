"""SmartFitter endpoints for 3D scan stacks and sequential-map spot tracking.

Maps are used at native resolution (no regridding). See nvfit/scan_stack.py.
"""

from __future__ import annotations

from . import core

from nvfit import scan_stack as ss

from .core import _map_arrays, clean


def _loader(ws, source: str):
    def load(path, detector, mode):
        return _map_arrays(ws, path, detector, mode, source)

    return load


def _check_units(ws, paths, detectors, mode):
    from nvfit.units import unit, unit_scale
    from nvfit.scan_stack import axis_key

    reference = None
    for path in paths:
        for detector in detectors:
            try:
                tr = ws.trace(path, detector, mode)
            except (OSError, ValueError):
                continue  # existing loader reports unavailable maps
            meta = tr.metadata or {}
            axes = meta.get("axis_units") or {}
            if not isinstance(axes, dict):
                axes = {}
            units = {
                axis_key(a): unit(axes.get(axis_key(a), axes.get(a, meta.get("x_unit") or ""))) for a in tr.scan_axes
            }
            if reference is None:
                reference = (units, tr.file_name)
                continue
            previous, name = reference
            if set(units) != set(previous):
                raise ValueError(f"Stack: incompatible scan planes for {name} and {tr.file_name}")
            for key, value in units.items():
                if unit_scale(value, previous[key]) != 1:
                    raise ValueError(
                        f"Stack: different {key} units in {name} and {tr.file_name}; convert coordinates to the same recorded units first."
                    )


def _opts_source(opts: dict | None) -> str:
    return str((opts or {}).get("source", "auto"))


@core.analysis_result(aggregate=True)
def stack_build(ws, paths, detector="detector1", mode="contrast", opts=None):
    src = _opts_source(opts)
    _check_units(ws, paths, [detector], mode)
    stack = ss.build_stack(paths, detector, mode, source=src, loader=_loader(ws, src))
    return clean(stack.to_dict(include_maps=bool((opts or {}).get("include_maps", False))))


@core.analysis_result(aggregate=True)
def stack_trajectory(ws, paths, detectors=None, mode="contrast", opts=None):
    opts = opts or {}
    dets = [detectors] if isinstance(detectors, str) else list(detectors or ["detector1"])
    src = _opts_source(opts)
    _check_units(ws, paths, dets, mode)
    stack = ss.build_stack(paths, dets[0], mode, source=src, loader=_loader(ws, src))
    res = ss.stack_spot_trajectory(stack, dets, opts, max_jump_fwhm=float(opts.get("max_jump_fwhm", 3.0)))
    res["stack"] = stack.to_dict()
    return clean(res)


@core.analysis_result(aggregate=True)
def track_spots(ws, paths, detector="detector1", mode="contrast", opts=None):
    opts = opts or {}
    src = _opts_source(opts)
    _check_units(ws, paths, [detector], mode)
    gate = opts.get("gate")
    res = ss.track_spots(
        paths,
        detector,
        mode,
        opts,
        gate_fwhm=float(opts.get("gate_fwhm", 3.0)),
        gate=float(gate) if gate not in (None, "") else None,
        source=src,
        loader=_loader(ws, src),
    )
    return clean(res)


ENDPOINTS = {"stack_build": stack_build, "stack_trajectory": stack_trajectory, "track_spots": track_spots}
