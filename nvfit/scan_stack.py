"""3D scan stacks (2D maps at different fixed third-axis values) and spot tracking.

Conventions
-----------
* Maps are never resampled, regridded or interpolated.  Each slice keeps its native
  1D column/row coordinate vectors and its native 2D value map.
* A trace's ``scan_axes = (a0, a1)`` names the axes of the map *columns* (``x2d[0, :]``)
  and *rows* (``y2d[:, 0]``) respectively.  Axis names are preserved; positions are
  reported keyed by normalised axis key (``"x"``, ``"y"``, ``"z"``) so XY, YX, XZ, YZ
  orientations are all handled without transposing data.
* Lengths are in the stage units stored in the file.  The MAT files do not record the
  stage unit, so results are labelled "stage units (as recorded)".
* Position uncertainty: ``SpotFit`` carries no covariance, so the error used is the
  proxy ``FWHM_major / SNR`` (flagged ``err_source="FWHM/SNR proxy"``).  If a future
  SpotFit provides ``center_x_err``/``center_y_err`` those are used instead.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .scan_spots import detect_fluorescent_spots

UNITS = "stage units (as recorded; unit not stored in file)"
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


# ---------------------------------------------------------------------------
# helpers: axis names, filename / metadata parsing
# ---------------------------------------------------------------------------

def axis_key(name: str) -> str:
    """Normalise an axis label ("x", "Stage Z", ...) to "x"/"y"/"z" (else lower-cased name)."""
    text = str(name).strip().lower()
    if text in ("x", "y", "z"):
        return text
    m = re.search(r"(?:^|[^a-z])([xyz])(?:[^a-z]|$)", text)
    return m.group(1) if m else text


def _num(token: str) -> float:
    """'n784' -> -784, '9936_5' -> 9936.5 (underscore = decimal point)."""
    neg = token.startswith("n")
    token = token[1:] if neg else token
    value = float(token.replace("_", "."))
    return -value if neg else value


def parse_axis_from_filename(file_name: str, axis: str) -> float | None:
    """Parse e.g. ``Z10006``, ``Z_10022``, ``Z9936_5``, ``Y826_9``, ``Xn2244_4`` for ``axis``."""
    stem = Path(file_name).stem
    pat = re.compile(
        r"(?<![0-9])" + re.escape(axis) + r"_?(n?\d+(?:_\d+)?)(?![A-Za-z0-9])", re.IGNORECASE
    )
    for m in pat.finditer(stem):
        try:
            return _num(m.group(1).lower())
        except ValueError:
            continue
    return None


def parse_trailing_series_value(file_name: str) -> float | None:
    """Fallback for names like ``..._XYSeries1_10027`` or ``...Scanat10035`` (assumed third-axis value)."""
    stem = Path(file_name).stem
    m = re.search(r"(?:at|_)(n?\d{3,6})(?:_(\d+))?$", stem, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).lower() + (("_" + m.group(2)) if m.group(2) else "")
    try:
        return _num(token)
    except ValueError:
        return None


def metadata_axis_value(metadata: dict | None, axis: str) -> tuple[float | None, str]:
    """Look for a stored fixed stage position of ``axis`` in trace/dataset metadata."""
    if not metadata:
        return None, ""
    pat = re.compile(
        rf"^(?:stage_|fixed_|current_|static_|hold_)?{axis}(?:_?(?:pos|position|stage|um|nm|fixed))?$|"
        rf"^(?:stage|fixed|current|static)_{axis}(?:_.*)?$|^{axis}_(?:pos|position|stage)(?:_.*)?$",
        re.IGNORECASE,
    )
    for key, value in metadata.items():
        if not pat.match(str(key)):
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f, str(key)
    return None, ""


def parse_file_time(path: str | Path) -> tuple[float, str]:
    """Epoch seconds and source. Prefers a ``YYYYMMDD_HHMMSS`` stamp in the file name, then ``MMDD_HHMM``
    at its start, else mtime."""
    p = Path(path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = float("nan")
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?!\d)", p.name)
    if m:
        try:
            return _dt.datetime(*(int(g) for g in m.groups())).timestamp(), "filename (YYYYMMDD_HHMMSS, local time)"
        except ValueError:
            pass
    m = re.match(r"^(\d{2})(\d{2})_(\d{2})(\d{2})", p.name)
    if m:
        year = _dt.datetime.fromtimestamp(mtime).year if math.isfinite(mtime) else _dt.datetime.now().year
        try:
            t = _dt.datetime(year, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            return t.timestamp(), "filename (MMDD_HHMM; year taken from file mtime)"
        except ValueError:
            pass
    return mtime, "file mtime"


def linear_fit(t: Sequence[float], v: Sequence[float]) -> dict[str, Any]:
    """Ordinary least squares v = a + b t with standard error of b (None if n < 3)."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    n = int(len(t))
    out: dict[str, Any] = {"n": n, "slope": None, "slope_se": None, "intercept": None, "r_squared": None}
    if n < 2 or np.ptp(t) == 0:
        return out
    tm, vm = t.mean(), v.mean()
    sxx = float(np.sum((t - tm) ** 2))
    b = float(np.sum((t - tm) * (v - vm)) / sxx)
    a = float(vm - b * tm)
    out.update(slope=b, intercept=a)
    resid = v - (a + b * t)
    if n >= 3:
        s2 = float(np.sum(resid ** 2) / (n - 2))
        out["slope_se"] = math.sqrt(s2 / sxx)
        sst = float(np.sum((v - vm) ** 2))
        out["r_squared"] = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else None
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

MapLoader = Callable[[str, str, str], tuple[Any, np.ndarray, np.ndarray, np.ndarray]]


def default_map_loader(path: str, detector: str, mode: str, source: str = "auto"):
    """Load a 2D map natively: returns (trace, col_centres, row_centres, value_map)."""
    from .io_mat import load_experiment_dataset

    ds = load_experiment_dataset(path, mode)
    det = detector if detector in ds.detectors else (ds.available_detectors[0] if ds.available_detectors else None)
    if det is None or det not in ds.detectors:
        raise ValueError(f"No detector data in {Path(path).name}")
    tr = ds.detectors[det]
    if tr.scan_dim != "scan2d" or tr.z2d is None:
        raise ValueError(f"{Path(path).name} is not a 2D map")
    x = np.asarray(tr.x2d, float)[0, :]
    y = np.asarray(tr.y2d, float)[:, 0]
    z = tr.z2d
    if source == "signal" and tr.signal2d is not None:
        z = tr.signal2d
    elif source == "reference" and tr.reference2d is not None:
        z = tr.reference2d
    elif source == "auto":
        best = -1.0
        for m in [m for m in (tr.signal2d, tr.reference2d) if m is not None] + [tr.z2d]:
            m = np.asarray(m, float)
            finite = m[np.isfinite(m)]
            if finite.size and np.nanmean(np.abs(finite)) > 0:
                score = float(np.nanstd(finite) / (np.nanmean(np.abs(finite)) + 1e-300))
                if score > best:
                    best, z = score, m
    return tr, x, y, np.asarray(z, float)


@dataclass
class StackSlice:
    path: str
    file_name: str
    detector: str
    axes: tuple[str, str]            # names of (column axis, row axis) as stored in the file
    axis_keys: tuple[str, str]       # normalised keys
    x: np.ndarray                    # native column centres
    y: np.ndarray                    # native row centres
    values: np.ndarray               # native map, shape (len(y), len(x)); never resampled
    third_axis: str
    third_value: float | None
    third_source: str                # "metadata:<key>", "filename", "filename (trailing number, assumed)", ""
    time: float
    time_source: str

    @property
    def step(self) -> tuple[float, float]:
        sx = float(np.median(np.abs(np.diff(self.x)))) if len(self.x) > 1 else float("nan")
        sy = float(np.median(np.abs(np.diff(self.y)))) if len(self.y) > 1 else float("nan")
        return sx, sy

    def summary(self) -> dict[str, Any]:
        sx, sy = self.step
        return {
            "path": self.path, "file_name": self.file_name, "detector": self.detector,
            "axes": list(self.axes), "shape": list(self.values.shape),
            "x_range": [float(self.x.min()), float(self.x.max())],
            "y_range": [float(self.y.min()), float(self.y.max())],
            "x_step": sx, "y_step": sy,
            "third_axis": self.third_axis, "third_value": self.third_value,
            "third_source": self.third_source, "time": self.time, "time_source": self.time_source,
        }


@dataclass
class ScanStack:
    detector: str
    mode: str
    source: str
    plane_axes: tuple[str, str]        # of the reference (first) slice, names as stored
    plane_keys: tuple[str, str]
    third_axis: str
    slices: list[StackSlice]           # sorted by third-axis value
    warnings: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    loader: MapLoader | None = None
    units: str = UNITS

    @property
    def paths(self) -> list[str]:
        return [s.path for s in self.slices] + [s["path"] for s in self.skipped]

    @property
    def z_values(self) -> list[float]:
        return [s.third_value for s in self.slices]

    def to_dict(self, include_maps: bool = False) -> dict[str, Any]:
        sl = []
        for s in self.slices:
            d = s.summary()
            if include_maps:
                d["x"], d["y"], d["values"] = s.x.tolist(), s.y.tolist(), s.values.tolist()
            sl.append(d)
        return {
            "detector": self.detector, "mode": self.mode, "source": self.source,
            "plane_axes": list(self.plane_axes), "plane_keys": list(self.plane_keys),
            "third_axis": self.third_axis, "units": self.units,
            "z_values": self.z_values, "slices": sl, "skipped": self.skipped, "warnings": self.warnings,
            "regridded": False,
        }


def _third_value(tr: Any, file_name: str, third: str, dataset_meta: dict | None = None) -> tuple[float | None, str]:
    v, key = metadata_axis_value(getattr(tr, "metadata", None), third)
    if v is None:
        v, key = metadata_axis_value(dataset_meta, third)
    if v is not None:
        return v, f"metadata:{key}"
    v = parse_axis_from_filename(file_name, third)
    if v is not None:
        return v, "filename"
    v = parse_trailing_series_value(file_name)
    if v is not None:
        return v, "filename (trailing number, assumed)"
    return None, ""


def build_stack(
    paths: Sequence[str | Path],
    detector: str = "detector1",
    mode: str = "contrast",
    *,
    source: str = "auto",
    loader: MapLoader | None = None,
) -> ScanStack:
    """Group 2D maps into a stack ordered by the fixed third-axis value (no regridding).

    ``loader(path, detector, mode)`` must return ``(trace, col_centres, row_centres, map)``;
    it defaults to loading the MAT file natively.
    """
    load = loader or (lambda p, d, m: default_map_loader(p, d, m, source))
    warnings: list[str] = []
    skipped: list[dict[str, str]] = []
    slices: list[StackSlice] = []
    for p in paths:
        p = str(p)
        try:
            tr, x, y, vals = load(p, detector, mode)
        except Exception as exc:  # keep going; report
            skipped.append({"path": p, "reason": f"load failed: {exc}"})
            continue
        axes = tuple(getattr(tr, "scan_axes", ()) or ())
        if len(axes) != 2:
            skipped.append({"path": p, "path_name": Path(p).name, "reason": "trace has no two named scan axes"})
            continue
        keys = (axis_key(axes[0]), axis_key(axes[1]))
        third_set = {"x", "y", "z"} - set(keys)
        third = next(iter(third_set)) if len(third_set) == 1 else ""
        file_name = Path(p).name
        tv, tsrc = _third_value(tr, file_name, third) if third else (None, "")
        t, tsrc_time = parse_file_time(p)
        slices.append(StackSlice(
            path=p, file_name=file_name, detector=getattr(tr, "detector_id", detector),
            axes=(str(axes[0]), str(axes[1])), axis_keys=keys,
            x=np.asarray(x, float), y=np.asarray(y, float), values=np.asarray(vals, float),
            third_axis=third, third_value=tv, third_source=tsrc, time=t, time_source=tsrc_time,
        ))
    if not slices:
        return ScanStack(detector, mode, source, ("", ""), ("", ""), "", [], warnings + ["No usable 2D maps."], skipped, loader)

    ref = slices[0]
    plane_set = set(ref.axis_keys)
    kept: list[StackSlice] = []
    for s in slices:
        if set(s.axis_keys) != plane_set:
            skipped.append({"path": s.path, "reason": f"different plane {s.axes} vs {ref.axes}"})
            warnings.append(f"{s.file_name}: plane {s.axes} differs from reference plane {ref.axes}; excluded.")
            continue
        if s.third_value is None:
            skipped.append({"path": s.path, "reason": f"no {s.third_axis}-axis value in metadata or filename"})
            warnings.append(f"{s.file_name}: fixed {s.third_axis} value not found; excluded.")
            continue
        kept.append(s)
    kept.sort(key=lambda s: s.third_value)
    if kept:
        ref = kept[0]
        for s in kept:
            if s.axes != ref.axes:
                warnings.append(f"{s.file_name}: axis order {s.axes} differs from {ref.axes} (kept native; positions are keyed by axis name).")
            if s.values.shape != ref.values.shape:
                warnings.append(f"{s.file_name}: map shape {s.values.shape} differs from {ref.values.shape}.")
            else:
                rs, ss = ref.step, s.step
                if not (np.allclose(rs, ss, rtol=1e-3, equal_nan=True) or np.allclose(rs, ss[::-1], rtol=1e-3, equal_nan=True)):
                    warnings.append(f"{s.file_name}: grid step {ss} differs from {rs}.")
        zs = [s.third_value for s in kept]
        if len(set(zs)) < len(zs):
            warnings.append("Duplicate third-axis values present (repeat maps at the same position).")
        heur = [s.file_name for s in kept if "assumed" in s.third_source]
        if heur:
            warnings.append(f"{len(heur)} slice(s) use the trailing filename number as {ref.third_axis} (assumed, not verified).")
    return ScanStack(
        detector=detector, mode=mode, source=source,
        plane_axes=ref.axes if kept else ("", ""), plane_keys=ref.axis_keys if kept else ("", ""),
        third_axis=ref.third_axis if kept else "", slices=kept, warnings=warnings, skipped=skipped, loader=loader,
    )


# ---------------------------------------------------------------------------
# spot detection plumbing
# ---------------------------------------------------------------------------

_DETECT_KEYS = ("polarity", "threshold_sigma", "min_pixels", "max_spots", "background_window_pixels")


def _detect_kwargs(opts: dict | None) -> dict[str, Any]:
    opts = dict(opts or {})
    if "threshold" in opts and "threshold_sigma" not in opts:
        opts["threshold_sigma"] = opts["threshold"]
    if "bg_window" in opts and "background_window_pixels" not in opts:
        opts["background_window_pixels"] = opts["bg_window"] or 0
    pol = opts.get("polarity")
    if pol is not None:
        opts["polarity"] = {"Auto": "auto", "Bright": "bright", "Dark": "dark", "Bright and dark": "both"}.get(pol, str(pol).lower())
    kw = {k: opts[k] for k in _DETECT_KEYS if k in opts and opts[k] is not None}
    for k in ("threshold_sigma",):
        if k in kw:
            kw[k] = float(kw[k])
    for k in ("min_pixels", "max_spots", "background_window_pixels"):
        if k in kw:
            kw[k] = int(kw[k])
    return kw


def _spot_error(spot: Any) -> tuple[float, float, str]:
    ex = getattr(spot, "center_x_err", None)
    ey = getattr(spot, "center_y_err", None)
    if ex is not None and ey is not None and np.isfinite(ex) and np.isfinite(ey):
        return float(ex), float(ey), "fit uncertainty"
    snr = float(spot.snr) if np.isfinite(spot.snr) and spot.snr > 0 else float("nan")
    e = spot.fwhm_major / snr if np.isfinite(snr) else float("nan")
    return e, e, "FWHM/SNR proxy"


def _detect_slice(s: StackSlice, kw: dict) -> list[dict[str, Any]]:
    """Spots of one slice as dicts keyed by axis key; fit failures are dropped."""
    try:
        res = detect_fluorescent_spots(s.x, s.y, s.values, **kw)
    except ValueError:
        return []
    out = []
    for sp in res.spots:
        if not sp.fit_success:
            continue
        ex, ey, esrc = _spot_error(sp)
        k0, k1 = s.axis_keys
        out.append({
            "pos": {k0: float(sp.center_x), k1: float(sp.center_y)},
            "err": {k0: ex, k1: ey}, "err_source": esrc,
            "fwhm": float(sp.fwhm_major), "fwhm_minor": float(sp.fwhm_minor),
            "amplitude": float(sp.amplitude), "snr": float(sp.snr), "r_squared": float(sp.r_squared),
            "polarity": sp.polarity,
        })
    return out


def _dist(a: dict, b: dict, keys: Sequence[str]) -> float:
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in keys))


# ---------------------------------------------------------------------------
# item 9: trajectory through a z stack
# ---------------------------------------------------------------------------

def _link_stack(stack: ScanStack, detections: list[list[dict]], max_jump_fwhm: float,
                seed_hint: dict | None = None, seed_hint_keys: Sequence[str] = ()) -> tuple[list[dict | None], int]:
    n = len(detections)
    keys = stack.plane_keys
    seed_idx, seed = -1, None
    if seed_hint is not None:
        best = None
        for i, dets in enumerate(detections):
            for d in dets:
                dist = _dist(d["pos"], seed_hint, keys)
                if dist <= max_jump_fwhm * d["fwhm"] and (best is None or dist < best[0]):
                    best = (dist, i, d)
        if best:
            seed_idx, seed = best[1], best[2]
    if seed is None:
        for i, dets in enumerate(detections):
            for d in dets:
                if seed is None or abs(d["amplitude"]) > abs(seed["amplitude"]):
                    seed_idx, seed = i, d
    chain: list[dict | None] = [None] * n
    if seed is None:
        return chain, -1
    chain[seed_idx] = seed
    for direction in (1, -1):
        last = seed
        i = seed_idx + direction
        while 0 <= i < n:
            cands = detections[i]
            if cands:
                dists = [_dist(c["pos"], last["pos"], keys) for c in cands]
                j = int(np.argmin(dists))
                if dists[j] <= max_jump_fwhm * last["fwhm"]:
                    chain[i] = cands[j]
                    last = cands[j]
            i += direction
    return chain, seed_idx


def _trajectory_for(stack: ScanStack, opts: dict | None, max_jump_fwhm: float,
                    seed_hint: dict | None = None) -> dict[str, Any]:
    kw = _detect_kwargs(opts)
    detections = [_detect_slice(s, kw) for s in stack.slices]
    chain, seed_idx = _link_stack(stack, detections, max_jump_fwhm, seed_hint)
    k0, k1 = stack.plane_keys
    points = []
    for s, c, dets in zip(stack.slices, chain, detections):
        row = {"z": s.third_value, "z_source": s.third_source, "file_name": s.file_name,
               "n_detected": len(dets), "linked": c is not None}
        if c is not None:
            row.update({"pos": c["pos"], "err": c["err"], "err_source": c["err_source"],
                        "fwhm": c["fwhm"], "amplitude": c["amplitude"], "snr": c["snr"]})
        points.append(row)
    linked = [p for p in points if p["linked"]]
    walk: dict[str, Any] = {}
    for k in (k0, k1):
        f = linear_fit([p["z"] for p in linked], [p["pos"][k] for p in linked])
        f["units"] = f"{k} per {stack.third_axis} (both in stage units)"
        walk[k] = f
    warn = []
    if len(linked) < 3:
        warn.append("Fewer than 3 slices linked; walk slope SE unavailable or unreliable.")
    if linked and any(p["err_source"] != "fit uncertainty" for p in linked):
        warn.append("Position errors are a FWHM/SNR proxy (SpotFit has no covariance); slope SE comes from scatter about the line (OLS), not from these errors.")
    return {"detector": stack.detector, "points": points, "walk": walk, "seed_index": seed_idx,
            "n_linked": len(linked), "n_slices": len(points), "warnings": warn}


def stack_spot_trajectory(
    stack: ScanStack,
    detectors: Sequence[str] | str | None = None,
    opts: dict | None = None,
    *,
    max_jump_fwhm: float = 3.0,
) -> dict[str, Any]:
    """Track the brightest spot through a z stack; optionally compare two detectors.

    For extra detectors the stack is rebuilt (same files, same loader) and the seed is
    chosen as the spot nearest the first detector's seed, so both detectors follow the
    same physical feature.  ``offset`` = detector[1] - detector[0] per z, keyed by axis.
    """
    if isinstance(detectors, str):
        detectors = [detectors]
    dets = list(detectors) if detectors else [stack.detector]
    out: dict[str, Any] = {"plane_axes": list(stack.plane_axes), "plane_keys": list(stack.plane_keys),
                           "third_axis": stack.third_axis, "units": stack.units,
                           "max_jump_fwhm": max_jump_fwhm, "warnings": list(stack.warnings),
                           "detectors": {}, "offset": None}
    if not stack.slices:
        out["warnings"].append("Empty stack.")
        return out
    stacks = {}
    for d in dets:
        if d == stack.detector:
            stacks[d] = stack
        else:
            stacks[d] = build_stack(stack.paths, d, stack.mode, source=stack.source, loader=stack.loader)
            out["warnings"] += [f"[{d}] {w}" for w in stacks[d].warnings if w not in stack.warnings]
    first = None
    for d in dets:
        hint = None
        if first is not None:
            fp = [p for p in first["points"] if p["linked"]]
            if fp:
                seed = first["points"][first["seed_index"]]
                hint = seed["pos"] if seed.get("linked") else fp[0]["pos"]
        res = _trajectory_for(stacks[d], opts, max_jump_fwhm, hint)
        out["detectors"][d] = res
        out["warnings"] += [f"[{d}] {w}" for w in res["warnings"]]
        if first is None:
            first = res
    if len(dets) >= 2:
        a, b = out["detectors"][dets[0]], out["detectors"][dets[1]]
        bz = {p["z"]: p for p in b["points"] if p["linked"]}
        k0, k1 = stack.plane_keys
        rows = []
        for p in a["points"]:
            q = bz.get(p["z"])
            if not p["linked"] or q is None:
                continue
            rows.append({"z": p["z"],
                         "offset": {k: q["pos"][k] - p["pos"][k] for k in (k0, k1)},
                         "err": {k: math.hypot(q["err"][k], p["err"][k]) for k in (k0, k1)}})
        fits = {k: linear_fit([r["z"] for r in rows], [r["offset"][k] for r in rows]) for k in (k0, k1)}
        out["offset"] = {"from": dets[0], "to": dets[1], "definition": f"{dets[1]} - {dets[0]} (stage units)",
                         "rows": rows, "fit_vs_z": fits}
    return out


# ---------------------------------------------------------------------------
# item 14: tracking across sequential maps (same plane, different times)
# ---------------------------------------------------------------------------

def track_spots(
    paths: Sequence[str | Path],
    detector: str = "detector1",
    mode: str = "contrast",
    opts: dict | None = None,
    *,
    gate_fwhm: float = 3.0,
    gate: float | None = None,
    source: str = "auto",
    loader: MapLoader | None = None,
) -> dict[str, Any]:
    """Detect spots in each map and match them between consecutive maps (Hungarian + gate).

    Maps are used at native resolution. Gate = ``gate`` (stage units) if given, else
    ``gate_fwhm`` x median FWHM of the spots in the earlier map. Drift per file is the
    cumulative sum of the median displacement of spots matched between consecutive maps
    (common-mode); ``drift_rate`` is an OLS fit of that drift vs time in minutes.
    """
    load = loader or (lambda p, d, m: default_map_loader(p, d, m, source))
    kw = _detect_kwargs(opts)
    warnings: list[str] = []
    files = []
    for p in paths:
        p = str(p)
        try:
            tr, x, y, vals = load(p, detector, mode)
        except Exception as exc:
            warnings.append(f"{Path(p).name}: load failed: {exc}")
            continue
        axes = tuple(getattr(tr, "scan_axes", ()) or ())
        if len(axes) != 2:
            warnings.append(f"{Path(p).name}: no two named scan axes; skipped.")
            continue
        t, tsrc = parse_file_time(p)
        files.append(StackSlice(p, Path(p).name, getattr(tr, "detector_id", detector), (str(axes[0]), str(axes[1])),
                                (axis_key(axes[0]), axis_key(axes[1])), np.asarray(x, float), np.asarray(y, float),
                                np.asarray(vals, float), "", None, "", t, tsrc))
    base = {"detector": detector, "units": UNITS, "warnings": warnings, "files": [], "tracks": [], "drift": [], "drift_rate": None}
    if not files:
        warnings.append("No usable maps.")
        return base
    files.sort(key=lambda s: (s.time if np.isfinite(s.time) else 0.0, s.file_name))
    ref = files[0]
    plane = set(ref.axis_keys)
    kept = []
    for s in files:
        if set(s.axis_keys) != plane:
            warnings.append(f"{s.file_name}: plane {s.axes} differs from {ref.axes}; skipped.")
        else:
            kept.append(s)
            if s.axes != ref.axes:
                warnings.append(f"{s.file_name}: axis order differs from {ref.axes} (positions keyed by axis name).")
    files = kept
    keys = ref.axis_keys
    t0 = files[0].time
    if len({s.time_source for s in files}) > 1:
        warnings.append("Mixed time sources (filename vs mtime); drift rate may be unreliable.")
    err_srcs = set()

    detections = [_detect_slice(s, kw) for s in files]
    tracks: list[dict[str, Any]] = []
    increments: list[dict[str, Any] | None] = [None]
    for i, (s, dets) in enumerate(zip(files, detections)):
        tmin = (s.time - t0) / 60.0 if np.isfinite(s.time) else float("nan")
        if i == 0:
            for d in dets:
                tracks.append({"track_id": len(tracks) + 1, "points": [{"file_index": 0, "t_min": tmin, **d}]})
            continue
        active = [tr for tr in tracks if tr["points"][-1]["file_index"] == i - 1]
        matched_pairs: list[tuple[dict, dict]] = []
        used_det: set[int] = set()
        if active and dets:
            cost = np.array([[_dist(tr["points"][-1]["pos"], d["pos"], keys) for d in dets] for tr in active])
            gates = np.array([gate if gate is not None else gate_fwhm * tr["points"][-1]["fwhm"] for tr in active])
            big = 1e12
            masked = np.where(cost <= gates[:, None], cost, big)
            rr, cc = linear_sum_assignment(masked)
            for r, c in zip(rr, cc):
                if masked[r, c] >= big:
                    continue
                prev = active[r]["points"][-1]
                active[r]["points"].append({"file_index": i, "t_min": tmin, **dets[c]})
                matched_pairs.append((prev, dets[c]))
                used_det.add(c)
        for c, d in enumerate(dets):
            if c not in used_det:
                tracks.append({"track_id": len(tracks) + 1, "points": [{"file_index": i, "t_min": tmin, **d}]})
        if matched_pairs:
            disp = {k: np.array([b["pos"][k] - a["pos"][k] for a, b in matched_pairs]) for k in keys}
            increments.append({"n": len(matched_pairs),
                               "median": {k: float(np.median(disp[k])) for k in keys},
                               "spread": {k: float(1.4826 * np.median(np.abs(disp[k] - np.median(disp[k])))) for k in keys}})
        else:
            increments.append(None)
            warnings.append(f"{s.file_name}: no spots matched to the previous map (gate too tight or no spots).")

    drift = []
    cum = {k: 0.0 for k in keys}
    broken = False
    for i, s in enumerate(files):
        inc = increments[i]
        if i > 0:
            if inc is None:
                broken = True
            elif not broken:
                for k in keys:
                    cum[k] += inc["median"][k]
        tmin = (s.time - t0) / 60.0 if np.isfinite(s.time) else float("nan")
        drift.append({"file_name": s.file_name, "t_min": tmin, "n_detected": len(detections[i]),
                      "n_matched": (inc["n"] if inc else (len(detections[0]) if i == 0 else 0)),
                      "drift": None if broken else dict(cum),
                      "increment_spread": inc["spread"] if inc else None})
    good = [d for d in drift if d["drift"] is not None]
    rate = {}
    for k in keys:
        f = linear_fit([d["t_min"] for d in good], [d["drift"][k] for d in good])
        f["units"] = f"{k} stage units per minute"
        rate[k] = f
    if len(good) < 3:
        warnings.append("Fewer than 3 files with drift estimates; drift-rate SE unavailable.")
    if good and np.ptp([d["t_min"] for d in good]) == 0:
        warnings.append("All files share one timestamp; drift rate undefined.")
    for tr in tracks:
        for p in tr["points"]:
            err_srcs.add(p["err_source"])
    return {
        "detector": detector, "plane_axes": list(ref.axes), "plane_keys": list(keys), "units": UNITS,
        "time_note": "t_min measured from the first map; filename MMDD_HHMM is scan start at 1-min resolution.",
        "gate": {"abs": gate, "fwhm_multiple": gate_fwhm},
        "position_err_source": sorted(err_srcs),
        "files": [{"file_name": s.file_name, "path": s.path, "time": s.time, "time_source": s.time_source,
                   "t_min": (s.time - t0) / 60.0 if np.isfinite(s.time) else None} for s in files],
        "tracks": tracks, "drift": drift, "drift_rate": rate, "warnings": warnings,
    }
