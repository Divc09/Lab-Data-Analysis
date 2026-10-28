"""Acquisition parameters saved in the .mat file (params / scanInfo / metadata structs), flattened for the Info tab.

Values are shown as stored: no unit conversion, no interpretation. Large arrays are summarised, not dumped."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque, mat_struct

SECTIONS = ("params", "scanInfo", "metadata", "Data")
_MAX_ITEMS = 12      # arrays up to this many elements are listed in full
_MAX_DEPTH = 4


def _fmt(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.10g}" if isinstance(value, (float, np.floating)) else str(int(value))
    return str(value)


def _flatten(value, prefix: str, out: list, depth: int, skip_numeric: bool) -> None:
    if isinstance(value, mat_struct):
        if depth >= _MAX_DEPTH:
            out.append([prefix, f"struct ({len(value._fieldnames)} fields)"])
            return
        for name in value._fieldnames:
            _flatten(getattr(value, name), f"{prefix}.{name}", out, depth + 1, skip_numeric)
        return
    if isinstance(value, MatlabOpaque):  # MATLAB string/datetime objects cannot be decoded outside MATLAB
        out.append([prefix, "(MATLAB object: open in MATLAB to read)"])
        return
    if isinstance(value, np.ndarray):
        # {label, value; ...} cells such as metadata.dataInfo read as named settings
        if value.dtype == object and value.ndim == 2 and value.shape[1] == 2 and all(isinstance(v, str) for v in value[:, 0]):
            for label, v in value:
                _flatten(v, f"{prefix}.{label}", out, depth + 1, skip_numeric)
            return
        if value.dtype == object:
            # struct lists (e.g. pulse-sequence steps): one compact row per element
            if 0 < value.size <= 64 and all(isinstance(v, mat_struct) for v in value.ravel()):
                for i, v in enumerate(value.ravel()):
                    out.append([f"{prefix}[{i + 1}]", " · ".join(f"{n} {_fmt(getattr(v, n))}" for n in v._fieldnames if not isinstance(getattr(v, n), mat_struct))])
                return
            if value.size <= _MAX_ITEMS and all(isinstance(v, (str, bytes, int, float, np.generic)) for v in value.ravel()):
                out.append([prefix, ", ".join(_fmt(v) for v in value.ravel())])
            elif value.size <= _MAX_ITEMS and depth < _MAX_DEPTH:
                for i, v in enumerate(value.ravel()):
                    _flatten(v, f"{prefix}[{i + 1}]", out, depth + 1, skip_numeric)
            else:
                out.append([prefix, f"cell {'×'.join(map(str, value.shape))}"])
            return
        if value.size == 0:
            out.append([prefix, "[]"])
        elif value.size <= _MAX_ITEMS and value.ndim <= 1:
            out.append([prefix, "[" + ", ".join(_fmt(v) for v in value.ravel()) + "]"])
        elif not skip_numeric:
            out.append([prefix, f"{value.dtype} array {'×'.join(map(str, value.shape))}"])
        return
    out.append([prefix, _fmt(value)])


def mat_params(ws, path):
    raw = loadmat(Path(path), struct_as_record=False, squeeze_me=True)
    sections = []
    for key in SECTIONS:
        if key not in raw:
            continue
        rows: list = []
        # Data holds the measured arrays; only its scalar/short settings are useful here.
        _flatten(raw[key], key, rows, 0, skip_numeric=key == "Data")
        sections.append({"name": key, "rows": rows})
    if not sections:  # legacy files: list any top-level scalar settings
        rows = []
        for key, value in raw.items():
            if not key.startswith("__"):
                _flatten(value, key, rows, 1, skip_numeric=True)
        sections.append({"name": "file", "rows": rows})
    return {"sections": sections}


ENDPOINTS = {"mat_params": mat_params}
