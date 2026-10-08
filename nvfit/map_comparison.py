"""Coordinate-preserving, two-color comparisons of measured 2D maps."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


PALETTES = {
    "Pink / teal": ((1.0, 0.0, 0.6), (0.0, 1.0, 0.6)),
    "Green / magenta": ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
    "Red / cyan": ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
}


def axis_key(label: str) -> str:
    text = str(label).strip().lower().replace("µ", "u").replace("μ", "u")
    text = re.sub(r"\bstage\s*", "", text)
    text = re.sub(r"[\[\]()]", " ", text)
    text = " ".join(text.split())
    if text in {"x", "y", "z"}:
        text += " um"
    return text


def compatible_axes(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    return len(first) == len(second) == 2 and sorted(map(axis_key, first)) == sorted(map(axis_key, second))


def axis_edges(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=float).ravel()
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Map coordinates must be nonempty and finite.")
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    steps = np.diff(values)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError("Map coordinates must be strictly monotonic.")
    return np.r_[values[0] - steps[0] / 2, (values[:-1] + values[1:]) / 2, values[-1] + steps[-1] / 2]


@dataclass(frozen=True)
class MapOverlay:
    x_edges: np.ndarray
    y_edges: np.ndarray
    rgba: np.ndarray
    ranges: tuple[tuple[float, float], tuple[float, float]]
    primary: np.ndarray
    secondary: np.ndarray


def _ascending_map(x, y, z):
    x, y, z = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    if z.shape != (len(y), len(x)):
        raise ValueError("Map values do not match their coordinate grid.")
    xe, ye = axis_edges(x), axis_edges(y)
    if xe[-1] < xe[0]:
        xe, z = xe[::-1], z[:, ::-1]
    if ye[-1] < ye[0]:
        ye, z = ye[::-1], z[::-1, :]
    return xe, ye, z


def _sample_cells(xe, ye, z, x, y):
    # Split native pixel rectangles at the union of both maps' boundaries.
    # Values are copied from the measured cell, never interpolated or stretched.
    ix, iy = np.searchsorted(xe, x, side="right") - 1, np.searchsorted(ye, y, side="right") - 1
    valid_x, valid_y = (ix >= 0) & (ix < z.shape[1]), (iy >= 0) & (iy < z.shape[0])
    values = z[np.ix_(np.clip(iy, 0, z.shape[0] - 1), np.clip(ix, 0, z.shape[1] - 1))].copy()
    values[~(valid_y[:, None] & valid_x[None, :])] = np.nan
    return values


def _range(values, background_percentile):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Both comparison maps need at least one measured finite cell.")
    return float(np.percentile(finite, background_percentile)), float(np.max(finite))


def build_map_overlay(
    primary: tuple[np.ndarray, np.ndarray, np.ndarray],
    secondary: tuple[np.ndarray, np.ndarray, np.ndarray],
    *, shared: bool = False, background_percentile: float = 10.0,
    palette: str = "Pink / teal", max_cells: int = 2_000_000,
) -> MapOverlay:
    if not 0 <= background_percentile <= 100:
        raise ValueError("Overlay background percentile must be between 0 and 100.")
    first, second = _ascending_map(*primary), _ascending_map(*secondary)
    ranges = (_range(first[2], background_percentile), _range(second[2], background_percentile))
    if shared:
        common = _range(np.r_[first[2].ravel(), second[2].ravel()], background_percentile)
        ranges = (common, common)
    xe, ye = np.unique(np.r_[first[0], second[0]]), np.unique(np.r_[first[1], second[1]])
    if (len(xe) - 1) * (len(ye) - 1) > max_cells:
        raise ValueError("Comparison grid exceeds two million cells; crop or reduce the input maps.")
    x, y = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    a, b = (_sample_cells(*layer, x, y) for layer in (first, second))
    colors = PALETTES[palette]
    rgba = np.zeros((*a.shape, 4), dtype=float)
    for values, (low, high), color in zip((a, b), ranges, colors):
        normalized = np.zeros_like(values)
        if high > low:
            normalized = np.clip((values - low) / (high - low), 0, 1)
        rgba[:, :, :3] += np.nan_to_num(normalized)[:, :, None] * np.asarray(color)
    rgba[:, :, :3] = np.clip(rgba[:, :, :3], 0, 1)
    rgba[:, :, 3] = np.isfinite(a) | np.isfinite(b)
    return MapOverlay(xe, ye, rgba, ranges, a, b)
