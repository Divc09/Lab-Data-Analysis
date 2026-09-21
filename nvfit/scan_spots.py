from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label, median_filter
from scipy.optimize import curve_fit


@dataclass
class SpotFit:
    """Characterization of one fluorescent feature on a physical XY grid."""

    spot_id: int
    polarity: str
    center_x: float
    center_y: float
    sigma_major: float
    sigma_minor: float
    angle_deg: float
    amplitude: float
    background: float
    contrast_percent: float
    snr: float
    r_squared: float
    integrated_signal: float
    component_pixels: int
    fit_success: bool
    fit_message: str = ""
    source: str = "Current observable"

    @property
    def fwhm_major(self) -> float:
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sigma_major

    @property
    def fwhm_minor(self) -> float:
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sigma_minor

    @property
    def half_max_radius_major(self) -> float:
        return 0.5 * self.fwhm_major

    @property
    def half_max_radius_minor(self) -> float:
        return 0.5 * self.fwhm_minor

    @property
    def equivalent_half_max_radius(self) -> float:
        return float(np.sqrt(self.half_max_radius_major * self.half_max_radius_minor))

    @property
    def axis_ratio(self) -> float:
        return float(self.sigma_major / self.sigma_minor) if self.sigma_minor > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update(
            {
                "fwhm_major": self.fwhm_major,
                "fwhm_minor": self.fwhm_minor,
                "half_max_radius_major": self.half_max_radius_major,
                "half_max_radius_minor": self.half_max_radius_minor,
                "equivalent_half_max_radius": self.equivalent_half_max_radius,
                "axis_ratio": self.axis_ratio,
            }
        )
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SpotFit":
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})


@dataclass
class SpotDetectionResult:
    spots: list[SpotFit]
    noise_sigma: float
    background_window_pixels: int
    tested_polarities: tuple[str, ...]


def _odd_window(requested: int, shape: tuple[int, int]) -> int:
    limit = max(3, min(shape))
    if requested <= 0:
        requested = max(3, int(round(limit / 5.0)))
    requested = max(3, min(int(requested), limit))
    if requested % 2 == 0:
        requested = requested - 1 if requested == limit else requested + 1
    return max(3, requested)


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    median = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = float(np.std(finite))
    return max(sigma, np.finfo(float).eps)


def _gaussian_plane(
    coordinates: tuple[np.ndarray, np.ndarray],
    offset: float,
    slope_x: float,
    slope_y: float,
    amplitude: float,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
) -> np.ndarray:
    x, y = coordinates
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = x - center_x, y - center_y
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    gaussian = amplitude * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    return offset + slope_x * dx + slope_y * dy + gaussian


def _component_seed(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    signed_residual: np.ndarray,
    component: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[float, float, float, float, float]:
    weights = np.where(component, np.clip(signed_residual, 0.0, None), 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        weights = component.astype(float)
        total = float(np.sum(weights))
    center_x = float(np.sum(weights * x_grid) / total)
    center_y = float(np.sum(weights * y_grid) / total)
    x_delta, y_delta = x_grid - center_x, y_grid - center_y
    covariance = np.array(
        [
            [np.sum(weights * x_delta * x_delta), np.sum(weights * x_delta * y_delta)],
            [np.sum(weights * x_delta * y_delta), np.sum(weights * y_delta * y_delta)],
        ],
        dtype=float,
    ) / total
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], [dx * dx * 0.25, dy * dy * 0.25])
        vector = eigenvectors[:, order[0]]
        angle = float(np.arctan2(vector[1], vector[0]))
        major, minor = float(np.sqrt(eigenvalues[0])), float(np.sqrt(eigenvalues[1]))
    except np.linalg.LinAlgError:
        major, minor, angle = max(dx, dy), min(dx, dy), 0.0
    return center_x, center_y, max(major, dx * 0.5), max(minor, dy * 0.5), angle


def _fit_component(
    spot_id: int,
    sign: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    residual: np.ndarray,
    component: np.ndarray,
    noise_sigma: float,
) -> SpotFit:
    rows, cols = np.where(component)
    dx = float(np.median(np.abs(np.diff(x)))) if len(x) > 1 else 1.0
    dy = float(np.median(np.abs(np.diff(y)))) if len(y) > 1 else 1.0
    dx, dy = max(dx, np.finfo(float).eps), max(dy, np.finfo(float).eps)
    x_grid, y_grid = np.meshgrid(x, y)
    cx, cy, sigma_major, sigma_minor, theta = _component_seed(
        x_grid, y_grid, sign * residual, component, dx, dy
    )
    pad = max(3, int(np.ceil(2.5 * max(sigma_major / dx, sigma_minor / dy))))
    row0, row1 = max(0, int(rows.min()) - pad), min(z.shape[0], int(rows.max()) + pad + 1)
    col0, col1 = max(0, int(cols.min()) - pad), min(z.shape[1], int(cols.max()) + pad + 1)
    roi_x, roi_y = x_grid[row0:row1, col0:col1], y_grid[row0:row1, col0:col1]
    roi_z = z[row0:row1, col0:col1]
    valid = np.isfinite(roi_z)
    fit_x, fit_y, fit_z = roi_x[valid], roi_y[valid], roi_z[valid]
    background = float(np.median(fit_z))
    amplitude = sign * max(float(np.max(sign * residual[component])), noise_sigma)
    x_span = max(float(np.ptp(x[col0:col1])), dx)
    y_span = max(float(np.ptp(y[row0:row1])), dy)
    scale = max(float(np.ptp(fit_z)), abs(amplitude), noise_sigma, np.finfo(float).eps)
    amp_bounds = (0.0, 5.0 * scale) if sign > 0 else (-5.0 * scale, 0.0)
    lower = [-np.inf, -np.inf, -np.inf, amp_bounds[0], float(np.min(fit_x)), float(np.min(fit_y)), dx * 0.30, dy * 0.30, -np.pi]
    upper = [np.inf, np.inf, np.inf, amp_bounds[1], float(np.max(fit_x)), float(np.max(fit_y)), x_span * 1.5, y_span * 1.5, np.pi]
    initial = [background, 0.0, 0.0, amplitude, cx, cy, sigma_major, sigma_minor, theta]
    success = True
    message = ""
    try:
        params, _ = curve_fit(
            _gaussian_plane,
            (fit_x, fit_y),
            fit_z,
            p0=np.clip(initial, np.asarray(lower) + 1e-12, np.asarray(upper) - 1e-12),
            bounds=(lower, upper),
            maxfev=20000,
        )
        predicted = _gaussian_plane((fit_x, fit_y), *params)
    except Exception as exc:
        params = np.asarray(initial, dtype=float)
        predicted = _gaussian_plane((fit_x, fit_y), *params)
        success = False
        message = str(exc)
    offset, slope_x, slope_y, amplitude, cx, cy, sigma_x, sigma_y, theta = [float(v) for v in params]
    if sigma_y > sigma_x:
        sigma_x, sigma_y = sigma_y, sigma_x
        theta += np.pi / 2.0
    theta_deg = float(((np.degrees(theta) + 90.0) % 180.0) - 90.0)
    background_at_center = offset
    total_variation = float(np.sum((fit_z - np.mean(fit_z)) ** 2))
    residual_variation = float(np.sum((fit_z - predicted) ** 2))
    r_squared = 1.0 - residual_variation / total_variation if total_variation > 0 else float("nan")
    contrast = 100.0 * amplitude / abs(background_at_center) if abs(background_at_center) > np.finfo(float).eps else float("nan")
    return SpotFit(
        spot_id=spot_id,
        polarity="bright" if sign > 0 else "dark",
        center_x=cx,
        center_y=cy,
        sigma_major=abs(sigma_x),
        sigma_minor=abs(sigma_y),
        angle_deg=theta_deg,
        amplitude=amplitude,
        background=background_at_center,
        contrast_percent=contrast,
        snr=abs(amplitude) / noise_sigma,
        r_squared=r_squared,
        integrated_signal=2.0 * np.pi * amplitude * abs(sigma_x) * abs(sigma_y),
        component_pixels=int(np.count_nonzero(component)),
        fit_success=success,
        fit_message=message,
    )


def detect_fluorescent_spots(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    polarity: str = "auto",
    threshold_sigma: float = 3.0,
    min_pixels: int = 4,
    max_spots: int = 12,
    background_window_pixels: int = 0,
) -> SpotDetectionResult:
    """Detect and fit fluorescent spots on a rectangular physical XY grid.

    A median background model removes slow illumination and scan offsets. Each
    thresholded component is then fitted with a rotated elliptical Gaussian on
    a local planar background. Radii are reported in the units of ``x`` and
    ``y`` rather than in pixels.
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape != (len(y), len(x)) or min(z.shape) < 3:
        raise ValueError("Spot detection requires a 2D map matching one-dimensional X and Y axes.")
    finite = np.isfinite(z)
    if np.count_nonzero(finite) < 9:
        raise ValueError("The map does not contain enough finite samples for spot detection.")
    fill = float(np.nanmedian(z))
    clean = np.where(finite, z, fill)
    window = _odd_window(background_window_pixels, z.shape)
    background_map = median_filter(clean, size=(window, window), mode="nearest")
    residual = clean - background_map
    noise_sigma = _robust_sigma(residual[finite])
    if not np.isfinite(noise_sigma):
        raise ValueError("The scan noise level could not be estimated.")

    requested = polarity.strip().lower()
    if requested == "auto":
        positive_score = float(np.nanmax(residual)) / noise_sigma
        negative_score = abs(float(np.nanmin(residual))) / noise_sigma
        signs = (1.0,) if positive_score >= negative_score else (-1.0,)
    elif requested in {"bright and dark", "both"}:
        signs = (1.0, -1.0)
    elif requested == "dark":
        signs = (-1.0,)
    else:
        signs = (1.0,)

    candidates: list[tuple[float, float, np.ndarray]] = []
    structure = np.ones((3, 3), dtype=bool)
    for sign in signs:
        signed_peak = float(np.nanmax(sign * residual[finite]))
        # A perfectly smooth synthetic or heavily averaged scan can have a MAD
        # noise estimate near machine precision. Keep the threshold above a
        # small fraction of the feature itself so its infinite Gaussian tail
        # does not turn the entire map into one rejected component.
        threshold = max(
            max(0.5, float(threshold_sigma)) * noise_sigma,
            0.01 * max(0.0, signed_peak),
        )
        mask = finite & ((sign * residual) >= threshold)
        mask = binary_fill_holes(binary_closing(mask, structure=structure))
        labels, count = label(mask, structure=structure)
        for component_id in range(1, count + 1):
            component = labels == component_id
            pixels = int(np.count_nonzero(component))
            if pixels < max(1, int(min_pixels)) or pixels > int(0.45 * z.size):
                continue
            score = float(np.nanmax(sign * residual[component]))
            candidates.append((score, sign, component))

    candidates.sort(key=lambda item: item[0], reverse=True)
    spots: list[SpotFit] = []
    for _score, sign, component in candidates[: max(1, int(max_spots))]:
        spots.append(_fit_component(len(spots) + 1, sign, x, y, clean, residual, component, noise_sigma))
    spots.sort(key=lambda spot: abs(spot.amplitude), reverse=True)
    for index, spot in enumerate(spots, start=1):
        spot.spot_id = index
    return SpotDetectionResult(
        spots=spots,
        noise_sigma=noise_sigma,
        background_window_pixels=window,
        tested_polarities=tuple("bright" if sign > 0 else "dark" for sign in signs),
    )
