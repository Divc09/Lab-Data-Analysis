import numpy as np
import pytest

from nvfit.scan_spots import SpotFit, detect_fluorescent_spots


def test_detects_rotated_fluorescent_spot_on_varying_background():
    rng = np.random.default_rng(42)
    x = np.linspace(-4.0, 4.0, 81)
    y = np.linspace(-3.5, 3.5, 71)
    xx, yy = np.meshgrid(x, y)
    center_x, center_y = 1.2, -0.8
    sigma_major, sigma_minor = 0.70, 0.34
    theta = np.deg2rad(27.0)
    dx, dy = xx - center_x, yy - center_y
    xr = np.cos(theta) * dx + np.sin(theta) * dy
    yr = -np.sin(theta) * dx + np.cos(theta) * dy
    background = 1200.0 + 24.0 * xx - 15.0 * yy
    z = background + 460.0 * np.exp(-0.5 * ((xr / sigma_major) ** 2 + (yr / sigma_minor) ** 2))
    z += rng.normal(0.0, 9.0, size=z.shape)

    result = detect_fluorescent_spots(
        x, y, z, polarity="bright", threshold_sigma=3.5, min_pixels=6, max_spots=4,
        background_window_pixels=19,
    )

    assert result.spots
    spot = result.spots[0]
    assert spot.fit_success
    assert spot.center_x == pytest.approx(center_x, abs=0.08)
    assert spot.center_y == pytest.approx(center_y, abs=0.08)
    assert spot.sigma_major == pytest.approx(sigma_major, rel=0.15)
    assert spot.sigma_minor == pytest.approx(sigma_minor, rel=0.18)
    assert spot.r_squared > 0.95


def test_spot_round_trip_preserves_analysis_source():
    values = dict(
        spot_id=1, polarity="bright", center_x=1.0, center_y=2.0,
        sigma_major=0.5, sigma_minor=0.25, angle_deg=15.0, amplitude=20.0,
        background=100.0, contrast_percent=20.0, snr=8.0, r_squared=0.98,
        integrated_signal=15.0, component_pixels=12, fit_success=True, source="Signal",
    )
    restored = SpotFit.from_dict(SpotFit(**values).to_dict())
    assert restored.source == "Signal"
    assert restored.equivalent_half_max_radius > 0.0
