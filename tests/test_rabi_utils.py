import numpy as np

from nvfit.rabi_utils import (
    build_rabi_nv_metrics,
    fit_rabi_envelope,
    rabi_envelope_bounds,
    rabi_envelope_metrics,
    rabi_timing_metrics_from_frequency,
)


def test_rabi_timing_metrics_apply_dead_time_correction():
    metrics = rabi_timing_metrics_from_frequency(0.002, dead_time_ns=25.0)
    assert np.isclose(metrics["pi_time_ns_nominal"], 250.0)
    assert np.isclose(metrics["pi_over_2_time_ns_nominal"], 125.0)
    assert np.isclose(metrics["pi_time_ns"], 250.0)
    assert np.isclose(metrics["pi_over_2_time_ns"], 125.0)
    assert np.isclose(metrics["rabi_dead_time_ns"], 25.0)


def test_build_rabi_nv_metrics_carries_uncertainty_and_prefers_max_frequency():
    params = {"f": 0.002, "f2": 0.003, "tau": 500.0}
    errors = {"f2": 1.0e-4}
    metrics = build_rabi_nv_metrics(params, metadata={"rabi_dead_time_ns": 10.0}, errors=errors)
    assert np.isclose(metrics["rabi_freq_MHz"], 3.0)
    assert np.isclose(metrics["pi_time_ns_nominal"], 1.0 / (2.0 * 0.003))
    assert np.isclose(metrics["pi_time_ns"], metrics["pi_time_ns_nominal"])
    assert metrics["pi_time_ns_err"] > 0.0
    assert np.isclose(metrics["pi_over_2_time_ns_err"], 0.5 * metrics["pi_time_ns_err"])


def test_build_rabi_nv_metrics_derives_first_peak_and_delay_from_shifted_fit():
    params = {"A": -0.03, "f": 0.029, "phi": -2.0, "t0": 25.0}
    metrics = build_rabi_nv_metrics(params, metadata=None, errors=None)
    assert metrics["first_peak_ns"] > 0.0
    assert metrics["delay_ns"] >= 0.0
    assert np.isclose(metrics["rabi_dead_time_ns"], metrics["delay_ns"])


def test_build_rabi_nv_metrics_reports_first_observable_peak_inside_trace_window():
    params = {"A": 0.03, "f": 0.0291, "phi": -3.3047}
    metrics = build_rabi_nv_metrics(params, metadata={"x_min_ns": 10.0, "x_max_ns": 450.0}, errors=None)
    assert metrics["first_peak_ns"] >= 10.0


def test_fit_rabi_envelope_recovers_exponential_decay():
    x = np.linspace(0.0, 1200.0, 600)
    baseline = 0.04
    amp = 0.035 * np.exp(-(x / 420.0))
    y = baseline + amp * np.cos(2.0 * np.pi * 0.018 * x + 0.2)

    payload = fit_rabi_envelope(
        x,
        y,
        model_name="RabiCosine",
        param_names=["y0", "A", "T", "beta", "f", "phi"],
        params=np.array([baseline, 0.035, 420.0, 1.0, 0.018, 0.2]),
        y_fit=y,
    )
    metrics = rabi_envelope_metrics(payload)

    assert payload["success"]
    assert 250.0 <= metrics["T2rho_ns"] <= 650.0
    assert 0.5 <= metrics["rabi_envelope_beta"] <= 1.8
    bounds = rabi_envelope_bounds(x, payload)
    assert bounds is not None
    upper, lower = bounds
    assert upper.shape == x.shape
    assert lower.shape == x.shape
    assert np.all(upper >= lower)


def test_fit_rabi_envelope_recovers_stretched_decay():
    x = np.linspace(0.0, 1600.0, 700)
    baseline = 0.02
    beta = 1.7
    amp = 0.05 * np.exp(-((x / 780.0) ** beta))
    y = baseline + amp * np.cos(2.0 * np.pi * 0.014 * x - 0.4)

    payload = fit_rabi_envelope(
        x,
        y,
        model_name="RabiChirp",
        param_names=["y0", "A", "T", "beta", "f", "alpha", "phi"],
        params=np.array([baseline, 0.05, 780.0, beta, 0.014, 0.0, -0.4]),
        y_fit=y,
    )
    metrics = rabi_envelope_metrics(payload)

    assert payload["success"]
    assert 450.0 <= metrics["T2rho_ns"] <= 1100.0
    assert metrics["rabi_envelope_beta"] > 1.0
