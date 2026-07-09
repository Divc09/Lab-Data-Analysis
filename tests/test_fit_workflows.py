from pathlib import Path

import numpy as np
import pytest

from nvfit.fit_workflows import FitWorkflowConfig, fit_non_ramsey
from nvfit.io_mat import load_saved_data_mat
from nvfit.rabi_utils import build_rabi_nv_metrics, rabi_envelope_metrics


def test_rabi_default_prefers_phase_ramp_on_checkpoint():
    root = Path(__file__).resolve().parents[1]
    tr = load_saved_data_mat(root / "TestData" / "NewCodebaseTests" / "323RabiFullRun__checkpoint.mat", mode="contrast")
    res = fit_non_ramsey(
        "Rabi",
        tr.x_ns,
        tr.y,
        config=FitWorkflowConfig(multistart=20, rabi_mode="Single reliable Rabi"),
        trace_experiment_type="Rabi",
        trace=tr,
    )
    assert res.model_name == "RabiPhaseRamp"
    assert res.r2 > 0.90
    assert "phase-ramp" in res.selection_note.lower()
    p = {n: float(v) for n, v in zip(res.param_names, res.params)}
    assert "tau_ramp" in p
    metrics = build_rabi_nv_metrics(p, metadata=tr.metadata or {}, errors=None)
    assert 0.0 < metrics["delay_ns"] < 30.0
    assert 25.0 <= metrics["first_peak_ns"] <= 36.0
    assert "rabi_envelope" in res.extras
    envelope_metrics = rabi_envelope_metrics(res.extras["rabi_envelope"])
    assert envelope_metrics["T2rho_ns"] > 0.0


@pytest.mark.parametrize(
    "path",
    [
        Path(r"D:\BacklundLabResearch\Data\Experiments\0130_TDDWaveguideDay2\Jan31\Rabi10us.mat"),
        Path(r"D:\BacklundLabResearch\Data\Experiments\0319_TDD\March23rd\1142_323RabiLong.mat"),
        Path(r"D:\BacklundLabResearch\Data\Experiments\429_TDD\May3\0145_RabiLongAfterAlign.mat"),
    ],
)
def test_long_rabi_scans_auto_select_long_damped_fit(path: Path):
    if not path.exists():
        pytest.skip(f"missing external long Rabi fixture: {path}")
    tr = load_saved_data_mat(path, mode="contrast")
    phase = fit_non_ramsey(
        "Rabi",
        tr.x_ns,
        tr.y,
        config=FitWorkflowConfig(multistart=8, rabi_mode="Phase-ramp (recommended physical fit)"),
        trace_experiment_type=tr.experiment_type,
        trace=tr,
    )
    long_fit = fit_non_ramsey(
        "Rabi",
        tr.x_ns,
        tr.y,
        config=FitWorkflowConfig(multistart=8, rabi_mode="Long damped Rabi"),
        trace_experiment_type=tr.experiment_type,
        trace=tr,
    )
    assert phase.model_name == "RabiLongDamped"
    assert "long damped rabi" in phase.selection_note.lower()
    assert long_fit.model_name == "RabiLongDamped"
    assert long_fit.r2 > 0.25
    assert long_fit.r2 > phase.extras.get("rabi_long_phase_reference_r2", -1.0)
    p = {n: float(v) for n, v in zip(long_fit.param_names, long_fit.params)}
    assert 0.02 <= p["f"] <= 0.05
    env = rabi_envelope_metrics(long_fit.extras["rabi_envelope"])
    assert env["T2rho_ns"] > 100.0


def test_odmr_raw_signal_no_longer_collapses_to_invalid_scale():
    root = Path(__file__).resolve().parents[1]
    tr = load_saved_data_mat(root / "TestData" / "ODMRRedo.mat", mode="raw_signal")
    res = fit_non_ramsey(
        "ODMR",
        tr.x_ns,
        tr.y,
        config=FitWorkflowConfig(multistart=20, odmr_multipeak=True, odmr_peak_count=4, odmr_auto_peak=True, odmr_polarity="Auto"),
        trace_experiment_type="ODMR",
        trace=tr,
    )
    assert res.success
    assert res.model_name in {"ODMR", "ODMRMulti"}
    assert res.r2 > 0.0
    p = {n: float(v) for n, v in zip(res.param_names, res.params)}
    x0 = p.get("x0", p.get("x0_1"))
    assert x0 is not None
    assert float(tr.x_ns.min()) <= x0 <= float(tr.x_ns.max())


def test_odmr_multipeak_selection_limits_detected_peak_count():
    root = Path(__file__).resolve().parents[1]
    tr = load_saved_data_mat(root / "TestData" / "ODMR5MHzOffAxis.mat", mode="contrast")
    res = fit_non_ramsey(
        "ODMR",
        tr.x_ns,
        tr.y,
        config=FitWorkflowConfig(multistart=20, odmr_multipeak=True, odmr_peak_count=4, odmr_auto_peak=True, odmr_polarity="Auto"),
        trace_experiment_type="ODMR",
        trace=tr,
    )
    detected = list(res.extras.get("detected_peaks", []))
    assert len(detected) <= 4
    if res.model_name == "ODMRMulti":
        centers = [name for name in res.param_names if name.startswith("x0_")]
        assert 2 <= len(centers) <= 4


def test_t1_newcodebasetests_support_direct_and_normalized_modes():
    root = Path(__file__).resolve().parents[1]
    tr = load_saved_data_mat(root / "TestData" / "NewCodebaseTests" / "T1_20260324_041805.mat", mode="contrast")
    direct = fit_non_ramsey(
        "T1",
        tr.x_ns,
        tr.y,
        config=FitWorkflowConfig(multistart=20, t1_prep_mode="Direct T1", t1_model_family="Exponential"),
        trace_experiment_type="T1",
        trace=tr,
    )
    normalized = fit_non_ramsey(
        "T1",
        tr.x_ns,
        tr.y,
        config=FitWorkflowConfig(multistart=20, t1_prep_mode="Normalize then 1-data", t1_model_family="Stretched exponential"),
        trace_experiment_type="T1",
        trace=tr,
    )
    assert direct.model_name == "T1Exp"
    assert normalized.model_name == "T1NormStretch"
    assert direct.success and normalized.success
    assert direct.extras["t1_prep_mode"] == "direct"
    assert normalized.extras["t1_prep_mode"] == "normalized_1_minus_data"


def test_odmr_auto_polarity_selects_peak_for_peak_shaped_trace():
    root = Path(__file__).resolve().parents[1]
    tr = load_saved_data_mat(root / "TestData" / "NewCodebaseTests" / "ODMR_20260324_024029.mat", mode="contrast")
    res = fit_non_ramsey(
        "ODMR",
        tr.x_ns,
        tr.y,
        config=FitWorkflowConfig(multistart=20, odmr_polarity="Auto", odmr_multipeak=False),
        trace_experiment_type="ODMR",
        trace=tr,
    )
    assert res.success
    assert res.extras["odmr_polarity"] == "peak"
    center = {n: float(v) for n, v in zip(res.param_names, res.params)}["x0"]
    assert 2.116 <= center <= 2.119


def test_odmr_peak_fit_range_constrains_selected_resonance():
    x = np.linspace(0.0, 10.0, 401)
    y = (
        0.1
        + 0.9 * (0.18**2 / ((x - 3.0) ** 2 + 0.18**2))
        + 0.6 * (0.22**2 / ((x - 7.0) ** 2 + 0.22**2))
    )
    res = fit_non_ramsey(
        "ODMR",
        x,
        y,
        config=FitWorkflowConfig(
            multistart=12,
            odmr_polarity="Peak",
            odmr_multipeak=False,
            odmr_peak_ranges=((6.6, 7.4),),
        ),
        trace_experiment_type="ODMR",
    )

    center = {n: float(v) for n, v in zip(res.param_names, res.params)}["x0"]
    assert 6.6 <= center <= 7.4
    assert res.extras["detected_peaks"] == pytest.approx([7.0])
