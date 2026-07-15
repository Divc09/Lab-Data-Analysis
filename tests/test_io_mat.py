from pathlib import Path

import numpy as np
import pytest

from nvfit.io_mat import infer_experiment_type, load_saved_data_mat


TDD_ROOT = Path(r"D:\BacklundLabResearch\Data\Experiments\429_TDD")


def test_load_saved_data_long_ramsey():
    root = Path(__file__).resolve().parents[1]
    path = root / "LongRamsey.mat"
    if not path.exists():
        path = root / "TestData" / "LongRamsey.mat"
    tr = load_saved_data_mat(path, mode="contrast")
    assert tr.file_name == "LongRamsey.mat"
    assert tr.experiment_type == "Ramsey"
    assert len(tr.x_ns) == len(tr.y)
    assert len(tr.x_ns) > 100
    assert np.isfinite(tr.y).all()
    assert np.min(tr.x_ns) >= 0
    assert np.max(tr.x_ns) > 3000


def test_detect_scan2d_from_metadata():
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "1305_XZScan.mat"
    if not p.exists():
        return
    tr = load_saved_data_mat(p, mode="contrast")
    assert tr.scan_dim == "scan2d"
    assert not tr.fit_allowed
    assert tr.z2d is not None


def test_detect_odmr_from_axis_label():
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "1704_PostAlignSpectra.mat"
    if not p.exists():
        return
    tr = load_saved_data_mat(p, mode="contrast")
    assert tr.experiment_type == "ODMR"


def test_load_new_matlab_dynamic_decoupling_trace():
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "NewCodebaseTests" / "xy8-128_overnight.mat"
    if not p.exists():
        return
    tr = load_saved_data_mat(p, mode="contrast")
    assert tr.experiment_type == "DynamicDecoupling"
    assert tr.scan_dim == "time"
    assert tr.fit_allowed
    assert len(tr.x_ns) == len(tr.y) == 251
    assert tr.raw_x is not None
    assert tr.raw_x_label == "Pulse duration (ns)"
    assert np.isclose(tr.raw_x[0], 17.0)
    assert np.isclose(tr.raw_x[-1], 267.0)
    assert np.isclose(tr.x_ns[0], 25600.0)
    assert np.isclose(tr.x_ns[-1], 89600.0)
    assert tr.x_label == "Total evolution time (ns)"
    assert tr.metadata is not None
    assert tr.metadata.get("sequence_name") == "XY8-128"
    assert tr.metadata.get("dd_pulse_count") == 128
    assert np.isclose(float(tr.metadata.get("pi_pulse_ns", 0.0)), 44.0)
    assert np.isclose(float(tr.metadata.get("rf_frequency_ghz", 0.0)), 2.245)
    assert np.nanmean(tr.y) > 0.0


def test_load_snapshot_rabi_checkpoint():
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "NewCodebaseTests" / "323RabiFullRun__checkpoint.mat"
    if not p.exists():
        return
    tr = load_saved_data_mat(p, mode="contrast")
    assert tr.experiment_type == "Rabi"
    assert tr.scan_dim == "time"
    assert tr.fit_allowed
    assert len(tr.x_ns) == len(tr.y) == 221
    assert tr.metadata is not None
    assert tr.metadata.get("schema") == "snapshot_run"
    assert tr.metadata.get("save_type") == "checkpoint"
    assert tr.metadata.get("run_status") == "checkpoint"
    assert tr.metadata.get("completed_iterations") == 17
    assert tr.metadata.get("target_iterations") == 50
    assert tr.metadata.get("checkpoint_count") == 17
    assert tr.metadata.get("scan_notes") == "Rabi (RF: 2.118 GHz)"
    assert np.isclose(float(tr.metadata.get("rf_frequency_ghz", 0.0)), 2.118)
    assert np.isfinite(tr.y).sum() > 150


def test_reference_mode_uses_reference_vector():
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "NewCodebaseTests" / "323RabiFullRun__checkpoint.mat"
    tr = load_saved_data_mat(p, mode="reference")
    assert tr.mode == "reference"
    assert tr.y_label.startswith("Reference")
    assert np.allclose(tr.y, tr.reference)


@pytest.mark.parametrize(
    ("mode", "expected_y"),
    [
        ("contrast", "contrast"),
        ("raw_signal", "signal"),
        ("reference", "reference"),
        ("difference", "difference"),
    ],
)
def test_snapshot_iteration_matrices_are_available_for_all_observable_modes(mode, expected_y):
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "NewCodebaseTests" / "323RabiFullRun__checkpoint.mat"
    if not p.exists():
        return
    tr = load_saved_data_mat(p, mode=mode)

    assert tr.iteration_contrast is not None
    assert tr.iteration_signal is not None
    assert tr.iteration_reference is not None
    assert tr.iteration_valid is not None
    assert tr.iteration_contrast.shape == (221, 17)
    assert tr.iteration_signal.shape == (221, 17)
    assert tr.iteration_reference.shape == (221, 17)
    assert tr.iteration_valid.shape == (221, 17)

    if expected_y == "signal":
        assert np.allclose(tr.y, tr.signal, equal_nan=True)
    elif expected_y == "reference":
        assert np.allclose(tr.y, tr.reference, equal_nan=True)
    elif expected_y == "difference":
        assert np.allclose(tr.y, tr.reference - tr.signal, equal_nan=True)


def test_detect_t1_from_notes_and_filename():
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "NewCodebaseTests" / "T1_20260324_041805.mat"
    tr = load_saved_data_mat(p, mode="contrast")
    assert tr.experiment_type == "T1"
    assert tr.scan_dim == "time"
    assert tr.fit_allowed


def test_load_snapshot_stage_scan_plot_only():
    root = Path(__file__).resolve().parents[1]
    p = root / "Stage_scan_20260323_044312.mat"
    if not p.exists():
        return
    tr = load_saved_data_mat(p, mode="contrast")
    assert tr.experiment_type == "LineScan"
    assert tr.scan_dim == "scan1d"
    assert not tr.fit_allowed
    assert tr.metadata is not None
    assert tr.metadata.get("schema") == "snapshot_run"
    assert tr.metadata.get("save_type") == "final"
    assert tr.metadata.get("run_status") == "completed"
    assert tr.metadata.get("completed_iterations") == 1
    assert tr.metadata.get("target_iterations") == 1
    assert tr.metadata.get("average_contrast_png")


def test_snapshot_modes_use_saved_analysis_exports():
    root = Path(__file__).resolve().parents[1]
    p = root / "Stage_scan_20260323_044312.mat"
    if not p.exists():
        return
    tr_signal = load_saved_data_mat(p, mode="raw_signal")
    tr_diff = load_saved_data_mat(p, mode="difference")
    assert tr_signal.scan_dim == "scan1d"
    assert tr_diff.scan_dim == "scan1d"
    assert np.isfinite(tr_signal.y).all()
    assert np.isfinite(tr_diff.y).all()
    assert np.allclose(tr_diff.reference - tr_diff.signal, tr_diff.y)


def test_load_param_grid_stage_scan_fallback_2d():
    root = Path(__file__).resolve().parents[1]
    for name in ("14102DScan.mat", "13382DScan.mat"):
        p = root / "TestData" / "NewCodebaseTests" / name
        tr = load_saved_data_mat(p, mode="contrast")
        assert tr.experiment_type == "Scan2D"
        assert tr.scan_dim == "scan2d"
        assert not tr.fit_allowed
        assert tr.z2d is not None
        assert tr.x2d is not None and tr.y2d is not None
        assert tr.z2d.shape == tr.x2d.shape == tr.y2d.shape


def test_experiment_detection_prefers_explicit_sequence_and_stage_metadata():
    assert infer_experiment_type("unknown.mat", ("duration",), notes="Ramsey free precession") == "Ramsey"
    assert infer_experiment_type("unknown.mat", ("duration",), notes="T1 relaxation") == "T1"
    assert infer_experiment_type("rabi_alignment.mat", ("Stage X", "Stage Y"), notes="Rabi alignment map") == "LineScan"
    assert infer_experiment_type("spectrum.mat", ("RF frequency",)) == "ODMR"


@pytest.mark.skipif(not TDD_ROOT.exists(), reason="429_TDD data directory is not available")
def test_429_tdd_all_mat_files_load_and_classify():
    files = sorted(TDD_ROOT.rglob("*.mat"))
    assert len(files) >= 255
    traces = [load_saved_data_mat(p, mode="contrast") for p in files]
    assert len(traces) == len(files)
    families = {tr.experiment_type for tr in traces}
    assert {"DEERFrequency", "DEERDuration", "DEERDecay", "DEERPosition"} <= families
    assert all(len(tr.x_ns) and len(tr.y) for tr in traces)


@pytest.mark.skipif(not TDD_ROOT.exists(), reason="429_TDD data directory is not available")
def test_429_tdd_checkpoint_metadata_and_iteration_matrices():
    p = TDD_ROOT / "May11" / "AfterCombiner" / "1653_May11_DEERFreqSweeptau600_Stepsize1AfterRF2Fixes__checkpoint.mat"
    tr = load_saved_data_mat(p, mode="contrast")
    assert tr.experiment_type == "DEERFrequency"
    assert tr.metadata is not None
    assert tr.metadata.get("save_type") == "checkpoint"
    assert tr.metadata.get("run_status") == "checkpoint"
    assert tr.metadata.get("completed_iterations") == 17
    assert tr.metadata.get("target_iterations") == 25
    assert tr.iteration_contrast is not None
    assert tr.iteration_signal is not None
    assert tr.iteration_reference is not None
    assert tr.iteration_valid is not None


@pytest.mark.skipif(not TDD_ROOT.exists(), reason="429_TDD data directory is not available")
def test_429_tdd_snapshot_dynamic_decoupling_uses_physical_axis():
    p = TDD_ROOT / "May11" / "BeforeCombiner" / "0620_May11_XY88LogSpacePi16ns2130MHzRefPeak.mat"
    tr = load_saved_data_mat(p, mode="contrast")
    assert tr.experiment_type == "DynamicDecoupling"
    assert tr.x_label == "Total evolution time (ns)"
    assert tr.raw_x is not None
    assert tr.raw_x_label is not None
    assert tr.x_ns[0] > tr.raw_x[0]
    assert tr.metadata is not None
    assert tr.metadata.get("dd_pulse_count") == 8

