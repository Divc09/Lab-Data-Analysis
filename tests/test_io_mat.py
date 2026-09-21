from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from nvfit.io_mat import MatDataError, infer_experiment_type, load_experiment_dataset, load_saved_data_mat


TDD_ROOT = Path(r"D:\BacklundLabResearch\Data\Experiments\429_TDD")


def _metric_1d(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    return {"x": x, "y": y, "xLabel": "Stage Z", "yLabel": label}


def _write_dual_snapshot_1d(path: Path, *, reference_only: bool = False) -> None:
    x = np.arange(4, dtype=float)
    detector_values = np.empty((4, 2), dtype=object)
    detector2_values = np.empty((4, 2), dtype=object)
    for i in range(4):
        for j in range(2):
            signal1 = 0.0 if reference_only else 80.0 + i + j
            signal2 = 0.0 if reference_only else 40.0 + 2 * i + j
            detector_values[i, j] = np.array([100.0 + i, signal1])
            detector2_values[i, j] = np.array([60.0 + i, signal2])
    signal1 = np.array([np.mean([detector_values[i, j][1] for j in range(2)]) for i in range(4)])
    reference1 = np.array([100.0 + i for i in range(4)])
    signal2 = np.array([np.mean([detector2_values[i, j][1] for j in range(2)]) for i in range(4)])
    reference2 = np.array([60.0 + i for i in range(4)])
    analysis = {
        "dimension": 1,
        "averageContrast": _metric_1d(x, (reference1 - signal1) / reference1, "Contrast"),
        "averageSignal": _metric_1d(x, signal1, "Signal [counts]"),
        "averageReference": _metric_1d(x, reference1, "Reference [counts]"),
        "detector2": {
            "dimension": 1,
            "averageContrast": _metric_1d(x, (reference2 - signal2) / reference2, "Contrast"),
            "averageSignal": _metric_1d(x, signal2, "Signal [counts]"),
            "averageReference": _metric_1d(x, reference2, "Reference [counts]"),
        },
    }
    savemat(
        path,
        {
            "Data": {
                "values": detector_values,
                "detector2Values": detector2_values,
                "iteration": np.full(4, 2),
                "failedPoints": np.zeros((4, 2)),
                "detectorMode": "dual",
            },
            "scanInfo": {
                "bounds": np.array([0.0, 3.0]),
                "stepSize": 1.0,
                "nSteps": 4,
                "parameter": "stage z",
                "identifier": "PI Stage",
                "notes": "Synthetic dual Z scan",
            },
            "params": {"detectorMode": "dual"},
            "progress": {"status": "completed", "completedIterations": 2, "targetIterations": 2},
            "metadata": {"runStatus": "completed", "saveType": "final"},
            "analysis": analysis,
        },
    )


def _write_unacquired_dual_snapshot_2d(path: Path) -> None:
    x = np.array([0.0, 1.0])
    y = np.array([10.0, 11.0])
    values1 = np.empty((2, 2), dtype=object)
    values2 = np.empty((2, 2), dtype=object)
    for index in np.ndindex(values1.shape):
        values1[index] = np.array([0.0, 0.0])
        values2[index] = np.array([0.0, 0.0])

    def metric(z: np.ndarray, label: str) -> dict:
        return {"x": x, "y": y, "z": z, "xLabel": "Stage X", "yLabel": "Stage Y", "zLabel": label}

    detector_analysis = {
        "dimension": 2,
        "averageContrast": metric(np.zeros((2, 2)), "Contrast"),
        "averageSignal": metric(np.zeros((2, 2)), "Signal [counts]"),
        "averageReference": metric(np.zeros((2, 2)), "Reference [counts]"),
    }
    scan_info = np.empty(2, dtype=object)
    scan_info[0] = {"bounds": np.array([0.0, 1.0]), "stepSize": 1.0, "nSteps": 2, "parameter": "stage x", "identifier": "PI Stage"}
    scan_info[1] = {"bounds": np.array([10.0, 11.0]), "stepSize": 1.0, "nSteps": 2, "parameter": "stage y", "identifier": "PI Stage"}
    savemat(
        path,
        {
            "Data": {
                "values": values1,
                "detector2Values": values2,
                "iteration": np.zeros((2, 2)),
                "failedPoints": np.zeros((2, 2)),
                "detectorMode": "dual",
            },
            "scanInfo": scan_info,
            "params": {"detectorMode": "dual", "scanAxes": np.array(["x", "y"], dtype=object)},
            "progress": {"status": "error", "completedIterations": 0, "targetIterations": 1},
            "metadata": {
                "runStatus": "error",
                "saveType": "error",
                "errorInfo": {
                    "identifier": "daq:channel:clockNotConfigured",
                    "message": "Sample clock was not configured.",
                },
            },
            "analysis": {**detector_analysis, "detector2": dict(detector_analysis)},
        },
    )


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


def test_dual_snapshot_loader_exposes_distinct_stable_detector_streams(tmp_path: Path):
    path = tmp_path / "dual_z.mat"
    _write_dual_snapshot_1d(path)

    dataset = load_experiment_dataset(path, mode="contrast")

    assert dataset.detector_mode == "dual"
    assert dataset.available_detectors == ("detector1", "detector2")
    assert set(dataset.detectors) == {"detector1", "detector2"}
    detector1 = dataset.detectors["detector1"]
    detector2 = dataset.detectors["detector2"]
    assert detector1.detector_label == "Detector 1"
    assert detector2.detector_label == "Detector 2"
    assert detector1.scan_dim == detector2.scan_dim == "scan1d"
    assert detector1.iteration_signal is not None
    assert detector2.iteration_signal is not None
    assert not np.allclose(detector1.signal, detector2.signal)
    assert np.allclose(load_saved_data_mat(path, "contrast").y, detector1.y)
    assert np.allclose(load_saved_data_mat(path, "signal", detector_id="detector2").y, detector2.signal)


def test_reference_only_zero_signal_cells_remain_valid_when_acquired(tmp_path: Path):
    path = tmp_path / "dual_reference_only.mat"
    _write_dual_snapshot_1d(path, reference_only=True)

    dataset = load_experiment_dataset(path, mode="contrast")

    assert set(dataset.detectors) == {"detector1", "detector2"}
    for trace in dataset.detectors.values():
        assert np.all(trace.signal == 0.0)
        assert np.all(trace.reference > 0.0)
        assert np.allclose(trace.y, 1.0)


@pytest.mark.parametrize("mode", ["contrast", "signal", "reference", "difference"])
def test_unacquired_failed_2d_snapshot_never_becomes_a_zero_map(tmp_path: Path, mode: str):
    path = tmp_path / "failed_xy.mat"
    _write_unacquired_dual_snapshot_2d(path)

    dataset = load_experiment_dataset(path, mode=mode)

    assert dataset.available_detectors == ("detector1", "detector2")
    assert dataset.detectors == {}
    assert set(dataset.detector_errors) == {"detector1", "detector2"}
    for failure in dataset.detector_errors.values():
        assert failure.run_status == "error"
        assert failure.error_identifier == "daq:channel:clockNotConfigured"
        assert "No 2D cells were acquired" in failure.message
    with pytest.raises(MatDataError, match="clockNotConfigured"):
        load_saved_data_mat(path, mode=mode)


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

