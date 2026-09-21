import csv
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.io import savemat

from nvfit.pro_batch import run_batch


def _write_dual_scan(path: Path, *, omit_detector2_values: bool = False) -> None:
    x = np.arange(3, dtype=float)
    values1 = np.empty((3, 1), dtype=object)
    values2 = np.empty((3, 1), dtype=object)
    for i in range(3):
        values1[i, 0] = np.array([100.0 + i, 70.0 + i])
        values2[i, 0] = np.array([50.0 + i, 20.0 + i])

    def metric(y: np.ndarray, label: str) -> dict:
        return {"x": x, "y": y, "xLabel": "Stage Z", "yLabel": label}

    ref1, sig1 = 100.0 + x, 70.0 + x
    ref2, sig2 = 50.0 + x, 20.0 + x
    data = {
        "values": values1,
        "iteration": np.ones(3),
        "failedPoints": np.zeros((3, 1)),
        "detectorMode": "dual",
    }
    if not omit_detector2_values:
        data["detector2Values"] = values2
    savemat(
        path,
        {
            "Data": data,
            "scanInfo": {
                "bounds": np.array([0.0, 2.0]),
                "stepSize": 1.0,
                "nSteps": 3,
                "parameter": "stage z",
                "identifier": "PI Stage",
            },
            "params": {"detectorMode": "dual"},
            "progress": {"status": "completed", "completedIterations": 1, "targetIterations": 1},
            "metadata": {"runStatus": "completed", "saveType": "final"},
            "analysis": {
                "dimension": 1,
                "averageContrast": metric((ref1 - sig1) / ref1, "Contrast"),
                "averageSignal": metric(sig1, "Signal [counts]"),
                "averageReference": metric(ref1, "Reference [counts]"),
                "detector2": {
                    "dimension": 1,
                    "averageContrast": metric((ref2 - sig2) / ref2, "Contrast"),
                    "averageSignal": metric(sig2, "Signal [counts]"),
                    "averageReference": metric(ref2, "Reference [counts]"),
                },
            },
        },
    )


def test_batch_outputs_schema(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "TestData" if (root / "TestData").exists() else root
    out = tmp_path / "out"
    run_batch(input_dir=data_dir, output_dir=out, mode="contrast", bin_size=1, smooth_window=1)

    assert (out / "batch_summary.csv").exists()
    assert (out / "validation_report.csv").exists()
    assert (out / "aggregate_report.csv").exists()
    result_json = out / "LongRamsey_result.json"
    assert result_json.exists()
    txt = result_json.read_text(encoding="utf-8")
    assert '"nv_metrics"' in txt
    assert '"profile"' in txt


def test_batch_dynamic_decoupling_single_file_outputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "NewCodebaseTests" / "xy8-128_overnight.mat"
    if not p.exists():
        return
    out = tmp_path / "out_dd"
    run_batch(input_dir=p.parent, output_dir=out, single_file=p, mode="contrast", bin_size=1, smooth_window=1)

    result_json = out / "xy8-128_overnight_result.json"
    fit_png = out / "xy8-128_overnight_fit.png"
    assert result_json.exists()
    assert fit_png.exists()
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["experiment_type"] == "DynamicDecoupling"
    assert payload["trace_metadata"]["sequence_name"] == "XY8-128"
    assert payload["trace_metadata"]["dd_pulse_count"] == 128
    assert "T2_dd_ns" in payload["nv_metrics"]
    assert payload["nv_metrics"]["dd_pulse_count"] == 128.0


def test_batch_snapshot_single_file_outputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "NewCodebaseTests" / "323RabiFullRun__checkpoint.mat"
    if not p.exists():
        return
    out = tmp_path / "out_snapshot"
    run_batch(input_dir=p.parent, output_dir=out, single_file=p, mode="contrast", bin_size=1, smooth_window=1)

    result_json = out / "323RabiFullRun__checkpoint_result.json"
    assert result_json.exists()
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["experiment_type"] == "Rabi"
    assert payload["trace_metadata"]["schema"] == "snapshot_run"
    assert payload["trace_metadata"]["save_type"] == "checkpoint"
    assert payload["trace_metadata"]["completed_iterations"] == 17
    assert "pi_time_ns" in payload["nv_metrics"]
    assert "pi_time_ns_nominal" in payload["nv_metrics"]
    assert "rabi_dead_time_ns" in payload["nv_metrics"]
    assert "T2rho_ns" in payload["nv_metrics"]
    assert "rabi_envelope" in payload["fit_extras"]


def test_batch_recursive_outputs_are_collision_safe_and_can_skip_checkpoints(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    scan = root / "TestData" / "1305_XZScan.mat"
    checkpoint = root / "TestData" / "NewCodebaseTests" / "323RabiFullRun__checkpoint.mat"
    if not scan.exists() or not checkpoint.exists():
        return
    inp = tmp_path / "input"
    (inp / "a").mkdir(parents=True)
    (inp / "b").mkdir(parents=True)
    shutil.copy2(scan, inp / "a" / "same.mat")
    shutil.copy2(scan, inp / "b" / "same.mat")
    shutil.copy2(checkpoint, inp / "b" / checkpoint.name)

    out = tmp_path / "out_recursive"
    run_batch(input_dir=inp, output_dir=out, mode="contrast", recursive=True, skip_checkpoints=True)

    assert (out / "a__same_result.json").exists()
    assert (out / "b__same_result.json").exists()
    assert not (out / f"b__{checkpoint.stem}_result.json").exists()
    summary = (out / "batch_summary.csv").read_text(encoding="utf-8")
    assert "relative_path" in summary
    assert "scan_dim" in summary


def test_batch_defaults_to_both_detectors_with_distinct_outputs_and_schema(tmp_path: Path):
    source = tmp_path / "dual_scan.mat"
    out = tmp_path / "dual_out"
    _write_dual_scan(source)

    run_batch(input_dir=tmp_path, output_dir=out, single_file=source, mode="contrast")

    assert (out / "dual_scan__detector1_result.json").exists()
    assert (out / "dual_scan__detector2_result.json").exists()
    with (out / "batch_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["detector_id"] for row in rows} == {"detector1", "detector2"}
    assert {row["detector_label"] for row in rows} == {"Detector 1", "Detector 2"}
    assert {row["detector_count"] for row in rows} == {"2"}


def test_batch_contains_one_invalid_detector_and_continues(tmp_path: Path):
    source = tmp_path / "partial_dual_scan.mat"
    out = tmp_path / "partial_out"
    _write_dual_scan(source, omit_detector2_values=True)

    run_batch(input_dir=tmp_path, output_dir=out, single_file=source, mode="contrast")

    with (out / "batch_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_detector = {row["detector_id"]: row for row in rows}
    assert by_detector["detector1"]["status"] == "PASS"
    assert by_detector["detector2"]["status"] == "FAIL"
    assert "does not contain Detector 2 values" in by_detector["detector2"]["status_reason"]
    assert (out / "partial_dual_scan__detector1_result.json").exists()
    assert (out / "partial_dual_scan__detector2_result.json").exists()

