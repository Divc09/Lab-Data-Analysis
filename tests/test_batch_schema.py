import json
import shutil
from pathlib import Path

from nvfit.pro_batch import run_batch


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

