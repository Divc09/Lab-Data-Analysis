import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings, QCoreApplication, QEvent, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog

from nvfit.fit_workflows import FitWorkflowConfig, fit_non_ramsey
from nvfit.gui_app import SmartFitterMainWindow
from nvfit.io_mat import MatDataError, load_experiment_dataset, load_saved_data_mat
from nvfit.models import time_trace_decay
from nvfit.sessions import save_session


def write_time_snapshot(path, *, source="ODMR", reference_only=False, n=4, analysis=False, raw_timing=True, points=True, acquired=None):
    times = np.array([0., .7, 2.4, 4.05])[:n]
    ref = np.array([120., 110., 100., 90.])[:n]
    sig = np.zeros(n) if reference_only else .8 * ref
    values, values2 = np.empty((n, 1), object), np.empty((n, 1), object)
    for i in range(n):
        values[i, 0] = [ref[i], sig[i]]
        values2[i, 0] = [2 * ref[i], 2 * sig[i]]
    events = {"type": "optimization", "elapsedSeconds": np.array([.9, 1.6]), "details": {"axes": "z"}}
    timing = {"sourceExperiment": source, "sourceScan": {"parameter": "frequency", "identifier": "RF", "notes": source}, "fixedValue": 2.87,
              "timing": "scheduled", "durationSeconds": 5., "cadenceSeconds": 1., "events": events,
              "elapsedSeconds": times, "scheduledSeconds": np.arange(n), "latenessSeconds": np.arange(n) * .02}
    data = {"values": values, "detector2Values": values2, "iteration": np.ones(n) if acquired is None else acquired,
            "failedPoints": np.array([0, 2, 0, 0])[:n], "detectorMode": "dual"}
    if raw_timing:
        data["timeTrace"] = timing
    scan = {"bounds": np.array([0., 5.]), "stepSize": 1., "nSteps": n, "parameter": "Elapsed time (s)", "identifier": "Time"}
    if points:
        scan["points"] = times
    payload = {"Data": data, "scanInfo": scan, "params": {"scanMode": "time", "referenceOnly": reference_only, "perSecond": False},
               "metadata": {"timeTrace": timing, "runStatus": "checkpoint", "saveType": "checkpoint"}}
    if analysis:
        def metric(y, label):
            return {"x": np.linspace(0, 5, n), "y": y, "xLabel": "frequency", "yLabel": label}
        a = {"dimension": 1, "averageReference": metric(ref, "Reference"), "averageSignal": metric(sig, "Signal"), "averageContrast": metric(np.ones(n), "Contrast")}
        payload["analysis"] = {**a, "detector2": a}
    savemat(path, payload)
    return times, ref


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr("nvfit.gui_app.QSettings", lambda *_: settings)
    win = SmartFitterMainWindow()
    messages = []
    win._message = lambda *args: messages.append(args)
    win.test_messages = messages
    yield win
    app.processEvents()
    win.close()
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


@pytest.mark.parametrize("source", ["ODMR", "Rabi", "DynamicDecoupling", "FluorescenceTrace"])
def test_measured_timestamps_override_sequence_and_analysis_axes(tmp_path, source):
    path = tmp_path / f"{source}_Time_Trace.mat"
    x, ref = write_time_snapshot(path, source=source, reference_only=source == "FluorescenceTrace", analysis=True)
    dataset = load_experiment_dataset(path)
    assert not dataset.detector_errors
    for detector, scale in (("detector1", 1), ("detector2", 2)):
        trace = dataset.detectors[detector]
        assert trace.experiment_type == "TimeTrace"
        assert trace.scan_dim == "time"
        np.testing.assert_array_equal(trace.x_ns, x)
        np.testing.assert_array_equal(trace.reference, scale * ref)
        assert trace.iteration_contrast is None
        assert trace.metadata["time_sampling_irregular"]
        assert trace.metadata["time_trace"]["failedPoints"][1] == 2
        assert trace.metadata["time_trace"]["fixedValue"] == 2.87


def test_reference_only_has_no_fabricated_signal_or_contrast(tmp_path):
    path = tmp_path / "fluorescence.mat"
    x, ref = write_time_snapshot(path, source="FluorescenceTrace", reference_only=True)
    for mode in ("contrast", "reference", "signal", "difference"):
        trace = load_saved_data_mat(path, mode=mode)
        assert trace.mode == "reference"
        np.testing.assert_array_equal(trace.y, ref)
        assert np.all(np.isnan(trace.signal))
        assert trace.y_label == "Fluorescence [counts]"


@pytest.mark.parametrize("n", [1, 4])
def test_raw_checkpoint_fallback_uses_scan_points_including_single_sample(tmp_path, n):
    path = tmp_path / "checkpoint.mat"
    x, _ = write_time_snapshot(path, n=n, raw_timing=False)
    trace = load_saved_data_mat(path)
    np.testing.assert_array_equal(trace.x_ns, x)
    assert trace.fit_allowed == (n > 3)
    assert trace.metadata["time_axis_source"] == "scanInfo.points"


def test_missing_measured_timestamps_are_not_reconstructed_from_cadence(tmp_path):
    path = tmp_path / "missing.mat"
    write_time_snapshot(path, raw_timing=False, points=False)
    with pytest.raises(MatDataError, match="measured elapsed"):
        load_saved_data_mat(path)


def test_pending_checkpoint_cell_is_omitted_but_successful_retry_is_kept(tmp_path):
    path = tmp_path / "partial.mat"
    x, ref = write_time_snapshot(path, acquired=[1, 1, 1, 0])
    trace = load_saved_data_mat(path, mode="reference")
    np.testing.assert_array_equal(trace.x_ns, x[:3])
    np.testing.assert_array_equal(trace.y, ref[:3])
    assert trace.metadata["source_point_indices"] == [0, 1, 2]


def test_exponential_fit_on_irregular_seconds_recovers_decay_time():
    times = 400 * np.linspace(0, 1, 71) ** 1.3
    expected = [180000., 900000., 42.5]
    y = time_trace_decay(times, *expected)
    result = fit_non_ramsey("TimeTraceDecay", times, y, config=FitWorkflowConfig(multistart=5))
    np.testing.assert_allclose(result.params, expected, rtol=1e-5)
    assert result.extras["parameter_units"]["tau_s"] == "s"


def test_gui_time_units_observables_events_and_domain_compatibility(window, tmp_path):
    path = tmp_path / "fluorescence.mat"
    write_time_snapshot(path, source="FluorescenceTrace", reference_only=True)
    assert window._load_file(str(path))
    window.canvas.draw()
    assert window.mode_combo.currentText() == "reference"
    assert window._active_profile().name == "TimeTrace"
    assert window.model_combo.currentText() == "TimeTraceDecay"
    assert window.roi_min_label.text() == "ROI min (s)"
    assert window.ax_main.get_xlabel() == "Elapsed time (s)"
    assert window.ax_main.get_ylabel() == "Fluorescence [counts]"
    assert window.quick_signal_btn.isHidden() and window.quick_fft_btn.isHidden()
    assert len(window.ax_main.patches) == 1
    window.quick_time_events_btn.setChecked(False)
    assert len(window.ax_main.patches) == 0
    normal_path = tmp_path / "Rabi.mat"
    write_time_snapshot(normal_path)
    payload = loadmat(normal_path, simplify_cells=True)
    payload["params"]["scanMode"] = "parameter"
    payload["scanInfo"].update(parameter="duration", bounds=[0., 3.], stepSize=1.)
    payload["metadata"].pop("timeTrace")
    payload["Data"].pop("timeTrace")
    savemat(normal_path, {key: val for key, val in payload.items() if not key.startswith('__')})
    normal = load_saved_data_mat(normal_path)
    assert not window._overlay_is_compatible(normal)
    window.on_fft()
    assert window.test_messages[-1][0] == "FFT unavailable"


def test_gui_time_decay_worker_fits_both_detectors_in_seconds(window, tmp_path):
    path = tmp_path / "decay.mat"
    write_time_snapshot(path, reference_only=True, source="FluorescenceTrace")
    payload = loadmat(path, simplify_cells=True)
    times = 240 * np.linspace(0, 1, 45) ** 1.2
    expected = np.array([180000., 900000., 42.5])
    reference = time_trace_decay(times, *expected)
    for field, scale in (("values", 1), ("detector2Values", 2)):
        cells = np.empty((len(times), 1), object)
        for index, value in enumerate(reference):
            cells[index, 0] = [scale * value, 0.]
        payload["Data"][field] = cells
    payload["Data"]["iteration"] = np.ones(len(times))
    payload["Data"]["failedPoints"] = np.zeros(len(times))
    payload["Data"]["timeTrace"]["elapsedSeconds"] = times
    payload["scanInfo"]["points"] = times
    payload["scanInfo"]["nSteps"] = len(times)
    savemat(path, {key: val for key, val in payload.items() if not key.startswith('__')})
    assert window._load_file(str(path))
    window.detector_scope_combo.setCurrentIndex(window.detector_scope_combo.findData("both"))
    window.multistart_spin.setValue(4)
    window._start_fit()
    loop, poll = QEventLoop(), QTimer()
    poll.timeout.connect(lambda: loop.quit() if window._active_fit_request is None and window._dual_fit_origin is None else None)
    poll.start(20)
    QTimer.singleShot(20000, loop.quit)
    loop.exec()
    poll.stop()
    assert window._active_fit_request is None
    assert len(window._dual_fit_summaries) == 2
    assert window.ctx.fit_result.model_name == "TimeTraceDecay"
    np.testing.assert_allclose(window.ctx.fit_result.params, expected, rtol=1e-5)
    assert window.ax_main.get_xlabel() == "Elapsed time (s)"
    assert "tau_s" in window.summary_text.toPlainText()
    assert not window.test_messages


def test_time_trace_rejects_nanosecond_physical_fit(tmp_path):
    path = tmp_path / "trace.mat"
    write_time_snapshot(path)
    trace = load_saved_data_mat(path)
    with pytest.raises(ValueError, match="seconds"):
        fit_non_ramsey("Rabi", trace.x_ns, trace.y, trace=trace)


REAL_TIME_TRACE = Path(r"C:\Users\Div\Downloads\Raw_Fluorescence_Time_Trace1.mat")


@pytest.mark.skipif(not REAL_TIME_TRACE.exists(), reason="Real fluorescence trace is not available")
def test_real_dual_trace_matches_all_raw_values_and_preserves_rates(window):
    raw = loadmat(REAL_TIME_TRACE, squeeze_me=True, struct_as_record=False)
    assert window._load_file(str(REAL_TIME_TRACE))
    window.compare_detectors_chk.setChecked(True)
    for detector, attr in (("detector1", "values"), ("detector2", "detector2Values")):
        trace = window._primary_dataset.detectors[detector]
        expected = np.array([np.asarray(cell).ravel()[0] for cell in getattr(raw["Data"], attr)])
        np.testing.assert_array_equal(trace.reference, expected)
        np.testing.assert_array_equal(trace.x_ns, raw["Data"].timeTrace.elapsedSeconds)
        assert trace.y_label == "Fluorescence [counts/s]"
    assert window.ctx.trace.metadata["time_trace_point_count"] == 614
    window.canvas.draw()
    assert not window.test_messages


def test_time_trace_session_and_export_preserve_timing(window, tmp_path, monkeypatch):
    path = tmp_path / "fluorescence.mat"
    write_time_snapshot(path, source="FluorescenceTrace", reference_only=True)
    assert window._load_file(str(path))
    session = tmp_path / "trace.nvfit-session.json"
    window.quick_time_events_btn.setChecked(False)
    save_session(session, window._session_payload(str(session)))
    window.quick_time_events_btn.setChecked(True)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(session), ""))
    window.on_load_session()
    assert not window.quick_time_events_btn.isChecked()
    assert window.mode_combo.currentText() == "reference"
    out = tmp_path / "export"
    out.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(out))
    window.exp_png_chk.setChecked(False)
    window.exp_pdf_chk.setChecked(False)
    window.exp_svg_chk.setChecked(False)
    window.exp_json_chk.setChecked(True)
    window.on_save_results()
    result = json.loads(next(out.glob('*_gui_result.json')).read_text())
    assert result["trace_metadata"]["x_unit"] == "s"
    assert result["trace_metadata"]["time_trace"]["elapsedSeconds"] == [0., .7, 2.4, 4.05]
    assert result["series"]["x_processed"] == [0., .7, 2.4, 4.05]
    import csv
    rows = list(csv.DictReader(next(out.glob('*__time_trace.csv')).open(newline='')))
    assert len(rows) == 8
    assert rows[1]["retries"] == "2.0"
    assert all(row["signal"] == row["contrast"] == row["difference"] == "" for row in rows)


def test_batch_time_trace_defaults_to_plotting_and_exports_seconds(tmp_path):
    from nvfit.pro_batch import run_batch
    source = tmp_path / "source"
    source.mkdir()
    write_time_snapshot(source / "fluorescence.mat", source="FluorescenceTrace", reference_only=True)
    out = tmp_path / "out"
    run_batch(source, out, multistart=1, dpi=70)
    results = list(out.glob('*_result.json'))
    assert len(results) == 2
    for file in results:
        document = json.loads(file.read_text())
        assert document["trace_metadata"]["x_unit"] == "s"
        assert document["provenance"]["mode"] == "reference"
        assert document["status"] == "PASS"


@pytest.mark.parametrize("model", ["TimeTraceDecay", "Rabi"])
def test_batch_time_trace_optional_fit_and_wrong_domain_model(tmp_path, model):
    from nvfit.pro_batch import run_batch
    source = tmp_path / "trace.mat"
    times, _ = write_time_snapshot(source, source="FluorescenceTrace", reference_only=True)
    payload = loadmat(source, simplify_cells=True)
    reference = time_trace_decay(times, 11., 80., 1.3)
    for field, scale in (("values", 1), ("detector2Values", 2)):
        values = np.empty((len(times), 1), object)
        for i, y in enumerate(reference):
            values[i, 0] = [scale * y, 0.]
        payload["Data"][field] = values
    savemat(source, {key: val for key, val in payload.items() if not key.startswith('__')})
    out = tmp_path / "out"
    run_batch(tmp_path, out, single_file=source, model_override=model, multistart=4, dpi=70)
    for file in out.glob('*_result.json'):
        document = json.loads(file.read_text())
        if model == "TimeTraceDecay":
            assert document["selected_model"] == "TimeTraceDecay"
            assert document["params"]["tau_s"] == pytest.approx(1.3, rel=1e-5)
            assert document["fit_extras"]["parameter_units"]["tau_s"] == "s"
        else:
            assert document["status"] == "FAIL"
            assert "seconds" in document["status_reason"]


def test_batch_time_trace_honors_explicit_exclusions(tmp_path):
    from nvfit.pro_batch import run_batch
    source = tmp_path / "trace.mat"
    write_time_snapshot(source)
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({source.name: {"exclude": True}}))
    out = tmp_path / "out"
    run_batch(tmp_path, out, single_file=source, exclude_file=exclusions, dpi=70)
    for file in out.glob('*_result.json'):
        assert json.loads(file.read_text())["status"] == "EXCLUDED"
