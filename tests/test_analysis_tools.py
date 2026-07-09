from __future__ import annotations

import numpy as np
import pytest

from nvfit.analysis_tools import AnalysisStep, apply_steps, process_series
from nvfit.sessions import load_session, resolve_source, save_session, source_descriptor


def test_process_series_preserves_sources_and_raw_index_masks():
    x = np.arange(8.0)
    y = x.copy()
    original = y.copy()
    processed = process_series(x, y, bin_size=2, excluded_raw_indices={0}, exclusion_ranges=[(5.5, 6.5)])
    assert np.allclose(y, original)
    assert np.allclose(processed.fit_x, [1.0, 2.5, 4.5, 7.0])
    assert [group.tolist() for group in processed.source_groups] == [[1], [2, 3], [4, 5], [7]]


@pytest.mark.parametrize(
    "step",
    [
        AnalysisStep("baseline", {"method": "median"}),
        AnalysisStep("baseline", {"method": "linear"}),
        AnalysisStep("detrend"),
        AnalysisStep("normalize", {"method": "minmax"}),
        AnalysisStep("normalize", {"method": "zscore"}),
        AnalysisStep("normalize", {"method": "area"}),
        AnalysisStep("derivative"),
        AnalysisStep("integral"),
        AnalysisStep("resample", {"count": 11}),
        AnalysisStep("expression", {"expression": "sin(x) + y"}),
    ],
)
def test_analysis_steps_return_finite_vector(step):
    x = np.linspace(0.0, 1.0, 9)
    y = 1.0 + 2.0 * x
    out_x, out_y, provenance = apply_steps(x, y, [step])
    assert len(out_x) == len(out_y)
    assert np.all(np.isfinite(out_y))
    assert provenance[0]["kind"] == step.kind


def test_expression_rejects_code_like_syntax():
    with pytest.raises(ValueError):
        apply_steps(np.arange(3.0), np.arange(3.0), [AnalysisStep("expression", {"expression": "__import__('os')"})])


def test_session_round_trip_and_changed_source_warning(tmp_path):
    source = tmp_path / "trace.mat"
    source.write_bytes(b"first")
    session_path = tmp_path / "session.nvfit-session.json"
    descriptor = source_descriptor(source, session_path)
    save_session(session_path, {"primary_source": descriptor, "analysis_steps": []})
    loaded = load_session(session_path)
    resolved, changed = resolve_source(loaded["primary_source"], session_path)
    assert resolved == source.resolve() and not changed
    source.write_bytes(b"second")
    _resolved, changed = resolve_source(loaded["primary_source"], session_path)
    assert changed
