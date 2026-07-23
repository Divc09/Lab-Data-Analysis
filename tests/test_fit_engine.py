from pathlib import Path

import numpy as np
import pytest

from nvfit import fit_engine
from nvfit.fit_engine import fit_model_multistart
from nvfit.io_mat import load_saved_data_mat
from nvfit.models import ramsey_simple
from nvfit.pro_batch import _fit_non_ramsey
from nvfit.ramsey import fit_ramsey_physics_first


def test_fit_ramsey_physics_first_real_data():
    root = Path(__file__).resolve().parents[1]
    path = root / "LongRamsey.mat"
    if not path.exists():
        path = root / "TestData" / "LongRamsey.mat"
    tr = load_saved_data_mat(path, mode="contrast")
    sel = fit_ramsey_physics_first(tr.x_ns, tr.y)
    assert sel.best.success
    assert np.isfinite(sel.best.r2)
    # Keep threshold moderate for robustness across machines.
    assert sel.best.r2 > 0.75


def test_ramsey_simple_synthetic_recovers_reasonable_r2():
    rng = np.random.default_rng(5)
    x = np.linspace(20, 4020, 401)
    y_true = ramsey_simple(x, -0.06, 0.02, 0.0032, 0.2, 1200.0, 1.8)
    y_noisy = y_true + rng.normal(0, 0.0015, size=len(x))
    sel = fit_ramsey_physics_first(x, y_noisy)
    assert sel.best.r2 > 0.8


def test_dynamic_decoupling_fit_uses_expanded_t2_range():
    root = Path(__file__).resolve().parents[1]
    p = root / "TestData" / "NewCodebaseTests" / "xy8-128_overnight.mat"
    if not p.exists():
        return
    tr = load_saved_data_mat(p, mode="contrast")
    res = _fit_non_ramsey("DynamicDecoupling", tr.x_ns, tr.y, n_starts=12)
    assert res.success
    assert np.isfinite(res.r2)
    t2 = float(res.params[list(res.param_names).index("T2")])
    assert t2 > 20000.0
    assert not res.bound_hits[list(res.param_names).index("T2")]


def test_multistart_retains_best_values_when_one_start_raises(monkeypatch: pytest.MonkeyPatch):
    real_least_squares = fit_engine.least_squares
    calls = 0

    def flaky_least_squares(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic failed start")
        return real_least_squares(*args, **kwargs)

    monkeypatch.setattr(fit_engine, "least_squares", flaky_least_squares)
    x = np.linspace(0.0, 5.0, 30)
    y = 2.0 * x + 1.0
    result = fit_model_multistart(
        model_name="Linear",
        model_func=lambda t, m, b: m * t + b,
        x=x,
        y=y,
        param_names=["m", "b"],
        lower=[-10.0, -10.0],
        upper=[10.0, 10.0],
        seeds=[np.array([0.0, 0.0]), np.array([1.5, 0.5])],
    )

    assert result.success
    assert result.params == pytest.approx([2.0, 1.0])
    assert "retained best result after 1 of 2 starts failed" in result.message

