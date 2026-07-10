import numpy as np
import pytest

from nvfit.models import (
    build_custom_expression_model,
    odmr_multi_lorentzian,
    rabi_dual_model,
    ramsey_extended,
    ramsey_hyperfine_n14,
    ramsey_simple,
)


def test_ramsey_models_return_finite():
    t = np.linspace(20, 4020, 401)
    y1 = ramsey_simple(t, -0.06, 0.02, 0.003, 0.0, 1000.0, 2.0)
    y2 = ramsey_hyperfine_n14(t, -0.06, 0.02, 0.003, 0.0, 1000.0, 2.0)
    y3 = ramsey_extended(t, 0.01, 0.005, 800.0, 1.8, 0.003, 0.0018, 0.1, -0.2, -0.06, 0.0, 1000.0)
    assert np.isfinite(y1).all()
    assert np.isfinite(y2).all()
    assert np.isfinite(y3).all()
    assert y1.shape == t.shape


def test_rabi_dual_model_return_finite():
    t = np.linspace(0, 5000, 800)
    y = rabi_dual_model(t, 0.01, 0.0, 0.03, 0.015, 0.002, 0.0025, 0.0, 0.4, 1800.0)
    assert np.isfinite(y).all()
    assert y.shape == t.shape


def test_odmr_multi_and_custom_model_finite():
    t = np.linspace(2.1, 2.3, 300)
    y = odmr_multi_lorentzian(t, 0.1, -0.02, 0.002, 2.18, -0.03, 0.0025, 2.205)
    assert np.isfinite(y).all()
    model = build_custom_expression_model("y0 + A*np.cos(2*np.pi*f*t+phi)", ["y0", "A", "f", "phi"])
    y2 = model(t, 0.01, 0.02, 0.005, 0.0)
    assert np.isfinite(y2).all()


@pytest.mark.parametrize(
    "expression",
    ["np.__dict__", "t[0]", "[value for value in t]", "__import__('os')", "(1).__class__"],
)
def test_custom_model_expression_rejects_unsafe_syntax(expression):
    with pytest.raises(ValueError):
        build_custom_expression_model(expression, ["A"])

