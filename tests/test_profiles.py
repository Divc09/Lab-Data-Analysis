from nvfit.profiles import PROFILES, classify_status


def test_profiles_include_core_types():
    for name in ("Ramsey", "Rabi", "SpinEcho", "DynamicDecoupling", "T1", "ODMR"):
        assert name in PROFILES


def test_classify_status_rules():
    p = PROFILES["Ramsey"]
    status, _ = classify_status(r2=p.r2_min + 0.01, bound_hits=0, profile=p)
    assert status == "PASS"
    status, _ = classify_status(r2=p.r2_min - 0.2, bound_hits=0, profile=p)
    assert status == "FAIL"
    status, _ = classify_status(r2=p.r2_min + 0.01, bound_hits=p.max_bound_hits + 2, profile=p)
    assert status == "WARN"

