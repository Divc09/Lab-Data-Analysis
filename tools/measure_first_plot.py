"""Measure SmartFitter's file-to-first-plot backend stages (audit EN04).

Opening a file in the app makes these server calls: one ``load`` without a detector (header/info for the
file state), then one ``load`` per detector for the plot. This script times the same work stage by stage,
cold (fresh Workspace, empty caches) and warm (same Workspace again), and reports the JSON size the browser
must download and parse. Browser transport and Plotly rendering are not included; measure those in the app.

Usage:  .venv\\Scripts\\python.exe tools\\measure_first_plot.py [file.mat ...]
        (default: every .mat file under TestData\\NewCodebaseTests)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nvfit.io_mat import load_experiment_dataset  # noqa: E402
from smartfitter import core  # noqa: E402


def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def measure(path: Path) -> dict:
    out = {"file": path.name, "size_kb": path.stat().st_size / 1024}
    t = time.perf_counter()
    load_experiment_dataset(path, "contrast")
    out["decode_ms"] = ms(t)  # MAT read + trace construction, no caches

    ws = core.Workspace()
    t = time.perf_counter()
    first = core.trace_payload(ws, str(path), None, "contrast", None)  # the app's initial "load"
    out["cold_first_load_ms"] = ms(t)
    detectors = list(first["info"]["detectors"])
    t = time.perf_counter()
    payloads = [core.trace_payload(ws, str(path), d, "contrast", {}) for d in detectors]  # loadData, per detector
    out["cold_detector_loads_ms"] = ms(t)
    t = time.perf_counter()
    body = [json.dumps({"ok": True, "result": core.clean(p)}, allow_nan=False) for p in [first, *payloads]]
    out["json_encode_ms"] = ms(t)
    out["json_kb"] = sum(len(b) for b in body) / 1024
    out["detectors"] = len(detectors)

    t = time.perf_counter()
    for d in detectors:
        core.trace_payload(ws, str(path), d, "contrast", {})
    out["warm_detector_loads_ms"] = ms(t)
    return out


def main(argv: list[str]) -> None:
    files = [Path(a) for a in argv] or sorted((ROOT / "TestData" / "NewCodebaseTests").glob("*.mat"))
    cols = ["file", "size_kb", "detectors", "decode_ms", "cold_first_load_ms", "cold_detector_loads_ms",
            "json_encode_ms", "json_kb", "warm_detector_loads_ms"]
    print("\t".join(cols))
    for path in files:
        try:
            row = measure(path)
        except Exception as exc:  # report and continue: one unreadable file should not stop the survey
            print(f"{path.name}\tERROR {exc}")
            continue
        print("\t".join(f"{row[c]:.1f}" if isinstance(row[c], float) else str(row[c]) for c in cols))


if __name__ == "__main__":
    main(sys.argv[1:])
