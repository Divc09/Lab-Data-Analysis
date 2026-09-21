"""Portable, versioned SmartFitter analysis-session helpers."""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any

SESSION_VERSION = 3
SUPPORTED_SESSION_VERSIONS = {1, 2, SESSION_VERSION}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_descriptor(path: str | Path, session_path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path).resolve()
    stat = p.stat()
    result: dict[str, Any] = {
        "absolute_path": str(p),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(p),
    }
    if session_path is not None:
        try:
            result["relative_path"] = os.path.relpath(p, Path(session_path).resolve().parent)
        except ValueError:
            # Windows cannot express a relative path across drive letters.
            pass
    return result


def resolve_source(descriptor: dict[str, Any], session_path: str | Path) -> tuple[Path | None, bool]:
    session_dir = Path(session_path).resolve().parent
    candidates = []
    if descriptor.get("relative_path"):
        candidates.append(session_dir / str(descriptor["relative_path"]))
    if descriptor.get("absolute_path"):
        candidates.append(Path(str(descriptor["absolute_path"])))
    for candidate in candidates:
        if candidate.exists():
            stat = candidate.stat()
            changed = stat.st_size != descriptor.get("size_bytes") or stat.st_mtime_ns != descriptor.get("mtime_ns")
            if descriptor.get("sha256") and not changed:
                changed = _sha256(candidate) != descriptor.get("sha256")
            return candidate.resolve(), changed
    return None, False


def save_session(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {"format": "nvfit-session", "version": SESSION_VERSION, **payload}
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, out)
    return out


def load_session(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != "nvfit-session" or document.get("version") not in SUPPORTED_SESSION_VERSIONS:
        raise ValueError("Unsupported SmartFitter session file.")
    source_version = int(document.get("version", 0))
    if source_version in {1, 2}:
        document = {
            **document,
            "version": SESSION_VERSION,
            "migrated_from": source_version,
            "active_detector_id": str(document.get("active_detector_id", "detector1")),
            "compare_detectors": bool(document.get("compare_detectors", False)),
            "detector_labels": dict(document.get("detector_labels") or {"detector1": "Detector 1", "detector2": "Detector 2"}),
        }
    return document
