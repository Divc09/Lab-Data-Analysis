"""Source-preserving iteration selection and exact-grid pooling (no interpolation)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
from typing import Callable

import numpy as np

from .io_mat import ExperimentDataset, ExperimentTrace


def iteration_matrix(tr: ExperimentTrace, mode: str) -> np.ndarray | None:
    if mode == "contrast":
        values = tr.iteration_contrast
    elif mode == "signal":
        values = tr.iteration_signal
    elif mode == "reference":
        values = tr.iteration_reference
    elif mode == "difference" and tr.iteration_reference is not None and tr.iteration_signal is not None:
        values = np.asarray(tr.iteration_reference, float) - np.asarray(tr.iteration_signal, float)
    else:
        values = None
    if values is None:
        return None
    matrix = np.asarray(values, float)
    if matrix.ndim != 2 or matrix.shape[0] != len(tr.x_ns):
        return None
    if tr.iteration_valid is not None and np.shape(tr.iteration_valid) == matrix.shape:
        matrix = np.where(np.asarray(tr.iteration_valid, bool), matrix, np.nan)
    return matrix


def selection_indices(count: int, selection: list[int] | None) -> list[int]:
    if selection is None:
        return list(range(count))
    if not isinstance(selection, list) or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < count for i in selection):
        raise ValueError(f"Iteration indices must be integers from 0 to {count - 1} (or null for all).")
    return sorted(set(int(i) for i in selection))


def iteration_sources(tr: ExperimentTrace) -> list[dict]:
    matrix = iteration_matrix(tr, tr.mode)
    if matrix is None:
        return []
    meta = tr.metadata or {}
    sources = meta.get("iteration_sources")
    if isinstance(sources, list) and len(sources) == matrix.shape[1]:
        return sources
    ids = meta.get("complete_iteration_indices") or list(range(1, matrix.shape[1] + 1))
    return [{"path": tr.source_path, "iteration": int(ids[i])} for i in range(matrix.shape[1])]


def matrix_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise sample SD (ddof=1) and SEM=SD/sqrt(n); n<2 is unknown, not zero."""
    matrix = np.asarray(matrix, float)
    valid = np.isfinite(matrix)
    n = valid.sum(axis=1)
    mean = np.divide(np.where(valid, matrix, 0).sum(axis=1), n, out=np.full(matrix.shape[0], np.nan), where=n > 0)
    ss = np.where(valid, matrix - mean[:, None], 0) ** 2
    var = np.divide(ss.sum(axis=1), n - 1, out=np.full(matrix.shape[0], np.nan), where=n > 1)
    sd = np.sqrt(var)
    return mean, sd, sd / np.sqrt(np.maximum(n, 1)), n


def selected_trace(tr: ExperimentTrace, mode: str, selection: list[int] | None, *, recompute: bool = False) -> ExperimentTrace:
    """Apply an iteration subset. With no selection the loader's own averages are kept unchanged: for some files
    (e.g. 317RabiFullRun, ODMR_20260324_024029) the loader's signal/reference/difference include data outside the
    complete-iteration matrix, so recomputing them would silently change the default curves. `recompute` forces the
    matrix mean (used for pooled datasets, which have no loader average)."""
    if selection is None and not recompute:
        return tr
    matrix = iteration_matrix(tr, mode)
    if matrix is None:
        if selection is not None:
            raise ValueError(f"No complete {mode} iteration matrix is available for this detector.")
        return tr
    indices = selection_indices(matrix.shape[1], selection)
    matrices = {}
    for key in ("contrast", "signal", "reference"):
        m = iteration_matrix(tr, key)
        matrices["iteration_" + key] = None if m is None else m[:, indices]
    mean, _, sem, _ = matrix_stats(matrix[:, indices])
    sources = iteration_sources(tr)
    meta = {**(tr.metadata or {}), "complete_iterations": len(indices), "selected_iterations": [i + 1 for i in indices],
            "iteration_sources": [sources[i] for i in indices], "selection_applied": True}
    sig = matrix_stats(matrices["iteration_signal"])[0] if matrices["iteration_signal"] is not None else tr.signal.copy()
    ref = matrix_stats(matrices["iteration_reference"])[0] if matrices["iteration_reference"] is not None else tr.reference.copy()
    return replace(tr, y=mean, signal=sig, reference=ref, y_sem=sem if len(indices) > 1 else None,
                   iteration_valid=None, metadata=meta, **matrices)


def _reject_shared_iterations(runs: list, traces: list, matrices: list) -> None:
    """Refuse to pool the same observation twice.

    A checkpoint file holds the first iterations of its final file, and a copied file holds all of them. Such
    iterations are bit-identical measured columns, so an exact column match between two runs is detected from
    the data itself, not guessed from file names.
    """
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            a, b = matrices[i], matrices[j]
            shared = sum(
                any(np.array_equal(a[:, p], b[:, q], equal_nan=True) for q in range(b.shape[1])) for p in range(a.shape[1])
            )
            if shared:
                raise ValueError(
                    f"{traces[i].file_name} and {traces[j].file_name} contain {shared} identical iteration(s): the same "
                    "observations (a checkpoint and its final run, or a copy). Pooling would count them twice; use the "
                    "final file, or deselect the shared iterations."
                )


def combine_datasets(load: Callable, recipe: dict, mode: str) -> ExperimentDataset:
    """Pool selected columns, retaining each source's exact sampled x coordinates."""
    runs = recipe.get("runs") or []
    if len(runs) < 2:
        raise ValueError("Choose at least two different runs.")
    paths = [r["path"] for r in runs]
    identities = [os.path.normcase(str(Path(p).resolve())) for p in paths]
    if len(set(identities)) != len(paths) or any(p.startswith("combined:") for p in paths):
        raise ValueError("Choose distinct source files, not already-combined datasets.")
    policy = recipe.get("grid", "identical")
    if policy not in ("identical", "intersect"):
        raise ValueError("Choose identical x grids or an explicit exact intersection.")
    datasets = [load(p, mode) for p in paths]
    first = datasets[0]
    if any(set(ds.available_detectors) != set(first.available_detectors) or ds.detector_mode != first.detector_mode for ds in datasets):
        raise ValueError("Detector layouts/modes differ. Only matching detector IDs and modes can be pooled.")
    if any(set(ds.detectors) != set(ds.available_detectors) for ds in datasets):
        raise ValueError("Every detector must load successfully before pooling; failed detectors cannot be substituted.")
    detectors = {}
    for det in first.available_detectors:
        traces = [ds.detectors[det] for ds in datasets]
        base = traces[0]
        if base.scan_dim == "scan2d" or base.experiment_type == "TimeTrace":
            raise ValueError("Pooling requires 1D experiment iterations, not maps or elapsed-time traces.")
        signature = lambda t: (t.experiment_type, t.scan_dim, t.x_label, (t.metadata or {}).get("x_unit"),
                               (t.metadata or {}).get("value_unit"), t.detector_label,
                               tuple(sorted((t.metadata or {}).get("available_observables") or [])),
                               bool((t.metadata or {}).get("reference_only")))
        if any(signature(t) != signature(base) for t in traces):
            raise ValueError("Experiment, detector label, observable availability, or native x/value units differ.")
        xs = [np.asarray(t.x_ns, float) for t in traces]
        if any(not np.all(np.isfinite(x)) or len(np.unique(x)) != len(x) for x in xs):
            raise ValueError("Pooling requires finite, unique measured x coordinates.")
        if policy == "identical":
            if any(not np.array_equal(xs[0], x) for x in xs[1:]):
                raise ValueError("X grids are not identical. Explicitly choose exact intersection; no interpolation is performed.")
            common = xs[0]
        else:
            common = xs[0]
            for x in xs[1:]:
                common = common[np.isin(common, x)]
            if len(common) < 5:
                raise ValueError("Exact intersection has fewer than 5 measured x coordinates.")
        pooled = {key: [] for key in ("contrast", "signal", "reference")}
        provenance = []
        retained = []
        for run, tr, x in zip(runs, traces, xs):
            selected = selected_trace(tr, mode, run.get("iterations"))
            m = iteration_matrix(selected, mode)
            if m is None or m.shape[1] == 0:
                raise ValueError(f"{tr.file_name}: choose at least one complete {mode} iteration for {det}.")
            idx = np.array([np.flatnonzero(x == v)[0] for v in common])
            if not np.all(np.isfinite(m[idx])):
                raise ValueError(f"{tr.file_name}: selected iterations contain missing/failed values on the pooled grid.")
            provenance.extend(iteration_sources(selected))
            retained.append({"path": tr.source_path, "original_points": len(x), "retained_points": len(common),
                             "iterations": [s["iteration"] for s in iteration_sources(selected)]})
            for key in pooled:
                matrix = iteration_matrix(selected, key)
                pooled[key].append(None if matrix is None else matrix[idx])
        _reject_shared_iterations(runs, traces, pooled["signal"] if all(m is not None for m in pooled["signal"]) else pooled["contrast"])
        matrices = {"iteration_" + key: np.concatenate(ms, axis=1) if all(m is not None for m in ms) else None for key, ms in pooled.items()}
        meta = {**(base.metadata or {}), "iteration_sources": provenance, "combined": {"grid": policy, "runs": retained,
                "note": "Pooled iteration mean; sample SEM assumes independent iterations. Checked: no identical iteration "
                        "appears in two runs (a checkpoint and its final file share them); no interpolation."},
                "complete_iterations": len(provenance), "complete_iteration_indices": list(range(1, len(provenance) + 1)),
                "target_iterations": len(provenance), "completed_iterations": len(provenance), "save_type": "combined", "run_status": "combined"}
        seed = replace(base, source_path=recipe["path"], file_name=recipe["name"], x_ns=common.copy(), raw_x=common.copy(),
                       y=np.zeros(len(common)), signal=np.zeros(len(common)), reference=np.zeros(len(common)),
                       y_sem=None, iteration_valid=None, metadata=meta, **matrices)
        detectors[det] = selected_trace(seed, mode, None, recompute=True)
    return ExperimentDataset(recipe["path"], recipe["name"], first.detector_mode, first.available_detectors,
                             detectors, {}, {"combined_recipe": recipe})
