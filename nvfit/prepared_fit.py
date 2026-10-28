"""Shared, immutable request used by ordinary, robust and resampled fitting."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, replace
import numpy as np

from .diagnostics import aic, bic, durbin_watson, r_squared, rmse
from .fit_engine import sem_to_weights
from .fit_workflows import (
    FitWorkflowConfig,
    aligned_sem,
    fit_non_ramsey,
    model_candidate_summary,
    _prepare_t1_series,
    _rabi_fit_weights,
    parameter_control_model,
)
from .models import MODEL_REGISTRY, build_custom_expression_model
from .ramsey import fit_ramsey_physics_first, ramsey_weights, ramsey_control_model


def evaluate_result(result, x):
    model = result.model_func or MODEL_REGISTRY.get(result.model_name)
    if model is None and result.extras.get("custom_spec"):
        spec = result.extras["custom_spec"]
        model = build_custom_expression_model(spec["expression"], spec["param_names"])
    if model is None:
        raise ValueError(f"No evaluator for {result.model_name}")
    offset = float(result.extras.get("x_offset_ns") or 0.0)
    return np.asarray(model(np.asarray(x, float) - offset, *result.params), float)


@dataclass(frozen=True)
class PreparedFit:
    x: np.ndarray
    y: np.ndarray
    sem: np.ndarray | None
    family: str
    config: FitWorkflowConfig
    locks: dict | None
    trace: object | None
    ramsey_model: str = "Auto"

    @classmethod
    def create(cls, x, y, family, config=None, locks=None, trace=None, ramsey_model="Auto"):
        x, y = np.array(x, float, copy=True), np.array(y, float, copy=True)
        if x.shape != y.shape or x.ndim != 1 or not np.all(np.isfinite(x) & np.isfinite(y)):
            raise ValueError("Fit target requires paired finite one-dimensional arrays.")
        cfg = deepcopy(config or FitWorkflowConfig())
        if int(cfg.multistart) < 1:
            raise ValueError("Multistart must be positive.")
        sem = aligned_sem(trace, x)
        sem = np.array(sem, copy=True) if sem is not None else None
        if family == "T1":
            norm = {}
            _, prep = _prepare_t1_series(
                x, y, prep_mode=cfg.t1_prep_mode, norm_extras=norm, normalization=cfg.t1_normalization
            )
            modes = {
                "normalized_1_minus_data": "Normalize then 1-data",
                "direct_decay": "Direct decay",
                "direct": "Direct T1",
            }
            cfg = replace(
                cfg,
                t1_prep_mode=modes[prep],
                t1_normalization=(norm["t1_norm_lo"], norm["t1_norm_span"]) if norm else None,
            )
        if cfg.parameter_model is None:
            cfg = replace(
                cfg,
                parameter_model=(
                    ramsey_control_model(ramsey_model)
                    if family == "Ramsey"
                    else parameter_control_model(family, cfg, x, y)
                ),
            )
        x.setflags(write=False)
        y.setflags(write=False)
        if sem is not None:
            sem.setflags(write=False)
        return cls(x, y, sem, family, cfg, deepcopy(locks), trace, ramsey_model)

    def trace_for(self, indices):
        if self.trace is None:
            return None
        tr = copy(self.trace)
        tr.x_ns, tr.y = self.x[indices].copy(), self.y[indices].copy()
        tr.y_sem = self.sem[indices].copy() if self.sem is not None else None
        return tr

    def single(self, indices=None):
        indices = np.arange(len(self.x)) if indices is None else indices
        x, y, tr = self.x[indices], self.y[indices], self.trace_for(indices)
        if self.family == "Ramsey":
            sel = fit_ramsey_physics_first(
                x,
                y,
                sem=tr.y_sem if tr is not None and self.config.sem_weights else None,
                multistart=self.config.multistart,
                locks=self.locks,
                model=self.ramsey_model,
                parameter_model=self.config.parameter_model,
            )
            candidates = [model_candidate_summary(c, chosen=c is sel.best) for c in sel.candidates]
            sel.best.extras.update(ramsey_candidates=candidates, ramsey_selected_by=sel.selected_by)
            return sel.best, candidates
        result = fit_non_ramsey(
            self.family,
            x,
            y,
            config=self.config,
            locks=self.locks,
            custom_spec=self.config.custom_spec,
            trace_experiment_type=getattr(tr, "experiment_type", self.family),
            trace=tr,
        )
        if self.family == "CustomModel":
            result.extras["custom_spec"] = deepcopy(self.config.custom_spec)
        return result, None

    def target(self):
        if self.family != "T1":
            return self.y
        return _prepare_t1_series(
            self.x, self.y, prep_mode=self.config.t1_prep_mode, normalization=self.config.t1_normalization
        )[0]

    def weights(self):
        if self.family == "Ramsey":
            return ramsey_weights(self.x, self.y, self.sem if self.config.sem_weights else None)
        if self.family in ("Rabi", "DEERDuration"):
            return _rabi_fit_weights(self.trace_for(np.arange(len(self.x))), self.x, enabled=self.config.rabi_use_sem)
        return sem_to_weights(self.sem) if self.config.sem_weights else None

    def run(self, robust=False, progress=None):
        base, candidates = self.single()
        if not robust:
            return base, candidates
        target, weights = self.target(), self.weights()
        weights = np.ones_like(target) if weights is None else weights
        scored, failures = [], []
        n = len(self.x)
        cuts = [
            ("full", 0, 0),
            ("trim_start4", 0.04, 0),
            ("trim_start8", 0.08, 0),
            ("trim_end5", 0, 0.05),
            ("trim_both4", 0.04, 0.04),
            ("trim_start12", 0.12, 0),
            ("trim_end10", 0, 0.1),
        ]
        for label, start, end in cuts:
            idx = np.arange(int(n * start), n - int(n * end))
            if label != "full" and len(idx) < max(10, len(base.params) + 1):
                continue
            if progress:
                progress(f"Robust: {label}")
            try:
                result = base if label == "full" else self.single(idx)[0]
                pred = evaluate_result(result, self.x)
                if not result.success or pred.shape != target.shape or not np.all(np.isfinite(pred)):
                    raise ValueError("unconverged or invalid candidate")
                score = float(np.sum(weights * (target - pred) ** 2) / np.sum(weights))
                scored.append((score, label, result, pred))
            except Exception as exc:
                failures.append(f"{label}: {exc}")
        if not scored:
            base.extras["robust_failures"] = failures
            return base, candidates
        _, label, best, pred = min(scored, key=lambda row: row[0])
        extras = dict(best.extras)
        if "plot_y" in extras:
            extras["plot_y"] = target.copy()
        extras.update(
            robust_subset=label,
            robust_candidates=[{"subset": s, "weighted_mse": v} for v, s, _, _ in scored],
            robust_failures=failures,
        )
        nfree = sum(best.free_mask) if best.free_mask is not None else len(best.params)
        result = replace(
            best,
            y_fit=pred,
            r2=r_squared(target, pred),
            rmse=rmse(target, pred),
            aic=aic(target, pred, nfree),
            bic=bic(target, pred, nfree),
            durbin_watson=durbin_watson(target - pred),
            extras=extras,
            message=best.message
            + f"\nRobust subset: {label}; candidates scored on the full prepared target with its weights.",
        )
        return result, candidates
