from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProfileName = Literal[
    "Ramsey",
    "Rabi",
    "SpinEcho",
    "DynamicDecoupling",
    "T1",
    "ODMR",
    "LineScan",
    "DEERFrequency",
    "DEERDuration",
    "DEERDecay",
    "DEERPosition",
]


@dataclass(frozen=True)
class FitProfile:
    name: ProfileName
    mode: str
    bin_size: int
    smooth_window: int
    roi_head_fraction: float | None
    candidate_models: tuple[str, ...]
    r2_min: float
    max_bound_hits: int
    notes: str


PROFILES: dict[ProfileName, FitProfile] = {
    "Ramsey": FitProfile(
        name="Ramsey",
        mode="contrast",
        bin_size=1,
        smooth_window=1,
        roi_head_fraction=None,
        candidate_models=("RamseyAuto", "RamseySimple", "RamseyHyperfine", "RamseyExtended"),
        r2_min=0.80,
        max_bound_hits=3,
        notes="Use contrast and preserve early oscillations; robust multistart recommended.",
    ),
    "Rabi": FitProfile(
        name="Rabi",
        mode="contrast",
        bin_size=1,
        smooth_window=3,
        roi_head_fraction=None,
        candidate_models=("Rabi",),
        r2_min=0.75,
        max_bound_hits=2,
        notes="Moderate smoothing allowed; extract pi and pi/2 times from fitted frequency.",
    ),
    "SpinEcho": FitProfile(
        name="SpinEcho",
        mode="contrast",
        bin_size=1,
        smooth_window=3,
        roi_head_fraction=None,
        candidate_models=("SpinEcho",),
        r2_min=0.70,
        max_bound_hits=2,
        notes="Stretched exponential decay expected; check residuals for baseline drift.",
    ),
    "DynamicDecoupling": FitProfile(
        name="DynamicDecoupling",
        mode="contrast",
        bin_size=1,
        smooth_window=3,
        roi_head_fraction=None,
        candidate_models=("SpinEcho",),
        r2_min=0.70,
        max_bound_hits=2,
        notes="DD sequences are fit on total evolution time with a stretched exponential decay model.",
    ),
    "T1": FitProfile(
        name="T1",
        mode="contrast",
        bin_size=1,
        smooth_window=5,
        roi_head_fraction=None,
        candidate_models=("T1",),
        r2_min=0.70,
        max_bound_hits=2,
        notes="Support direct T1 and normalized 1-data T1 fits with exponential or stretched-exponential models.",
    ),
    "ODMR": FitProfile(
        name="ODMR",
        mode="contrast",
        bin_size=1,
        smooth_window=1,
        roi_head_fraction=None,
        candidate_models=("ODMR",),
        r2_min=0.70,
        max_bound_hits=2,
        notes="Fit ODMR line shape with automatic peak/dip polarity detection and center/linewidth extraction.",
    ),
    "LineScan": FitProfile(
        name="LineScan",
        mode="contrast",
        bin_size=1,
        smooth_window=1,
        roi_head_fraction=None,
        candidate_models=("SpinEcho",),
        r2_min=0.60,
        max_bound_hits=4,
        notes="Fallback profile for unknown experiments.",
    ),
    "DEERFrequency": FitProfile(
        name="DEERFrequency",
        mode="contrast",
        bin_size=1,
        smooth_window=1,
        roi_head_fraction=None,
        candidate_models=("DEERFrequency",),
        r2_min=0.65,
        max_bound_hits=3,
        notes="DEER RF2 frequency sweeps use ODMR-style peak/dip fitting while preserving DEER metadata.",
    ),
    "DEERDuration": FitProfile(
        name="DEERDuration",
        mode="contrast",
        bin_size=1,
        smooth_window=3,
        roi_head_fraction=None,
        candidate_models=("DEERDuration",),
        r2_min=0.60,
        max_bound_hits=3,
        notes="DEER RF2 duration sweeps use Rabi-style oscillation fits and report DEER context.",
    ),
    "DEERDecay": FitProfile(
        name="DEERDecay",
        mode="contrast",
        bin_size=1,
        smooth_window=3,
        roi_head_fraction=None,
        candidate_models=("DEERDecay",),
        r2_min=0.60,
        max_bound_hits=3,
        notes="DEER tau/evolution sweeps use stretched-decay fits.",
    ),
    "DEERPosition": FitProfile(
        name="DEERPosition",
        mode="contrast",
        bin_size=1,
        smooth_window=1,
        roi_head_fraction=None,
        candidate_models=("LineScan",),
        r2_min=0.60,
        max_bound_hits=4,
        notes="DEER position scans are plot-only unless a physical model is defined later.",
    ),
}


def classify_status(*, r2: float, bound_hits: int, profile: FitProfile) -> tuple[str, str]:
    if r2 < profile.r2_min:
        return "FAIL", f"R^2 below profile minimum ({r2:.3f} < {profile.r2_min:.3f})"
    if bound_hits > profile.max_bound_hits:
        return "WARN", f"Many bound hits ({bound_hits})"
    return "PASS", "Quality checks passed"

