"""NV center fitting toolkit (Python-first SmartFitter)."""

from .io_mat import ExperimentDataset, ExperimentTrace, MatDataError, load_experiment_dataset, load_saved_data_mat
from .preprocess import bin_trace, apply_roi, smooth_trace
from .fit_engine import FitResult, fit_model_multistart
from .scan_spots import SpotDetectionResult, SpotFit, detect_fluorescent_spots

__all__ = [
    "ExperimentDataset",
    "ExperimentTrace",
    "MatDataError",
    "load_experiment_dataset",
    "load_saved_data_mat",
    "bin_trace",
    "apply_roi",
    "smooth_trace",
    "FitResult",
    "fit_model_multistart",
    "SpotDetectionResult",
    "SpotFit",
    "detect_fluorescent_spots",
]

