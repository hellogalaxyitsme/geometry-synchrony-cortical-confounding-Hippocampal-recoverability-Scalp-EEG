"""Structured synthetic experiments for hippocampal EEG recoverability."""

from .synthetic import (
    build_cortical_subspace,
    build_sensor_modes,
    compute_recoverability_metrics,
    perturb_contributions,
    source_covariance,
    structured_contributions,
)

__all__ = [
    "build_cortical_subspace",
    "build_sensor_modes",
    "compute_recoverability_metrics",
    "perturb_contributions",
    "source_covariance",
    "structured_contributions",
]
