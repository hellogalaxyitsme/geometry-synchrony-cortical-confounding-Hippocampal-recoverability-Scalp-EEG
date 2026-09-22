"""Mathematical utilities for hippocampal EEG recoverability."""

from .recoverability import (
    ar1_finite_window_mi_rate_bits,
    ar1_spectral_mi_rate_bits,
    equicorrelated_sensor_power,
    forward_error_dprime_interval,
    gaussian_mi_bits,
    gaussian_posterior_covariance,
    geometry_efficiency,
    helmert_reference,
    kl_present_vs_absent_nats,
    matched_filter_auc,
    matched_filter_dprime,
    perturbation_mi_interval_bits,
    sensor_power_from_covariance,
    subset_gaussian_mi_bits,
    whitened_signal_eigenvalues,
    weighted_projection_residual,
)

__all__ = [
    "ar1_finite_window_mi_rate_bits",
    "ar1_spectral_mi_rate_bits",
    "equicorrelated_sensor_power",
    "forward_error_dprime_interval",
    "gaussian_mi_bits",
    "gaussian_posterior_covariance",
    "geometry_efficiency",
    "helmert_reference",
    "kl_present_vs_absent_nats",
    "matched_filter_auc",
    "matched_filter_dprime",
    "perturbation_mi_interval_bits",
    "sensor_power_from_covariance",
    "subset_gaussian_mi_bits",
    "whitened_signal_eigenvalues",
    "weighted_projection_residual",
]
