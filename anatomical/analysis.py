"""Recoverability metrics for a validated anatomical bundle."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from theory.recoverability import (
    gaussian_mi_bits,
    geometry_efficiency,
    kl_present_vs_absent_nats,
    weighted_projection_residual,
    whitened_signal_eigenvalues,
)

from .bundle import AnatomicalBundle
from .operators import (
    density_operator,
    propagated_covariance,
    reference_covariance,
    reference_leadfield,
    validate_sensor_indices,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CorticalRestriction:
    raw_basis: FloatArray
    raw_nuisance_covariance: FloatArray
    requested_rank: int
    actual_rank: int
    retained_spectral_energy_fraction: float
    variance_normalization: str


def _leading_topography(covariance: FloatArray) -> FloatArray:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    vector = vectors[:, int(np.argmax(values))]
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0:
        vector *= -1.0
    return vector


def build_cortical_restriction(
    bundle: AnatomicalBundle,
    rank: int,
    cortical_total_variance: float,
    normalization: str = "area_average",
    variance_normalization: str = "fixed_total",
) -> CorticalRestriction:
    """Build one full-sensor cortical model shared by all nested montages."""

    maximum_rank = bundle.number_of_sensors - 1
    if rank < 1 or rank > maximum_rank:
        raise ValueError(f"rank must lie in [1, {maximum_rank}]")
    if cortical_total_variance < 0 or not np.isfinite(cortical_total_variance):
        raise ValueError("cortical_total_variance must be non-negative and finite")
    if variance_normalization not in {"fixed_total", "fixed_mode"}:
        raise ValueError(
            "variance_normalization must be 'fixed_total' or 'fixed_mode'"
        )

    full_referenced = reference_leadfield(bundle.cortical_leadfield)
    weighted = density_operator(
        full_referenced, bundle.cortical_area_weights_m2, normalization
    )
    left, singular_values, _ = np.linalg.svd(weighted, full_matrices=False)
    tolerance = (
        max(weighted.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    numerical_rank = int(np.sum(singular_values > tolerance))
    actual_rank = min(rank, numerical_rank)
    if actual_rank < 1:
        raise ValueError("cortical operator has zero referenced rank")
    referenced_basis = left[:, :actual_rank]

    # Lift zero-mean Helmert coordinates back to raw sensor coordinates.  The
    # same joint nuisance model can then be restricted to every physical
    # montage, which preserves the data-processing relationship.
    from theory.recoverability import helmert_reference

    contrasts = helmert_reference(bundle.number_of_sensors)
    raw_basis = contrasts.T @ referenced_basis
    if cortical_total_variance == 0:
        cortical_covariance = np.zeros(
            (bundle.number_of_sensors, bundle.number_of_sensors), dtype=float
        )
    else:
        variance_per_mode = cortical_total_variance
        if variance_normalization == "fixed_total":
            variance_per_mode /= actual_rank
        cortical_covariance = variance_per_mode * (raw_basis @ raw_basis.T)
    raw_nuisance = bundle.sensor_noise_covariance + cortical_covariance
    raw_nuisance = 0.5 * (raw_nuisance + raw_nuisance.T)
    total_spectral_energy = float(np.sum(singular_values**2))
    retained = (
        float(np.sum(singular_values[:actual_rank] ** 2)) / total_spectral_energy
        if total_spectral_energy > 0
        else 0.0
    )
    return CorticalRestriction(
        raw_basis=raw_basis,
        raw_nuisance_covariance=raw_nuisance,
        requested_rank=rank,
        actual_rank=actual_rank,
        retained_spectral_energy_fraction=retained,
        variance_normalization=variance_normalization,
    )


def analyze_condition(
    bundle: AnatomicalBundle,
    source_covariance: ArrayLike,
    normalization: str,
    restriction: CorticalRestriction,
    sensor_indices: ArrayLike | None = None,
) -> tuple[dict[str, float], FloatArray]:
    """Evaluate one source model, cortical restriction, and physical montage."""

    indices = validate_sensor_indices(sensor_indices, bundle.number_of_sensors)
    hippocampal = reference_leadfield(bundle.hippocampal_leadfield, indices)
    forward = density_operator(
        hippocampal, bundle.hippocampal_area_weights_m2, normalization
    )
    source_covariance = np.asarray(source_covariance, dtype=np.float64)
    signal = propagated_covariance(forward, source_covariance)
    nuisance = reference_covariance(restriction.raw_nuisance_covariance, indices)
    cortical = reference_leadfield(restriction.raw_basis, indices)

    eigenvalues = whitened_signal_eigenvalues(
        forward, source_covariance, nuisance
    )
    mi_bits = gaussian_mi_bits(forward, source_covariance, nuisance)
    direct_sign_0, direct_logdet_0 = np.linalg.slogdet(nuisance)
    direct_sign_1, direct_logdet_1 = np.linalg.slogdet(nuisance + signal)
    if direct_sign_0 <= 0 or direct_sign_1 <= 0:
        raise FloatingPointError("non-positive covariance determinant")
    direct_mi = 0.5 * (direct_logdet_1 - direct_logdet_0) / log(2.0)

    target = _leading_topography(signal)
    precision = np.linalg.solve(nuisance, np.eye(nuisance.shape[0]))
    _, residual_energy = weighted_projection_residual(target, cortical, precision)
    target_energy = float(target @ precision @ target)

    cholesky = np.linalg.cholesky(nuisance)
    whitened_forward = np.linalg.solve(cholesky, forward)
    metrics = {
        "signal_power": float(np.trace(signal)),
        "whitened_signal_power": float(np.sum(eigenvalues)),
        "mutual_information_bits": mi_bits,
        "direct_mutual_information_bits": direct_mi,
        "mi_identity_absolute_error_bits": abs(mi_bits - direct_mi),
        "presence_kl_nats": kl_present_vs_absent_nats(eigenvalues),
        "dominant_mode_dprime": sqrt(float(eigenvalues[0]))
        if eigenvalues.size
        else 0.0,
        "largest_recoverability_eigenvalue": float(eigenvalues[0])
        if eigenvalues.size
        else 0.0,
        "effective_recoverability_rank": float(np.sum(eigenvalues > 1e-10)),
        "masking_residual_energy": residual_energy,
        "masking_residual_fraction": residual_energy / target_energy
        if target_energy > 0
        else 0.0,
        "geometry_efficiency": geometry_efficiency(whitened_forward),
        "cortical_actual_rank": float(restriction.actual_rank),
        "cortical_retained_spectral_energy_fraction": (
            restriction.retained_spectral_energy_fraction
        ),
    }
    return metrics, eigenvalues
