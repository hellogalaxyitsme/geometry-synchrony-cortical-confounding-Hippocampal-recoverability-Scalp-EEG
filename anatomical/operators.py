"""Continuous-density quadrature and sensor-space operators."""

from __future__ import annotations

from math import pi

import numpy as np
from numpy.typing import ArrayLike, NDArray

from theory.recoverability import helmert_reference


FloatArray = NDArray[np.float64]


def _finite_matrix(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def validate_sensor_indices(
    sensor_indices: ArrayLike | None, number_of_sensors: int
) -> NDArray[np.int64]:
    if sensor_indices is None:
        return np.arange(number_of_sensors, dtype=np.int64)
    indices = np.asarray(sensor_indices, dtype=np.int64).reshape(-1)
    if indices.size < 2:
        raise ValueError("a montage must contain at least two sensors")
    if len(set(int(value) for value in indices)) != indices.size:
        raise ValueError("sensor indices must be unique")
    if np.any(indices < 0) or np.any(indices >= number_of_sensors):
        raise ValueError("sensor index is out of range")
    return indices


def reference_leadfield(
    leadfield: ArrayLike, sensor_indices: ArrayLike | None = None
) -> FloatArray:
    leadfield = _finite_matrix(leadfield, "leadfield")
    indices = validate_sensor_indices(sensor_indices, leadfield.shape[0])
    contrasts = helmert_reference(indices.size)
    return contrasts @ leadfield[indices, :]


def reference_covariance(
    covariance: ArrayLike, sensor_indices: ArrayLike | None = None
) -> FloatArray:
    covariance = _finite_matrix(covariance, "covariance")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    indices = validate_sensor_indices(sensor_indices, covariance.shape[0])
    selected = covariance[np.ix_(indices, indices)]
    contrasts = helmert_reference(indices.size)
    result = contrasts @ selected @ contrasts.T
    return 0.5 * (result + result.T)


def quadrature_weights(
    area_weights_m2: ArrayLike, normalization: str
) -> FloatArray:
    weights = np.asarray(area_weights_m2, dtype=np.float64).reshape(-1)
    if weights.size == 0 or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("area weights must be finite and strictly positive")
    if normalization == "physical_density":
        return weights
    if normalization == "area_average":
        return weights / float(np.sum(weights))
    raise ValueError("normalization must be 'physical_density' or 'area_average'")


def density_operator(
    referenced_leadfield: ArrayLike,
    area_weights_m2: ArrayLike,
    normalization: str,
) -> FloatArray:
    leadfield = _finite_matrix(referenced_leadfield, "referenced_leadfield")
    weights = quadrature_weights(area_weights_m2, normalization)
    if leadfield.shape[1] != weights.size:
        raise ValueError("leadfield and area weights have incompatible dimensions")
    return leadfield * weights[None, :]


def density_covariance(
    positions_m: ArrayLike,
    longitudinal_coordinate: ArrayLike,
    coherence_length_m: float,
    synchrony_fraction: float,
    phase_cycles: float,
    source_density_scale: float = 1.0,
) -> FloatArray:
    """PSD fixed-marginal covariance sampled from continuous spatial kernels."""

    positions = _finite_matrix(positions_m, "positions_m")
    if positions.shape[1] != 3:
        raise ValueError("positions_m must have three columns")
    coordinate = np.asarray(longitudinal_coordinate, dtype=np.float64).reshape(-1)
    if coordinate.size != positions.shape[0] or not np.all(np.isfinite(coordinate)):
        raise ValueError("longitudinal_coordinate has incompatible dimensions")
    if coherence_length_m <= 0 or not np.isfinite(coherence_length_m):
        raise ValueError("coherence_length_m must be positive and finite")
    if not 0.0 <= synchrony_fraction <= 1.0:
        raise ValueError("synchrony_fraction must be in [0, 1]")
    if source_density_scale < 0 or not np.isfinite(source_density_scale):
        raise ValueError("source_density_scale must be non-negative and finite")

    differences = positions[:, None, :] - positions[None, :, :]
    squared_distance = np.sum(differences * differences, axis=2)
    local = np.exp(-0.5 * squared_distance / (coherence_length_m**2))
    phases = 2.0 * pi * phase_cycles * coordinate
    wave = np.cos(phases[:, None] - phases[None, :])
    covariance = (source_density_scale**2) * (
        (1.0 - synchrony_fraction) * local + synchrony_fraction * wave
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise FloatingPointError("density covariance is not positive semidefinite")
    return covariance


def propagated_covariance(
    density_forward_operator: ArrayLike, source_covariance: ArrayLike
) -> FloatArray:
    forward = _finite_matrix(density_forward_operator, "density_forward_operator")
    covariance = _finite_matrix(source_covariance, "source_covariance")
    if covariance.shape != (forward.shape[1], forward.shape[1]):
        raise ValueError("source covariance has incompatible dimensions")
    result = forward @ covariance @ forward.T
    return 0.5 * (result + result.T)


def farthest_point_order(positions_m: ArrayLike) -> NDArray[np.int64]:
    """Deterministic spatially dispersed ordering for mock nested montages."""

    positions = _finite_matrix(positions_m, "positions_m")
    if positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("positions_m must have shape (N, 3), N >= 2")
    start = int(np.argmax(positions[:, 2]))
    selected = [start]
    remaining = set(range(positions.shape[0]))
    remaining.remove(start)
    minimum_squared_distance = np.sum(
        (positions - positions[start]) ** 2, axis=1
    )
    while remaining:
        next_index = max(
            remaining, key=lambda index: (minimum_squared_distance[index], -index)
        )
        selected.append(next_index)
        remaining.remove(next_index)
        new_distance = np.sum((positions - positions[next_index]) ** 2, axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, new_distance)
    return np.asarray(selected, dtype=np.int64)
