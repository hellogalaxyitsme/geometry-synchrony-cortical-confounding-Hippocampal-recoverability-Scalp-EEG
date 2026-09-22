"""Geometry and scoring primitives for the stimulation-control cortical control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from simulation.cortical_restriction import orthonormal_basis, somp


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Interpolation:
    indices: IntArray
    weights: FloatArray
    nearest_angle_degrees: FloatArray


def fiducial_head_coordinates(
    coordinates_mm: ArrayLike, fiducials_mm: Mapping[str, ArrayLike]
) -> FloatArray:
    """Transform native MRI coordinates to MNE-style head coordinates in metres."""
    points = np.asarray(coordinates_mm, dtype=np.float64)
    lpa = np.asarray(fiducials_mm["LPA"], dtype=np.float64)
    nas = np.asarray(fiducials_mm["NAS"], dtype=np.float64)
    rpa = np.asarray(fiducials_mm["RPA"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or any(value.shape != (3,) for value in (lpa, nas, rpa)):
        raise ValueError("coordinates and fiducials must be three-dimensional")
    origin = 0.5 * (lpa + rpa)
    x_axis = rpa - lpa
    x_axis /= np.linalg.norm(x_axis)
    y_axis = nas - origin
    y_axis -= x_axis * float(x_axis @ y_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    transformed = (points - origin) @ rotation * 1e-3
    if not np.all(np.isfinite(transformed)):
        raise ValueError("fiducial transformation produced non-finite coordinates")
    return np.asarray(transformed, dtype=np.float64)


def fit_sphere(points: ArrayLike) -> tuple[FloatArray, float]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 4:
        raise ValueError("sphere fitting requires at least four 3-D points")
    design = np.column_stack((2.0 * values, np.ones(len(values))))
    target = np.sum(values * values, axis=1)
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    center = solution[:3]
    radius_squared = float(solution[3] + center @ center)
    if radius_squared <= 0.0:
        raise ValueError("invalid fitted sphere")
    return np.asarray(center, dtype=np.float64), float(np.sqrt(radius_squared))


def spherical_directions(points: ArrayLike) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    center, _ = fit_sphere(values)
    centered = values - center
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("a sensor lies at the fitted sphere centre")
    return np.asarray(centered / norms[:, None], dtype=np.float64)


def spherical_interpolation(
    source_points: ArrayLike,
    target_points: ArrayLike,
    neighbours: int = 8,
    sigma_degrees: float = 12.0,
) -> Interpolation:
    source = spherical_directions(source_points)
    target = spherical_directions(target_points)
    if neighbours < 1 or neighbours > len(source) or sigma_degrees <= 0.0:
        raise ValueError("invalid spherical interpolation parameters")
    chord, indices = cKDTree(source).query(target, k=neighbours, workers=1)
    if neighbours == 1:
        chord = chord[:, None]
        indices = indices[:, None]
    angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    sigma = np.deg2rad(sigma_degrees)
    weights = np.exp(-0.5 * (angle / sigma) ** 2)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return Interpolation(
        np.asarray(indices, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.rad2deg(angle[:, 0]),
    )


def interpolate_rows(matrix: ArrayLike, interpolation: Interpolation) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or np.max(interpolation.indices) >= values.shape[0]:
        raise ValueError("interpolation and sensor matrix are incompatible")
    selected = values[interpolation.indices]
    return np.asarray(np.sum(selected * interpolation.weights[:, :, None], axis=1), dtype=np.float64)


def leave_one_out_interpolation(
    points: ArrayLike, matrix: ArrayLike, neighbours: int, sigma_degrees: float
) -> dict[str, float]:
    coords = np.asarray(points, dtype=np.float64)
    values = np.asarray(matrix, dtype=np.float64)
    directions = spherical_directions(coords)
    if values.shape[0] != len(coords) or neighbours >= len(coords):
        raise ValueError("invalid leave-one-out interpolation inputs")
    tree = cKDTree(directions)
    chord, indices = tree.query(directions, k=neighbours + 1, workers=1)
    chord = chord[:, 1:]
    indices = indices[:, 1:]
    angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    weights = np.exp(-0.5 * (angle / np.deg2rad(sigma_degrees)) ** 2)
    weights /= np.sum(weights, axis=1, keepdims=True)
    predicted = np.sum(values[indices] * weights[:, :, None], axis=1)
    residual = predicted - values
    denominator = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    reference = values - values.mean(axis=0, keepdims=True)
    estimate = predicted - predicted.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(reference, axis=0) * np.linalg.norm(estimate, axis=0)
    cosine = np.divide(
        np.sum(reference * estimate, axis=0),
        norms,
        out=np.zeros(reference.shape[1], dtype=np.float64),
        where=norms > 0.0,
    )
    return {
        "relative_frobenius_error": float(np.linalg.norm(residual) / denominator),
        "median_referenced_topography_cosine": float(np.median(cosine)),
        "p05_referenced_topography_cosine": float(np.quantile(cosine, 0.05)),
        "median_nearest_angle_degrees": float(np.median(np.rad2deg(angle[:, 0]))),
        "maximum_nearest_angle_degrees": float(np.max(np.rad2deg(angle[:, 0]))),
    }


def robust_channel_scales(baseline_epochs: ArrayLike) -> FloatArray:
    values = np.asarray(baseline_epochs, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("baseline epochs must be trial-by-channel-by-time")
    flattened = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    centre = np.median(flattened, axis=1, keepdims=True)
    scales = 1.4826 * np.median(np.abs(flattened - centre), axis=1)
    positive = scales[np.isfinite(scales) & (scales > 0.0)]
    if positive.size == 0:
        raise ValueError("all robust channel scales are zero")
    floor = max(float(np.median(positive)) * 1e-6, np.finfo(float).tiny)
    if np.any(~np.isfinite(scales)) or np.any(scales <= floor):
        raise ValueError("a robust channel scale is invalid or negligible")
    return np.asarray(scales, dtype=np.float64)


def average_reference(values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim < 2:
        raise ValueError("average referencing requires a sensor dimension")
    sensor_axis = matrix.ndim - 2
    return np.asarray(matrix - np.mean(matrix, axis=sensor_axis, keepdims=True), dtype=np.float64)


def transform_patient_operator(
    electrode_operator: ArrayLike, good_indices: ArrayLike, scales: ArrayLike
) -> FloatArray:
    values = np.asarray(electrode_operator, dtype=np.float64)
    indices = np.asarray(good_indices, dtype=np.int64)
    scale = np.asarray(scales, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or len(indices) != len(scale) or np.max(indices) >= values.shape[0]:
        raise ValueError("patient operator transform inputs are incompatible")
    transformed = values[indices] / scale[:, None]
    transformed -= transformed.mean(axis=0, keepdims=True)
    return np.asarray(transformed, dtype=np.float64)


def pca_targets(matrix: ArrayLike, components: int) -> tuple[FloatArray, float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or components < 1:
        raise ValueError("invalid PCA target request")
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    keep = min(components, len(singular))
    targets = left[:, :keep] * singular[:keep][None, :]
    total = float(np.sum(singular * singular))
    retained = float(np.sum(singular[:keep] ** 2) / total) if total > 0.0 else 0.0
    return np.asarray(targets, dtype=np.float64), retained


def peak_topography(matrix: ArrayLike) -> tuple[FloatArray, int]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("peak topography requires sensor-by-time data")
    index = int(np.argmax(np.sum(values * values, axis=0)))
    return np.asarray(values[:, index : index + 1], dtype=np.float64), index


def matrix_false_attribution(
    targets: ArrayLike, hippocampal_factor: ArrayLike, cortical_basis: ArrayLike
) -> float:
    observations = np.asarray(targets, dtype=np.float64)
    hippocampal = np.asarray(hippocampal_factor, dtype=np.float64)
    basis = np.asarray(cortical_basis, dtype=np.float64)
    if observations.ndim != 2 or hippocampal.ndim != 2 or basis.shape[0] != observations.shape[0]:
        raise ValueError("false-attribution dimensions disagree")
    residual_y = observations - basis @ (basis.T @ observations)
    residual_h = hippocampal - basis @ (basis.T @ hippocampal)
    h_basis = orthonormal_basis(residual_h)
    total = float(np.sum(observations * observations))
    explained = float(np.sum((h_basis.T @ residual_y) ** 2)) if h_basis.shape[1] else 0.0
    return float(np.clip(explained / total if total > 0.0 else 0.0, 0.0, 1.0))


def matrix_attribution_components(
    targets: ArrayLike, hippocampal_factor: ArrayLike, cortical_basis: ArrayLike
) -> dict[str, float]:
    """Return total, residual, and conditional anatomical attribution fractions."""
    observations = np.asarray(targets, dtype=np.float64)
    hippocampal = np.asarray(hippocampal_factor, dtype=np.float64)
    basis = np.asarray(cortical_basis, dtype=np.float64)
    if observations.ndim != 2 or hippocampal.ndim != 2 or basis.shape[0] != observations.shape[0]:
        raise ValueError("attribution dimensions disagree")
    residual_y = observations - basis @ (basis.T @ observations)
    residual_h = hippocampal - basis @ (basis.T @ hippocampal)
    h_basis = orthonormal_basis(residual_h)
    total = float(np.sum(observations * observations))
    residual = float(np.sum(residual_y * residual_y))
    explained = float(np.sum((h_basis.T @ residual_y) ** 2)) if h_basis.shape[1] else 0.0
    tiny = np.finfo(float).tiny
    return {
        "total_attribution_fraction": float(np.clip(explained / max(total, tiny), 0.0, 1.0)),
        "cortical_residual_energy_fraction": float(np.clip(residual / max(total, tiny), 0.0, 1.0)),
        "conditional_hippocampal_fraction": float(np.clip(explained / max(residual, tiny), 0.0, 1.0)),
    }


def sparse_basis(dictionary: ArrayLike, targets: ArrayLike, atoms: int) -> tuple[FloatArray, IntArray]:
    design = np.asarray(dictionary, dtype=np.float64)
    observations = np.asarray(targets, dtype=np.float64)
    selected, _ = somp(design, observations, atoms)
    return orthonormal_basis(design[:, selected]), selected


def normalized_dictionary(matrix: ArrayLike, minimum_norm: float = 1e-12) -> tuple[FloatArray, IntArray]:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=0)
    keep = np.flatnonzero(norms > minimum_norm * max(float(np.max(norms)), np.finfo(float).tiny))
    if len(keep) == 0:
        raise ValueError("mapped cortical dictionary has no usable atoms")
    return np.asarray(values[:, keep] / norms[keep][None, :], dtype=np.float64), keep.astype(np.int64)
