"""Cortical-restriction and false-attribution primitives for cortical-restriction."""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from theory.recoverability import helmert_reference


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def referenced_operator(matrix: ArrayLike, sensor_indices: ArrayLike | None = None) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("operator must be a finite sensor-by-source matrix")
    if sensor_indices is not None:
        indices = np.asarray(sensor_indices, dtype=np.int64).reshape(-1)
        if len(indices) < 2 or len(np.unique(indices)) != len(indices):
            raise ValueError("sensor indices must be unique and contain at least two sensors")
        values = values[indices]
    return np.asarray(helmert_reference(values.shape[0]) @ values, dtype=np.float64)


def orthonormal_basis(matrix: ArrayLike, relative_tolerance: float = 1e-12) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("basis input must be a finite matrix")
    if values.shape[1] == 0:
        return np.zeros((values.shape[0], 0), dtype=np.float64)
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.zeros((values.shape[0], 0), dtype=np.float64)
    keep = singular > relative_tolerance * singular[0]
    return np.asarray(left[:, keep], dtype=np.float64)


def covariance_basis(
    covariance: ArrayLike, retained_fraction: float
) -> tuple[FloatArray, FloatArray, float]:
    matrix = np.asarray(covariance, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not 0.0 < retained_fraction <= 1.0
    ):
        raise ValueError("invalid covariance basis inputs")
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("covariance has zero trace")
    stop = int(np.searchsorted(np.cumsum(values), retained_fraction * total, side="left")) + 1
    stop = min(stop, len(values))
    basis = vectors[:, :stop]
    realized = float(np.sum(values[:stop]) / total)
    return np.asarray(basis, dtype=np.float64), values, realized


def area_weighted_covariance(
    leadfield: ArrayLike, area_weights: ArrayLike
) -> FloatArray:
    operator = np.asarray(leadfield, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64).reshape(-1)
    if operator.ndim != 2 or operator.shape[1] != len(weights) or np.any(weights <= 0.0):
        raise ValueError("lead field and positive area weights are required")
    weights = weights / float(np.sum(weights))
    weighted = operator * np.sqrt(weights)[None, :]
    covariance = weighted @ weighted.T
    return np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64)


def farthest_anchors(
    positions_m: ArrayLike,
    hemisphere_code: ArrayLike,
    area_weights: ArrayLike,
    anchors_per_hemisphere: int,
) -> IntArray:
    positions = np.asarray(positions_m, dtype=np.float64)
    hemispheres = np.asarray(hemisphere_code, dtype=np.int8).reshape(-1)
    areas = np.asarray(area_weights, dtype=np.float64).reshape(-1)
    if positions.shape != (len(hemispheres), 3) or len(areas) != len(hemispheres):
        raise ValueError("cortical geometry arrays are misaligned")
    if anchors_per_hemisphere < 1 or set(np.unique(hemispheres)) != {-1, 1}:
        raise ValueError("invalid hemisphere anchor request")
    result: list[int] = []
    for code in (-1, 1):
        candidates = np.flatnonzero(hemispheres == code)
        first = int(candidates[np.argmax(areas[candidates])])
        selected = [first]
        minimum_squared = np.sum((positions[candidates] - positions[first]) ** 2, axis=1)
        for _ in range(1, anchors_per_hemisphere):
            local = int(np.argmax(minimum_squared))
            chosen = int(candidates[local])
            selected.append(chosen)
            distance = np.sum((positions[candidates] - positions[chosen]) ** 2, axis=1)
            minimum_squared = np.minimum(minimum_squared, distance)
            minimum_squared[np.isin(candidates, selected)] = -1.0
        result.extend(selected)
    return np.asarray(result, dtype=np.int64)


def smooth_topographies(
    leadfield: ArrayLike,
    positions_m: ArrayLike,
    hemisphere_code: ArrayLike,
    area_weights: ArrayLike,
    anchor_indices: ArrayLike,
    widths_m: float | Sequence[float],
) -> FloatArray:
    operator = np.asarray(leadfield, dtype=np.float64)
    positions = np.asarray(positions_m, dtype=np.float64)
    hemispheres = np.asarray(hemisphere_code, dtype=np.int8).reshape(-1)
    areas = np.asarray(area_weights, dtype=np.float64).reshape(-1)
    anchors = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    widths = np.asarray(widths_m, dtype=np.float64).reshape(-1)
    if widths.size == 1:
        widths = np.repeat(widths, len(anchors))
    if (
        operator.shape[1] != len(positions)
        or positions.shape != (len(areas), 3)
        or len(hemispheres) != len(areas)
        or len(widths) != len(anchors)
        or np.any(widths <= 0.0)
    ):
        raise ValueError("smooth cortical inputs are misaligned")
    patterns = np.zeros((len(areas), len(anchors)), dtype=np.float64)
    for column, (anchor, width) in enumerate(zip(anchors, widths)):
        same = hemispheres == hemispheres[anchor]
        distance_squared = np.sum((positions - positions[anchor]) ** 2, axis=1)
        weights = np.exp(-0.5 * distance_squared / float(width * width)) * areas * same
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("smooth cortical kernel has zero weight")
        patterns[:, column] = weights / total
    return np.asarray(operator @ patterns, dtype=np.float64)


def neighborhood_indices(
    cortical_positions_m: ArrayLike,
    active_hippocampal_positions_m: ArrayLike,
    radius_m: float,
) -> IntArray:
    cortical = np.asarray(cortical_positions_m, dtype=np.float64)
    hippocampal = np.asarray(active_hippocampal_positions_m, dtype=np.float64)
    if cortical.ndim != 2 or cortical.shape[1] != 3 or hippocampal.ndim != 2 or hippocampal.shape[1] != 3:
        raise ValueError("neighborhood positions must be point-by-three matrices")
    if len(hippocampal) == 0 or radius_m <= 0.0:
        raise ValueError("hippocampal support and positive radius are required")
    distances, _ = cKDTree(hippocampal).query(cortical, k=1, workers=1)
    return np.flatnonzero(distances <= radius_m).astype(np.int64)


def normalized_columns(matrix: ArrayLike, minimum_norm: float = 1e-14) -> tuple[FloatArray, IntArray]:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=0)
    keep = np.flatnonzero(norms > minimum_norm)
    return np.asarray(values[:, keep] / norms[keep][None, :], dtype=np.float64), keep.astype(np.int64)


def deterministic_dictionary_split(
    area_weights: ArrayLike,
    usable_indices: ArrayLike,
    dictionary_size: int,
    heldout_size: int,
    seed: int,
) -> tuple[IntArray, IntArray]:
    areas = np.asarray(area_weights, dtype=np.float64).reshape(-1)
    usable = np.asarray(usable_indices, dtype=np.int64).reshape(-1)
    if dictionary_size < 1 or heldout_size < 1 or dictionary_size + heldout_size > len(usable):
        raise ValueError("dictionary split is larger than the usable cortex")
    probabilities = areas[usable] / float(np.sum(areas[usable]))
    generator = np.random.default_rng(seed)
    selected = generator.choice(
        usable, size=dictionary_size + heldout_size, replace=False, p=probabilities
    )
    dictionary = np.sort(selected[:dictionary_size]).astype(np.int64)
    heldout = np.sort(selected[dictionary_size:]).astype(np.int64)
    if np.intersect1d(dictionary, heldout).size:
        raise AssertionError("dictionary and held-out cortex overlap")
    return dictionary, heldout


def somp(dictionary: ArrayLike, targets: ArrayLike, atoms: int) -> tuple[IntArray, FloatArray]:
    design = np.asarray(dictionary, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, None]
    if design.ndim != 2 or response.ndim != 2 or design.shape[0] != response.shape[0]:
        raise ValueError("SOMP design and response dimensions disagree")
    if atoms < 1 or atoms > design.shape[1]:
        raise ValueError("invalid SOMP atom count")
    selected: list[int] = []
    residual = response.copy()
    coefficients = np.zeros((0, response.shape[1]), dtype=np.float64)
    for _ in range(atoms):
        correlation = np.sum((design.T @ residual) ** 2, axis=1)
        if selected:
            correlation[np.asarray(selected, dtype=np.int64)] = -np.inf
        chosen = int(np.argmax(correlation))
        selected.append(chosen)
        active = design[:, selected]
        coefficients, _, _, _ = np.linalg.lstsq(active, response, rcond=None)
        residual = response - active @ coefficients
    return np.asarray(selected, dtype=np.int64), np.asarray(residual, dtype=np.float64)


def residual_sensitivity(factor: ArrayLike, cortical_basis: ArrayLike) -> dict[str, float]:
    hippocampal = np.asarray(factor, dtype=np.float64)
    basis = np.asarray(cortical_basis, dtype=np.float64)
    total = float(np.sum(hippocampal * hippocampal))
    if hippocampal.ndim != 2 or total <= 0.0 or basis.shape[0] != hippocampal.shape[0]:
        raise ValueError("invalid sensitivity inputs")
    residual = hippocampal - basis @ (basis.T @ hippocampal)
    residual_energy = float(np.sum(residual * residual))
    return {
        "hippocampal_energy": total,
        "residual_energy": residual_energy,
        "sensitivity_fraction": min(max(residual_energy / total, 0.0), 1.0),
        "cortical_captured_fraction": min(max(1.0 - residual_energy / total, 0.0), 1.0),
    }


def sparse_sensitivity(
    factor: ArrayLike, dictionary: ArrayLike, atoms: int
) -> dict[str, float]:
    hippocampal = np.asarray(factor, dtype=np.float64)
    selected, residual = somp(dictionary, hippocampal, atoms)
    total = float(np.sum(hippocampal * hippocampal))
    remaining = float(np.sum(residual * residual))
    return {
        "hippocampal_energy": total,
        "residual_energy": remaining,
        "sensitivity_fraction": min(max(remaining / total, 0.0), 1.0),
        "cortical_captured_fraction": min(max(1.0 - remaining / total, 0.0), 1.0),
        "selected_atoms": float(len(selected)),
    }


def false_attribution(
    probes: ArrayLike, factor: ArrayLike, cortical_basis: ArrayLike
) -> FloatArray:
    cortical = np.asarray(probes, dtype=np.float64)
    hippocampal = np.asarray(factor, dtype=np.float64)
    basis = np.asarray(cortical_basis, dtype=np.float64)
    if cortical.ndim != 2 or hippocampal.ndim != 2 or basis.shape[0] != cortical.shape[0]:
        raise ValueError("false-attribution inputs are incompatible")
    residual_probe = cortical - basis @ (basis.T @ cortical)
    residual_hippocampal = hippocampal - basis @ (basis.T @ hippocampal)
    h_basis = orthonormal_basis(residual_hippocampal)
    if h_basis.shape[1] == 0:
        return np.zeros(cortical.shape[1], dtype=np.float64)
    explained = np.sum((h_basis.T @ residual_probe) ** 2, axis=0)
    total = np.sum(cortical * cortical, axis=0)
    result = np.divide(explained, total, out=np.zeros_like(explained), where=total > 0.0)
    return np.clip(result, 0.0, 1.0)


def sparse_false_attribution(
    probes: ArrayLike, factor: ArrayLike, dictionary: ArrayLike, atoms: int
) -> FloatArray:
    cortical = np.asarray(probes, dtype=np.float64)
    hippocampal = np.asarray(factor, dtype=np.float64)
    design = np.asarray(dictionary, dtype=np.float64)
    values = np.empty(cortical.shape[1], dtype=np.float64)
    for column in range(cortical.shape[1]):
        selected, _ = somp(design, cortical[:, column], atoms)
        basis = orthonormal_basis(design[:, selected])
        values[column] = false_attribution(
            cortical[:, column : column + 1], hippocampal, basis
        )[0]
    return values


def sparse_probe_bases(
    probes: ArrayLike, dictionary: ArrayLike, atoms: int
) -> list[FloatArray]:
    """Fit the cortical sparse nuisance once per probe, independently of H."""
    cortical = np.asarray(probes, dtype=np.float64)
    design = np.asarray(dictionary, dtype=np.float64)
    result = []
    for column in range(cortical.shape[1]):
        selected, _ = somp(design, cortical[:, column], atoms)
        result.append(orthonormal_basis(design[:, selected]))
    return result


def sparse_false_from_bases(
    probes: ArrayLike, factor: ArrayLike, fitted_bases: Sequence[ArrayLike]
) -> FloatArray:
    cortical = np.asarray(probes, dtype=np.float64)
    if cortical.shape[1] != len(fitted_bases):
        raise ValueError("one fitted sparse basis is required per cortical probe")
    values = np.empty(cortical.shape[1], dtype=np.float64)
    for column, basis in enumerate(fitted_bases):
        values[column] = false_attribution(
            cortical[:, column : column + 1], factor, np.asarray(basis, dtype=np.float64)
        )[0]
    return values


def summarize_false_attribution(values: ArrayLike, threshold: float) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("false-attribution summary requires finite values")
    return {
        "probe_count": float(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
        "fraction_above_threshold": float(np.mean(array > threshold)),
        "threshold": float(threshold),
    }


def tensor_project(matrix: ArrayLike, spatial_basis: ArrayLike, temporal_basis: ArrayLike) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    spatial = np.asarray(spatial_basis, dtype=np.float64)
    temporal = np.asarray(temporal_basis, dtype=np.float64)
    if values.ndim != 2 or spatial.shape[0] != values.shape[0] or temporal.shape[0] != values.shape[1]:
        raise ValueError("tensor projection dimensions disagree")
    return np.asarray(spatial @ (spatial.T @ values @ temporal) @ temporal.T, dtype=np.float64)


def dynamic_hippocampal_factors(
    spatial_factor: ArrayLike, samples: int, sampling_frequency: float, carrier_hz: float
) -> list[FloatArray]:
    factor = np.asarray(spatial_factor, dtype=np.float64)
    if factor.ndim != 2 or factor.shape[1] not in (1, 2) or samples < 4:
        raise ValueError("dynamic hippocampal factor must have one or two spatial columns")
    time = np.arange(samples, dtype=np.float64) / float(sampling_frequency)
    cosine = np.cos(2.0 * np.pi * carrier_hz * time)
    sine = np.sin(2.0 * np.pi * carrier_hz * time)
    real = factor[:, 0]
    imaginary = factor[:, 1] if factor.shape[1] == 2 else np.zeros_like(real)
    first = np.outer(real, cosine) - np.outer(imaginary, sine)
    second = np.outer(real, sine) + np.outer(imaginary, cosine)
    return [np.asarray(first), np.asarray(second)]


def tensor_sensitivity(
    hippocampal_factors: Iterable[ArrayLike],
    spatial_basis: ArrayLike,
    temporal_basis: ArrayLike,
) -> dict[str, float]:
    factors = [np.asarray(value, dtype=np.float64) for value in hippocampal_factors]
    total = float(sum(np.sum(value * value) for value in factors))
    residual = [value - tensor_project(value, spatial_basis, temporal_basis) for value in factors]
    remaining = float(sum(np.sum(value * value) for value in residual))
    return {
        "hippocampal_energy": total,
        "residual_energy": remaining,
        "sensitivity_fraction": min(max(remaining / total, 0.0), 1.0),
        "cortical_captured_fraction": min(max(1.0 - remaining / total, 0.0), 1.0),
    }


def tensor_false_attribution(
    probes: Iterable[ArrayLike],
    hippocampal_factors: Iterable[ArrayLike],
    spatial_basis: ArrayLike,
    temporal_basis: ArrayLike,
) -> FloatArray:
    h_residual = []
    for value in hippocampal_factors:
        matrix = np.asarray(value, dtype=np.float64)
        h_residual.append((matrix - tensor_project(matrix, spatial_basis, temporal_basis)).reshape(-1))
    h_basis = orthonormal_basis(np.column_stack(h_residual))
    result = []
    for value in probes:
        matrix = np.asarray(value, dtype=np.float64)
        residual = matrix - tensor_project(matrix, spatial_basis, temporal_basis)
        vector = residual.reshape(-1)
        denominator = float(np.sum(matrix * matrix))
        explained = float(np.sum((h_basis.T @ vector) ** 2)) if h_basis.shape[1] else 0.0
        result.append(explained / denominator if denominator > 0.0 else 0.0)
    return np.clip(np.asarray(result, dtype=np.float64), 0.0, 1.0)


def basis_diagnostics(basis: ArrayLike) -> dict[str, float]:
    values = np.asarray(basis, dtype=np.float64)
    gram = values.T @ values
    identity = np.eye(values.shape[1])
    projector = values @ values.T
    return {
        "rank": float(values.shape[1]),
        "orthonormality_max_error": float(np.max(np.abs(gram - identity))) if gram.size else 0.0,
        "projector_idempotence_relative_error": float(
            np.linalg.norm(projector @ projector - projector)
            / max(np.linalg.norm(projector), np.finfo(float).tiny)
        ),
    }
