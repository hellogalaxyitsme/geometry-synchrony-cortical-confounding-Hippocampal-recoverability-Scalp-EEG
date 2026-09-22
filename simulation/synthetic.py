"""Structured, forward-model-agnostic synthetic generators.

The constructions in this module are intentionally interpretable.  They do not
attempt to imitate an anatomical lead field; instead, they isolate curvature,
active extent, phase structure, cortical subspace overlap, and forward error
while keeping otherwise confounding quantities fixed.
"""

from __future__ import annotations

import hashlib
import json
from math import log, pi, sqrt
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from theory.recoverability import geometry_efficiency


FloatArray = NDArray[np.float64]


def stable_seed(base_seed: int, payload: Mapping[str, object] | str) -> int:
    """Derive a platform-independent uint32 seed from parameters."""

    rendered = payload if isinstance(payload, str) else json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(f"{base_seed}|{rendered}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _fix_qr_signs(matrix: FloatArray) -> FloatArray:
    result = np.array(matrix, copy=True)
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def build_sensor_modes(
    contrast_dimension: int, seed: int, number_of_modes: int = 3
) -> FloatArray:
    """Return deterministic orthonormal sensor-space modes."""

    if contrast_dimension < number_of_modes:
        raise ValueError("contrast dimension is smaller than the requested mode count")
    rng = np.random.default_rng(seed)
    candidate = rng.normal(size=(contrast_dimension, number_of_modes))
    modes, _ = np.linalg.qr(candidate, mode="reduced")
    return _fix_qr_signs(modes)


def full_sheet_positions(number_of_elements: int) -> FloatArray:
    if number_of_elements < 2:
        raise ValueError("at least two source elements are required")
    return (np.arange(number_of_elements, dtype=float) + 0.5) / number_of_elements - 0.5


def active_indices(number_of_elements: int, active_fraction: float) -> NDArray[np.int64]:
    if not 0.0 < active_fraction <= 1.0:
        raise ValueError("active_fraction must be in (0, 1]")
    count = max(1, int(round(number_of_elements * active_fraction)))
    start = (number_of_elements - count) // 2
    return np.arange(start, start + count, dtype=np.int64)


def structured_contributions(
    sensor_modes: ArrayLike,
    number_of_elements: int,
    active_fraction: float,
    orientation_span_radians: float,
    source_gain: float,
    depth_gradient: float,
    secondary_geometry_weight: float,
    normalization: str,
) -> tuple[FloatArray, FloatArray]:
    """Construct elementary sensor contributions and active coordinates.

    ``fixed_total`` makes the Frobenius norm of the returned matrix exactly
    ``source_gain``.  ``fixed_element`` keeps the full-sheet element scale fixed,
    allowing active extent to change total elementary power.
    """

    modes = np.asarray(sensor_modes, dtype=float)
    if modes.ndim != 2 or modes.shape[1] < 3:
        raise ValueError("sensor_modes must contain at least three columns")
    if source_gain < 0:
        raise ValueError("source_gain must be non-negative")
    if not 0.0 <= depth_gradient < 1.0:
        raise ValueError("depth_gradient must be in [0, 1)")
    if secondary_geometry_weight < 0:
        raise ValueError("secondary_geometry_weight must be non-negative")
    if normalization not in {"fixed_total", "fixed_element"}:
        raise ValueError("unknown source normalization")

    positions = full_sheet_positions(number_of_elements)
    indices = active_indices(number_of_elements, active_fraction)
    active_positions = positions[indices]
    angles = orientation_span_radians * active_positions
    directions = (
        modes[:, [0]] * np.cos(angles)[None, :]
        + modes[:, [1]] * np.sin(angles)[None, :]
        + secondary_geometry_weight
        * modes[:, [2]]
        * np.sin(2.0 * angles)[None, :]
    )
    norms = np.linalg.norm(directions, axis=0)
    if np.any(norms == 0):
        raise FloatingPointError("a synthetic source direction has zero norm")
    directions /= norms[None, :]

    attenuation = 1.0 - depth_gradient * (active_positions + 0.5)
    if normalization == "fixed_total":
        denominator = float(np.linalg.norm(attenuation))
        scale = 0.0 if denominator == 0.0 else source_gain / denominator
    else:
        scale = source_gain / sqrt(number_of_elements)
    contributions = directions * (scale * attenuation)[None, :]
    return contributions, active_positions


def source_covariance(
    positions: ArrayLike,
    synchrony_fraction: float,
    phase_cycles_full_extent: float,
) -> FloatArray:
    """Fixed-marginal source covariance with a coherent traveling component."""

    positions = np.asarray(positions, dtype=float).reshape(-1)
    if positions.size == 0:
        raise ValueError("positions cannot be empty")
    if not 0.0 <= synchrony_fraction <= 1.0:
        raise ValueError("synchrony_fraction must be in [0, 1]")
    phases = 2.0 * pi * phase_cycles_full_extent * positions
    wave = np.cos(phases[:, None] - phases[None, :])
    covariance = (
        (1.0 - synchrony_fraction) * np.eye(positions.size)
        + synchrony_fraction * wave
    )
    return 0.5 * (covariance + covariance.T)


def signal_covariance(contributions: ArrayLike, covariance: ArrayLike) -> FloatArray:
    contributions = np.asarray(contributions, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if contributions.ndim != 2:
        raise ValueError("contributions must be a matrix")
    if covariance.shape != (contributions.shape[1], contributions.shape[1]):
        raise ValueError("source covariance has incompatible dimensions")
    result = contributions @ covariance @ contributions.T
    return 0.5 * (result + result.T)


def leading_topography(covariance: ArrayLike) -> FloatArray:
    covariance = np.asarray(covariance, dtype=float)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    topography = vectors[:, int(np.argmax(values))]
    pivot = int(np.argmax(np.abs(topography)))
    if topography[pivot] < 0:
        topography *= -1.0
    return topography


def build_cortical_subspace(
    target_topography: ArrayLike,
    rank: int,
    overlap_fraction: float,
    seed: int,
) -> FloatArray:
    """Build an orthonormal cortical basis with controlled target overlap.

    For ``rank < m``, the squared projection of the normalized target onto the
    returned subspace equals ``overlap_fraction`` up to floating-point error.
    For ``rank >= m``, a full basis is returned and masking is necessarily exact.
    """

    target = np.asarray(target_topography, dtype=float).reshape(-1)
    dimension = target.size
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        raise ValueError("target_topography cannot be zero")
    if rank < 1:
        raise ValueError("cortical rank must be positive")
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1]")
    target = target / target_norm

    rng = np.random.default_rng(seed)
    complement_columns = dimension - 1 if rank >= dimension else rank
    candidate = rng.normal(size=(dimension, complement_columns))
    candidate -= target[:, None] * (target @ candidate)[None, :]
    complement, _ = np.linalg.qr(candidate, mode="reduced")
    complement = _fix_qr_signs(complement)

    if rank >= dimension:
        return np.column_stack((target, complement))

    first = (
        sqrt(overlap_fraction) * target
        + sqrt(1.0 - overlap_fraction) * complement[:, 0]
    )
    if rank == 1:
        return first[:, None]
    return np.column_stack((first, complement[:, 1:rank]))


def nuisance_covariance(
    cortical_subspace: ArrayLike,
    sensor_noise_variance: float,
    cortical_variance: float,
) -> FloatArray:
    cortical = np.asarray(cortical_subspace, dtype=float)
    if cortical.ndim != 2:
        raise ValueError("cortical_subspace must be a matrix")
    if sensor_noise_variance <= 0 or cortical_variance < 0:
        raise ValueError("invalid nuisance variances")
    result = (
        sensor_noise_variance * np.eye(cortical.shape[0])
        + cortical_variance * cortical @ cortical.T
    )
    return 0.5 * (result + result.T)


def _whiten_covariance(
    signal: FloatArray, cholesky: FloatArray
) -> FloatArray:
    left = np.linalg.solve(cholesky, signal)
    whitened = np.linalg.solve(cholesky, left.T).T
    return 0.5 * (whitened + whitened.T)


def compute_recoverability_metrics(
    contributions: ArrayLike,
    signal: ArrayLike,
    nuisance: ArrayLike,
    cortical_subspace: ArrayLike,
) -> dict[str, float]:
    """Compute the registered scalar outputs for one condition."""

    contributions = np.asarray(contributions, dtype=float)
    signal = np.asarray(signal, dtype=float)
    nuisance = np.asarray(nuisance, dtype=float)
    cortical = np.asarray(cortical_subspace, dtype=float)
    dimension = signal.shape[0]
    if signal.shape != (dimension, dimension):
        raise ValueError("signal covariance must be square")
    if nuisance.shape != signal.shape:
        raise ValueError("nuisance covariance has incompatible dimensions")
    if contributions.shape[0] != dimension or cortical.shape[0] != dimension:
        raise ValueError("sensor dimensions do not agree")

    cholesky = np.linalg.cholesky(0.5 * (nuisance + nuisance.T))
    whitened = _whiten_covariance(signal, cholesky)
    eigenvalues = np.linalg.eigvalsh(whitened)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1e-9 * scale:
        raise FloatingPointError("whitened signal covariance is not PSD")
    eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]
    mi_bits = 0.5 * float(np.sum(np.log1p(eigenvalues))) / log(2.0)
    kl_nats = 0.5 * float(np.sum(eigenvalues - np.log1p(eigenvalues)))

    target = leading_topography(signal)
    whitened_target = np.linalg.solve(cholesky, target)
    whitened_cortical = np.linalg.solve(cholesky, cortical)
    coefficients = np.linalg.lstsq(
        whitened_cortical, whitened_target, rcond=None
    )[0]
    masking_residual = whitened_target - whitened_cortical @ coefficients
    masking_energy = float(masking_residual @ masking_residual)
    target_energy = float(whitened_target @ whitened_target)
    masking_fraction = masking_energy / target_energy if target_energy > 0 else 0.0

    whitened_contributions = np.linalg.solve(cholesky, contributions)
    return {
        "signal_power": float(np.trace(signal)),
        "whitened_signal_power": float(np.trace(whitened)),
        "mutual_information_bits": mi_bits,
        "presence_kl_nats": kl_nats,
        "dominant_mode_dprime": sqrt(float(eigenvalues[0])) if eigenvalues.size else 0.0,
        "largest_recoverability_eigenvalue": float(eigenvalues[0]) if eigenvalues.size else 0.0,
        "effective_recoverability_rank": float(np.sum(eigenvalues > 1e-10)),
        "masking_residual_energy": masking_energy,
        "masking_residual_fraction": masking_fraction,
        "geometry_efficiency": geometry_efficiency(whitened_contributions),
    }


def perturb_contributions(
    contributions: ArrayLike, relative_spectral_error: float, seed: int
) -> tuple[FloatArray, float]:
    """Apply a norm-preserving random topographic forward perturbation.

    Every elementary contribution keeps its Euclidean norm, so the perturbation
    cannot manufacture source power. A bisection search sets the achieved
    matrix 2-norm error to the requested fraction of the nominal 2-norm.
    """

    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim != 2:
        raise ValueError("contributions must be a matrix")
    if relative_spectral_error < 0:
        raise ValueError("relative_spectral_error must be non-negative")
    if relative_spectral_error == 0:
        return np.array(contributions, copy=True), 0.0
    nominal_norm = float(np.linalg.norm(contributions, ord=2))
    if nominal_norm == 0:
        return np.array(contributions, copy=True), 0.0
    column_norms = np.linalg.norm(contributions, axis=0)
    if np.any(column_norms == 0):
        raise ValueError("cannot norm-preserve a zero elementary contribution")
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=contributions.shape)
    # Make the direction tangent to each column's constant-norm sphere.  This
    # makes the norm-preserving retraction well behaved for the small errors in
    # the benchmark and avoids a needlessly wide bisection bracket.
    radial_coefficients = np.sum(contributions * direction, axis=0) / (
        column_norms**2
    )
    direction -= contributions * radial_coefficients[None, :]
    direction_norm = float(np.linalg.norm(direction, ord=2))
    if direction_norm == 0:
        raise FloatingPointError("random perturbation direction vanished")

    def candidate(step: float) -> tuple[FloatArray, float]:
        perturbed = contributions + step * direction
        perturbed_norms = np.linalg.norm(perturbed, axis=0)
        if np.any(perturbed_norms == 0):
            raise FloatingPointError("forward perturbation produced a zero column")
        perturbed *= (column_norms / perturbed_norms)[None, :]
        achieved_error = (
            float(np.linalg.norm(perturbed - contributions, ord=2)) / nominal_norm
        )
        return perturbed, achieved_error

    lower = 0.0
    upper = relative_spectral_error * nominal_norm / direction_norm
    upper_candidate, upper_error = candidate(upper)
    for _ in range(20):
        if upper_error >= relative_spectral_error:
            break
        upper *= 2.0
        upper_candidate, upper_error = candidate(upper)
    else:
        raise FloatingPointError("could not attain the requested forward error")

    result = upper_candidate
    achieved = upper_error
    # Twenty-four iterations locate the requested relative error to much better
    # than the precision warranted by the simulation parameters while keeping
    # the uncertainty sweep inexpensive.
    for _ in range(24):
        midpoint = 0.5 * (lower + upper)
        midpoint_candidate, midpoint_error = candidate(midpoint)
        result, achieved = midpoint_candidate, midpoint_error
        if midpoint_error < relative_spectral_error:
            lower = midpoint
        else:
            upper = midpoint
    return result, achieved
