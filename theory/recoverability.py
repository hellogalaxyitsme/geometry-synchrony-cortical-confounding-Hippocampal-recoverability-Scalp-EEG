"""Reference implementations for the recoverability results.

The functions in this module deliberately operate on small dense matrices.  They
are intended as auditable mathematical reference implementations, not as the
eventual large-scale source-imaging pipeline.
"""

from __future__ import annotations

from math import erf, log, pi, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _as_float_matrix(value: ArrayLike, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _as_complex_matrix(value: ArrayLike, name: str) -> ComplexArray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _symmetrise(matrix: FloatArray) -> FloatArray:
    return 0.5 * (matrix + matrix.T)


def _assert_spd(matrix: FloatArray, name: str) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    try:
        np.linalg.cholesky(_symmetrise(matrix))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be symmetric positive definite") from exc


def _psd_sqrt(matrix: FloatArray, tolerance: float = 1e-12) -> FloatArray:
    matrix = _symmetrise(matrix)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < -tolerance * scale:
        raise ValueError("matrix must be positive semidefinite")
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def _inverse_sqrt_spd(matrix: FloatArray) -> FloatArray:
    _assert_spd(matrix, "matrix")
    values, vectors = np.linalg.eigh(_symmetrise(matrix))
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _logdet_spd(matrix: FloatArray, name: str = "matrix") -> float:
    _assert_spd(matrix, name)
    sign, value = np.linalg.slogdet(_symmetrise(matrix))
    if sign <= 0:
        raise ValueError(f"{name} has non-positive determinant")
    return float(value)


def helmert_reference(number_of_electrodes: int) -> FloatArray:
    """Return orthonormal average-reference contrasts of shape ``(N-1, N)``.

    The rows span the subspace orthogonal to the all-ones vector.  This avoids
    carrying a singular ``N x N`` average-reference covariance through the
    theoretical calculations.
    """

    if number_of_electrodes < 2:
        raise ValueError("at least two electrodes are required")
    result = np.zeros((number_of_electrodes - 1, number_of_electrodes))
    for row in range(1, number_of_electrodes):
        normaliser = sqrt(row * (row + 1))
        result[row - 1, :row] = 1.0 / normaliser
        result[row - 1, row] = -row / normaliser
    return result


def geometry_efficiency(contributions: ArrayLike) -> float:
    """Return the cancellation efficiency of whitened elementary contributions.

    ``contributions[:, i]`` is the signed sensor-space contribution from the
    ``i``-th elementary laminar generator after nuisance whitening.  The result
    is in ``[0, 1]`` up to floating-point error.
    """

    contributions = _as_float_matrix(contributions, "contributions")
    denominator = float(np.sum(np.linalg.norm(contributions, axis=0)))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.linalg.norm(np.sum(contributions, axis=1)))
    return numerator / denominator


def equicorrelated_sensor_power(
    contributions: ArrayLike, rho: float, variance: float = 1.0
) -> float:
    """Whitened sensor power for equicorrelated elementary generators.

    If there are ``p`` generators, validity requires
    ``-1/(p-1) <= rho <= 1`` (or ``rho == 1`` for ``p == 1``).
    """

    contributions = _as_float_matrix(contributions, "contributions")
    if variance < 0:
        raise ValueError("variance must be non-negative")
    number_of_generators = contributions.shape[1]
    lower = -1.0 / (number_of_generators - 1) if number_of_generators > 1 else 1.0
    if rho < lower - 1e-12 or rho > 1.0 + 1e-12:
        raise ValueError("rho does not define a positive-semidefinite covariance")
    incoherent = float(np.sum(contributions * contributions))
    coherent = float(np.sum(np.sum(contributions, axis=1) ** 2))
    return variance * ((1.0 - rho) * incoherent + rho * coherent)


def sensor_power_from_covariance(
    contributions: ArrayLike, source_covariance: ArrayLike
) -> float:
    """Return ``E[||Bq||^2] = tr(B Cq B*)`` for real or complex fields.

    This is the frequency-domain generalization of
    :func:`equicorrelated_sensor_power`.  The covariance must be Hermitian
    positive semidefinite; complex phases are retained rather than replaced by
    a scalar coherence factor.
    """

    contributions = _as_complex_matrix(contributions, "contributions")
    covariance = _as_complex_matrix(source_covariance, "source_covariance")
    generators = contributions.shape[1]
    if covariance.shape != (generators, generators):
        raise ValueError("source covariance has incompatible dimensions")
    scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
    hermitian_error = float(np.linalg.norm(covariance - covariance.conj().T, ord=2))
    if hermitian_error > 1e-10 * scale:
        raise ValueError("source covariance must be Hermitian")
    covariance = 0.5 * (covariance + covariance.conj().T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError("source covariance must be positive semidefinite")
    value = np.trace(contributions @ covariance @ contributions.conj().T)
    if abs(float(value.imag)) > 1e-10 * max(1.0, abs(float(value.real))):
        raise FloatingPointError("sensor power acquired a non-negligible imaginary part")
    return max(0.0, float(value.real))


def weighted_projection_residual(
    hippocampal_topography: ArrayLike,
    cortical_leadfield: ArrayLike,
    precision: ArrayLike,
) -> tuple[FloatArray, float]:
    """Return the minimum weighted cortical-masking residual and its energy.

    The energy equals ``min_z (h - Gc z)^T precision (h - Gc z)``.  The returned
    residual is in whitened sensor coordinates.
    """

    topography = np.asarray(hippocampal_topography, dtype=np.float64).reshape(-1)
    cortical = _as_float_matrix(cortical_leadfield, "cortical_leadfield")
    precision = _as_float_matrix(precision, "precision")
    if cortical.shape[0] != topography.size:
        raise ValueError("topography and lead field have incompatible dimensions")
    if precision.shape != (topography.size, topography.size):
        raise ValueError("precision has incompatible dimensions")
    _assert_spd(precision, "precision")

    # If precision = L L^T, then C=L^T obeys C^T C=precision.
    whitening = np.linalg.cholesky(_symmetrise(precision)).T
    whitened_topography = whitening @ topography
    whitened_cortical = whitening @ cortical
    coefficients = np.linalg.lstsq(
        whitened_cortical, whitened_topography, rcond=None
    )[0]
    residual = whitened_topography - whitened_cortical @ coefficients
    return residual, float(residual @ residual)


def whitened_signal_eigenvalues(
    hippocampal_leadfield: ArrayLike,
    source_covariance: ArrayLike,
    nuisance_covariance: ArrayLike,
) -> FloatArray:
    """Generalized recoverability eigenvalues in nuisance-whitened sensor space."""

    leadfield = _as_float_matrix(hippocampal_leadfield, "hippocampal_leadfield")
    source_covariance = _as_float_matrix(source_covariance, "source_covariance")
    nuisance_covariance = _as_float_matrix(
        nuisance_covariance, "nuisance_covariance"
    )
    if source_covariance.shape != (leadfield.shape[1], leadfield.shape[1]):
        raise ValueError("source covariance has incompatible dimensions")
    if nuisance_covariance.shape != (leadfield.shape[0], leadfield.shape[0]):
        raise ValueError("nuisance covariance has incompatible dimensions")
    _psd_sqrt(source_covariance)
    inverse_sqrt = _inverse_sqrt_spd(nuisance_covariance)
    signal_covariance = leadfield @ source_covariance @ leadfield.T
    whitened = _symmetrise(inverse_sqrt @ signal_covariance @ inverse_sqrt)
    values = np.linalg.eigvalsh(whitened)
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < -1e-10 * scale:
        raise FloatingPointError("whitened covariance acquired a negative eigenvalue")
    return np.maximum(values, 0.0)[::-1]


def gaussian_mi_bits(
    hippocampal_leadfield: ArrayLike,
    source_covariance: ArrayLike,
    nuisance_covariance: ArrayLike,
) -> float:
    """Mutual information ``I(x_h; y)`` in bits for the linear Gaussian model."""

    values = whitened_signal_eigenvalues(
        hippocampal_leadfield, source_covariance, nuisance_covariance
    )
    return 0.5 * float(np.sum(np.log1p(values))) / log(2.0)


def gaussian_posterior_covariance(
    hippocampal_leadfield: ArrayLike,
    source_covariance: ArrayLike,
    nuisance_covariance: ArrayLike,
) -> FloatArray:
    """Posterior covariance for an SPD Gaussian source prior."""

    leadfield = _as_float_matrix(hippocampal_leadfield, "hippocampal_leadfield")
    source_covariance = _as_float_matrix(source_covariance, "source_covariance")
    nuisance_covariance = _as_float_matrix(
        nuisance_covariance, "nuisance_covariance"
    )
    _assert_spd(source_covariance, "source_covariance")
    _assert_spd(nuisance_covariance, "nuisance_covariance")
    prior_precision = np.linalg.inv(_symmetrise(source_covariance))
    nuisance_precision = np.linalg.inv(_symmetrise(nuisance_covariance))
    posterior_precision = (
        prior_precision + leadfield.T @ nuisance_precision @ leadfield
    )
    return _symmetrise(np.linalg.inv(_symmetrise(posterior_precision)))


def kl_present_vs_absent_nats(recoverability_eigenvalues: ArrayLike) -> float:
    """Return ``D[N(0,R0+Kh) || N(0,R0)]`` in nats."""

    values = np.asarray(recoverability_eigenvalues, dtype=np.float64).reshape(-1)
    if np.any(values < -1e-12):
        raise ValueError("recoverability eigenvalues must be non-negative")
    values = np.maximum(values, 0.0)
    return 0.5 * float(np.sum(values - np.log1p(values)))


def matched_filter_dprime(signal: ArrayLike, nuisance_covariance: ArrayLike) -> float:
    """Optimal equal-covariance Gaussian detection index for a known signal."""

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    covariance = _as_float_matrix(nuisance_covariance, "nuisance_covariance")
    if covariance.shape != (signal.size, signal.size):
        raise ValueError("signal and nuisance covariance have incompatible dimensions")
    _assert_spd(covariance, "nuisance_covariance")
    value = float(signal @ np.linalg.solve(_symmetrise(covariance), signal))
    return sqrt(max(value, 0.0))


def matched_filter_auc(dprime: float) -> float:
    """Theoretical AUC of the optimal equal-covariance Gaussian matched filter."""

    if dprime < 0:
        raise ValueError("dprime must be non-negative")
    # Phi(d'/sqrt(2)) = 0.5 * [1 + erf(d'/2)].
    return 0.5 * (1.0 + erf(dprime / 2.0))


def perturbation_mi_interval_bits(
    nominal_eigenvalues: ArrayLike, spectral_error_bound: float
) -> tuple[float, float]:
    """Weyl-based MI interval for a whitened covariance perturbation."""

    values = np.asarray(nominal_eigenvalues, dtype=np.float64).reshape(-1)
    if np.any(values < -1e-12):
        raise ValueError("nominal eigenvalues must be non-negative")
    if spectral_error_bound < 0:
        raise ValueError("spectral_error_bound must be non-negative")
    lower_values = np.maximum(0.0, values - spectral_error_bound)
    upper_values = np.maximum(0.0, values + spectral_error_bound)
    factor = 0.5 / log(2.0)
    return (
        factor * float(np.sum(np.log1p(lower_values))),
        factor * float(np.sum(np.log1p(upper_values))),
    )


def forward_error_dprime_interval(
    nominal_signal: ArrayLike,
    whitened_forward_error: ArrayLike,
    source_amplitudes: ArrayLike,
) -> tuple[float, float]:
    """Triangle-inequality interval for true ``d'`` under forward error.

    ``nominal_signal`` and ``whitened_forward_error`` must already be in
    nuisance-whitened sensor coordinates.
    """

    nominal_signal = np.asarray(nominal_signal, dtype=np.float64).reshape(-1)
    error = _as_float_matrix(whitened_forward_error, "whitened_forward_error")
    source = np.asarray(source_amplitudes, dtype=np.float64).reshape(-1)
    if error.shape != (nominal_signal.size, source.size):
        raise ValueError("forward error has incompatible dimensions")
    radius = float(np.linalg.norm(error, ord=2) * np.linalg.norm(source))
    nominal = float(np.linalg.norm(nominal_signal))
    return max(0.0, nominal - radius), nominal + radius


def _ar1_covariance(length: int, rho: float) -> FloatArray:
    indices = np.arange(length)
    return rho ** np.abs(indices[:, None] - indices[None, :])


def ar1_finite_window_mi_rate_bits(
    length: int, rho: float, gain: float, noise_variance: float
) -> float:
    """Exact per-sample Gaussian MI for a finite AR(1) source window."""

    if length < 1:
        raise ValueError("length must be positive")
    if abs(rho) >= 1:
        raise ValueError("rho must satisfy |rho| < 1")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    covariance = _ar1_covariance(length, rho)
    matrix = np.eye(length) + (gain * gain / noise_variance) * covariance
    return 0.5 * _logdet_spd(matrix) / (length * log(2.0))


def ar1_spectral_mi_rate_bits(
    rho: float,
    gain: float,
    noise_variance: float,
    integration_points: int = 65536,
) -> float:
    """Stationary Gaussian spectral MI rate for a unit-variance AR(1) source."""

    if abs(rho) >= 1:
        raise ValueError("rho must satisfy |rho| < 1")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    if integration_points < 128:
        raise ValueError("integration_points is too small")
    omega = np.linspace(-pi, pi, integration_points, endpoint=False)
    spectrum = (1.0 - rho * rho) / (
        1.0 + rho * rho - 2.0 * rho * np.cos(omega)
    )
    integrand = np.log1p((gain * gain / noise_variance) * spectrum)
    # The grid covers 2*pi, so mean(integrand)/2 is 1/(4*pi) integral.
    return 0.5 * float(np.mean(integrand)) / log(2.0)


def subset_gaussian_mi_bits(
    hippocampal_leadfield: ArrayLike,
    source_covariance: ArrayLike,
    nuisance_covariance: ArrayLike,
    sensors: Iterable[int],
) -> float:
    """Gaussian MI for a selected sensor subset."""

    leadfield = _as_float_matrix(hippocampal_leadfield, "hippocampal_leadfield")
    nuisance = _as_float_matrix(nuisance_covariance, "nuisance_covariance")
    indices = np.asarray(list(sensors), dtype=int)
    if indices.size == 0:
        return 0.0
    if np.any(indices < 0) or np.any(indices >= leadfield.shape[0]):
        raise ValueError("sensor index is out of range")
    return gaussian_mi_bits(
        leadfield[indices, :],
        source_covariance,
        nuisance[np.ix_(indices, indices)],
    )
