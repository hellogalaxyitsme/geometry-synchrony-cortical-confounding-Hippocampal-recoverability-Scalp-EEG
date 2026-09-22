"""Leakage-safe inverse-method benchmarks for method-benchmark.

The primitives in this module deliberately separate three concerns:

* sensor-level feature extraction from event epochs;
* fitting an operator using training data only; and
* anatomical scoring after applying a frozen operator.

Arrays use events-by-sensors-by-time for epochs, sensors-by-sources for lead
fields, and sources-by-sensors for inverse operators.  The implementation is
small enough to audit and has no outcome-adaptive defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from empirical.working_memory import roc_auc


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _finite_matrix(value: ArrayLike, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return matrix


def _symmetric(value: ArrayLike, name: str) -> FloatArray:
    matrix = _finite_matrix(value, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def average_reference_epochs(epochs: ArrayLike) -> FloatArray:
    """Average-reference finite event epochs without changing their shape."""
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("epochs must be finite events-by-sensors-by-time")
    return np.asarray(values - values.mean(axis=1, keepdims=True), dtype=np.float64)


def event_covariances(epochs: ArrayLike) -> FloatArray:
    """Return one mean-removed sensor covariance per event."""
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("epochs must be finite events-by-sensors-by-time")
    centered = values - values.mean(axis=2, keepdims=True)
    return np.asarray(
        np.einsum("est,eft->esf", centered, centered, optimize=True) / centered.shape[2],
        dtype=np.float64,
    )


def log_channel_power(epochs: ArrayLike) -> FloatArray:
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("epochs must be finite events-by-sensors-by-time")
    power = np.mean(values * values, axis=2)
    return np.asarray(np.log(np.maximum(power, np.finfo(np.float64).tiny)), dtype=np.float64)


def regularized_covariance(samples: ArrayLike, ridge_fraction: float) -> FloatArray:
    matrix = _finite_matrix(samples, "samples")
    if matrix.shape[0] < 2 or not 0.0 <= ridge_fraction <= 1.0:
        raise ValueError("invalid covariance request")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    scale = float(np.trace(covariance) / covariance.shape[0])
    if scale <= 0.0:
        raise ValueError("training samples have zero covariance")
    return np.asarray(
        (1.0 - ridge_fraction) * covariance
        + ridge_fraction * scale * np.eye(covariance.shape[0]),
        dtype=np.float64,
    )


def _inverse_sqrt(matrix: ArrayLike, relative_floor: float = 1e-9) -> FloatArray:
    covariance = _symmetric(matrix, "covariance")
    if relative_floor <= 0.0:
        raise ValueError("relative floor must be positive")
    values, vectors = np.linalg.eigh(covariance)
    ceiling = max(float(values[-1]), np.finfo(np.float64).tiny)
    values = np.maximum(values, relative_floor * ceiling)
    return np.asarray((vectors / np.sqrt(values)[None, :]) @ vectors.T, dtype=np.float64)


@dataclass(frozen=True)
class FastICA:
    center: FloatArray
    unmixing: FloatArray
    converged: bool
    iterations: int

    def transform(self, samples: ArrayLike) -> FloatArray:
        matrix = _finite_matrix(samples, "ICA samples")
        if matrix.shape[1] != len(self.center):
            raise ValueError("ICA sensor dimension mismatch")
        return np.asarray((matrix - self.center) @ self.unmixing.T, dtype=np.float64)


def _symmetric_decorrelation(matrix: FloatArray) -> FloatArray:
    gram = matrix @ matrix.T
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    floor = 1e-12 * max(float(values[-1]), np.finfo(np.float64).tiny)
    if np.any(values <= floor):
        raise ValueError("ICA iterate lost row rank")
    return np.asarray((vectors / np.sqrt(values)[None, :]) @ vectors.T @ matrix)


def fit_fastica(
    samples: ArrayLike,
    components: int,
    *,
    seed: int,
    maximum_iterations: int = 500,
    tolerance: float = 1e-7,
) -> FastICA:
    """Fit symmetric FastICA on training samples only (tanh contrast)."""
    matrix = _finite_matrix(samples, "ICA training samples")
    if not 1 <= components <= matrix.shape[1] or maximum_iterations < 1 or tolerance <= 0.0:
        raise ValueError("invalid FastICA settings")
    center = matrix.mean(axis=0)
    centered = matrix - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1][:components]
    selected = values[order]
    if selected[-1] <= 1e-12 * max(float(selected[0]), np.finfo(np.float64).tiny):
        raise ValueError("requested ICA dimension exceeds training rank")
    whitening = (vectors[:, order] / np.sqrt(selected)[None, :]).T
    whitened = centered @ whitening.T
    generator = np.random.default_rng(seed)
    weight = _symmetric_decorrelation(generator.normal(size=(components, components)))
    converged = False
    iterations = maximum_iterations
    for iteration in range(1, maximum_iterations + 1):
        projection = whitened @ weight.T
        nonlinearity = np.tanh(projection)
        derivative = 1.0 - nonlinearity * nonlinearity
        updated = nonlinearity.T @ whitened / len(whitened)
        updated -= derivative.mean(axis=0)[:, None] * weight
        updated = _symmetric_decorrelation(updated)
        alignment = np.abs(np.diag(updated @ weight.T))
        weight = updated
        if float(np.max(np.abs(alignment - 1.0))) <= tolerance:
            converged = True
            iterations = iteration
            break
    return FastICA(
        center=np.asarray(center, dtype=np.float64),
        unmixing=np.asarray(weight @ whitening, dtype=np.float64),
        converged=converged,
        iterations=iterations,
    )


def ica_event_features(model: FastICA, epochs: ArrayLike) -> FloatArray:
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(model.center):
        raise ValueError("ICA epochs have the wrong dimensions")
    flattened = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    sources = model.transform(flattened).reshape(values.shape[0], values.shape[2], -1)
    power = np.mean(sources * sources, axis=1)
    return np.asarray(np.log(np.maximum(power, np.finfo(np.float64).tiny)), dtype=np.float64)


def select_signed_feature(features: ArrayLike, labels: ArrayLike) -> tuple[int, float, float]:
    matrix = _finite_matrix(features, "features")
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(matrix) != len(truth) or set(np.unique(truth)) != {0, 1}:
        raise ValueError("feature selection needs aligned binary training labels")
    candidates: list[tuple[float, int, float]] = []
    for column in range(matrix.shape[1]):
        raw = roc_auc(truth, matrix[:, column])
        sign = 1.0 if raw >= 0.5 else -1.0
        candidates.append((max(raw, 1.0 - raw), column, sign))
    auc, column, sign = max(candidates, key=lambda item: (item[0], -item[1]))
    return int(column), float(sign), float(auc)


def _validate_leadfield(leadfield: ArrayLike) -> FloatArray:
    matrix = _finite_matrix(leadfield, "lead field")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("lead field is too small")
    if np.max(np.linalg.norm(matrix, axis=0)) <= 0.0:
        raise ValueError("lead field has zero gain")
    return matrix


def minimum_norm_operator(
    leadfield: ArrayLike,
    noise_covariance: ArrayLike,
    regularization: float,
    source_variances: ArrayLike | None = None,
) -> FloatArray:
    """Whitened fixed-orientation minimum-norm inverse operator."""
    gain = _validate_leadfield(leadfield)
    noise = _symmetric(noise_covariance, "noise covariance")
    if noise.shape[0] != gain.shape[0] or regularization <= 0.0:
        raise ValueError("invalid MNE covariance or regularization")
    if source_variances is None:
        variances = np.ones(gain.shape[1], dtype=np.float64)
    else:
        variances = np.asarray(source_variances, dtype=np.float64).reshape(-1)
        if len(variances) != gain.shape[1] or np.any(variances <= 0.0):
            raise ValueError("source variances must be positive and aligned")
    whitening = _inverse_sqrt(noise)
    white_gain = whitening @ gain
    sensor = (white_gain * variances[None, :]) @ white_gain.T
    sensor += regularization * np.eye(sensor.shape[0])
    inverse_white = (variances[:, None] * white_gain.T) @ np.linalg.pinv(sensor, rcond=1e-12)
    return np.asarray(inverse_white @ whitening, dtype=np.float64)


def eloreta_type_operator(
    leadfield: ArrayLike,
    noise_covariance: ArrayLike,
    regularization: float,
    *,
    maximum_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[FloatArray, dict[str, float | int | bool]]:
    """Iterative fixed-orientation eLORETA-type standardized inverse.

    This follows the diagonal weight fixed point used by exact-low-resolution
    tomography for scalar, fixed-orientation sources.  We label it "type"
    because the project source dictionary contains physiological factors as
    well as elementary dipoles.
    """
    gain = _validate_leadfield(leadfield)
    noise = _symmetric(noise_covariance, "noise covariance")
    if noise.shape[0] != gain.shape[0] or regularization <= 0.0:
        raise ValueError("invalid eLORETA settings")
    whitening = _inverse_sqrt(noise)
    white_gain = whitening @ gain
    norms = np.linalg.norm(white_gain, axis=0)
    if np.any(norms <= 1e-14 * float(np.max(norms))):
        raise ValueError("eLORETA lead field contains negligible columns")
    weights = norms.copy()
    converged = False
    relative_change = math.inf
    iterations = maximum_iterations
    for iteration in range(1, maximum_iterations + 1):
        inverse_weights = 1.0 / np.maximum(weights, np.finfo(np.float64).tiny)
        sensor = (white_gain * inverse_weights[None, :]) @ white_gain.T
        sensor += regularization * np.eye(sensor.shape[0])
        metric = np.linalg.pinv(sensor, rcond=1e-12)
        updated = np.sqrt(
            np.maximum(np.einsum("si,st,ti->i", white_gain, metric, white_gain), 0.0)
        )
        updated /= np.median(updated)
        weights_normalized = weights / np.median(weights)
        relative_change = float(
            np.max(np.abs(updated - weights_normalized) / np.maximum(weights_normalized, 1e-12))
        )
        weights = updated
        if relative_change <= tolerance:
            converged = True
            iterations = iteration
            break
    inverse_weights = 1.0 / weights
    sensor = (white_gain * inverse_weights[None, :]) @ white_gain.T
    sensor += regularization * np.eye(sensor.shape[0])
    operator = (inverse_weights[:, None] * white_gain.T) @ np.linalg.pinv(sensor, rcond=1e-12)
    return np.asarray(operator @ whitening, dtype=np.float64), {
        "converged": converged,
        "iterations": iterations,
        "maximum_relative_weight_change": relative_change,
    }


def lcmv_operator(
    leadfield: ArrayLike,
    training_covariance: ArrayLike,
    ridge_fraction: float,
) -> FloatArray:
    """Return scalar fixed-orientation unit-gain LCMV filters."""
    gain = _validate_leadfield(leadfield)
    covariance = _symmetric(training_covariance, "training covariance")
    if covariance.shape[0] != gain.shape[0] or not 0.0 < ridge_fraction <= 1.0:
        raise ValueError("invalid LCMV settings")
    scale = float(np.trace(covariance) / covariance.shape[0])
    if scale <= 0.0:
        raise ValueError("LCMV covariance has zero trace")
    regularized = covariance + ridge_fraction * scale * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(regularized, rcond=1e-12)
    numerator = precision @ gain
    denominator = np.einsum("si,si->i", gain, numerator)
    floor = 1e-14 * max(float(np.max(np.abs(denominator))), np.finfo(np.float64).tiny)
    if np.any(denominator <= floor):
        raise ValueError("LCMV has a non-positive or negligible unit-gain denominator")
    return np.asarray((numerator / denominator[None, :]).T, dtype=np.float64)


def anatomical_log_power_ratio(
    inverse_operator: ArrayLike,
    covariances: ArrayLike,
    hippocampal_indices: ArrayLike,
    cortical_indices: ArrayLike,
) -> FloatArray:
    """Score hippocampal versus cortical mean reconstructed source power."""
    operator = _finite_matrix(inverse_operator, "inverse operator")
    matrices = np.asarray(covariances, dtype=np.float64)
    hippocampus = np.asarray(hippocampal_indices, dtype=np.int64).reshape(-1)
    cortex = np.asarray(cortical_indices, dtype=np.int64).reshape(-1)
    if (
        matrices.ndim != 3
        or matrices.shape[1:] != (operator.shape[1], operator.shape[1])
        or len(hippocampus) == 0
        or len(cortex) == 0
        or np.intersect1d(hippocampus, cortex).size
        or np.any(np.concatenate((hippocampus, cortex)) >= operator.shape[0])
    ):
        raise ValueError("invalid anatomical score inputs")
    h_gram = operator[hippocampus].T @ operator[hippocampus] / len(hippocampus)
    c_gram = operator[cortex].T @ operator[cortex] / len(cortex)
    h_power = np.einsum("eij,ji->e", matrices, h_gram, optimize=True)
    c_power = np.einsum("eij,ji->e", matrices, c_gram, optimize=True)
    floor = np.finfo(np.float64).tiny
    return np.asarray(np.log(np.maximum(h_power, floor) / np.maximum(c_power, floor)))


def covariance_factors(covariance: ArrayLike, components: int) -> FloatArray:
    matrix = _symmetric(covariance, "event covariance")
    if not 1 <= components <= len(matrix):
        raise ValueError("invalid covariance factor count")
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1][:components]
    keep = values[order] > 1e-14 * max(float(values[-1]), np.finfo(np.float64).tiny)
    if not np.any(keep):
        return np.zeros((len(matrix), 1), dtype=np.float64)
    indices = order[keep]
    return np.asarray(vectors[:, indices] * np.sqrt(np.maximum(values[indices], 0.0))[None, :])


def _soft_threshold(values: FloatArray, threshold: float) -> FloatArray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def sparse_group_fista(
    design: ArrayLike,
    targets: ArrayLike,
    l1_penalty: float,
    *,
    groups: Sequence[ArrayLike] = (),
    group_penalty: float = 0.0,
    group_weights: ArrayLike | None = None,
    maximum_iterations: int = 500,
    tolerance: float = 1e-7,
) -> tuple[FloatArray, dict[str, float | int | bool]]:
    """Solve a multi-target sparse-group inverse with FISTA.

    Groups must be disjoint.  An empty group list and zero group penalty reduce
    exactly to the ordinary L1 sparse inverse benchmark.
    """
    matrix = _finite_matrix(design, "sparse design")
    response = np.asarray(targets, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, None]
    if response.ndim != 2 or response.shape[0] != matrix.shape[0] or not np.all(np.isfinite(response)):
        raise ValueError("sparse targets do not match the design")
    if l1_penalty < 0.0 or group_penalty < 0.0 or maximum_iterations < 1 or tolerance <= 0.0:
        raise ValueError("invalid sparse penalties or iteration settings")
    parsed = [np.asarray(group, dtype=np.int64).reshape(-1) for group in groups]
    if parsed:
        joined = np.concatenate(parsed)
        if (
            any(len(group) == 0 for group in parsed)
            or len(np.unique(joined)) != len(joined)
            or np.any(joined < 0)
            or np.any(joined >= matrix.shape[1])
        ):
            raise ValueError("sparse groups must be nonempty, valid, and disjoint")
    if group_weights is None:
        weights = np.sqrt(np.asarray([len(group) for group in parsed], dtype=np.float64))
    else:
        weights = np.asarray(group_weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(parsed) or np.any(weights <= 0.0):
            raise ValueError("group weights must be positive and aligned")
    lipschitz = float(np.linalg.norm(matrix, ord=2) ** 2)
    if lipschitz <= 0.0:
        raise ValueError("sparse design has zero norm")
    step = 1.0 / lipschitz
    coefficients = np.zeros((matrix.shape[1], response.shape[1]), dtype=np.float64)
    momentum_point = coefficients.copy()
    acceleration = 1.0
    converged = False
    relative_change = math.inf
    stationarity = math.inf
    iterations = maximum_iterations

    def proximal(values: FloatArray) -> FloatArray:
        result = _soft_threshold(values, step * l1_penalty)
        if group_penalty > 0.0:
            for group, weight in zip(parsed, weights):
                norm = float(np.linalg.norm(result[group]))
                shrink = max(
                    0.0,
                    1.0 - step * group_penalty * float(weight) / max(norm, 1e-30),
                )
                result[group] *= shrink
        return result

    for iteration in range(1, maximum_iterations + 1):
        gradient = matrix.T @ (matrix @ momentum_point - response)
        updated = proximal(momentum_point - step * gradient)
        denominator = max(float(np.linalg.norm(coefficients)), 1.0)
        relative_change = float(np.linalg.norm(updated - coefficients) / denominator)
        fixed_point = proximal(
            updated - step * (matrix.T @ (matrix @ updated - response))
        )
        stationarity = float(
            np.linalg.norm(fixed_point - updated) / max(float(np.linalg.norm(updated)), 1.0)
        )
        restart = float(np.sum((momentum_point - updated) * (updated - coefficients))) > 0.0
        next_acceleration = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration))
        if restart:
            next_acceleration = 1.0
            momentum_point = updated.copy()
        else:
            momentum_point = updated + ((acceleration - 1.0) / next_acceleration) * (
                updated - coefficients
            )
        coefficients = updated
        acceleration = next_acceleration
        if stationarity <= tolerance:
            converged = True
            iterations = iteration
            break
    residual = matrix @ coefficients - response
    objective = 0.5 * float(np.sum(residual * residual))
    objective += l1_penalty * float(np.sum(np.abs(coefficients)))
    if group_penalty > 0.0:
        objective += group_penalty * sum(
            float(weight) * float(np.linalg.norm(coefficients[group]))
            for group, weight in zip(parsed, weights)
        )
    return coefficients, {
        "converged": converged,
        "iterations": iterations,
        "relative_change": relative_change,
        "proximal_gradient_stationarity": stationarity,
        "objective": objective,
    }


def sparse_anatomical_score(
    coefficients: ArrayLike,
    hippocampal_indices: ArrayLike,
    cortical_indices: ArrayLike,
) -> float:
    matrix = _finite_matrix(coefficients, "sparse coefficients")
    hippocampus = np.asarray(hippocampal_indices, dtype=np.int64).reshape(-1)
    cortex = np.asarray(cortical_indices, dtype=np.int64).reshape(-1)
    if len(hippocampus) == 0 or len(cortex) == 0:
        raise ValueError("both anatomical source families are required")
    h_power = float(np.mean(matrix[hippocampus] ** 2))
    c_power = float(np.mean(matrix[cortex] ** 2))
    return float(np.log(max(h_power, np.finfo(float).tiny) / max(c_power, np.finfo(float).tiny)))


def patient_block_folds(
    patients: ArrayLike,
    blocks: ArrayLike,
    training_blocks: Iterable[int],
    testing_blocks: Iterable[int],
) -> list[dict[str, object]]:
    """Construct audited leave-one-patient-out, chronological transfer folds."""
    groups = np.asarray(patients, dtype=str).reshape(-1)
    time = np.asarray(blocks, dtype=np.int64).reshape(-1)
    train_values = tuple(sorted(set(int(value) for value in training_blocks)))
    test_values = tuple(sorted(set(int(value) for value in testing_blocks)))
    if len(groups) != len(time) or not train_values or not test_values or set(train_values) & set(test_values):
        raise ValueError("invalid patient/block fold request")
    folds: list[dict[str, object]] = []
    for heldout in np.unique(groups):
        train = (groups != heldout) & np.isin(time, train_values)
        test = (groups == heldout) & np.isin(time, test_values)
        if not np.any(train) or not np.any(test):
            raise ValueError(f"empty patient/block partition for {heldout}")
        folds.append(
            {
                "heldout_patient": str(heldout),
                "train_indices": np.flatnonzero(train).astype(np.int64),
                "test_indices": np.flatnonzero(test).astype(np.int64),
                "training_blocks": list(train_values),
                "testing_blocks": list(test_values),
            }
        )
    return folds
