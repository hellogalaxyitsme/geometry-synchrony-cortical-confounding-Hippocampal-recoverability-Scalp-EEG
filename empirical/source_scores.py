"""Seven-method cortical-control scoring for the sparse-group experiment."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from empirical.estimators import FastICA


FloatArray = NDArray[np.float64]


def factor_covariance(factors: ArrayLike) -> FloatArray:
    matrix = np.asarray(factors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("factors must be finite sensors-by-components")
    return np.asarray(matrix @ matrix.T / matrix.shape[1], dtype=np.float64)


def stochastic_injection_covariance(
    baseline_covariance: ArrayLike, topography: ArrayLike, energy_ratio: float
) -> FloatArray:
    covariance = np.asarray(baseline_covariance, dtype=np.float64)
    vector = np.asarray(topography, dtype=np.float64).reshape(-1)
    if (
        covariance.ndim != 2
        or covariance.shape != (len(vector), len(vector))
        or energy_ratio <= 0.0
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError("invalid stochastic injection inputs")
    norm = float(np.linalg.norm(vector))
    trace = float(np.trace(covariance))
    if norm <= 0.0 or trace <= 0.0:
        raise ValueError("injection topography/baseline has zero energy")
    unit = vector / norm
    return np.asarray(covariance + energy_ratio * trace * np.outer(unit, unit), dtype=np.float64)


def best_channel_anatomical_scores(
    baseline_covariance: ArrayLike,
    response_covariance: ArrayLike,
    injection_covariance: ArrayLike,
    injection_topography: ArrayLike,
) -> tuple[float, float, float, int]:
    baseline = np.asarray(baseline_covariance, dtype=np.float64)
    response = np.asarray(response_covariance, dtype=np.float64)
    injected = np.asarray(injection_covariance, dtype=np.float64)
    topography = np.asarray(injection_topography, dtype=np.float64).reshape(-1)
    noise = np.maximum(np.diag(baseline), np.finfo(float).tiny)
    channel = int(np.argmax(topography * topography / noise))

    def score(covariance: FloatArray) -> float:
        return float(np.log(max(float(covariance[channel, channel]), np.finfo(float).tiny)))

    base = score(baseline)
    return base, score(response), score(injected), channel


def ica_anatomical_scores(
    model: FastICA,
    baseline_covariance: ArrayLike,
    response_covariance: ArrayLike,
    injection_covariance: ArrayLike,
    injection_topography: ArrayLike,
) -> tuple[float, float, float, int]:
    baseline = np.asarray(baseline_covariance, dtype=np.float64)
    response = np.asarray(response_covariance, dtype=np.float64)
    injected = np.asarray(injection_covariance, dtype=np.float64)
    topography = np.asarray(injection_topography, dtype=np.float64).reshape(-1)
    unmixing = np.asarray(model.unmixing, dtype=np.float64)
    noise = np.einsum("is,st,it->i", unmixing, baseline, unmixing, optimize=True)
    signal = (unmixing @ topography) ** 2
    component = int(np.argmax(signal / np.maximum(noise, np.finfo(float).tiny)))
    vector = unmixing[component]

    def score(covariance: FloatArray) -> float:
        power = float(vector @ covariance @ vector)
        return float(np.log(max(power, np.finfo(float).tiny)))

    base = score(baseline)
    return base, score(response), score(injected), component


def heldout_injection_threshold(values: ArrayLike, sensitivity_target: float) -> float:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(scores) == 0 or not np.all(np.isfinite(scores)) or not 0.0 < sensitivity_target < 1.0:
        raise ValueError("invalid injection-threshold inputs")
    return float(np.quantile(scores, 1.0 - sensitivity_target, method="lower"))


def stable_anatomical_log_power_ratio(
    coefficients: ArrayLike,
    hippocampal_indices: ArrayLike,
    cortical_indices: ArrayLike,
) -> float:
    """Log mean-power ratio without forming an overflow-prone raw ratio."""
    matrix = np.asarray(coefficients, dtype=np.float64)
    hippocampus = np.asarray(hippocampal_indices, dtype=np.int64).reshape(-1)
    cortex = np.asarray(cortical_indices, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("coefficients must be a finite matrix")
    if len(hippocampus) == 0 or len(cortex) == 0:
        raise ValueError("both anatomical source families are required")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        return 0.0
    normalized = matrix / scale
    h_power = float(np.mean(normalized[hippocampus] ** 2))
    c_power = float(np.mean(normalized[cortex] ** 2))
    tiny = np.finfo(float).tiny
    value = float(np.log(max(h_power, tiny)) - np.log(max(c_power, tiny)))
    if not np.isfinite(value):
        raise FloatingPointError("non-finite anatomical log-power ratio")
    return value
