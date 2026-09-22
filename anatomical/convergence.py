"""Scale-aware and scale-free forward-model convergence metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from theory.recoverability import helmert_reference


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SensorOperators:
    hippocampal_covariance: FloatArray
    cortical_covariance: FloatArray
    hippocampal_wave_covariance: FloatArray
    hippocampal_coherent_topography: FloatArray


def _validate_forward_geometry(
    leadfield: FloatArray,
    area_weights_m2: FloatArray,
    label: str,
) -> tuple[FloatArray, FloatArray]:
    leadfield = np.asarray(leadfield, dtype=np.float64)
    weights = np.asarray(area_weights_m2, dtype=np.float64).reshape(-1)
    if leadfield.ndim != 2 or leadfield.shape[1] != len(weights):
        raise ValueError(f"{label} leadfield/area dimensions disagree")
    if not np.all(np.isfinite(leadfield)):
        raise ValueError(f"{label} leadfield contains non-finite values")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(f"{label} areas must be finite and positive")
    return leadfield - leadfield.mean(axis=0, keepdims=True), weights / weights.sum()


def sensor_operators(
    hippocampal_leadfield: FloatArray,
    cortical_leadfield: FloatArray,
    hippocampal_area_weights_m2: FloatArray,
    cortical_area_weights_m2: FloatArray,
    hippocampal_longitudinal_coordinate: FloatArray | None,
    *,
    hippocampal_wave_basis: FloatArray | None = None,
) -> SensorOperators:
    """Construct mesh-stable geometry operators without physiological scaling.

    The covariance quadrature approximates ``A^-1 integral l(s)l(s)^T dA``.
    The wave operator spans one cosine/sine cycle along the declared
    hippocampal longitudinal coordinate. These are numerical convergence
    probes, not source-amplitude priors for physiological inference.
    """

    hippocampal, h_weights = _validate_forward_geometry(
        hippocampal_leadfield, hippocampal_area_weights_m2, "hippocampal"
    )
    cortical, c_weights = _validate_forward_geometry(
        cortical_leadfield, cortical_area_weights_m2, "cortical"
    )
    if hippocampal.shape[0] != cortical.shape[0]:
        raise ValueError("hippocampal and cortical sensor counts disagree")
    if hippocampal_wave_basis is None:
        if hippocampal_longitudinal_coordinate is None:
            raise ValueError("a hippocampal coordinate or explicit wave basis is required")
        coordinate = np.asarray(
            hippocampal_longitudinal_coordinate, dtype=np.float64
        ).reshape(-1)
        if len(coordinate) != hippocampal.shape[1] or np.any(
            (coordinate < 0.0) | (coordinate > 1.0)
        ) or not np.all(np.isfinite(coordinate)):
            raise ValueError("invalid hippocampal longitudinal coordinate")
        phase = 2.0 * np.pi * coordinate
        wave_basis = np.vstack((np.cos(phase), np.sin(phase)))
    else:
        wave_basis = np.asarray(hippocampal_wave_basis, dtype=np.float64)
        if wave_basis.shape != (2, hippocampal.shape[1]) or not np.all(
            np.isfinite(wave_basis)
        ):
            raise ValueError("hippocampal wave basis must be finite with shape (2, n)")
        if np.any(np.linalg.norm(wave_basis, axis=0) > 1.0 + 1e-12):
            raise ValueError("area-averaged hippocampal wave basis lies outside the unit disk")
    h_weighted = hippocampal * np.sqrt(h_weights)[None, :]
    c_weighted = cortical * np.sqrt(c_weights)[None, :]
    h_covariance = h_weighted @ h_weighted.T
    c_covariance = c_weighted @ c_weighted.T
    coherent = hippocampal @ h_weights
    cosine = hippocampal @ (h_weights * wave_basis[0])
    sine = hippocampal @ (h_weights * wave_basis[1])
    wave_covariance = np.outer(cosine, cosine) + np.outer(sine, sine)
    return SensorOperators(
        hippocampal_covariance=0.5 * (h_covariance + h_covariance.T),
        cortical_covariance=0.5 * (c_covariance + c_covariance.T),
        hippocampal_wave_covariance=0.5 * (
            wave_covariance + wave_covariance.T
        ),
        hippocampal_coherent_topography=coherent,
    )


def _unit_trace(matrix: FloatArray) -> FloatArray:
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 0.0:
        raise ValueError("sensor covariance must have positive finite trace")
    return matrix / trace


def standardized_recoverability_spectrum(
    operators: SensorOperators,
    reference_hippocampal_trace: float,
    reference_cortical_trace: float,
    noise_fraction: float,
) -> FloatArray:
    if reference_hippocampal_trace <= 0.0 or reference_cortical_trace <= 0.0:
        raise ValueError("reference covariance traces must be positive")
    if noise_fraction <= 0.0 or not np.isfinite(noise_fraction):
        raise ValueError("noise_fraction must be finite and positive")
    sensors = operators.hippocampal_covariance.shape[0]
    contrasts = helmert_reference(sensors)
    signal = (
        contrasts
        @ operators.hippocampal_covariance
        @ contrasts.T
        / reference_hippocampal_trace
    )
    nuisance = (
        contrasts
        @ operators.cortical_covariance
        @ contrasts.T
        / reference_cortical_trace
    )
    nuisance += noise_fraction * np.eye(sensors - 1) / (sensors - 1)
    values = eigh(
        0.5 * (signal + signal.T),
        0.5 * (nuisance + nuisance.T),
        eigvals_only=True,
        check_finite=True,
    )[::-1]
    scale = max(float(values[0]), np.finfo(np.float64).tiny)
    if float(values[-1]) < -1e-10 * scale:
        raise FloatingPointError("recoverability spectrum has a negative mode")
    return np.maximum(values, 0.0)


def summarize_condition(
    operators: SensorOperators,
    reference_hippocampal_trace: float,
    reference_cortical_trace: float,
    noise_fraction: float,
) -> tuple[dict[str, float], FloatArray]:
    h_trace = float(np.trace(operators.hippocampal_covariance))
    c_trace = float(np.trace(operators.cortical_covariance))
    wave_trace = float(np.trace(operators.hippocampal_wave_covariance))
    spectrum = standardized_recoverability_spectrum(
        operators,
        reference_hippocampal_trace,
        reference_cortical_trace,
        noise_fraction,
    )
    return {
        "hippocampal_covariance_trace": h_trace,
        "cortical_covariance_trace": c_trace,
        "hippocampal_wave_covariance_trace": wave_trace,
        "hippocampal_coherent_topography_norm": float(
            np.linalg.norm(operators.hippocampal_coherent_topography)
        ),
        "recoverability_spectrum_sum": float(spectrum.sum()),
        "recoverability_spectrum_largest": float(spectrum[0]),
        "recoverability_numerical_rank": float(
            np.count_nonzero(spectrum > 1e-10 * max(float(spectrum[0]), 1.0))
        ),
    }, spectrum


def _cosine(left: FloatArray, right: FloatArray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        raise ValueError("cannot compare a zero topography")
    return float(np.dot(left, right) / denominator)


def compare_conditions(
    variant: SensorOperators,
    reference: SensorOperators,
    variant_spectrum: FloatArray,
    reference_spectrum: FloatArray,
) -> dict[str, float]:
    h_trace = float(np.trace(variant.hippocampal_covariance))
    h_reference_trace = float(np.trace(reference.hippocampal_covariance))
    c_trace = float(np.trace(variant.cortical_covariance))
    c_reference_trace = float(np.trace(reference.cortical_covariance))
    wave_trace = float(np.trace(variant.hippocampal_wave_covariance))
    wave_reference_trace = float(np.trace(reference.hippocampal_wave_covariance))
    spectrum_scale = max(
        float(np.linalg.norm(reference_spectrum)), np.finfo(np.float64).tiny
    )
    return {
        "hippocampal_covariance_shape_distance": float(
            np.linalg.norm(
                _unit_trace(variant.hippocampal_covariance)
                - _unit_trace(reference.hippocampal_covariance),
                ord="fro",
            )
        ),
        "cortical_covariance_shape_distance": float(
            np.linalg.norm(
                _unit_trace(variant.cortical_covariance)
                - _unit_trace(reference.cortical_covariance),
                ord="fro",
            )
        ),
        "hippocampal_wave_shape_distance": float(
            np.linalg.norm(
                _unit_trace(variant.hippocampal_wave_covariance)
                - _unit_trace(reference.hippocampal_wave_covariance),
                ord="fro",
            )
        ),
        "hippocampal_coherent_cosine": _cosine(
            variant.hippocampal_coherent_topography,
            reference.hippocampal_coherent_topography,
        ),
        "hippocampal_amplitude_ratio": h_trace / h_reference_trace,
        "cortical_amplitude_ratio": c_trace / c_reference_trace,
        "hippocampal_wave_amplitude_ratio": wave_trace / wave_reference_trace,
        "recoverability_spectrum_relative_error": float(
            np.linalg.norm(variant_spectrum - reference_spectrum) / spectrum_scale
        ),
    }


def archive_arrays(
    operators: SensorOperators, spectrum: FloatArray
) -> dict[str, FloatArray]:
    return {
        "hippocampal_covariance": operators.hippocampal_covariance,
        "cortical_covariance": operators.cortical_covariance,
        "hippocampal_wave_covariance": operators.hippocampal_wave_covariance,
        "hippocampal_coherent_topography": operators.hippocampal_coherent_topography,
        "recoverability_spectrum": np.asarray(spectrum, dtype=np.float64),
    }
