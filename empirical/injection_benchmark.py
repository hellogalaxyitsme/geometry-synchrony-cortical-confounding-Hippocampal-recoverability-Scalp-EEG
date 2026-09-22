"""Corrected calibration and design utilities for injection-benchmark.

The v2 protocol treats the empirical event-to-baseline ratio as a *total*
energy ratio.  Only the excess above one is injected as signal energy.  It
also supplies deterministic anatomy derangements, balanced source schedules,
and patient-equal held-out threshold calibration.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def added_signal_ratio_from_total_energy_ratio(total_ratio: float) -> float:
    """Convert event/baseline total energy to added-signal/baseline energy."""

    value = float(total_ratio)
    if not np.isfinite(value) or value <= 1.0:
        raise ValueError("event-to-baseline total energy ratio must exceed one")
    result = value - 1.0
    if result <= 0.0:
        raise FloatingPointError("added signal energy is not positive")
    return result


def stochastic_factor_injection_covariance(
    baseline_covariance: ArrayLike,
    sensor_factor: ArrayLike,
    added_signal_energy_ratio: float,
) -> tuple[FloatArray, FloatArray]:
    """Add a rank-r factor with an exact declared trace-energy increment.

    ``added_signal_energy_ratio`` is signal trace divided by baseline trace,
    not the total event-to-baseline ratio.  Columns of ``sensor_factor`` may
    span the cosine/sine subspace of a traveling or random-phase source.
    """

    baseline = np.asarray(baseline_covariance, dtype=np.float64)
    factor = np.asarray(sensor_factor, dtype=np.float64)
    ratio = float(added_signal_energy_ratio)
    if baseline.ndim != 2 or baseline.shape[0] != baseline.shape[1]:
        raise ValueError("baseline covariance must be square")
    if factor.ndim == 1:
        factor = factor[:, None]
    if (
        factor.ndim != 2
        or factor.shape[0] != baseline.shape[0]
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(factor))
        or not np.isfinite(ratio)
        or ratio <= 0.0
    ):
        raise ValueError("invalid stochastic factor injection inputs")
    baseline_trace = float(np.trace(baseline))
    factor_energy = float(np.sum(factor * factor))
    if baseline_trace <= 0.0 or factor_energy <= 0.0:
        raise ValueError("baseline and source factor must have positive energy")
    normalized = factor / np.sqrt(factor_energy)
    excess = ratio * baseline_trace * (normalized @ normalized.T)
    excess = np.asarray(0.5 * (excess + excess.T), dtype=np.float64)
    injected = np.asarray(0.5 * (baseline + excess + (baseline + excess).T), dtype=np.float64)
    realized = float(np.trace(excess) / baseline_trace)
    if not np.isclose(realized, ratio, rtol=1e-12, atol=1e-12):
        raise FloatingPointError("injected signal trace does not match its declaration")
    return injected, excess


def best_channel_factor_scores(
    baseline_covariance: ArrayLike,
    response_covariance: ArrayLike,
    injection_covariance: ArrayLike,
    signal_excess_covariance: ArrayLike,
) -> tuple[float, float, float, int]:
    """Score a channel selected only from injected excess versus baseline."""

    baseline = np.asarray(baseline_covariance, dtype=np.float64)
    response = np.asarray(response_covariance, dtype=np.float64)
    injected = np.asarray(injection_covariance, dtype=np.float64)
    excess = np.asarray(signal_excess_covariance, dtype=np.float64)
    if any(value.shape != baseline.shape for value in (response, injected, excess)):
        raise ValueError("channel-score covariances differ in shape")
    noise = np.maximum(np.diag(baseline), np.finfo(float).tiny)
    channel = int(np.argmax(np.maximum(np.diag(excess), 0.0) / noise))

    def score(covariance: FloatArray) -> float:
        return float(np.log(max(float(covariance[channel, channel]), np.finfo(float).tiny)))

    base = score(baseline)
    return base, score(response), score(injected), channel


def ica_factor_scores(
    unmixing: ArrayLike,
    baseline_covariance: ArrayLike,
    response_covariance: ArrayLike,
    injection_covariance: ArrayLike,
    signal_excess_covariance: ArrayLike,
) -> tuple[float, float, float, int]:
    """Score an ICA component selected from injected excess, never response."""

    matrix = np.asarray(unmixing, dtype=np.float64)
    baseline = np.asarray(baseline_covariance, dtype=np.float64)
    response = np.asarray(response_covariance, dtype=np.float64)
    injected = np.asarray(injection_covariance, dtype=np.float64)
    excess = np.asarray(signal_excess_covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != baseline.shape[0]:
        raise ValueError("ICA unmixing and covariance dimensions disagree")
    noise = np.einsum("is,st,it->i", matrix, baseline, matrix, optimize=True)
    signal = np.einsum("is,st,it->i", matrix, excess, matrix, optimize=True)
    component = int(np.argmax(signal / np.maximum(noise, np.finfo(float).tiny)))
    vector = matrix[component]

    def score(covariance: FloatArray) -> float:
        power = float(vector @ covariance @ vector)
        return float(np.log(max(power, np.finfo(float).tiny)))

    base = score(baseline)
    return base, score(response), score(injected), component


def anatomy_derangement(subjects: Sequence[str], offset: int) -> dict[str, str]:
    """Return a balanced cyclic detector-to-source derangement."""

    ordered = tuple(str(value) for value in subjects)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("subjects must be unique and contain at least two entries")
    shift = int(offset) % len(ordered)
    if shift == 0:
        raise ValueError("derangement offset cannot be zero modulo cohort size")
    mapping = {subject: ordered[(index + shift) % len(ordered)] for index, subject in enumerate(ordered)}
    if set(mapping) != set(mapping.values()) or any(key == value for key, value in mapping.items()):
        raise AssertionError("cyclic mapping is not a complete derangement")
    return mapping


def _seed(master_seed: int, *components: object) -> int:
    payload = "|".join((str(int(master_seed)), *(str(value) for value in components)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def balanced_source_schedule(
    observations: int,
    source_count: int,
    master_seed: int,
    *seed_components: object,
) -> NDArray[np.int64]:
    """Generate reproducible randomized blocks with count imbalance at most one."""

    if observations < 1 or source_count < 2:
        raise ValueError("schedule requires observations and at least two sources")
    generator = np.random.default_rng(_seed(master_seed, *seed_components))
    blocks = []
    remaining = int(observations)
    while remaining:
        permutation = generator.permutation(source_count)
        take = min(remaining, source_count)
        blocks.append(permutation[:take])
        remaining -= take
    schedule = np.concatenate(blocks).astype(np.int64, copy=False)
    counts = np.bincount(schedule, minlength=source_count)
    if int(np.max(counts) - np.min(counts)) > 1:
        raise AssertionError("balanced source schedule is imbalanced")
    return schedule


def patient_equal_sensitivity(scores: FloatArray, patients: np.ndarray, threshold: float) -> float:
    unique = np.unique(patients)
    return float(np.mean([np.mean(scores[patients == patient] > threshold) for patient in unique]))


def patient_equal_injection_threshold(
    values: ArrayLike,
    patient_labels: Iterable[object],
    sensitivity_target: float,
) -> float:
    """Highest threshold meeting target mean sensitivity across patients."""

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    patients = np.asarray([str(value) for value in patient_labels], dtype=str).reshape(-1)
    target = float(sensitivity_target)
    if (
        len(scores) == 0
        or len(scores) != len(patients)
        or len(np.unique(patients)) < 2
        or not np.all(np.isfinite(scores))
        or not 0.0 < target < 1.0
    ):
        raise ValueError("invalid patient-equal threshold inputs")
    candidates = np.nextafter(np.unique(scores), -np.inf)
    feasible = [
        float(value)
        for value in candidates
        if patient_equal_sensitivity(scores, patients, float(value)) >= target - 1e-15
    ]
    if not feasible:
        raise FloatingPointError("no threshold attains the requested sensitivity")
    threshold = max(feasible)
    if patient_equal_sensitivity(scores, patients, threshold) < target - 1e-15:
        raise AssertionError("selected threshold violates its sensitivity constraint")
    return threshold


def complete_primary_method_metrics(
    population_rows: Sequence[dict[str, object]],
    methods: Sequence[str],
    endpoint: str,
    required_metrics: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Construct a complete method-keyed headline or fail closed."""

    result: dict[str, dict[str, float]] = {}
    expected_metrics = set(str(value) for value in required_metrics)
    for method in methods:
        selected = {
            str(row["metric"]): float(row["median"])
            for row in population_rows
            if str(row["endpoint"]) == str(endpoint) and str(row["method"]) == str(method)
        }
        if set(selected) != expected_metrics:
            raise ValueError(f"headline metrics are incomplete for method {method}")
        result[str(method)] = selected
    if set(result) != set(str(value) for value in methods):
        raise AssertionError("headline method coverage is incomplete")
    return result
