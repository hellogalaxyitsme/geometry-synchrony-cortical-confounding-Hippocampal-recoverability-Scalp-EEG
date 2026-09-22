"""Leakage-resistant primitives for Koessler/Ternisien external validation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from empirical.estimators import fit_fastica, ica_event_features, roc_auc, select_signed_feature


ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalize_channel(name: str) -> str:
    value = re.sub(r"\s+", "", str(name)).upper()
    if value.startswith("S") and len(value) > 1:
        value = value[1:]
    value = ALIASES.get(value, value)
    return value


def load_cell_epochs(
    path: Path, variable: str, samples: int = 512, *, normalize_scalp_names: bool = True
) -> tuple[list[str], np.ndarray, dict[str, object]]:
    payload = loadmat(path, squeeze_me=False, struct_as_record=True)
    if set(name for name in payload if not name.startswith("__")) != {variable}:
        raise ValueError(f"{path} does not contain exactly {variable}")
    cells = payload[variable]
    if cells.ndim != 2 or cells.shape[0] < 4 or cells.shape[1] < 2 or cells.dtype != object:
        raise ValueError(f"invalid cell matrix: {path}")
    names = [str(np.asarray(cells[0, column]).reshape(-1)[0]) for column in range(cells.shape[1])]
    output_names = [normalize_channel(name) for name in names] if normalize_scalp_names else names
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"channel aliases collide in {path}")
    trial_rows = []
    auxiliary_lengths: dict[int, int] = {}
    for row in range(1, cells.shape[0]):
        lengths = {int(np.asarray(cells[row, column]).size) for column in range(cells.shape[1])}
        if len(lengths) != 1:
            raise ValueError(f"row {row} has inconsistent channel lengths")
        length = next(iter(lengths))
        if length == samples:
            trial_rows.append(row)
        else:
            auxiliary_lengths[length] = auxiliary_lengths.get(length, 0) + 1
    if not trial_rows:
        raise ValueError(f"no {samples}-sample epochs in {path}")
    epochs = np.empty((len(trial_rows), len(names), samples), dtype=np.float64)
    for event, row in enumerate(trial_rows):
        for channel in range(len(names)):
            epochs[event, channel] = np.asarray(cells[row, channel], dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(epochs)):
        raise ValueError(f"non-finite epochs in {path}")
    return output_names, epochs, {
        "cell_shape": list(cells.shape), "trial_rows": len(trial_rows),
        "auxiliary_row_lengths": {str(key): value for key, value in sorted(auxiliary_lengths.items())},
    }


def preprocess_scalp(
    epochs: np.ndarray, sampling_frequency: float, band_hz: tuple[float, float], order: int,
    artifact_peak_uv: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("scalp epochs must be event by channel by time")
    keep = np.max(np.abs(values), axis=(1, 2)) <= artifact_peak_uv
    if np.count_nonzero(keep) < 20:
        raise ValueError("artifact rule leaves fewer than 20 events")
    values = values[keep]
    sos = butter(order, band_hz, btype="bandpass", fs=sampling_frequency, output="sos")
    values = sosfiltfilt(sos, values, axis=-1)
    values -= values.mean(axis=1, keepdims=True)
    return np.asarray(values, dtype=np.float64), keep


def window(center: int, half_width: int) -> slice:
    return slice(center - half_width, center + half_width + 1)


def spatial_scores(epochs: np.ndarray, spatial_filter: np.ndarray, interval: slice) -> np.ndarray:
    projected = np.einsum("ect,c->et", epochs[:, :, interval], spatial_filter)
    return np.max(np.abs(projected), axis=1)


def fit_matched_filter(
    training: np.ndarray, event_interval: slice, baseline_intervals: tuple[slice, ...], shrinkage: float,
) -> tuple[np.ndarray, dict[str, float]]:
    event_average = training[:, :, event_interval].mean(axis=0)
    event_average -= event_average.mean(axis=0, keepdims=True)
    peak = int(np.argmax(np.sqrt(np.mean(event_average * event_average, axis=0))))
    template = event_average[:, peak]
    baseline = np.concatenate([training[:, :, interval] for interval in baseline_intervals], axis=2)
    samples = baseline.transpose(0, 2, 1).reshape(-1, baseline.shape[1])
    samples -= samples.mean(axis=0, keepdims=True)
    covariance = samples.T @ samples / max(len(samples) - 1, 1)
    ridge = shrinkage * float(np.trace(covariance)) / len(covariance)
    covariance += ridge * np.eye(len(covariance))
    spatial_filter = np.linalg.solve(covariance, template)
    norm = float(np.sqrt(spatial_filter @ covariance @ spatial_filter))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("matched filter has zero norm")
    spatial_filter /= norm
    return spatial_filter, {"training_peak_offset_samples": peak, "ridge": ridge}


def paired_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    scores = np.concatenate((positive, negative)); labels = np.concatenate((np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)))
    return roc_auc(labels, scores)


def fastica_auc(
    training: np.ndarray, testing: np.ndarray, event_interval: slice, baseline_interval: slice, seed: int,
) -> tuple[float, bool, int]:
    samples = training.transpose(0, 2, 1).reshape(-1, training.shape[1])
    components = min(training.shape[1] - 1, 12)
    model = fit_fastica(samples, components, seed=seed, maximum_iterations=1000, tolerance=1e-7)
    train_event = ica_event_features(model, training[:, :, event_interval])
    train_base = ica_event_features(model, training[:, :, baseline_interval])
    features = np.vstack((train_event, train_base)); labels = np.concatenate((np.ones(len(training), dtype=int), np.zeros(len(training), dtype=int)))
    component, sign, _ = select_signed_feature(features, labels)
    test_event = ica_event_features(model, testing[:, :, event_interval])[:, component]
    test_base = ica_event_features(model, testing[:, :, baseline_interval])[:, component]
    return paired_auc(sign * test_event, sign * test_base), model.converged, model.iterations


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort"); result = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]: end += 1
            result[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return result
    x = ranks(np.asarray(left, dtype=float)); y = ranks(np.asarray(right, dtype=float))
    if np.std(x) == 0.0 or np.std(y) == 0.0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
