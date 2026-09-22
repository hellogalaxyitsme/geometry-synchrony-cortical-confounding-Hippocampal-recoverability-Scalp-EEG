"""Leakage-safe utilities for the ds004752 working-memory experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, sosfiltfilt
from scipy.stats import rankdata, spearmanr


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _edf_text(block: bytes) -> str:
    return block.decode("ascii", errors="replace").strip()


def read_edf_selected(path: Path, requested: Sequence[str]) -> tuple[FloatArray, float]:
    """Read selected EDF channels without interpreting nonessential date/time fields."""
    names_requested = list(requested)
    if not names_requested or len(names_requested) != len(set(names_requested)):
        raise ValueError("requested EDF channels must be nonempty and unique")
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError("truncated EDF fixed header")
        header_bytes = int(_edf_text(fixed[184:192]))
        records = int(_edf_text(fixed[236:244]))
        duration = float(_edf_text(fixed[244:252]))
        signals = int(_edf_text(fixed[252:256]))
        variable = stream.read(header_bytes - 256)
    if records <= 0 or duration <= 0.0 or signals <= 0:
        raise ValueError("invalid EDF dimensions")
    if len(variable) != 256 * signals:
        raise ValueError("truncated EDF variable header")
    offset = 0

    def fields(width: int) -> list[str]:
        nonlocal offset
        values = [
            _edf_text(variable[offset + index * width : offset + (index + 1) * width])
            for index in range(signals)
        ]
        offset += width * signals
        return values

    labels = fields(16)
    fields(80)
    fields(8)
    physical_minimum = np.asarray([float(value) for value in fields(8)])
    physical_maximum = np.asarray([float(value) for value in fields(8)])
    digital_minimum = np.asarray([float(value) for value in fields(8)])
    digital_maximum = np.asarray([float(value) for value in fields(8)])
    fields(80)
    samples = np.asarray([int(value) for value in fields(8)], dtype=np.int64)
    fields(32)
    if offset != len(variable) or len(labels) != len(set(labels)):
        raise ValueError("invalid or duplicate EDF signal header")
    missing = sorted(set(names_requested) - set(labels))
    if missing:
        raise ValueError(f"requested EDF channels are missing: {missing}")
    indices = np.asarray([labels.index(name) for name in names_requested], dtype=np.int64)
    requested_samples = samples[indices]
    if np.any(requested_samples != requested_samples[0]):
        raise ValueError("selected EDF channels have unequal sampling frequencies")
    words_per_record = int(np.sum(samples))
    expected_words = records * words_per_record
    expected_bytes = header_bytes + 2 * expected_words
    if path.stat().st_size < expected_bytes:
        raise ValueError("EDF payload is shorter than its declared records")
    raw = np.memmap(path, mode="r", dtype="<i2", offset=header_bytes, shape=(expected_words,))
    records_by_word = raw.reshape(records, words_per_record)
    channel_offsets = np.concatenate(([0], np.cumsum(samples[:-1]))).astype(np.int64)
    result = np.empty((len(indices), records * int(requested_samples[0])), dtype=np.float64)
    for output_index, signal_index in enumerate(indices):
        left = int(channel_offsets[signal_index])
        right = left + int(samples[signal_index])
        digital = np.asarray(records_by_word[:, left:right], dtype=np.float64).reshape(-1)
        denominator = digital_maximum[signal_index] - digital_minimum[signal_index]
        if denominator == 0.0:
            raise ValueError("EDF digital calibration range is zero")
        scale = (physical_maximum[signal_index] - physical_minimum[signal_index]) / denominator
        result[output_index] = (digital - digital_minimum[signal_index]) * scale + physical_minimum[
            signal_index
        ]
    return result, float(requested_samples[0] / duration)


def robust_zscore(values: ArrayLike, axis: int = 0) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("robust_zscore requires a nonempty finite array")
    median = np.median(matrix, axis=axis, keepdims=True)
    mad = np.median(np.abs(matrix - median), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    standard = np.std(matrix, axis=axis, ddof=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, standard)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.asarray((matrix - median) / scale, dtype=np.float64)


def repeat_string(value: str, count: int) -> NDArray[np.str_]:
    if not value or count < 0:
        raise ValueError("invalid repeated string")
    return np.repeat(np.asarray([value], dtype=f"<U{len(value)}"), count)


def _sos(band: Sequence[float], sampling_frequency: float, order: int) -> FloatArray:
    low, high = (float(value) for value in band)
    if not (0.0 < low < high < 0.5 * sampling_frequency):
        raise ValueError("invalid band-pass frequencies")
    return np.asarray(
        butter(order, (low, high), btype="bandpass", fs=sampling_frequency, output="sos"),
        dtype=np.float64,
    )


def filtered_power(
    data: ArrayLike,
    sampling_frequency: float,
    band: Sequence[float],
    order: int,
) -> FloatArray:
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 16 or not np.all(np.isfinite(matrix)):
        raise ValueError("filtered_power expects finite channel-by-time data")
    filtered = sosfiltfilt(_sos(band, sampling_frequency, order), matrix, axis=1)
    return np.asarray(np.mean(filtered * filtered, axis=1), dtype=np.float64)


def continuous_bandpasses(
    data: ArrayLike,
    sampling_frequency: float,
    bands: Sequence[Sequence[float]],
    order: int,
) -> list[FloatArray]:
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 16 or not np.all(np.isfinite(matrix)):
        raise ValueError("continuous_bandpasses expects finite channel-by-time data")
    return [
        np.asarray(
            sosfiltfilt(_sos(band, sampling_frequency, order), matrix, axis=1),
            dtype=np.float64,
        )
        for band in bands
    ]


def window_log_power(filtered: ArrayLike, starts: ArrayLike, stops: ArrayLike) -> FloatArray:
    matrix = np.asarray(filtered, dtype=np.float64)
    begin = np.asarray(starts, dtype=np.int64).reshape(-1)
    end = np.asarray(stops, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or len(begin) != len(end):
        raise ValueError("invalid filtered data or windows")
    if np.any(begin < 0) or np.any(end > matrix.shape[1]) or np.any(end <= begin):
        raise ValueError("analysis window exceeds recording bounds")
    result = np.empty((len(begin), matrix.shape[0]), dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    for index, (left, right) in enumerate(zip(begin, end)):
        segment = matrix[:, left:right]
        result[index] = np.log(np.maximum(np.mean(segment * segment, axis=1), tiny))
    return result


def stratified_tail_labels(
    scores: ArrayLike,
    strata: ArrayLike,
    lower_quantile: float,
    upper_quantile: float,
    minimum_per_stratum: int,
) -> IntArray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups = np.asarray(strata).reshape(-1)
    if len(values) != len(groups) or not np.all(np.isfinite(values)):
        raise ValueError("invalid scores or strata")
    if not (0.0 < lower_quantile < upper_quantile < 1.0):
        raise ValueError("invalid tail quantiles")
    labels = np.full(len(values), -1, dtype=np.int64)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < minimum_per_stratum:
            continue
        order = indices[np.lexsort((indices, values[indices]))]
        low_count = int(math.floor(lower_quantile * len(order)))
        high_count = int(math.floor((1.0 - upper_quantile) * len(order)))
        if low_count < 1 or high_count < 1 or low_count + high_count >= len(order):
            continue
        labels[order[:low_count]] = 0
        labels[order[-high_count:]] = 1
    return labels


def roc_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(truth) != len(values) or not np.all(np.isfinite(values)):
        raise ValueError("invalid AUC inputs")
    positives = truth == 1
    negatives = truth == 0
    n_positive = int(np.count_nonzero(positives))
    n_negative = int(np.count_nonzero(negatives))
    if n_positive == 0 or n_negative == 0 or np.any(~(positives | negatives)):
        raise ValueError("AUC requires both binary classes")
    ranks = rankdata(values, method="average")
    statistic = float(np.sum(ranks[positives]) - n_positive * (n_positive + 1) / 2.0)
    return statistic / (n_positive * n_negative)


def fit_shrinkage_lda(features: ArrayLike, labels: ArrayLike, shrinkage: float) -> tuple[FloatArray, float]:
    matrix = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(truth) or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid LDA inputs")
    if not 0.0 <= shrinkage <= 1.0 or set(np.unique(truth)) != {0, 1}:
        raise ValueError("invalid shrinkage or labels")
    low = matrix[truth == 0]
    high = matrix[truth == 1]
    low_mean = low.mean(axis=0)
    high_mean = high.mean(axis=0)
    centered = np.vstack((low - low_mean, high - high_mean))
    covariance = centered.T @ centered / max(len(centered) - 2, 1)
    isotropic = float(np.trace(covariance) / matrix.shape[1])
    regularized = (1.0 - shrinkage) * covariance + shrinkage * isotropic * np.eye(
        matrix.shape[1]
    )
    weight = np.linalg.pinv(regularized, rcond=1e-12) @ (high_mean - low_mean)
    midpoint = 0.5 * (high_mean + low_mean)
    intercept = -float(weight @ midpoint)
    if not np.all(np.isfinite(weight)) or not math.isfinite(intercept):
        raise ValueError("non-finite LDA solution")
    return np.asarray(weight, dtype=np.float64), intercept


def predict_linear(features: ArrayLike, model: tuple[ArrayLike, float]) -> FloatArray:
    matrix = np.asarray(features, dtype=np.float64)
    weight = np.asarray(model[0], dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != len(weight):
        raise ValueError("linear model feature mismatch")
    return np.asarray(matrix @ weight + float(model[1]), dtype=np.float64)


def macro_group_auc(labels: IntArray, scores: FloatArray, groups: NDArray[np.str_]) -> float:
    values = [roc_auc(labels[groups == group], scores[groups == group]) for group in np.unique(groups)]
    return float(np.mean(values))


def choose_shrinkage_nested(
    features: FloatArray,
    labels: IntArray,
    patients: NDArray[np.str_],
    grid: Iterable[float],
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    unique = np.unique(patients)
    if len(unique) < 3:
        raise ValueError("nested selection needs at least three training patients")
    for shrinkage in sorted(float(value) for value in grid):
        aucs: list[float] = []
        for heldout in unique:
            train = patients != heldout
            test = ~train
            model = fit_shrinkage_lda(features[train], labels[train], shrinkage)
            aucs.append(roc_auc(labels[test], predict_linear(features[test], model)))
        rows.append(
            {
                "shrinkage": shrinkage,
                "macro_inner_patient_auc": float(np.mean(aucs)),
            }
        )
    best = max(rows, key=lambda row: (row["macro_inner_patient_auc"], row["shrinkage"]))
    return float(best["shrinkage"]), rows


def outer_patient_lda(
    features: ArrayLike,
    labels: ArrayLike,
    patients: ArrayLike,
    shrinkage_grid: Iterable[float],
) -> tuple[FloatArray, list[dict[str, Any]]]:
    matrix = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(patients, dtype=str).reshape(-1)
    scores = np.full(len(truth), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for heldout in np.unique(groups):
        train = groups != heldout
        test = ~train
        chosen, inner = choose_shrinkage_nested(
            matrix[train], truth[train], groups[train], shrinkage_grid
        )
        model = fit_shrinkage_lda(matrix[train], truth[train], chosen)
        scores[test] = predict_linear(matrix[test], model)
        folds.append(
            {
                "heldout_patient": str(heldout),
                "train_events": int(np.count_nonzero(train)),
                "test_events": int(np.count_nonzero(test)),
                "chosen_shrinkage": chosen,
                "inner_grid": inner,
                "heldout_auc": roc_auc(truth[test], scores[test]),
            }
        )
    if not np.all(np.isfinite(scores)):
        raise AssertionError("outer scores are incomplete")
    return scores, folds


def outer_best_channel(
    features: ArrayLike, labels: ArrayLike, patients: ArrayLike
) -> tuple[FloatArray, list[dict[str, Any]]]:
    matrix = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(patients, dtype=str).reshape(-1)
    scores = np.full(len(truth), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for heldout in np.unique(groups):
        train = groups != heldout
        test = ~train
        candidates: list[tuple[float, int, float]] = []
        for channel in range(matrix.shape[1]):
            sign = 1.0 if matrix[train & (truth == 1), channel].mean() >= matrix[
                train & (truth == 0), channel
            ].mean() else -1.0
            aucs = [
                roc_auc(
                    truth[train & (groups == patient)],
                    sign * matrix[train & (groups == patient), channel],
                )
                for patient in np.unique(groups[train])
            ]
            candidates.append((float(np.mean(aucs)), channel, sign))
        training_auc, channel, sign = max(candidates, key=lambda item: (item[0], -item[1]))
        scores[test] = sign * matrix[test, channel]
        folds.append(
            {
                "heldout_patient": str(heldout),
                "selected_feature_index": int(channel),
                "selected_sign": float(sign),
                "training_macro_patient_auc": training_auc,
                "heldout_auc": roc_auc(truth[test], scores[test]),
            }
        )
    return scores, folds


def permutation_spearman(
    left: ArrayLike, right: ArrayLike, replicates: int, seed: int
) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) < 4 or len(x) != len(y) or not np.all(np.isfinite(x + y)):
        raise ValueError("invalid permutation association inputs")
    observed = float(spearmanr(x, y).statistic)
    generator = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        null[index] = float(spearmanr(generator.permutation(x), y).statistic)
    return {
        "spearman_rho": observed,
        "permutation_replicates": int(replicates),
        "one_sided_positive_p": float((1 + np.count_nonzero(null >= observed)) / (replicates + 1)),
        "two_sided_p": float(
            (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (replicates + 1)
        ),
        "null_mean": float(np.mean(null)),
        "null_standard_deviation": float(np.std(null)),
    }


def bootstrap_macro_auc(
    patient_aucs: ArrayLike, replicates: int, seed: int
) -> dict[str, float]:
    values = np.asarray(patient_aucs, dtype=np.float64).reshape(-1)
    generator = np.random.default_rng(seed)
    samples = np.mean(
        values[generator.integers(0, len(values), size=(replicates, len(values)))], axis=1
    )
    return {
        "mean": float(np.mean(values)),
        "bootstrap_replicates": int(replicates),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
