"""Patient-held-out contiguous-block validation utilities for temporal-transfer."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from empirical.working_memory import (
    bootstrap_macro_auc,
    fit_shrinkage_lda,
    predict_linear,
    roc_auc,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


def contiguous_block_ids(length: int, count: int) -> IntArray:
    if length < count or count < 2:
        raise ValueError("contiguous blocks require at least one row per block")
    result = np.empty(length, dtype=np.int64)
    for block, indices in enumerate(np.array_split(np.arange(length), count)):
        result[indices] = block
    return result


def fit_early_calibration(features: ArrayLike, blocks: ArrayLike) -> tuple[FloatArray, FloatArray]:
    matrix = np.asarray(features, dtype=np.float64)
    block = np.asarray(blocks, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(block) or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid calibration inputs")
    early = matrix[block == 0]
    if len(early) < 2:
        raise ValueError("early calibration block is too small")
    center = np.median(early, axis=0)
    mad = np.median(np.abs(early - center), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(early, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def apply_calibration(
    features: ArrayLike, center: ArrayLike, scale: ArrayLike
) -> FloatArray:
    matrix = np.asarray(features, dtype=np.float64)
    location = np.asarray(center, dtype=np.float64).reshape(-1)
    spread = np.asarray(scale, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != len(location) or len(location) != len(spread):
        raise ValueError("calibration dimensions disagree")
    if np.any(spread <= 0.0) or not np.all(np.isfinite(matrix)):
        raise ValueError("calibration contains invalid values")
    return np.asarray((matrix - location) / spread, dtype=np.float64)


def _mask(
    patients: StringArray,
    blocks: IntArray,
    included_patients: Sequence[str],
    included_blocks: Sequence[int],
) -> NDArray[np.bool_]:
    return np.isin(patients, included_patients) & np.isin(blocks, included_blocks)


def choose_transfer_shrinkage(
    features: FloatArray,
    labels: IntArray,
    patients: StringArray,
    blocks: IntArray,
    outer_training_patients: Sequence[str],
    training_blocks: Sequence[int],
    test_blocks: Sequence[int],
    grid: Iterable[float],
) -> tuple[float, list[dict[str, Any]]]:
    cohort = sorted(str(value) for value in outer_training_patients)
    if len(cohort) < 3:
        raise ValueError("nested transfer needs at least three training patients")
    rows: list[dict[str, Any]] = []
    for shrinkage in sorted(float(value) for value in grid):
        inner_rows = []
        for heldout in cohort:
            inner_training = [patient for patient in cohort if patient != heldout]
            train = _mask(patients, blocks, inner_training, training_blocks)
            test = _mask(patients, blocks, [heldout], test_blocks)
            model = fit_shrinkage_lda(features[train], labels[train], shrinkage)
            score = predict_linear(features[test], model)
            inner_rows.append(
                {
                    "heldout_patient": heldout,
                    "train_events": int(np.count_nonzero(train)),
                    "test_events": int(np.count_nonzero(test)),
                    "auc": roc_auc(labels[test], score),
                }
            )
        rows.append(
            {
                "shrinkage": shrinkage,
                "macro_inner_patient_auc": float(np.mean([row["auc"] for row in inner_rows])),
                "inner_folds": inner_rows,
            }
        )
    chosen = max(rows, key=lambda row: (row["macro_inner_patient_auc"], row["shrinkage"]))
    return float(chosen["shrinkage"]), rows


def outer_transfer_lda(
    features: ArrayLike,
    labels: ArrayLike,
    patients: ArrayLike,
    blocks: ArrayLike,
    training_blocks: Sequence[int],
    test_blocks: Sequence[int],
    shrinkage_grid: Iterable[float],
) -> tuple[FloatArray, NDArray[np.bool_], list[dict[str, Any]]]:
    matrix = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(patients, dtype=str).reshape(-1)
    block = np.asarray(blocks, dtype=np.int64).reshape(-1)
    if not (len(matrix) == len(truth) == len(groups) == len(block)):
        raise ValueError("transfer arrays differ in length")
    evaluated = np.isin(block, test_blocks)
    scores = np.full(len(truth), np.nan, dtype=np.float64)
    folds = []
    cohort = sorted(np.unique(groups).tolist())
    for heldout in cohort:
        training_patients = [patient for patient in cohort if patient != heldout]
        chosen, inner = choose_transfer_shrinkage(
            matrix,
            truth,
            groups,
            block,
            training_patients,
            training_blocks,
            test_blocks,
            shrinkage_grid,
        )
        train = _mask(groups, block, training_patients, training_blocks)
        test = _mask(groups, block, [heldout], test_blocks)
        model = fit_shrinkage_lda(matrix[train], truth[train], chosen)
        scores[test] = predict_linear(matrix[test], model)
        folds.append(
            {
                "heldout_patient": heldout,
                "training_patients": training_patients,
                "training_blocks": list(int(value) for value in training_blocks),
                "test_blocks": list(int(value) for value in test_blocks),
                "train_events": int(np.count_nonzero(train)),
                "test_events": int(np.count_nonzero(test)),
                "chosen_shrinkage": chosen,
                "inner_grid": inner,
                "heldout_auc": roc_auc(truth[test], scores[test]),
            }
        )
    if not np.all(np.isfinite(scores[evaluated])) or np.any(np.isfinite(scores[~evaluated])):
        raise AssertionError("transfer scores do not match the declared test blocks")
    return scores, evaluated, folds


def outer_transfer_best_channel(
    features: ArrayLike,
    labels: ArrayLike,
    patients: ArrayLike,
    blocks: ArrayLike,
    training_blocks: Sequence[int],
    test_blocks: Sequence[int],
) -> tuple[FloatArray, NDArray[np.bool_], list[dict[str, Any]]]:
    matrix = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(patients, dtype=str).reshape(-1)
    block = np.asarray(blocks, dtype=np.int64).reshape(-1)
    evaluated = np.isin(block, test_blocks)
    scores = np.full(len(truth), np.nan, dtype=np.float64)
    folds = []
    cohort = sorted(np.unique(groups).tolist())
    for heldout in cohort:
        training_patients = [patient for patient in cohort if patient != heldout]
        candidates = []
        for channel in range(matrix.shape[1]):
            inner_aucs = []
            for inner_heldout in training_patients:
                inner_training = [
                    patient for patient in training_patients if patient != inner_heldout
                ]
                train = _mask(groups, block, inner_training, training_blocks)
                test = _mask(groups, block, [inner_heldout], test_blocks)
                high_mean = float(matrix[train & (truth == 1), channel].mean())
                low_mean = float(matrix[train & (truth == 0), channel].mean())
                sign = 1.0 if high_mean >= low_mean else -1.0
                inner_aucs.append(roc_auc(truth[test], sign * matrix[test, channel]))
            candidates.append((float(np.mean(inner_aucs)), channel))
        inner_auc, selected_channel = max(candidates, key=lambda item: (item[0], -item[1]))
        train = _mask(groups, block, training_patients, training_blocks)
        test = _mask(groups, block, [heldout], test_blocks)
        sign = (
            1.0
            if matrix[train & (truth == 1), selected_channel].mean()
            >= matrix[train & (truth == 0), selected_channel].mean()
            else -1.0
        )
        scores[test] = sign * matrix[test, selected_channel]
        folds.append(
            {
                "heldout_patient": heldout,
                "training_patients": training_patients,
                "training_blocks": list(int(value) for value in training_blocks),
                "test_blocks": list(int(value) for value in test_blocks),
                "selected_feature_index": int(selected_channel),
                "selected_sign": float(sign),
                "macro_inner_patient_auc": inner_auc,
                "train_events": int(np.count_nonzero(train)),
                "test_events": int(np.count_nonzero(test)),
                "heldout_auc": roc_auc(truth[test], scores[test]),
            }
        )
    return scores, evaluated, folds


def summarize_transfer(
    labels: IntArray,
    patients: StringArray,
    scores: FloatArray,
    evaluated: NDArray[np.bool_],
    folds: list[dict[str, Any]],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    patient_rows = []
    for patient in np.unique(patients):
        selected = evaluated & (patients == patient)
        patient_rows.append(
            {
                "patient": str(patient),
                "events": int(np.count_nonzero(selected)),
                "low_events": int(np.count_nonzero(selected & (labels == 0))),
                "high_events": int(np.count_nonzero(selected & (labels == 1))),
                "auc": roc_auc(labels[selected], scores[selected]),
            }
        )
    return {
        "macro_patient_auc": bootstrap_macro_auc(
            [row["auc"] for row in patient_rows], bootstrap_replicates, seed
        ),
        "pooled_event_auc": roc_auc(labels[evaluated], scores[evaluated]),
        "evaluated_events": int(np.count_nonzero(evaluated)),
        "patients": patient_rows,
        "folds": folds,
    }


def shift_labels_within_session_blocks(
    labels: ArrayLike,
    sessions: ArrayLike,
    blocks: ArrayLike,
    generator: np.random.Generator,
) -> IntArray:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    session = np.asarray(sessions, dtype=str).reshape(-1)
    block = np.asarray(blocks, dtype=np.int64).reshape(-1)
    shifted = truth.copy()
    for session_id in np.unique(session):
        for block_id in np.unique(block[session == session_id]):
            indices = np.flatnonzero((session == session_id) & (block == block_id))
            if len(indices) > 1:
                shifted[indices] = np.roll(truth[indices], int(generator.integers(1, len(indices))))
    return shifted
