#!/usr/bin/env python3
"""Deterministic stress tests for the temporal-transfer blocked validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.working_memory import atomic_json, repeat_string, roc_auc
from empirical.temporal_transfer import (
    apply_calibration,
    contiguous_block_ids,
    fit_early_calibration,
    outer_transfer_best_channel,
    outer_transfer_lda,
    shift_labels_within_session_blocks,
)


PROTOCOL = "openneuro-ds004752/patient-blocked-checks-v1"


def check(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(81)
    feature_blocks = []
    label_blocks = []
    patient_blocks = []
    time_blocks = []
    session_blocks = []
    for patient_index in range(5):
        patient = f"p{patient_index}"
        for block in range(3):
            labels = np.repeat([0, 1], 8)
            features = generator.normal(0.0, 0.5, size=(16, 3))
            features[:, 0] += 1.2 * labels
            features[:, 1] += 0.15 * block
            feature_blocks.append(features)
            label_blocks.append(labels)
            patient_blocks.append(repeat_string(patient, len(labels)))
            time_blocks.append(np.full(len(labels), block, dtype=np.int64))
            session_blocks.append(repeat_string(f"{patient}_s1", len(labels)))
    return (
        np.vstack(feature_blocks),
        np.concatenate(label_blocks),
        np.concatenate(patient_blocks),
        np.concatenate(time_blocks),
        np.concatenate(session_blocks),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tests: list[dict[str, object]] = []

    def test(name: str, function) -> None:
        try:
            function()
            tests.append({"name": name, "passed": True, "detail": "ok"})
        except Exception as error:
            tests.append(
                {
                    "name": name,
                    "passed": False,
                    "detail": f"{type(error).__name__}: {error}",
                }
            )

    def blocks() -> None:
        value = contiguous_block_ids(10, 3)
        check(value.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2], "wrong split")
        check(all(np.all(np.diff(np.flatnonzero(value == block)) == 1) for block in range(3)), "not contiguous")

    test("blocks_are_ordered_contiguous_and_exhaustive", blocks)

    def calibration() -> None:
        raw = np.arange(30.0).reshape(10, 3)
        block = contiguous_block_ids(10, 3)
        center, scale = fit_early_calibration(raw, block)
        changed = raw.copy()
        changed[block > 0] += 1e6
        changed_center, changed_scale = fit_early_calibration(changed, block)
        check(np.array_equal(center, changed_center), "future changed calibration center")
        check(np.array_equal(scale, changed_scale), "future changed calibration scale")
        check(np.all(np.isfinite(apply_calibration(raw, center, scale))), "invalid calibrated values")

    test("future_scalp_samples_cannot_change_early_calibration", calibration)

    def transfer() -> None:
        features, labels, patients, block, _ = synthetic()
        scores, evaluated, folds = outer_transfer_lda(
            features, labels, patients, block, (0, 1), (2,), (0.1, 0.5, 1.0)
        )
        check(len(folds) == 5, "wrong outer fold count")
        check(np.all(block[evaluated] == 2), "non-late event evaluated")
        for fold in folds:
            check(fold["heldout_patient"] not in fold["training_patients"], "patient leakage")
            check(fold["training_blocks"] == [0, 1] and fold["test_blocks"] == [2], "block leakage")
        check(roc_auc(labels[evaluated], scores[evaluated]) > 0.8, "transfer signal missed")

    test("prospective_transfer_has_patient_and_block_separation", transfer)

    def baseline() -> None:
        features, labels, patients, block, _ = synthetic()
        scores, evaluated, folds = outer_transfer_best_channel(
            features, labels, patients, block, (0, 1), (2,)
        )
        check(all(row["selected_feature_index"] == 0 for row in folds), "wrong channel")
        check(roc_auc(labels[evaluated], scores[evaluated]) > 0.8, "baseline missed signal")

    test("single_channel_selection_is_nested_by_patient", baseline)

    def shifts() -> None:
        _, labels, _, block, sessions = synthetic()
        shifted = shift_labels_within_session_blocks(
            labels, sessions, block, np.random.default_rng(5)
        )
        check(np.any(shifted != labels), "labels were not shifted")
        for session in np.unique(sessions):
            for value in range(3):
                selected = (sessions == session) & (block == value)
                check(
                    np.array_equal(np.sort(labels[selected]), np.sort(shifted[selected])),
                    "shift changed a session/block class count",
                )

    test("circular_control_preserves_session_block_class_counts", shifts)

    passed = sum(bool(row["passed"]) for row in tests)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": passed == len(tests),
        "tests_passed": passed,
        "tests_total": len(tests),
        "tests": tests,
        "numpy": np.__version__,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
