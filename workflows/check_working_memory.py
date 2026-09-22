#!/usr/bin/env python3
"""Deterministic numerical tests for the working-memory empirical pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.working_memory import (
    atomic_json,
    fit_shrinkage_lda,
    outer_patient_lda,
    permutation_spearman,
    predict_linear,
    read_edf_selected,
    repeat_string,
    robust_zscore,
    roc_auc,
    stratified_tail_labels,
)
from workflows.build_working_memory_epochs import closest_area_patch


PROTOCOL = "openneuro-ds004752/empirical-checks-v1"


def check(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


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

    def robust() -> None:
        result = robust_zscore([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
        check(np.all(result[:, 0] == 0.0), "constant feature was not handled")
        check(abs(float(np.median(result[:, 1]))) < 1e-12, "median is not zero")

    test("robust_scaling_handles_constant_features", robust)
    test(
        "repeated_patient_codes_are_not_truncated",
        lambda: check(
            repeat_string("example-subject", 3).tolist()
            == ["example-subject", "example-subject", "example-subject"],
            "patient identifier was truncated",
        ),
    )

    def edf_reader() -> None:
        def fixed(value: object, width: int) -> bytes:
            return str(value).encode("ascii").ljust(width, b" ")

        labels = ["F3", "F4"]
        fixed_header = b"".join(
            (
                fixed("0", 8), fixed("X", 80), fixed("X", 80), fixed("01.01.01", 8),
                fixed("00.00.60", 8), fixed(768, 8), fixed("EDF+C", 44), fixed(2, 8),
                fixed(1, 8), fixed(2, 4),
            )
        )
        variable = b"".join(
            (
                b"".join(fixed(value, 16) for value in labels),
                b"".join(fixed("", 80) for _ in labels),
                b"".join(fixed("uV", 8) for _ in labels),
                b"".join(fixed(-1, 8) for _ in labels),
                b"".join(fixed(1, 8) for _ in labels),
                b"".join(fixed(-32768, 8) for _ in labels),
                b"".join(fixed(32767, 8) for _ in labels),
                b"".join(fixed("", 80) for _ in labels),
                b"".join(fixed(2, 8) for _ in labels),
                b"".join(fixed("", 32) for _ in labels),
            )
        )
        digital = np.asarray(
            [0, 32767, -32768, 0, 100, -100, -100, 100], dtype="<i2"
        ).tobytes()
        with tempfile.TemporaryDirectory(prefix="working_memory-edf-") as directory:
            path = Path(directory) / "invalid-clock.edf"
            path.write_bytes(fixed_header + variable + digital)
            data, frequency = read_edf_selected(path, ["F4", "F3"])
        check(data.shape == (2, 4), "selected EDF shape is wrong")
        check(frequency == 2.0, "selected EDF sampling frequency is wrong")
        check(data[0, 0] == -1.0, "EDF calibration or requested order is wrong")
        check(data[1, 1] == 1.0, "EDF maximum calibration is wrong")

    test("edf_reader_ignores_invalid_nonessential_clock", edf_reader)

    def labels() -> None:
        result = stratified_tail_labels(
            np.arange(12.0), np.repeat([4, 6], 6), 1 / 3, 2 / 3, 6
        )
        check(np.count_nonzero(result == 0) == 4, "wrong low-tail count")
        check(np.count_nonzero(result == 1) == 4, "wrong high-tail count")
        check(np.count_nonzero(result < 0) == 4, "wrong middle count")

    test("stratified_labels_are_balanced_and_nested", labels)
    test(
        "auc_handles_ties_exactly",
        lambda: check(
            abs(roc_auc([0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0]) - 0.5) < 1e-12,
            "tied AUC is not one half",
        ),
    )

    def lda() -> None:
        x = np.asarray([[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
        y = np.asarray([0, 0, 1, 1])
        scores = predict_linear(x, fit_shrinkage_lda(x, y, 0.1))
        check(roc_auc(y, scores) == 1.0, "LDA failed a separable problem")

    test("shrinkage_lda_separable_case", lda)

    def heldout() -> None:
        generator = np.random.default_rng(9)
        patients = np.repeat([f"p{index}" for index in range(5)], 20)
        labels_value = np.tile(np.repeat([0, 1], 10), 5)
        features = generator.normal(size=(100, 3))
        features[:, 0] += 1.5 * labels_value
        scores, folds = outer_patient_lda(features, labels_value, patients, [0.1, 1.0])
        check(len(folds) == 5, "wrong outer fold count")
        check(roc_auc(labels_value, scores) > 0.75, "held-out signal was not recovered")
        check(np.all(np.isfinite(scores)), "held-out scores are incomplete")

    test("leave_one_patient_out_is_complete", heldout)

    def patch() -> None:
        anterior = np.asarray([0.0, 0.2, 0.8, 1.0, 0.1, 0.9])
        pd = np.zeros(6)
        hemisphere = np.asarray([-1, -1, -1, -1, 1, 1])
        area = np.ones(6)
        selected = closest_area_patch(anterior, pd, hemisphere, area, -1, 0.85, 0.5)
        check(set(selected) == {2, 3}, "AP-centred area patch is incorrect")

    test("forward_patch_respects_hemisphere_and_AP", patch)

    def permutation() -> None:
        first = permutation_spearman(np.arange(8.0), np.arange(8.0), 100, 17)
        second = permutation_spearman(np.arange(8.0), np.arange(8.0), 100, 17)
        check(first == second, "permutation result is not deterministic")
        check(first["spearman_rho"] == 1.0, "perfect association was not recovered")

    test("association_permutation_is_deterministic", permutation)

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
