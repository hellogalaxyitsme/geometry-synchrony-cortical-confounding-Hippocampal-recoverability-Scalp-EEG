#!/usr/bin/env python3
"""Deterministic numerical and leakage tests for method-benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.working_memory import atomic_json, roc_auc  # noqa: E402
from empirical.estimators import (  # noqa: E402
    anatomical_log_power_ratio,
    average_reference_epochs,
    covariance_factors,
    eloreta_type_operator,
    event_covariances,
    fit_fastica,
    ica_event_features,
    lcmv_operator,
    log_channel_power,
    minimum_norm_operator,
    patient_block_folds,
    select_signed_feature,
    sparse_anatomical_score,
    sparse_group_fista,
)


PROTOCOL = "benchmark/method-checks-v1"


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

    generator = np.random.default_rng(20260830)

    def referencing_and_covariance() -> None:
        epochs = generator.normal(size=(9, 8, 64))
        referenced = average_reference_epochs(epochs)
        if np.max(np.abs(referenced.sum(axis=1))) > 1e-12:
            raise AssertionError("average reference failed")
        covariances = event_covariances(referenced)
        if covariances.shape != (9, 8, 8):
            raise AssertionError("wrong covariance shape")
        if np.max(np.abs(covariances - covariances.transpose(0, 2, 1))) > 1e-12:
            raise AssertionError("event covariance is asymmetric")
        if log_channel_power(referenced).shape != (9, 8):
            raise AssertionError("wrong channel feature shape")

    test("referencing_and_event_covariance_are_exact", referencing_and_covariance)

    def ica_recovers_non_gaussian_component() -> None:
        samples = 12000
        sources = np.column_stack(
            (
                generator.laplace(size=samples),
                generator.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=samples),
                generator.standard_t(5, size=samples) / np.sqrt(5.0 / 3.0),
            )
        )
        mixing = np.asarray([[1.0, 0.4, -0.2], [0.3, 1.1, 0.5], [-0.4, 0.2, 0.9]])
        observed = sources @ mixing.T
        model = fit_fastica(observed, 3, seed=19, maximum_iterations=1000, tolerance=1e-7)
        recovered = model.transform(observed)
        correlations = np.abs(np.corrcoef(sources.T, recovered.T)[:3, 3:])
        if not model.converged or np.min(np.max(correlations, axis=1)) < 0.96:
            raise AssertionError("FastICA did not recover the independent sources")
        epochs = observed[:12000].reshape(20, 600, 3).transpose(0, 2, 1)
        if ica_event_features(model, epochs).shape != (20, 3):
            raise AssertionError("wrong ICA event features")

    test("fastica_recovers_auditable_non_gaussian_sources", ica_recovers_non_gaussian_component)

    def feature_selection_is_signed_and_training_only() -> None:
        labels = np.repeat([0, 1], 20)
        features = generator.normal(size=(40, 4))
        features[:, 2] -= 2.0 * labels
        column, sign, auc = select_signed_feature(features, labels)
        if column != 2 or sign != -1.0 or auc < 0.90:
            raise AssertionError("signed feature selection failed")

    test("channel_or_component_selection_handles_polarity", feature_selection_is_signed_and_training_only)

    def linear_inverse_operators_detect_injected_hippocampus() -> None:
        sensors = 7
        cortex = np.eye(sensors)[:, :4]
        hippocampus = np.eye(sensors)[:, 4:]
        gain = np.column_stack((cortex, hippocampus))
        noise = np.eye(sensors)
        low = np.repeat(np.eye(sensors)[None, :, :], 12, axis=0)
        high = low.copy()
        high[:, 4:, 4:] += 20.0 * np.eye(3)[None, :, :]
        covariances = np.concatenate((low, high))
        labels = np.repeat([0, 1], 12)
        h = np.arange(4, 7)
        c = np.arange(4)
        operators = [
            minimum_norm_operator(gain, noise, 0.05),
            eloreta_type_operator(gain, noise, 0.05)[0],
            lcmv_operator(gain, np.mean(low, axis=0), 0.05),
        ]
        for operator in operators:
            score = anatomical_log_power_ratio(operator, covariances, h, c)
            if roc_auc(labels, score) < 0.99:
                raise AssertionError("linear inverse missed an isolated hippocampal injection")
        beamformer = operators[-1]
        if np.max(np.abs(np.diag(beamformer @ gain) - 1.0)) > 1e-12:
            raise AssertionError("LCMV unit-gain constraint failed")

    test("mne_eloreta_and_lcmv_detect_known_hippocampal_power", linear_inverse_operators_detect_injected_hippocampus)

    def sparse_methods_recover_and_hierarchy_changes_attribution() -> None:
        design, _ = np.linalg.qr(generator.normal(size=(16, 12)))
        truth = np.zeros((12, 3))
        truth[9:11] = np.asarray([[2.0, -1.0, 0.5], [-1.0, 0.5, 1.5]])
        targets = design @ truth
        sparse, report = sparse_group_fista(
            design, targets, 0.01, maximum_iterations=2000, tolerance=1e-9
        )
        if not report["converged"] or np.linalg.norm(design @ sparse - targets) > 0.1:
            raise AssertionError("ordinary sparse inverse failed")
        ordinary_score = sparse_anatomical_score(sparse, np.arange(8, 12), np.arange(8))
        hierarchical, hierarchical_report = sparse_group_fista(
            design,
            targets,
            0.005,
            groups=(np.arange(0, 4), np.arange(4, 8), np.arange(8, 12)),
            group_penalty=0.01,
            group_weights=np.ones(3),
            maximum_iterations=2000,
            tolerance=1e-9,
        )
        hierarchical_score = sparse_anatomical_score(
            hierarchical, np.arange(8, 12), np.arange(8)
        )
        if not hierarchical_report["converged"] or min(ordinary_score, hierarchical_score) <= 2.0:
            raise AssertionError("sparse anatomical attribution failed")

    test("ordinary_and_hierarchical_sparse_inverse_recover_known_support", sparse_methods_recover_and_hierarchy_changes_attribution)

    def covariance_factorization_reconstructs_rank_three() -> None:
        factor = generator.normal(size=(8, 3))
        covariance = factor @ factor.T
        recovered = covariance_factors(covariance, 3)
        relative = np.linalg.norm(recovered @ recovered.T - covariance) / np.linalg.norm(covariance)
        if relative > 1e-12:
            raise AssertionError("covariance factorization is inaccurate")

    test("covariance_factors_preserve_declared_rank", covariance_factorization_reconstructs_rank_three)

    def folds_have_no_patient_or_future_block_leakage() -> None:
        patients = np.repeat(np.asarray(["p1", "p2", "p3"]), 12)
        blocks = np.tile(np.repeat(np.arange(3), 4), 3)
        folds = patient_block_folds(patients, blocks, (0, 1), (2,))
        if len(folds) != 3:
            raise AssertionError("wrong outer fold count")
        for fold in folds:
            train = np.asarray(fold["train_indices"])
            test_indices = np.asarray(fold["test_indices"])
            heldout = str(fold["heldout_patient"])
            if heldout in set(patients[train]) or set(patients[test_indices]) != {heldout}:
                raise AssertionError("patient leakage")
            if set(blocks[train]) != {0, 1} or set(blocks[test_indices]) != {2}:
                raise AssertionError("block leakage")

    test("outer_folds_are_patient_held_out_and_prospective", folds_have_no_patient_or_future_block_leakage)

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
