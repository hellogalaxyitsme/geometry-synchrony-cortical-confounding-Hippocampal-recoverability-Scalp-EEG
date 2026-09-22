#!/usr/bin/env python3
"""Deterministic numerical tests for the corrected injection-benchmark design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from empirical.injection_benchmark import (  # noqa: E402
    added_signal_ratio_from_total_energy_ratio,
    anatomy_derangement,
    balanced_source_schedule,
    best_channel_factor_scores,
    complete_primary_method_metrics,
    ica_factor_scores,
    patient_equal_injection_threshold,
    patient_equal_sensitivity,
    stochastic_factor_injection_covariance,
)


class InjectionBenchmarkTests(unittest.TestCase):
    def test_total_ratio_is_not_reused_as_added_signal_ratio(self) -> None:
        self.assertAlmostEqual(added_signal_ratio_from_total_energy_ratio(1.4186608319), 0.4186608319)
        with self.assertRaises(ValueError):
            added_signal_ratio_from_total_energy_ratio(1.0)

    def test_factor_injection_has_exact_total_trace_ratio(self) -> None:
        baseline = np.diag([2.0, 3.0, 5.0])
        factor = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        total = 1.4186608319
        injected, excess = stochastic_factor_injection_covariance(
            baseline, factor, added_signal_ratio_from_total_energy_ratio(total)
        )
        self.assertAlmostEqual(float(np.trace(injected) / np.trace(baseline)), total)
        self.assertAlmostEqual(float(np.trace(excess) / np.trace(baseline)), total - 1.0)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(excess))), -1e-12)

    def test_best_channel_selection_never_uses_response(self) -> None:
        baseline = np.eye(3)
        injected, excess = stochastic_factor_injection_covariance(baseline, [0.0, 2.0, 0.0], 0.4)
        response = np.diag([1000.0, 1.0, 1.0])
        _, _, positive, channel = best_channel_factor_scores(baseline, response, injected, excess)
        self.assertEqual(channel, 1)
        self.assertGreater(positive, 0.0)

    def test_ica_selection_uses_excess_covariance(self) -> None:
        baseline = np.eye(3)
        injected, excess = stochastic_factor_injection_covariance(baseline, [0.0, 0.0, 1.0], 0.5)
        _, _, positive, component = ica_factor_scores(np.eye(3), baseline, np.diag([50.0, 1.0, 1.0]), injected, excess)
        self.assertEqual(component, 2)
        self.assertGreater(positive, 0.0)

    def test_anatomy_pairing_is_complete_derangement(self) -> None:
        subjects = [f"s{index}" for index in range(7)]
        mapping = anatomy_derangement(subjects, 3)
        self.assertEqual(set(mapping), set(subjects))
        self.assertEqual(set(mapping.values()), set(subjects))
        self.assertTrue(all(key != value for key, value in mapping.items()))
        with self.assertRaises(ValueError):
            anatomy_derangement(subjects, 7)

    def test_source_schedule_is_balanced_and_reproducible(self) -> None:
        first = balanced_source_schedule(323, 23, 19, "integrated")
        second = balanced_source_schedule(323, 23, 19, "integrated")
        other = balanced_source_schedule(323, 23, 19, "n1")
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, other))
        counts = np.bincount(first, minlength=23)
        self.assertLessEqual(int(np.max(counts) - np.min(counts)), 1)

    def test_patient_equal_threshold_is_invariant_to_patient_replication(self) -> None:
        scores = np.asarray([1.0, 2.0, 3.0, 4.0])
        patients = np.asarray(["a", "a", "b", "b"])
        base = patient_equal_injection_threshold(scores, patients, 0.5)
        replicated_scores = np.concatenate((np.repeat(scores[:2], 20), scores[2:]))
        replicated_patients = np.concatenate((np.repeat(patients[:2], 20), patients[2:]))
        repeated = patient_equal_injection_threshold(replicated_scores, replicated_patients, 0.5)
        self.assertEqual(base, repeated)

    def test_patient_equal_threshold_is_maximally_specific_subject_to_target(self) -> None:
        scores = np.asarray([1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
        patients = np.asarray(["a", "a", "b", "b", "c", "c"])
        target = 2.0 / 3.0
        threshold = patient_equal_injection_threshold(scores, patients, target)
        self.assertGreaterEqual(patient_equal_sensitivity(scores, patients, threshold), target)
        higher = np.nextafter(np.min(scores[scores > threshold]), np.inf)
        self.assertLess(patient_equal_sensitivity(scores, patients, higher), target)

    def test_headline_builder_cannot_drop_a_method(self) -> None:
        methods = ("a", "b", "c")
        metrics = ("false_fire", "sensitivity")
        rows = [
            {"endpoint": "primary", "method": method, "metric": metric, "median": index + offset}
            for index, method in enumerate(methods)
            for offset, metric in enumerate(metrics)
        ]
        result = complete_primary_method_metrics(rows, methods, "primary", metrics)
        self.assertEqual(set(result), set(methods))
        self.assertTrue(all(set(value) == set(metrics) for value in result.values()))
        with self.assertRaises(ValueError):
            complete_primary_method_metrics(rows[:-1], methods, "primary", metrics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sources = list(config["source_ensemble"]["sources"])
    design_checks = {
        "checks_cortical_method_control": config.get("protocol") == "benchmark/cortical-method-control-v2",
        "source_count_23": len(sources) == 23,
        "source_ids_unique": len({str(row["id"]) for row in sources}) == len(sources),
        "laterality_covered": {str(row["hemisphere"]) for row in sources} == {"left", "right", "bilateral"},
        "location_covered": {str(row["location"]) for row in sources} == {"whole", "anterior", "middle", "posterior"},
        "extent_ladder_covered": {float(row["extent"]) for row in sources} == {0.05, 0.10, 0.25, 0.50, 1.0},
        "wave_ladder_covered": {
            float(row["wave_cycles"]) for row in sources if row["phase_model"] == "deterministic_wave"
        } == {0.0, 0.25, 0.50, 1.0, 2.0},
        "random_phase_ladder_covered": {
            float(row["correlation_length"]) for row in sources if row["phase_model"] == "correlated_random"
        } == {0.05, 0.10, 0.25, 0.50},
        "matched_anatomy_prohibited": config.get("matched_injection_reconstruction_anatomy_permitted") is False,
        "energy_semantics_explicit": config["energy_calibration"].get("conversion") == "added_equals_total_minus_one",
    }
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InjectionBenchmarkTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 2,
        "protocol": "benchmark/cortical-method-control-v2",
        "ok": result.wasSuccessful() and all(design_checks.values()),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "design_checks": design_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
