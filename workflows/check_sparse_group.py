#!/usr/bin/env python3
"""Deterministic numerical tests for sparse-group method controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from empirical.estimators import fit_fastica  # noqa: E402
from empirical.source_scores import (  # noqa: E402
    best_channel_anatomical_scores,
    factor_covariance,
    heldout_injection_threshold,
    ica_anatomical_scores,
    stable_anatomical_log_power_ratio,
    stochastic_injection_covariance,
)


class SparseGroupTests(unittest.TestCase):
    def test_factor_covariance_identity(self) -> None:
        factors = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(np.allclose(factor_covariance(factors), factors @ factors.T / 2.0))

    def test_injection_has_declared_trace_energy(self) -> None:
        baseline = np.diag([2.0, 3.0, 5.0])
        injected = stochastic_injection_covariance(baseline, [1.0, 2.0, 0.0], 0.4)
        self.assertAlmostEqual(np.trace(injected) - np.trace(baseline), 0.4 * np.trace(baseline))

    def test_best_channel_is_selected_from_injection_not_response(self) -> None:
        baseline = np.eye(3)
        injected = stochastic_injection_covariance(baseline, [0.0, 3.0, 0.0], 1.0)
        response = np.diag([100.0, 1.0, 1.0])
        _, _, injection_score, channel = best_channel_anatomical_scores(
            baseline, response, injected, [0.0, 3.0, 0.0]
        )
        self.assertEqual(channel, 1)
        self.assertGreater(injection_score, 0.0)

    def test_ica_selection_uses_injection_projection(self) -> None:
        generator = np.random.default_rng(7)
        samples = generator.normal(size=(1000, 3))
        model = fit_fastica(samples, 3, seed=8)
        baseline = np.eye(3)
        injected = stochastic_injection_covariance(baseline, [1.0, 0.0, 0.0], 1.0)
        base, _, positive, component = ica_anatomical_scores(
            model, baseline, baseline, injected, [1.0, 0.0, 0.0]
        )
        self.assertIn(component, (0, 1, 2))
        self.assertGreater(positive, base)

    def test_heldout_threshold_targets_lower_tail(self) -> None:
        values = np.arange(10, dtype=np.float64)
        self.assertEqual(heldout_injection_threshold(values, 0.8), 1.0)

    def test_log_power_ratio_is_finite_at_extreme_dynamic_range(self) -> None:
        coefficients = np.asarray([[1e150], [1e-150]], dtype=np.float64)
        value = stable_anatomical_log_power_ratio(coefficients, [0], [1])
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 700.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SparseGroupTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "protocol": "benchmark/cortical-method-control-v1",
        "ok": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 2


if __name__ == "__main__":
    raise SystemExit(main())
