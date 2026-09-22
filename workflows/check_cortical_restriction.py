#!/usr/bin/env python3
"""Deterministic unit and adversarial tests for cortical-restriction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from simulation.cortical_restriction import (  # noqa: E402
    basis_diagnostics,
    covariance_basis,
    deterministic_dictionary_split,
    dynamic_hippocampal_factors,
    false_attribution,
    neighborhood_indices,
    orthonormal_basis,
    residual_sensitivity,
    smooth_topographies,
    somp,
    sparse_false_attribution,
    sparse_sensitivity,
    tensor_false_attribution,
    tensor_sensitivity,
)
from workflows.build_empirical_covariance_basis import (  # noqa: E402
    patient_fraction_is_usable,
    session_is_usable,
)


class CorticalRestrictionTests(unittest.TestCase):
    def test_empirical_session_gate_has_exact_frozen_boundaries(self) -> None:
        self.assertEqual(session_is_usable(10, 20, 10, 0.50), (True, None))
        self.assertEqual(
            session_is_usable(9, 20, 10, 0.50),
            (False, "too_few_artifact_screened_windows"),
        )
        self.assertEqual(
            session_is_usable(10, 21, 10, 0.50),
            (False, "retained_fraction_below_gate"),
        )

    def test_empirical_patient_coverage_gate_has_exact_boundary(self) -> None:
        self.assertTrue(patient_fraction_is_usable(13, 14, 0.90))
        self.assertTrue(patient_fraction_is_usable(9, 10, 0.90))
        self.assertFalse(patient_fraction_is_usable(8, 10, 0.90))
        self.assertFalse(patient_fraction_is_usable(0, 0, 0.90))

    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260830)

    def test_covariance_basis_meets_trace_and_projector_gate(self) -> None:
        matrix = self.rng.normal(size=(11, 7))
        covariance = matrix @ matrix.T
        basis, values, retained = covariance_basis(covariance, 0.90)
        self.assertGreaterEqual(retained, 0.90)
        self.assertLess(retained - values[basis.shape[1] - 1] / values.sum(), 0.90)
        diagnostics = basis_diagnostics(basis)
        self.assertLess(diagnostics["orthonormality_max_error"], 1e-12)
        self.assertLess(diagnostics["projector_idempotence_relative_error"], 1e-12)

    def test_unrestricted_span_erases_sensitivity_without_false_gain(self) -> None:
        sensors = 13
        cortex = self.rng.normal(size=(sensors, sensors + 8))
        basis = orthonormal_basis(cortex)
        self.assertEqual(basis.shape[1], sensors)
        hippocampal = self.rng.normal(size=(sensors, 2))
        probes = self.rng.normal(size=(sensors, 19))
        self.assertLess(residual_sensitivity(hippocampal, basis)["sensitivity_fraction"], 1e-24)
        self.assertLess(float(np.max(false_attribution(probes, hippocampal, basis))), 1e-24)

    def test_false_attribution_detects_omitted_cortical_direction(self) -> None:
        basis = np.eye(8)[:, :3]
        hippocampal = np.eye(8)[:, 3:4]
        probes = np.column_stack((np.eye(8)[:, 3], np.eye(8)[:, 4]))
        values = false_attribution(probes, hippocampal, basis)
        self.assertAlmostEqual(values[0], 1.0, places=13)
        self.assertAlmostEqual(values[1], 0.0, places=13)

    def test_dictionary_split_is_reproducible_and_disjoint(self) -> None:
        areas = np.linspace(1.0, 3.0, 200)
        usable = np.arange(10, 190)
        first = deterministic_dictionary_split(areas, usable, 80, 30, 14)
        second = deterministic_dictionary_split(areas, usable, 80, 30, 14)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(np.intersect1d(first[0], first[1]).size, 0)

    def test_somp_exact_sparse_recovery_and_sparse_false_control(self) -> None:
        dictionary, _ = np.linalg.qr(self.rng.normal(size=(24, 16)))
        truth = dictionary[:, [2, 9]] @ np.asarray([[1.2, -0.3], [-0.7, 0.6]])
        selected, residual = somp(dictionary, truth, 2)
        self.assertEqual(set(selected.tolist()), {2, 9})
        self.assertLess(np.linalg.norm(residual), 1e-12)
        hippocampal = dictionary[:, 12:13]
        sensitivity = sparse_sensitivity(hippocampal, dictionary[:, :10], 2)
        self.assertAlmostEqual(sensitivity["sensitivity_fraction"], 1.0, places=12)
        false = sparse_false_attribution(
            dictionary[:, 2:3], hippocampal, dictionary[:, :10], 2
        )
        self.assertAlmostEqual(float(false[0]), 0.0, places=12)

    def test_smooth_kernels_are_hemisphere_local_and_finite(self) -> None:
        positions = np.asarray(
            [[-0.03, 0.0, 0.0], [-0.02, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]]
        )
        hemispheres = np.asarray([-1, -1, 1, 1], dtype=np.int8)
        areas = np.ones(4)
        leadfield = np.eye(4)
        smooth = smooth_topographies(
            leadfield, positions, hemispheres, areas, np.asarray([0, 3]), 0.02
        )
        self.assertTrue(np.all(np.isfinite(smooth)))
        self.assertAlmostEqual(float(smooth[2:, 0].sum()), 0.0, places=15)
        self.assertAlmostEqual(float(smooth[:2, 1].sum()), 0.0, places=15)

    def test_neighborhood_selection_uses_declared_radius(self) -> None:
        cortical = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.05, 0.0, 0.0]])
        hippocampal = np.asarray([[0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(
            neighborhood_indices(cortical, hippocampal, 0.02), np.asarray([0, 1])
        )

    def test_tensor_restriction_exposes_unmodeled_theta(self) -> None:
        spatial = np.eye(5)[:, :2]
        temporal = np.eye(16)[:, :3]
        hippocampal = dynamic_hippocampal_factors(
            np.eye(5)[:, 3:4], samples=16, sampling_frequency=16.0, carrier_hz=4.0
        )
        sensitivity = tensor_sensitivity(hippocampal, spatial, temporal)
        self.assertAlmostEqual(sensitivity["sensitivity_fraction"], 1.0, places=12)
        false = tensor_false_attribution(
            [hippocampal[0]], hippocampal, spatial, temporal
        )
        self.assertAlmostEqual(float(false[0]), 1.0, places=12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorticalRestrictionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 1,
        "protocol": "restriction/ladder-checks-v1",
        "ok": result.wasSuccessful(),
        "tests_total": result.testsRun,
        "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
        "failures": [str(value[0]) for value in result.failures],
        "errors": [str(value[0]) for value in result.errors],
        "numpy": np.__version__,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
