#!/usr/bin/env python3
"""Adversarial and identity tests for source-ensemble source ensembles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from anatomical.hippunfold_ensembles import (
    bilateral_interference,
    build_source_supports,
    oriented_intrinsic_coordinates,
    patch_coherence_metrics,
    phase_locked_metrics,
    project_free_cartesian_leadfield,
    random_correlated_phases,
    support_identifier,
    tilted_directions,
    whitened_subspace_projection,
)


def _geometry(points: int = 40) -> dict[str, np.ndarray]:
    half = points // 2
    ap = np.linspace(0.0, 1.0, half)
    pd = np.mod(np.arange(half), 5) / 4.0
    left = np.column_stack((-0.02 * np.ones(half), -0.03 + 0.06 * ap, 0.01 * pd))
    right = left.copy()
    right[:, 0] *= -1.0
    return {
        "positions": np.vstack((left, right)),
        "ap": np.concatenate((ap, ap)),
        "pd": np.concatenate((pd, pd)),
        "hemisphere": np.concatenate(
            (-np.ones(half, dtype=np.int8), np.ones(half, dtype=np.int8))
        ),
        "area": np.linspace(0.8, 1.2, points),
    }


class EnsembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = _geometry()
        self.anterior, self.pd, _ = oriented_intrinsic_coordinates(
            self.geometry["positions"],
            self.geometry["ap"],
            self.geometry["pd"],
            self.geometry["hemisphere"],
            self.geometry["area"],
            endpoint_decile=0.1,
            minimum_endpoint_separation_m=0.02,
        )
        self.supports, self.support_report = build_source_supports(
            self.anterior,
            self.pd,
            self.geometry["hemisphere"],
            self.geometry["area"],
            hemisphere_families=("left", "right", "bilateral"),
            focal_locations=("anterior", "middle", "posterior"),
            focal_extents=(0.05, 0.10, 0.25, 0.50),
            include_whole_extent=True,
        )
        rng = np.random.default_rng(20260830)
        self.leadfield = rng.normal(size=(11, 40))
        nuisance = rng.normal(size=(11, 11))
        nuisance = nuisance @ nuisance.T + 0.5 * np.eye(11)
        self.cholesky = np.linalg.cholesky(nuisance)
        weights = self.geometry["area"] / np.sum(self.geometry["area"])
        self.full_trace = float(np.sum((self.leadfield * self.leadfield) @ weights))

    def test_AP_polarity_and_support_nesting(self) -> None:
        self.assertGreater(
            np.corrcoef(self.anterior, self.geometry["positions"][:, 1])[0, 1],
            0.99,
        )
        self.assertEqual(len(self.supports), 39)
        self.assertTrue(self.support_report["all_exactly_nested"])
        for family in ("left", "right", "bilateral"):
            for location in ("anterior", "middle", "posterior"):
                previous: set[int] = set()
                for extent in (0.05, 0.10, 0.25, 0.50):
                    support = self.supports[
                        support_identifier(family, location, extent)
                    ]
                    current = set(int(value) for value in support.indices)
                    self.assertTrue(previous.issubset(current))
                    previous = current

    def test_phase_power_bound_and_wave_reversal(self) -> None:
        support = self.supports[support_identifier("bilateral", "whole", 1.0)]
        phase = 2.0 * np.pi * self.anterior[support.indices]
        metrics, factor = phase_locked_metrics(
            self.leadfield,
            support,
            phase,
            self.full_trace,
            self.cholesky,
        )
        reverse_metrics, reverse = phase_locked_metrics(
            self.leadfield,
            support,
            -phase,
            self.full_trace,
            self.cholesky,
        )
        self.assertGreaterEqual(metrics["support_normalized_retained_power_ratio"], 0.0)
        self.assertLessEqual(metrics["support_normalized_retained_power_ratio"], 1.0 + 1e-12)
        self.assertAlmostEqual(
            metrics["support_normalized_retained_power_ratio"],
            reverse_metrics["support_normalized_retained_power_ratio"],
            places=13,
        )
        np.testing.assert_allclose(factor @ factor.T, reverse @ reverse.T, atol=1e-12)

    def test_patch_coherence_is_PSD_factor_and_bounded(self) -> None:
        support = self.supports[support_identifier("left", "whole", 1.0)]
        metrics, factor = patch_coherence_metrics(
            self.leadfield,
            support,
            self.anterior,
            self.pd,
            self.geometry["hemisphere"],
            1.0,
            self.full_trace,
            self.cholesky,
        )
        zero, zero_factor = phase_locked_metrics(
            self.leadfield,
            support,
            np.zeros(len(support.indices)),
            self.full_trace,
            self.cholesky,
        )
        self.assertEqual(metrics["coherent_patch_count"], 1)
        self.assertLessEqual(metrics["support_normalized_retained_power_ratio"], 1.0 + 1e-12)
        np.testing.assert_allclose(factor[:, 0], zero_factor[:, 0], atol=1e-12)
        self.assertAlmostEqual(
            metrics["support_normalized_retained_power_ratio"],
            zero["support_normalized_retained_power_ratio"],
        )

    def test_random_phase_is_seed_reproducible_and_unit_magnitude(self) -> None:
        support = self.supports[
            support_identifier("bilateral", "middle", 0.25)
        ]
        kwargs = dict(
            correlation_length=0.25,
            features=64,
            phase_standard_deviation_rad=np.pi,
            master_seed=20260830,
            seed_components=("subject", "support", 0),
        )
        first, report = random_correlated_phases(
            self.anterior[support.indices],
            self.pd[support.indices],
            self.geometry["hemisphere"][support.indices],
            support.conditional_weights,
            **kwargs,
        )
        second, _ = random_correlated_phases(
            self.anterior[support.indices],
            self.pd[support.indices],
            self.geometry["hemisphere"][support.indices],
            support.conditional_weights,
            **kwargs,
        )
        np.testing.assert_array_equal(first, second)
        self.assertLess(report["unit_complex_magnitude_maximum_error"], 1e-14)

    def test_physical_orientation_projection_and_tilt(self) -> None:
        rng = np.random.default_rng(12)
        directions = rng.normal(size=(17, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        free = rng.normal(size=(9, 17, 3))
        fixed = np.einsum("snc,nc->sn", free, directions)
        np.testing.assert_allclose(
            project_free_cartesian_leadfield(free.reshape(9, -1), directions),
            fixed,
            atol=1e-13,
        )
        identity, _ = tilted_directions(directions, 0.0)
        orthogonal, audit = tilted_directions(directions, 90.0)
        np.testing.assert_allclose(identity, directions, atol=1e-13)
        np.testing.assert_allclose(
            np.sum(orthogonal * directions, axis=1), 0.0, atol=1e-13
        )
        self.assertAlmostEqual(audit["minimum_realized_angle_degrees"], 90.0)
        self.assertAlmostEqual(audit["maximum_realized_angle_degrees"], 90.0)

    def test_whitened_projection_identity_and_orthogonal_control(self) -> None:
        truth = np.zeros((11, 1))
        truth[0, 0] = 1.0
        identity = whitened_subspace_projection(truth, truth, np.eye(11))
        self.assertAlmostEqual(identity["whitened_truth_projection_efficiency"], 1.0)
        orthogonal = np.zeros((11, 1))
        orthogonal[1, 0] = 1.0
        control = whitened_subspace_projection(truth, orthogonal, np.eye(11))
        self.assertAlmostEqual(control["whitened_truth_projection_efficiency"], 0.0)
        self.assertAlmostEqual(
            control["whitened_subspace_maximum_principal_angle_degrees"], 90.0
        )

    def test_bilateral_interference_identity_and_exact_cancellation(self) -> None:
        left = np.asarray([[1.0], [2.0], [-1.0]])
        record = bilateral_interference(left, -left)
        self.assertAlmostEqual(record["bilateral_locked_power"], 0.0)
        self.assertAlmostEqual(record["bilateral_interference_gain"], 0.0)
        self.assertAlmostEqual(
            record["cross_term_fraction_of_independent_power"], -1.0
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "hippunfold_ensemble_tests.json",
    )
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EnsembleTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 1,
        "protocol": "hcp/source-ensembles-v1.1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
