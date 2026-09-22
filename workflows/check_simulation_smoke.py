#!/usr/bin/env python3
"""Limiting-case and invariance tests for the synthetic experiment."""

from __future__ import annotations

import math
import argparse
import copy
import json
import platform
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.synthetic import (  # noqa: E402
    build_cortical_subspace,
    build_sensor_modes,
    compute_recoverability_metrics,
    leading_topography,
    nuisance_covariance,
    perturb_contributions,
    signal_covariance,
    source_covariance,
    stable_seed,
    structured_contributions,
)
from theory.recoverability import weighted_projection_residual  # noqa: E402
from simulation.phase_models import load_config, run_experiment  # noqa: E402


class SyntheticFrameworkTests(unittest.TestCase):
    seed = 20260825

    def setUp(self) -> None:
        self.modes = build_sensor_modes(31, self.seed)

    def contributions(
        self,
        active_fraction: float = 1.0,
        orientation_span: float = 0.0,
        normalization: str = "fixed_total",
        depth_gradient: float = 0.0,
        secondary_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        return structured_contributions(
            self.modes,
            number_of_elements=64,
            active_fraction=active_fraction,
            orientation_span_radians=orientation_span,
            source_gain=0.4,
            depth_gradient=depth_gradient,
            secondary_geometry_weight=secondary_weight,
            normalization=normalization,
        )

    def test_source_covariance_psd_unit_diagonal_and_fixed_trace(self) -> None:
        _, positions = self.contributions(active_fraction=0.75)
        for synchrony in (0.0, 0.25, 0.75, 1.0):
            for cycles in (0.0, 0.25, 1.0, 2.0):
                covariance = source_covariance(positions, synchrony, cycles)
                self.assertTrue(np.allclose(np.diag(covariance), 1.0, atol=1e-12))
                self.assertAlmostEqual(float(np.trace(covariance)), positions.size)
                self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(covariance))), -1e-10)

    def test_incoherent_model_is_exactly_phase_invariant(self) -> None:
        contributions, positions = self.contributions(
            orientation_span=2.0 * math.pi,
            depth_gradient=0.2,
            secondary_weight=0.15,
        )
        reference_covariance = source_covariance(positions, 0.0, 0.0)
        reference_signal = signal_covariance(contributions, reference_covariance)
        for cycles in (0.25, 0.5, 1.0, 1.5, 2.0):
            covariance = source_covariance(positions, 0.0, cycles)
            self.assertTrue(np.array_equal(covariance, reference_covariance))
            self.assertTrue(
                np.array_equal(
                    signal_covariance(contributions, covariance), reference_signal
                )
            )

    def test_fixed_total_incoherent_power_is_extent_invariant(self) -> None:
        powers = []
        for active_fraction in (0.125, 0.25, 0.5, 0.75, 1.0):
            contributions, positions = self.contributions(
                active_fraction=active_fraction,
                orientation_span=2.0 * math.pi,
                depth_gradient=0.25,
                secondary_weight=0.2,
            )
            covariance = source_covariance(positions, 0.0, 0.0)
            powers.append(float(np.trace(signal_covariance(contributions, covariance))))
        self.assertLess(max(powers) - min(powers), 1e-12)
        self.assertAlmostEqual(powers[0], 0.4**2, places=12)

    def test_aligned_coherence_has_exact_amplification(self) -> None:
        contributions, positions = self.contributions(orientation_span=0.0)
        covariance = source_covariance(positions, 1.0, 0.0)
        power = float(np.trace(signal_covariance(contributions, covariance)))
        self.assertAlmostEqual(power, positions.size * 0.4**2, places=11)

    def test_curvature_cancels_stationary_coherence_and_phase_rescues_it(self) -> None:
        contributions, positions = self.contributions(
            orientation_span=2.0 * math.pi
        )
        stationary = signal_covariance(
            contributions, source_covariance(positions, 1.0, 0.0)
        )
        traveling = signal_covariance(
            contributions, source_covariance(positions, 1.0, 1.0)
        )
        stationary_power = float(np.trace(stationary))
        traveling_power = float(np.trace(traveling))
        self.assertLess(stationary_power, 1e-24)
        self.assertGreater(traveling_power, 1.0)

    def test_prescribed_cortical_overlap_and_full_rank_masking(self) -> None:
        rng = np.random.default_rng(self.seed)
        target = rng.normal(size=31)
        target /= np.linalg.norm(target)
        for overlap in (0.0, 0.25, 0.5, 0.9, 1.0):
            cortical = build_cortical_subspace(target, 6, overlap, self.seed)
            _, energy = weighted_projection_residual(target, cortical, np.eye(31))
            self.assertAlmostEqual(energy, 1.0 - overlap, places=11)
        full = build_cortical_subspace(target, 31, 0.0, self.seed)
        _, energy = weighted_projection_residual(target, full, np.eye(31))
        self.assertLess(energy, 1e-24)

    def test_nested_measurement_information_is_monotone(self) -> None:
        contributions, positions = self.contributions(
            orientation_span=math.pi,
            depth_gradient=0.2,
            secondary_weight=0.15,
        )
        covariance = source_covariance(positions, 0.7, 0.5)
        signal = signal_covariance(contributions, covariance)
        cortical = build_cortical_subspace(
            leading_topography(signal), 8, 0.7, self.seed
        )
        nuisance = nuisance_covariance(cortical, 1.0, 2.0)
        values = []
        for dimension in (7, 15, 31):
            selection = slice(0, dimension)
            metrics = compute_recoverability_metrics(
                contributions[selection],
                signal[selection, selection],
                nuisance[selection, selection],
                cortical[selection],
            )
            values.append(metrics["mutual_information_bits"])
        self.assertTrue(all(right >= left - 1e-12 for left, right in zip(values, values[1:])))

    def test_orthogonal_sensor_rotation_preserves_metrics(self) -> None:
        contributions, positions = self.contributions(
            orientation_span=math.pi,
            depth_gradient=0.2,
            secondary_weight=0.15,
        )
        covariance = source_covariance(positions, 0.6, 0.75)
        signal = signal_covariance(contributions, covariance)
        cortical = build_cortical_subspace(
            leading_topography(signal), 9, 0.4, self.seed
        )
        nuisance = nuisance_covariance(cortical, 1.0, 1.5)
        original = compute_recoverability_metrics(
            contributions, signal, nuisance, cortical
        )
        rng = np.random.default_rng(self.seed + 1)
        rotation, _ = np.linalg.qr(rng.normal(size=(31, 31)))
        rotated = compute_recoverability_metrics(
            rotation @ contributions,
            rotation @ signal @ rotation.T,
            rotation @ nuisance @ rotation.T,
            rotation @ cortical,
        )
        for metric in (
            "signal_power",
            "whitened_signal_power",
            "mutual_information_bits",
            "presence_kl_nats",
            "dominant_mode_dprime",
            "masking_residual_fraction",
            "geometry_efficiency",
        ):
            self.assertAlmostEqual(original[metric], rotated[metric], places=10)

    def test_forward_perturbation_norm_and_reproducibility(self) -> None:
        contributions, _ = self.contributions(orientation_span=math.pi)
        seed = stable_seed(self.seed, {"test": "perturbation", "replicate": 3})
        first, achieved_first = perturb_contributions(contributions, 0.15, seed)
        second, achieved_second = perturb_contributions(contributions, 0.15, seed)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(achieved_first, achieved_second)
        self.assertAlmostEqual(achieved_first, 0.15, places=7)
        self.assertTrue(
            np.allclose(
                np.linalg.norm(first, axis=0),
                np.linalg.norm(contributions, axis=0),
                rtol=0.0,
                atol=1e-13,
            )
        )

    def test_registered_pipeline_is_bitwise_reproducible(self) -> None:
        config = copy.deepcopy(
            load_config(PROJECT_ROOT / "configs" / "phase_ensembles_v1.json")
        )
        config["maximum_contrast_dimension"] = 31
        config["full_source_elements"] = 16
        phase = config["phase_diagram"]
        phase["active_fractions"] = [1.0]
        phase["orientation_spans_radians"] = [0.0, 2.0 * math.pi]
        phase["synchrony_fractions"] = [0.0, 1.0]
        phase["phase_cycles_full_extent"] = [0.0, 1.0]
        phase["normalizations"] = ["fixed_total"]
        phase["contrast_dimension"] = 31
        phase["cortical_rank"] = 6
        ladder = config["confounding_ladder"]
        ladder["contrast_dimension"] = 31
        ladder["cortical_ranks"] = [4, 31]
        ladder["cortical_overlaps"] = [0.0, 1.0]
        montage = config["montage_uncertainty"]
        montage["contrast_dimensions"] = [7, 15, 31]
        montage["cortical_rank"] = 6
        montage["relative_forward_errors"] = [0.0, 0.1]
        montage["replicates_per_nonzero_error"] = 2
        montage["scenarios"] = montage["scenarios"][:1]
        first_rows, first_summary = run_experiment(config)
        second_rows, second_summary = run_experiment(config)
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_summary, second_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticFrameworkTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 1,
        "seed": SyntheticFrameworkTests.seed,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
