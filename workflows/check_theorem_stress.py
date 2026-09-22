#!/usr/bin/env python3
"""Numerically falsify or corroborate the results stated in ``theorems.md``.

The suite uses randomized matrices, adversarial cancellation examples, Monte
Carlo detection, ill-conditioned covariances, and a finite-window/spectral
comparison.  A non-zero exit code means at least one theoretical invariant was
violated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from theory.recoverability import (  # noqa: E402
    ar1_finite_window_mi_rate_bits,
    ar1_spectral_mi_rate_bits,
    equicorrelated_sensor_power,
    forward_error_dprime_interval,
    gaussian_mi_bits,
    gaussian_posterior_covariance,
    geometry_efficiency,
    helmert_reference,
    kl_present_vs_absent_nats,
    matched_filter_auc,
    matched_filter_dprime,
    perturbation_mi_interval_bits,
    sensor_power_from_covariance,
    subset_gaussian_mi_bits,
    whitened_signal_eigenvalues,
    weighted_projection_residual,
)


def random_spd(
    rng: np.random.Generator, dimension: int, condition_number: float = 1e3
) -> np.ndarray:
    orthogonal, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    values = np.geomspace(1.0, condition_number, dimension)
    rng.shuffle(values)
    return (orthogonal * values) @ orthogonal.T


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(1.0, abs(expected))


class StressSuite:
    def __init__(self, seed: int, trials: int) -> None:
        self.seed = seed
        self.trials = trials
        self.rng = np.random.default_rng(seed)
        self.results: list[dict[str, object]] = []

    def record(self, test_id: str, passed: bool, **metrics: object) -> None:
        result = {"test_id": test_id, "passed": bool(passed), **metrics}
        self.results.append(result)
        status = "PASS" if passed else "FAIL"
        metric_text = ", ".join(f"{key}={value}" for key, value in metrics.items())
        print(f"[{status}] {test_id}: {metric_text}")

    def reference_rank(self) -> None:
        maximum_error = 0.0
        minimum_rank_margin = math.inf
        for electrodes in (2, 4, 19, 64, 256):
            reference = helmert_reference(electrodes)
            ones_error = np.linalg.norm(reference @ np.ones(electrodes))
            orthogonality_error = np.linalg.norm(
                reference @ reference.T - np.eye(electrodes - 1), ord=2
            )
            maximum_error = max(maximum_error, ones_error, orthogonality_error)
            minimum_rank_margin = min(
                minimum_rank_margin,
                float(np.min(np.linalg.svd(reference, compute_uv=False))),
            )
        self.record(
            "ST-01-reference-rank",
            maximum_error < 1e-12 and minimum_rank_margin > 1.0 - 1e-12,
            max_error=f"{maximum_error:.3e}",
            min_singular_value=f"{minimum_rank_margin:.12f}",
        )

    def geometry_bound(self) -> None:
        maximum_efficiency = 0.0
        for _ in range(self.trials):
            sensors = int(self.rng.integers(2, 30))
            generators = int(self.rng.integers(2, 80))
            contributions = self.rng.normal(size=(sensors, generators))
            maximum_efficiency = max(
                maximum_efficiency, geometry_efficiency(contributions)
            )
        direction = self.rng.normal(size=12)
        direction /= np.linalg.norm(direction)
        positive_weights = self.rng.uniform(0.1, 3.0, size=20)
        aligned = direction[:, None] * positive_weights[None, :]
        cancelled = np.column_stack((direction, -direction))
        aligned_efficiency = geometry_efficiency(aligned)
        cancelled_efficiency = geometry_efficiency(cancelled)
        passed = (
            maximum_efficiency <= 1.0 + 1e-12
            and abs(aligned_efficiency - 1.0) < 1e-12
            and cancelled_efficiency < 1e-12
        )
        self.record(
            "ST-02-geometry-cancellation-bound",
            passed,
            max_random_efficiency=f"{maximum_efficiency:.6f}",
            aligned=f"{aligned_efficiency:.12f}",
            cancellation=f"{cancelled_efficiency:.3e}",
        )

    def synchrony_geometry_law(self) -> None:
        maximum_error = 0.0
        maximum_complex_error = 0.0
        maximum_rank_one_error = 0.0
        for _ in range(self.trials):
            sensors = int(self.rng.integers(2, 20))
            generators = int(self.rng.integers(2, 30))
            contributions = self.rng.normal(size=(sensors, generators))
            lower = -1.0 / (generators - 1)
            rho = float(self.rng.uniform(lower, 1.0))
            covariance = (1.0 - rho) * np.eye(generators) + rho * np.ones(
                (generators, generators)
            )
            direct = float(np.trace(contributions @ covariance @ contributions.T))
            formula = equicorrelated_sensor_power(contributions, rho)
            maximum_error = max(maximum_error, relative_error(direct, formula))

            complex_contributions = self.rng.normal(
                size=(sensors, generators)
            ) + 1j * self.rng.normal(size=(sensors, generators))
            covariance_rank = int(self.rng.integers(1, generators + 1))
            covariance_factor = self.rng.normal(
                size=(generators, covariance_rank)
            ) + 1j * self.rng.normal(size=(generators, covariance_rank))
            complex_covariance = covariance_factor @ covariance_factor.conj().T
            complex_formula = sensor_power_from_covariance(
                complex_contributions, complex_covariance
            )
            complex_factored = float(
                np.linalg.norm(complex_contributions @ covariance_factor, ord="fro")
                ** 2
            )
            maximum_complex_error = max(
                maximum_complex_error,
                relative_error(complex_formula, complex_factored),
            )

            phases = self.rng.uniform(-math.pi, math.pi, size=generators)
            phasor = self.rng.uniform(0.1, 2.0, size=generators) * np.exp(1j * phases)
            rank_one_covariance = np.outer(phasor, phasor.conj())
            rank_one_formula = sensor_power_from_covariance(
                complex_contributions, rank_one_covariance
            )
            rank_one_direct = float(
                np.linalg.norm(complex_contributions @ phasor) ** 2
            )
            maximum_rank_one_error = max(
                maximum_rank_one_error,
                relative_error(rank_one_formula, rank_one_direct),
            )

        direction = self.rng.normal(size=10)
        aligned = np.column_stack((direction, direction, direction))
        cancelling = np.column_stack((direction, -direction))
        aligned_slope = equicorrelated_sensor_power(aligned, 1.0) - (
            equicorrelated_sensor_power(aligned, 0.0)
        )
        cancelling_slope = equicorrelated_sensor_power(cancelling, 1.0) - (
            equicorrelated_sensor_power(cancelling, 0.0)
        )
        coherent_cancellation = sensor_power_from_covariance(
            np.column_stack((direction, direction)),
            np.outer(np.array([1.0, -1.0]), np.array([1.0, -1.0])),
        )
        passed = (
            maximum_error < 1e-11
            and maximum_complex_error < 1e-11
            and maximum_rank_one_error < 1e-11
            and aligned_slope > 0
            and cancelling_slope < 0
            and coherent_cancellation < 1e-12
        )
        self.record(
            "ST-03-synchrony-geometry-interaction",
            passed,
            max_relative_error=f"{maximum_error:.3e}",
            max_complex_covariance_error=f"{maximum_complex_error:.3e}",
            max_rank_one_phasor_error=f"{maximum_rank_one_error:.3e}",
            aligned_power_change=f"{aligned_slope:.6f}",
            cancelling_power_change=f"{cancelling_slope:.6f}",
            coherent_phase_cancellation=f"{coherent_cancellation:.3e}",
        )

    def cortical_confounding(self) -> None:
        maximum_full_rank_residual = 0.0
        maximum_projection_error = 0.0
        maximum_nuisance_leakage = 0.0
        maximum_detector_identity_error = 0.0
        positive_deficient_residuals = []
        for _ in range(self.trials):
            sensors = int(self.rng.integers(3, 25))
            cortical = self.rng.normal(size=(sensors, sensors + 10))
            hippocampal = self.rng.normal(size=(sensors, 4))
            source = self.rng.normal(size=4)
            topography = hippocampal @ source
            precision = random_spd(self.rng, sensors, condition_number=1e4)
            residual, energy = weighted_projection_residual(
                topography, cortical, precision
            )
            maximum_full_rank_residual = max(
                maximum_full_rank_residual, float(np.linalg.norm(residual))
            )

            whitening = np.linalg.cholesky(precision).T
            weighted_cortical = whitening @ cortical
            weighted_topography = whitening @ topography
            projector = weighted_cortical @ np.linalg.pinv(weighted_cortical)
            expected_residual = (np.eye(sensors) - projector) @ weighted_topography
            maximum_projection_error = max(
                maximum_projection_error,
                float(np.linalg.norm(residual - expected_residual)),
                abs(energy - float(expected_residual @ expected_residual)),
            )

            deficient = self.rng.normal(size=(sensors, sensors - 2))
            q, _ = np.linalg.qr(whitening @ deficient, mode="complete")
            adversarial_whitened = q[:, -1]
            adversarial_topography = np.linalg.solve(whitening, adversarial_whitened)
            deficient_residual, deficient_energy = weighted_projection_residual(
                adversarial_topography, deficient, precision
            )
            positive_deficient_residuals.append(deficient_energy)

            # Theorem 4: the residualized statistic must annihilate every
            # cortical amplitude and retain exactly alpha * ||r||^2.
            nuisance_amplitudes = self.rng.normal(size=deficient.shape[1])
            alpha = float(self.rng.normal())
            nuisance_only = deficient @ nuisance_amplitudes
            observation = alpha * adversarial_topography + nuisance_only
            nuisance_leakage = float(
                deficient_residual @ (whitening @ nuisance_only)
            )
            detector_numerator = float(
                deficient_residual @ (whitening @ observation)
            )
            expected_numerator = alpha * deficient_energy
            maximum_nuisance_leakage = max(
                maximum_nuisance_leakage, abs(nuisance_leakage)
            )
            maximum_detector_identity_error = max(
                maximum_detector_identity_error,
                abs(detector_numerator - expected_numerator),
            )

        min_deficient = min(positive_deficient_residuals)
        passed = (
            maximum_full_rank_residual < 2e-9
            and maximum_projection_error < 2e-9
            and min_deficient > 0.99
            and maximum_nuisance_leakage < 2e-9
            and maximum_detector_identity_error < 2e-9
        )
        self.record(
            "ST-04-cortical-observational-equivalence",
            passed,
            max_full_rank_residual=f"{maximum_full_rank_residual:.3e}",
            max_projection_identity_error=f"{maximum_projection_error:.3e}",
            min_rank_deficient_energy=f"{min_deficient:.6f}",
            max_nuisance_leakage=f"{maximum_nuisance_leakage:.3e}",
            max_detector_identity_error=f"{maximum_detector_identity_error:.3e}",
        )

    def gaussian_information_identities(self) -> None:
        max_sensor_source_error = 0.0
        max_posterior_error = 0.0
        condition_numbers = (1.0, 1e2, 1e4, 1e6)
        for trial in range(self.trials):
            sensors = int(self.rng.integers(2, 18))
            sources = int(self.rng.integers(1, 10))
            leadfield = self.rng.normal(size=(sensors, sources))
            condition = condition_numbers[trial % len(condition_numbers)]
            source_covariance = random_spd(self.rng, sources, condition)
            nuisance_covariance = random_spd(self.rng, sensors, condition)
            implementation = gaussian_mi_bits(
                leadfield, source_covariance, nuisance_covariance
            )

            source_sqrt = psd_sqrt(source_covariance)
            source_channel = (
                source_sqrt
                @ leadfield.T
                @ np.linalg.solve(nuisance_covariance, leadfield)
                @ source_sqrt
            )
            source_sign, source_logdet = np.linalg.slogdet(
                np.eye(sources) + source_channel
            )
            if source_sign <= 0:
                raise AssertionError("source channel determinant was non-positive")
            source_value = 0.5 * source_logdet / math.log(2.0)
            max_sensor_source_error = max(
                max_sensor_source_error,
                relative_error(implementation, source_value),
            )

            posterior = gaussian_posterior_covariance(
                leadfield, source_covariance, nuisance_covariance
            )
            prior_logdet = np.linalg.slogdet(source_covariance)[1]
            posterior_logdet = np.linalg.slogdet(posterior)[1]
            posterior_value = 0.5 * (prior_logdet - posterior_logdet) / math.log(2.0)
            max_posterior_error = max(
                max_posterior_error,
                relative_error(implementation, posterior_value),
            )
        passed = max_sensor_source_error < 2e-7 and max_posterior_error < 2e-7
        self.record(
            "ST-05-gaussian-information-identities",
            passed,
            max_sensor_vs_source_relative_error=f"{max_sensor_source_error:.3e}",
            max_mi_vs_posterior_relative_error=f"{max_posterior_error:.3e}",
        )

    def gaussian_detection_divergence(self) -> None:
        maximum_error = 0.0
        for _ in range(self.trials):
            sensors = int(self.rng.integers(2, 20))
            sources = int(self.rng.integers(1, 8))
            leadfield = self.rng.normal(size=(sensors, sources))
            source_covariance = random_spd(self.rng, sources, 1e3)
            nuisance_covariance = random_spd(self.rng, sensors, 1e3)
            signal_covariance = leadfield @ source_covariance @ leadfield.T
            present_covariance = nuisance_covariance + signal_covariance
            direct = 0.5 * (
                np.trace(np.linalg.solve(nuisance_covariance, present_covariance))
                - sensors
                - np.linalg.slogdet(present_covariance)[1]
                + np.linalg.slogdet(nuisance_covariance)[1]
            )
            eigenvalues = whitened_signal_eigenvalues(
                leadfield, source_covariance, nuisance_covariance
            )
            formula = kl_present_vs_absent_nats(eigenvalues)
            maximum_error = max(maximum_error, relative_error(float(direct), formula))
        self.record(
            "ST-06-gaussian-detection-kl",
            maximum_error < 2e-8,
            max_relative_error=f"{maximum_error:.3e}",
        )

    def matched_filter_monte_carlo(self) -> None:
        sensors = 10
        covariance = random_spd(self.rng, sensors, 30.0)
        signal = self.rng.normal(size=sensors)
        signal *= 1.75 / matched_filter_dprime(signal, covariance)
        dprime = matched_filter_dprime(signal, covariance)
        expected_auc = matched_filter_auc(dprime)
        weights = np.linalg.solve(covariance, signal)
        samples = max(50_000, self.trials * 200)
        noise0 = self.rng.multivariate_normal(np.zeros(sensors), covariance, samples)
        noise1 = self.rng.multivariate_normal(np.zeros(sensors), covariance, samples)
        scores0 = noise0 @ weights
        scores1 = (noise1 + signal) @ weights
        empirical_auc = float(np.mean(scores1 > scores0))
        error = abs(empirical_auc - expected_auc)
        self.record(
            "ST-07-matched-filter-monte-carlo",
            error < 0.012,
            dprime=f"{dprime:.6f}",
            theoretical_auc=f"{expected_auc:.6f}",
            empirical_auc=f"{empirical_auc:.6f}",
            absolute_error=f"{error:.6f}",
            samples=samples,
        )

    def electrode_information(self) -> None:
        maximum_monotonicity_violation = 0.0
        for _ in range(self.trials):
            sensors = int(self.rng.integers(4, 35))
            sources = int(self.rng.integers(1, 8))
            leadfield = self.rng.normal(size=(sensors, sources))
            source_covariance = random_spd(self.rng, sources, 100.0)
            nuisance_covariance = random_spd(self.rng, sensors, 30.0)
            order = self.rng.permutation(sensors)
            previous = 0.0
            for count in range(1, sensors + 1):
                current = subset_gaussian_mi_bits(
                    leadfield,
                    source_covariance,
                    nuisance_covariance,
                    order[:count],
                )
                maximum_monotonicity_violation = max(
                    maximum_monotonicity_violation, previous - current
                )
                previous = current

        # Counterexample 1: independent channels with zero source sensitivity add
        # no information, regardless of how many are added.
        informative = np.array([[1.0]])
        padded = np.vstack((informative, np.zeros((31, 1))))
        identity_noise = np.eye(32)
        one_sensor = subset_gaussian_mi_bits(
            padded, np.eye(1), identity_noise, [0]
        )
        all_sensors = subset_gaussian_mi_bits(
            padded, np.eye(1), identity_noise, range(32)
        )

        # Counterexample 2: repeated independently noisy measurements continue to
        # add information: I_N = 0.5 log2(1 + N*SNR).
        repeated = np.ones((256, 1))
        repeated_values = [
            subset_gaussian_mi_bits(repeated, np.eye(1), np.eye(256), range(count))
            for count in (1, 16, 64, 256)
        ]
        repeated_strict = all(
            right > left for left, right in zip(repeated_values, repeated_values[1:])
        )
        passed = (
            maximum_monotonicity_violation < 2e-9
            and abs(one_sensor - all_sensors) < 1e-12
            and repeated_strict
        )
        self.record(
            "ST-08-electrode-information-no-universal-count",
            passed,
            max_monotonicity_violation=f"{maximum_monotonicity_violation:.3e}",
            zero_sensitivity_gain=f"{all_sensors - one_sensor:.3e}",
            repeated_sensor_mi_bits=[round(value, 6) for value in repeated_values],
        )

    def perturbation_bounds(self) -> None:
        maximum_weyl_violation = 0.0
        maximum_mi_violation = 0.0
        maximum_dprime_violation = 0.0
        for _ in range(self.trials):
            dimension = int(self.rng.integers(2, 20))
            rank = int(self.rng.integers(1, dimension + 1))
            nominal_factor = self.rng.normal(size=(dimension, rank))
            true_factor = nominal_factor + self.rng.normal(
                scale=0.05, size=(dimension, rank)
            )
            nominal = nominal_factor @ nominal_factor.T
            true = true_factor @ true_factor.T
            delta = float(np.linalg.norm(true - nominal, ord=2))
            nominal_values = np.linalg.eigvalsh(nominal)[::-1]
            true_values = np.linalg.eigvalsh(true)[::-1]
            maximum_weyl_violation = max(
                maximum_weyl_violation,
                float(np.max(np.abs(true_values - nominal_values))) - delta,
            )
            lower, upper = perturbation_mi_interval_bits(nominal_values, delta)
            true_mi = 0.5 * float(np.sum(np.log1p(true_values))) / math.log(2.0)
            maximum_mi_violation = max(
                maximum_mi_violation, lower - true_mi, true_mi - upper
            )

            sensors = dimension
            sources = int(self.rng.integers(1, 8))
            nominal_forward = self.rng.normal(size=(sensors, sources))
            forward_error = self.rng.normal(scale=0.05, size=(sensors, sources))
            source = self.rng.normal(size=sources)
            nominal_signal = nominal_forward @ source
            true_dprime = float(
                np.linalg.norm((nominal_forward + forward_error) @ source)
            )
            dprime_lower, dprime_upper = forward_error_dprime_interval(
                nominal_signal, forward_error, source
            )
            maximum_dprime_violation = max(
                maximum_dprime_violation,
                dprime_lower - true_dprime,
                true_dprime - dprime_upper,
            )
        passed = (
            maximum_weyl_violation < 1e-9
            and maximum_mi_violation < 1e-9
            and maximum_dprime_violation < 1e-9
        )
        self.record(
            "ST-09-forward-model-perturbation-bounds",
            passed,
            max_weyl_violation=f"{max(0.0, maximum_weyl_violation):.3e}",
            max_mi_interval_violation=f"{max(0.0, maximum_mi_violation):.3e}",
            max_dprime_interval_violation=f"{max(0.0, maximum_dprime_violation):.3e}",
        )

    def spectral_information_rate(self) -> None:
        maximum_error = 0.0
        comparisons = []
        for rho in (0.0, 0.5, 0.9, 0.97):
            finite = ar1_finite_window_mi_rate_bits(
                length=384, rho=rho, gain=0.8, noise_variance=1.3
            )
            spectral = ar1_spectral_mi_rate_bits(
                rho=rho,
                gain=0.8,
                noise_variance=1.3,
                integration_points=65536,
            )
            error = abs(finite - spectral)
            maximum_error = max(maximum_error, error)
            comparisons.append(
                {
                    "rho": rho,
                    "finite_bits_per_sample": round(finite, 8),
                    "spectral_bits_per_sample": round(spectral, 8),
                    "absolute_error": round(error, 8),
                }
            )
        self.record(
            "ST-10-spectral-information-rate",
            maximum_error < 0.006,
            max_absolute_error_bits_per_sample=f"{maximum_error:.6f}",
            comparisons=comparisons,
        )

    def run(self) -> list[dict[str, object]]:
        self.reference_rank()
        self.geometry_bound()
        self.synchrony_geometry_law()
        self.cortical_confounding()
        self.gaussian_information_identities()
        self.gaussian_detection_divergence()
        self.matched_filter_monte_carlo()
        self.electrode_information()
        self.perturbation_bounds()
        self.spectral_information_rate()
        return self.results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    suite = StressSuite(seed=args.seed, trials=args.trials)
    results = suite.run()
    summary = {
        "seed": args.seed,
        "randomized_trials_per_test": args.trials,
        "elapsed_seconds": round(time.time() - started, 3),
        "python": sys.version,
        "numpy": np.__version__,
        "all_passed": all(bool(result["passed"]) for result in results),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
