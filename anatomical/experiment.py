"""Registered grid execution for validated anatomical lead-field bundles."""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .analysis import analyze_condition, build_cortical_restriction
from .bundle import AnatomicalBundle
from .operators import density_covariance, farthest_point_order


def run_anatomical_grid(
    bundle: AnatomicalBundle, config: Mapping[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run the mock-registered factors against any compatible bundle."""

    sensor_counts = [int(value) for value in config["sensor_counts"]]
    if min(sensor_counts) < 2 or max(sensor_counts) > bundle.number_of_sensors:
        raise ValueError("sensor_counts are incompatible with the bundle")
    if max(int(value) for value in config["cortical_ranks"]) > (
        bundle.number_of_sensors - 1
    ):
        raise ValueError("a cortical rank exceeds referenced sensor dimension")

    order = farthest_point_order(bundle.electrode_positions_m)
    restrictions = {
        (int(rank), str(variance_normalization)): build_cortical_restriction(
            bundle,
            rank=int(rank),
            cortical_total_variance=float(config["cortical_total_variance"]),
            normalization="area_average",
            variance_normalization=str(variance_normalization),
        )
        for rank, variance_normalization in product(
            config["cortical_ranks"],
            config["cortical_variance_normalizations"],
        )
    }
    covariance_cache: dict[tuple[float, float, float], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    spectra: list[dict[str, object]] = []
    grid = product(
        config["weight_normalizations"],
        config["coherence_lengths_m"],
        config["synchrony_fractions"],
        config["phase_cycles"],
        config["cortical_ranks"],
        config["cortical_variance_normalizations"],
        sensor_counts,
    )
    for condition_id, (
        normalization,
        coherence_length,
        synchrony,
        phase_cycles,
        cortical_rank,
        cortical_variance_normalization,
        sensor_count,
    ) in enumerate(grid):
        covariance_key = (
            float(coherence_length),
            float(synchrony),
            float(phase_cycles),
        )
        if covariance_key not in covariance_cache:
            covariance_cache[covariance_key] = density_covariance(
                bundle.hippocampal_positions_m,
                bundle.hippocampal_longitudinal_coordinate,
                coherence_length_m=float(coherence_length),
                synchrony_fraction=float(synchrony),
                phase_cycles=float(phase_cycles),
                source_density_scale=float(config["source_density_scale"]),
            )
        covariance = covariance_cache[covariance_key]
        indices = order[: int(sensor_count)]
        metrics, eigenvalues = analyze_condition(
            bundle,
            covariance,
            str(normalization),
            restrictions[
                (int(cortical_rank), str(cortical_variance_normalization))
            ],
            indices,
        )
        covariance_eigenvalues = np.linalg.eigvalsh(covariance)
        row = {
            "condition_id": condition_id,
            "subject_id": bundle.manifest["subject_id"],
            "arrays_sha256": bundle.manifest["arrays_sha256"],
            "normalization": str(normalization),
            "coherence_length_m": float(coherence_length),
            "synchrony_fraction": float(synchrony),
            "phase_cycles": float(phase_cycles),
            "cortical_rank": int(cortical_rank),
            "cortical_variance_normalization": str(
                cortical_variance_normalization
            ),
            "sensor_count": int(sensor_count),
            "contrast_dimension": int(sensor_count) - 1,
            "montage": f"mock_farthest_{int(sensor_count)}",
            "source_covariance_min_eigenvalue": float(
                np.min(covariance_eigenvalues)
            ),
            "source_covariance_max_diagonal_error": float(
                np.max(
                    np.abs(
                        np.diag(covariance)
                        - float(config["source_density_scale"]) ** 2
                    )
                )
            ),
            **metrics,
        }
        rows.append(row)
        for eigen_index, value in enumerate(eigenvalues):
            spectra.append(
                {
                    "condition_id": condition_id,
                    "eigenvalue_index": eigen_index,
                    "recoverability_eigenvalue": float(value),
                }
            )

    summary = summarize_anatomical_grid(bundle, config, rows)
    return rows, spectra, summary


def summarize_anatomical_grid(
    bundle: AnatomicalBundle,
    config: Mapping[str, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    numeric = [
        float(value)
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    maximum_mi_identity_error = max(
        float(row["mi_identity_absolute_error_bits"]) for row in rows
    )
    maximum_covariance_diagonal_error = max(
        float(row["source_covariance_max_diagonal_error"]) for row in rows
    )
    minimum_covariance_eigenvalue = min(
        float(row["source_covariance_min_eigenvalue"]) for row in rows
    )

    full_rank = bundle.number_of_sensors - 1
    full_rank_residuals = [
        float(row["masking_residual_fraction"])
        for row in rows
        if int(row["cortical_rank"]) == full_rank
        and int(row["sensor_count"]) == bundle.number_of_sensors
    ]

    nested_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["normalization"],
            row["coherence_length_m"],
            row["synchrony_fraction"],
            row["phase_cycles"],
            row["cortical_rank"],
            row["cortical_variance_normalization"],
        )
        nested_groups[key].append(row)
    maximum_monotonicity_violation = 0.0
    for group in nested_groups.values():
        ordered = sorted(group, key=lambda item: int(item["sensor_count"]))
        for left, right in zip(ordered, ordered[1:]):
            maximum_monotonicity_violation = max(
                maximum_monotonicity_violation,
                float(left["mutual_information_bits"])
                - float(right["mutual_information_bits"]),
            )

    synchrony_groups: dict[tuple[object, ...], dict[float, float]] = defaultdict(dict)
    for row in rows:
        key = (
            row["normalization"],
            row["coherence_length_m"],
            row["phase_cycles"],
            row["cortical_rank"],
            row["cortical_variance_normalization"],
            row["sensor_count"],
        )
        synchrony_groups[key][float(row["synchrony_fraction"])] = float(
            row["signal_power"]
        )
    synchrony_differences = [
        values[1.0] - values[0.0]
        for values in synchrony_groups.values()
        if 0.0 in values and 1.0 in values
    ]

    checks = {
        "all_numeric_outputs_finite": bool(np.all(np.isfinite(numeric))),
        "source_covariance_psd": minimum_covariance_eigenvalue > -1e-9,
        "source_covariance_fixed_diagonal": maximum_covariance_diagonal_error
        < 1e-12,
        "direct_and_spectral_mi_agree": maximum_mi_identity_error < 1e-9,
        "full_rank_cortical_masking": bool(
            full_rank_residuals and max(full_rank_residuals) < 1e-9
        ),
        "nested_montage_mi_monotone": maximum_monotonicity_violation < 1e-9,
        "contains_synchrony_amplification": max(synchrony_differences) > 1e-12,
        "contains_synchrony_suppression": min(synchrony_differences) < -1e-12,
    }
    return {
        "schema_version": 1,
        "seed": int(config["seed"]),
        "subject_id": bundle.manifest["subject_id"],
        "arrays_sha256": bundle.manifest["arrays_sha256"],
        "row_count": len(rows),
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
        "diagnostics": {
            "minimum_source_covariance_eigenvalue": minimum_covariance_eigenvalue,
            "maximum_source_covariance_diagonal_error": (
                maximum_covariance_diagonal_error
            ),
            "maximum_mi_identity_error_bits": maximum_mi_identity_error,
            "maximum_full_rank_masking_residual_fraction": max(
                full_rank_residuals
            ),
            "maximum_nested_mi_violation_bits": max(
                0.0, maximum_monotonicity_violation
            ),
            "minimum_synchrony_power_change": min(synchrony_differences),
            "maximum_synchrony_power_change": max(synchrony_differences),
        },
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    fields = sorted({field for row in materialized for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
