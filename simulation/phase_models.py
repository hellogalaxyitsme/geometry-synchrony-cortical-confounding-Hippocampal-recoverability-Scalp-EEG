"""Execution logic for the registered synthetic phase-diagram experiment."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .synthetic import (
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


def load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "seed",
        "maximum_contrast_dimension",
        "full_source_elements",
        "phase_diagram",
        "confounding_ladder",
        "montage_uncertainty",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"configuration is missing keys: {sorted(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported configuration schema")
    maximum = int(payload["maximum_contrast_dimension"])
    if maximum < 3:
        raise ValueError("maximum contrast dimension is too small")
    montage = payload["montage_uncertainty"]
    if max(int(value) for value in montage["contrast_dimensions"]) > maximum:
        raise ValueError("a montage contrast dimension exceeds the maximum")
    return payload


def _source_parameters(
    active_fraction: float,
    orientation_span_radians: float,
    synchrony_fraction: float,
    phase_cycles_full_extent: float,
    normalization: str,
) -> dict[str, object]:
    return {
        "active_fraction": float(active_fraction),
        "orientation_span_radians": float(orientation_span_radians),
        "synchrony_fraction": float(synchrony_fraction),
        "phase_cycles_full_extent": float(phase_cycles_full_extent),
        "normalization": normalization,
    }


def _build_source(
    config: Mapping[str, object],
    sensor_modes: np.ndarray,
    parameters: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contributions, positions = structured_contributions(
        sensor_modes=sensor_modes,
        number_of_elements=int(config["full_source_elements"]),
        active_fraction=float(parameters["active_fraction"]),
        orientation_span_radians=float(parameters["orientation_span_radians"]),
        source_gain=float(config["source_gain"]),
        depth_gradient=float(config["depth_gradient"]),
        secondary_geometry_weight=float(config["secondary_geometry_weight"]),
        normalization=str(parameters["normalization"]),
    )
    covariance = source_covariance(
        positions,
        synchrony_fraction=float(parameters["synchrony_fraction"]),
        phase_cycles_full_extent=float(parameters["phase_cycles_full_extent"]),
    )
    signal = signal_covariance(contributions, covariance)
    return contributions, positions, covariance, signal


def _evaluate_subset(
    block: str,
    parameters: Mapping[str, object],
    contributions: np.ndarray,
    source_covariance_matrix: np.ndarray,
    signal: np.ndarray,
    cortical: np.ndarray,
    nuisance: np.ndarray,
    contrast_dimension: int,
) -> dict[str, object]:
    selection = slice(0, contrast_dimension)
    metrics = compute_recoverability_metrics(
        contributions[selection, :],
        signal[selection, selection],
        nuisance[selection, selection],
        cortical[selection, :],
    )
    covariance_eigenvalues = np.linalg.eigvalsh(source_covariance_matrix)
    row: dict[str, object] = {
        "block": block,
        **parameters,
        "contrast_dimension": int(contrast_dimension),
        "electrode_equivalent": int(contrast_dimension + 1),
        "active_elements": int(contributions.shape[1]),
        "source_covariance_trace": float(np.trace(source_covariance_matrix)),
        "source_covariance_min_eigenvalue": float(np.min(covariance_eigenvalues)),
        **metrics,
    }
    return row


def run_phase_diagram(
    config: Mapping[str, object], sensor_modes: np.ndarray
) -> list[dict[str, object]]:
    block = config["phase_diagram"]
    rows: list[dict[str, object]] = []
    keys = (
        "active_fractions",
        "orientation_spans_radians",
        "synchrony_fractions",
        "phase_cycles_full_extent",
        "normalizations",
    )
    grids = [block[key] for key in keys]
    maximum = int(config["maximum_contrast_dimension"])
    base_seed = int(config["seed"])
    # Use one random complement construction throughout this controlled block.
    # The target-dependent first cortical direction still enforces the declared
    # overlap, while parameters that are irrelevant to the signal cannot alter
    # the nuisance model through a seed side channel.
    cortical_seed = stable_seed(base_seed, {"phase_cortical": "common"})
    for active, orientation, synchrony, phase, normalization in product(*grids):
        source_parameters = _source_parameters(
            active, orientation, synchrony, phase, str(normalization)
        )
        contributions, positions, covariance, signal = _build_source(
            config, sensor_modes, source_parameters
        )
        cortical = build_cortical_subspace(
            leading_topography(signal),
            rank=int(block["cortical_rank"]),
            overlap_fraction=float(block["cortical_overlap"]),
            seed=cortical_seed,
        )
        nuisance = nuisance_covariance(
            cortical,
            sensor_noise_variance=float(config["sensor_noise_variance"]),
            cortical_variance=float(config["cortical_variance"]),
        )
        parameters = {
            **source_parameters,
            "cortical_rank": int(block["cortical_rank"]),
            "cortical_overlap": float(block["cortical_overlap"]),
            "forward_error_fraction": 0.0,
            "forward_error_achieved": 0.0,
            "replicate": 0,
            "scenario": "phase_grid",
            "maximum_contrast_dimension": maximum,
        }
        rows.append(
            _evaluate_subset(
                "phase_diagram",
                parameters,
                contributions,
                covariance,
                signal,
                cortical,
                nuisance,
                int(block["contrast_dimension"]),
            )
        )
    return rows


def run_confounding_ladder(
    config: Mapping[str, object], sensor_modes: np.ndarray
) -> list[dict[str, object]]:
    block = config["confounding_ladder"]
    source_parameters = _source_parameters(
        block["active_fraction"],
        block["orientation_span_radians"],
        block["synchrony_fraction"],
        block["phase_cycles_full_extent"],
        str(block["normalization"]),
    )
    contributions, positions, covariance, signal = _build_source(
        config, sensor_modes, source_parameters
    )
    target = leading_topography(signal)
    common_seed = stable_seed(int(config["seed"]), {"ladder": source_parameters})
    rows: list[dict[str, object]] = []
    for rank, overlap in product(
        block["cortical_ranks"], block["cortical_overlaps"]
    ):
        cortical = build_cortical_subspace(
            target,
            rank=int(rank),
            overlap_fraction=float(overlap),
            seed=common_seed,
        )
        nuisance = nuisance_covariance(
            cortical,
            sensor_noise_variance=float(config["sensor_noise_variance"]),
            cortical_variance=float(config["cortical_variance"]),
        )
        parameters = {
            **source_parameters,
            "cortical_rank": int(rank),
            "cortical_overlap": float(overlap),
            "forward_error_fraction": 0.0,
            "forward_error_achieved": 0.0,
            "replicate": 0,
            "scenario": "registered_ladder_source",
            "maximum_contrast_dimension": int(config["maximum_contrast_dimension"]),
        }
        rows.append(
            _evaluate_subset(
                "confounding_ladder",
                parameters,
                contributions,
                covariance,
                signal,
                cortical,
                nuisance,
                int(block["contrast_dimension"]),
            )
        )
    return rows


def run_montage_uncertainty(
    config: Mapping[str, object], sensor_modes: np.ndarray
) -> list[dict[str, object]]:
    block = config["montage_uncertainty"]
    rows: list[dict[str, object]] = []
    base_seed = int(config["seed"])
    for scenario in block["scenarios"]:
        source_parameters = _source_parameters(
            scenario["active_fraction"],
            scenario["orientation_span_radians"],
            scenario["synchrony_fraction"],
            scenario["phase_cycles_full_extent"],
            str(block["normalization"]),
        )
        nominal, positions, covariance, nominal_signal = _build_source(
            config, sensor_modes, source_parameters
        )
        cortical = build_cortical_subspace(
            leading_topography(nominal_signal),
            rank=int(block["cortical_rank"]),
            overlap_fraction=float(block["cortical_overlap"]),
            seed=stable_seed(base_seed, {"montage_cortical": scenario["name"]}),
        )
        nuisance = nuisance_covariance(
            cortical,
            sensor_noise_variance=float(config["sensor_noise_variance"]),
            cortical_variance=float(config["cortical_variance"]),
        )
        for relative_error in block["relative_forward_errors"]:
            replicate_count = (
                1
                if float(relative_error) == 0.0
                else int(block["replicates_per_nonzero_error"])
            )
            for replicate in range(replicate_count):
                perturbation_seed = stable_seed(
                    base_seed,
                    {
                        "scenario": scenario["name"],
                        "relative_error": relative_error,
                        "replicate": replicate,
                    },
                )
                perturbed, achieved = perturb_contributions(
                    nominal, float(relative_error), perturbation_seed
                )
                perturbed_signal = signal_covariance(perturbed, covariance)
                for contrast_dimension in block["contrast_dimensions"]:
                    parameters = {
                        **source_parameters,
                        "cortical_rank": int(block["cortical_rank"]),
                        "cortical_overlap": float(block["cortical_overlap"]),
                        "forward_error_fraction": float(relative_error),
                        "forward_error_achieved": float(achieved),
                        "replicate": int(replicate),
                        "scenario": str(scenario["name"]),
                        "maximum_contrast_dimension": int(
                            config["maximum_contrast_dimension"]
                        ),
                    }
                    rows.append(
                        _evaluate_subset(
                            "montage_uncertainty",
                            parameters,
                            perturbed,
                            covariance,
                            perturbed_signal,
                            cortical,
                            nuisance,
                            int(contrast_dimension),
                        )
                    )
    return rows


def summarize_results(
    config: Mapping[str, object], rows: list[dict[str, object]]
) -> dict[str, object]:
    block_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        block_counts[str(row["block"])] += 1

    phase = [row for row in rows if row["block"] == "phase_diagram"]
    synchrony_groups: dict[tuple[object, ...], dict[float, float]] = defaultdict(dict)
    for row in phase:
        key = (
            row["active_fraction"],
            row["orientation_span_radians"],
            row["phase_cycles_full_extent"],
            row["normalization"],
        )
        synchrony_groups[key][float(row["synchrony_fraction"])] = float(
            row["signal_power"]
        )
    synchrony_differences = [
        values[1.0] - values[0.0]
        for values in synchrony_groups.values()
        if 0.0 in values and 1.0 in values
    ]

    incoherent_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in phase:
        if float(row["synchrony_fraction"]) == 0.0:
            key = (
                row["active_fraction"],
                row["orientation_span_radians"],
                row["normalization"],
            )
            incoherent_groups[key].append(row)
    phase_invariant_metrics = (
        "signal_power",
        "whitened_signal_power",
        "mutual_information_bits",
        "presence_kl_nats",
        "dominant_mode_dprime",
        "masking_residual_fraction",
        "geometry_efficiency",
    )
    incoherent_phase_spreads = [
        max(float(row[metric]) for row in group)
        - min(float(row[metric]) for row in group)
        for group in incoherent_groups.values()
        for metric in phase_invariant_metrics
    ]
    maximum_incoherent_phase_spread = max(incoherent_phase_spreads, default=float("inf"))

    ladder = [row for row in rows if row["block"] == "confounding_ladder"]
    full_rank_threshold = int(config["maximum_contrast_dimension"])
    full_rank_residuals = [
        float(row["masking_residual_fraction"])
        for row in ladder
        if int(row["cortical_rank"]) >= full_rank_threshold
    ]

    montage = [row for row in rows if row["block"] == "montage_uncertainty"]
    montage_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in montage:
        key = (
            row["scenario"],
            row["forward_error_fraction"],
            row["replicate"],
        )
        montage_groups[key].append(row)
    maximum_monotonicity_violation = 0.0
    for group in montage_groups.values():
        ordered = sorted(group, key=lambda item: int(item["contrast_dimension"]))
        for left, right in zip(ordered, ordered[1:]):
            maximum_monotonicity_violation = max(
                maximum_monotonicity_violation,
                float(left["mutual_information_bits"])
                - float(right["mutual_information_bits"]),
            )

    numeric_values = [
        float(value)
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    checks = {
        "all_numeric_outputs_finite": bool(np.all(np.isfinite(numeric_values))),
        "contains_synchrony_amplification": bool(max(synchrony_differences) > 1e-8),
        "contains_synchrony_suppression": bool(min(synchrony_differences) < -1e-8),
        "full_rank_masking_residual_below_tolerance": bool(
            full_rank_residuals and max(full_rank_residuals) < 1e-9
        ),
        "nested_measurement_mi_monotone": maximum_monotonicity_violation < 1e-9,
        "incoherent_limit_is_phase_invariant": maximum_incoherent_phase_spread
        < 1e-12,
    }
    return {
        "schema_version": 1,
        "seed": int(config["seed"]),
        "row_count": len(rows),
        "block_counts": dict(sorted(block_counts.items())),
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
        "diagnostics": {
            "minimum_synchrony_power_change": min(synchrony_differences),
            "maximum_synchrony_power_change": max(synchrony_differences),
            "maximum_full_rank_masking_residual_fraction": max(full_rank_residuals),
            "maximum_nested_measurement_mi_violation_bits": max(
                0.0, maximum_monotonicity_violation
            ),
            "maximum_incoherent_phase_metric_spread": maximum_incoherent_phase_spread,
        },
    }


def run_experiment(
    config: Mapping[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    maximum = int(config["maximum_contrast_dimension"])
    modes = build_sensor_modes(maximum, seed=int(config["seed"]), number_of_modes=3)
    rows = []
    rows.extend(run_phase_diagram(config, modes))
    rows.extend(run_confounding_ladder(config, modes))
    rows.extend(run_montage_uncertainty(config, modes))
    summary = summarize_results(config, rows)
    return rows, summary


def write_rows_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty result table")
    fields = sorted({key for row in materialized for key in row})
    preferred = [
        "block",
        "scenario",
        "normalization",
        "active_fraction",
        "orientation_span_radians",
        "synchrony_fraction",
        "phase_cycles_full_extent",
        "cortical_rank",
        "cortical_overlap",
        "contrast_dimension",
        "electrode_equivalent",
        "forward_error_fraction",
        "replicate",
    ]
    fieldnames = [field for field in preferred if field in fields]
    fieldnames.extend(field for field in fields if field not in fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
