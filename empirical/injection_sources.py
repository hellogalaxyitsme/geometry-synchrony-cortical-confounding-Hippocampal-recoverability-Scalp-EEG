"""Frozen multi-source hippocampal injection library for injection-benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from anatomical.hippunfold_ensembles import (
    build_source_supports,
    deterministic_phase,
    oriented_intrinsic_coordinates,
    random_correlated_phases,
    support_identifier,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InjectionSource:
    identifier: str
    family: str
    hemisphere: str
    location: str
    extent: float
    phase_model: str
    phase_parameter: float
    factor: FloatArray


def _sensor_factor(
    leadfield: FloatArray,
    indices: NDArray[np.int64],
    weights: FloatArray,
    full_sheet_fraction: float,
    phase: ArrayLike,
) -> FloatArray:
    angle = np.asarray(phase, dtype=np.float64).reshape(-1)
    if len(angle) != len(indices) or not np.all(np.isfinite(angle)):
        raise ValueError("phase vector and source support disagree")
    selected = np.asarray(leadfield[:, indices], dtype=np.float64)
    factor = float(full_sheet_fraction) * np.column_stack(
        (selected @ (weights * np.cos(angle)), selected @ (weights * np.sin(angle)))
    )
    factor -= factor.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(factor, axis=0)
    keep = norms > 1e-12 * max(float(np.max(norms)), np.finfo(float).tiny)
    if not np.any(keep):
        raise ValueError("injection source has zero referenced sensor energy")
    return np.asarray(factor[:, keep], dtype=np.float64)


def build_injection_sources(
    leadfield: ArrayLike,
    metadata: dict[str, np.ndarray],
    design: dict[str, Any],
    donor_subject: str,
    master_seed: int,
) -> tuple[list[InjectionSource], dict[str, object]]:
    """Build the exact deterministic/random source list declared by config."""

    matrix = np.asarray(leadfield, dtype=np.float64)
    required = {
        "positions_m", "area_weights_m2", "longitudinal_coordinate",
        "proximal_distal_coordinate", "hemisphere_code",
    }
    if required.difference(metadata):
        raise ValueError("hippocampal source metadata is incomplete")
    if matrix.ndim != 2 or matrix.shape[1] != len(metadata["area_weights_m2"]):
        raise ValueError("hippocampal lead field and metadata are misaligned")
    support_design = dict(design["support_construction"])
    anterior, pd, coordinate_report = oriented_intrinsic_coordinates(
        metadata["positions_m"],
        metadata["longitudinal_coordinate"],
        metadata["proximal_distal_coordinate"],
        metadata["hemisphere_code"],
        metadata["area_weights_m2"],
        endpoint_decile=float(support_design["endpoint_decile"]),
        minimum_endpoint_separation_m=float(support_design["minimum_endpoint_separation_m"]),
    )
    supports, support_report = build_source_supports(
        anterior,
        pd,
        metadata["hemisphere_code"],
        metadata["area_weights_m2"],
        hemisphere_families=("left", "right", "bilateral"),
        focal_locations=("anterior", "middle", "posterior"),
        focal_extents=(0.05, 0.10, 0.25, 0.50),
        include_whole_extent=True,
    )
    rows = list(design["sources"])
    identifiers = [str(row["id"]) for row in rows]
    if len(rows) < 2 or len(set(identifiers)) != len(rows):
        raise ValueError("source ensemble IDs must be unique")
    result: list[InjectionSource] = []
    maximum_unit_phase_error = 0.0
    for row in rows:
        hemisphere = str(row["hemisphere"])
        location = str(row["location"])
        extent = float(row["extent"])
        support_id = support_identifier(hemisphere, location, extent)
        if support_id not in supports:
            raise ValueError(f"source ensemble requests absent support: {support_id}")
        support = supports[support_id]
        model = str(row["phase_model"])
        if model == "deterministic_wave":
            parameter = float(row["wave_cycles"])
            phase = deterministic_phase(anterior[support.indices], parameter)
            family = "deterministic"
        elif model == "correlated_random":
            parameter = float(row["correlation_length"])
            settings = dict(design["random_phase"])
            phase, phase_report = random_correlated_phases(
                anterior[support.indices],
                pd[support.indices],
                metadata["hemisphere_code"][support.indices],
                support.conditional_weights,
                correlation_length=parameter,
                features=int(settings["random_fourier_features"]),
                phase_standard_deviation_rad=float(settings["phase_standard_deviation_rad"]),
                master_seed=int(master_seed),
                seed_components=(donor_subject, str(row["id"])),
            )
            maximum_unit_phase_error = max(
                maximum_unit_phase_error,
                float(phase_report["unit_complex_magnitude_maximum_error"]),
            )
            family = "random_correlated"
        else:
            raise ValueError(f"unknown source phase model: {model}")
        factor = _sensor_factor(
            matrix,
            support.indices,
            support.conditional_weights,
            support.full_sheet_area_fraction,
            phase,
        )
        result.append(
            InjectionSource(
                identifier=str(row["id"]),
                family=family,
                hemisphere=hemisphere,
                location=location,
                extent=extent,
                phase_model=model,
                phase_parameter=parameter,
                factor=factor,
            )
        )
    return result, {
        "source_count": len(result),
        "source_ids": identifiers,
        "all_supports_exactly_nested": bool(support_report["all_exactly_nested"]),
        "maximum_random_phase_unit_error": maximum_unit_phase_error,
        "coordinate_orientation": coordinate_report,
    }

