#!/usr/bin/env python3
"""Build one subject's frozen source-ensemble HippUnfold source ensembles."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Iterable

import mne
from mne.io.constants import FIFF
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from anatomical.hippunfold_ensembles import (
    SourceSupport,
    bilateral_interference,
    build_source_supports,
    deterministic_phase,
    oriented_intrinsic_coordinates,
    patch_coherence_metrics,
    phase_locked_metrics,
    project_free_cartesian_leadfield,
    random_correlated_phases,
    referenced_leadfield,
    standardized_nuisance,
    support_identifier,
    tilted_directions,
    whitened_subspace_projection,
)


PROTOCOL = "hcp/source-ensembles-v1.1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path, required: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in required}
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"{path} contains non-finite arrays")
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _support_fields(support: SourceSupport) -> dict[str, object]:
    return {
        "support_id": support.identifier,
        "hemisphere": support.hemisphere,
        "location": support.location,
        "target_extent": support.target_extent,
        "source_count": len(support.indices),
        "full_sheet_area_fraction": support.full_sheet_area_fraction,
        "realized_extent_within_selected_hemispheres": (
            support.realized_extent_within_selected_hemispheres
        ),
        "realized_extent_by_hemisphere_json": json.dumps(
            support.realized_extent_by_hemisphere, sort_keys=True, separators=(",", ":")
        ),
    }


def _regimes(cycles: Iterable[float]) -> list[tuple[str, float]]:
    return [("zero_phase", 0.0)] + [
        (f"wave_{cycle:g}_cycles", float(cycle)) for cycle in cycles
    ]


def _canonical_support(config_row: dict[str, object]) -> str:
    return support_identifier(
        str(config_row["hemisphere"]),
        str(config_row["location"]),
        float(config_row["extent"]),
    )


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected)) / scale)


def _finite_rows(rows: list[dict[str, object]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return False
    return True


def build(
    subject: str,
    config_path: Path,
    forward_root: Path,
    legacy_root: Path,
    output_directory: Path,
) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected source-ensemble protocol")
    if output_directory.exists():
        raise FileExistsError(f"output already exists: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=False)

    subject_forward = forward_root / subject
    subject_legacy = legacy_root / subject
    paths = {
        "forward_models_report": subject_forward / "report.json",
        "forward_models_marker": subject_forward / ".hippunfold_forward.json",
        "fixed_leadfield": subject_forward / "hippunfold-hippocampal-fixed-leadfield.npy",
        "free_forward": subject_forward / "hippunfold-hippocampal-fwd.fif",
        "metadata": subject_forward / "hippunfold-hippocampal-source-metadata.npz",
        "cortical_leadfield": subject_legacy / "cortical-fixed-leadfield.npy",
        "cortical_metadata": subject_legacy / "cortical-source-metadata.npz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source-ensemble inputs: {missing}")

    forward_models_report = json.loads(paths["forward_models_report"].read_text(encoding="utf-8"))
    forward_models_marker = json.loads(paths["forward_models_marker"].read_text(encoding="utf-8"))
    if (
        forward_models_report.get("ok") is not True
        or forward_models_report.get("subject") != subject
        or forward_models_report.get("protocol") != config["forward_models_protocol"]
        or forward_models_report["source_model"]["dentate"]["included_in_primary_source_model"]
        is not False
        or forward_models_marker.get("status") != "complete"
        or forward_models_marker.get("report_sha256") != _sha256(paths["forward_models_report"])
    ):
        raise ValueError("forward-model subject provenance is not complete")

    metadata = _load_npz(
        paths["metadata"],
        (
            "positions_m",
            "directions",
            "area_weights_m2",
            "longitudinal_coordinate",
            "proximal_distal_coordinate",
            "hemisphere_code",
        ),
    )
    cortical_metadata = _load_npz(paths["cortical_metadata"], ("area_weights_m2",))
    fixed = np.load(paths["fixed_leadfield"], allow_pickle=False)
    cortical = np.load(paths["cortical_leadfield"], mmap_mode="r", allow_pickle=False)
    source_count = len(metadata["area_weights_m2"])
    sensors = int(config["expected_sensors"])
    if fixed.shape != (sensors, source_count) or not np.all(np.isfinite(fixed)):
        raise ValueError("fixed HippUnfold lead field is invalid")
    if cortical.shape[0] != sensors or not np.all(np.isfinite(cortical)):
        raise ValueError("cortical lead field is invalid")

    restored = mne.read_forward_solution(paths["free_forward"], verbose=False)
    if (
        int(restored["source_ori"]) != int(FIFF.FIFFV_MNE_FREE_ORI)
        or bool(restored["surf_ori"])
        or restored["sol"]["data"].shape != (sensors, 3 * source_count)
    ):
        raise ValueError("serialized MNE forward does not expose Cartesian free orientations")
    free = np.asarray(restored["sol"]["data"], dtype=np.float64)
    projected = project_free_cartesian_leadfield(free, metadata["directions"])
    projection_error = _relative_error(projected, fixed)
    if projection_error > float(config["maximum_fixed_projection_relative_error"]):
        raise ValueError("free-orientation projection does not reproduce forward-model")

    support_config = dict(config["support"])
    anterior, pd, coordinate_report = oriented_intrinsic_coordinates(
        metadata["positions_m"],
        metadata["longitudinal_coordinate"],
        metadata["proximal_distal_coordinate"],
        metadata["hemisphere_code"],
        metadata["area_weights_m2"],
        endpoint_decile=float(support_config["endpoint_decile"]),
        minimum_endpoint_separation_m=float(
            support_config["minimum_endpoint_separation_m"]
        ),
    )
    supports, support_report = build_source_supports(
        anterior,
        pd,
        metadata["hemisphere_code"],
        metadata["area_weights_m2"],
        hemisphere_families=support_config["hemisphere_families"],
        focal_locations=support_config["focal_locations"],
        focal_extents=support_config["focal_extents"],
        include_whole_extent=bool(support_config["include_whole_extent"]),
    )
    overshoot_limit = float(
        support_config["maximum_fraction_overshoot_patch_multiplier"]
    )
    maximum_support_overshoot = 0.0
    for support in supports.values():
        if support.location == "whole":
            continue
        for realized in support.realized_extent_by_hemisphere.values():
            overshoot = realized - support.target_extent
            maximum_support_overshoot = max(maximum_support_overshoot, overshoot)
            if (
                overshoot
                > overshoot_limit * support.maximum_selected_hemisphere_patch_fraction
                + 1e-12
            ):
                raise ValueError("support area overshot by more than one source patch")

    hippocampal_referenced = referenced_leadfield(fixed)
    cholesky, h_trace, c_trace, nuisance_report = standardized_nuisance(
        hippocampal_referenced,
        metadata["area_weights_m2"],
        np.asarray(cortical),
        cortical_metadata["area_weights_m2"],
        float(config["standardized_noise_fraction"]),
    )

    regimes = _regimes(config["deterministic_phase"]["wave_cycles"])
    deterministic_rows: list[dict[str, object]] = []
    factor_cache: dict[tuple[str, str], np.ndarray] = {}
    maximum_bound_excess = 0.0
    maximum_wave_reversal_error = 0.0
    for identifier in sorted(supports):
        support = supports[identifier]
        for regime, cycles in regimes:
            phase = deterministic_phase(anterior[support.indices], cycles)
            metrics, factor = phase_locked_metrics(
                hippocampal_referenced,
                support,
                phase,
                h_trace,
                cholesky,
            )
            maximum_bound_excess = max(
                maximum_bound_excess, float(metrics["analytical_bound_excess"])
            )
            row = {
                "subject": subject,
                **_support_fields(support),
                "regime": regime,
                "wave_cycles": cycles,
                **metrics,
            }
            deterministic_rows.append(row)
            factor_cache[(identifier, regime)] = factor
            if cycles != 0.0 and support.location == "whole":
                reverse_phase = deterministic_phase(
                    anterior[support.indices], -cycles
                )
                _, reverse = phase_locked_metrics(
                    hippocampal_referenced,
                    support,
                    reverse_phase,
                    h_trace,
                    cholesky,
                )
                covariance_error = float(
                    np.max(
                        np.abs(
                            factor @ factor.T - reverse @ reverse.T
                        )
                    )
                    / max(float(np.max(np.abs(factor @ factor.T))), np.finfo(float).tiny)
                )
                maximum_wave_reversal_error = max(
                    maximum_wave_reversal_error, covariance_error
                )

    coherence_rows: list[dict[str, object]] = []
    for identifier in sorted(supports):
        support = supports[identifier]
        for width in config["patch_coherence"]["intrinsic_bin_widths"]:
            metrics, _ = patch_coherence_metrics(
                hippocampal_referenced,
                support,
                anterior,
                pd,
                metadata["hemisphere_code"],
                float(width),
                h_trace,
                cholesky,
            )
            maximum_bound_excess = max(
                maximum_bound_excess, float(metrics["analytical_bound_excess"])
            )
            coherence_rows.append(
                {
                    "subject": subject,
                    **_support_fields(support),
                    "intrinsic_bin_width": float(width),
                    **metrics,
                }
            )

    random_config = dict(config["random_phase"])
    random_rows: list[dict[str, object]] = []
    maximum_unit_phase_error = 0.0
    for support_row in random_config["supports"]:
        identifier = _canonical_support(support_row)
        support = supports[identifier]
        local = support.indices
        for length in random_config["intrinsic_correlation_lengths"]:
            for replicate in range(int(random_config["replicates"])):
                phase, phase_report = random_correlated_phases(
                    anterior[local],
                    pd[local],
                    metadata["hemisphere_code"][local],
                    support.conditional_weights,
                    correlation_length=float(length),
                    features=int(random_config["random_fourier_features"]),
                    phase_standard_deviation_rad=float(
                        random_config["marginal_phase_standard_deviation_rad"]
                    ),
                    master_seed=int(random_config["master_seed"]),
                    seed_components=(subject, identifier, float(length), replicate),
                )
                metrics, _ = phase_locked_metrics(
                    hippocampal_referenced,
                    support,
                    phase,
                    h_trace,
                    cholesky,
                )
                maximum_bound_excess = max(
                    maximum_bound_excess, float(metrics["analytical_bound_excess"])
                )
                maximum_unit_phase_error = max(
                    maximum_unit_phase_error,
                    float(phase_report["unit_complex_magnitude_maximum_error"]),
                )
                random_rows.append(
                    {
                        "subject": subject,
                        **_support_fields(support),
                        "intrinsic_correlation_length": float(length),
                        "replicate": replicate,
                        "phase_seed_manifest_json": json.dumps(
                            phase_report["hemispheres"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        **metrics,
                    }
                )

    mismatch_config = dict(config["orientation_mismatch"])
    mismatch_rows: list[dict[str, object]] = []
    zero_degree_referenced = referenced_leadfield(projected)
    maximum_zero_degree_projection_loss = 0.0
    tilted_cache: dict[float, tuple[np.ndarray, dict[str, float]]] = {}
    for angle in mismatch_config["angles_degrees"]:
        tilted, tilt_report = tilted_directions(metadata["directions"], float(angle))
        tilted_fixed = project_free_cartesian_leadfield(free, tilted)
        tilted_cache[float(angle)] = (referenced_leadfield(tilted_fixed), tilt_report)
    for support_row in mismatch_config["supports"]:
        identifier = _canonical_support(support_row)
        support = supports[identifier]
        for regime, cycles in regimes:
            phase = deterministic_phase(anterior[support.indices], cycles)
            truth = factor_cache[(identifier, regime)]
            _, zero_degree_factor = phase_locked_metrics(
                zero_degree_referenced,
                support,
                phase,
                h_trace,
                cholesky,
            )
            zero_control = whitened_subspace_projection(
                truth, zero_degree_factor, cholesky
            )
            maximum_zero_degree_projection_loss = max(
                maximum_zero_degree_projection_loss,
                1.0
                - float(zero_control["whitened_truth_projection_efficiency"]),
            )
            for angle in mismatch_config["angles_degrees"]:
                tilted_referenced, tilt_report = tilted_cache[float(angle)]
                _, assumed = phase_locked_metrics(
                    tilted_referenced,
                    support,
                    phase,
                    h_trace,
                    cholesky,
                )
                mismatch_rows.append(
                    {
                        "subject": subject,
                        **_support_fields(support),
                        "regime": regime,
                        "wave_cycles": cycles,
                        "mismatch_angle_degrees": float(angle),
                        **tilt_report,
                        **whitened_subspace_projection(truth, assumed, cholesky),
                    }
                )

    interference_rows: list[dict[str, object]] = []
    support_shapes = [("whole", 1.0)] + [
        (location, float(extent))
        for location in support_config["focal_locations"]
        for extent in support_config["focal_extents"]
    ]
    maximum_bilateral_identity_error = 0.0
    for location, extent in support_shapes:
        left_id = support_identifier("left", location, extent)
        right_id = support_identifier("right", location, extent)
        bilateral_id = support_identifier("bilateral", location, extent)
        for regime, cycles in regimes:
            left = factor_cache[(left_id, regime)] * supports[left_id].full_sheet_area_fraction
            right = factor_cache[(right_id, regime)] * supports[right_id].full_sheet_area_fraction
            bilateral = (
                factor_cache[(bilateral_id, regime)]
                * supports[bilateral_id].full_sheet_area_fraction
            )
            identity_error = _relative_error(bilateral, left + right)
            maximum_bilateral_identity_error = max(
                maximum_bilateral_identity_error, identity_error
            )
            interference_rows.append(
                {
                    "subject": subject,
                    "location": location,
                    "target_extent": extent,
                    "regime": regime,
                    "wave_cycles": cycles,
                    "bilateral_factor_identity_relative_error": identity_error,
                    **bilateral_interference(left, right),
                }
            )

    expected_counts = {
        "supports": 39,
        "deterministic": 39 * len(regimes),
        "patch_coherence": 39
        * len(config["patch_coherence"]["intrinsic_bin_widths"]),
        "random_phase": len(random_config["supports"])
        * len(random_config["intrinsic_correlation_lengths"])
        * int(random_config["replicates"]),
        "orientation_mismatch": len(mismatch_config["supports"])
        * len(regimes)
        * len(mismatch_config["angles_degrees"]),
        "bilateral_interference": len(support_shapes) * len(regimes),
    }
    actual_counts = {
        "supports": len(supports),
        "deterministic": len(deterministic_rows),
        "patch_coherence": len(coherence_rows),
        "random_phase": len(random_rows),
        "orientation_mismatch": len(mismatch_rows),
        "bilateral_interference": len(interference_rows),
    }
    tolerance = float(config["maximum_numerical_bound_error"])
    qc = {
        "forward_models_report_ok": True,
        "dentate_excluded": True,
        "fixed_projection_reproduced": projection_error
        <= float(config["maximum_fixed_projection_relative_error"]),
        "AP_polarity_auditable": all(
            row["endpoint_separation_m"]
            >= float(support_config["minimum_endpoint_separation_m"])
            for row in coordinate_report["hemispheres"]
        ),
        "support_count_exact": actual_counts["supports"] == expected_counts["supports"],
        "supports_exactly_nested": support_report["all_exactly_nested"] is True,
        "support_overshoot_bounded": all(
            support.location == "whole"
            or all(
                realized - support.target_extent
                <= overshoot_limit
                * support.maximum_selected_hemisphere_patch_fraction
                + 1e-12
                for realized in support.realized_extent_by_hemisphere.values()
            )
            for support in supports.values()
        ),
        "condition_counts_exact": actual_counts == expected_counts,
        "all_rows_finite": all(
            _finite_rows(rows)
            for rows in (
                deterministic_rows,
                coherence_rows,
                random_rows,
                mismatch_rows,
                interference_rows,
            )
        ),
        "analytical_power_bounds": maximum_bound_excess <= tolerance,
        "random_phase_unit_magnitude": maximum_unit_phase_error <= 1e-12,
        "wave_direction_covariance_invariant": maximum_wave_reversal_error <= 1e-10,
        "bilateral_factor_identity": maximum_bilateral_identity_error <= 1e-10,
        "orientation_90_degree_orthogonality": max(
            abs(tilted_cache[90.0][1][key] - 90.0)
            for key in (
                "minimum_realized_angle_degrees",
                "maximum_realized_angle_degrees",
            )
        )
        <= 1e-8,
        "orientation_zero_degree_projection_identity": (
            maximum_zero_degree_projection_loss <= 1e-10
        ),
    }
    if not all(qc.values()):
        raise ValueError(f"source-ensemble subject QC failed: {qc}")

    tables = {
        "deterministic_conditions.csv": deterministic_rows,
        "patch_coherence_conditions.csv": coherence_rows,
        "random_phase_conditions.csv": random_rows,
        "orientation_mismatch_conditions.csv": mismatch_rows,
        "bilateral_interference.csv": interference_rows,
    }
    for name, rows in tables.items():
        _write_csv(output_directory / name, rows)

    report = {
        "schema_version": 1,
        "ok": True,
        "protocol": PROTOCOL,
        "subject": subject,
        "created_at_utc": _now(),
        "scope": "source-ensemble numerical source ensembles; no physiological calibration",
        "python": platform.python_version(),
        "mne": mne.__version__,
        "numpy": np.__version__,
        "source_model": {
            "family": forward_models_report["source_model"]["primary"],
            "orientation": forward_models_report["source_model"]["orientation"],
            "histological_laminar_ground_truth": False,
            "sources": source_count,
            "dentate_included": False,
        },
        "coordinate_orientation": coordinate_report,
        "supports": {
            **support_report,
            "maximum_realized_fraction_overshoot": maximum_support_overshoot,
        },
        "nuisance": nuisance_report,
        "free_orientation_projection": {
            "source_orientation": "Cartesian XYZ",
            "relative_error_against_forward_models_fixed": projection_error,
        },
        "counts": {"expected": expected_counts, "actual": actual_counts},
        "numerical_diagnostics": {
            "maximum_analytical_power_bound_excess": maximum_bound_excess,
            "maximum_random_phase_unit_magnitude_error": maximum_unit_phase_error,
            "maximum_wave_reversal_covariance_relative_error": maximum_wave_reversal_error,
            "maximum_bilateral_factor_identity_relative_error": (
                maximum_bilateral_identity_error
            ),
            "maximum_zero_degree_whitened_projection_loss": (
                maximum_zero_degree_projection_loss
            ),
        },
        "qc": qc,
        "inputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "outputs": {
            name: {
                "bytes": (output_directory / name).stat().st_size,
                "sha256": _sha256(output_directory / name),
                "rows": len(rows),
            }
            for name, rows in tables.items()
        },
        "physiological_inference_authorized": False,
        "shared_storage_touched": False,
    }
    report_path = output_directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--forward-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.subject,
        args.config.resolve(strict=True),
        args.forward_root.resolve(strict=True),
        args.legacy_root.resolve(strict=True),
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
