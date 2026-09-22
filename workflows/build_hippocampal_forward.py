#!/usr/bin/env python3
"""Container entry point for one paired HippUnfold/legacy HCP forward model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import mne
from mne.io.constants import FIFF
import nibabel as nib
import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from anatomical.convergence import sensor_operators, standardized_recoverability_spectrum
from anatomical.hcp_forward import (
    _compute_fixed_forward,
    _make_info,
    _referenced_rank,
    _source_containment,
    _validate_forward_roundtrip,
    register_standard_montage,
    scanner_ras_to_bem_transform,
    sha256_file,
)
from anatomical.hippunfold import hippunfold_source_hierarchy


def _npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        result = {name: np.asarray(archive[name]) for name in required}
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"{path} contains non-finite arrays")
    return result


def _registered_montage(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    names = [row["name"] for row in rows]
    positions = np.asarray(
        [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in rows]
    )
    if len(names) < 2 or len(set(names)) != len(names) or not np.all(np.isfinite(positions)):
        raise ValueError("legacy registered montage is invalid")
    return names, positions


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _column_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)
    if np.any(denominator <= 0.0):
        raise ValueError("cannot compare zero lead-field columns")
    return np.sum(left * right, axis=0) / denominator


def _source_agreement(
    new_metadata: dict[str, np.ndarray],
    old_metadata: dict[str, np.ndarray],
    new_leadfield: np.ndarray,
    old_leadfield: np.ndarray,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    all_new_distances = []
    all_old_distances = []
    all_orientation = []
    all_topography = []
    for hemisphere_code, hemisphere in ((-1, "L"), (1, "R")):
        new_indices = np.flatnonzero(new_metadata["hemisphere_code"] == hemisphere_code)
        old_indices = np.flatnonzero(old_metadata["hemisphere_code"] == hemisphere_code)
        if len(new_indices) == 0 or len(old_indices) == 0:
            raise ValueError("one source model lacks a hemisphere")
        old_tree = cKDTree(old_metadata["positions_m"][old_indices])
        new_to_old_distance, local_old = old_tree.query(
            new_metadata["positions_m"][new_indices], k=1, workers=1
        )
        nearest_old = old_indices[np.asarray(local_old, dtype=int)]
        new_tree = cKDTree(new_metadata["positions_m"][new_indices])
        old_to_new_distance, _ = new_tree.query(
            old_metadata["positions_m"][old_indices], k=1, workers=1
        )
        orientation = np.sum(
            new_metadata["directions"][new_indices]
            * old_metadata["directions"][nearest_old],
            axis=1,
        )
        topography = _column_cosines(
            new_leadfield[:, new_indices], old_leadfield[:, nearest_old]
        )
        records.append(
            {
                "hemisphere": hemisphere,
                "new_sources": int(len(new_indices)),
                "old_sources": int(len(old_indices)),
                "new_to_old_nearest_distance_m": _quantiles(new_to_old_distance),
                "old_to_new_nearest_distance_m": _quantiles(old_to_new_distance),
                "signed_orientation_cosine": _quantiles(orientation),
                "absolute_orientation_cosine": _quantiles(np.abs(orientation)),
                "signed_nearest_topography_cosine": _quantiles(topography),
                "absolute_nearest_topography_cosine": _quantiles(np.abs(topography)),
            }
        )
        all_new_distances.append(new_to_old_distance)
        all_old_distances.append(old_to_new_distance)
        all_orientation.append(orientation)
        all_topography.append(topography)
    return {
        "matching": "within-hemisphere Euclidean nearest neighbour; no anatomical correspondence asserted",
        "hemispheres": records,
        "pooled_new_to_old_nearest_distance_m": _quantiles(np.concatenate(all_new_distances)),
        "pooled_old_to_new_nearest_distance_m": _quantiles(np.concatenate(all_old_distances)),
        "pooled_signed_orientation_cosine": _quantiles(np.concatenate(all_orientation)),
        "pooled_absolute_orientation_cosine": _quantiles(np.abs(np.concatenate(all_orientation))),
        "pooled_signed_nearest_topography_cosine": _quantiles(np.concatenate(all_topography)),
        "pooled_absolute_nearest_topography_cosine": _quantiles(
            np.abs(np.concatenate(all_topography))
        ),
    }


def _unit_trace(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 0.0:
        raise ValueError("covariance trace must be positive")
    return np.asarray(matrix, dtype=np.float64) / trace


def _model_metrics(
    h_leadfield: np.ndarray,
    h_metadata: dict[str, np.ndarray],
    c_leadfield: np.ndarray,
    c_weights: np.ndarray,
    noise_fraction: float,
) -> tuple[dict[str, float], object, np.ndarray]:
    coordinate = h_metadata["longitudinal_coordinate"]
    phase = 2.0 * np.pi * coordinate
    operators = sensor_operators(
        h_leadfield,
        c_leadfield,
        h_metadata["area_weights_m2"],
        c_weights,
        None,
        hippocampal_wave_basis=np.vstack((np.cos(phase), np.sin(phase))),
    )
    h_trace = float(np.trace(operators.hippocampal_covariance))
    c_trace = float(np.trace(operators.cortical_covariance))
    coherent_trace = float(np.dot(
        operators.hippocampal_coherent_topography,
        operators.hippocampal_coherent_topography,
    ))
    wave_trace = float(np.trace(operators.hippocampal_wave_covariance))
    spectrum = standardized_recoverability_spectrum(
        operators, h_trace, c_trace, noise_fraction
    )
    return (
        {
            "hippocampal_covariance_trace": h_trace,
            "coherent_cancellation_ratio": coherent_trace / h_trace,
            "one_cycle_wave_cancellation_ratio": wave_trace / h_trace,
            "standardized_recoverability_index_bits": float(
                0.5 * np.sum(np.log2(1.0 + spectrum))
            ),
            "largest_recoverability_eigenvalue": float(spectrum[0]),
            "effective_recoverability_rank": float(
                np.sum(spectrum) ** 2 / np.sum(spectrum**2)
            ),
        },
        operators,
        spectrum,
    )


def _sensor_agreement(
    new_metrics: dict[str, float],
    old_metrics: dict[str, float],
    new_operators: object,
    old_operators: object,
    new_spectrum: np.ndarray,
    old_spectrum: np.ndarray,
) -> dict[str, object]:
    new_cov = _unit_trace(new_operators.hippocampal_covariance)
    old_cov = _unit_trace(old_operators.hippocampal_covariance)
    new_values, new_vectors = np.linalg.eigh(new_cov)
    old_values, old_vectors = np.linalg.eigh(old_cov)
    new_vectors = new_vectors[:, np.argsort(new_values)[::-1]]
    old_vectors = old_vectors[:, np.argsort(old_values)[::-1]]
    angles = {}
    for rank in (1, 2, 4, 8):
        values = subspace_angles(new_vectors[:, :rank], old_vectors[:, :rank])
        angles[str(rank)] = {
            "maximum_degrees": float(np.degrees(np.max(values))),
            "rms_degrees": float(np.sqrt(np.mean(np.degrees(values) ** 2))),
        }
    coherent_cosine = float(
        np.dot(
            new_operators.hippocampal_coherent_topography,
            old_operators.hippocampal_coherent_topography,
        )
        / (
            np.linalg.norm(new_operators.hippocampal_coherent_topography)
            * np.linalg.norm(old_operators.hippocampal_coherent_topography)
        )
    )
    scale = max(float(np.linalg.norm(old_spectrum)), np.finfo(float).tiny)
    return {
        "new": new_metrics,
        "old": old_metrics,
        "new_over_old": {
            key: float(new_metrics[key] / old_metrics[key])
            for key in (
                "hippocampal_covariance_trace",
                "coherent_cancellation_ratio",
                "one_cycle_wave_cancellation_ratio",
                "standardized_recoverability_index_bits",
            )
            if old_metrics[key] != 0.0
        },
        "unit_trace_covariance_frobenius_distance": float(
            np.linalg.norm(new_cov - old_cov, ord="fro")
        ),
        "coherent_topography_signed_cosine": coherent_cosine,
        "coherent_topography_absolute_cosine": abs(coherent_cosine),
        "recoverability_spectrum_relative_l2_error": float(
            np.linalg.norm(new_spectrum - old_spectrum) / scale
        ),
        "leading_covariance_subspace_principal_angles": angles,
    }


def build(
    subject: str,
    hippunfold_root: Path,
    bem_root: Path,
    old_forward_root: Path,
    output_directory: Path,
    config: dict[str, object],
) -> dict[str, object]:
    subject_bem = bem_root / subject / "bem"
    brain_surface_path = subject_bem / "brain.surf"
    bem_surfaces_path = subject_bem / f"{subject}-ico4-bem.fif"
    bem_solution_path = subject_bem / f"{subject}-ico4-bem-sol.fif"
    old_directory = old_forward_root / subject
    required = [
        brain_surface_path,
        bem_surfaces_path,
        bem_solution_path,
        old_directory / "hippocampal-source-metadata.npz",
        old_directory / "cortical-source-metadata.npz",
        old_directory / "hippocampal-fixed-leadfield.npy",
        old_directory / "cortical-fixed-leadfield.npy",
        old_directory / "registered-montage.tsv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing paired forward inputs: {missing}")
    output_directory.mkdir(parents=True, exist_ok=False)
    levels = [
        {"name": row["name"], "spatial_bin_mm": row["hippocampal_spatial_bin_mm"]}
        for row in config["source_levels"]
    ]
    scanner_to_bem, center_ras_mm = scanner_ras_to_bem_transform(brain_surface_path)
    hierarchy, geometry_report = hippunfold_source_hierarchy(
        hippunfold_root,
        subject,
        scanner_to_bem,
        levels,
        density=str(config["surface_density"]),
        normal_bin_width=float(config["normal_bin_width"]),
        audit_dentate=True,
    )
    source = hierarchy["reference"]
    surfaces = mne.read_bem_surfaces(bem_surfaces_path, verbose=False)
    by_id = {int(surface["id"]): surface for surface in surfaces}
    inner_skull = by_id[int(FIFF.FIFFV_BEM_SURF_ID_BRAIN)]
    outer_skin = by_id[int(FIFF.FIFFV_BEM_SURF_ID_HEAD)]
    containment = _source_containment(
        source["positions_m"], inner_skull, "HippUnfold hippocampal"
    )
    sensor_names, electrode_positions_m, montage_report = register_standard_montage(
        outer_skin, str(config["montage"])
    )
    old_names, old_electrode_positions_m = _registered_montage(
        old_directory / "registered-montage.tsv"
    )
    montage_error = float(np.max(np.abs(electrode_positions_m - old_electrode_positions_m)))
    if sensor_names != old_names or montage_error > 1e-11:
        raise ValueError("paired old/new montage is not identical")
    info = _make_info(sensor_names, electrode_positions_m)
    bem_solution = mne.read_bem_solution(bem_solution_path, verbose=False)
    forward = _compute_fixed_forward(
        info, source["positions_m"], source["directions"], bem_solution
    )
    leadfield = np.asarray(forward["sol"]["data"], dtype=np.float64)
    rank, singular = _referenced_rank(leadfield)
    if rank != len(sensor_names) - 1:
        raise ValueError("HippUnfold lead field is not full referenced sensor rank")

    forward_path = output_directory / "hippunfold-hippocampal-fwd.fif"
    matrix_path = output_directory / "hippunfold-hippocampal-fixed-leadfield.npy"
    metadata_path = output_directory / "hippunfold-hippocampal-source-metadata.npz"
    mne.write_forward_solution(forward_path, forward, overwrite=True, verbose=False)
    np.save(matrix_path, leadfield, allow_pickle=False)
    np.savez_compressed(
        metadata_path,
        positions_m=source["positions_m"],
        directions=source["directions"],
        area_weights_m2=source["area_weights_m2"],
        longitudinal_coordinate=source["longitudinal_coordinate"],
        proximal_distal_coordinate=source["proximal_distal_coordinate"],
        hemisphere_code=source["hemisphere_code"],
    )
    roundtrip = _validate_forward_roundtrip(
        forward_path, leadfield, source["positions_m"], source["directions"]
    )

    old_h = _npz(
        old_directory / "hippocampal-source-metadata.npz",
        ("positions_m", "directions", "area_weights_m2", "longitudinal_coordinate", "hemisphere_code"),
    )
    old_c = _npz(
        old_directory / "cortical-source-metadata.npz",
        ("area_weights_m2",),
    )
    old_h_leadfield = np.load(
        old_directory / "hippocampal-fixed-leadfield.npy", allow_pickle=False
    )
    cortical_leadfield = np.load(
        old_directory / "cortical-fixed-leadfield.npy", mmap_mode="r", allow_pickle=False
    )
    new_metadata = {key: np.asarray(value) for key, value in source.items()}
    source_agreement = _source_agreement(
        new_metadata, old_h, leadfield, old_h_leadfield
    )
    noise_fraction = float(config["standardized_noise_fraction"])
    new_metrics, new_operators, new_spectrum = _model_metrics(
        leadfield, new_metadata, np.asarray(cortical_leadfield), old_c["area_weights_m2"], noise_fraction
    )
    old_metrics, old_operators, old_spectrum = _model_metrics(
        old_h_leadfield, old_h, np.asarray(cortical_leadfield), old_c["area_weights_m2"], noise_fraction
    )
    sensor_agreement = _sensor_agreement(
        new_metrics,
        old_metrics,
        new_operators,
        old_operators,
        new_spectrum,
        old_spectrum,
    )
    spectra_path = output_directory / "paired-recoverability-spectra.npz"
    np.savez_compressed(spectra_path, hippunfold=new_spectrum, boundary_normal=old_spectrum)
    output_paths = [forward_path, matrix_path, metadata_path, spectra_path]
    report = {
        "schema_version": 1,
        "ok": True,
        "protocol": config["protocol"],
        "subject": subject,
        "scope": "paired geometry-only comparison; no physiological calibration",
        "python": platform.python_version(),
        "mne": mne.__version__,
        "nibabel": nib.__version__,
        "numpy": np.__version__,
        "coordinate_frame": "mne-head-identical-to-bem-surface-ras",
        "scanner_ras_to_bem_surface_ras": scanner_to_bem.tolist(),
        "watershed_center_ras_mm": center_ras_mm.tolist(),
        "source_model": {
            "primary": "HippUnfold label-hipp CA/subiculum midthickness",
            "adapter_revision": config["source_adapter_revision"],
            "orientation": geometry_report["orientation_definition"],
            "midthickness_segment_policy": config["midthickness_segment_policy"],
            "zero_measure_vertex_policy": config["zero_measure_vertex_policy"],
            "histological_laminar_ground_truth": False,
            "dentate": geometry_report["dentate"],
            "sources": int(len(source["positions_m"])),
            "area_m2": float(np.sum(source["area_weights_m2"])),
            "exact_nested": geometry_report["exact_nested"],
        },
        "geometry_report": geometry_report,
        "containment": containment,
        "montage": {
            **montage_report,
            "sensors": len(sensor_names),
            "legacy_name_order_identical": True,
            "legacy_maximum_position_error_m": montage_error,
        },
        "forward": {
            "shape": list(leadfield.shape),
            "referenced_rank": rank,
            "expected_sensor_rank": len(sensor_names) - 1,
            "largest_singular_value": float(singular[0]),
            "fixed_projection_roundtrip": roundtrip,
            "finite": bool(np.all(np.isfinite(leadfield))),
        },
        "agreement": {
            "source_space": source_agreement,
            "sensor_space": sensor_agreement,
        },
        "input_sha256": {str(path): sha256_file(path) for path in required},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_paths
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
    parser.add_argument("--hippunfold-root", type=Path, required=True)
    parser.add_argument("--bem-root", type=Path, required=True)
    parser.add_argument("--old-forward-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build(
        args.subject,
        args.hippunfold_root,
        args.bem_root,
        args.old_forward_root,
        args.output,
        config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
