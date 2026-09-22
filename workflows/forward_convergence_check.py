#!/usr/bin/env python3
"""Container entry point for the HCP forward-model convergence smoke study."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
from pathlib import Path
import platform
import sys

import mne
from mne.io.constants import FIFF
from mne.surface import _points_outside_surface
import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from anatomical.bundle import _deterministic_savez, sha256_file
from anatomical.convergence import (
    SensorOperators,
    archive_arrays,
    compare_conditions,
    sensor_operators,
    summarize_condition,
)
from anatomical.hcp_forward import (
    _closed_surface_centroid,
    _make_info,
    _ray_surface_intersections,
    compute_combined_fixed_leadfields,
    cortical_surface_geometry,
    hippocampal_boundary_geometry,
    scanner_ras_to_bem_transform,
)


def _load_metadata(path: Path, hippocampal: bool) -> dict[str, np.ndarray]:
    names = ["positions_m", "directions", "area_weights_m2"]
    if hippocampal:
        names.append("longitudinal_coordinate")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in names}


def _load_montage(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    positions: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["name", "x_m", "y_m", "z_m"]:
            raise ValueError("registered montage TSV has an unexpected header")
        for row in reader:
            names.append(str(row["name"]))
            positions.append([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
    result = np.asarray(positions, dtype=np.float64)
    if len(names) < 2 or len(set(names)) != len(names) or result.shape != (len(names), 3):
        raise ValueError("registered montage is invalid")
    return names, result


def _inside_count(positions_m: np.ndarray, model: list[dict[str, object]]) -> int:
    inner = next(
        surface
        for surface in model
        if int(surface["id"]) == int(FIFF.FIFFV_BEM_SURF_ID_BRAIN)
    )
    return int(np.count_nonzero(_points_outside_surface(positions_m, inner, n_jobs=1)))


def _bem_model_report(model: list[dict[str, object]], ico: int) -> dict[str, object]:
    by_id = {int(surface["id"]): surface for surface in model}
    expected = {
        int(FIFF.FIFFV_BEM_SURF_ID_BRAIN),
        int(FIFF.FIFFV_BEM_SURF_ID_SKULL),
        int(FIFF.FIFFV_BEM_SURF_ID_HEAD),
    }
    if set(by_id) != expected:
        raise ValueError("BEM refinement model has incorrect surface identities")
    surfaces = []
    for surface_id in sorted(by_id):
        surface = by_id[surface_id]
        rr = np.asarray(surface["rr"])
        tris = np.asarray(surface["tris"])
        if not np.all(np.isfinite(rr)) or tris.min() < 0 or tris.max() >= len(rr):
            raise ValueError("BEM refinement surface is invalid")
        surfaces.append(
            {
                "fiff_id": surface_id,
                "vertices": int(len(rr)),
                "triangles": int(len(tris)),
                "conductivity_s_per_m": float(surface["sigma"]),
            }
        )
    return {"ico": ico, "surfaces": surfaces}


def _set_conductivity(
    model: list[dict[str, object]], values: list[float]
) -> list[dict[str, object]]:
    result = copy.deepcopy(model)
    by_id = {int(surface["id"]): surface for surface in result}
    assignment = {
        int(FIFF.FIFFV_BEM_SURF_ID_BRAIN): float(values[0]),
        int(FIFF.FIFFV_BEM_SURF_ID_SKULL): float(values[1]),
        int(FIFF.FIFFV_BEM_SURF_ID_HEAD): float(values[2]),
    }
    if any(value <= 0.0 or not np.isfinite(value) for value in assignment.values()):
        raise ValueError("conductivity values must be finite and positive")
    for surface_id, sigma in assignment.items():
        by_id[surface_id]["sigma"] = sigma
    return result


def _perturb_electrodes(
    nominal_m: np.ndarray,
    outer_skin: dict[str, object],
    tangent_directions: np.ndarray,
    requested_displacement_mm: float,
) -> tuple[np.ndarray, dict[str, float]]:
    center = _closed_surface_centroid(outer_skin)
    radial = nominal_m - center
    radii = np.linalg.norm(radial, axis=1)
    radial /= radii[:, None]
    tangent = tangent_directions - np.sum(tangent_directions * radial, axis=1)[:, None] * radial
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_norm <= 0.0):
        raise ValueError("electrode perturbation produced a zero tangent")
    tangent /= tangent_norm[:, None]
    angle = (requested_displacement_mm / 1000.0) / radii
    directions = np.cos(angle)[:, None] * radial + np.sin(angle)[:, None] * tangent
    perturbed = _ray_surface_intersections(center, directions, outer_skin)
    displacement = np.linalg.norm(perturbed - nominal_m, axis=1) * 1000.0
    return perturbed, {
        "requested_displacement_mm": requested_displacement_mm,
        "actual_rms_displacement_mm": float(np.sqrt(np.mean(displacement**2))),
        "actual_median_displacement_mm": float(np.median(displacement)),
        "actual_maximum_displacement_mm": float(np.max(displacement)),
    }


def _maximum_covariance_distance(comparison: dict[str, float]) -> float:
    return max(
        comparison["hippocampal_covariance_shape_distance"],
        comparison["cortical_covariance_shape_distance"],
        comparison["hippocampal_wave_shape_distance"],
    )


def _amplitudes_in_range(
    comparison: dict[str, float], minimum: float, maximum: float
) -> bool:
    return all(
        minimum <= comparison[key] <= maximum
        for key in (
            "hippocampal_amplitude_ratio",
            "cortical_amplitude_ratio",
            "hippocampal_wave_amplitude_ratio",
        )
    )


def _monotone_toward_reference(
    earlier: dict[str, float], later: dict[str, float]
) -> bool:
    distance_keys = (
        "hippocampal_covariance_shape_distance",
        "cortical_covariance_shape_distance",
        "hippocampal_wave_shape_distance",
        "recoverability_spectrum_relative_error",
    )
    if not all(later[key] <= earlier[key] + 1e-12 for key in distance_keys):
        return False
    if 1.0 - later["hippocampal_coherent_cosine"] > 1.0 - earlier["hippocampal_coherent_cosine"] + 1e-12:
        return False
    return True


def _assess(
    comparisons: dict[str, dict[str, float]], config: dict[str, object]
) -> dict[str, object]:
    criteria = dict(config["criteria"])
    source_criteria = dict(criteria["source"])
    coarse = comparisons["source_coarse"]
    medium = comparisons["source_medium"]
    source_checks = {
        "monotone_refinement": _monotone_toward_reference(coarse, medium),
        "medium_covariance_shape": _maximum_covariance_distance(medium)
        <= float(source_criteria["medium_covariance_shape_distance_max"]),
        "medium_coherent_cosine": medium["hippocampal_coherent_cosine"]
        >= float(source_criteria["medium_coherent_cosine_min"]),
        "medium_recoverability_spectrum": medium["recoverability_spectrum_relative_error"]
        <= float(source_criteria["medium_recoverability_spectrum_relative_error_max"]),
        "medium_amplitudes": _amplitudes_in_range(
            medium,
            float(source_criteria["medium_amplitude_ratio_min"]),
            float(source_criteria["medium_amplitude_ratio_max"]),
        ),
    }

    bem_criteria = dict(criteria["bem"])
    ico2 = comparisons["bem_ico2"]
    ico3 = comparisons["bem_ico3"]
    bem_checks = {
        "monotone_refinement": _monotone_toward_reference(ico2, ico3),
        "ico3_covariance_shape": _maximum_covariance_distance(ico3)
        <= float(bem_criteria["ico3_covariance_shape_distance_max"]),
        "ico3_coherent_cosine": ico3["hippocampal_coherent_cosine"]
        >= float(bem_criteria["ico3_coherent_cosine_min"]),
        "ico3_recoverability_spectrum": ico3["recoverability_spectrum_relative_error"]
        <= float(bem_criteria["ico3_recoverability_spectrum_relative_error_max"]),
        "ico3_amplitudes": _amplitudes_in_range(
            ico3,
            float(bem_criteria["ico3_amplitude_ratio_min"]),
            float(bem_criteria["ico3_amplitude_ratio_max"]),
        ),
    }

    conductivity_criteria = dict(criteria["conductivity"])
    conductivity_rows = [
        value
        for key, value in comparisons.items()
        if key.startswith("conductivity_")
    ]
    conductivity_checks = {
        "all_covariance_shapes": all(
            _maximum_covariance_distance(row)
            <= float(conductivity_criteria["covariance_shape_distance_max"])
            for row in conductivity_rows
        ),
        "all_coherent_cosines": all(
            row["hippocampal_coherent_cosine"]
            >= float(conductivity_criteria["coherent_cosine_min"])
            for row in conductivity_rows
        ),
        "all_amplitudes_bounded": all(
            _amplitudes_in_range(
                row,
                float(conductivity_criteria["amplitude_ratio_min"]),
                float(conductivity_criteria["amplitude_ratio_max"]),
            )
            for row in conductivity_rows
        ),
    }

    electrode_criteria = dict(criteria["electrode"])
    e10 = comparisons["electrode_10mm"]
    e5 = comparisons["electrode_5mm"]
    e25 = comparisons["electrode_2p5mm"]
    electrode_checks = {
        "monotone_refinement": _monotone_toward_reference(e10, e5)
        and _monotone_toward_reference(e5, e25),
        "small_covariance_shape": _maximum_covariance_distance(e25)
        <= float(electrode_criteria["small_covariance_shape_distance_max"]),
        "small_coherent_cosine": e25["hippocampal_coherent_cosine"]
        >= float(electrode_criteria["small_coherent_cosine_min"]),
        "small_recoverability_spectrum": e25["recoverability_spectrum_relative_error"]
        <= float(electrode_criteria["small_recoverability_spectrum_relative_error_max"]),
    }
    families = {
        "source_resolution": {"checks": source_checks, "passed": all(source_checks.values())},
        "bem_mesh": {"checks": bem_checks, "passed": all(bem_checks.values())},
        "conductivity_sensitivity": {
            "checks": conductivity_checks,
            "passed": all(conductivity_checks.values()),
            "interpretation": "bounded sensitivity, not a convergence claim",
        },
        "electrode_registration": {
            "checks": electrode_checks,
            "passed": all(electrode_checks.values()),
        },
    }
    return {
        "families": families,
        "all_families_passed": all(value["passed"] for value in families.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--bem-root", type=Path, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    subject = str(config["subject"])
    args.output.mkdir(parents=True, exist_ok=False)
    forward = args.forward
    subject_input = args.input_root / subject
    subject_bem = args.bem_root / subject
    segmentation = subject_input / "T1w" / "aparc+aseg.nii.gz"
    brain_surface = subject_bem / "bem" / "brain.surf"
    bem_surface_path = subject_bem / "bem" / f"{subject}-ico4-bem.fif"
    bem_solution_path = subject_bem / "bem" / f"{subject}-ico4-bem-sol.fif"
    source_paths = {
        "segmentation": segmentation,
        "brain_surface": brain_surface,
        "bem_surfaces": bem_surface_path,
        "bem_solution": bem_solution_path,
        "left_white": subject_input / "T1w" / "Native" / f"{subject}.L.white.native.surf.gii",
        "right_white": subject_input / "T1w" / "Native" / f"{subject}.R.white.native.surf.gii",
        "forward_report": forward / "report.json",
        "bundle_arrays": args.bundle / "bundle" / "arrays.npz",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing convergence inputs: {missing}")
    if sha256_file(source_paths["forward_report"]) != config["source_forward_report_sha256"]:
        raise ValueError("source forward report digest mismatch")
    if sha256_file(source_paths["bundle_arrays"]) != config["bundle_arrays_sha256"]:
        raise ValueError("bundle arrays digest mismatch")

    expected_levels = ["coarse", "medium", "reference"]
    if [row["name"] for row in config["source_levels"]] != expected_levels:
        raise ValueError("source levels are not the frozen coarse/medium/reference sequence")
    if [int(value) for value in config["bem_ico_levels"]] != [2, 3, 4]:
        raise ValueError("BEM levels are not the frozen ico-2/3/4 sequence")

    h_reference_geometry = _load_metadata(
        forward / "hippocampal-source-metadata.npz", True
    )
    c_reference_geometry = _load_metadata(
        forward / "cortical-source-metadata.npz", False
    )
    h_reference_leadfield = np.load(
        forward / "hippocampal-fixed-leadfield.npy", allow_pickle=False, mmap_mode="r"
    )
    c_reference_leadfield = np.load(
        forward / "cortical-fixed-leadfield.npy", allow_pickle=False, mmap_mode="r"
    )
    sensor_names, nominal_electrodes = _load_montage(forward / "registered-montage.tsv")
    scanner_to_bem, center_ras = scanner_ras_to_bem_transform(brain_surface)
    nominal_surfaces = mne.read_bem_surfaces(bem_surface_path, verbose=False)
    nominal_solution = mne.read_bem_solution(bem_solution_path, verbose=False)
    nominal_info = _make_info(sensor_names, nominal_electrodes)
    outer_skin = next(
        surface
        for surface in nominal_surfaces
        if int(surface["id"]) == int(FIFF.FIFFV_BEM_SURF_ID_HEAD)
    )

    operators: dict[str, SensorOperators] = {}
    spectra: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, object]] = {}
    archive: dict[str, np.ndarray] = {}

    reference = sensor_operators(
        h_reference_leadfield,
        c_reference_leadfield,
        h_reference_geometry["area_weights_m2"],
        c_reference_geometry["area_weights_m2"],
        h_reference_geometry["longitudinal_coordinate"],
    )
    h_reference_trace = float(np.trace(reference.hippocampal_covariance))
    c_reference_trace = float(np.trace(reference.cortical_covariance))
    noise_fraction = float(config["standardized_noise_fraction"])

    def register_condition(
        name: str,
        family: str,
        current: SensorOperators,
        reference_name: str,
        details: dict[str, object],
    ) -> None:
        reference_operator = reference if reference_name == "reference" else operators[reference_name]
        summary, spectrum = summarize_condition(
            current, h_reference_trace, c_reference_trace, noise_fraction
        )
        if reference_name == "reference":
            reference_spectrum = spectra.get("reference")
        else:
            reference_spectrum = spectra[reference_name]
        comparison = (
            {key: 0.0 for key in (
                "hippocampal_covariance_shape_distance",
                "cortical_covariance_shape_distance",
                "hippocampal_wave_shape_distance",
                "recoverability_spectrum_relative_error",
            )}
            | {
                "hippocampal_coherent_cosine": 1.0,
                "hippocampal_amplitude_ratio": 1.0,
                "cortical_amplitude_ratio": 1.0,
                "hippocampal_wave_amplitude_ratio": 1.0,
            }
            if name == reference_name
            else compare_conditions(current, reference_operator, spectrum, reference_spectrum)
        )
        operators[name] = current
        spectra[name] = spectrum
        records[name] = {
            "condition": name,
            "family": family,
            "reference_condition": reference_name,
            **details,
            **summary,
            **comparison,
        }
        for array_name, value in archive_arrays(current, spectrum).items():
            archive[f"{name}__{array_name}"] = value
        print(f"completed condition {name}", flush=True)

    register_condition(
        "reference",
        "reference",
        reference,
        "reference",
        {
            "hippocampal_sources": int(len(h_reference_geometry["positions_m"])),
            "cortical_sources": int(len(c_reference_geometry["positions_m"])),
        },
    )

    for level in config["source_levels"]:
        if level["name"] == "reference":
            continue
        h_geometry = hippocampal_boundary_geometry(
            segmentation,
            scanner_to_bem,
            spatial_bin_mm=float(level["hippocampal_spatial_bin_mm"]),
            normal_bin_width=float(config["normal_bin_width"]),
            smoothing_sigma_voxels=float(config["hippocampal_smoothing_sigma_voxels"]),
        )
        c_geometry = cortical_surface_geometry(
            subject,
            args.input_root,
            scanner_to_bem,
            surface_name="white",
            spatial_bin_mm=float(level["cortical_spatial_bin_mm"]),
            normal_bin_width=float(config["normal_bin_width"]),
        )
        h_area_error = abs(
            float(h_geometry["area_weights_m2"].sum())
            / float(h_reference_geometry["area_weights_m2"].sum())
            - 1.0
        )
        c_area_error = abs(
            float(c_geometry["area_weights_m2"].sum())
            / float(c_reference_geometry["area_weights_m2"].sum())
            - 1.0
        )
        if max(h_area_error, c_area_error) > 1e-12:
            raise ValueError("source aggregation did not conserve surface area")
        h_leadfield, c_leadfield = compute_combined_fixed_leadfields(
            nominal_info,
            h_geometry["positions_m"],
            h_geometry["directions"],
            c_geometry["positions_m"],
            c_geometry["directions"],
            nominal_solution,
        )
        current = sensor_operators(
            h_leadfield,
            c_leadfield,
            h_geometry["area_weights_m2"],
            c_geometry["area_weights_m2"],
            h_geometry["longitudinal_coordinate"],
        )
        register_condition(
            f"source_{level['name']}",
            "source_resolution",
            current,
            "reference",
            {
                "hippocampal_spatial_bin_mm": float(level["hippocampal_spatial_bin_mm"]),
                "cortical_spatial_bin_mm": float(level["cortical_spatial_bin_mm"]),
                "hippocampal_sources": int(len(h_geometry["positions_m"])),
                "cortical_sources": int(len(c_geometry["positions_m"])),
                "hippocampal_area_relative_error": h_area_error,
                "cortical_area_relative_error": c_area_error,
            },
        )
        del h_leadfield, c_leadfield, h_geometry, c_geometry
        gc.collect()

    nominal_conductivity = tuple(float(value) for value in config["nominal_conductivity_s_per_m"])
    ico3_model = None
    ico3_solution = None
    bem_reports = [_bem_model_report(nominal_surfaces, 4)]
    for ico in config["bem_ico_levels"]:
        ico = int(ico)
        if ico == 4:
            continue
        model = mne.make_bem_model(
            subject=subject,
            ico=ico,
            conductivity=nominal_conductivity,
            subjects_dir=args.bem_root,
            verbose=False,
        )
        bem_reports.append(_bem_model_report(model, ico))
        h_outside = _inside_count(h_reference_geometry["positions_m"], model)
        c_outside = _inside_count(c_reference_geometry["positions_m"], model)
        if h_outside or c_outside:
            raise ValueError(
                f"ico-{ico} excludes {h_outside} hippocampal and {c_outside} cortical sources"
            )
        solution = mne.make_bem_solution(model, verbose=False)
        h_leadfield, c_leadfield = compute_combined_fixed_leadfields(
            nominal_info,
            h_reference_geometry["positions_m"],
            h_reference_geometry["directions"],
            c_reference_geometry["positions_m"],
            c_reference_geometry["directions"],
            solution,
        )
        current = sensor_operators(
            h_leadfield,
            c_leadfield,
            h_reference_geometry["area_weights_m2"],
            c_reference_geometry["area_weights_m2"],
            h_reference_geometry["longitudinal_coordinate"],
        )
        register_condition(
            f"bem_ico{ico}",
            "bem_mesh",
            current,
            "reference",
            {"bem_ico": ico, "hippocampal_sources_outside": 0, "cortical_sources_outside": 0},
        )
        if ico == int(config["conductivity_mesh_ico"]):
            ico3_model, ico3_solution = model, solution
        else:
            del model, solution
        del h_leadfield, c_leadfield
        gc.collect()
    if ico3_model is None or ico3_solution is None:
        raise ValueError("conductivity mesh was not constructed")

    for variant in config["conductivity_variants"]:
        values = [float(value) for value in variant["values_s_per_m"]]
        model = _set_conductivity(ico3_model, values)
        solution = mne.make_bem_solution(model, verbose=False)
        h_leadfield, c_leadfield = compute_combined_fixed_leadfields(
            nominal_info,
            h_reference_geometry["positions_m"],
            h_reference_geometry["directions"],
            c_reference_geometry["positions_m"],
            c_reference_geometry["directions"],
            solution,
        )
        current = sensor_operators(
            h_leadfield,
            c_leadfield,
            h_reference_geometry["area_weights_m2"],
            c_reference_geometry["area_weights_m2"],
            h_reference_geometry["longitudinal_coordinate"],
        )
        register_condition(
            f"conductivity_{variant['name']}",
            "conductivity_sensitivity",
            current,
            "bem_ico3",
            {"bem_ico": int(config["conductivity_mesh_ico"]), "conductivity_s_per_m": values},
        )
        del model, solution, h_leadfield, c_leadfield
        gc.collect()

    rng = np.random.default_rng(int(config["electrode_perturbation_seed"]))
    tangent_directions = rng.normal(size=nominal_electrodes.shape)
    for displacement in config["electrode_displacement_mm"]:
        displacement = float(displacement)
        perturbed, perturbation_report = _perturb_electrodes(
            nominal_electrodes, outer_skin, tangent_directions, displacement
        )
        info = _make_info(sensor_names, perturbed)
        h_leadfield, c_leadfield = compute_combined_fixed_leadfields(
            info,
            h_reference_geometry["positions_m"],
            h_reference_geometry["directions"],
            c_reference_geometry["positions_m"],
            c_reference_geometry["directions"],
            nominal_solution,
        )
        current = sensor_operators(
            h_leadfield,
            c_leadfield,
            h_reference_geometry["area_weights_m2"],
            c_reference_geometry["area_weights_m2"],
            h_reference_geometry["longitudinal_coordinate"],
        )
        suffix = str(displacement).replace(".0", "").replace(".", "p")
        register_condition(
            f"electrode_{suffix}mm",
            "electrode_registration",
            current,
            "reference",
            perturbation_report,
        )
        del perturbed, info, h_leadfield, c_leadfield
        gc.collect()

    comparisons = {
        name: {
            key: float(value)
            for key, value in record.items()
            if key
            in {
                "hippocampal_covariance_shape_distance",
                "cortical_covariance_shape_distance",
                "hippocampal_wave_shape_distance",
                "hippocampal_coherent_cosine",
                "hippocampal_amplitude_ratio",
                "cortical_amplitude_ratio",
                "hippocampal_wave_amplitude_ratio",
                "recoverability_spectrum_relative_error",
            }
        }
        for name, record in records.items()
        if name != "reference"
    }
    assessment = _assess(comparisons, config)
    archive_path = args.output / "sensor_operators.npz"
    _deterministic_savez(archive_path, archive)
    config_snapshot = args.output / "config.snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    conditions_path = args.output / "conditions.csv"
    rows = [records[name] for name in sorted(records)]
    fieldnames = sorted({key for row in rows for key in row})
    with conditions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    report = {
        "schema_version": 1,
        "ok": True,
        "protocol": config["protocol"],
        "subject": subject,
        "python": platform.python_version(),
        "mne": mne.__version__,
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "estimand": "geometry-only area-normalized sensor operators; not physiological detectability",
        "reference_condition": "source bins 2/8 mm, BEM ico-4, conductivities 0.3/0.006/0.3 S/m, nominal synthetic standard_1005",
        "scanner_ras_to_bem_surface_ras": scanner_to_bem.tolist(),
        "watershed_center_ras_mm": center_ras.tolist(),
        "standardized_noise_fraction": noise_fraction,
        "conditions": records,
        "bem_models": sorted(bem_reports, key=lambda value: value["ico"]),
        "assessment": assessment,
        "population_forward_baseline_authorized": bool(assessment["all_families_passed"]),
        "physiological_inference_authorized": False,
        "histological_laminar_claim_authorized": False,
        "conductivity_ico4_interaction_tested": False,
        "input_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "outputs": {
            "sensor_operators_sha256": sha256_file(archive_path),
            "sensor_operators_bytes": archive_path.stat().st_size,
            "conditions_sha256": sha256_file(conditions_path),
            "config_snapshot_sha256": sha256_file(config_snapshot),
        },
        "shared_storage_touched": False,
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
