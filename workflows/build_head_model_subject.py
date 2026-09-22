#!/usr/bin/env python3
"""Compute controlled forward-model uncertainty conditions for one HCP anatomy."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import sys

import mne
from mne.io.constants import FIFF
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from anatomical.convergence import compare_conditions, sensor_operators, summarize_condition  # noqa: E402
from anatomical.hcp_forward import (  # noqa: E402
    _closed_surface_centroid, _make_info, compute_combined_fixed_leadfields,
)
from workflows.forward_convergence_check import _load_montage, _perturb_electrodes, _set_conductivity  # noqa: E402
from simulation.cortical_restriction import farthest_anchors  # noqa: E402
from simulation.head_model_uncertainty import radial_displacement, radial_shift, rotated_directions  # noqa: E402

PROTOCOL = "uncertainty/head-model-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing: raise ValueError(f"{path} lacks {missing}")
        return {name: np.asarray(archive[name]) for name in required}


def shifted_bem_model(
    source_bem: Path, temporary_root: Path, subject: str, target_name: str, shift_mm: float,
    conductivity: tuple[float, float, float],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    destination = temporary_root / subject / "bem"; destination.mkdir(parents=True, exist_ok=False)
    displacement = None
    for name in ("inner_skull.surf", "outer_skull.surf", "outer_skin.surf"):
        vertices, triangles = mne.read_surface(source_bem / name, read_metadata=False, return_dict=False, verbose=False)
        if name == target_name:
            shifted = radial_shift(vertices, shift_mm); displacement = radial_displacement(vertices, shifted); vertices = shifted
        mne.write_surface(destination / name, vertices, triangles, overwrite=False, verbose=False)
    if displacement is None: raise ValueError(f"unknown perturbed surface: {target_name}")
    model = mne.make_bem_model(subject=subject, ico=3, conductivity=conductivity, subjects_dir=temporary_root, verbose=False)
    return model, displacement


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--subject", required=True); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hippocampal-root", type=Path, required=True); parser.add_argument("--cortical-root", type=Path, required=True)
    parser.add_argument("--bem-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8")); subject = args.subject
    if config.get("protocol") != PROTOCOL: raise ValueError("unexpected head-model protocol")
    h_dir = args.hippocampal_root / subject; c_dir = args.cortical_root / subject; bem_dir = args.bem_root / subject / "bem"
    paths = {
        "h_leadfield": h_dir / "hippunfold-hippocampal-fixed-leadfield.npy", "h_metadata": h_dir / "hippunfold-hippocampal-source-metadata.npz",
        "c_leadfield": c_dir / "cortical-fixed-leadfield.npy", "c_metadata": c_dir / "cortical-source-metadata.npz",
        "montage": c_dir / "registered-montage.tsv", "bem_surfaces": bem_dir / f"{subject}-ico4-bem.fif",
        "bem_solution": bem_dir / f"{subject}-ico4-bem-sol.fif", "inner_skull": bem_dir / "inner_skull.surf",
        "outer_skull": bem_dir / "outer_skull.surf", "outer_skin": bem_dir / "outer_skin.surf",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing: raise FileNotFoundError(f"missing head-model inputs: {missing}")
    args.output.mkdir(parents=True, exist_ok=False)
    hm = npz(paths["h_metadata"], ("positions_m", "directions", "area_weights_m2", "longitudinal_coordinate"))
    cm = npz(paths["c_metadata"], ("positions_m", "directions", "area_weights_m2", "hemisphere_code"))
    h_reference = np.load(paths["h_leadfield"], mmap_mode="r", allow_pickle=False); c_full = np.load(paths["c_leadfield"], mmap_mode="r", allow_pickle=False)
    names, electrodes = _load_montage(paths["montage"]); info = _make_info(names, electrodes)
    anchors = farthest_anchors(cm["positions_m"], cm["hemisphere_code"], cm["area_weights_m2"], int(config["cortical_anchors_per_hemisphere"]))
    width = float(config["cortical_smoothing_width_m"]); total_area = float(np.sum(cm["area_weights_m2"])); anchor_weights = []
    for anchor in anchors:
        same = cm["hemisphere_code"] == cm["hemisphere_code"][anchor]
        squared = np.sum((cm["positions_m"] - cm["positions_m"][anchor]) ** 2, axis=1)
        anchor_weights.append(float(np.sum(np.exp(-0.5 * squared / width**2) * cm["area_weights_m2"] * same)) / total_area)
    anchor_weights = np.asarray(anchor_weights); c_positions = cm["positions_m"][anchors]; c_directions = cm["directions"][anchors]
    c_reference = np.asarray(c_full[:, anchors]); nominal_surfaces = mne.read_bem_surfaces(paths["bem_surfaces"], verbose=False)
    nominal_solution = mne.read_bem_solution(paths["bem_solution"], verbose=False)
    outer_skin = next(surface for surface in nominal_surfaces if int(surface["id"]) == int(FIFF.FIFFV_BEM_SURF_ID_HEAD))
    reference_operators = sensor_operators(np.asarray(h_reference), c_reference, hm["area_weights_m2"], anchor_weights, hm["longitudinal_coordinate"])
    h_trace = float(np.trace(reference_operators.hippocampal_covariance)); c_trace = float(np.trace(reference_operators.cortical_covariance))
    noise = float(config["standardized_noise_fraction"]); operators = {}; spectra = {}; records = {}; archive = {}

    def register(name: str, family: str, h_lead: np.ndarray, c_lead: np.ndarray, reference_name: str, details: dict[str, object]):
        current = sensor_operators(h_lead, c_lead, hm["area_weights_m2"], anchor_weights, hm["longitudinal_coordinate"])
        summary, spectrum = summarize_condition(current, h_trace, c_trace, noise)
        if name == reference_name:
            comparison = {"hippocampal_covariance_shape_distance": 0.0, "cortical_covariance_shape_distance": 0.0,
                          "hippocampal_wave_shape_distance": 0.0, "hippocampal_coherent_cosine": 1.0,
                          "hippocampal_amplitude_ratio": 1.0, "cortical_amplitude_ratio": 1.0,
                          "hippocampal_wave_amplitude_ratio": 1.0, "recoverability_spectrum_relative_error": 0.0}
        else: comparison = compare_conditions(current, operators[reference_name], spectrum, spectra[reference_name])
        operators[name] = current; spectra[name] = spectrum
        records[name] = {"condition": name, "family": family, "reference_condition": reference_name, **details, **summary, **comparison}
        archive[f"{name}__hippocampal_covariance"] = current.hippocampal_covariance
        archive[f"{name}__cortical_covariance"] = current.cortical_covariance
        archive[f"{name}__hippocampal_wave_covariance"] = current.hippocampal_wave_covariance
        archive[f"{name}__hippocampal_coherent_topography"] = current.hippocampal_coherent_topography
        archive[f"{name}__recoverability_spectrum"] = spectrum
        print(json.dumps({"status": "complete", "condition": name}), flush=True)

    register("reference", "reference", np.asarray(h_reference), c_reference, "reference", {"bem_ico": 4})
    nominal_conductivity = tuple(float(value) for value in config["nominal_conductivity_s_per_m"])
    ico3_model = mne.make_bem_model(subject=subject, ico=3, conductivity=nominal_conductivity, subjects_dir=args.bem_root, verbose=False)
    ico3_solution = mne.make_bem_solution(ico3_model, verbose=False)
    h_ico3, c_ico3 = compute_combined_fixed_leadfields(info, hm["positions_m"], hm["directions"], c_positions, c_directions, ico3_solution)
    register("bem_ico3", "mesh_reference", h_ico3, c_ico3, "reference", {"bem_ico": 3})
    for variant in config["conductivity_variants"]:
        model = _set_conductivity(ico3_model, [float(value) for value in variant["values_s_per_m"]]); solution = mne.make_bem_solution(model, verbose=False)
        h, c = compute_combined_fixed_leadfields(info, hm["positions_m"], hm["directions"], c_positions, c_directions, solution)
        register(f"conductivity_{variant['name']}", "conductivity", h, c, "bem_ico3", {"values_s_per_m": variant["values_s_per_m"]})
    rng = np.random.default_rng(int(config["electrode_perturbation_seed"])); tangent = rng.normal(size=electrodes.shape)
    for displacement in config["electrode_displacement_mm"]:
        perturbed, report = _perturb_electrodes(electrodes, outer_skin, tangent, float(displacement)); perturbed_info = _make_info(names, perturbed)
        h, c = compute_combined_fixed_leadfields(perturbed_info, hm["positions_m"], hm["directions"], c_positions, c_directions, nominal_solution)
        suffix = str(float(displacement)).replace(".0", "").replace(".", "p")
        register(f"electrode_{suffix}mm", "electrode_registration", h, c, "reference", report)
    for variant in config["boundary_variants"]:
        with tempfile.TemporaryDirectory(prefix="head_model_uncertainty-bem-", dir=args.output) as temporary:
            model, displacement = shifted_bem_model(bem_dir, Path(temporary), subject, str(variant["surface"]), float(variant["shift_mm"]), nominal_conductivity)
            solution = mne.make_bem_solution(model, verbose=False)
            h, c = compute_combined_fixed_leadfields(info, hm["positions_m"], hm["directions"], c_positions, c_directions, solution)
        register(f"boundary_{variant['name']}", "segmentation_boundary", h, c, "bem_ico3", {"surface": variant["surface"], "shift_mm": float(variant["shift_mm"]), "actual_displacement": displacement, "bem_ico": 3})
    for degrees in config["orientation_degrees"]:
        directions = rotated_directions(hm["directions"], float(degrees), int(config["orientation_seed"]) + int(round(10 * float(degrees))))
        h, c = compute_combined_fixed_leadfields(info, hm["positions_m"], directions, c_positions, c_directions, nominal_solution)
        suffix = str(float(degrees)).replace(".0", "").replace(".", "p")
        register(f"orientation_{suffix}deg", "source_orientation", h, c, "reference", {"rotation_degrees": float(degrees)})
    center = _closed_surface_centroid(outer_skin); head_radius = float(np.median(np.linalg.norm(electrodes - center, axis=1)))
    maximum_source_ratio = float(np.max(np.linalg.norm(np.vstack((hm["positions_m"], c_positions)) - center, axis=1)) / head_radius)
    sphere_config = config["sphere_model"]; inner = max(float(sphere_config["minimum_inner_relative_radius"]), maximum_source_ratio + float(sphere_config["source_clearance_fraction"]))
    if inner > float(sphere_config["maximum_inner_relative_radius"]): raise ValueError("sources do not fit the prospective analytical sphere")
    middle = 0.5 * (inner + 1.0)
    sphere = mne.make_sphere_model(r0=tuple(center), head_radius=head_radius, relative_radii=(inner, middle, 1.0), sigmas=nominal_conductivity, verbose=False)
    h, c = compute_combined_fixed_leadfields(info, hm["positions_m"], hm["directions"], c_positions, c_directions, sphere)
    register("analytical_sphere", "alternative_conductor", h, c, "reference", {"head_radius_m": head_radius, "inner_relative_radius": inner, "maximum_source_radius_fraction": maximum_source_ratio})
    archive_path = args.output / "operators.npz"; np.savez_compressed(archive_path, **archive)
    rows = [records[name] for name in sorted(records)]; fields = sorted({key for row in rows for key in row})
    conditions_path = args.output / "conditions.csv"
    with conditions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    expected = 2 + len(config["conductivity_variants"]) + len(config["electrode_displacement_mm"]) + len(config["boundary_variants"]) + len(config["orientation_degrees"]) + 1
    report = {"schema_version": 1, "protocol": PROTOCOL, "ok": len(records) == expected and all(np.all(np.isfinite(value)) for value in archive.values()),
              "subject": subject, "condition_count": len(records), "families": sorted({row["family"] for row in records.values()}),
              "estimand": "geometry-only forward sensitivity; no physiological amplitude calibration", "reduced_cortical_anchor_count": len(anchors),
              "hcp_fem_run": False, "alternative_conductor_run": "three-shell analytical sphere",
              "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
              "outputs": {"operators_sha256": sha256(archive_path), "conditions_sha256": sha256(conditions_path)},
              "shared_storage_touched": False, "physiological_inference_authorized": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
