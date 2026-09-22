#!/usr/bin/env python3
"""Build one HCP subject's montage physical-montage value table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from anatomical.hippunfold_ensembles import (  # noqa: E402
    build_source_supports,
    oriented_intrinsic_coordinates,
    support_identifier,
)
from workflows.build_cortical_restriction_subject import phase_factor  # noqa: E402
from simulation.cortical_restriction import farthest_anchors, smooth_topographies  # noqa: E402
from simulation.montage_information import load_montage, montage_information, physical_montages  # noqa: E402


PROTOCOL = "montage/physical-value-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"{path} lacks {missing}")
        return {name: np.asarray(archive[name]) for name in required}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build(subject: str, config: dict[str, object], h_root: Path, c_root: Path, output: Path) -> dict[str, object]:
    h_dir = h_root / subject; c_dir = c_root / subject
    paths = {
        "h_leadfield": h_dir / "hippunfold-hippocampal-fixed-leadfield.npy",
        "h_metadata": h_dir / "hippunfold-hippocampal-source-metadata.npz",
        "h_report": h_dir / "report.json",
        "c_leadfield": c_dir / "cortical-fixed-leadfield.npy",
        "c_metadata": c_dir / "cortical-source-metadata.npz",
        "c_report": c_dir / "report.json",
        "montage": c_dir / "registered-montage.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing montage inputs: {missing}")
    if any(json.loads(paths[key].read_text(encoding="utf-8")).get("ok") is not True for key in ("h_report", "c_report")):
        raise ValueError("an upstream forward report failed")
    h_meta = load_npz(paths["h_metadata"], ("positions_m", "area_weights_m2", "longitudinal_coordinate", "proximal_distal_coordinate", "hemisphere_code"))
    c_meta = load_npz(paths["c_metadata"], ("positions_m", "area_weights_m2", "hemisphere_code"))
    h_leadfield = np.load(paths["h_leadfield"], allow_pickle=False, mmap_mode="r")
    c_leadfield = np.load(paths["c_leadfield"], allow_pickle=False, mmap_mode="r")
    names, positions = load_montage(paths["montage"])
    if h_leadfield.shape[0] != len(names) or c_leadfield.shape[0] != len(names):
        raise ValueError("lead-field and montage dimensions disagree")
    montage_config = config["montages"]
    montages, montage_report = physical_montages(
        names, positions,
        list(montage_config["conventional32_channels"]),
        list(montage_config["conventional64_channels"]),
        list(montage_config["inferior_addition_channels"]),
        tuple(int(value) for value in montage_config["high_density_sizes"]),
    )

    source_config = config["source_model"]
    anchors = farthest_anchors(
        c_meta["positions_m"], c_meta["hemisphere_code"], c_meta["area_weights_m2"],
        int(source_config["cortical_anchors_per_hemisphere"]),
    )
    cortical_factors = smooth_topographies(
        np.asarray(c_leadfield), c_meta["positions_m"], c_meta["hemisphere_code"],
        c_meta["area_weights_m2"], anchors, float(source_config["cortical_smoothing_width_m"]),
    )
    total_c_area = float(np.sum(c_meta["area_weights_m2"])); width = float(source_config["cortical_smoothing_width_m"])
    fractions = []
    for anchor in anchors:
        same = c_meta["hemisphere_code"] == c_meta["hemisphere_code"][anchor]
        squared = np.sum((c_meta["positions_m"] - c_meta["positions_m"][anchor]) ** 2, axis=1)
        fractions.append(float(np.sum(np.exp(-0.5 * squared / width**2) * c_meta["area_weights_m2"] * same)) / total_c_area)
    cortical_factors *= np.asarray(fractions)[None, :]
    cortical_covariance = cortical_factors @ cortical_factors.T / cortical_factors.shape[1]

    support_settings = source_config["support_construction"]
    anterior, pd, orientation_report = oriented_intrinsic_coordinates(
        h_meta["positions_m"], h_meta["longitudinal_coordinate"], h_meta["proximal_distal_coordinate"],
        h_meta["hemisphere_code"], h_meta["area_weights_m2"],
        endpoint_decile=float(support_settings["endpoint_decile"]),
        minimum_endpoint_separation_m=float(support_settings["minimum_endpoint_separation_m"]),
    )
    supports, support_report = build_source_supports(
        anterior, pd, h_meta["hemisphere_code"], h_meta["area_weights_m2"],
        hemisphere_families=("left", "right", "bilateral"),
        focal_locations=("anterior", "middle", "posterior"), focal_extents=(0.25,),
        include_whole_extent=True,
    )
    h_covariances: dict[str, np.ndarray] = {}
    for regime in source_config["ensembles"]:
        identifier = support_identifier(str(regime["hemisphere"]), str(regime["location"]), float(regime["extent"]))
        support = supports[identifier]
        factor = phase_factor(
            np.asarray(h_leadfield), support.indices, support.conditional_weights,
            support.full_sheet_area_fraction, anterior, float(regime["wave_cycles"]),
        )
        h_covariances[str(regime["name"])] = factor @ factor.T / factor.shape[1]
    reference_name = str(source_config["reference_ensemble"])
    sensor_dimension = len(names) - 1
    c_variance = float(np.trace(cortical_covariance)) / sensor_dimension
    h_variance = float(np.trace(h_covariances[reference_name])) / sensor_dimension
    if min(c_variance, h_variance) <= 0.0:
        raise ValueError("reference source variance is zero")
    signal_scale = float(config["standardization"]["reference_signal_to_cortical_variance"]) * c_variance / h_variance

    rows: list[dict[str, object]] = []
    for ensemble, h_covariance in sorted(h_covariances.items()):
        for cortical_ratio in config["standardization"]["cortical_variance_ratios"]:
            for noise_fraction in config["standardization"]["sensor_noise_variance_fractions"]:
                for montage, indices in montages.items():
                    bits = montage_information(
                        h_covariance, cortical_covariance, indices, signal_scale,
                        float(cortical_ratio), float(noise_fraction) * c_variance,
                    )
                    rows.append({
                        "subject": subject, "ensemble": ensemble, "montage": montage,
                        "sensors": len(indices), "sensor_dimension": len(indices) - 1,
                        "cortical_variance_ratio": float(cortical_ratio),
                        "noise_variance_fraction": float(noise_fraction),
                        "information_bits": bits,
                    })
    lookup = {(row["ensemble"], row["montage"], row["cortical_variance_ratio"], row["noise_variance_fraction"]): row for row in rows}
    increments: list[dict[str, object]] = []
    comparisons = list(zip(montage_report["nested_ladder"], montage_report["nested_ladder"][1:])) + [("conventional64", "conventional64_plus_inferior")]
    for ensemble in sorted(h_covariances):
        for cortical_ratio in config["standardization"]["cortical_variance_ratios"]:
            for noise_fraction in config["standardization"]["sensor_noise_variance_fractions"]:
                for existing, augmented in comparisons:
                    left = lookup[(ensemble, existing, float(cortical_ratio), float(noise_fraction))]
                    right = lookup[(ensemble, augmented, float(cortical_ratio), float(noise_fraction))]
                    increments.append({
                        "subject": subject, "ensemble": ensemble, "existing_montage": existing,
                        "augmented_montage": augmented, "existing_sensors": left["sensors"],
                        "augmented_sensors": right["sensors"], "cortical_variance_ratio": float(cortical_ratio),
                        "noise_variance_fraction": float(noise_fraction),
                        "conditional_information_bits": float(right["information_bits"]) - float(left["information_bits"]),
                    })
    minimum_increment = min(float(row["conditional_information_bits"]) for row in increments if row["augmented_montage"] != "conventional64_plus_inferior")
    output.mkdir(parents=True, exist_ok=False)
    metrics_path = output / "metrics.csv"; increments_path = output / "increments.csv"
    write_csv(metrics_path, rows); write_csv(increments_path, increments)
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "ok": bool(
            support_report["all_exactly_nested"] and montage_report["exactly_nested"] and minimum_increment >= -1e-9
        ),
        "subject": subject, "estimand": "geometry-only Gaussian information under a declared relative nuisance grid; not physiological bits",
        "montages": montage_report, "orientation": orientation_report,
        "source_supports_exact": bool(support_report["all_exactly_nested"]),
        "ensemble_count": len(h_covariances), "condition_count": len(rows), "increment_count": len(increments),
        "minimum_nested_conditional_information_bits": minimum_increment,
        "standardization": {"cortical_full_sensor_variance": c_variance, "reference_hippocampal_full_sensor_variance": h_variance, "signal_scale": signal_scale},
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "outputs": {"metrics_sha256": sha256(metrics_path), "increments_sha256": sha256(increments_path)},
        "face_neck_evaluated": False, "physiological_inference_authorized": False, "shared_storage_touched": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hippocampal-root", type=Path, required=True); parser.add_argument("--cortical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise SystemExit("unexpected montage protocol")
    report = build(args.subject, config, args.hippocampal_root, args.cortical_root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
