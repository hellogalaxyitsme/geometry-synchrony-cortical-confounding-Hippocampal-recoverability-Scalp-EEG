#!/usr/bin/env python3
"""Build one subject of the cortical-restriction cortical-restriction ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from anatomical.hippunfold_ensembles import (  # noqa: E402
    build_source_supports,
    deterministic_phase,
    oriented_intrinsic_coordinates,
    support_identifier,
)
from simulation.cortical_restriction import (  # noqa: E402
    area_weighted_covariance,
    basis_diagnostics,
    covariance_basis,
    deterministic_dictionary_split,
    dynamic_hippocampal_factors,
    false_attribution,
    farthest_anchors,
    neighborhood_indices,
    normalized_columns,
    orthonormal_basis,
    referenced_operator,
    residual_sensitivity,
    smooth_topographies,
    sparse_false_from_bases,
    sparse_probe_bases,
    sparse_sensitivity,
    stable_seed,
    summarize_false_attribution,
    tensor_false_attribution,
    tensor_sensitivity,
)


PROTOCOL = "restriction/ladder-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(names).difference(archive.files)
        if missing:
            raise ValueError(f"missing arrays in {path}: {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in names}
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"non-finite arrays in {path}")
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def montage_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [row["name"] for row in csv.DictReader(stream, delimiter="\t")]


def normalized_probes(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= 1e-14):
        raise ValueError("cortical probe has negligible referenced energy")
    return np.asarray(matrix / norms[None, :], dtype=np.float64)


def phase_factor(
    leadfield: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    full_sheet_fraction: float,
    anterior: np.ndarray,
    cycles: float,
) -> np.ndarray:
    phase = deterministic_phase(anterior[indices], cycles)
    selected = leadfield[:, indices]
    real = selected @ (weights * np.cos(phase))
    imaginary = selected @ (weights * np.sin(phase))
    factor = full_sheet_fraction * np.column_stack((real, imaginary))
    norms = np.linalg.norm(factor, axis=0)
    keep = norms > 1e-12 * max(float(np.max(norms)), np.finfo(float).tiny)
    if not np.any(keep):
        raise ValueError("hippocampal probe has zero sensor field")
    return np.asarray(factor[:, keep], dtype=np.float64)


def dense_probes(covariance: np.ndarray, count: int, seed: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    keep = values > 1e-14 * max(float(values[-1]), np.finfo(float).tiny)
    generator = np.random.default_rng(seed)
    draws = generator.normal(size=(int(np.count_nonzero(keep)), count))
    result = vectors[:, keep] @ (np.sqrt(values[keep])[:, None] * draws)
    return normalized_probes(result)


def select_from_pool(pool: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(pool) < count:
        raise ValueError("cortical probe pool is too small")
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(pool, size=count, replace=False)).astype(np.int64)


def dynamic_probe_sets(
    static: dict[str, np.ndarray],
    temporal_covariance: np.ndarray,
    count: int,
    sampling_frequency: float,
    subject: str,
    montage: str,
) -> dict[str, list[np.ndarray]]:
    samples = temporal_covariance.shape[0]
    values, vectors = np.linalg.eigh(0.5 * (temporal_covariance + temporal_covariance.T))
    values = np.maximum(values, 0.0)
    generator = np.random.default_rng(stable_seed(PROTOCOL, subject, montage, "dynamic"))
    empirical_time = vectors @ (np.sqrt(values)[:, None] * generator.normal(size=(samples, count)))
    empirical_time /= np.maximum(np.linalg.norm(empirical_time, axis=0, keepdims=True), 1e-14)
    time = np.arange(samples, dtype=np.float64) / sampling_frequency
    theta_time = np.empty((samples, count), dtype=np.float64)
    for index in range(count):
        frequency = (4.0, 6.0, 8.0)[index % 3]
        phase = generator.uniform(0.0, 2.0 * np.pi)
        theta_time[:, index] = np.sin(2.0 * np.pi * frequency * time + phase)
    theta_time /= np.linalg.norm(theta_time, axis=0, keepdims=True)
    transient_time = np.empty((samples, count), dtype=np.float64)
    for index in range(count):
        center = generator.uniform(0.25, 0.75)
        width = generator.uniform(0.025, 0.10)
        transient_time[:, index] = np.exp(-0.5 * ((time - center) / width) ** 2)
        transient_time[:, index] -= transient_time[:, index].mean()
    transient_time /= np.linalg.norm(transient_time, axis=0, keepdims=True)
    dense = static["dense_covariance"]
    smooth = static["smooth_heldout"]
    focal = static["focal_heldout"]
    return {
        "empirical_temporal": [np.outer(dense[:, i % dense.shape[1]], empirical_time[:, i]) for i in range(count)],
        "cortical_theta": [np.outer(smooth[:, i % smooth.shape[1]], theta_time[:, i]) for i in range(count)],
        "transient": [np.outer(focal[:, i % focal.shape[1]], transient_time[:, i]) for i in range(count)],
    }


def primary_supports(config: dict[str, Any]) -> list[tuple[str, str, float]]:
    result = []
    for row in config["hippocampal_probes"]["supports"]:
        result.append((str(row["hemisphere"]), str(row["location"]), float(row["extent"])))
    return result


def build(
    subject: str,
    config: dict[str, Any],
    forward_root: Path,
    legacy_root: Path,
    empirical_path: Path,
    output: Path,
) -> dict[str, object]:
    source_dir = forward_root / subject
    cortical_dir = legacy_root / subject
    paths = {
        "hippocampal": source_dir / "hippunfold-hippocampal-fixed-leadfield.npy",
        "hippocampal_metadata": source_dir / "hippunfold-hippocampal-source-metadata.npz",
        "forward_models_report": source_dir / "report.json",
        "cortical": cortical_dir / "cortical-fixed-leadfield.npy",
        "cortical_metadata": cortical_dir / "cortical-source-metadata.npz",
        "cortical_report": cortical_dir / "report.json",
        "montage": cortical_dir / "registered-montage.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cortical-restriction input files: {missing}")
    forward_models = json.loads(paths["forward_models_report"].read_text(encoding="utf-8"))
    cortical_report = json.loads(paths["cortical_report"].read_text(encoding="utf-8"))
    if forward_models.get("ok") is not True or cortical_report.get("ok") is not True:
        raise ValueError("upstream forward report is not complete")
    h_metadata = load_npz(
        paths["hippocampal_metadata"],
        (
            "positions_m",
            "directions",
            "area_weights_m2",
            "longitudinal_coordinate",
            "proximal_distal_coordinate",
            "hemisphere_code",
        ),
    )
    c_metadata = load_npz(
        paths["cortical_metadata"],
        ("positions_m", "directions", "area_weights_m2", "hemisphere_code"),
    )
    hippocampal_raw = np.load(paths["hippocampal"], mmap_mode="r", allow_pickle=False)
    cortical_raw = np.load(paths["cortical"], mmap_mode="r", allow_pickle=False)
    if hippocampal_raw.shape[0] != 339 or cortical_raw.shape[0] != 339:
        raise ValueError("cortical-restriction requires the validated 339-sensor operators")
    with np.load(empirical_path, allow_pickle=False) as archive:
        empirical_channels = np.asarray(archive["channels"], dtype=str)
        empirical_spatial = np.asarray(archive["spatial_covariance"], dtype=np.float64)
        empirical_temporal = np.asarray(archive["temporal_covariance"], dtype=np.float64)
        empirical_frequency = float(np.asarray(archive["target_sampling_frequency_hz"]).item())
    clinical_channels = list(config["montages"]["clinical8_channels"])
    if not np.array_equal(empirical_channels, np.asarray(clinical_channels, dtype=str)):
        raise ValueError("empirical covariance channels differ from the frozen montage")
    names = montage_names(paths["montage"])
    if len(names) != 339 or any(name not in names for name in clinical_channels):
        raise ValueError("registered montage does not contain the clinical channels")
    clinical_indices = np.asarray([names.index(name) for name in clinical_channels], dtype=np.int64)

    support_settings = config["hippocampal_probes"]["support_construction"]
    anterior, pd, _ = oriented_intrinsic_coordinates(
        h_metadata["positions_m"],
        h_metadata["longitudinal_coordinate"],
        h_metadata["proximal_distal_coordinate"],
        h_metadata["hemisphere_code"],
        h_metadata["area_weights_m2"],
        endpoint_decile=float(support_settings["endpoint_decile"]),
        minimum_endpoint_separation_m=float(support_settings["minimum_endpoint_separation_m"]),
    )
    supports, support_report = build_source_supports(
        anterior,
        pd,
        h_metadata["hemisphere_code"],
        h_metadata["area_weights_m2"],
        hemisphere_families=("left", "right", "bilateral"),
        focal_locations=("anterior", "middle", "posterior"),
        focal_extents=(0.05, 0.10, 0.25, 0.50),
        include_whole_extent=True,
    )
    requested_supports = [support_identifier(*row) for row in primary_supports(config)]
    if any(identifier not in supports for identifier in requested_supports):
        raise ValueError("frozen hippocampal support is absent")

    anchors = farthest_anchors(
        c_metadata["positions_m"],
        c_metadata["hemisphere_code"],
        c_metadata["area_weights_m2"],
        int(config["restrictions"]["smooth_anchors_per_hemisphere"]),
    )
    threshold = float(config["false_attribution"]["reporting_threshold"])
    probe_count = int(config["false_attribution"]["probes_per_family"])
    sensitivity_rows: list[dict[str, object]] = []
    false_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=False)

    for montage, sensor_indices in (("full339", None), ("clinical8", clinical_indices)):
        h_leadfield = referenced_operator(hippocampal_raw, sensor_indices)
        c_leadfield = referenced_operator(cortical_raw, sensor_indices)
        dimensions = c_leadfield.shape[0]
        covariance = area_weighted_covariance(c_leadfield, c_metadata["area_weights_m2"])
        forward_basis, _, forward_retained = covariance_basis(
            covariance, float(config["restrictions"]["covariance_retained_fraction"])
        )
        if montage == "clinical8":
            covariance_primary = empirical_spatial
            covariance_kind = "label_blind_ds004752"
        else:
            covariance_primary = covariance
            covariance_kind = "area_weighted_forward_ensemble"
        covariance_primary_basis, _, covariance_primary_retained = covariance_basis(
            covariance_primary, float(config["restrictions"]["covariance_retained_fraction"])
        )
        unrestricted = orthonormal_basis(c_leadfield)
        smooth_topography = smooth_topographies(
            c_leadfield,
            c_metadata["positions_m"],
            c_metadata["hemisphere_code"],
            c_metadata["area_weights_m2"],
            anchors,
            float(config["restrictions"]["smooth_width_m"]),
        )
        smooth_basis = orthonormal_basis(smooth_topography)
        normalized_cortical, usable_local = normalized_columns(c_leadfield)
        usable_source_indices = usable_local
        dictionary_indices, heldout_indices = deterministic_dictionary_split(
            c_metadata["area_weights_m2"],
            usable_source_indices,
            int(config["restrictions"]["sparse_dictionary_size"]),
            probe_count,
            stable_seed(PROTOCOL, subject, montage, "dictionary"),
        )
        dictionary = normalized_probes(c_leadfield[:, dictionary_indices])
        focal = normalized_probes(c_leadfield[:, heldout_indices])
        smooth_widths = (
            [float(value) for value in config["false_attribution"]["smooth_probe_widths_m"]]
            * (probe_count // len(config["false_attribution"]["smooth_probe_widths_m"]) + 1)
        )[:probe_count]
        smooth_heldout = normalized_probes(
            smooth_topographies(
                c_leadfield,
                c_metadata["positions_m"],
                c_metadata["hemisphere_code"],
                c_metadata["area_weights_m2"],
                heldout_indices,
                smooth_widths,
            )
        )
        dense = dense_probes(
            covariance, probe_count, stable_seed(PROTOCOL, subject, montage, "dense")
        )
        static_common = {
            "focal_heldout": focal,
            "smooth_heldout": smooth_heldout,
            "dense_covariance": dense,
        }
        common_sparse_bases = {
            (int(atoms), family): sparse_probe_bases(probes, dictionary, int(atoms))
            for atoms in config["restrictions"]["sparse_atom_ladder"]
            for family, probes in static_common.items()
        }
        for name, basis in (
            ("unrestricted", unrestricted),
            ("covariance", covariance_primary_basis),
            ("smooth", smooth_basis),
            ("forward_covariance_secondary", forward_basis),
        ):
            diagnostics.append(
                {
                    "subject": subject,
                    "montage": montage,
                    "restriction": name,
                    **basis_diagnostics(basis),
                }
            )
        if unrestricted.shape[1] != dimensions:
            raise ValueError("unrestricted cortical operator lacks full referenced rank")

        temporal_basis, _, temporal_retained = covariance_basis(
            empirical_temporal, float(config["restrictions"]["temporal_retained_fraction"])
        )
        for support_id in requested_supports:
            support = supports[support_id]
            neighborhood = neighborhood_indices(
                c_metadata["positions_m"],
                h_metadata["positions_m"][support.indices],
                float(config["restrictions"]["support_radius_m"]),
            )
            if len(neighborhood) < int(config["restrictions"]["minimum_support_sources"]):
                raise ValueError("hippocampal-near cortical support is too small")
            support_covariance = area_weighted_covariance(
                c_leadfield[:, neighborhood], c_metadata["area_weights_m2"][neighborhood]
            )
            support_basis, _, support_retained = covariance_basis(
                support_covariance, float(config["restrictions"]["support_retained_fraction"])
            )
            support_probe_indices = select_from_pool(
                neighborhood,
                probe_count,
                stable_seed(PROTOCOL, subject, montage, support_id, "near"),
            )
            temporal_neighborhood = normalized_probes(c_leadfield[:, support_probe_indices])
            static_probes = {**static_common, "temporal_neighborhood": temporal_neighborhood}
            support_sparse_bases = {
                int(atoms): sparse_probe_bases(temporal_neighborhood, dictionary, int(atoms))
                for atoms in config["restrictions"]["sparse_atom_ladder"]
            }
            diagnostics.append(
                {
                    "subject": subject,
                    "montage": montage,
                    "restriction": "support",
                    "support_id": support_id,
                    "support_source_count": len(neighborhood),
                    "support_area_m2": float(np.sum(c_metadata["area_weights_m2"][neighborhood])),
                    "retained_fraction": support_retained,
                    **basis_diagnostics(support_basis),
                }
            )
            for cycles in config["hippocampal_probes"]["wave_cycles"]:
                regime = "zero_phase" if float(cycles) == 0.0 else f"wave_{float(cycles):g}_cycles"
                factor = phase_factor(
                    h_leadfield,
                    support.indices,
                    support.conditional_weights,
                    support.full_sheet_area_fraction,
                    anterior,
                    float(cycles),
                )
                base = {
                    "subject": subject,
                    "montage": montage,
                    "sensor_dimension": dimensions,
                    "support_id": support_id,
                    "hemisphere": support.hemisphere,
                    "location": support.location,
                    "extent": support.target_extent,
                    "regime": regime,
                    "wave_cycles": float(cycles),
                    "hippocampal_factor_rank": factor.shape[1],
                }
                linear_models = {
                    "unrestricted": (unrestricted, True, 1.0),
                    "covariance": (covariance_primary_basis, True, covariance_primary_retained),
                    "smooth": (smooth_basis, True, float("nan")),
                    "support": (support_basis, True, support_retained),
                    "forward_covariance_secondary": (forward_basis, False, forward_retained),
                }
                for restriction, (basis, primary, retained) in linear_models.items():
                    metrics = residual_sensitivity(factor, basis)
                    sensitivity_rows.append(
                        {
                            **base,
                            "restriction": restriction,
                            "primary_ladder": primary,
                            "restriction_rank": basis.shape[1],
                            "restriction_retained_fraction": retained,
                            "covariance_kind": covariance_kind if restriction == "covariance" else "",
                            **metrics,
                        }
                    )
                    for family, probes in static_probes.items():
                        summary = summarize_false_attribution(
                            false_attribution(probes, factor, basis), threshold
                        )
                        false_rows.append(
                            {
                                **base,
                                "restriction": restriction,
                                "primary_ladder": primary,
                                "probe_domain": "static",
                                "probe_family": family,
                                **summary,
                            }
                        )
                for atoms in config["restrictions"]["sparse_atom_ladder"]:
                    restriction = f"sparse_k{int(atoms)}"
                    primary = int(atoms) == int(config["restrictions"]["sparse_primary_atoms"])
                    metrics = sparse_sensitivity(factor, dictionary, int(atoms))
                    sensitivity_rows.append(
                        {
                            **base,
                            "restriction": restriction,
                            "primary_ladder": primary,
                            "restriction_rank": int(atoms),
                            "restriction_retained_fraction": float("nan"),
                            "covariance_kind": "",
                            **metrics,
                        }
                    )
                    for family, probes in static_probes.items():
                        fitted = (
                            support_sparse_bases[int(atoms)]
                            if family == "temporal_neighborhood"
                            else common_sparse_bases[(int(atoms), family)]
                        )
                        summary = summarize_false_attribution(
                            sparse_false_from_bases(probes, factor, fitted),
                            threshold,
                        )
                        false_rows.append(
                            {
                                **base,
                                "restriction": restriction,
                                "primary_ladder": primary,
                                "probe_domain": "static",
                                "probe_family": family,
                                **summary,
                            }
                        )
                dynamic_h = dynamic_hippocampal_factors(
                    factor,
                    empirical_temporal.shape[0],
                    empirical_frequency,
                    float(config["restrictions"]["hippocampal_carrier_hz"]),
                )
                dynamic_metrics = tensor_sensitivity(
                    dynamic_h, covariance_primary_basis, temporal_basis
                )
                sensitivity_rows.append(
                    {
                        **base,
                        "restriction": "spatiotemporal",
                        "primary_ladder": True,
                        "restriction_rank": covariance_primary_basis.shape[1] * temporal_basis.shape[1],
                        "restriction_retained_fraction": covariance_primary_retained * temporal_retained,
                        "covariance_kind": covariance_kind,
                        **dynamic_metrics,
                    }
                )
                dynamic_sets = dynamic_probe_sets(
                    static_probes,
                    empirical_temporal,
                    probe_count,
                    empirical_frequency,
                    subject,
                    montage,
                )
                for family, probes in dynamic_sets.items():
                    summary = summarize_false_attribution(
                        tensor_false_attribution(
                            probes,
                            dynamic_h,
                            covariance_primary_basis,
                            temporal_basis,
                        ),
                        threshold,
                    )
                    false_rows.append(
                        {
                            **base,
                            "restriction": "spatiotemporal",
                            "primary_ladder": True,
                            "probe_domain": "spatiotemporal",
                            "probe_family": family,
                            **summary,
                        }
                    )

    sensitivity_path = output / "sensitivity.csv"
    false_path = output / "false_attribution.csv"
    diagnostics_path = output / "basis_diagnostics.csv"
    write_csv(sensitivity_path, sensitivity_rows)
    write_csv(false_path, false_rows)
    write_csv(diagnostics_path, diagnostics)
    tolerance = float(config["gates"]["numerical_fraction_tolerance"])
    sensitivity_values = np.asarray([float(row["sensitivity_fraction"]) for row in sensitivity_rows])
    false_values = np.asarray(
        [float(row[key]) for row in false_rows for key in ("median", "p95", "maximum", "fraction_above_threshold")]
    )
    unrestricted_sensitivity = np.asarray(
        [float(row["sensitivity_fraction"]) for row in sensitivity_rows if row["restriction"] == "unrestricted"]
    )
    unrestricted_false = np.asarray(
        [float(row["maximum"]) for row in false_rows if row["restriction"] == "unrestricted"]
    )
    max_orthonormality = max(float(row["orthonormality_max_error"]) for row in diagnostics)
    max_idempotence = max(float(row["projector_idempotence_relative_error"]) for row in diagnostics)
    expected_primary_per_montage = len(requested_supports) * len(config["hippocampal_probes"]["wave_cycles"]) * 6
    actual_primary = sum(bool(row["primary_ladder"]) for row in sensitivity_rows)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(
            np.all(np.isfinite(sensitivity_values))
            and np.all(np.isfinite(false_values))
            and np.min(sensitivity_values) >= -tolerance
            and np.max(sensitivity_values) <= 1.0 + tolerance
            and np.min(false_values) >= -tolerance
            and np.max(false_values) <= 1.0 + tolerance
            and np.max(unrestricted_sensitivity) <= float(config["gates"]["unrestricted_maximum_residual"])
            and np.max(unrestricted_false) <= float(config["gates"]["unrestricted_maximum_false_gain"])
            and max_orthonormality <= float(config["gates"]["maximum_basis_error"])
            and max_idempotence <= float(config["gates"]["maximum_basis_error"])
            and actual_primary == 2 * expected_primary_per_montage
        ),
        "subject": subject,
        "montages": ["full339", "clinical8"],
        "hippocampal_supports": requested_supports,
        "hippocampal_regimes": len(config["hippocampal_probes"]["wave_cycles"]),
        "sensitivity_rows": len(sensitivity_rows),
        "primary_sensitivity_rows": actual_primary,
        "expected_primary_sensitivity_rows": 2 * expected_primary_per_montage,
        "false_attribution_rows": len(false_rows),
        "maximum_basis_orthonormality_error": max_orthonormality,
        "maximum_projector_idempotence_error": max_idempotence,
        "maximum_unrestricted_sensitivity": float(np.max(unrestricted_sensitivity)),
        "maximum_unrestricted_false_attribution": float(np.max(unrestricted_false)),
        "sparse_dictionary_size": int(config["restrictions"]["sparse_dictionary_size"]),
        "sparse_dictionary_heldout_overlap": 0,
        "smooth_anchor_count": len(anchors),
        "support_construction_exact": bool(support_report["all_exactly_nested"]),
        "empirical_covariance_sha256": sha256(empirical_path),
        "inputs": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in paths.items()},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (sensitivity_path, false_path, diagnostics_path)
        },
        "physiological_inference_authorized": False,
        "shared_storage_touched": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("inputs", "outputs")}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--forward-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--empirical-covariance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise SystemExit("unexpected cortical-restriction protocol")
    report = build(
        args.subject,
        config,
        args.forward_root,
        args.legacy_root,
        args.empirical_covariance,
        args.output,
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
