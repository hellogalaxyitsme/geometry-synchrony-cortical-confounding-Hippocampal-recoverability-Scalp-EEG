#!/usr/bin/env python3
"""Build one transportable HCP anatomy bundle for the stimulation-control detector."""

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
from empirical.stimulation_control import leave_one_out_interpolation  # noqa: E402
from simulation.cortical_restriction import (  # noqa: E402
    deterministic_dictionary_split,
    farthest_anchors,
    neighborhood_indices,
    smooth_topographies,
    stable_seed,
)


PROTOCOL = "cortical-control/adversarial-stimulation-v1"
CORTICAL_RESTRICTION_PROTOCOL = "restriction/ladder-v1"


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
        return {name: np.asarray(archive[name]) for name in names}


def montage(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return (
        np.asarray([row["name"] for row in rows], dtype=str),
        np.asarray([[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows], dtype=np.float64),
    )


def requested_supports(config: dict[str, Any]) -> list[str]:
    return [
        support_identifier(str(row["hemisphere"]), str(row["location"]), float(row["extent"]))
        for row in config["hippocampal_probes"]["supports"]
    ]


def phase_factor_raw(
    leadfield: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    full_sheet_fraction: float,
    anterior: np.ndarray,
    cycles: float,
) -> np.ndarray:
    phase = deterministic_phase(anterior[indices], cycles)
    selected = np.asarray(leadfield[:, indices], dtype=np.float64)
    factor = full_sheet_fraction * np.column_stack(
        (selected @ (weights * np.cos(phase)), selected @ (weights * np.sin(phase)))
    )
    centred = factor - factor.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=0)
    keep = norms > 1e-12 * max(float(np.max(norms)), np.finfo(float).tiny)
    if not np.any(keep):
        raise ValueError("hippocampal factor has zero referenced energy")
    return np.asarray(factor[:, keep], dtype=np.float64)


def covariance_factor(operator: np.ndarray, weights: np.ndarray, retained: float) -> tuple[np.ndarray, float]:
    normalized = np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
    weighted = np.asarray(operator, dtype=np.float64) * np.sqrt(normalized)[None, :]
    covariance = weighted @ weighted.T
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("support covariance has zero trace")
    rank = int(np.searchsorted(np.cumsum(values), retained * total, side="left")) + 1
    realized = float(np.sum(values[:rank]) / total)
    return np.asarray(vectors[:, :rank] * np.sqrt(values[:rank])[None, :], dtype=np.float64), realized


def build(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise SystemExit("unexpected stimulation-control protocol")
    h_dir = args.hippunfold_root / args.subject
    c_dir = args.cortical_root / args.subject
    p5_report_path = args.cortical_restriction_root / "subjects" / args.subject / "report.json"
    paths = {
        "hippocampal": h_dir / "hippunfold-hippocampal-fixed-leadfield.npy",
        "hippocampal_metadata": h_dir / "hippunfold-hippocampal-source-metadata.npz",
        "hippocampal_report": h_dir / "report.json",
        "cortical": c_dir / "cortical-fixed-leadfield.npy",
        "cortical_metadata": c_dir / "cortical-source-metadata.npz",
        "cortical_report": c_dir / "report.json",
        "montage": c_dir / "registered-montage.tsv",
        "cortical_restriction_report": p5_report_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing stimulation-control template inputs: {missing}")
    if any(json.loads(paths[name].read_text(encoding="utf-8")).get("ok") is not True for name in ("hippocampal_report", "cortical_report", "cortical_restriction_report")):
        raise ValueError("an upstream report did not pass")
    h_meta = load_npz(
        paths["hippocampal_metadata"],
        ("positions_m", "directions", "area_weights_m2", "longitudinal_coordinate", "proximal_distal_coordinate", "hemisphere_code"),
    )
    c_meta = load_npz(paths["cortical_metadata"], ("positions_m", "area_weights_m2", "hemisphere_code"))
    h_raw = np.load(paths["hippocampal"], mmap_mode="r", allow_pickle=False)
    c_raw = np.load(paths["cortical"], mmap_mode="r", allow_pickle=False)
    names, positions = montage(paths["montage"])
    if h_raw.shape[0] != 339 or c_raw.shape[0] != 339 or len(names) != 339:
        raise ValueError("stimulation-control requires the validated 339-position HCP operators")
    support_settings = config["hippocampal_probes"]["support_construction"]
    anterior, pd, orientation_report = oriented_intrinsic_coordinates(
        h_meta["positions_m"],
        h_meta["longitudinal_coordinate"],
        h_meta["proximal_distal_coordinate"],
        h_meta["hemisphere_code"],
        h_meta["area_weights_m2"],
        endpoint_decile=float(support_settings["endpoint_decile"]),
        minimum_endpoint_separation_m=float(support_settings["minimum_endpoint_separation_m"]),
    )
    supports, support_report = build_source_supports(
        anterior,
        pd,
        h_meta["hemisphere_code"],
        h_meta["area_weights_m2"],
        hemisphere_families=("left", "right", "bilateral"),
        focal_locations=("anterior", "middle", "posterior"),
        focal_extents=(0.05, 0.10, 0.25, 0.50),
        include_whole_extent=True,
    )
    support_ids = requested_supports(config)
    if any(identifier not in supports for identifier in support_ids):
        raise ValueError("a frozen hippocampal support is absent")

    factors: list[np.ndarray] = []
    factor_ids: list[str] = []
    factor_support_index: list[int] = []
    factor_cycles: list[float] = []
    for support_index, identifier in enumerate(support_ids):
        support = supports[identifier]
        for cycles in config["hippocampal_probes"]["wave_cycles"]:
            factors.append(
                phase_factor_raw(
                    h_raw,
                    support.indices,
                    support.conditional_weights,
                    support.full_sheet_area_fraction,
                    anterior,
                    float(cycles),
                )
            )
            factor_ids.append(f"{identifier}__cycles_{float(cycles):g}")
            factor_support_index.append(support_index)
            factor_cycles.append(float(cycles))
    factor_ranks = np.asarray([value.shape[1] for value in factors], dtype=np.int64)
    factor_array = np.zeros((len(factors), 339, int(np.max(factor_ranks))), dtype=np.float64)
    for index, value in enumerate(factors):
        factor_array[index, :, : value.shape[1]] = value

    anchors = farthest_anchors(
        c_meta["positions_m"],
        c_meta["hemisphere_code"],
        c_meta["area_weights_m2"],
        int(config["restrictions"]["smooth_anchors_per_hemisphere"]),
    )
    smooth = smooth_topographies(
        c_raw,
        c_meta["positions_m"],
        c_meta["hemisphere_code"],
        c_meta["area_weights_m2"],
        anchors,
        float(config["restrictions"]["smooth_width_m"]),
    )

    support_factors: list[np.ndarray] = []
    support_retained: list[float] = []
    support_source_count: list[int] = []
    for identifier in support_ids:
        neighborhood = neighborhood_indices(
            c_meta["positions_m"],
            h_meta["positions_m"][supports[identifier].indices],
            float(config["restrictions"]["support_radius_m"]),
        )
        if len(neighborhood) < int(config["restrictions"]["minimum_support_sources"]):
            raise ValueError(f"too few cortical support sources for {identifier}")
        factor, realized = covariance_factor(
            np.asarray(c_raw[:, neighborhood], dtype=np.float64),
            c_meta["area_weights_m2"][neighborhood],
            float(config["restrictions"]["support_retained_fraction"]),
        )
        support_factors.append(factor)
        support_retained.append(realized)
        support_source_count.append(len(neighborhood))
    support_ranks = np.asarray([value.shape[1] for value in support_factors], dtype=np.int64)
    support_array = np.zeros((len(support_factors), 339, int(np.max(support_ranks))), dtype=np.float64)
    for index, value in enumerate(support_factors):
        support_array[index, :, : value.shape[1]] = value

    centred_norm = np.linalg.norm(np.asarray(c_raw, dtype=np.float64) - np.mean(c_raw, axis=0, keepdims=True), axis=0)
    usable = np.flatnonzero(centred_norm > 1e-14)
    dictionary_indices, heldout = deterministic_dictionary_split(
        c_meta["area_weights_m2"],
        usable,
        int(config["restrictions"]["sparse_dictionary_size"]),
        int(config["interpolation"]["cortical_validation_columns"]),
        stable_seed(CORTICAL_RESTRICTION_PROTOCOL, args.subject, "full339", "dictionary"),
    )
    dictionary = np.asarray(c_raw[:, dictionary_indices], dtype=np.float64)
    heldout_cortical = np.asarray(c_raw[:, heldout], dtype=np.float64)
    interpolation_config = config["interpolation"]
    hippocampal_validation = leave_one_out_interpolation(
        positions,
        np.column_stack(factors),
        int(interpolation_config["neighbours"]),
        float(interpolation_config["sigma_degrees"]),
    )
    cortical_validation = leave_one_out_interpolation(
        positions,
        heldout_cortical,
        int(interpolation_config["neighbours"]),
        float(interpolation_config["sigma_degrees"]),
    )
    validation_ok = all(
        report["relative_frobenius_error"] <= float(interpolation_config["maximum_loo_relative_error"])
        and report["median_referenced_topography_cosine"] >= float(interpolation_config["minimum_loo_median_topography_cosine"])
        for report in (hippocampal_validation, cortical_validation)
    )
    injection = config["hippocampal_probes"]["injection_support"]
    injection_id = (
        f"{support_identifier(str(injection['hemisphere']), str(injection['location']), float(injection['extent']))}"
        f"__cycles_{float(injection['wave_cycles']):g}"
    )
    if injection_id not in factor_ids:
        raise ValueError("injection factor is absent from the frozen scan")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle_path = args.output / "bundle.npz"
    np.savez_compressed(
        bundle_path,
        montage_names=names,
        montage_positions_head_m=positions,
        factor_ids=np.asarray(factor_ids, dtype=str),
        factor_support_index=np.asarray(factor_support_index, dtype=np.int64),
        factor_cycles=np.asarray(factor_cycles, dtype=np.float64),
        factor_ranks=factor_ranks,
        hippocampal_factors=factor_array,
        support_ids=np.asarray(support_ids, dtype=str),
        support_ranks=support_ranks,
        support_factors=support_array,
        support_retained_fraction=np.asarray(support_retained, dtype=np.float64),
        support_source_count=np.asarray(support_source_count, dtype=np.int64),
        smooth_topographies=smooth,
        sparse_dictionary=dictionary,
        sparse_dictionary_indices=dictionary_indices,
        sparse_validation_indices=heldout,
        injection_factor_index=np.asarray(factor_ids.index(injection_id), dtype=np.int64),
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(validation_ok and support_report["all_exactly_nested"]),
        "hcp_subject": args.subject,
        "factor_count": len(factors),
        "factor_ids": factor_ids,
        "support_ids": support_ids,
        "support_ranks": support_ranks.tolist(),
        "support_source_counts": support_source_count,
        "support_retained_fractions": support_retained,
        "smooth_topography_count": smooth.shape[1],
        "sparse_dictionary_size": dictionary.shape[1],
        "sparse_dictionary_heldout_overlap": int(np.intersect1d(dictionary_indices, heldout).size),
        "hippocampal_leave_one_out": hippocampal_validation,
        "cortical_leave_one_out": cortical_validation,
        "injection_factor_id": injection_id,
        "orientation": orientation_report,
        "support_construction_exact": bool(support_report["all_exactly_nested"]),
        "bundle": {"path": bundle_path.name, "bytes": bundle_path.stat().st_size, "sha256": sha256(bundle_path)},
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "inputs": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in paths.items()},
        "shared_storage_touched": False,
    }
    report_path = args.output / "bundle_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("inputs", "orientation")}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hippunfold-root", type=Path, required=True)
    parser.add_argument("--cortical-root", type=Path, required=True)
    parser.add_argument("--cortical_restriction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if build(args)["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
