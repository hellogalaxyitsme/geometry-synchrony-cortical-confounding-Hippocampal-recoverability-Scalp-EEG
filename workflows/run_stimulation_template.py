#!/usr/bin/env python3
"""Evaluate one HCP detector anatomy on all requested cortical CCEP sessions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.stimulation_control import (  # noqa: E402
    interpolate_rows,
    matrix_attribution_components,
    normalized_dictionary,
    sparse_basis,
    spherical_interpolation,
    transform_patient_operator,
)
from simulation.cortical_restriction import basis_diagnostics, orthonormal_basis  # noqa: E402
from theory.recoverability import helmert_reference  # noqa: E402


PROTOCOL = "cortical-control/adversarial-stimulation-v1"
ENDPOINTS = ("n1_peak", "n2_peak", "integrated_pca3")
RESTRICTIONS = ("unrestricted", "empirical_covariance", "smooth", "support", "sparse_k4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def attribution(targets: np.ndarray, factor: np.ndarray, basis: np.ndarray, restriction: str) -> dict[str, float]:
    if restriction == "unrestricted":
        return {
            "total_attribution_fraction": 0.0,
            "cortical_residual_energy_fraction": 0.0,
            "conditional_hippocampal_fraction": 0.0,
        }
    return matrix_attribution_components(targets, factor, basis)


def build(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cache_report = json.loads((args.cache_root / "cache_report.json").read_text(encoding="utf-8"))
    bundle_report = json.loads((args.bundle_root / "bundle_report.json").read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL or cache_report.get("ok") is not True or bundle_report.get("ok") is not True:
        raise SystemExit("stimulation-control configuration/cache/template gate failed")
    config_hash = sha256(args.config)
    if cache_report.get("config", {}).get("sha256") != config_hash or bundle_report.get("config", {}).get("sha256") != config_hash:
        raise SystemExit("stimulation-control cache/template configuration hash mismatch")
    requested = set(args.subject or cache_report["subjects"])
    session_rows = [row for row in cache_report["sessions"] if row["subject"] in requested]
    if requested != {row["subject"] for row in session_rows}:
        raise ValueError("requested CCEP subject absent from cache")
    with np.load(args.bundle_root / "bundle.npz", allow_pickle=False) as archive:
        bundle = {name: np.asarray(archive[name]) for name in archive.files}
    factor_ids = bundle["factor_ids"].astype(str).tolist()
    support_ids = bundle["support_ids"].astype(str).tolist()
    hcp_subject = str(bundle_report["hcp_subject"])
    interpolation_config = config["interpolation"]
    sparse_atoms = int(config["restrictions"]["sparse_atoms"])
    threshold = float(config["detector"]["reporting_threshold"])
    injection_index = int(bundle["injection_factor_index"])
    rows: list[dict[str, object]] = []
    injection_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    maximum_basis_error = 0.0
    maximum_unrestricted = 0.0
    minimum_dictionary_size = int(config["restrictions"]["sparse_dictionary_size"])

    for patient in sorted(requested):
        with np.load(args.cache_root / "montages" / f"{patient}.npz", allow_pickle=False) as montage:
            patient_positions = np.asarray(montage["coordinates_head_m"], dtype=np.float64)
        interpolation = spherical_interpolation(
            bundle["montage_positions_head_m"],
            patient_positions,
            int(interpolation_config["neighbours"]),
            float(interpolation_config["sigma_degrees"]),
        )
        coverage_rows.append(
            {
                "hcp_subject": hcp_subject,
                "ccepcoreg_subject": patient,
                "median_nearest_angle_degrees": float(np.median(interpolation.nearest_angle_degrees)),
                "p95_nearest_angle_degrees": float(np.quantile(interpolation.nearest_angle_degrees, 0.95)),
                "maximum_nearest_angle_degrees": float(np.max(interpolation.nearest_angle_degrees)),
            }
        )
        mapped_h = np.stack(
            [interpolate_rows(bundle["hippocampal_factors"][index], interpolation) for index in range(len(factor_ids))],
            axis=0,
        )
        mapped_smooth = interpolate_rows(bundle["smooth_topographies"], interpolation)
        mapped_dictionary = interpolate_rows(bundle["sparse_dictionary"], interpolation)
        mapped_support = np.stack(
            [interpolate_rows(bundle["support_factors"][index], interpolation) for index in range(len(support_ids))],
            axis=0,
        )

        for session in (row for row in session_rows if row["subject"] == patient):
            cache_path = args.cache_root / str(session["cache"])
            with np.load(cache_path, allow_pickle=False) as cached:
                cache = {name: np.asarray(cached[name]) for name in cached.files}
            good = cache["good_indices"].astype(np.int64)
            scales = np.asarray(cache["robust_scales"], dtype=np.float64)
            dimension = len(good)
            unrestricted = np.asarray(helmert_reference(dimension).T, dtype=np.float64)
            empirical = np.asarray(cache["covariance_basis"], dtype=np.float64)
            smooth = orthonormal_basis(transform_patient_operator(mapped_smooth, good, scales))
            support_bases = [
                orthonormal_basis(
                    transform_patient_operator(
                        mapped_support[index, :, : int(bundle["support_ranks"][index])], good, scales
                    )
                )
                for index in range(len(support_ids))
            ]
            dictionary, dictionary_keep = normalized_dictionary(
                transform_patient_operator(mapped_dictionary, good, scales)
            )
            minimum_dictionary_size = min(minimum_dictionary_size, dictionary.shape[1])
            transformed_h = [
                transform_patient_operator(
                    mapped_h[index, :, : int(bundle["factor_ranks"][index])], good, scales
                )
                for index in range(len(factor_ids))
            ]
            common_bases = {
                "unrestricted": unrestricted,
                "empirical_covariance": empirical,
                "smooth": smooth,
            }
            for basis in (unrestricted, empirical, smooth, *support_bases):
                diagnostics = basis_diagnostics(basis)
                maximum_basis_error = max(
                    maximum_basis_error,
                    diagnostics["orthonormality_max_error"],
                    diagnostics["projector_idempotence_relative_error"],
                )

            for endpoint in ENDPOINTS:
                response = np.asarray(cache[f"{endpoint}_response"], dtype=np.float64)
                baseline = np.asarray(cache[f"{endpoint}_baseline"], dtype=np.float64)
                sparse_response, selected_response = sparse_basis(dictionary, response, sparse_atoms)
                sparse_baseline, selected_baseline = sparse_basis(dictionary, baseline, sparse_atoms)
                for factor_index, factor_id in enumerate(factor_ids):
                    support_index = int(bundle["factor_support_index"][factor_index])
                    bases = {
                        **common_bases,
                        "support": support_bases[support_index],
                        "sparse_k4": sparse_response,
                    }
                    baseline_bases = {**bases, "sparse_k4": sparse_baseline}
                    for restriction in RESTRICTIONS:
                        response_components = attribution(
                            response, transformed_h[factor_index], bases[restriction], restriction
                        )
                        baseline_components = attribution(
                            baseline, transformed_h[factor_index], baseline_bases[restriction], restriction
                        )
                        if restriction == "unrestricted":
                            maximum_unrestricted = max(
                                maximum_unrestricted,
                                *response_components.values(),
                                *baseline_components.values(),
                            )
                        rows.append(
                            {
                                "hcp_subject": hcp_subject,
                                "ccepcoreg_subject": patient,
                                "run": session["run"],
                                "stem": session["stem"],
                                "endpoint": endpoint,
                                "restriction": restriction,
                                "factor_id": factor_id,
                                "support_id": support_ids[support_index],
                                "wave_cycles": float(bundle["factor_cycles"][factor_index]),
                                "factor_rank": int(bundle["factor_ranks"][factor_index]),
                                "restriction_rank_response": int(selected_response.size if restriction == "sparse_k4" else bases[restriction].shape[1]),
                                "restriction_rank_baseline": int(selected_baseline.size if restriction == "sparse_k4" else baseline_bases[restriction].shape[1]),
                                "response_score": response_components["total_attribution_fraction"],
                                "response_residual_energy_fraction": response_components["cortical_residual_energy_fraction"],
                                "response_conditional_score": response_components["conditional_hippocampal_fraction"],
                                "baseline_score": baseline_components["total_attribution_fraction"],
                                "baseline_residual_energy_fraction": baseline_components["cortical_residual_energy_fraction"],
                                "baseline_conditional_score": baseline_components["conditional_hippocampal_fraction"],
                                "response_above_threshold": response_components["conditional_hippocampal_fraction"] > threshold,
                                "baseline_above_threshold": baseline_components["conditional_hippocampal_fraction"] > threshold,
                            }
                        )

                if endpoint == str(config["windows"]["primary_endpoint"]):
                    injection_factor = transformed_h[injection_index]
                    injection_support = int(bundle["factor_support_index"][injection_index])
                    signal_topography = injection_factor[:, 0]
                    signal_topography /= max(float(np.linalg.norm(signal_topography)), np.finfo(float).tiny)
                    baseline_energy = float(np.sum(baseline * baseline))
                    for energy_ratio in config["detector"]["injection_energy_ratios"]:
                        injected = baseline.copy()
                        injected[:, 0] += signal_topography * np.sqrt(float(energy_ratio) * baseline_energy)
                        injected_sparse, selected_injected = sparse_basis(dictionary, injected, sparse_atoms)
                        injection_bases = {
                            **common_bases,
                            "support": support_bases[injection_support],
                            "sparse_k4": injected_sparse,
                        }
                        for restriction in RESTRICTIONS:
                            injection_components = attribution(
                                injected, injection_factor, injection_bases[restriction], restriction
                            )
                            injection_rows.append(
                                {
                                    "hcp_subject": hcp_subject,
                                    "ccepcoreg_subject": patient,
                                    "run": session["run"],
                                    "stem": session["stem"],
                                    "endpoint": endpoint,
                                    "restriction": restriction,
                                    "factor_id": factor_ids[injection_index],
                                    "injection_energy_ratio": float(energy_ratio),
                                    "restriction_rank": int(selected_injected.size if restriction == "sparse_k4" else injection_bases[restriction].shape[1]),
                                    "injection_score": injection_components["total_attribution_fraction"],
                                    "injection_residual_energy_fraction": injection_components["cortical_residual_energy_fraction"],
                                    "injection_conditional_score": injection_components["conditional_hippocampal_fraction"],
                                    "above_threshold": injection_components["conditional_hippocampal_fraction"] > threshold,
                                }
                            )

    args.output.mkdir(parents=True, exist_ok=False)
    attribution_path = args.output / "attribution.csv"
    injection_path = args.output / "injection.csv"
    coverage_path = args.output / "coverage.csv"
    scan: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["ccepcoreg_subject"]), str(row["run"]), str(row["endpoint"]), str(row["restriction"]))
        if key not in scan:
            scan[key] = {
                "hcp_subject": hcp_subject,
                "ccepcoreg_subject": key[0],
                "run": key[1],
                "endpoint": key[2],
                "restriction": key[3],
                "response_scan_max": float(row["response_score"]),
                "response_conditional_scan_max": float(row["response_conditional_score"]),
                "response_residual_at_conditional_max": float(row["response_residual_energy_fraction"]),
                "response_total_at_conditional_max": float(row["response_score"]),
                "baseline_scan_max": float(row["baseline_score"]),
                "baseline_conditional_scan_max": float(row["baseline_conditional_score"]),
                "baseline_residual_at_conditional_max": float(row["baseline_residual_energy_fraction"]),
                "baseline_total_at_conditional_max": float(row["baseline_score"]),
                "response_factor_id": str(row["factor_id"]),
                "baseline_factor_id": str(row["factor_id"]),
            }
        else:
            if float(row["response_score"]) > float(scan[key]["response_scan_max"]):
                scan[key]["response_scan_max"] = float(row["response_score"])
            if float(row["baseline_score"]) > float(scan[key]["baseline_scan_max"]):
                scan[key]["baseline_scan_max"] = float(row["baseline_score"])
            if float(row["response_conditional_score"]) > float(scan[key]["response_conditional_scan_max"]):
                scan[key]["response_conditional_scan_max"] = float(row["response_conditional_score"])
                scan[key]["response_residual_at_conditional_max"] = float(row["response_residual_energy_fraction"])
                scan[key]["response_total_at_conditional_max"] = float(row["response_score"])
                scan[key]["response_factor_id"] = str(row["factor_id"])
            if float(row["baseline_conditional_score"]) > float(scan[key]["baseline_conditional_scan_max"]):
                scan[key]["baseline_conditional_scan_max"] = float(row["baseline_conditional_score"])
                scan[key]["baseline_residual_at_conditional_max"] = float(row["baseline_residual_energy_fraction"])
                scan[key]["baseline_total_at_conditional_max"] = float(row["baseline_score"])
                scan[key]["baseline_factor_id"] = str(row["factor_id"])
    scan_rows = [scan[key] for key in sorted(scan)]
    scan_path = args.output / "scan_max.csv"
    write_csv(attribution_path, rows)
    write_csv(injection_path, injection_rows)
    write_csv(coverage_path, coverage_rows)
    write_csv(scan_path, scan_rows)
    expected_rows = len(session_rows) * len(ENDPOINTS) * len(factor_ids) * len(RESTRICTIONS)
    expected_injections = (
        len(session_rows) * len(config["detector"]["injection_energy_ratios"]) * len(RESTRICTIONS)
    )
    expected_scan_rows = len(session_rows) * len(ENDPOINTS) * len(RESTRICTIONS)
    values = np.asarray(
        [
            float(row[key])
            for row in rows
            for key in (
                "response_score",
                "response_residual_energy_fraction",
                "response_conditional_score",
                "baseline_score",
                "baseline_residual_energy_fraction",
                "baseline_conditional_score",
            )
        ]
        + [
            float(row[key])
            for row in injection_rows
            for key in ("injection_score", "injection_residual_energy_fraction", "injection_conditional_score")
        ],
        dtype=np.float64,
    )
    tolerance = float(config["gates"]["numerical_fraction_tolerance"])
    maximum_coverage = max(float(row["maximum_nearest_angle_degrees"]) for row in coverage_rows)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(
            len(rows) == expected_rows
            and len(injection_rows) == expected_injections
            and len(scan_rows) == expected_scan_rows
            and np.all(np.isfinite(values))
            and float(np.min(values)) >= -tolerance
            and float(np.max(values)) <= 1.0 + tolerance
            and maximum_unrestricted <= float(config["gates"]["maximum_unrestricted_score"])
            and maximum_basis_error <= float(config["gates"]["maximum_basis_error"])
            and maximum_coverage <= float(interpolation_config["maximum_patient_nearest_angle_degrees"])
            and minimum_dictionary_size == int(config["restrictions"]["sparse_dictionary_size"])
        ),
        "hcp_subject": hcp_subject,
        "ccepcoreg_subjects": sorted(requested),
        "ccepcoreg_subject_count": len(requested),
        "session_count": len(session_rows),
        "factor_count": len(factor_ids),
        "restriction_count": len(RESTRICTIONS),
        "endpoint_count": len(ENDPOINTS),
        "attribution_rows": len(rows),
        "expected_attribution_rows": expected_rows,
        "injection_rows": len(injection_rows),
        "expected_injection_rows": expected_injections,
        "scan_rows": len(scan_rows),
        "expected_scan_rows": expected_scan_rows,
        "maximum_basis_error": maximum_basis_error,
        "maximum_unrestricted_score": maximum_unrestricted,
        "maximum_patient_nearest_angle_degrees": maximum_coverage,
        "minimum_mapped_sparse_dictionary_size": minimum_dictionary_size,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (attribution_path, injection_path, coverage_path, scan_path)
        },
        "inputs": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "cache_report": {"path": str(args.cache_root / "cache_report.json"), "sha256": sha256(args.cache_root / "cache_report.json")},
            "bundle_report": {"path": str(args.bundle_root / "bundle_report.json"), "sha256": sha256(args.bundle_root / "bundle_report.json")},
            "bundle": {"path": str(args.bundle_root / "bundle.npz"), "sha256": sha256(args.bundle_root / "bundle.npz")},
        },
        "ieeg_waveforms_read": False,
        "physiological_hippocampal_recovery_inference_authorized": False,
        "shared_storage_touched": False,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("inputs", "outputs")}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", action="append")
    args = parser.parse_args()
    return 0 if build(args)["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
