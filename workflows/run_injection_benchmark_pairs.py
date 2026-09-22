#!/usr/bin/env python3
"""Run the corrected seven-method control for one mismatched HCP anatomy pair."""

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
sys.path.insert(0, str(PROJECT))

from empirical.stimulation_control import interpolate_rows, spherical_interpolation, transform_patient_operator  # noqa: E402
from empirical.estimators import (  # noqa: E402
    anatomical_log_power_ratio,
    covariance_factors,
    eloreta_type_operator,
    fit_fastica,
    lcmv_operator,
    minimum_norm_operator,
    sparse_group_fista,
)
from empirical.source_scores import factor_covariance, stable_anatomical_log_power_ratio  # noqa: E402
from empirical.injection_sources import build_injection_sources  # noqa: E402
from empirical.injection_benchmark import (  # noqa: E402
    added_signal_ratio_from_total_energy_ratio,
    anatomy_derangement,
    balanced_source_schedule,
    best_channel_factor_scores,
    ica_factor_scores,
    stochastic_factor_injection_covariance,
)
from theory.recoverability import helmert_reference  # noqa: E402


PROTOCOL = "benchmark/cortical-method-control-v2"
METHODS = (
    "best_channel", "fastica_bss", "minimum_norm", "eloreta_type", "lcmv",
    "sparse_l1", "hierarchical_sparse_group",
)
ENDPOINTS = ("n1_peak", "n2_peak", "integrated_pca3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def inverse_score(operator: np.ndarray, covariance: np.ndarray, hippocampus: np.ndarray, cortex: np.ndarray) -> float:
    return float(anatomical_log_power_ratio(operator, covariance[None, :, :], hippocampus, cortex)[0])


def sparse_score(
    method: str,
    design: np.ndarray,
    covariance: np.ndarray,
    hippocampus: np.ndarray,
    cortex: np.ndarray,
    groups: list[np.ndarray],
    settings: dict[str, Any],
) -> tuple[float, dict[str, float | int | bool]]:
    target = covariance_factors(covariance, int(settings["covariance_components"]))
    maximum = float(np.max(np.abs(design.T @ target)))
    penalty = float(settings["penalty_fraction"]) * max(maximum, np.finfo(float).tiny)
    if method == "sparse_l1":
        coefficients, report = sparse_group_fista(
            design, target, penalty,
            maximum_iterations=int(settings["maximum_iterations"]),
            tolerance=float(settings["tolerance"]),
        )
    else:
        coefficients, report = sparse_group_fista(
            design,
            target,
            float(settings["l1_fraction_of_group_penalty"]) * penalty,
            groups=groups,
            group_penalty=penalty,
            maximum_iterations=int(settings["maximum_iterations"]),
            tolerance=float(settings["tolerance"]),
        )
    return stable_anatomical_log_power_ratio(coefficients, hippocampus, cortex), report


def load_metadata(path: Path) -> dict[str, np.ndarray]:
    required = (
        "positions_m", "area_weights_m2", "longitudinal_coordinate",
        "proximal_distal_coordinate", "hemisphere_code",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"source metadata is missing arrays: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detector-bundle", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-forward-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--detector-subject", required=True)
    parser.add_argument("--source-subject", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected injection-benchmark protocol")
    subjects = [
        value.strip() for value in (PROJECT / str(config["hcp_subjects_file"])).read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    mapping = anatomy_derangement(subjects, int(config["cross_anatomy"]["cyclic_offset"]))
    if mapping.get(args.detector_subject) != args.source_subject or args.detector_subject == args.source_subject:
        raise ValueError("detector/source anatomy pair violates the frozen derangement")

    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    if calibration.get("protocol") != "calibration/continuous-false-positive-v1.1" or calibration.get("ok") is not True:
        raise ValueError("invalid empirical calibration report")
    total_ratio = float(calibration["mesial_event_physical_calibration"]["injection_calibration_energy_ratio"])
    added_ratio = added_signal_ratio_from_total_energy_ratio(total_ratio)
    declared = config["energy_calibration"]
    if declared.get("input_semantics") != "event_to_baseline_total_energy_ratio" or declared.get("injected_semantics") != "added_signal_to_baseline_energy_ratio":
        raise ValueError("energy calibration semantics are not explicit")

    cache_report = json.loads((args.cache_root / "cache_report.json").read_text(encoding="utf-8"))
    with np.load(args.detector_bundle, allow_pickle=False) as archive:
        detector = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.source_bundle, allow_pickle=False) as archive:
        source_montage_positions = np.asarray(archive["montage_positions_head_m"], dtype=np.float64)
    source_forward = args.source_forward_dir
    source_leadfield = np.load(source_forward / "hippunfold-hippocampal-fixed-leadfield.npy", mmap_mode="r", allow_pickle=False)
    source_metadata = load_metadata(source_forward / "hippunfold-hippocampal-source-metadata.npz")
    sources, source_report = build_injection_sources(
        source_leadfield,
        source_metadata,
        config["source_ensemble"],
        args.source_subject,
        int(config["master_seed"]),
    )
    expected_ids = [str(row["id"]) for row in config["source_ensemble"]["sources"]]
    if [source.identifier for source in sources] != expected_ids:
        raise ValueError("realized source ensemble differs from frozen ordering")
    source_ranks = np.asarray([source.factor.shape[1] for source in sources], dtype=np.int64)
    source_offsets = np.concatenate(([0], np.cumsum(source_ranks)))
    source_matrix = np.column_stack([source.factor for source in sources])

    schedules = {
        endpoint: balanced_source_schedule(
            len(cache_report["sessions"]), len(sources), int(config["master_seed"]), "source_schedule", endpoint
        )
        for endpoint in ENDPOINTS
    }
    mapped_detector: dict[str, dict[str, np.ndarray]] = {}
    mapped_sources: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    methods = config["methods"]
    maximum_total_ratio_error = 0.0
    for session_index, session in enumerate(cache_report["sessions"]):
        patient = str(session["subject"])
        if patient not in mapped_detector:
            with np.load(args.cache_root / "montages" / f"{patient}.npz", allow_pickle=False) as montage:
                patient_positions = np.asarray(montage["coordinates_head_m"], dtype=np.float64)
            detector_interpolation = spherical_interpolation(
                np.asarray(detector["montage_positions_head_m"], dtype=np.float64),
                patient_positions,
                int(config["interpolation"]["neighbours"]),
                float(config["interpolation"]["sigma_degrees"]),
            )
            h_shape = detector["hippocampal_factors"].shape
            mapped_h = interpolate_rows(
                np.asarray(detector["hippocampal_factors"], dtype=np.float64).transpose(1, 0, 2).reshape(h_shape[1], -1),
                detector_interpolation,
            ).reshape(len(patient_positions), h_shape[0], h_shape[2]).transpose(1, 0, 2)
            mapped_detector[patient] = {
                "smooth": interpolate_rows(np.asarray(detector["smooth_topographies"], dtype=np.float64), detector_interpolation),
                "hippocampus": mapped_h,
            }
            source_interpolation = spherical_interpolation(
                source_montage_positions,
                patient_positions,
                int(config["interpolation"]["neighbours"]),
                float(config["interpolation"]["sigma_degrees"]),
            )
            mapped_sources[patient] = interpolate_rows(source_matrix, source_interpolation)

        with np.load(args.cache_root / str(session["cache"]), allow_pickle=False) as cache:
            arrays = {name: np.asarray(cache[name], dtype=np.float64) for name in cache.files}
        good = np.asarray(arrays["good_indices"], dtype=np.int64)
        scales = np.asarray(arrays["robust_scales"], dtype=np.float64)
        contrast = np.asarray(helmert_reference(len(good)), dtype=np.float64)
        smooth = contrast @ transform_patient_operator(mapped_detector[patient]["smooth"], good, scales)
        h_columns: list[np.ndarray] = []
        h_groups: list[int] = []
        for factor_index, rank in enumerate(np.asarray(detector["factor_ranks"], dtype=np.int64)):
            transformed = contrast @ transform_patient_operator(
                mapped_detector[patient]["hippocampus"][factor_index, :, : int(rank)], good, scales
            )
            h_columns.extend(transformed[:, component] for component in range(transformed.shape[1]))
            h_groups.extend([2 + factor_index] * transformed.shape[1])
        hippocampal = np.column_stack(h_columns)
        gain = np.column_stack((smooth, hippocampal))
        cortical_indices = np.arange(smooth.shape[1], dtype=np.int64)
        hippocampal_indices = np.arange(smooth.shape[1], gain.shape[1], dtype=np.int64)
        group_ids = np.concatenate(
            (
                np.repeat([0, 1], [smooth.shape[1] // 2, smooth.shape[1] - smooth.shape[1] // 2]),
                np.asarray(h_groups, dtype=np.int64),
            )
        )
        groups = [np.flatnonzero(group_ids == value) for value in np.unique(group_ids)]
        endpoint_samples = []
        for endpoint in ENDPOINTS:
            endpoint_samples.extend((contrast @ arrays[f"{endpoint}_response"], contrast @ arrays[f"{endpoint}_baseline"]))
        ica_samples = np.column_stack(endpoint_samples).T
        components = min(int(methods["fastica_bss"]["components"]), ica_samples.shape[0] - 1, ica_samples.shape[1])
        ica = fit_fastica(
            ica_samples,
            components,
            seed=int(config["master_seed"]) + session_index + 1,
            maximum_iterations=int(methods["fastica_bss"]["maximum_iterations"]),
            tolerance=float(methods["fastica_bss"]["tolerance"]),
        )

        for endpoint in ENDPOINTS:
            response_covariance = factor_covariance(contrast @ arrays[f"{endpoint}_response"])
            baseline_covariance = factor_covariance(contrast @ arrays[f"{endpoint}_baseline"])
            source_index = int(schedules[endpoint][session_index])
            source = sources[source_index]
            start, stop = int(source_offsets[source_index]), int(source_offsets[source_index + 1])
            source_factor = contrast @ transform_patient_operator(mapped_sources[patient][:, start:stop], good, scales)
            injection_covariance, excess_covariance = stochastic_factor_injection_covariance(
                baseline_covariance, source_factor, added_ratio
            )
            realized_total = float(np.trace(injection_covariance) / np.trace(baseline_covariance))
            maximum_total_ratio_error = max(maximum_total_ratio_error, abs(realized_total - total_ratio))
            common = {
                "detector_hcp_subject": args.detector_subject,
                "source_hcp_subject": args.source_subject,
                "ccepcoreg_subject": patient,
                "stem": session["stem"],
                "endpoint": endpoint,
                "source_id": source.identifier,
                "source_family": source.family,
                "source_hemisphere": source.hemisphere,
                "source_location": source.location,
                "source_extent": source.extent,
                "source_phase_model": source.phase_model,
                "source_phase_parameter": source.phase_parameter,
                "source_rank": source_factor.shape[1],
                "event_to_baseline_total_energy_ratio": total_ratio,
                "added_signal_to_baseline_energy_ratio": added_ratio,
            }
            base, response, injected, selected = best_channel_factor_scores(
                baseline_covariance, response_covariance, injection_covariance, excess_covariance
            )
            rows.append({
                **common, "method": "best_channel", "parameter": selected,
                "baseline_score": base, "response_score": response, "injection_score": injected,
                "response_delta": response - base, "injection_delta": injected - base,
            })
            base, response, injected, selected = ica_factor_scores(
                ica.unmixing, baseline_covariance, response_covariance, injection_covariance, excess_covariance
            )
            rows.append({
                **common, "method": "fastica_bss", "parameter": selected,
                "baseline_score": base, "response_score": response, "injection_score": injected,
                "response_delta": response - base, "injection_delta": injected - base,
                "converged": ica.converged, "iterations": ica.iterations,
            })
            for method in ("minimum_norm", "eloreta_type", "lcmv"):
                parameter = float(methods[method]["parameter"])
                if method == "minimum_norm":
                    norms = np.linalg.norm(gain, axis=0)
                    variances = (norms / np.max(norms)) ** (-2.0 * float(methods[method]["depth_exponent"]))
                    operator = minimum_norm_operator(gain, baseline_covariance, parameter, variances)
                    detail: dict[str, object] = {}
                elif method == "eloreta_type":
                    operator, detail = eloreta_type_operator(
                        gain, baseline_covariance, parameter,
                        maximum_iterations=int(methods[method]["maximum_iterations"]),
                        tolerance=float(methods[method]["tolerance"]),
                    )
                else:
                    operator = lcmv_operator(gain, baseline_covariance, parameter)
                    unit_error = float(np.max(np.abs(np.diag(operator @ gain) - 1.0)))
                    noise_power = np.einsum("is,st,it->i", operator, baseline_covariance, operator, optimize=True)
                    operator = operator / np.sqrt(np.maximum(noise_power, np.finfo(float).tiny))[:, None]
                    detail = {"maximum_unit_gain_error": unit_error}
                base = inverse_score(operator, baseline_covariance, hippocampal_indices, cortical_indices)
                response = inverse_score(operator, response_covariance, hippocampal_indices, cortical_indices)
                injected = inverse_score(operator, injection_covariance, hippocampal_indices, cortical_indices)
                rows.append({
                    **common, "method": method, "parameter": parameter,
                    "baseline_score": base, "response_score": response, "injection_score": injected,
                    "response_delta": response - base, "injection_delta": injected - base, **detail,
                })
            normalized = gain / np.linalg.norm(gain, axis=0)[None, :]
            for method in ("sparse_l1", "hierarchical_sparse_group"):
                reports = []
                values = []
                for covariance in (baseline_covariance, response_covariance, injection_covariance):
                    value, detail = sparse_score(
                        method, normalized, covariance, hippocampal_indices, cortical_indices, groups, methods[method]
                    )
                    values.append(value)
                    reports.append(detail)
                base, response, injected = values
                rows.append({
                    **common, "method": method, "parameter": float(methods[method]["penalty_fraction"]),
                    "baseline_score": base, "response_score": response, "injection_score": injected,
                    "response_delta": response - base, "injection_delta": injected - base,
                    "converged": all(bool(value["converged"]) for value in reports),
                    "maximum_stationarity": max(float(value["proximal_gradient_stationarity"]) for value in reports),
                })
        if (session_index + 1) % 10 == 0 or session_index + 1 == len(cache_report["sessions"]):
            print(json.dumps({"detector": args.detector_subject, "source": args.source_subject, "sessions": session_index + 1}), flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    score_path = args.output / "method_scores.csv"
    write_csv(score_path, rows)
    score_fields = ("baseline_score", "response_score", "injection_score", "response_delta", "injection_delta")
    nonfinite = sum(not all(np.isfinite(float(row[field])) for field in score_fields) for row in rows)
    all_ica_converged = all(bool(row.get("converged", True)) for row in rows if row["method"] == "fastica_bss")
    sparse_converged_fraction = float(
        np.mean([bool(row.get("converged", False)) for row in rows if row["method"] in {"sparse_l1", "hierarchical_sparse_group"}])
    )
    maximum_sparse_stationarity = max(float(row.get("maximum_stationarity", 0.0)) for row in rows)
    schedule_counts = {
        endpoint: {
            "minimum": int(np.min(np.bincount(schedule, minlength=len(sources)))),
            "maximum": int(np.max(np.bincount(schedule, minlength=len(sources)))),
        }
        for endpoint, schedule in schedules.items()
    }
    report = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "ok": bool(
            len(rows) == len(cache_report["sessions"]) * len(ENDPOINTS) * len(METHODS)
            and nonfinite == 0
            and args.detector_subject != args.source_subject
            and maximum_total_ratio_error <= 1e-10
            and all(value["maximum"] - value["minimum"] <= 1 for value in schedule_counts.values())
            and all_ica_converged
            and sparse_converged_fraction == 1.0
        ),
        "detector_hcp_subject": args.detector_subject,
        "source_hcp_subject": args.source_subject,
        "anatomies_mismatched": args.detector_subject != args.source_subject,
        "session_count": len(cache_report["sessions"]),
        "patient_count": len({str(row["subject"]) for row in cache_report["sessions"]}),
        "methods": list(METHODS),
        "endpoints": list(ENDPOINTS),
        "source_ensemble": source_report,
        "source_schedule_counts": schedule_counts,
        "score_rows": len(rows),
        "nonfinite_score_rows": nonfinite,
        "event_to_baseline_total_energy_ratio": total_ratio,
        "added_signal_to_baseline_energy_ratio": added_ratio,
        "maximum_total_energy_ratio_error": maximum_total_ratio_error,
        "calibration_report_sha256": sha256(args.calibration_report),
        "all_ica_converged": all_ica_converged,
        "sparse_converged_fraction": sparse_converged_fraction,
        "maximum_sparse_stationarity": maximum_sparse_stationarity,
        "scores": {"path": score_path.name, "bytes": score_path.stat().st_size, "sha256": sha256(score_path)},
        "shared_storage_touched": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
