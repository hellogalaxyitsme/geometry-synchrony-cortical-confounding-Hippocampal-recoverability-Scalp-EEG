#!/usr/bin/env python3
"""Run all seven method-benchmark methods on stimulation-control cortical responses for one HCP template."""

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
from empirical.stimulation_control import (  # noqa: E402
    interpolate_rows,
    spherical_interpolation,
    transform_patient_operator,
)
from empirical.estimators import (  # noqa: E402
    anatomical_log_power_ratio,
    covariance_factors,
    eloreta_type_operator,
    fit_fastica,
    lcmv_operator,
    minimum_norm_operator,
    sparse_group_fista,
)
from empirical.source_scores import (  # noqa: E402
    best_channel_anatomical_scores,
    factor_covariance,
    ica_anatomical_scores,
    stable_anatomical_log_power_ratio,
    stochastic_injection_covariance,
)
from theory.recoverability import helmert_reference  # noqa: E402


PROTOCOL = "benchmark/cortical-method-control-v1"
METHODS = (
    "best_channel",
    "fastica_bss",
    "minimum_norm",
    "eloreta_type",
    "lcmv",
    "sparse_l1",
    "hierarchical_sparse_group",
)
ENDPOINTS = ("n1_peak", "n2_peak", "integrated_pca3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
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
            design,
            target,
            penalty,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected sparse-group protocol")
    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    if calibration.get("protocol") != "calibration/continuous-false-positive-v1.1" or calibration.get("ok") is not True:
        raise ValueError("invalid empirical calibration report")
    if config.get("calibration_report_sha256") and sha256(args.calibration_report) != config["calibration_report_sha256"]:
        raise ValueError("empirical calibration report hash changed")
    energy_ratio = float(
        calibration["mesial_event_physical_calibration"]["injection_calibration_energy_ratio"]
    )
    cache_report = json.loads((args.cache_root / "cache_report.json").read_text(encoding="utf-8"))
    with np.load(args.bundle, allow_pickle=False) as archive:
        bundle = {name: np.asarray(archive[name]) for name in archive.files}
    source_positions = np.asarray(bundle["montage_positions_head_m"], dtype=np.float64)
    mapped_by_patient: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    methods = config["methods"]
    for session_index, session in enumerate(cache_report["sessions"], start=1):
        patient = str(session["subject"])
        if patient not in mapped_by_patient:
            with np.load(args.cache_root / "montages" / f"{patient}.npz", allow_pickle=False) as montage:
                patient_positions = np.asarray(montage["coordinates_head_m"], dtype=np.float64)
            interpolation = spherical_interpolation(
                source_positions,
                patient_positions,
                int(config["interpolation"]["neighbours"]),
                float(config["interpolation"]["sigma_degrees"]),
            )
            h_shape = bundle["hippocampal_factors"].shape
            mapped_h = interpolate_rows(
                np.asarray(bundle["hippocampal_factors"], dtype=np.float64)
                .transpose(1, 0, 2)
                .reshape(h_shape[1], -1),
                interpolation,
            ).reshape(len(patient_positions), h_shape[0], h_shape[2]).transpose(1, 0, 2)
            mapped_by_patient[patient] = {
                "smooth": interpolate_rows(np.asarray(bundle["smooth_topographies"], dtype=np.float64), interpolation),
                "hippocampus": mapped_h,
            }
        cache_path = args.cache_root / str(session["cache"])
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_arrays = {name: np.asarray(cache[name], dtype=np.float64) for name in cache.files}
        good = np.asarray(cache_arrays["good_indices"], dtype=np.int64)
        scales = np.asarray(cache_arrays["robust_scales"], dtype=np.float64)
        contrast = np.asarray(helmert_reference(len(good)), dtype=np.float64)
        smooth = contrast @ transform_patient_operator(mapped_by_patient[patient]["smooth"], good, scales)
        h_columns: list[np.ndarray] = []
        h_groups: list[int] = []
        for factor_index, rank in enumerate(np.asarray(bundle["factor_ranks"], dtype=np.int64)):
            transformed = contrast @ transform_patient_operator(
                mapped_by_patient[patient]["hippocampus"][factor_index, :, : int(rank)], good, scales
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
        injection_factor = int(bundle["injection_factor_index"])
        preceding = int(np.sum(np.asarray(bundle["factor_ranks"], dtype=np.int64)[:injection_factor]))
        injection_topography = hippocampal[:, preceding]
        endpoint_samples = []
        for endpoint in ENDPOINTS:
            endpoint_samples.append(contrast @ cache_arrays[f"{endpoint}_response"])
            endpoint_samples.append(contrast @ cache_arrays[f"{endpoint}_baseline"])
        ica_samples = np.column_stack(endpoint_samples).T
        components = min(int(methods["fastica_bss"]["components"]), ica_samples.shape[0] - 1, ica_samples.shape[1])
        ica = fit_fastica(
            ica_samples,
            components,
            seed=int(config["master_seed"]) + session_index,
            maximum_iterations=int(methods["fastica_bss"]["maximum_iterations"]),
            tolerance=float(methods["fastica_bss"]["tolerance"]),
        )
        for endpoint in ENDPOINTS:
            response_factors = contrast @ cache_arrays[f"{endpoint}_response"]
            baseline_factors = contrast @ cache_arrays[f"{endpoint}_baseline"]
            response_covariance = factor_covariance(response_factors)
            baseline_covariance = factor_covariance(baseline_factors)
            injection_covariance = stochastic_injection_covariance(
                baseline_covariance, injection_topography, energy_ratio
            )
            base, response, injected, selected = best_channel_anatomical_scores(
                baseline_covariance, response_covariance, injection_covariance, injection_topography
            )
            rows.append({
                "hcp_subject": args.subject, "ccepcoreg_subject": patient, "stem": session["stem"],
                "endpoint": endpoint, "method": "best_channel", "parameter": selected,
                "baseline_score": base, "response_score": response, "injection_score": injected,
                "response_delta": response - base, "injection_delta": injected - base,
                "injection_energy_ratio": energy_ratio,
            })
            base, response, injected, selected = ica_anatomical_scores(
                ica, baseline_covariance, response_covariance, injection_covariance, injection_topography
            )
            rows.append({
                "hcp_subject": args.subject, "ccepcoreg_subject": patient, "stem": session["stem"],
                "endpoint": endpoint, "method": "fastica_bss", "parameter": selected,
                "baseline_score": base, "response_score": response, "injection_score": injected,
                "response_delta": response - base, "injection_delta": injected - base,
                "injection_energy_ratio": energy_ratio, "converged": ica.converged,
                "iterations": ica.iterations,
            })
            training_covariance = baseline_covariance
            for method in ("minimum_norm", "eloreta_type", "lcmv"):
                parameter = float(methods[method]["parameter"])
                if method == "minimum_norm":
                    norms = np.linalg.norm(gain, axis=0)
                    variances = (norms / np.max(norms)) ** (-2.0 * float(methods[method]["depth_exponent"]))
                    operator = minimum_norm_operator(gain, training_covariance, parameter, variances)
                    detail: dict[str, object] = {}
                elif method == "eloreta_type":
                    operator, detail = eloreta_type_operator(
                        gain,
                        training_covariance,
                        parameter,
                        maximum_iterations=int(methods[method]["maximum_iterations"]),
                        tolerance=float(methods[method]["tolerance"]),
                    )
                else:
                    operator = lcmv_operator(gain, training_covariance, parameter)
                    unit_error = float(np.max(np.abs(np.diag(operator @ gain) - 1.0)))
                    noise_power = np.einsum("is,st,it->i", operator, training_covariance, operator, optimize=True)
                    operator = operator / np.sqrt(np.maximum(noise_power, np.finfo(float).tiny))[:, None]
                    detail = {"maximum_unit_gain_error": unit_error}
                base = inverse_score(operator, baseline_covariance, hippocampal_indices, cortical_indices)
                response = inverse_score(operator, response_covariance, hippocampal_indices, cortical_indices)
                injected = inverse_score(operator, injection_covariance, hippocampal_indices, cortical_indices)
                rows.append({
                    "hcp_subject": args.subject, "ccepcoreg_subject": patient, "stem": session["stem"],
                    "endpoint": endpoint, "method": method, "parameter": parameter,
                    "baseline_score": base, "response_score": response, "injection_score": injected,
                    "response_delta": response - base, "injection_delta": injected - base,
                    "injection_energy_ratio": energy_ratio, **detail,
                })
            normalized = gain / np.linalg.norm(gain, axis=0)[None, :]
            for method in ("sparse_l1", "hierarchical_sparse_group"):
                method_reports = []
                values = []
                for covariance in (baseline_covariance, response_covariance, injection_covariance):
                    value, detail = sparse_score(
                        method, normalized, covariance, hippocampal_indices, cortical_indices,
                        groups, methods[method],
                    )
                    values.append(value)
                    method_reports.append(detail)
                base, response, injected = values
                rows.append({
                    "hcp_subject": args.subject, "ccepcoreg_subject": patient, "stem": session["stem"],
                    "endpoint": endpoint, "method": method, "parameter": float(methods[method]["penalty_fraction"]),
                    "baseline_score": base, "response_score": response, "injection_score": injected,
                    "response_delta": response - base, "injection_delta": injected - base,
                    "injection_energy_ratio": energy_ratio,
                    "converged": all(bool(value["converged"]) for value in method_reports),
                    "maximum_stationarity": max(float(value["proximal_gradient_stationarity"]) for value in method_reports),
                })
        if session_index % 10 == 0 or session_index == len(cache_report["sessions"]):
            print(json.dumps({"hcp_subject": args.subject, "sessions": session_index, "total": len(cache_report["sessions"])}), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    score_path = args.output / "method_scores.csv"
    write_csv(score_path, rows)
    score_fields = ("baseline_score", "response_score", "injection_score", "response_delta", "injection_delta")
    nonfinite_rows = sum(
        not all(np.isfinite(float(row[field])) for field in score_fields)
        for row in rows
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": (
            len(rows) == len(cache_report["sessions"]) * len(ENDPOINTS) * len(METHODS)
            and nonfinite_rows == 0
        ),
        "hcp_subject": args.subject,
        "session_count": len(cache_report["sessions"]),
        "patient_count": len({str(row["subject"]) for row in cache_report["sessions"]}),
        "methods": list(METHODS),
        "endpoints": list(ENDPOINTS),
        "score_rows": len(rows),
        "nonfinite_score_rows": nonfinite_rows,
        "injection_energy_ratio": energy_ratio,
        "calibration_report_sha256": sha256(args.calibration_report),
        "all_ica_converged": all(bool(row.get("converged", True)) for row in rows if row["method"] == "fastica_bss"),
        "sparse_converged_fraction": float(np.mean([bool(row.get("converged", False)) for row in rows if row["method"] in {"sparse_l1", "hierarchical_sparse_group"}])),
        "maximum_sparse_stationarity": max(float(row.get("maximum_stationarity", 0.0)) for row in rows),
        "scores": {"path": score_path.name, "bytes": score_path.stat().st_size, "sha256": sha256(score_path)},
        "shared_storage_touched": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
