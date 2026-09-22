#!/usr/bin/env python3
"""Aggregate corrected cross-anatomy injection-benchmark results."""

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
from empirical.injection_benchmark import (  # noqa: E402
    added_signal_ratio_from_total_energy_ratio,
    anatomy_derangement,
    complete_primary_method_metrics,
    patient_equal_injection_threshold,
    patient_equal_sensitivity,
)


PROTOCOL = "benchmark/cortical-method-control-v2"
METHODS = (
    "best_channel", "fastica_bss", "minimum_norm", "eloreta_type", "lcmv",
    "sparse_l1", "hierarchical_sparse_group",
)
ENDPOINTS = ("n1_peak", "n2_peak", "integrated_pca3")
METRICS = (
    "cortical_false_fire_fraction", "calibrated_injection_sensitivity",
    "median_cortical_response_delta", "median_injection_delta",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_median(values: list[float], replicates: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = np.median(array[generator.integers(0, len(array), size=(replicates, len(array)))], axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected injection-benchmark protocol")
    subjects = [
        value.strip() for value in (PROJECT / str(config["hcp_subjects_file"])).read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    expected_pairs = anatomy_derangement(subjects, int(config["cross_anatomy"]["cyclic_offset"]))
    pair_dirs = sorted(
        path for path in args.pair_root.iterdir()
        if not path.name.startswith(".") and path.is_dir() and (path / "report.json").is_file()
    )
    if len(pair_dirs) != len(expected_pairs):
        raise ValueError(f"expected {len(expected_pairs)} complete pairs, found {len(pair_dirs)}")

    reports = []
    raw: dict[tuple[str, str, str, str], dict[str, object]] = {}
    source_fields = (
        "source_id", "source_family", "source_hemisphere", "source_location",
        "source_extent", "source_phase_model", "source_phase_parameter", "source_rank",
    )
    for directory in pair_dirs:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        detector = str(report.get("detector_hcp_subject"))
        if (
            report.get("ok") is not True
            or report.get("protocol") != PROTOCOL
            or expected_pairs.get(detector) != report.get("source_hcp_subject")
            or report.get("anatomies_mismatched") is not True
            or int(report.get("nonfinite_score_rows", -1)) != 0
            or report.get("all_ica_converged") is not True
            or float(report.get("sparse_converged_fraction", 0.0)) != 1.0
        ):
            raise ValueError(f"invalid v2 pair report: {directory}")
        reports.append(report)
        for row in read_csv(directory / "method_scores.csv"):
            key = (row["ccepcoreg_subject"], row["stem"], row["endpoint"], row["method"])
            numeric = (float(row["response_delta"]), float(row["injection_delta"]))
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"non-finite pair score: {directory} {key}")
            target = raw.setdefault(
                key,
                {
                    "response": [], "injection": [],
                    "source": {field: row[field] for field in source_fields},
                },
            )
            if target["source"] != {field: row[field] for field in source_fields}:
                raise ValueError(f"source schedule differs across anatomy pairs: {key}")
            target["response"].append(numeric[0])
            target["injection"].append(numeric[1])

    consensus: list[dict[str, object]] = []
    for (patient, stem, endpoint, method), values in sorted(raw.items()):
        response = np.asarray(values["response"], dtype=np.float64)
        injection = np.asarray(values["injection"], dtype=np.float64)
        if len(response) != len(expected_pairs) or len(injection) != len(expected_pairs):
            raise ValueError(f"incomplete cross-anatomy consensus: {patient} {stem} {endpoint} {method}")
        consensus.append({
            "patient": patient, "stem": stem, "endpoint": endpoint, "method": method,
            **values["source"],
            "anatomy_pair_count": len(expected_pairs),
            "response_delta_median": float(np.median(response)),
            "response_delta_p05": float(np.quantile(response, 0.05)),
            "response_delta_p95": float(np.quantile(response, 0.95)),
            "injection_delta_median": float(np.median(injection)),
            "injection_delta_p05": float(np.quantile(injection, 0.05)),
            "injection_delta_p95": float(np.quantile(injection, 0.95)),
        })
    patients = sorted({str(row["patient"]) for row in consensus})
    sensitivity_target = float(config["threshold_calibration"]["injection_sensitivity_target"])
    patient_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for method in METHODS:
            selected = [row for row in consensus if row["endpoint"] == endpoint and row["method"] == method]
            for heldout in patients:
                training = [row for row in selected if row["patient"] != heldout]
                testing = [row for row in selected if row["patient"] == heldout]
                training_scores = np.asarray([float(row["injection_delta_median"]) for row in training])
                training_patients = np.asarray([str(row["patient"]) for row in training])
                threshold = patient_equal_injection_threshold(training_scores, training_patients, sensitivity_target)
                training_sensitivity = patient_equal_sensitivity(training_scores, training_patients, threshold)
                response_fire = [float(row["response_delta_median"]) > threshold for row in testing]
                injection_fire = [float(row["injection_delta_median"]) > threshold for row in testing]
                patient_rows.append({
                    "patient": heldout, "endpoint": endpoint, "method": method,
                    "threshold": threshold,
                    "training_patients": len(set(training_patients)),
                    "training_injection_sessions": len(training),
                    "training_patient_equal_sensitivity": training_sensitivity,
                    "heldout_sessions": len(testing),
                    "cortical_false_fire_fraction": float(np.mean(response_fire)),
                    "calibrated_injection_sensitivity": float(np.mean(injection_fire)),
                    "median_cortical_response_delta": float(np.median([float(row["response_delta_median"]) for row in testing])),
                    "median_injection_delta": float(np.median([float(row["injection_delta_median"]) for row in testing])),
                })
                for row, cortical_fire, positive_fire in zip(testing, response_fire, injection_fire):
                    decision_rows.append({
                        "patient": heldout, "stem": row["stem"], "endpoint": endpoint, "method": method,
                        **{field: row[field] for field in source_fields},
                        "threshold": threshold,
                        "cortical_false_fire": int(cortical_fire),
                        "injection_detected": int(positive_fire),
                        "injection_delta_median": row["injection_delta_median"],
                        "injection_anatomy_p90_width": float(row["injection_delta_p95"]) - float(row["injection_delta_p05"]),
                    })

    replicates = int(config["summary"]["bootstrap_replicates"])
    seed = int(config["summary"]["seed"])
    population_rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for method_index, method in enumerate(METHODS):
            selected = [row for row in patient_rows if row["endpoint"] == endpoint and row["method"] == method]
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                low, high = bootstrap_median(values, replicates, seed + method_index + 100 * ENDPOINTS.index(endpoint))
                population_rows.append({
                    "endpoint": endpoint, "method": method, "metric": metric,
                    "patients": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
                    "minimum": float(np.min(values)), "maximum": float(np.max(values)),
                    "bootstrap_median_ci95_low": low, "bootstrap_median_ci95_high": high,
                })

    source_rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for method in METHODS:
            for source_id in [str(row["id"]) for row in config["source_ensemble"]["sources"]]:
                selected = [
                    row for row in decision_rows
                    if row["endpoint"] == endpoint and row["method"] == method and row["source_id"] == source_id
                ]
                if not selected:
                    raise ValueError(f"source has no held-out decisions: {endpoint} {method} {source_id}")
                source_rows.append({
                    "endpoint": endpoint, "method": method,
                    **{field: selected[0][field] for field in source_fields},
                    "sessions": len(selected),
                    "patients": len({str(row["patient"]) for row in selected}),
                    "heldout_injection_sensitivity": float(np.mean([int(row["injection_detected"]) for row in selected])),
                    "median_injection_delta": float(np.median([float(row["injection_delta_median"]) for row in selected])),
                    "median_injection_anatomy_p90_width": float(np.median([float(row["injection_anatomy_p90_width"]) for row in selected])),
                })

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "template_consensus_session_scores.csv", consensus)
    write_csv(args.output / "patient_metrics.csv", patient_rows)
    write_csv(args.output / "population_metrics.csv", population_rows)
    write_csv(args.output / "source_sensitivity.csv", source_rows)
    primary = complete_primary_method_metrics(population_rows, METHODS, "integrated_pca3", METRICS)
    total_ratio = float(reports[0]["event_to_baseline_total_energy_ratio"])
    added_ratio = float(reports[0]["added_signal_to_baseline_energy_ratio"])
    source_ids = [str(row["id"]) for row in config["source_ensemble"]["sources"]]
    report = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "ok": bool(
            len(pair_dirs) == len(subjects)
            and len(patients) == int(config["expected_ccepcoreg_patients"])
            and len({row["stem"] for row in consensus}) == int(config["expected_ccepcoreg_sessions"])
            and set(primary) == set(METHODS)
            and np.isclose(added_ratio, added_signal_ratio_from_total_energy_ratio(total_ratio), rtol=0.0, atol=1e-15)
        ),
        "pair_count": len(pair_dirs),
        "detector_subject_count": len({str(report["detector_hcp_subject"]) for report in reports}),
        "source_subject_count": len({str(report["source_hcp_subject"]) for report in reports}),
        "all_anatomies_mismatched": all(bool(report["anatomies_mismatched"]) for report in reports),
        "patient_count": len(patients),
        "session_count": len({row["stem"] for row in consensus}),
        "methods": list(METHODS), "endpoints": list(ENDPOINTS),
        "source_count": len(source_ids), "source_ids": source_ids,
        "threshold_calibration": f"leave-one-patient-out, patient-equal training sensitivity >= {sensitivity_target:.3f}",
        "event_to_baseline_total_energy_ratio": total_ratio,
        "added_signal_to_baseline_energy_ratio": added_ratio,
        "maximum_pair_total_energy_ratio_error": max(float(report["maximum_total_energy_ratio_error"]) for report in reports),
        "primary_integrated_pca3_medians": primary,
        "patient_specific_anatomy_used": False,
        "physiological_hippocampal_recovery_inference_authorized": False,
        "outputs": {},
    }
    for path in args.output.glob("*.csv"):
        report["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
