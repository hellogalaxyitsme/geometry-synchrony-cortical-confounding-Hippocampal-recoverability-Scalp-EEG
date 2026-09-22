#!/usr/bin/env python3
"""Independent, dynamically counted audit for injection-benchmark."""

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
    patient_equal_injection_threshold,
    patient_equal_sensitivity,
)


PROTOCOL = "benchmark/cortical-method-control-v2"
METHODS = {
    "best_channel", "fastica_bss", "minimum_norm", "eloreta_type", "lcmv",
    "sparse_l1", "hierarchical_sparse_group",
}
ENDPOINTS = {"n1_peak", "n2_peak", "integrated_pca3"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = json.loads((args.result_root / "report.json").read_text(encoding="utf-8"))
    consensus = read_csv(args.result_root / "template_consensus_session_scores.csv")
    patients = read_csv(args.result_root / "patient_metrics.csv")
    population = read_csv(args.result_root / "population_metrics.csv")
    source_sensitivity = read_csv(args.result_root / "source_sensitivity.csv")
    cohort = [
        value.strip() for value in (PROJECT / str(config["hcp_subjects_file"])).read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    expected_pairs = anatomy_derangement(cohort, int(config["cross_anatomy"]["cyclic_offset"]))
    pair_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.pair_root.glob("*/report.json"))
        if not path.parent.name.startswith(".")
    ]
    source_ids = [str(row["id"]) for row in config["source_ensemble"]["sources"]]
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object = None) -> None:
        checks.append({"name": name, "ok": bool(condition), "detail": detail})

    check("report_identity", report.get("protocol") == PROTOCOL and report.get("ok") is True)
    check("method_coverage", set(report.get("methods", [])) == METHODS)
    check("endpoint_coverage", set(report.get("endpoints", [])) == ENDPOINTS)
    check("pair_count", len(pair_reports) == len(cohort), len(pair_reports))
    realized_pairs = {str(row.get("detector_hcp_subject")): str(row.get("source_hcp_subject")) for row in pair_reports}
    check("exact_frozen_derangement", realized_pairs == expected_pairs)
    check("no_matched_anatomies", all(key != value for key, value in realized_pairs.items()))
    check("detector_cohort_complete", set(realized_pairs) == set(cohort))
    check("source_cohort_complete", set(realized_pairs.values()) == set(cohort))
    check("all_pair_reports_passed", all(row.get("ok") is True for row in pair_reports))
    check("all_ica_converged", all(row.get("all_ica_converged") is True for row in pair_reports))
    check("all_sparse_converged", all(float(row.get("sparse_converged_fraction", 0.0)) == 1.0 for row in pair_reports))
    check("all_pair_scores_finite", all(int(row.get("nonfinite_score_rows", -1)) == 0 for row in pair_reports))
    maximum_energy_error = max((float(row.get("maximum_total_energy_ratio_error", np.inf)) for row in pair_reports), default=np.inf)
    check("injection_energy_identity", maximum_energy_error <= 1e-10, maximum_energy_error)
    total_ratio = float(report.get("event_to_baseline_total_energy_ratio", np.nan))
    added_ratio = float(report.get("added_signal_to_baseline_energy_ratio", np.nan))
    check(
        "explicit_total_to_added_conversion",
        np.isfinite(total_ratio) and np.isclose(added_ratio, added_signal_ratio_from_total_energy_ratio(total_ratio), rtol=0.0, atol=1e-15),
        {"total": total_ratio, "added": added_ratio},
    )
    check("source_ids_exact", report.get("source_ids") == source_ids)
    check("source_count_exact", int(report.get("source_count", 0)) == len(source_ids))
    expected_consensus = int(config["expected_ccepcoreg_sessions"]) * len(METHODS) * len(ENDPOINTS)
    check("consensus_row_count", len(consensus) == expected_consensus, len(consensus))
    numeric_fields = (
        "response_delta_median", "response_delta_p05", "response_delta_p95",
        "injection_delta_median", "injection_delta_p05", "injection_delta_p95",
    )
    finite = all(np.all(np.isfinite([float(row[field]) for field in numeric_fields])) for row in consensus)
    check("consensus_finite", finite)
    ordered_intervals = all(
        float(row["response_delta_p05"]) <= float(row["response_delta_median"]) <= float(row["response_delta_p95"])
        and float(row["injection_delta_p05"]) <= float(row["injection_delta_median"]) <= float(row["injection_delta_p95"])
        for row in consensus
    )
    check("anatomy_intervals_ordered", ordered_intervals)
    source_assignments = [row for row in consensus if row["method"] == "minimum_norm"]
    balanced = True
    for endpoint in ENDPOINTS:
        counts = [sum(row["endpoint"] == endpoint and row["source_id"] == source for row in source_assignments) for source in source_ids]
        balanced &= max(counts) - min(counts) <= 1
    check("source_schedule_balanced", balanced)
    expected_patient_rows = int(config["expected_ccepcoreg_patients"]) * len(METHODS) * len(ENDPOINTS)
    check("patient_metric_row_count", len(patients) == expected_patient_rows, len(patients))
    target = float(config["threshold_calibration"]["injection_sensitivity_target"])
    maximum_threshold_error = 0.0
    maximum_fraction_error = 0.0
    minimum_training_sensitivity = 1.0
    for row in patients:
        selected = [item for item in consensus if item["endpoint"] == row["endpoint"] and item["method"] == row["method"]]
        training = [item for item in selected if item["patient"] != row["patient"]]
        heldout = [item for item in selected if item["patient"] == row["patient"]]
        scores = np.asarray([float(item["injection_delta_median"]) for item in training])
        labels = np.asarray([str(item["patient"]) for item in training])
        threshold = patient_equal_injection_threshold(scores, labels, target)
        realized_training = patient_equal_sensitivity(scores, labels, threshold)
        minimum_training_sensitivity = min(minimum_training_sensitivity, realized_training)
        maximum_threshold_error = max(maximum_threshold_error, abs(threshold - float(row["threshold"])))
        false_fire = float(np.mean([float(item["response_delta_median"]) > threshold for item in heldout]))
        injection_fire = float(np.mean([float(item["injection_delta_median"]) > threshold for item in heldout]))
        maximum_fraction_error = max(
            maximum_fraction_error,
            abs(false_fire - float(row["cortical_false_fire_fraction"])),
            abs(injection_fire - float(row["calibrated_injection_sensitivity"])),
            abs(realized_training - float(row["training_patient_equal_sensitivity"])),
        )
    check("heldout_thresholds_reconstruct", maximum_threshold_error <= 1e-12, maximum_threshold_error)
    check("heldout_fractions_reconstruct", maximum_fraction_error <= 1e-12, maximum_fraction_error)
    check("training_patient_equal_target_met", minimum_training_sensitivity >= target - 1e-15, minimum_training_sensitivity)
    check("population_table_complete", len(population) == len(METHODS) * len(ENDPOINTS) * 4, len(population))
    check("source_sensitivity_complete", len(source_sensitivity) == len(METHODS) * len(ENDPOINTS) * len(source_ids), len(source_sensitivity))
    check("primary_report_has_all_methods", set(report.get("primary_integrated_pca3_medians", {})) == METHODS)
    required_metrics = {
        "cortical_false_fire_fraction", "calibrated_injection_sensitivity",
        "median_cortical_response_delta", "median_injection_delta",
    }
    check(
        "primary_report_has_all_metrics",
        all(set(value) == required_metrics for value in report.get("primary_integrated_pca3_medians", {}).values()),
    )
    hash_errors = []
    for name, record in report.get("outputs", {}).items():
        path = args.result_root / str(name)
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            hash_errors.append(name)
    check("output_hashes_verify", not hash_errors, hash_errors)
    check("physiological_inference_not_authorized", report.get("physiological_hippocampal_recovery_inference_authorized") is False)
    errors = [str(row["name"]) for row in checks if not row["ok"]]
    audit = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "ok": not errors,
        "errors": errors,
        "checks_total": len(checks),
        "checks_passed": sum(bool(row["ok"]) for row in checks),
        "checks": checks,
        "maximum_threshold_reconstruction_error": maximum_threshold_error,
        "maximum_fraction_reconstruction_error": maximum_fraction_error,
        "maximum_total_energy_ratio_error": maximum_energy_error,
        "minimum_training_patient_equal_sensitivity": minimum_training_sensitivity,
        "pair_count": len(pair_reports),
        "patient_count": int(report.get("patient_count", 0)),
        "session_count": int(report.get("session_count", 0)),
        "source_count": int(report.get("source_count", 0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
