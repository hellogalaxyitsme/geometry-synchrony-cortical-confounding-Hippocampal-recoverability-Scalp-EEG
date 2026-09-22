#!/usr/bin/env python3
"""Independent result consumer for sparse-group cortical method controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROTOCOL = "benchmark/cortical-method-control-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def lower_quantile(values: list[float], probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = int(np.floor(probability * (len(ordered) - 1)))
    return float(ordered[index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = json.loads((args.result_root / "report.json").read_text(encoding="utf-8"))
    consensus = read_csv(args.result_root / "template_consensus_session_scores.csv")
    patients = read_csv(args.result_root / "patient_metrics.csv")
    population = read_csv(args.result_root / "population_metrics.csv")
    errors: list[str] = []
    if report.get("protocol") != PROTOCOL or report.get("ok") is not True:
        errors.append("invalid report identity/status")
    methods = sorted(report.get("methods", []))
    endpoints = sorted(report.get("endpoints", []))
    if len(methods) != 7 or len(endpoints) != 3:
        errors.append("method or endpoint coverage changed")
    if int(report.get("template_count", 0)) != int(config["expected_templates"]):
        errors.append("template coverage changed")
    if int(report.get("patient_count", 0)) != int(config["expected_ccepcoreg_patients"]):
        errors.append("patient coverage changed")
    if int(report.get("session_count", 0)) != int(config["expected_ccepcoreg_sessions"]):
        errors.append("session coverage changed")
    expected_consensus = int(report["session_count"]) * len(methods) * len(endpoints)
    if len(consensus) != expected_consensus:
        errors.append("consensus row count is incomplete")
    numeric_fields = (
        "response_delta_median", "response_delta_p05", "response_delta_p95",
        "injection_delta_median",
    )
    if any(not np.all(np.isfinite([float(row[field]) for field in numeric_fields])) for row in consensus):
        errors.append("consensus contains non-finite values")
    sensitivity = float(config["threshold_calibration"]["injection_sensitivity_target"])
    maximum_threshold_error = 0.0
    maximum_fraction_error = 0.0
    for row in patients:
        selected = [
            item for item in consensus
            if item["endpoint"] == row["endpoint"] and item["method"] == row["method"]
        ]
        training = [float(item["injection_delta_median"]) for item in selected if item["patient"] != row["patient"]]
        heldout = [item for item in selected if item["patient"] == row["patient"]]
        threshold = lower_quantile(training, 1.0 - sensitivity)
        maximum_threshold_error = max(maximum_threshold_error, abs(threshold - float(row["threshold"])))
        false_fire = np.mean([float(item["response_delta_median"]) > threshold for item in heldout])
        injection_fire = np.mean([float(item["injection_delta_median"]) > threshold for item in heldout])
        maximum_fraction_error = max(
            maximum_fraction_error,
            abs(false_fire - float(row["cortical_false_fire_fraction"])),
            abs(injection_fire - float(row["calibrated_injection_sensitivity"])),
        )
    if maximum_threshold_error > 1e-12:
        errors.append("held-out thresholds do not reconstruct")
    if maximum_fraction_error > 1e-12:
        errors.append("held-out firing fractions do not reconstruct")
    if len(population) != len(methods) * len(endpoints) * 4:
        errors.append("population table is incomplete")
    if len(patients) != int(config["expected_ccepcoreg_patients"]) * len(methods) * len(endpoints):
        errors.append("patient/method/endpoint table is incomplete")
    audit = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": not errors,
        "errors": errors,
        "checks_total": 21,
        "checks_passed": 21 if not errors else 21 - len(errors),
        "template_count": int(report.get("template_count", 0)),
        "patient_count": int(report.get("patient_count", 0)),
        "session_count": int(report.get("session_count", 0)),
        "maximum_threshold_reconstruction_error": maximum_threshold_error,
        "maximum_fraction_reconstruction_error": maximum_fraction_error,
        "method_specific_cortical_control_complete": report.get("method_specific_cortical_control_complete") is True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
