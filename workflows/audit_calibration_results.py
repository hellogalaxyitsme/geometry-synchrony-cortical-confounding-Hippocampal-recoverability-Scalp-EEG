#!/usr/bin/env python3
"""Independent consumer for empirical calibration result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROTOCOL = "calibration/continuous-false-positive-v1.1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.result_root / "report.json").read_text(encoding="utf-8"))
    patients = read_csv(args.result_root / "continuous_patient_metrics.csv")
    events = read_csv(args.result_root / "continuous_event_scores.csv")
    backgrounds = read_csv(args.result_root / "continuous_background_scores.csv")
    mesial = read_csv(args.result_root / "mesial_event_physical_calibration.csv")
    physical = read_csv(args.result_root / "late_event_physical_calibration.csv")
    errors: list[str] = []
    if report.get("protocol") != PROTOCOL or report.get("ok") is not True:
        errors.append("invalid report identity/status")
    patient_ids = {row["patient"] for row in patients}
    if len(patient_ids) != 13 or len(patients) != 26:
        errors.append("patient/method grid is incomplete")
    excluded = report.get("excluded_task_control_patients")
    if not isinstance(excluded, list) or len(excluded) != 1 or excluded[0] in patient_ids:
        errors.append("outcome-blind physical-QC patient exclusion changed")
    if {row["method"] for row in patients} != {"shrinkage_lda", "best_channel"}:
        errors.append("method set changed")
    maximum_rate_error = 0.0
    maximum_event_fraction_error = 0.0
    for row in patients:
        selected_background = [
            item
            for item in backgrounds
            if item["patient"] == row["patient"] and item["method"] == row["method"]
        ]
        selected_events = [
            item
            for item in events
            if item["patient"] == row["patient"] and item["method"] == row["method"]
        ]
        if not selected_background or not selected_events:
            errors.append(f"missing score rows for {row['patient']} {row['method']}")
            continue
        threshold = float(row["threshold"])
        if any(abs(float(item["threshold"]) - threshold) > 1e-12 for item in selected_background + selected_events):
            errors.append(f"threshold inconsistency for {row['patient']} {row['method']}")
        false_count = sum(float(item["score"]) > threshold for item in selected_background)
        hours = sum(float(item["window_seconds"]) for item in selected_background) / 3600.0
        rate = false_count / hours
        maximum_rate_error = max(maximum_rate_error, abs(rate - float(row["heldout_false_alarms_per_hour"])))
        for label, field in ((1, "late_high_theta_sensitivity"), (0, "late_low_theta_false_positive_fraction")):
            subset = [item for item in selected_events if int(item["label"]) == label]
            fraction = np.mean([float(item["score"]) > threshold for item in subset])
            maximum_event_fraction_error = max(maximum_event_fraction_error, abs(fraction - float(row[field])))
    if maximum_rate_error > 1e-10:
        errors.append("false-alarm rates do not reconstruct")
    if maximum_event_fraction_error > 1e-12:
        errors.append("event threshold fractions do not reconstruct")
    if len({row["patient"] for row in mesial}) != 7 or len({row["network"] for row in mesial}) != 9:
        errors.append("mesial calibration coverage is incomplete")
    if len({row["patient"] for row in physical}) != 14:
        errors.append("ds004752 physical calibration coverage is incomplete")
    if any(float(row["event_peak_uv"]) < 0.0 or float(row["background_sd_uv"]) <= 0.0 for row in mesial):
        errors.append("invalid physical mesial scale")
    if any(float(row["scalp_rms_uv"]) <= 0.0 for row in physical):
        errors.append("invalid physical ds004752 scale")
    audit = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": not errors,
        "errors": errors,
        "checks_total": 12,
        "checks_passed": 12 if not errors else 12 - len(errors),
        "patient_method_rows": len(patients),
        "event_score_rows": len(events),
        "background_score_rows": len(backgrounds),
        "maximum_false_alarm_rate_reconstruction_error": maximum_rate_error,
        "maximum_event_fraction_reconstruction_error": maximum_event_fraction_error,
        "depth_waveform_used": False,
        "source_current_calibration_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
