#!/usr/bin/env python3
"""Independent consumer audit for mesial-event."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--epoch-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True); parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--expected-templates", type=int, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); checks = []; errors = []
    try:
        manifest = json.loads((args.epoch_root / "manifest.json").read_text(encoding="utf-8"))
        checks.extend([manifest.get("ok") is True, manifest.get("network_count") == 9, len(manifest.get("patients", [])) == 7,
                       manifest.get("depth_waveform_stored") is False, manifest.get("depth_waveform_used_by_scalp_detector") is False,
                       manifest.get("participants_tsv_read") is False])
        retained = 0
        for row in manifest["networks"]:
            path = args.epoch_root / row["bundle"]
            checks.append(sha256(path) == row["bundle_sha256"])
            with np.load(path, allow_pickle=False) as archive:
                checks.append(set(archive.files) == {"protocol", "patient", "network", "channel_names", "eeg_epochs_uv", "retained_original_indices", "sampling_frequency_hz", "t0_sample"})
                checks.append(archive["eeg_epochs_uv"].shape[0] == int(row["retained_events"]))
                retained += int(archive["eeg_epochs_uv"].shape[0])
        checks.append(retained == int(manifest["retained_event_count"]))
        prediction_report = json.loads((args.prediction_root / "report.json").read_text(encoding="utf-8"))
        template_predictions = rows(args.prediction_root / "template_predictions.csv"); population_predictions = rows(args.prediction_root / "population_predictions.csv")
        checks.extend([prediction_report.get("ok") is True, prediction_report.get("template_count") == args.expected_templates,
                       len(template_predictions) == args.expected_templates * 18, len(population_predictions) == 18])
        analysis = json.loads((args.analysis_root / "report.json").read_text(encoding="utf-8")); network = rows(args.analysis_root / "network_metrics.csv")
        patient = rows(args.analysis_root / "patient_metrics.csv"); averaging = rows(args.analysis_root / "averaging_ladder.csv")
        checks.extend([analysis.get("ok") is True, len(network) == 9, len(patient) == 7, len(averaging) == 54,
                       analysis.get("depth_waveform_used") is False, analysis.get("random_epoch_split_used") is False,
                       analysis.get("patient_is_resampling_unit") is True,
                       all(0.0 <= float(row[key]) <= 1.0 for row in network for key in ("matched_filter_auc", "best_channel_auc", "fastica_bss_auc", "time_shift_control_auc"))])
    except Exception as error: errors.append(f"{type(error).__name__}: {error}")
    report = {"schema_version": 1, "protocol": "mesial-events/external-validation-v1", "ok": not errors and all(checks),
              "checks_passed": sum(bool(value) for value in checks), "checks_total": len(checks), "errors": errors,
              "expected_templates": args.expected_templates}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
