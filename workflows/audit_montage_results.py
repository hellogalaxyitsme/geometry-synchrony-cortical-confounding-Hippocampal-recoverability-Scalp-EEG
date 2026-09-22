#!/usr/bin/env python3
"""Independent structural and numerical consumer for montage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest(); return digest


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True); parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    checks = []; errors = []
    for subject in sorted(args.subjects):
        directory = args.subject_root / subject
        try:
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            metrics = rows(directory / "metrics.csv"); increments = rows(directory / "increments.csv")
            checks.extend([
                report.get("ok") is True, report.get("subject") == subject,
                len(metrics) == int(report["condition_count"]), len(increments) == int(report["increment_count"]),
                sha256(directory / "metrics.csv") == report["outputs"]["metrics_sha256"],
                sha256(directory / "increments.csv") == report["outputs"]["increments_sha256"],
                min(float(row["conditional_information_bits"]) for row in increments if row["augmented_montage"] != "conventional64_plus_inferior") >= -1e-9,
                report.get("physiological_inference_authorized") is False,
            ])
        except Exception as error: errors.append(f"{subject}: {error}")
    try:
        summary = json.loads((args.summary_root / "report.json").read_text(encoding="utf-8"))
        checks.extend([summary.get("ok") is True, summary.get("subjects") == sorted(args.subjects), summary.get("face_neck_evaluated") is False])
        population_metrics = rows(args.summary_root / "population_metrics.csv")
        population_increments = rows(args.summary_root / "population_increments.csv")
        checks.extend([len(population_metrics) > 0, len(population_increments) > 0])
    except Exception as error: errors.append(f"summary: {error}")
    report = {
        "schema_version": 1, "protocol": "montage/physical-value-v1", "ok": not errors and all(checks),
        "checks_passed": sum(bool(value) for value in checks), "checks_total": len(checks), "errors": errors,
        "subjects": sorted(args.subjects),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
