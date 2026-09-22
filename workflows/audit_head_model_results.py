#!/usr/bin/env python3
"""Independent operator-level audit for head-model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from anatomical.convergence import SensorOperators, compare_conditions  # noqa: E402


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    with path.open("r", encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--subject-root", type=Path, required=True); parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    checks = []; errors = []; maximum_error = 0.0
    for subject in sorted(args.subjects):
        directory = args.subject_root / subject
        try:
            report = json.loads((directory / "report.json").read_text(encoding="utf-8")); conditions = read(directory / "conditions.csv")
            checks.extend([report.get("ok") is True, report.get("subject") == subject, len(conditions) == 15,
                           sha256(directory / "conditions.csv") == report["outputs"]["conditions_sha256"], sha256(directory / "operators.npz") == report["outputs"]["operators_sha256"]])
            by_name = {row["condition"]: row for row in conditions}
            with np.load(directory / "operators.npz", allow_pickle=False) as archive:
                def operator(name):
                    return SensorOperators(archive[f"{name}__hippocampal_covariance"], archive[f"{name}__cortical_covariance"],
                                           archive[f"{name}__hippocampal_wave_covariance"], archive[f"{name}__hippocampal_coherent_topography"])
                for name, row in by_name.items():
                    if name == row["reference_condition"]: continue
                    observed = compare_conditions(operator(name), operator(row["reference_condition"]), archive[f"{name}__recoverability_spectrum"], archive[f"{row['reference_condition']}__recoverability_spectrum"])
                    for key, value in observed.items(): maximum_error = max(maximum_error, abs(float(row[key]) - float(value)))
            checks.append(maximum_error <= 1e-10)
        except Exception as error: errors.append(f"{subject}: {type(error).__name__}: {error}")
    try:
        summary = json.loads((args.summary_root / "report.json").read_text(encoding="utf-8"))
        checks.extend([summary.get("ok") is True, summary.get("subjects") == sorted(args.subjects), summary.get("hcp_fem_run") is False,
                       summary.get("new_york_head_fem_context", {}).get("hippocampal_operator_used") is False])
    except Exception as error: errors.append(f"summary: {error}")
    report = {"schema_version": 1, "protocol": "uncertainty/head-model-v1", "ok": not errors and all(checks),
              "checks_passed": sum(bool(value) for value in checks), "checks_total": len(checks), "maximum_comparison_reconstruction_error": maximum_error, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
