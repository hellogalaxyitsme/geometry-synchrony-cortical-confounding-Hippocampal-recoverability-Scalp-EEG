#!/usr/bin/env python3
"""Population uncertainty envelopes for head-model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("hippocampal_amplitude_ratio", "cortical_amplitude_ratio", "hippocampal_wave_amplitude_ratio",
           "hippocampal_covariance_shape_distance", "cortical_covariance_shape_distance", "hippocampal_wave_shape_distance",
           "hippocampal_coherent_cosine", "recoverability_spectrum_relative_error", "recoverability_spectrum_sum", "recoverability_spectrum_largest")


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))
def seed(master: int, text: str): return int.from_bytes(hashlib.sha256(f"{master}|{text}".encode()).digest()[:8], "little")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True); parser.add_argument("--fem-report", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8")); subjects = sorted(args.subjects); all_rows = []
    for subject in subjects:
        report = json.loads((args.subject_root / subject / "report.json").read_text(encoding="utf-8"))
        if report.get("ok") is not True or report.get("subject") != subject: raise ValueError(f"invalid subject {subject}")
        all_rows.extend(read(args.subject_root / subject / "conditions.csv"))
    grouped = {}
    for row in all_rows: grouped.setdefault((row["condition"], row["family"], row["reference_condition"]), []).append(row)
    output_rows = []; generator_count = int(config["bootstrap_replicates"])
    for key, rows in sorted(grouped.items()):
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in rows]); rng = np.random.default_rng(seed(int(config["master_seed"]), "|".join(key) + metric))
            boot = np.median(values[rng.integers(0, len(values), size=(generator_count, len(values)))], axis=1)
            output_rows.append({"condition": key[0], "family": key[1], "reference_condition": key[2], "metric": metric,
                                "subject_median": float(np.median(values)), "subject_minimum": float(np.min(values)), "subject_maximum": float(np.max(values)),
                                "bootstrap_ci_low": float(np.quantile(boot, 0.025)), "bootstrap_ci_high": float(np.quantile(boot, 0.975))})
    args.output.mkdir(parents=True, exist_ok=False); summary_path = args.output / "population_uncertainty.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    fem = json.loads(args.fem_report.read_text(encoding="utf-8"))
    amplitude_rows = [row for row in output_rows if row["metric"] == "hippocampal_amplitude_ratio" and row["condition"] != "reference"]
    report = {"schema_version": 1, "protocol": "uncertainty/head-model-v1", "ok": fem.get("ok") is True and len(grouped) == 15,
              "subject_count": len(subjects), "subjects": subjects, "condition_count": len(grouped),
              "hippocampal_amplitude_ratio_envelope": [min(row["subject_minimum"] for row in amplitude_rows), max(row["subject_maximum"] for row in amplitude_rows)],
              "new_york_head_fem_context": {"ok": fem.get("ok"), "benchmark_kind": fem.get("benchmark_kind"), "hippocampal_operator_used": fem.get("hippocampal_operator_used"),
                                                "physical_scale_established": fem.get("leadfield_physical_scale_status")},
              "hcp_fem_run": False, "hcp_fem_reason": config["hcp_fem_reason"],
              "alternative_hcp_conductor": "three-shell analytical sphere", "physiological_inference_authorized": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
