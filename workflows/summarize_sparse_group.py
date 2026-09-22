#!/usr/bin/env python3
"""Aggregate sparse-group templates and calibrate patient-held-out firing thresholds."""

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
from empirical.source_scores import heldout_injection_threshold


PROTOCOL = "benchmark/cortical-method-control-v1"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
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
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected sparse-group protocol")
    template_dirs = sorted(path for path in args.template_root.iterdir() if path.is_dir() and (path / "report.json").is_file())
    if len(template_dirs) != int(config["expected_templates"]):
        raise ValueError(f"expected {config['expected_templates']} complete templates, found {len(template_dirs)}")
    raw: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    reports = []
    for directory in template_dirs:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if report.get("ok") is not True or report.get("protocol") != PROTOCOL:
            raise ValueError(f"invalid template report: {directory}")
        if int(report.get("nonfinite_score_rows", -1)) != 0:
            raise ValueError(f"non-finite template scores: {directory}")
        reports.append(report)
        for row in read_csv(directory / "method_scores.csv"):
            key = (row["ccepcoreg_subject"], row["stem"], row["endpoint"], row["method"])
            target = raw.setdefault(key, {"response": [], "injection": []})
            values = (float(row["response_delta"]), float(row["injection_delta"]))
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite score row: {directory} {key}")
            target["response"].append(values[0])
            target["injection"].append(values[1])
    consensus = []
    for (patient, stem, endpoint, method), values in sorted(raw.items()):
        if len(values["response"]) != len(template_dirs):
            raise ValueError(f"incomplete template consensus: {patient} {stem} {endpoint} {method}")
        consensus.append({
            "patient": patient,
            "stem": stem,
            "endpoint": endpoint,
            "method": method,
            "template_count": len(template_dirs),
            "response_delta_median": float(np.median(values["response"])),
            "response_delta_p05": float(np.quantile(values["response"], 0.05)),
            "response_delta_p95": float(np.quantile(values["response"], 0.95)),
            "injection_delta_median": float(np.median(values["injection"])),
        })
    patients = sorted({row["patient"] for row in consensus})
    sensitivity_target = float(config["threshold_calibration"]["injection_sensitivity_target"])
    patient_rows = []
    for endpoint in ENDPOINTS:
        for method in METHODS:
            selected = [row for row in consensus if row["endpoint"] == endpoint and row["method"] == method]
            for heldout in patients:
                training = [float(row["injection_delta_median"]) for row in selected if row["patient"] != heldout]
                testing = [row for row in selected if row["patient"] == heldout]
                threshold = heldout_injection_threshold(training, sensitivity_target)
                response_fire = [float(row["response_delta_median"]) > threshold for row in testing]
                injection_fire = [float(row["injection_delta_median"]) > threshold for row in testing]
                patient_rows.append({
                    "patient": heldout,
                    "endpoint": endpoint,
                    "method": method,
                    "threshold": threshold,
                    "training_injection_sessions": len(training),
                    "heldout_sessions": len(testing),
                    "cortical_false_fire_fraction": float(np.mean(response_fire)),
                    "calibrated_injection_sensitivity": float(np.mean(injection_fire)),
                    "median_cortical_response_delta": float(np.median([float(row["response_delta_median"]) for row in testing])),
                    "median_injection_delta": float(np.median([float(row["injection_delta_median"]) for row in testing])),
                })
    population_rows = []
    replicates = int(config["summary"]["bootstrap_replicates"])
    seed = int(config["summary"]["seed"])
    for endpoint in ENDPOINTS:
        for method_index, method in enumerate(METHODS):
            selected = [row for row in patient_rows if row["endpoint"] == endpoint and row["method"] == method]
            for metric in ("cortical_false_fire_fraction", "calibrated_injection_sensitivity", "median_cortical_response_delta", "median_injection_delta"):
                values = [float(row[metric]) for row in selected]
                low, high = bootstrap_median(values, replicates, seed + method_index + 100 * ENDPOINTS.index(endpoint))
                population_rows.append({
                    "endpoint": endpoint,
                    "method": method,
                    "metric": metric,
                    "patients": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "bootstrap_median_ci95_low": low,
                    "bootstrap_median_ci95_high": high,
                })
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "template_consensus_session_scores.csv", consensus)
    write_csv(args.output / "patient_metrics.csv", patient_rows)
    write_csv(args.output / "population_metrics.csv", population_rows)
    primary = {
        row["method"]: {
            row["metric"]: row["median"]
            for row in population_rows
            if row["endpoint"] == "integrated_pca3" and row["method"] == method
        }
        for method in METHODS
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": (
            len(template_dirs) == int(config["expected_templates"])
            and len(patients) == int(config["expected_ccepcoreg_patients"])
            and len({row["stem"] for row in consensus}) == int(config["expected_ccepcoreg_sessions"])
            and len({report["calibration_report_sha256"] for report in reports}) == 1
            and reports[0]["calibration_report_sha256"] == config["calibration_report_sha256"]
        ),
        "template_count": len(template_dirs),
        "patient_count": len(patients),
        "session_count": len({row["stem"] for row in consensus}),
        "methods": list(METHODS),
        "endpoints": list(ENDPOINTS),
        "threshold_calibration": f"leave-one-patient-out threshold targeting {sensitivity_target:.3f} sensitivity to empirically calibrated mesial-event injection",
        "injection_energy_ratio": float(reports[0]["injection_energy_ratio"]),
        "primary_integrated_pca3_medians": primary,
        "method_specific_cortical_control_complete": True,
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
