#!/usr/bin/env python3
"""Patient-equal, template-ensemble summary for stimulation-control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


PROTOCOL = "cortical-control/adversarial-stimulation-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile_summary(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.median(array)), float(np.quantile(array, 0.95)), float(np.max(array))


def bootstrap_median(values: Iterable[float], replicates: int, seed: int) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        estimates[index] = np.median(generator.choice(array, size=len(array), replace=True))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def seed_for(*parts: object) -> int:
    text = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def summarize(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise SystemExit("unexpected stimulation-control protocol")
    template_dirs = sorted(path for path in args.results_root.iterdir() if (path / "report.json").is_file())
    reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in template_dirs]
    if len(template_dirs) != args.expected_templates or any(report.get("ok") is not True for report in reports):
        raise ValueError("template result count/pass gate failed")
    template_ids = [str(report["hcp_subject"]) for report in reports]
    if len(set(template_ids)) != len(template_ids):
        raise ValueError("duplicate HCP template results")
    threshold = float(config["detector"]["reporting_threshold"])

    scan_groups: dict[tuple[str, str, str, str], list[tuple[float, float, float, float, float, float]]] = defaultdict(list)
    injection_groups: dict[tuple[str, str, str, float], list[tuple[float, float, float]]] = defaultdict(list)
    provenance: dict[str, dict[str, object]] = {}
    for directory, report in zip(template_dirs, reports):
        scan_path = directory / "scan_max.csv"
        injection_path = directory / "injection.csv"
        for row in read_csv(scan_path):
            key = (row["ccepcoreg_subject"], row["run"], row["endpoint"], row["restriction"])
            scan_groups[key].append(
                (
                    float(row["response_conditional_scan_max"]),
                    float(row["baseline_conditional_scan_max"]),
                    float(row["response_total_at_conditional_max"]),
                    float(row["baseline_total_at_conditional_max"]),
                    float(row["response_residual_at_conditional_max"]),
                    float(row["baseline_residual_at_conditional_max"]),
                )
            )
        for row in read_csv(injection_path):
            key = (
                row["ccepcoreg_subject"], row["run"], row["restriction"], float(row["injection_energy_ratio"])
            )
            injection_groups[key].append(
                (
                    float(row["injection_conditional_score"]),
                    float(row["injection_score"]),
                    float(row["injection_residual_energy_fraction"]),
                )
            )
        provenance[str(report["hcp_subject"])] = {
            "report_sha256": sha256(directory / "report.json"),
            "scan_sha256": sha256(scan_path),
            "injection_sha256": sha256(injection_path),
        }
    if any(len(values) != args.expected_templates for values in scan_groups.values()):
        raise ValueError("a session/endpoint/restriction lacks a complete anatomy ensemble")
    if any(len(values) != args.expected_templates for values in injection_groups.values()):
        raise ValueError("an injection condition lacks a complete anatomy ensemble")

    session_rows: list[dict[str, object]] = []
    for key in sorted(scan_groups):
        values = np.asarray(scan_groups[key], dtype=np.float64)
        response_median, response_p95, response_maximum = percentile_summary(values[:, 0])
        baseline_median, baseline_p95, baseline_maximum = percentile_summary(values[:, 1])
        session_rows.append(
            {
                "ccepcoreg_subject": key[0],
                "run": key[1],
                "endpoint": key[2],
                "restriction": key[3],
                "template_count": len(values),
                "response_conditional_template_median": response_median,
                "response_conditional_template_p95": response_p95,
                "response_conditional_template_maximum": response_maximum,
                "baseline_conditional_template_median": baseline_median,
                "baseline_conditional_template_p95": baseline_p95,
                "baseline_conditional_template_maximum": baseline_maximum,
                "response_total_attribution_template_median": float(np.median(values[:, 2])),
                "baseline_total_attribution_template_median": float(np.median(values[:, 3])),
                "response_cortical_residual_template_median": float(np.median(values[:, 4])),
                "baseline_cortical_residual_template_median": float(np.median(values[:, 5])),
                "response_minus_baseline_conditional": response_median - baseline_median,
                "response_above_threshold": response_median > threshold,
                "baseline_above_threshold": baseline_median > threshold,
                "excess_response_fire": response_median > threshold and baseline_median <= threshold,
            }
        )

    by_patient: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in session_rows:
        by_patient[(str(row["endpoint"]), str(row["restriction"]), str(row["ccepcoreg_subject"]))].append(row)
    patient_rows: list[dict[str, object]] = []
    for key in sorted(by_patient):
        group = by_patient[key]
        response = np.asarray([float(row["response_conditional_template_median"]) for row in group])
        baseline = np.asarray([float(row["baseline_conditional_template_median"]) for row in group])
        patient_rows.append(
            {
                "endpoint": key[0],
                "restriction": key[1],
                "ccepcoreg_subject": key[2],
                "session_count": len(group),
                "response_conditional_session_median": float(np.median(response)),
                "response_conditional_session_p95": float(np.quantile(response, 0.95)),
                "baseline_conditional_session_median": float(np.median(baseline)),
                "baseline_conditional_session_p95": float(np.quantile(baseline, 0.95)),
                "response_minus_baseline_conditional_session_median": float(np.median(response - baseline)),
                "response_total_attribution_session_median": float(np.median([float(row["response_total_attribution_template_median"]) for row in group])),
                "response_cortical_residual_session_median": float(np.median([float(row["response_cortical_residual_template_median"]) for row in group])),
                "response_false_fire_fraction": float(np.mean(response > threshold)),
                "baseline_fire_fraction": float(np.mean(baseline > threshold)),
                "excess_response_fire_fraction": float(np.mean((response > threshold) & (baseline <= threshold))),
            }
        )

    replicates = int(config["summary"]["bootstrap_replicates"])
    population_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in patient_rows:
        population_groups[(str(row["endpoint"]), str(row["restriction"]))].append(row)
    population_rows: list[dict[str, object]] = []
    for key in sorted(population_groups):
        group = population_groups[key]
        response = [float(row["response_conditional_session_median"]) for row in group]
        baseline = [float(row["baseline_conditional_session_median"]) for row in group]
        false_fraction = [float(row["response_false_fire_fraction"]) for row in group]
        excess_fraction = [float(row["excess_response_fire_fraction"]) for row in group]
        response_ci = bootstrap_median(response, replicates, seed_for(PROTOCOL, *key, "response"))
        false_ci = bootstrap_median(false_fraction, replicates, seed_for(PROTOCOL, *key, "false"))
        population_rows.append(
            {
                "endpoint": key[0],
                "restriction": key[1],
                "patient_count": len(group),
                "patient_equal_response_conditional_median": float(np.median(response)),
                "response_median_ci025": response_ci[0],
                "response_median_ci975": response_ci[1],
                "patient_equal_response_conditional_p95": float(np.quantile(response, 0.95)),
                "patient_equal_baseline_conditional_median": float(np.median(baseline)),
                "patient_equal_response_minus_baseline_conditional": float(np.median(np.asarray(response) - np.asarray(baseline))),
                "patient_equal_response_total_attribution_median": float(np.median([float(row["response_total_attribution_session_median"]) for row in group])),
                "patient_equal_response_cortical_residual_median": float(np.median([float(row["response_cortical_residual_session_median"]) for row in group])),
                "patient_equal_false_fire_fraction_median": float(np.median(false_fraction)),
                "false_fire_fraction_ci025": false_ci[0],
                "false_fire_fraction_ci975": false_ci[1],
                "patient_equal_excess_fire_fraction_median": float(np.median(excess_fraction)),
                "threshold": threshold,
            }
        )

    injection_session_rows: list[dict[str, object]] = []
    for key in sorted(injection_groups):
        values = np.asarray(injection_groups[key], dtype=np.float64)
        injection_session_rows.append(
            {
                "ccepcoreg_subject": key[0],
                "run": key[1],
                "restriction": key[2],
                "injection_energy_ratio": key[3],
                "template_count": len(values),
                "injection_conditional_template_median": float(np.median(values[:, 0])),
                "injection_conditional_template_p95": float(np.quantile(values[:, 0], 0.95)),
                "injection_total_attribution_template_median": float(np.median(values[:, 1])),
                "injection_cortical_residual_template_median": float(np.median(values[:, 2])),
                "above_threshold": float(np.median(values[:, 0])) > threshold,
            }
        )
    injection_patient_groups: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in injection_session_rows:
        injection_patient_groups[(str(row["restriction"]), float(row["injection_energy_ratio"]), str(row["ccepcoreg_subject"]))].append(row)
    injection_patient_rows: list[dict[str, object]] = []
    for key in sorted(injection_patient_groups):
        group = injection_patient_groups[key]
        values = np.asarray([float(row["injection_conditional_template_median"]) for row in group])
        injection_patient_rows.append(
            {
                "restriction": key[0],
                "injection_energy_ratio": key[1],
                "ccepcoreg_subject": key[2],
                "session_count": len(group),
                "injection_conditional_session_median": float(np.median(values)),
                "injection_total_attribution_session_median": float(np.median([float(row["injection_total_attribution_template_median"]) for row in group])),
                "injection_cortical_residual_session_median": float(np.median([float(row["injection_cortical_residual_template_median"]) for row in group])),
                "injection_fire_fraction": float(np.mean(values > threshold)),
            }
        )
    injection_population_groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in injection_patient_rows:
        injection_population_groups[(str(row["restriction"]), float(row["injection_energy_ratio"]))].append(row)
    injection_population_rows: list[dict[str, object]] = []
    for key in sorted(injection_population_groups):
        group = injection_population_groups[key]
        scores = [float(row["injection_conditional_session_median"]) for row in group]
        fires = [float(row["injection_fire_fraction"]) for row in group]
        injection_population_rows.append(
            {
                "restriction": key[0],
                "injection_energy_ratio": key[1],
                "patient_count": len(group),
                "patient_equal_injection_conditional_median": float(np.median(scores)),
                "patient_equal_injection_conditional_p95": float(np.quantile(scores, 0.95)),
                "patient_equal_injection_total_attribution_median": float(np.median([float(row["injection_total_attribution_session_median"]) for row in group])),
                "patient_equal_injection_cortical_residual_median": float(np.median([float(row["injection_cortical_residual_session_median"]) for row in group])),
                "patient_equal_injection_fire_fraction_median": float(np.median(fires)),
                "threshold": threshold,
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    output_tables = {
        "session_consensus.csv": session_rows,
        "patient_summary.csv": patient_rows,
        "population_summary.csv": population_rows,
        "injection_session_consensus.csv": injection_session_rows,
        "injection_patient_summary.csv": injection_patient_rows,
        "injection_population_summary.csv": injection_population_rows,
    }
    for name, rows in output_tables.items():
        write_csv(args.output / name, rows)
    patients = sorted({str(row["ccepcoreg_subject"]) for row in session_rows})
    sessions = sorted({(str(row["ccepcoreg_subject"]), str(row["run"])) for row in session_rows})
    all_values = np.asarray(
        [
            float(row[key])
            for row in session_rows
            for key in (
                "response_conditional_template_median",
                "baseline_conditional_template_median",
                "response_total_attribution_template_median",
                "baseline_total_attribution_template_median",
                "response_cortical_residual_template_median",
                "baseline_cortical_residual_template_median",
            )
        ]
        + [
            float(row[key])
            for row in injection_session_rows
            for key in (
                "injection_conditional_template_median",
                "injection_total_attribution_template_median",
                "injection_cortical_residual_template_median",
            )
        ]
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(
            len(template_ids) == args.expected_templates
            and len(patients) == args.expected_patients
            and len(sessions) == args.expected_sessions
            and np.all(np.isfinite(all_values))
            and np.min(all_values) >= -float(config["gates"]["numerical_fraction_tolerance"])
            and np.max(all_values) <= 1.0 + float(config["gates"]["numerical_fraction_tolerance"])
        ),
        "template_count": len(template_ids),
        "templates": sorted(template_ids),
        "patient_count": len(patients),
        "patients": patients,
        "session_count": len(sessions),
        "session_consensus_rows": len(session_rows),
        "patient_summary_rows": len(patient_rows),
        "population_summary_rows": len(population_rows),
        "injection_session_rows": len(injection_session_rows),
        "patient_equal_weighting": True,
        "primary_endpoint": config["windows"]["primary_endpoint"],
        "template_primary_aggregation": "median",
        "outcome_gate_applied": False,
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "provenance": provenance,
        "outputs": {
            name: {"bytes": (args.output / name).stat().st_size, "sha256": sha256(args.output / name)}
            for name in output_tables
        },
    }
    report_path = args.output / "summary_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("provenance", "outputs")}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-templates", type=int, required=True)
    parser.add_argument("--expected-patients", type=int, required=True)
    parser.add_argument("--expected-sessions", type=int, required=True)
    args = parser.parse_args()
    return 0 if summarize(args)["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
