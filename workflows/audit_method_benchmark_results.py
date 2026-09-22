#!/usr/bin/env python3
"""Independent structural and numerical audit of method-benchmark benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.working_memory import atomic_json, roc_auc  # noqa: E402


PROTOCOL = "benchmark/method-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs-root", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--expected-patients", type=int, required=True)
    parser.add_argument("--expected-templates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report_path = args.benchmark_root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads((args.epochs_root / "extraction_manifest.json").read_text(encoding="utf-8"))
    methods = [str(value) for value in report["methods"]]
    expected_methods = [
        "best_channel", "fastica_bss", "minimum_norm", "eloreta_type", "lcmv",
        "sparse_l1", "hierarchical_sparse_group",
    ]
    events = rows(args.benchmark_root / "event_scores.csv")
    patients = rows(args.benchmark_root / "patient_metrics.csv")
    folds = rows(args.benchmark_root / "fold_metrics.csv")
    diagnostics = rows(args.benchmark_root / "operator_diagnostics.csv")
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    check("producer_report", report.get("ok") is True and report.get("protocol") == PROTOCOL, report.get("ok"))
    check("config_protocol", config.get("protocol") == PROTOCOL, config.get("protocol"))
    check("config_hash", report.get("config", {}).get("sha256") == sha256(args.config), report.get("config"))
    check("methods_exact", methods == expected_methods, methods)
    check("patient_count", len(patients) == args.expected_patients == report.get("patient_count"), len(patients))
    check("template_count", report.get("template_count") == args.expected_templates, report.get("template_count"))
    check("event_count", len(events) == report.get("evaluated_event_count") and len(events) > 0, len(events))
    check("late_block_only", {row["block"] for row in events} == {"2"}, sorted({row["block"] for row in events}))
    identities = [(row["patient"], row["session"], row["trial_number"]) for row in events]
    check("event_identity_unique", len(identities) == len(set(identities)), len(set(identities)))
    check(
        "no_depth_waveform",
        manifest.get("depth_waveform_included") is False
        and report.get("depth_waveform_in_scalp_solution") is False,
        {"manifest": manifest.get("depth_waveform_included"), "report": report.get("depth_waveform_in_scalp_solution")},
    )
    patient_names = sorted({row["patient"] for row in events})
    check("event_patient_coverage", patient_names == sorted(row["patient"] for row in patients), patient_names)
    score_finite = True
    auc_reproduced = True
    macro_reproduced = True
    for method in methods:
        score_key = f"{method}_score"
        auc_key = f"{method}_auc"
        all_scores = np.asarray([float(row[score_key]) for row in events], dtype=np.float64)
        score_finite &= bool(np.all(np.isfinite(all_scores)))
        patient_aucs = []
        for patient_row in patients:
            selected = [row for row in events if row["patient"] == patient_row["patient"]]
            labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
            scores = np.asarray([float(row[score_key]) for row in selected], dtype=np.float64)
            reproduced = roc_auc(labels, scores)
            stored = float(patient_row[auc_key])
            auc_reproduced &= abs(reproduced - stored) <= 1e-14
            patient_aucs.append(reproduced)
        macro = float(np.mean(patient_aucs))
        stored_macro = float(report["macro_patient_auc"][method]["mean"])
        macro_reproduced &= abs(macro - stored_macro) <= 1e-14
    check("all_scores_finite", score_finite, score_finite)
    check("patient_auc_reproduced", auc_reproduced, auc_reproduced)
    check("macro_auc_reproduced", macro_reproduced, macro_reproduced)
    expected_fold_pairs = {(patient, method) for patient in patient_names for method in methods}
    actual_fold_pairs = {(row["heldout_patient"], row["method"]) for row in folds}
    check("fold_method_grid_exact", actual_fold_pairs == expected_fold_pairs and len(folds) == len(expected_fold_pairs), len(folds))
    template_reports = []
    for subject in report["template_subjects"]:
        path = args.template_root / str(subject) / "report.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        template_reports.append(value)
    check(
        "templates_independently_valid",
        len(template_reports) == args.expected_templates
        and all(row.get("ok") is True and row.get("support_construction_exact") is True for row in template_reports),
        len(template_reports),
    )
    lcmv_errors = [
        float(row["maximum_unit_gain_error_before_NAI_normalization"])
        for row in diagnostics
        if row["method"] == "lcmv" and row.get("maximum_unit_gain_error_before_NAI_normalization", "")
    ]
    check(
        "lcmv_unit_gain",
        bool(lcmv_errors)
        and max(lcmv_errors) <= float(config["gates"]["maximum_lcmv_unit_gain_error"]),
        max(lcmv_errors) if lcmv_errors else None,
    )
    sparse_convergence = [
        float(row["converged_fraction"])
        for row in diagnostics
        if row["method"] in {"sparse_l1", "hierarchical_sparse_group"} and row.get("converged_fraction", "")
    ]
    check(
        "sparse_convergence",
        bool(sparse_convergence) and min(sparse_convergence) >= float(config["gates"]["minimum_sparse_converged_fraction"]),
        min(sparse_convergence) if sparse_convergence else None,
    )
    sparse_stationarity = [
        float(row["maximum_proximal_gradient_stationarity"])
        for row in diagnostics
        if row["method"] in {"sparse_l1", "hierarchical_sparse_group"}
        and row.get("maximum_proximal_gradient_stationarity", "")
    ]
    check(
        "sparse_stationarity",
        bool(sparse_stationarity)
        and max(sparse_stationarity) <= float(config["gates"]["maximum_sparse_stationarity"]),
        max(sparse_stationarity) if sparse_stationarity else None,
    )
    output_hashes_valid = all(
        (args.benchmark_root / name).is_file()
        and sha256(args.benchmark_root / name) == metadata["sha256"]
        for name, metadata in report["outputs"].items()
    )
    check("output_hashes", output_hashes_valid, output_hashes_valid)
    passed = sum(bool(row["passed"]) for row in checks)
    audit = {
        "schema_version": 1,
        "protocol": "benchmark/method-audit-v1",
        "ok": passed == len(checks),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "producer_report_sha256": sha256(report_path),
        "participants_tsv_read": False,
        "physiological_inference_authorized": False,
    }
    atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
