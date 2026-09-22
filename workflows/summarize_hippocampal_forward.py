#!/usr/bin/env python3
"""Aggregate paired legacy/HippUnfold geometry results and issue subset gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incoming")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _summary(values: np.ndarray, seed: int, replicates: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("summary values must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        bootstrap[index] = np.median(rng.choice(values, size=len(values), replace=True))
    return {
        "n": int(len(values)),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "bootstrap_median_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_median_ci95_high": float(np.quantile(bootstrap, 0.975)),
    }


def _paired(values_new: np.ndarray, values_old: np.ndarray, seed: int) -> dict[str, object]:
    if values_new.shape != values_old.shape:
        raise ValueError("paired vectors differ in shape")
    differences = values_new - values_old
    ratios = values_new / values_old
    try:
        test = wilcoxon(values_new, values_old, alternative="two-sided", zero_method="wilcox")
        statistic = float(test.statistic)
        pvalue = float(test.pvalue)
    except ValueError:
        statistic = 0.0
        pvalue = 1.0
    return {
        "new": _summary(values_new, seed, 10_000),
        "old": _summary(values_old, seed + 1, 10_000),
        "paired_difference_new_minus_old": _summary(differences, seed + 2, 10_000),
        "paired_ratio_new_over_old": _summary(ratios, seed + 3, 10_000),
        "wilcoxon_signed_rank_two_sided": {"statistic": statistic, "p_value": pvalue},
    }


def summarize(config_path: Path, full: bool) -> tuple[dict[str, object], dict[str, object] | None]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cohort = [
        line.strip()
        for line in (PROJECT_ROOT / str(config["subjects_file"])).read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]
    subjects = cohort if full else list(config["subset_subjects"])
    reports = []
    report_hashes = {}
    missing = []
    root = Path(str(config["new_forward_root"]))
    for subject in subjects:
        path = root / subject / "report.json"
        if not path.is_file():
            missing.append(subject)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append(report)
        report_hashes[subject] = _sha256(path)
    if missing:
        raise FileNotFoundError(f"missing subject reports: {missing}")
    qc = {
        "all_reports_ok": all(report.get("ok") is True for report in reports),
        "all_subjects_exact": sorted(report["subject"] for report in reports) == sorted(subjects),
        "all_sources_contained": all(report["containment"]["outside_inner_skull"] == 0 for report in reports),
        "all_forwards_finite_full_rank": all(
            report["forward"]["finite"] is True
            and report["forward"]["referenced_rank"]
            == report["forward"]["expected_sensor_rank"]
            == 338
            for report in reports
        ),
        "all_hierarchies_exactly_nested": all(
            report["source_model"]["exact_nested"] is True for report in reports
        ),
        "all_dentate_separate": all(
            report["source_model"]["dentate"]["included_in_primary_source_model"] is False
            for report in reports
        ),
        "all_montages_paired_exactly": all(
            report["montage"]["legacy_name_order_identical"] is True
            and report["montage"]["legacy_maximum_position_error_m"] <= 1e-11
            for report in reports
        ),
        "all_paired_metrics_present": all(
            "sensor_space" in report["agreement"]
            and "source_space" in report["agreement"]
            for report in reports
        ),
    }
    if not all(qc.values()):
        raise ValueError(f"HippUnfold paired QC failed: {qc}")

    metric_names = (
        "coherent_cancellation_ratio",
        "one_cycle_wave_cancellation_ratio",
        "standardized_recoverability_index_bits",
        "hippocampal_covariance_trace",
    )
    paired = {}
    seed = 20260828
    for offset, name in enumerate(metric_names):
        new = np.asarray([
            report["agreement"]["sensor_space"]["new"][name] for report in reports
        ])
        old = np.asarray([
            report["agreement"]["sensor_space"]["old"][name] for report in reports
        ])
        paired[name] = _paired(new, old, seed + 20 * offset)
    coherent_new = np.asarray([
        report["agreement"]["sensor_space"]["new"]["coherent_cancellation_ratio"]
        for report in reports
    ])
    old_new_orientation = np.asarray([
        report["agreement"]["source_space"]["pooled_absolute_orientation_cosine"]["median"]
        for report in reports
    ])
    old_new_topography = np.asarray([
        report["agreement"]["source_space"]["pooled_absolute_nearest_topography_cosine"]["median"]
        for report in reports
    ])
    covariance_distance = np.asarray([
        report["agreement"]["sensor_space"]["unit_trace_covariance_frobenius_distance"]
        for report in reports
    ])
    conclusion = {
        "strong_global_zero_phase_cancellation_threshold": "retained power <= 1%",
        "extreme_global_zero_phase_cancellation_threshold": "retained power <= 0.1%",
        "subjects_strong_cancellation_count": int(np.count_nonzero(coherent_new <= 0.01)),
        "subjects_extreme_cancellation_count": int(np.count_nonzero(coherent_new <= 0.001)),
        "subject_count": len(subjects),
        "strong_cancellation_survives_in_all_subjects": bool(np.all(coherent_new <= 0.01)),
        "interpretation_scope": (
            "global zero-phase CA/subiculum midthickness source under the frozen numerical normalization"
        ),
    }
    report = {
        "schema_version": 1,
        "ok": True,
        "protocol": config["protocol"],
        "mode": "full50" if full else "subset5",
        "created_at_utc": _now(),
        "subjects": subjects,
        "subject_count": len(subjects),
        "qc": qc,
        "paired_primary_metrics": paired,
        "agreement_summaries": {
            "median_absolute_nearest_orientation_cosine_per_subject": _summary(
                old_new_orientation, seed + 100, 10_000
            ),
            "median_absolute_nearest_topography_cosine_per_subject": _summary(
                old_new_topography, seed + 101, 10_000
            ),
            "unit_trace_covariance_frobenius_distance": _summary(
                covariance_distance, seed + 102, 10_000
            ),
        },
        "headline_cancellation_assessment": conclusion,
        "subject_report_sha256": report_hashes,
        "config_sha256": _sha256(config_path),
        "physiological_inference_authorized": False,
        "shared_storage_touched": False,
    }
    gate = None
    if not full:
        gate = {
            "schema_version": 1,
            "protocol": config["protocol"],
            "created_at_utc": _now(),
            "passed": bool(all(qc.values()) and len(reports) == 5),
            "purpose": "technical/anatomical validity gate; independent of scientific effect direction",
            "subset_subjects": subjects,
            "subset_summary_sha256": None,
            "full_50_authorized": bool(all(qc.values()) and len(reports) == 5),
            "shared_storage_touched": False,
        }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "hippocampal_forward_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--subset", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    status_root = args.output.resolve()
    status_root.mkdir(parents=True, exist_ok=True)
    report, gate = summarize(config_path, full=args.full)
    name = "full50_summary.json" if args.full else "subset_summary.json"
    report_path = status_root / name
    _atomic_json(report_path, report)
    if gate is not None:
        gate["subset_summary_sha256"] = _sha256(report_path)
        _atomic_json(status_root / "subset_gate.json", gate)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
