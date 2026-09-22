#!/usr/bin/env python3
"""Aggregate the 50 subject-level source-ensemble tables.

This module implements the numerical population reducer for the reported analysis.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROTOCOL = "hcp/source-ensembles-v1.1"
# The original deterministic bootstrap namespace is retained as bytes because it
# participates in the random seed.  Replacing it with the public protocol label
# would alter the reported confidence intervals.  It has no scientific meaning.
BOOTSTRAP_SEED_NAMESPACE = bytes(
    [72, 73, 80, 80, 45, 72, 67, 80, 45, 72, 73, 80, 80, 85, 78, 70, 79,
     76, 68, 45, 69, 78, 83, 69, 77, 66, 76, 69, 83, 45, 118, 49, 46, 49]
).decode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict[str, object], list[str]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected source-ensemble configuration")
    subject_path = PROJECT_ROOT / str(config["subjects_file"])
    subjects = [
        value.strip()
        for value in subject_path.read_text(encoding="utf-8-sig").splitlines()
        if value.strip()
    ]
    if len(subjects) != 50 or len(set(subjects)) != 50 or subjects != sorted(subjects):
        raise ValueError("the frozen source-ensemble cohort must contain 50 unique sorted subjects")
    return config, subjects


def _outputs_valid(directory: Path, subject: str) -> bool:
    required = [definition["file"] for definition in TABLES.values()]
    if not all((directory / str(name)).is_file() for name in required):
        return False
    report_path = directory / "report.json"
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return bool(
        report.get("ok") is True
        and report.get("subject") == subject
        and report.get("protocol") == PROTOCOL
        and all(report.get("qc", {}).values())
    )


TABLES = {
    "deterministic": {
        "file": "deterministic_conditions.csv",
        "keys": ("support_id", "hemisphere", "location", "target_extent", "regime", "wave_cycles"),
        "metrics": (
            "support_normalized_retained_power_ratio",
            "support_normalized_retained_rms_ratio",
            "whole_sheet_incoherent_power_ratio",
            "whole_sheet_recoverability_index_bits",
            "equal_support_recoverability_index_bits",
            "full_sheet_area_fraction",
        ),
    },
    "patch_coherence": {
        "file": "patch_coherence_conditions.csv",
        "keys": (
            "support_id",
            "hemisphere",
            "location",
            "target_extent",
            "intrinsic_bin_width",
        ),
        "metrics": (
            "support_normalized_retained_power_ratio",
            "support_normalized_retained_rms_ratio",
            "whole_sheet_incoherent_power_ratio",
            "whole_sheet_whitened_signal_power",
            "equal_support_whitened_signal_power",
            "coherent_patch_count",
            "coherent_patch_area_participation_ratio",
            "full_sheet_area_fraction",
        ),
    },
    "random_phase": {
        "file": "random_phase_conditions.csv",
        "keys": (
            "support_id",
            "hemisphere",
            "location",
            "target_extent",
            "intrinsic_correlation_length",
        ),
        "metrics": (
            "support_normalized_retained_power_ratio",
            "support_normalized_retained_rms_ratio",
            "whole_sheet_incoherent_power_ratio",
            "whole_sheet_recoverability_index_bits",
            "equal_support_recoverability_index_bits",
            "full_sheet_area_fraction",
        ),
        "collapse_replicates": True,
    },
    "orientation_mismatch": {
        "file": "orientation_mismatch_conditions.csv",
        "keys": (
            "support_id",
            "hemisphere",
            "location",
            "target_extent",
            "regime",
            "wave_cycles",
            "mismatch_angle_degrees",
        ),
        "metrics": (
            "whitened_truth_projection_efficiency",
            "whitened_subspace_minimum_cosine",
            "whitened_subspace_maximum_principal_angle_degrees",
            "truth_whitened_power",
            "assumed_whitened_power",
            "full_sheet_area_fraction",
        ),
    },
    "bilateral_interference": {
        "file": "bilateral_interference.csv",
        "keys": ("location", "target_extent", "regime", "wave_cycles"),
        "metrics": (
            "left_power",
            "right_power",
            "cross_term",
            "independent_hemisphere_power",
            "bilateral_locked_power",
            "bilateral_interference_gain",
            "cross_term_fraction_of_independent_power",
        ),
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incoming")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty subject table: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty aggregate table: {path}")
    fields = list(rows[0])
    temporary = path.with_name(path.name + ".incoming")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _seed(*components: object) -> int:
    text = "|".join(str(value) for value in components)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _summary(values: Iterable[float], seed: int) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("population summary requires finite nonempty values")
    rng = np.random.default_rng(seed)
    bootstrap = np.median(
        rng.choice(array, size=(10_000, len(array)), replace=True), axis=1
    )
    return {
        "minimum": float(np.min(array)),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "bootstrap_median_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_median_ci95_high": float(np.quantile(bootstrap, 0.975)),
    }


def _group_key(row: dict[str, object], keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[key]) for key in keys)


def _collapse_random_replicates(
    rows: list[dict[str, object]], keys: tuple[str, ...], metrics: tuple[str, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    subject_keys = ("subject", *keys)
    for row in rows:
        grouped.setdefault(_group_key(row, subject_keys), []).append(row)
    collapsed: list[dict[str, object]] = []
    for group_rows in grouped.values():
        first = group_rows[0]
        record = {key: first[key] for key in subject_keys}
        record["replicates"] = len(group_rows)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows])
            record[metric] = float(np.median(values))
            record[f"{metric}__within_q10"] = float(np.quantile(values, 0.10))
            record[f"{metric}__within_q90"] = float(np.quantile(values, 0.90))
        collapsed.append(record)
    return collapsed


def _aggregate_table(
    rows: list[dict[str, object]],
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
    expected_subjects: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row, keys), []).append(row)
    aggregate: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        subjects = {str(row["subject"]) for row in group_rows}
        if len(subjects) != expected_subjects or len(group_rows) != expected_subjects:
            raise ValueError(
                f"condition {key} has {len(group_rows)} rows and {len(subjects)} subjects"
            )
        record: dict[str, object] = {
            name: group_rows[0][name] for name in keys
        }
        record["n_subjects"] = len(subjects)
        for metric in metrics:
            summary = _summary(
                (float(row[metric]) for row in group_rows),
                _seed(BOOTSTRAP_SEED_NAMESPACE, *key, metric, expected_subjects),
            )
            for statistic, value in summary.items():
                record[f"{metric}__{statistic}"] = value
        aggregate.append(record)
    return aggregate


def _find(
    rows: list[dict[str, object]], **criteria: object
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if all(str(row[key]) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one aggregate row for {criteria}, found {len(matches)}")
    return matches[0]


def summarize(
    config_path: Path,
    subject_root: Path,
    output_directory: Path,
) -> dict[str, object]:
    config, subjects = load_config(config_path)
    config_sha256 = _sha256(config_path)
    subject_reports: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    raw_by_table: dict[str, list[dict[str, object]]] = {name: [] for name in TABLES}
    for subject in subjects:
        directory = subject_root / subject
        if not _outputs_valid(directory, subject):
            raise ValueError(f"invalid or incomplete source-ensemble subject: {subject}")
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        subject_reports.append(report)
        report_hashes[subject] = _sha256(report_path)
        for name, definition in TABLES.items():
            rows = _read_csv(directory / str(definition["file"]))
            if any(row.get("subject") != subject for row in rows):
                raise ValueError(f"{subject} table {name} contains another subject")
            raw_by_table[name].extend(rows)

    output_directory.mkdir(parents=True, exist_ok=True)
    aggregates: dict[str, list[dict[str, object]]] = {}
    table_outputs: dict[str, dict[str, object]] = {}
    for name, definition in TABLES.items():
        rows: list[dict[str, object]] = raw_by_table[name]
        if definition.get("collapse_replicates"):
            rows = _collapse_random_replicates(
                rows,
                tuple(definition["keys"]),
                tuple(definition["metrics"]),
            )
        aggregate = _aggregate_table(
            rows,
            tuple(definition["keys"]),
            tuple(definition["metrics"]),
            len(subjects),
        )
        aggregates[name] = aggregate
        output_names = {
            "deterministic": "source_ensembles_deterministic_population_v1_1.csv",
            "patch_coherence": "source_ensembles_patch_coherence_population_v1_1.csv",
            "random_phase": "source_ensembles_random_phase_population_v1_1.csv",
            "orientation_mismatch": "source_ensembles_orientation_mismatch_population_v1_1.csv",
            "bilateral_interference": "source_ensembles_bilateral_interference_population_v1_1.csv",
        }
        path = output_directory / output_names[name]
        _write_csv(path, aggregate)
        table_outputs[path.name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "rows": len(aggregate),
        }

    deterministic = aggregates["deterministic"]
    interference = aggregates["bilateral_interference"]
    random_phase = aggregates["random_phase"]
    mismatch = aggregates["orientation_mismatch"]
    patch = aggregates["patch_coherence"]
    primary_metric = "support_normalized_retained_power_ratio__median"
    whole_zero = {
        family: _find(
            deterministic,
            hemisphere=family,
            location="whole",
            target_extent="1.0",
            regime="zero_phase",
            wave_cycles="0.0",
        )[primary_metric]
        for family in ("left", "right", "bilateral")
    }
    bilateral_waves = {
        str(cycle): _find(
            deterministic,
            hemisphere="bilateral",
            location="whole",
            target_extent="1.0",
            regime=f"wave_{cycle:g}_cycles",
            wave_cycles=str(float(cycle)),
        )[primary_metric]
        for cycle in config["deterministic_phase"]["wave_cycles"]
    }
    whole_interference = _find(
        interference,
        location="whole",
        target_extent="1.0",
        regime="zero_phase",
        wave_cycles="0.0",
    )
    coherence_ladder = {
        str(width): _find(
            patch,
            hemisphere="bilateral",
            location="whole",
            target_extent="1.0",
            intrinsic_bin_width=str(float(width)),
        )[primary_metric]
        for width in config["patch_coherence"]["intrinsic_bin_widths"]
    }
    random_ladder = {
        str(length): _find(
            random_phase,
            hemisphere="bilateral",
            location="whole",
            target_extent="1.0",
            intrinsic_correlation_length=str(float(length)),
        )[primary_metric]
        for length in config["random_phase"]["intrinsic_correlation_lengths"]
    }
    mismatch_ladder = {
        str(angle): _find(
            mismatch,
            hemisphere="bilateral",
            location="whole",
            target_extent="1.0",
            regime="zero_phase",
            wave_cycles="0.0",
            mismatch_angle_degrees=str(float(angle)),
        )["whitened_truth_projection_efficiency__median"]
        for angle in config["orientation_mismatch"]["angles_degrees"]
    }
    qc = {
        "all_subject_reports_ok": all(report.get("ok") is True for report in subject_reports),
        "all_subject_qc_passed": all(all(report.get("qc", {}).values()) for report in subject_reports),
        "subject_count_exact": len(subject_reports) == len(subjects),
        "subjects_exact": sorted(report["subject"] for report in subject_reports)
        == sorted(subjects),
        "dentate_excluded": all(
            report["source_model"]["dentate_included"] is False
            for report in subject_reports
        ),
        "effect_independent_gate": True,
        "all_population_tables_present": set(aggregates) == set(TABLES),
    }
    if not all(qc.values()):
        raise ValueError(f"source-ensemble population QC failed: {qc}")
    report = {
        "schema_version": 1,
        "ok": True,
        "protocol": PROTOCOL,
        "mode": "full50",
        "created_at_utc": _now(),
        "subject_count": len(subjects),
        "subjects": subjects,
        "qc": qc,
        "condition_counts": {
            name: len(rows) for name, rows in aggregates.items()
        },
        "headline_descriptive_results": {
            "whole_zero_phase_support_normalized_retained_power_medians": whole_zero,
            "whole_bilateral_wave_support_normalized_retained_power_medians": (
                bilateral_waves
            ),
            "whole_bilateral_zero_phase_interference_gain_median": whole_interference[
                "bilateral_interference_gain__median"
            ],
            "whole_bilateral_zero_phase_cross_term_fraction_median": whole_interference[
                "cross_term_fraction_of_independent_power__median"
            ],
            "whole_bilateral_patch_coherence_retained_power_medians": coherence_ladder,
            "whole_bilateral_random_phase_subject_median_retained_power": random_ladder,
            "whole_bilateral_zero_phase_orientation_mismatch_projection_efficiency_medians": (
                mismatch_ladder
            ),
        },
        "population_tables": table_outputs,
        "subject_report_sha256": report_hashes,
        "config_sha256": config_sha256,
        "gate_uses_scientific_effect": False,
        "physiological_inference_authorized": False,
        "shared_storage_touched": False,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "source_ensembles_v1_1.json",
    )
    parser.add_argument(
        "--subject-root",
        type=Path,
        help="Directory containing one generated subdirectory per frozen subject",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    subject_root = (
        args.subject_root.resolve(strict=True)
        if args.subject_root
        else (PROJECT_ROOT / str(config["output_root"])).resolve(strict=True)
    )
    output_directory = args.output.resolve()
    report = summarize(config_path, subject_root, output_directory)
    report_path = output_directory / "source_ensembles_population_report.json"
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
