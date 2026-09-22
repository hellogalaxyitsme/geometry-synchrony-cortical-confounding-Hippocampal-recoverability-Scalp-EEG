#!/usr/bin/env python3
"""Population summaries for the completed cortical-restriction subject outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from simulation.cortical_restriction import stable_seed  # noqa: E402


PROTOCOL = "restriction/ladder-v1"
PRIMARY = ("unrestricted", "covariance", "smooth", "support", "sparse_k4", "spatiotemporal")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, values: list[dict[str, object]]) -> None:
    if not values:
        raise ValueError("population summary table is empty")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def bootstrap_median(values: Iterable[float], replicates: int, seed: int) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = np.median(
        array[generator.integers(0, len(array), size=(replicates, len(array)))], axis=1
    )
    return {
        "subjects": len(array),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "bootstrap_replicates": replicates,
        "median_ci95_low": float(np.quantile(samples, 0.025)),
        "median_ci95_high": float(np.quantile(samples, 0.975)),
    }


def grouped(values: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    result: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in values:
        key = tuple(row[name] for name in keys)
        result.setdefault(key, []).append(row)
    return result


def subject_aggregate(
    values: list[dict[str, str]],
    keys: tuple[str, ...],
    field: str,
    reducer: Callable[[np.ndarray], float],
) -> dict[tuple[str, ...], list[float]]:
    by_subject = grouped(values, ("subject",) + keys)
    result: dict[tuple[str, ...], list[float]] = {}
    for key, selected in by_subject.items():
        value = reducer(np.asarray([float(row[field]) for row in selected], dtype=np.float64))
        result.setdefault(key[1:], []).append(float(value))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects-file", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    subjects = [line.strip() for line in args.subjects_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if config.get("protocol") != PROTOCOL or not subjects:
        raise SystemExit("invalid cortical-restriction summary inputs")
    reports = []
    sensitivity: list[dict[str, str]] = []
    false: list[dict[str, str]] = []
    for subject in subjects:
        directory = args.input_root / subject
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("ok") is not True or report.get("subject") != subject or report.get("protocol") != PROTOCOL:
            raise ValueError(f"invalid subject report: {subject}")
        for name, declared in report["outputs"].items():
            path = directory / name
            if not path.is_file() or path.stat().st_size != declared["bytes"] or sha256(path) != declared["sha256"]:
                raise ValueError(f"subject output hash mismatch: {subject}/{name}")
        reports.append(report)
        sensitivity.extend(rows(directory / "sensitivity.csv"))
        false.extend(rows(directory / "false_attribution.csv"))
    if sorted({row["subject"] for row in sensitivity}) != subjects:
        raise ValueError("sensitivity cohort is incomplete")
    if any(row["primary_ladder"].lower() != "true" for row in sensitivity if row["restriction"] in PRIMARY):
        raise ValueError("primary restriction flag is inconsistent")
    replicates = int(config["population_summary"]["bootstrap_replicates"])
    master_seed = int(config["population_summary"]["bootstrap_seed"])

    primary_sensitivity = [row for row in sensitivity if row["restriction"] in PRIMARY]
    overview_values = subject_aggregate(
        primary_sensitivity,
        ("montage", "restriction"),
        "sensitivity_fraction",
        lambda value: float(np.median(value)),
    )
    overview = []
    for key in sorted(overview_values):
        montage, restriction = key
        summary = bootstrap_median(
            overview_values[key], replicates, stable_seed(master_seed, "sensitivity", *key)
        )
        overview.append({"montage": montage, "restriction": restriction, **summary})

    condition_values = subject_aggregate(
        primary_sensitivity,
        ("montage", "restriction", "support_id", "regime"),
        "sensitivity_fraction",
        lambda value: float(value[0]),
    )
    conditions = []
    for key in sorted(condition_values):
        montage, restriction, support_id, regime = key
        summary = bootstrap_median(
            condition_values[key], replicates, stable_seed(master_seed, "condition", *key)
        )
        conditions.append(
            {
                "montage": montage,
                "restriction": restriction,
                "support_id": support_id,
                "regime": regime,
                **summary,
            }
        )

    primary_false = [row for row in false if row["restriction"] in PRIMARY]
    false_keys = ("montage", "restriction", "probe_domain", "probe_family")
    false_median_values = subject_aggregate(
        primary_false, false_keys, "median", lambda value: float(np.median(value))
    )
    false_p95_values = subject_aggregate(
        primary_false, false_keys, "p95", lambda value: float(np.median(value))
    )
    false_max_values = subject_aggregate(
        primary_false, false_keys, "maximum", lambda value: float(np.max(value))
    )
    false_rate_values = subject_aggregate(
        primary_false,
        false_keys,
        "fraction_above_threshold",
        lambda value: float(np.mean(value)),
    )
    false_summary = []
    for key in sorted(false_median_values):
        montage, restriction, domain, family = key
        median_summary = bootstrap_median(
            false_median_values[key], replicates, stable_seed(master_seed, "false_median", *key)
        )
        false_summary.append(
            {
                "montage": montage,
                "restriction": restriction,
                "probe_domain": domain,
                "probe_family": family,
                "subject_median_false_attribution": median_summary["median"],
                "subject_median_ci95_low": median_summary["median_ci95_low"],
                "subject_median_ci95_high": median_summary["median_ci95_high"],
                "population_median_subject_p95": float(np.median(false_p95_values[key])),
                "population_median_subject_maximum": float(np.median(false_max_values[key])),
                "population_maximum": float(np.max(false_max_values[key])),
                "population_median_fraction_above_0_10": float(np.median(false_rate_values[key])),
                "subjects": len(false_median_values[key]),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    overview_path = args.output / "population_sensitivity_overview.csv"
    conditions_path = args.output / "population_sensitivity_conditions.csv"
    false_path = args.output / "population_false_attribution.csv"
    write_csv(overview_path, overview)
    write_csv(conditions_path, conditions)
    write_csv(false_path, false_summary)
    unrestricted = [row for row in overview if row["restriction"] == "unrestricted"]
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(
            len(reports) == len(subjects)
            and all(report["ok"] for report in reports)
            and max(float(row["maximum"]) for row in unrestricted)
            <= float(config["gates"]["unrestricted_maximum_residual"])
        ),
        "subjects": subjects,
        "subject_count": len(subjects),
        "sensitivity_row_count": len(sensitivity),
        "false_attribution_row_count": len(false),
        "primary_restrictions": list(PRIMARY),
        "montages": ["clinical8", "full339"],
        "population_sensitivity_overview": overview,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (overview_path, conditions_path, false_path)
        },
        "subject_report_sha256": {
            subject: sha256(args.input_root / subject / "report.json") for subject in subjects
        },
        "physiological_inference_authorized": False,
        "shared_storage_touched": False,
    }
    report_path = args.output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("population_sensitivity_overview", "subject_report_sha256")}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
