#!/usr/bin/env python3
"""Patient-equal population summaries for montage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROTOCOL = "montage/physical-value-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def seed(master: int, values: tuple[str, ...]) -> int:
    digest = hashlib.sha256((str(master) + "|" + "|".join(values)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def summarize(values: np.ndarray, replicates: int, random_seed: int) -> dict[str, float]:
    generator = np.random.default_rng(random_seed)
    draws = np.median(values[generator.integers(0, len(values), size=(replicates, len(values)))], axis=1)
    return {
        "subject_median": float(np.median(values)), "subject_minimum": float(np.min(values)),
        "subject_maximum": float(np.max(values)), "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
    }


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...], value: str) -> dict[tuple[str, ...], list[float]]:
    result: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        result.setdefault(tuple(row[key] for key in keys), []).append(float(row[value]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True); parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); subjects = sorted(args.subjects)
    metrics = []; increments = []
    for subject in subjects:
        directory = args.subject_root / subject
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if report.get("ok") is not True or report.get("subject") != subject:
            raise ValueError(f"invalid subject output: {subject}")
        metrics.extend(read_csv(directory / "metrics.csv")); increments.extend(read_csv(directory / "increments.csv"))
    args.output.mkdir(parents=True, exist_ok=False)
    replicates = int(config["bootstrap_replicates"]); master = int(config["master_seed"])
    metric_keys = ("ensemble", "montage", "sensors", "cortical_variance_ratio", "noise_variance_fraction")
    metric_summary = []
    for key, values in sorted(grouped(metrics, metric_keys, "information_bits").items()):
        metric_summary.append(dict(zip(metric_keys, key)) | summarize(np.asarray(values), replicates, seed(master, key)))
    increment_keys = ("ensemble", "existing_montage", "augmented_montage", "existing_sensors", "augmented_sensors", "cortical_variance_ratio", "noise_variance_fraction")
    increment_summary = []
    for key, values in sorted(grouped(increments, increment_keys, "conditional_information_bits").items()):
        increment_summary.append(dict(zip(increment_keys, key)) | summarize(np.asarray(values), replicates, seed(master, key)))
    metric_path = args.output / "population_metrics.csv"; increment_path = args.output / "population_increments.csv"
    write_csv(metric_path, metric_summary); write_csv(increment_path, increment_summary)
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "ok": True, "subject_count": len(subjects),
        "subjects": subjects, "subject_metric_rows": len(metrics), "subject_increment_rows": len(increments),
        "population_metric_conditions": len(metric_summary), "population_increment_conditions": len(increment_summary),
        "minimum_nested_population_increment": min(
            float(row["subject_median"]) for row in increment_summary
            if row["augmented_montage"] != "conventional64_plus_inferior"
        ),
        "face_neck_evaluated": False,
        "interpretation": "relative geometry/montage value only; absolute bits are normalization dependent",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
