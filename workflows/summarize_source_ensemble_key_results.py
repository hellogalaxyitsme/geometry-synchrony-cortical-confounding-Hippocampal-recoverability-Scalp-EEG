#!/usr/bin/env python3
"""Extract the manuscript source-ensemble summaries from verified tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "hcp/source-ensembles-v1.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def one(rows: Iterable[dict[str, str]], **conditions: object) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if all(
            (abs(float(row[key]) - float(value)) <= 1e-12)
            if isinstance(value, (float, int))
            else row[key] == str(value)
            for key, value in conditions.items()
        )
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {conditions}, found {len(selected)}")
    return selected[0]


def population_metric(row: dict[str, str], stem: str) -> dict[str, float]:
    return {
        "median": float(row[f"{stem}__median"]),
        "q1": float(row[f"{stem}__q1"]),
        "q3": float(row[f"{stem}__q3"]),
        "minimum": float(row[f"{stem}__minimum"]),
        "maximum": float(row[f"{stem}__maximum"]),
        "bootstrap_median_ci95_low": float(
            row[f"{stem}__bootstrap_median_ci95_low"]
        ),
        "bootstrap_median_ci95_high": float(
            row[f"{stem}__bootstrap_median_ci95_high"]
        ),
    }


def median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot summarize an empty collection")
    return float(statistics.median(materialized))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/source_ensembles_v1_1.json"
    )
    parser.add_argument("--subject-root", required=True)
    parser.add_argument("--population-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    config = load_json(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("protocol mismatch")

    subjects = [
        line.strip()
        for line in (project / str(config["subjects_file"])).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    root = Path(args.subject_root).resolve(strict=True)
    population = Path(args.population_dir).resolve(strict=True)
    deterministic_population = load_csv(
        population / "source_ensembles_deterministic_population_v1_1.csv"
    )
    patch_population = load_csv(population / "source_ensembles_patch_coherence_population_v1_1.csv")
    random_population = load_csv(population / "source_ensembles_random_phase_population_v1_1.csv")
    mismatch_population = load_csv(
        population / "source_ensembles_orientation_mismatch_population_v1_1.csv"
    )
    interference_population = load_csv(
        population / "source_ensembles_bilateral_interference_population_v1_1.csv"
    )

    raw_deterministic = {
        subject: load_csv(root / subject / "deterministic_conditions.csv")
        for subject in subjects
    }
    raw_interference = {
        subject: load_csv(root / subject / "bilateral_interference.csv")
        for subject in subjects
    }
    raw_mismatch = {
        subject: load_csv(root / subject / "orientation_mismatch_conditions.csv")
        for subject in subjects
    }

    whole_zero: dict[str, Any] = {}
    for hemisphere in ("left", "right", "bilateral"):
        row = one(
            deterministic_population,
            hemisphere=hemisphere,
            location="whole",
            target_extent=1.0,
            regime="zero_phase",
        )
        values = [
            float(
                one(
                    raw_deterministic[subject],
                    hemisphere=hemisphere,
                    location="whole",
                    target_extent=1.0,
                    regime="zero_phase",
                )["support_normalized_retained_power_ratio"]
            )
            for subject in subjects
        ]
        whole_zero[hemisphere] = {
            **population_metric(row, "support_normalized_retained_power_ratio"),
            "subjects_at_or_below_1_percent": sum(value <= 0.01 for value in values),
            "subjects_at_or_below_5_percent": sum(value <= 0.05 for value in values),
            "subjects_at_or_below_10_percent": sum(value <= 0.10 for value in values),
        }

    bilateral_whole_by_subject = {
        subject: float(
            one(
                rows,
                hemisphere="bilateral",
                location="whole",
                target_extent=1.0,
                regime="zero_phase",
            )["whole_sheet_incoherent_power_ratio"]
        )
        for subject, rows in raw_deterministic.items()
    }
    focal_zero: list[dict[str, Any]] = []
    for extent in (0.05, 0.10, 0.25, 0.50):
        for location in ("anterior", "middle", "posterior"):
            row = one(
                deterministic_population,
                hemisphere="bilateral",
                location=location,
                target_extent=extent,
                regime="zero_phase",
            )
            paired_ratios = []
            for subject, rows in raw_deterministic.items():
                focal = float(
                    one(
                        rows,
                        hemisphere="bilateral",
                        location=location,
                        target_extent=extent,
                        regime="zero_phase",
                    )["whole_sheet_incoherent_power_ratio"]
                )
                paired_ratios.append(focal / bilateral_whole_by_subject[subject])
            focal_zero.append(
                {
                    "location": location,
                    "target_extent": extent,
                    "support_normalized_retained_power": population_metric(
                        row, "support_normalized_retained_power_ratio"
                    ),
                    "whole_sheet_power": population_metric(
                        row, "whole_sheet_incoherent_power_ratio"
                    ),
                    "paired_focal_to_whole_power_ratio_median": median(paired_ratios),
                    "subjects_focal_power_exceeds_whole": sum(
                        value > 1.0 for value in paired_ratios
                    ),
                }
            )

    phase_ladder = []
    for cycles in (0.0, 0.25, 0.50, 1.0, 2.0):
        regime = "zero_phase" if cycles == 0.0 else f"wave_{cycles:g}_cycles"
        row = one(
            deterministic_population,
            hemisphere="bilateral",
            location="whole",
            target_extent=1.0,
            regime=regime,
        )
        phase_ladder.append(
            {
                "wave_cycles": cycles,
                "support_normalized_retained_power": population_metric(
                    row, "support_normalized_retained_power_ratio"
                ),
            }
        )

    patch_ladder = []
    for width in (0.05, 0.10, 0.25, 0.50, 1.0):
        row = one(
            patch_population,
            hemisphere="bilateral",
            location="whole",
            target_extent=1.0,
            intrinsic_bin_width=width,
        )
        patch_ladder.append(
            {
                "intrinsic_bin_width": width,
                "support_normalized_retained_power": population_metric(
                    row, "support_normalized_retained_power_ratio"
                ),
                "coherent_patch_count_median": float(row["coherent_patch_count__median"]),
            }
        )

    random_ladder = []
    for length in (0.05, 0.10, 0.25, 0.50):
        row = one(
            random_population,
            hemisphere="bilateral",
            location="whole",
            target_extent=1.0,
            intrinsic_correlation_length=length,
        )
        random_ladder.append(
            {
                "intrinsic_correlation_length": length,
                "subject_median_support_normalized_retained_power": population_metric(
                    row, "support_normalized_retained_power_ratio"
                ),
                "replicates_per_subject": int(config["random_phase"]["replicates"]),
            }
        )

    interference_row = one(
        interference_population,
        location="whole",
        target_extent=1.0,
        regime="zero_phase",
    )
    gains = [
        float(
            one(
                raw_interference[subject],
                location="whole",
                target_extent=1.0,
                regime="zero_phase",
            )["bilateral_interference_gain"]
        )
        for subject in subjects
    ]
    interference = {
        "gain": population_metric(interference_row, "bilateral_interference_gain"),
        "constructive_subjects_gain_above_one": sum(value > 1.0 for value in gains),
        "destructive_subjects_gain_below_one": sum(value < 1.0 for value in gains),
        "neutral_subjects_gain_equal_one": sum(value == 1.0 for value in gains),
    }

    mismatch_supports = [
        ("whole", 1.0),
        ("anterior", 0.25),
        ("middle", 0.25),
        ("posterior", 0.25),
    ]
    orientation: list[dict[str, Any]] = []
    for location, extent in mismatch_supports:
        angle_rows = []
        trajectories: dict[str, list[float]] = {subject: [] for subject in subjects}
        for angle in (15.0, 30.0, 60.0, 90.0):
            row = one(
                mismatch_population,
                hemisphere="bilateral",
                location=location,
                target_extent=extent,
                regime="zero_phase",
                mismatch_angle_degrees=angle,
            )
            angle_rows.append(
                {
                    "angle_degrees": angle,
                    "projection_efficiency": population_metric(
                        row, "whitened_truth_projection_efficiency"
                    ),
                }
            )
            for subject in subjects:
                trajectories[subject].append(
                    float(
                        one(
                            raw_mismatch[subject],
                            hemisphere="bilateral",
                            location=location,
                            target_extent=extent,
                            regime="zero_phase",
                            mismatch_angle_degrees=angle,
                        )["whitened_truth_projection_efficiency"]
                    )
                )
        orientation.append(
            {
                "location": location,
                "target_extent": extent,
                "angles": angle_rows,
                "subjects_with_monotone_nonincreasing_efficiency": sum(
                    all(a >= b for a, b in zip(values, values[1:]))
                    for values in trajectories.values()
                ),
                "subjects_with_monotone_nondecreasing_efficiency": sum(
                    all(a <= b for a, b in zip(values, values[1:]))
                    for values in trajectories.values()
                ),
            }
        )

    output = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "descriptive numerical geometry; no physiological inference",
        "subject_count": len(subjects),
        "whole_zero_phase_by_hemisphere": whole_zero,
        "bilateral_focal_zero_phase": focal_zero,
        "whole_bilateral_phase_ladder": phase_ladder,
        "whole_bilateral_patch_coherence_ladder": patch_ladder,
        "whole_bilateral_random_phase_ladder": random_ladder,
        "whole_zero_phase_bilateral_interference": interference,
        "bilateral_zero_phase_orientation_sensitivity": orientation,
        "physiological_inference_authorized": False,
        "shared_storage_touched": False,
    }
    atomic_json(Path(args.output), output)
    print(
        json.dumps(
            {
                "subject_count": output["subject_count"],
                "whole_zero_phase_by_hemisphere": whole_zero,
                "whole_zero_phase_bilateral_interference": interference,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

