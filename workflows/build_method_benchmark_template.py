#!/usr/bin/env python3
"""Build one auditable cortical/hippocampal factor dictionary for method-benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from anatomical.hippunfold_ensembles import (  # noqa: E402
    build_source_supports,
    oriented_intrinsic_coordinates,
    support_identifier,
)
from empirical.working_memory import atomic_json  # noqa: E402
from workflows.build_cortical_restriction_subject import phase_factor  # noqa: E402
from simulation.cortical_restriction import farthest_anchors, smooth_topographies  # noqa: E402
from theory.recoverability import helmert_reference  # noqa: E402


PROTOCOL = "benchmark/method-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(names) - set(archive.files))
        if missing:
            raise ValueError(f"missing arrays in {path}: {missing}")
        return {name: np.asarray(archive[name]) for name in names}


def montage_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [row["name"] for row in csv.DictReader(stream, delimiter="\t")]


def build(
    subject: str,
    config: dict[str, Any],
    hippocampal_root: Path,
    cortical_root: Path,
    output: Path,
) -> dict[str, Any]:
    hippocampal_dir = hippocampal_root / subject
    cortical_dir = cortical_root / subject
    paths = {
        "hippocampal": hippocampal_dir / "hippunfold-hippocampal-fixed-leadfield.npy",
        "hippocampal_metadata": hippocampal_dir / "hippunfold-hippocampal-source-metadata.npz",
        "hippocampal_report": hippocampal_dir / "report.json",
        "cortical": cortical_dir / "cortical-fixed-leadfield.npy",
        "cortical_metadata": cortical_dir / "cortical-source-metadata.npz",
        "cortical_report": cortical_dir / "report.json",
        "montage": cortical_dir / "registered-montage.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing method-benchmark template inputs: {missing}")
    for label in ("hippocampal_report", "cortical_report"):
        report = json.loads(paths[label].read_text(encoding="utf-8"))
        if report.get("ok") is not True:
            raise ValueError(f"upstream report failed: {paths[label]}")
    h_metadata = load_npz(
        paths["hippocampal_metadata"],
        (
            "positions_m",
            "area_weights_m2",
            "longitudinal_coordinate",
            "proximal_distal_coordinate",
            "hemisphere_code",
        ),
    )
    c_metadata = load_npz(
        paths["cortical_metadata"],
        ("positions_m", "area_weights_m2", "hemisphere_code"),
    )
    hippocampal = np.load(paths["hippocampal"], mmap_mode="r", allow_pickle=False)
    cortical = np.load(paths["cortical"], mmap_mode="r", allow_pickle=False)
    names = montage_names(paths["montage"])
    channels = list(config["scalp_channels"])
    if hippocampal.shape[0] != len(names) or cortical.shape[0] != len(names):
        raise ValueError("lead field and montage dimensions disagree")
    if any(channel not in names for channel in channels):
        raise ValueError("frozen scalp channel is absent from the registered montage")
    indices = np.asarray([names.index(channel) for channel in channels], dtype=np.int64)
    reference = helmert_reference(len(indices))
    h_leadfield = np.asarray(reference @ hippocampal[indices], dtype=np.float64)
    c_leadfield = np.asarray(reference @ cortical[indices], dtype=np.float64)

    geometry = config["source_dictionary"]
    anchors = farthest_anchors(
        c_metadata["positions_m"],
        c_metadata["hemisphere_code"],
        c_metadata["area_weights_m2"],
        int(geometry["cortical_anchors_per_hemisphere"]),
    )
    cortical_factors = smooth_topographies(
        c_leadfield,
        c_metadata["positions_m"],
        c_metadata["hemisphere_code"],
        c_metadata["area_weights_m2"],
        anchors,
        float(geometry["cortical_smoothing_width_m"]),
    )
    cortical_factor_area_fractions = []
    cortical_total_area = float(np.sum(c_metadata["area_weights_m2"]))
    width = float(geometry["cortical_smoothing_width_m"])
    for anchor in anchors:
        same_hemisphere = c_metadata["hemisphere_code"] == c_metadata["hemisphere_code"][anchor]
        squared_distance = np.sum(
            (c_metadata["positions_m"] - c_metadata["positions_m"][anchor]) ** 2,
            axis=1,
        )
        effective_area = float(
            np.sum(
                np.exp(-0.5 * squared_distance / (width * width))
                * c_metadata["area_weights_m2"]
                * same_hemisphere
            )
        )
        cortical_factor_area_fractions.append(effective_area / cortical_total_area)
    cortical_factor_area_fractions_array = np.asarray(
        cortical_factor_area_fractions, dtype=np.float64
    )
    cortical_factors *= cortical_factor_area_fractions_array[None, :]
    cortical_names = [
        f"cortex_{'lh' if int(c_metadata['hemisphere_code'][index]) == -1 else 'rh'}_anchor_{order:03d}"
        for order, index in enumerate(anchors)
    ]
    cortical_groups = np.asarray(
        [0 if int(c_metadata["hemisphere_code"][index]) == -1 else 1 for index in anchors],
        dtype=np.int64,
    )

    support_settings = geometry["support_construction"]
    anterior, pd, orientation_report = oriented_intrinsic_coordinates(
        h_metadata["positions_m"],
        h_metadata["longitudinal_coordinate"],
        h_metadata["proximal_distal_coordinate"],
        h_metadata["hemisphere_code"],
        h_metadata["area_weights_m2"],
        endpoint_decile=float(support_settings["endpoint_decile"]),
        minimum_endpoint_separation_m=float(support_settings["minimum_endpoint_separation_m"]),
    )
    supports, support_report = build_source_supports(
        anterior,
        pd,
        h_metadata["hemisphere_code"],
        h_metadata["area_weights_m2"],
        hemisphere_families=("left", "right", "bilateral"),
        focal_locations=("anterior", "middle", "posterior"),
        focal_extents=(0.05, 0.10, 0.25, 0.50),
        include_whole_extent=True,
    )
    hippocampal_columns: list[np.ndarray] = []
    hippocampal_names: list[str] = []
    hippocampal_groups: list[int] = []
    support_rows: list[dict[str, Any]] = []
    for support_order, row in enumerate(geometry["hippocampal_supports"]):
        identifier = support_identifier(
            str(row["hemisphere"]), str(row["location"]), float(row["extent"])
        )
        support = supports[identifier]
        for cycles in geometry["wave_cycles"]:
            factor = phase_factor(
                h_leadfield,
                support.indices,
                support.conditional_weights,
                support.full_sheet_area_fraction,
                anterior,
                float(cycles),
            )
            for component in range(factor.shape[1]):
                hippocampal_columns.append(factor[:, component])
                hippocampal_names.append(
                    f"hipp_{identifier}_cycles_{float(cycles):g}_component_{component}"
                )
                hippocampal_groups.append(2 + support_order)
            support_rows.append(
                {
                    "identifier": identifier,
                    "wave_cycles": float(cycles),
                    "factor_rank": factor.shape[1],
                    "source_vertices": len(support.indices),
                    "full_sheet_area_fraction": support.full_sheet_area_fraction,
                }
            )
    hippocampal_factors = np.column_stack(hippocampal_columns)
    leadfield = np.column_stack((cortical_factors, hippocampal_factors))
    family = np.concatenate(
        (
            np.zeros(cortical_factors.shape[1], dtype=np.int8),
            np.ones(hippocampal_factors.shape[1], dtype=np.int8),
        )
    )
    groups = np.concatenate((cortical_groups, np.asarray(hippocampal_groups, dtype=np.int64)))
    column_norms = np.linalg.norm(leadfield, axis=0)
    if np.any(column_norms <= 1e-14 * float(np.max(column_norms))):
        raise ValueError("method-benchmark dictionary contains a negligible factor")
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "source_dictionary.npz"
    temporary = bundle.with_name(f".{bundle.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        protocol=np.asarray(PROTOCOL),
        subject=np.asarray(subject),
        scalp_channels=np.asarray(channels, dtype=str),
        helmert_reference=reference,
        leadfield=leadfield,
        source_names=np.asarray(cortical_names + hippocampal_names, dtype=str),
        source_family=family,
        source_group=groups,
        column_norms=column_norms,
        cortical_anchor_indices=anchors,
        cortical_factor_area_fractions=cortical_factor_area_fractions_array,
        physical_leadfield_preserved=np.asarray(True),
    )
    temporary.replace(bundle)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(
            support_report["all_exactly_nested"]
            and leadfield.shape[0] == len(channels) - 1
            and np.linalg.matrix_rank(leadfield) == len(channels) - 1
            and np.all(np.isfinite(leadfield))
        ),
        "subject": subject,
        "sensor_dimension": leadfield.shape[0],
        "cortical_factor_count": cortical_factors.shape[1],
        "hippocampal_factor_count": hippocampal_factors.shape[1],
        "dictionary_rank": int(np.linalg.matrix_rank(leadfield)),
        "cortical_smoothing_width_m": float(geometry["cortical_smoothing_width_m"]),
        "cortical_factor_area_fraction_minimum": float(
            np.min(cortical_factor_area_fractions_array)
        ),
        "cortical_factor_area_fraction_median": float(
            np.median(cortical_factor_area_fractions_array)
        ),
        "cortical_factor_area_fraction_maximum": float(
            np.max(cortical_factor_area_fractions_array)
        ),
        "supports": support_rows,
        "orientation": orientation_report,
        "support_construction_exact": bool(support_report["all_exactly_nested"]),
        "physical_leadfield_preserved": True,
        "column_norm_minimum": float(np.min(column_norms)),
        "column_norm_median": float(np.median(column_norms)),
        "column_norm_maximum": float(np.max(column_norms)),
        "bundle": {"path": bundle.name, "bytes": bundle.stat().st_size, "sha256": sha256(bundle)},
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "shared_storage_touched": False,
    }
    atomic_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hippocampal-root", type=Path, required=True)
    parser.add_argument("--cortical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise SystemExit("unexpected method-benchmark protocol")
    report = build(
        args.subject,
        config,
        args.hippocampal_root,
        args.cortical_root,
        args.output,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "inputs"}, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
