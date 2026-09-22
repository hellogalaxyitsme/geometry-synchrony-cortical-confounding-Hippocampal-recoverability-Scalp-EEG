#!/usr/bin/env python3
"""Template-ensemble recoverability predictions for the nine mesial networks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from empirical.mesial_events import normalize_channel  # noqa: E402
from simulation.cortical_restriction import farthest_anchors, smooth_topographies  # noqa: E402
from simulation.montage_information import load_montage, montage_information  # noqa: E402

PROTOCOL = "mesial-events/external-validation-v1"


def npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing: raise ValueError(f"{path} lacks {missing}")
        return {name: np.asarray(archive[name]) for name in required}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epoch-root", type=Path, required=True); parser.add_argument("--hippocampal-root", type=Path, required=True)
    parser.add_argument("--cortical-root", type=Path, required=True); parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); prediction = config["prediction"]
    if config.get("protocol") != PROTOCOL: raise ValueError("unexpected mesial-event protocol")
    network_channels = {}
    for row in config["networks"]:
        with np.load(args.epoch_root / f"{row['id']}.npz", allow_pickle=False) as archive:
            network_channels[str(row["id"])] = [normalize_channel(value) for value in archive["channel_names"].astype(str)]
    rows: list[dict[str, object]] = []
    for subject in args.subjects:
        h_dir = args.hippocampal_root / subject; c_dir = args.cortical_root / subject
        h = np.load(h_dir / "hippunfold-hippocampal-fixed-leadfield.npy", mmap_mode="r", allow_pickle=False)
        c = np.load(c_dir / "cortical-fixed-leadfield.npy", mmap_mode="r", allow_pickle=False)
        hm = npz(h_dir / "hippunfold-hippocampal-source-metadata.npz", ("area_weights_m2", "longitudinal_coordinate"))
        cm = npz(c_dir / "cortical-source-metadata.npz", ("positions_m", "area_weights_m2", "hemisphere_code"))
        names, _ = load_montage(c_dir / "registered-montage.tsv")
        normalized_names = [normalize_channel(name) for name in names]
        if len(normalized_names) != len(set(normalized_names)): raise ValueError("forward montage aliases collide")
        lookup = {name: index for index, name in enumerate(normalized_names)}
        anchors = farthest_anchors(cm["positions_m"], cm["hemisphere_code"], cm["area_weights_m2"], int(prediction["cortical_anchors_per_hemisphere"]))
        width = float(prediction["cortical_smoothing_width_m"])
        c_factors = smooth_topographies(np.asarray(c), cm["positions_m"], cm["hemisphere_code"], cm["area_weights_m2"], anchors, width)
        total_area = float(np.sum(cm["area_weights_m2"])); fractions = []
        for anchor in anchors:
            same = cm["hemisphere_code"] == cm["hemisphere_code"][anchor]
            squared = np.sum((cm["positions_m"] - cm["positions_m"][anchor]) ** 2, axis=1)
            fractions.append(float(np.sum(np.exp(-0.5 * squared / width**2) * cm["area_weights_m2"] * same)) / total_area)
        c_factors *= np.asarray(fractions)[None, :]
        c_covariance = c_factors @ c_factors.T / c_factors.shape[1]
        weights = hm["area_weights_m2"] / float(np.sum(hm["area_weights_m2"]))
        coherent = np.asarray(h) @ weights
        phase = 2.0 * np.pi * hm["longitudinal_coordinate"]
        wave = np.column_stack((np.asarray(h) @ (weights * np.cos(phase)), np.asarray(h) @ (weights * np.sin(phase))))
        covariances = {"bilateral_coherent": np.outer(coherent, coherent), "bilateral_wave1": wave @ wave.T / 2.0}
        dimension = len(names) - 1; c_variance = float(np.trace(c_covariance)) / dimension
        h_variance = float(np.trace(covariances["bilateral_coherent"])) / dimension
        signal_scale = float(prediction["reference_signal_to_cortical_variance"]) * c_variance / h_variance
        for network, channels in sorted(network_channels.items()):
            missing = sorted(set(channels) - set(lookup))
            if missing: raise ValueError(f"{network} channels absent from HCP montage: {missing}")
            indices = [lookup[channel] for channel in channels]
            for ensemble in prediction["ensembles"]:
                information = montage_information(
                    covariances[str(ensemble)], c_covariance, indices, signal_scale,
                    float(prediction["cortical_variance_ratio"]),
                    float(prediction["sensor_noise_variance_fraction"]) * c_variance,
                )
                rows.append({"template_subject": subject, "network": network, "ensemble": ensemble,
                             "scalp_channels": len(channels), "information_bits": information})
    args.output.mkdir(parents=True, exist_ok=False); template_path = args.output / "template_predictions.csv"; write_csv(template_path, rows)
    population = []
    for network in sorted(network_channels):
        for ensemble in prediction["ensembles"]:
            values = np.asarray([float(row["information_bits"]) for row in rows if row["network"] == network and row["ensemble"] == ensemble])
            population.append({"network": network, "ensemble": ensemble, "scalp_channels": len(network_channels[network]),
                               "template_count": len(values), "information_median_bits": float(np.median(values)),
                               "information_minimum_bits": float(np.min(values)), "information_maximum_bits": float(np.max(values))})
    population_path = args.output / "population_predictions.csv"; write_csv(population_path, population)
    report = {"schema_version": 1, "protocol": PROTOCOL, "ok": len(rows) == len(args.subjects) * 9 * 2,
              "template_count": len(args.subjects), "network_count": 9, "ensemble_count": 2,
              "prediction_scope": "template-ensemble geometry prediction; no patient anatomy is available",
              "absolute_bits_physiological": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
