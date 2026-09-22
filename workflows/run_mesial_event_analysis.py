#!/usr/bin/env python3
"""Scalp-only event detection and predicted-to-observed association for mesial-event."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from empirical.mesial_events import (  # noqa: E402
    fastica_auc, fit_matched_filter, paired_auc, spatial_scores, spearman, window,
)

PROTOCOL = "mesial-events/external-validation-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def stable_seed(master: int, text: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{master}|{text}".encode()).digest()[:8], "little")


def interval(center: int, milliseconds: float, sampling: float, half_ms: float) -> slice:
    return window(center + int(round(milliseconds * sampling / 1000.0)), int(round(half_ms * sampling / 1000.0)))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epoch-root", type=Path, required=True); parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); signal = config["signal"]
    predictions = read_csv(args.prediction_root / "population_predictions.csv")
    prediction_lookup = {(row["network"], row["ensemble"]): float(row["information_median_bits"]) for row in predictions}
    network_rows = []; averaging_rows = []
    for network_spec in config["networks"]:
        identifier = str(network_spec["id"]); patient = str(network_spec["patient"])
        with np.load(args.epoch_root / f"{identifier}.npz", allow_pickle=False) as archive:
            epochs = np.asarray(archive["eeg_epochs_uv"], dtype=np.float64); sampling = float(archive["sampling_frequency_hz"]); center = int(archive["t0_sample"])
        split = int(np.floor(float(signal["training_fraction"]) * len(epochs)))
        if split < 20 or len(epochs) - split < 15: raise ValueError(f"insufficient contiguous split: {identifier}")
        training = epochs[:split]; testing = epochs[split:]
        event = interval(center, 0.0, sampling, float(signal["event_half_width_ms"]))
        baseline = interval(center, float(signal["baseline_center_ms"]), sampling, float(signal["event_half_width_ms"]))
        control = interval(center, float(signal["control_center_ms"]), sampling, float(signal["event_half_width_ms"]))
        baseline_covariance_intervals = (slice(0, center - int(0.2 * sampling)), slice(center + int(0.2 * sampling), epochs.shape[2]))
        matched, diagnostics = fit_matched_filter(training, event, baseline_covariance_intervals, float(signal["covariance_shrinkage"]))
        matched_event = spatial_scores(testing, matched, event); matched_base = spatial_scores(testing, matched, baseline)
        matched_auc = paired_auc(matched_event, matched_base)
        time_control_auc = paired_auc(spatial_scores(testing, matched, baseline), spatial_scores(testing, matched, control))
        train_event_channels = np.max(np.abs(training[:, :, event]), axis=2); train_base_channels = np.max(np.abs(training[:, :, baseline]), axis=2)
        channel_training_auc = [paired_auc(train_event_channels[:, column], train_base_channels[:, column]) for column in range(training.shape[1])]
        best = int(np.argmax(np.maximum(channel_training_auc, 1.0 - np.asarray(channel_training_auc))))
        sign = 1.0 if channel_training_auc[best] >= 0.5 else -1.0
        best_auc = paired_auc(sign * np.max(np.abs(testing[:, best, event]), axis=1), sign * np.max(np.abs(testing[:, best, baseline]), axis=1))
        ica_auc, ica_converged, ica_iterations = fastica_auc(training, testing, event, baseline, stable_seed(int(config["master_seed"]), identifier))
        projected_average = np.einsum("ect,c->et", testing, matched).mean(axis=0)
        event_peak = float(np.max(np.abs(projected_average[event])))
        baseline_values = np.concatenate((projected_average[: center - int(0.2 * sampling)], projected_average[center + int(0.2 * sampling):]))
        average_snr_db = float(20.0 * np.log10(event_peak / max(float(np.std(baseline_values)), np.finfo(float).tiny)))
        for requested in config["averaging_ladder"]:
            count = min(int(requested), len(testing)); groups = [testing[start:start + count] for start in range(0, len(testing) - count + 1, count)]
            snrs = []
            for group in groups:
                trace = np.einsum("ct,c->t", group.mean(axis=0), matched)
                peak = float(np.max(np.abs(trace[event]))); noise = np.concatenate((trace[: center - int(0.2 * sampling)], trace[center + int(0.2 * sampling):]))
                snrs.append(20.0 * np.log10(peak / max(float(np.std(noise)), np.finfo(float).tiny)))
            averaging_rows.append({"patient": patient, "network": identifier, "requested_events": int(requested), "events_per_average": count,
                                   "nonoverlapping_groups": len(groups), "median_snr_db": float(np.median(snrs))})
        network_rows.append({
            "patient": patient, "network": identifier, "scalp_channels": epochs.shape[1], "retained_events": len(epochs),
            "training_events": len(training), "testing_events": len(testing), "matched_filter_auc": matched_auc,
            "best_channel_auc": best_auc, "fastica_bss_auc": ica_auc, "fastica_converged": ica_converged,
            "fastica_iterations": ica_iterations, "time_shift_control_auc": time_control_auc,
            "all_test_events_average_snr_db": average_snr_db,
            "predicted_coherent_information_bits": prediction_lookup[(identifier, "bilateral_coherent")],
            "predicted_wave1_information_bits": prediction_lookup[(identifier, "bilateral_wave1")],
            "matched_filter_training_peak_offset_samples": diagnostics["training_peak_offset_samples"],
        })
    patient_rows = []
    for patient in sorted({row["patient"] for row in network_rows}):
        selected = [row for row in network_rows if row["patient"] == patient]
        patient_rows.append({"patient": patient, "network_count": len(selected),
                             "observed_auc": float(np.median([row["matched_filter_auc"] for row in selected])),
                             "predicted_coherent_bits": float(np.median([row["predicted_coherent_information_bits"] for row in selected])),
                             "predicted_wave1_bits": float(np.median([row["predicted_wave1_information_bits"] for row in selected]))})
    observed = np.asarray([row["observed_auc"] for row in patient_rows]); coherent = np.asarray([row["predicted_coherent_bits"] for row in patient_rows]); wave = np.asarray([row["predicted_wave1_bits"] for row in patient_rows])
    coherent_rho = spearman(coherent, observed); wave_rho = spearman(wave, observed)
    generator = np.random.default_rng(int(config["master_seed"])); boot = []
    for _ in range(int(config["bootstrap_replicates"])):
        indices = generator.integers(0, len(patient_rows), size=len(patient_rows)); value = spearman(coherent[indices], observed[indices])
        if np.isfinite(value): boot.append(value)
    args.output.mkdir(parents=True, exist_ok=False); write_csv(args.output / "network_metrics.csv", network_rows)
    write_csv(args.output / "averaging_ladder.csv", averaging_rows); write_csv(args.output / "patient_metrics.csv", patient_rows)
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "ok": len(network_rows) == 9 and len(patient_rows) == 7,
        "network_count": len(network_rows), "patient_count": len(patient_rows), "retained_event_count": int(sum(row["retained_events"] for row in network_rows)),
        "primary_endpoint": "patient-grouped association between template-ensemble predicted coherent information and held-out-block matched-filter AUC",
        "association": {"coherent_spearman_rho": coherent_rho, "wave1_spearman_rho": wave_rho,
                        "coherent_patient_bootstrap_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "finite_bootstrap_replicates": len(boot)},
        "macro_patient_auc": {method: float(np.mean([row[method] for row in network_rows])) for method in ("matched_filter_auc", "best_channel_auc", "fastica_bss_auc")},
        "depth_waveform_used": False, "random_epoch_split_used": False, "split": "contiguous early 60% training / late 40% testing within network",
        "patient_is_resampling_unit": True, "generalization_limit": "curated mesial epileptic spikes; not spontaneous physiological hippocampal activity",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
