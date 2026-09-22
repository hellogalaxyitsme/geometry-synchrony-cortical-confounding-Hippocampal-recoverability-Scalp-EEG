#!/usr/bin/env python3
"""Empirical sensor-scale calibration and continuous false-alarm evaluation.

Two complementary arms are kept separate:

* physical microvolt and single-event SNR calibration in the independent
  Koessler/Ternisien mesial-event release; and
* patient-held-out continuous-background scanning in ds004752 using the
  already frozen temporal-transfer theta-state detector and best-channel baseline.

No depth waveform is read in either scalp solution.  ds004752 depth data enter
only through the frozen temporal-transfer binary labels.
"""

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
sys.path.insert(0, str(PROJECT))
from empirical.calibration import (  # noqa: E402
    bids_selected_units,
    edf_selected_physical_dimensions,
    empirical_upper_threshold,
    false_alarms_per_hour,
    patient_equal_summary,
)
from empirical.working_memory import (  # noqa: E402
    continuous_bandpasses,
    fit_shrinkage_lda,
    predict_linear,
    read_edf_selected,
    read_tsv,
    roc_auc,
    window_log_power,
)
from empirical.temporal_transfer import choose_transfer_shrinkage  # noqa: E402
from empirical.mesial_events import fit_matched_filter, window  # noqa: E402


PROTOCOL = "calibration/continuous-false-positive-v1.1"
METHODS = ("shrinkage_lda", "best_channel")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(array: np.ndarray) -> object:
    return array.item() if array.ndim == 0 else array


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def p10_interval(center: int, milliseconds: float, sampling: float, half_ms: float) -> slice:
    offset = int(round(milliseconds * sampling / 1000.0))
    half = int(round(half_ms * sampling / 1000.0))
    return window(center + offset, half)


def mesial_physical_calibration(config: dict[str, Any], epoch_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    p10 = config["mesial_event_calibration"]
    signal = p10["signal"]
    rows: list[dict[str, object]] = []
    for specification in p10["networks"]:
        identifier = str(specification["id"])
        patient = str(specification["patient"])
        with np.load(epoch_root / f"{identifier}.npz", allow_pickle=False) as archive:
            epochs = np.asarray(archive["eeg_epochs_uv"], dtype=np.float64)
            sampling = float(archive["sampling_frequency_hz"])
            center = int(archive["t0_sample"])
        split = int(np.floor(float(signal["training_fraction"]) * len(epochs)))
        training, testing = epochs[:split], epochs[split:]
        event = p10_interval(center, 0.0, sampling, float(signal["event_half_width_ms"]))
        baseline = p10_interval(
            center,
            float(signal["baseline_center_ms"]),
            sampling,
            float(signal["event_half_width_ms"]),
        )
        covariance_intervals = (
            slice(0, center - int(0.2 * sampling)),
            slice(center + int(0.2 * sampling), epochs.shape[2]),
        )
        matched, _ = fit_matched_filter(
            training,
            event,
            covariance_intervals,
            float(signal["covariance_shrinkage"]),
        )
        projected = np.einsum("ect,c->et", testing, matched, optimize=True)
        background = np.concatenate(
            (projected[:, covariance_intervals[0]], projected[:, covariance_intervals[1]]), axis=1
        )
        event_peak = np.max(np.abs(projected[:, event]), axis=1)
        background_sd = np.std(background, axis=1, ddof=0)
        event_energy = np.mean(projected[:, event] ** 2, axis=1)
        baseline_energy = np.mean(projected[:, baseline] ** 2, axis=1)
        channel_background_rms = np.sqrt(
            np.mean(
                np.concatenate(
                    (testing[:, :, covariance_intervals[0]], testing[:, :, covariance_intervals[1]]),
                    axis=2,
                )
                ** 2,
                axis=(1, 2),
            )
        )
        for index in range(len(testing)):
            rows.append(
                {
                    "patient": patient,
                    "network": identifier,
                    "test_event_index": index,
                    "event_peak_uv": float(event_peak[index]),
                    "background_sd_uv": float(background_sd[index]),
                    "single_event_snr_db": float(
                        20.0
                        * np.log10(
                            event_peak[index]
                            / max(float(background_sd[index]), np.finfo(float).tiny)
                        )
                    ),
                    "event_to_baseline_energy_ratio": float(
                        event_energy[index]
                        / max(float(baseline_energy[index]), np.finfo(float).tiny)
                    ),
                    "scalp_background_rms_uv": float(channel_background_rms[index]),
                }
            )
    patient_ratios = []
    patient_snrs = []
    patient_rms = []
    for patient in sorted({str(row["patient"]) for row in rows}):
        selected = [row for row in rows if row["patient"] == patient]
        patient_ratios.append(float(np.median([float(row["event_to_baseline_energy_ratio"]) for row in selected])))
        patient_snrs.append(float(np.median([float(row["single_event_snr_db"]) for row in selected])))
        patient_rms.append(float(np.median([float(row["scalp_background_rms_uv"]) for row in selected])))
    summary = {
        "patient_count": len(patient_ratios),
        "network_count": len({str(row["network"]) for row in rows}),
        "test_event_count": len(rows),
        "single_event_snr_db_patient_equal": patient_equal_summary(patient_snrs),
        "event_to_baseline_energy_ratio_patient_equal": patient_equal_summary(patient_ratios),
        "scalp_background_rms_uv_patient_equal": patient_equal_summary(patient_rms),
        "injection_calibration_energy_ratio": float(np.median(patient_ratios)),
        "interpretation": "physical sensor-scale calibration for curated mesial epileptic events; not a source-current or spontaneous-theta calibration",
    }
    return rows, summary


def load_temporal_transfer_sessions(root: Path) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    sessions: list[dict[str, object]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "features": [],
        "labels": [],
        "patients": [],
        "sessions": [],
        "blocks": [],
        "trials": [],
    }
    for path in sorted((root / "sessions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            item = {name: scalar(np.asarray(archive[name])) for name in archive.files}
        keep = np.asarray(item["labels"], dtype=np.int64) >= 0
        features = np.asarray(item["early_calibrated_scalp_features"], dtype=np.float64)
        labels = np.asarray(item["labels"], dtype=np.int64)
        blocks = np.asarray(item["block_ids"], dtype=np.int64)
        trials = np.asarray(item["trial_number"], dtype=np.int64)
        arrays["features"].append(features[keep])
        arrays["labels"].append(labels[keep])
        arrays["blocks"].append(blocks[keep])
        arrays["trials"].append(trials[keep])
        arrays["patients"].append(np.repeat(str(item["subject"]), int(np.count_nonzero(keep))))
        arrays["sessions"].append(np.repeat(str(item["stem"]), int(np.count_nonzero(keep))))
        sessions.append(
            {
                "path": path,
                "stem": str(item["stem"]),
                "subject": str(item["subject"]),
                "session": str(item["session"]),
                "calibration_center": np.asarray(item["calibration_center"], dtype=np.float64),
                "calibration_scale": np.asarray(item["calibration_scale"], dtype=np.float64),
                "block_ids_all": np.asarray(item["block_ids"], dtype=np.int64),
                "labels_all": np.asarray(item["labels"], dtype=np.int64),
                "original_event_row": np.asarray(item["original_event_row"], dtype=np.int64),
            }
        )
    return sessions, {name: np.concatenate(value) for name, value in arrays.items()}


def extract_continuous_background(
    config: dict[str, Any],
    dataset_root: Path,
    audit: dict[str, Any],
    p4_sessions: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    continuous = config["continuous_false_positive"]
    channels = list(continuous["scalp_channels"])
    audit_lookup = {str(row["stem"]): row for row in audit["sessions"]}
    window_seconds = float(continuous["window_seconds"])
    background_rows: list[dict[str, object]] = []
    physical_rows: list[dict[str, object]] = []
    feature_blocks: list[np.ndarray] = []
    patient_blocks: list[np.ndarray] = []
    session_blocks: list[np.ndarray] = []
    start_blocks: list[np.ndarray] = []
    excluded_sessions: list[str] = []
    for order, session in enumerate(p4_sessions, start=1):
        stem = str(session["stem"])
        row = audit_lookup[stem]
        eeg_path = dataset_root / str(row["eeg_edf"])
        dimensions = edf_selected_physical_dimensions(str(eeg_path), channels)
        bids_units = bids_selected_units(
            str(eeg_path.with_name(f"{stem}_channels.tsv")), channels
        )
        normalized_units = {
            value.lower().replace("μ", "u").replace("µ", "u")
            for value in bids_units.values()
        }
        if normalized_units != {"uv"}:
            raise ValueError(f"{stem} is not uniformly declared in microvolts: {bids_units}")
        if any(value.strip() for value in dimensions.values()):
            normalized_edf = {
                value.lower().replace("μ", "u").replace("µ", "u")
                for value in dimensions.values()
            }
            if normalized_edf != {"uv"}:
                raise ValueError(f"{stem} EDF/BIDS unit conflict: {dimensions} versus {bids_units}")
        eeg, sampling = read_edf_selected(eeg_path, channels)
        eeg -= eeg.mean(axis=0, keepdims=True)
        theta, broadband = continuous_bandpasses(
            eeg,
            sampling,
            (continuous["theta_band_hz"], continuous["broadband_hz"]),
            int(continuous["butterworth_order"]),
        )
        event_path = eeg_path.with_name(f"{stem}_events.tsv")
        events = read_tsv(event_path)
        original = np.asarray(session["original_event_row"], dtype=np.int64)
        block_all = np.asarray(session["block_ids_all"], dtype=np.int64)
        late_original = original[block_all == 2]
        if len(late_original) == 0:
            raise ValueError(f"{stem} has no late trials")
        maintenance_left, maintenance_right = (
            float(value) for value in continuous["maintenance_window_seconds"]
        )
        labels = np.asarray(session["labels_all"], dtype=np.int64)
        late_labeled = np.flatnonzero((block_all == 2) & (labels >= 0))
        for local in late_labeled:
            zero = int(events[int(original[local])]["begSample"]) - 1
            left = zero + int(round(maintenance_left * sampling))
            right = zero + int(round(maintenance_right * sampling))
            physical_rows.append(
                {
                    "patient": str(session["subject"]),
                    "session": stem,
                    "label": int(labels[local]),
                    "scalp_rms_uv": float(np.sqrt(np.mean(eeg[:, left:right] ** 2))),
                }
            )
        # ds004752 is an epoched release: 8-second trials tile the EDF without
        # inter-trial background. Scan every frozen, non-overlapping late-trial
        # control interval instead of inventing unavailable continuous rest.
        # The resulting rate is exposure-normalized, not an ambulatory or
        # free-running clinical false-alarm rate.
        declared_intervals = [
            tuple(float(value) for value in pair)
            for pair in continuous["scan_intervals_seconds"]
        ]
        if any(abs((right - left) - window_seconds) > 1e-12 for left, right in declared_intervals):
            raise ValueError("scan intervals must match the declared window duration")
        starts_list: list[int] = []
        stops_list: list[int] = []
        for event_index in late_original:
            zero = int(events[int(event_index)]["begSample"]) - 1
            for left_seconds, right_seconds in declared_intervals:
                starts_list.append(zero + int(round(left_seconds * sampling)))
                stops_list.append(zero + int(round(right_seconds * sampling)))
        starts_all = np.asarray(starts_list, dtype=np.int64)
        stops_all = np.asarray(stops_list, dtype=np.int64)
        if (
            len(starts_all) == 0
            or np.any(starts_all < 0)
            or np.any(stops_all > eeg.shape[1])
            or np.any(stops_all <= starts_all)
        ):
            raise ValueError(f"{stem} has invalid available-exposure scan windows")
        candidate_raw = np.stack(
            [eeg[:, left:right] for left, right in zip(starts_all, stops_all)]
        )
        peak = np.max(np.abs(candidate_raw), axis=(1, 2))
        keep = peak <= float(continuous["artifact_peak_uv"])
        if not np.any(keep):
            excluded_sessions.append(stem)
            print(
                json.dumps(
                    {
                        "available_exposure_session": stem,
                        "index": order,
                        "windows": 0,
                        "exclusion": "no_window_below_frozen_artifact_ceiling",
                    }
                ),
                flush=True,
            )
            continue
        starts = starts_all[keep]
        stops = stops_all[keep]
        raw_features = window_log_power(theta, starts, stops) - window_log_power(
            broadband, starts, stops
        )
        center = np.asarray(session["calibration_center"], dtype=np.float64)
        scale = np.asarray(session["calibration_scale"], dtype=np.float64)
        calibrated = (raw_features - center) / scale
        rms = np.sqrt(np.mean(candidate_raw[keep] ** 2, axis=(1, 2)))
        feature_blocks.append(calibrated)
        patient_blocks.append(np.repeat(str(session["subject"]), len(calibrated)))
        session_blocks.append(np.repeat(stem, len(calibrated)))
        start_blocks.append(starts.astype(np.int64))
        for local in range(len(calibrated)):
            background_rows.append(
                {
                    "patient": str(session["subject"]),
                    "session": stem,
                    "start_sample": int(starts[local]),
                    "start_seconds": float(starts[local] / sampling),
                    "window_seconds": window_seconds,
                    "exposure_type": "late_trial_probe_response_control",
                    "bids_units": "microvolts",
                    "edf_dimension_field": "blank" if not any(dimensions.values()) else "microvolts",
                    "scalp_rms_uv": float(rms[local]),
                    **{f"feature_{index}": float(calibrated[local, index]) for index in range(calibrated.shape[1])},
                }
            )
        print(json.dumps({"available_exposure_session": stem, "index": order, "windows": len(calibrated)}), flush=True)
    return background_rows, physical_rows, {
        "features": np.concatenate(feature_blocks),
        "patients": np.concatenate(patient_blocks),
        "sessions": np.concatenate(session_blocks),
        "starts": np.concatenate(start_blocks),
        "excluded_sessions": np.asarray(excluded_sessions, dtype=str),
    }


def select_best_channel(
    features: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    blocks: np.ndarray,
    training_patients: list[str],
) -> tuple[int, float]:
    candidates = []
    for channel in range(features.shape[1]):
        aucs = []
        for heldout in training_patients:
            train = np.isin(patients, [value for value in training_patients if value != heldout]) & np.isin(blocks, (0, 1))
            test = (patients == heldout) & (blocks == 2)
            sign = 1.0 if features[train & (labels == 1), channel].mean() >= features[train & (labels == 0), channel].mean() else -1.0
            aucs.append(roc_auc(labels[test], sign * features[test, channel]))
        candidates.append((float(np.mean(aucs)), channel))
    _, selected = max(candidates, key=lambda value: (value[0], -value[1]))
    train = np.isin(patients, training_patients) & np.isin(blocks, (0, 1))
    sign = 1.0 if features[train & (labels == 1), selected].mean() >= features[train & (labels == 0), selected].mean() else -1.0
    return int(selected), float(sign)


def continuous_patient_heldout(
    config: dict[str, Any],
    events: dict[str, np.ndarray],
    background: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    settings = config["continuous_false_positive"]
    features = np.asarray(events["features"], dtype=np.float64)
    labels = np.asarray(events["labels"], dtype=np.int64)
    patients = np.asarray(events["patients"], dtype=str)
    blocks = np.asarray(events["blocks"], dtype=np.int64)
    sessions = np.asarray(events["sessions"], dtype=str)
    bg_features = np.asarray(background["features"], dtype=np.float64)
    bg_patients = np.asarray(background["patients"], dtype=str)
    bg_sessions = np.asarray(background["sessions"], dtype=str)
    bg_starts = np.asarray(background["starts"], dtype=np.int64)
    patient_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    background_rows: list[dict[str, object]] = []
    window_seconds = float(settings["window_seconds"])
    evaluable_patients = sorted(np.intersect1d(np.unique(patients), np.unique(bg_patients)))
    for heldout in evaluable_patients:
        training_patients = [value for value in sorted(np.unique(patients)) if value != heldout]
        event_train = np.isin(patients, training_patients) & np.isin(blocks, (0, 1))
        event_test = (patients == heldout) & (blocks == 2)
        background_train = np.isin(bg_patients, training_patients)
        background_test = bg_patients == heldout
        shrinkage, _ = choose_transfer_shrinkage(
            features,
            labels,
            patients,
            blocks,
            training_patients,
            (0, 1),
            (2,),
            settings["shrinkage_grid"],
        )
        model = fit_shrinkage_lda(features[event_train], labels[event_train], shrinkage)
        lda_event = predict_linear(features[event_test], model)
        lda_bg_train = predict_linear(bg_features[background_train], model)
        lda_bg_test = predict_linear(bg_features[background_test], model)
        channel, sign = select_best_channel(features, labels, patients, blocks, training_patients)
        channel_event = sign * features[event_test, channel]
        channel_bg_train = sign * bg_features[background_train, channel]
        channel_bg_test = sign * bg_features[background_test, channel]
        for method, event_score, train_score, test_score, parameter in (
            ("shrinkage_lda", lda_event, lda_bg_train, lda_bg_test, shrinkage),
            ("best_channel", channel_event, channel_bg_train, channel_bg_test, channel),
        ):
            threshold = empirical_upper_threshold(
                train_score,
                float(settings["target_false_alarms_per_hour"]),
                window_seconds,
            )
            false_count, hours, false_rate = false_alarms_per_hour(
                test_score, threshold, window_seconds
            )
            truth = labels[event_test]
            sensitivity = float(np.mean(event_score[truth == 1] > threshold))
            low_false = float(np.mean(event_score[truth == 0] > threshold))
            patient_rows.append(
                {
                    "patient": heldout,
                    "method": method,
                    "parameter": parameter,
                    "threshold": threshold,
                    "training_background_windows": int(len(train_score)),
                    "heldout_background_windows": int(len(test_score)),
                    "heldout_background_hours": hours,
                    "heldout_false_alarms": false_count,
                    "heldout_false_alarms_per_hour": false_rate,
                    "late_high_theta_sensitivity": sensitivity,
                    "late_low_theta_false_positive_fraction": low_false,
                    "late_high_vs_low_auc": roc_auc(truth, event_score),
                    "late_labeled_events": int(len(truth)),
                }
            )
            event_indices = np.flatnonzero(event_test)
            for local, index in enumerate(event_indices):
                event_rows.append(
                    {
                        "patient": heldout,
                        "session": str(sessions[index]),
                        "method": method,
                        "label": int(labels[index]),
                        "score": float(event_score[local]),
                        "threshold": threshold,
                        "above_threshold": bool(event_score[local] > threshold),
                    }
                )
            background_indices = np.flatnonzero(background_test)
            for local, index in enumerate(background_indices):
                background_rows.append(
                    {
                        "patient": heldout,
                        "session": str(bg_sessions[index]),
                        "start_sample": int(bg_starts[index]),
                        "method": method,
                        "score": float(test_score[local]),
                        "threshold": threshold,
                        "above_threshold": bool(test_score[local] > threshold),
                        "window_seconds": window_seconds,
                    }
                )
    summary: dict[str, object] = {}
    for method in METHODS:
        selected = [row for row in patient_rows if row["method"] == method]
        summary[method] = {
            "patient_count": len(selected),
            "false_alarms_per_hour_patient_equal": patient_equal_summary(
                [float(row["heldout_false_alarms_per_hour"]) for row in selected]
            ),
            "high_theta_sensitivity_patient_equal": patient_equal_summary(
                [float(row["late_high_theta_sensitivity"]) for row in selected]
            ),
            "low_theta_false_positive_fraction_patient_equal": patient_equal_summary(
                [float(row["late_low_theta_false_positive_fraction"]) for row in selected]
            ),
            "auc_patient_equal": patient_equal_summary(
                [float(row["late_high_vs_low_auc"]) for row in selected]
            ),
        }
    return patient_rows, event_rows, background_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--p10-epoch-root", type=Path, required=True)
    parser.add_argument("--p4-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected empirical calibration protocol")
    if args.output.exists():
        raise FileExistsError(args.output)
    mesial_rows, mesial_summary = mesial_physical_calibration(config, args.p10_epoch_root)
    p4_sessions, event_data = load_temporal_transfer_sessions(args.p4_root)
    audit = json.loads(args.input_audit.read_text(encoding="utf-8"))
    continuous_raw_rows, event_physical_rows, background_data = extract_continuous_background(
        config, args.dataset_root, audit, p4_sessions
    )
    patient_rows, event_rows, background_rows, continuous_summary = continuous_patient_heldout(
        config, event_data, background_data
    )
    expected_task_control_patients = int(
        config["continuous_false_positive"]["expected_patients_with_valid_exposure"]
    )
    observed_task_control_patients = sorted({str(row["patient"]) for row in patient_rows})
    all_event_patients = sorted(np.unique(np.asarray(event_data["patients"], dtype=str)))
    excluded_task_control_patients = sorted(set(all_event_patients) - set(observed_task_control_patients))
    args.output.mkdir(parents=True)
    write_csv(args.output / "mesial_event_physical_calibration.csv", mesial_rows)
    write_csv(args.output / "continuous_background_features.csv", continuous_raw_rows)
    write_csv(args.output / "late_event_physical_calibration.csv", event_physical_rows)
    write_csv(args.output / "continuous_patient_metrics.csv", patient_rows)
    write_csv(args.output / "continuous_event_scores.csv", event_rows)
    write_csv(args.output / "continuous_background_scores.csv", background_rows)
    late_high_rms = []
    late_low_rms = []
    for patient in sorted({str(row["patient"]) for row in event_physical_rows}):
        selected = [row for row in event_physical_rows if row["patient"] == patient]
        late_high_rms.append(float(np.median([float(row["scalp_rms_uv"]) for row in selected if int(row["label"]) == 1])))
        late_low_rms.append(float(np.median([float(row["scalp_rms_uv"]) for row in selected if int(row["label"]) == 0])))
    report = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "ok": (
            len(observed_task_control_patients) == expected_task_control_patients
            and len(patient_rows) == 2 * expected_task_control_patients
            and len(event_rows) > 0
            and len(background_rows) > 0
        ),
        "mesial_event_physical_calibration": mesial_summary,
        "continuous_false_positive": continuous_summary,
        "ds004752_physical_calibration": {
            "late_high_theta_scalp_rms_uv_patient_equal": patient_equal_summary(late_high_rms),
            "late_low_theta_scalp_rms_uv_patient_equal": patient_equal_summary(late_low_rms),
            "patient_count": len(late_high_rms),
        },
        "available_task_control_windows": len(continuous_raw_rows),
        "available_task_control_hours": float(
            len(continuous_raw_rows)
            * float(config["continuous_false_positive"]["window_seconds"])
            / 3600.0
        ),
        "task_control_patient_count": len(observed_task_control_patients),
        "task_control_patients": observed_task_control_patients,
        "excluded_task_control_patients": excluded_task_control_patients,
        "excluded_task_control_sessions": sorted(
            np.asarray(background_data["excluded_sessions"], dtype=str).tolist()
        ),
        "artifact_ceiling_uv": float(config["continuous_false_positive"]["artifact_peak_uv"]),
        "patient_specific_anatomy_used": False,
        "source_current_calibration_authorized": False,
        "physical_sensor_units": "microvolts as declared by every selected BIDS channels.tsv; blank EDF dimension fields were logged",
        "recording_scope": "exposure-normalized scan of frozen late-trial probe/response controls in an epoched release; not a free-running continuous clinical EEG estimate",
        "depth_waveform_used": False,
        "random_epoch_split_used": False,
        "shared_storage_touched": False,
        "outputs": {},
    }
    for path in args.output.glob("*.csv"):
        report["outputs"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
