#!/usr/bin/env python3
"""Build the temporal-transfer patient-held-out contiguous-block experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.working_memory import (
    atomic_json,
    continuous_bandpasses,
    fit_shrinkage_lda,
    predict_linear,
    read_edf_selected,
    read_tsv,
    repeat_string,
    robust_zscore,
    roc_auc,
    sha256,
    stratified_tail_labels,
    window_log_power,
)
from empirical.temporal_transfer import (
    apply_calibration,
    contiguous_block_ids,
    fit_early_calibration,
    outer_transfer_best_channel,
    outer_transfer_lda,
    shift_labels_within_session_blocks,
    summarize_transfer,
)


PROTOCOL = "openneuro-ds004752/patient-blocked-v1"
BLOCK_NAMES = ("early", "middle", "late")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def scalar(value: np.ndarray) -> object:
    return value.item() if value.ndim == 0 else value


def session_paths(dataset: Path, row: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    eeg = dataset / row["eeg_edf"]
    ieeg = dataset / row["ieeg_edf"]
    return (
        eeg,
        ieeg,
        eeg.with_name(f"{row['stem']}_events.tsv"),
        ieeg.with_name(f"{row['stem']}_events.tsv"),
    )


def extract_session(
    dataset: Path,
    audit_row: dict[str, Any],
    config: dict[str, Any],
    working_memory_root: Path,
    output: Path,
) -> dict[str, Any]:
    eeg_path, ieeg_path, eeg_events_path, ieeg_events_path = session_paths(dataset, audit_row)
    eeg_events = read_tsv(eeg_events_path)
    ieeg_events = read_tsv(ieeg_events_path)
    if len(eeg_events) != len(ieeg_events):
        raise ValueError("EEG/iEEG event count changed after input audit")
    selection = config["trial_selection"]
    signal = config["signal"]
    channels = list(config["primary_scalp_channels"])
    pairs = [tuple(pair) for pair in audit_row["hippocampal_bipolar_pairs"]]
    depth_names = sorted({name for pair in pairs for name in pair})
    eeg_data, eeg_frequency = read_edf_selected(eeg_path, channels)
    ieeg_data, ieeg_frequency = read_edf_selected(ieeg_path, depth_names)
    eeg_data -= eeg_data.mean(axis=0, keepdims=True)
    depth_index = {name: index for index, name in enumerate(depth_names)}
    bipolar = np.vstack(
        [ieeg_data[depth_index[left]] - ieeg_data[depth_index[right]] for left, right in pairs]
    )
    if abs(eeg_frequency - float(audit_row["eeg_sampling_frequency"])) > 1e-9:
        raise ValueError("EEG sampling frequency changed after audit")
    if abs(ieeg_frequency - float(audit_row["ieeg_sampling_frequency"])) > 1e-9:
        raise ValueError("iEEG sampling frequency changed after audit")
    eligible_indices = [
        index
        for index, trial in enumerate(eeg_events)
        if int(trial["Artifact"]) == int(selection["artifact_code"])
        and (not selection["correct_only"] or int(trial["Correct"]) == 1)
    ]
    eligible_eeg = [eeg_events[index] for index in eligible_indices]
    eligible_ieeg = [ieeg_events[index] for index in eligible_indices]
    block_ids = contiguous_block_ids(len(eligible_indices), int(config["contiguous_blocks"]["count"]))
    window_start, window_stop = (float(value) for value in selection["maintenance_window_seconds"])

    def windows(events: list[dict[str, str]], frequency: float) -> tuple[np.ndarray, np.ndarray]:
        starts = np.asarray(
            [int(row["begSample"]) - 1 + round(window_start * frequency) for row in events],
            dtype=np.int64,
        )
        stops = np.asarray(
            [int(row["begSample"]) - 1 + round(window_stop * frequency) for row in events],
            dtype=np.int64,
        )
        return starts, stops

    eeg_starts, eeg_stops = windows(eligible_eeg, eeg_frequency)
    ieeg_starts, ieeg_stops = windows(eligible_ieeg, ieeg_frequency)
    order = int(signal["butterworth_order"])
    eeg_theta, eeg_broadband = continuous_bandpasses(
        eeg_data, eeg_frequency, (signal["theta_band_hz"], signal["broadband_hz"]), order
    )
    (depth_theta,) = continuous_bandpasses(
        bipolar, ieeg_frequency, (signal["theta_band_hz"],), order
    )
    raw_scalp = window_log_power(eeg_theta, eeg_starts, eeg_stops) - window_log_power(
        eeg_broadband, eeg_starts, eeg_stops
    )
    calibration_center, calibration_scale = fit_early_calibration(raw_scalp, block_ids)
    calibrated_scalp = apply_calibration(raw_scalp, calibration_center, calibration_scale)
    depth_log_power = window_log_power(depth_theta, ieeg_starts, ieeg_stops)
    depth_z = robust_zscore(depth_log_power, axis=0)
    hippocampal_score = np.median(depth_z, axis=1)
    set_size = np.asarray([int(row["SetSize"]) for row in eligible_eeg], dtype=np.int64)
    labels = stratified_tail_labels(
        hippocampal_score,
        set_size,
        float(selection["lower_label_quantile"]),
        float(selection["upper_label_quantile"]),
        int(selection["minimum_trials_per_session_set_size_stratum"]),
    )
    trial_number = np.asarray([int(row["nTrial"]) for row in eligible_eeg], dtype=np.int64)
    working_memory_path = working_memory_root / "sessions" / f"{audit_row['stem']}.npz"
    with np.load(working_memory_path, allow_pickle=False) as archive:
        p3_trial = np.asarray(archive["trial_number"], dtype=np.int64)
        p3_set_size = np.asarray(archive["set_size"], dtype=np.int64)
        p3_labels = np.asarray(archive["labels"], dtype=np.int64)
        p3_score = np.asarray(archive["hippocampal_score"], dtype=np.float64)
    working_memory_identity = bool(
        np.array_equal(trial_number, p3_trial)
        and np.array_equal(set_size, p3_set_size)
        and np.array_equal(labels, p3_labels)
        and np.allclose(hippocampal_score, p3_score, rtol=0.0, atol=1e-12)
    )
    if not working_memory_identity:
        raise ValueError("temporal-transfer depth labels do not reproduce frozen working-memory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        protocol=np.asarray(PROTOCOL),
        subject=np.asarray(audit_row["subject"]),
        session=np.asarray(audit_row["session"]),
        stem=np.asarray(audit_row["stem"]),
        scalp_channels=np.asarray(channels, dtype=str),
        raw_scalp_features=raw_scalp,
        early_calibrated_scalp_features=calibrated_scalp,
        calibration_center=calibration_center,
        calibration_scale=calibration_scale,
        hippocampal_score=hippocampal_score,
        labels=labels,
        block_ids=block_ids,
        set_size=set_size,
        trial_number=trial_number,
        original_event_row=np.asarray(eligible_indices, dtype=np.int64),
        eeg_sampling_frequency=np.asarray(eeg_frequency),
        ieeg_sampling_frequency=np.asarray(ieeg_frequency),
        working_memory_label_identity=np.asarray(working_memory_identity),
    )
    temporary.replace(output)
    block_counts = {
        BLOCK_NAMES[block]: {
            "eligible": int(np.count_nonzero(block_ids == block)),
            "low": int(np.count_nonzero((block_ids == block) & (labels == 0))),
            "high": int(np.count_nonzero((block_ids == block) & (labels == 1))),
        }
        for block in range(3)
    }
    return {
        "subject": audit_row["subject"],
        "session": audit_row["session"],
        "stem": audit_row["stem"],
        "output": output.name,
        "eligible_trials": len(labels),
        "labeled_trials": int(np.count_nonzero(labels >= 0)),
        "block_counts": block_counts,
        "working_memory_label_identity": working_memory_identity,
        "finite": bool(
            np.all(np.isfinite(raw_scalp))
            and np.all(np.isfinite(calibrated_scalp))
            and np.all(np.isfinite(hippocampal_score))
        ),
        "npz_sha256": sha256(output),
    }


def extract_all(
    dataset: Path,
    audit: dict[str, Any],
    config: dict[str, Any],
    working_memory_root: Path,
    output_root: Path,
    subjects: list[str],
) -> dict[str, Any]:
    selected = [
        row
        for row in audit["sessions"]
        if row["subject"] in subjects and row["working_memory_eligible"]
    ]
    reports = []
    for index, row in enumerate(selected, start=1):
        report = extract_session(
            dataset,
            row,
            config,
            working_memory_root,
            output_root / "sessions" / f"{row['stem']}.npz",
        )
        reports.append(report)
        print(f"[{index}/{len(selected)}] {row['stem']}", flush=True)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(reports)
        and all(row["finite"] and row["working_memory_label_identity"] for row in reports),
        "subjects": subjects,
        "subject_count": len(subjects),
        "session_count": len(reports),
        "eligible_trials": sum(row["eligible_trials"] for row in reports),
        "labeled_trials": sum(row["labeled_trials"] for row in reports),
        "sessions": reports,
        "versions": {"mne": mne.__version__, "numpy": np.__version__},
        "participants_tsv_read": False,
        "shared_storage_touched": False,
    }
    atomic_json(output_root / "extraction_manifest.json", manifest)
    return manifest


def load_sessions(root: Path) -> list[dict[str, Any]]:
    sessions = []
    for path in sorted((root / "sessions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            item = {name: scalar(np.asarray(archive[name])) for name in archive.files}
        if str(item["protocol"]) != PROTOCOL:
            raise ValueError(f"wrong protocol: {path}")
        item["path"] = path
        sessions.append(item)
    if not sessions:
        raise ValueError("no temporal-transfer sessions found")
    return sessions


def concatenate_labeled(sessions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {
        "features": [],
        "labels": [],
        "patients": [],
        "sessions": [],
        "blocks": [],
        "trials": [],
        "set_size": [],
    }
    for row in sessions:
        labels = np.asarray(row["labels"], dtype=np.int64)
        keep = labels >= 0
        count = int(np.count_nonzero(keep))
        patient = str(row["subject"])
        session_id = f"{patient}_{row['session']}"
        result["features"].append(
            np.asarray(row["early_calibrated_scalp_features"], dtype=np.float64)[keep]
        )
        result["labels"].append(labels[keep])
        result["patients"].append(repeat_string(patient, count))
        result["sessions"].append(repeat_string(session_id, count))
        result["blocks"].append(np.asarray(row["block_ids"], dtype=np.int64)[keep])
        result["trials"].append(np.asarray(row["trial_number"], dtype=np.int64)[keep])
        result["set_size"].append(np.asarray(row["set_size"], dtype=np.int64)[keep])
    return {key: np.vstack(value) if key == "features" else np.concatenate(value) for key, value in result.items()}


def run_condition(
    data: dict[str, np.ndarray],
    config: dict[str, Any],
    name: str,
    training_blocks: Sequence[int],
    test_blocks: Sequence[int],
    seed: int,
    include_baseline: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    validation = config["validation"]
    scores, evaluated, folds = outer_transfer_lda(
        data["features"],
        data["labels"],
        data["patients"],
        data["blocks"],
        training_blocks,
        test_blocks,
        validation["shrinkage_grid"],
    )
    report = summarize_transfer(
        data["labels"],
        data["patients"],
        scores,
        evaluated,
        folds,
        int(validation["bootstrap_replicates"]),
        seed,
    )
    report.update(
        {
            "name": name,
            "training_blocks": [BLOCK_NAMES[value] for value in training_blocks],
            "test_blocks": [BLOCK_NAMES[value] for value in test_blocks],
        }
    )
    baseline_scores = None
    if include_baseline:
        baseline_scores, baseline_evaluated, baseline_folds = outer_transfer_best_channel(
            data["features"],
            data["labels"],
            data["patients"],
            data["blocks"],
            training_blocks,
            test_blocks,
        )
        if not np.array_equal(evaluated, baseline_evaluated):
            raise AssertionError("LDA and baseline evaluated different rows")
        report["best_single_channel"] = summarize_transfer(
            data["labels"],
            data["patients"],
            baseline_scores,
            baseline_evaluated,
            baseline_folds,
            int(validation["bootstrap_replicates"]),
            seed + 1,
        )
    return report, scores, baseline_scores


def fixed_primary_scores(
    features: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    blocks: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    evaluated = blocks == 2
    scores = np.full(len(labels), np.nan)
    cohort = np.unique(patients)
    for heldout in cohort:
        train = (patients != heldout) & np.isin(blocks, [0, 1])
        test = (patients == heldout) & (blocks == 2)
        model = fit_shrinkage_lda(features[train], labels[train], shrinkage)
        scores[test] = predict_linear(features[test], model)
    return scores, evaluated


def circular_null(data: dict[str, np.ndarray], primary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validation = config["validation"]
    shrinkage = float(np.median([row["chosen_shrinkage"] for row in primary["folds"]]))
    observed_scores, evaluated = fixed_primary_scores(
        data["features"], data["labels"], data["patients"], data["blocks"], shrinkage
    )
    patients = np.unique(data["patients"])
    observed = float(
        np.mean(
            [
                roc_auc(
                    data["labels"][evaluated & (data["patients"] == patient)],
                    observed_scores[evaluated & (data["patients"] == patient)],
                )
                for patient in patients
            ]
        )
    )
    generator = np.random.default_rng(int(validation["master_seed"]) + 700)
    replicates = int(validation["circular_shift_replicates"])
    null = np.empty(replicates)
    for index in range(replicates):
        shifted = shift_labels_within_session_blocks(
            data["labels"], data["sessions"], data["blocks"], generator
        )
        scores, mask = fixed_primary_scores(
            data["features"], shifted, data["patients"], data["blocks"], shrinkage
        )
        null[index] = float(
            np.mean(
                [
                    roc_auc(
                        shifted[mask & (data["patients"] == patient)],
                        scores[mask & (data["patients"] == patient)],
                    )
                    for patient in patients
                ]
            )
        )
    return {
        "fixed_shrinkage": shrinkage,
        "observed_macro_patient_auc": observed,
        "replicates": replicates,
        "null_mean": float(np.mean(null)),
        "null_standard_deviation": float(np.std(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "one_sided_p": float((1 + np.count_nonzero(null >= observed)) / (replicates + 1)),
        "shift_unit": "within_session_and_contiguous_block",
    }


def write_primary_events(
    path: Path,
    data: dict[str, np.ndarray],
    scores: np.ndarray,
    baseline: np.ndarray,
) -> None:
    evaluated = data["blocks"] == 2
    rows = []
    for index in np.flatnonzero(evaluated):
        rows.append(
            {
                "patient": str(data["patients"][index]),
                "session": str(data["sessions"][index]),
                "trial_number": int(data["trials"][index]),
                "set_size": int(data["set_size"][index]),
                "block": BLOCK_NAMES[int(data["blocks"][index])],
                "label": int(data["labels"][index]),
                "heldout_lda_score": float(scores[index]),
                "heldout_best_channel_score": float(baseline[index]),
            }
        )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze_full(sessions: list[dict[str, Any]], config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    data = concatenate_labeled(sessions)
    validation = config["validation"]
    minimum = int(config["contiguous_blocks"]["minimum_labeled_events_per_patient_test_partition"])
    for patient in np.unique(data["patients"]):
        for block in range(3):
            selected = (data["patients"] == patient) & (data["blocks"] == block)
            if np.count_nonzero(selected) < minimum or set(np.unique(data["labels"][selected])) != {0, 1}:
                raise ValueError(f"patient/block lacks the frozen minimum or both classes: {patient}/{block}")
    primary, primary_scores, baseline_scores = run_condition(
        data,
        config,
        "prospective_early_middle_to_late",
        (0, 1),
        (2,),
        int(validation["master_seed"]),
        include_baseline=True,
    )
    if baseline_scores is None:
        raise AssertionError("primary baseline was not computed")
    transfer = {}
    condition_index = 0
    for training_block in range(3):
        for test_block in range(3):
            key = f"{BLOCK_NAMES[training_block]}_to_{BLOCK_NAMES[test_block]}"
            transfer[key], _, _ = run_condition(
                data,
                config,
                key,
                (training_block,),
                (test_block,),
                int(validation["master_seed"]) + 20 + condition_index,
            )
            condition_index += 1
    all_blocks, _, _ = run_condition(
        data,
        config,
        "all_blocks_patient_held_out_reference",
        (0, 1, 2),
        (0, 1, 2),
        int(validation["master_seed"]) + 50,
    )
    control = circular_null(data, primary, config)
    event_path = output_root / "prospective_late_heldout_scores.csv"
    write_primary_events(event_path, data, primary_scores, baseline_scores)
    block_counts = {
        BLOCK_NAMES[block]: {
            "events": int(np.count_nonzero(data["blocks"] == block)),
            "low": int(np.count_nonzero((data["blocks"] == block) & (data["labels"] == 0))),
            "high": int(np.count_nonzero((data["blocks"] == block) & (data["labels"] == 1))),
        }
        for block in range(3)
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": True,
        "patient_count": len(np.unique(data["patients"])),
        "session_count": len(sessions),
        "labeled_event_count": len(data["labels"]),
        "block_counts": block_counts,
        "primary_prospective_late": primary,
        "cross_block_transfer": transfer,
        "matched_block_conditions": {
            name: transfer[f"{name}_to_{name}"] for name in BLOCK_NAMES
        },
        "all_blocks_reference": all_blocks,
        "circular_shift_negative_control": control,
        "primary_event_scores_csv": event_path.name,
        "primary_event_scores_sha256": sha256(event_path),
        "random_epoch_split_implemented": False,
        "heldout_ieeg_waveform_entered_scalp_model": False,
        "future_scalp_samples_entered_late_block_calibration": False,
        "participants_tsv_read": False,
        "shared_storage_touched": False,
    }
    atomic_json(output_root / "analysis_report.json", report)
    return report


def validate_subset(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "all_arrays_finite": all(
            np.all(np.isfinite(np.asarray(row["early_calibrated_scalp_features"])))
            for row in sessions
        ),
        "all_working_memory_labels_identical": all(
            bool(row["working_memory_label_identity"]) for row in sessions
        ),
        "all_sessions_have_three_ordered_contiguous_blocks": all(
            np.array_equal(
                np.asarray(row["block_ids"], dtype=np.int64),
                contiguous_block_ids(len(np.asarray(row["block_ids"])), 3),
            )
            for row in sessions
        ),
        "early_calibration_scales_positive": all(
            np.all(np.asarray(row["calibration_scale"], dtype=np.float64) > 0.0)
            for row in sessions
        ),
        "all_eight_features_vary": bool(
            np.all(
                np.std(
                    np.vstack(
                        [np.asarray(row["early_calibrated_scalp_features"]) for row in sessions]
                    ),
                    axis=0,
                )
                > 0.05
            )
        ),
    }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "mode": "technical_subset",
        "ok": all(checks.values()),
        "subject_count": len({str(row["subject"]) for row in sessions}),
        "session_count": len(sessions),
        "checks": checks,
        "analysis_run": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--working_memory-root", type=Path, required=True)
    parser.add_argument("--working_memory-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("subset", "full"), required=True)
    parser.add_argument("--stage", choices=("extract", "analyze", "all"), default="all")
    args = parser.parse_args()
    config = load_json(args.config)
    audit = load_json(args.audit)
    if config.get("protocol") != PROTOCOL or not audit.get("ok"):
        raise SystemExit("temporal-transfer protocol or input audit failed")
    for relative, expected in config.get("implementation_sha256", {}).items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"frozen implementation hash mismatch: {relative}")
    eligible = list(config["eligible_subjects"])
    if eligible != list(audit["working_memory_eligible_subjects"]):
        raise SystemExit("eligible cohort differs from input audit")
    subjects = list(config["minimum_subset_subjects"]) if args.mode == "subset" else eligible
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("extract", "all"):
        manifest = extract_all(
            args.dataset, audit, config, args.working_memory_root, args.output, subjects
        )
        if not manifest["ok"]:
            return 2
    sessions = load_sessions(args.output)
    if sorted({str(row["subject"]) for row in sessions}) != sorted(subjects):
        raise SystemExit("extracted subject cohort is incomplete or contaminated")
    if args.stage in ("analyze", "all"):
        if args.mode == "subset":
            gate = validate_subset(sessions)
            atomic_json(args.output / "subset_gate.json", gate)
            print(json.dumps(gate, indent=2, sort_keys=True))
            return 0 if gate["ok"] else 2
        report = analyze_full(sessions, config, args.output)
        print(
            json.dumps(
                {
                    "ok": report["ok"],
                    "patient_count": report["patient_count"],
                    "session_count": report["session_count"],
                    "block_counts": report["block_counts"],
                    "primary_macro_patient_auc": report["primary_prospective_late"][
                        "macro_patient_auc"
                    ],
                    "primary_baseline_macro_patient_auc": report[
                        "primary_prospective_late"
                    ]["best_single_channel"]["macro_patient_auc"],
                    "circular_shift_negative_control": report[
                        "circular_shift_negative_control"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
