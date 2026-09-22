#!/usr/bin/env python3
"""Extract and analyze the frozen ds004752 working-memory empirical experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from anatomical.hippunfold_ensembles import oriented_intrinsic_coordinates
from empirical.working_memory import (
    atomic_json,
    bootstrap_macro_auc,
    continuous_bandpasses,
    outer_best_channel,
    outer_patient_lda,
    permutation_spearman,
    read_edf_selected,
    read_tsv,
    repeat_string,
    robust_zscore,
    roc_auc,
    sha256,
    stratified_tail_labels,
    window_log_power,
    fit_shrinkage_lda,
    predict_linear,
)


PROTOCOL = "openneuro-ds004752/empirical-v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def scalar(value: np.ndarray) -> object:
    return value.item() if value.ndim == 0 else value


def session_paths(dataset: Path, row: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    eeg = dataset / row["eeg_edf"]
    ieeg = dataset / row["ieeg_edf"]
    eeg_events = eeg.with_name(f"{row['stem']}_events.tsv")
    ieeg_events = ieeg.with_name(f"{row['stem']}_events.tsv")
    return eeg, ieeg, eeg_events, ieeg_events


def extract_session(
    dataset: Path,
    row: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    eeg_path, ieeg_path, eeg_events_path, ieeg_events_path = session_paths(dataset, row)
    trials = read_tsv(eeg_events_path)
    ieeg_trials = read_tsv(ieeg_events_path)
    if len(trials) != len(ieeg_trials):
        raise ValueError("EEG/iEEG event count changed after audit")
    selection = config["trial_selection"]
    signal = config["signal"]
    channels = list(config["primary_scalp_channels"])
    pairs = [tuple(pair) for pair in row["hippocampal_bipolar_pairs"]]
    if not pairs:
        raise ValueError(f"eligible session has no bipolar pair: {row['stem']}")
    depth_names = sorted({name for pair in pairs for name in pair})

    eeg_data, eeg_frequency = read_edf_selected(eeg_path, channels)
    ieeg_data, ieeg_frequency = read_edf_selected(ieeg_path, depth_names)
    eeg_data -= eeg_data.mean(axis=0, keepdims=True)
    depth_index = {name: index for index, name in enumerate(depth_names)}
    bipolar = np.vstack(
        [ieeg_data[depth_index[left]] - ieeg_data[depth_index[right]] for left, right in pairs]
    )
    if abs(eeg_frequency - float(row["eeg_sampling_frequency"])) > 1e-9:
        raise ValueError("EEG sampling frequency changed after audit")
    if abs(ieeg_frequency - float(row["ieeg_sampling_frequency"])) > 1e-9:
        raise ValueError("iEEG sampling frequency changed after audit")

    eligible_indices = [
        index
        for index, trial in enumerate(trials)
        if int(trial["Artifact"]) == int(selection["artifact_code"])
        and (not selection["correct_only"] or int(trial["Correct"]) == 1)
    ]
    eligible = [trials[index] for index in eligible_indices]
    eligible_ieeg = [ieeg_trials[index] for index in eligible_indices]
    if not eligible:
        raise ValueError("session has no eligible trials")
    window_start, window_stop = (float(value) for value in selection["maintenance_window_seconds"])
    eeg_starts = np.asarray(
        [int(trial["begSample"]) - 1 + round(window_start * eeg_frequency) for trial in eligible],
        dtype=np.int64,
    )
    eeg_stops = np.asarray(
        [int(trial["begSample"]) - 1 + round(window_stop * eeg_frequency) for trial in eligible],
        dtype=np.int64,
    )
    ieeg_starts = np.asarray(
        [
            int(trial["begSample"]) - 1 + round(window_start * ieeg_frequency)
            for trial in eligible_ieeg
        ],
        dtype=np.int64,
    )
    ieeg_stops = np.asarray(
        [
            int(trial["begSample"]) - 1 + round(window_stop * ieeg_frequency)
            for trial in eligible_ieeg
        ],
        dtype=np.int64,
    )
    order = int(signal["butterworth_order"])
    theta_band = signal["theta_band_hz"]
    broadband = signal["broadband_hz"]
    eeg_theta, eeg_broadband = continuous_bandpasses(
        eeg_data, eeg_frequency, (theta_band, broadband), order
    )
    (ieeg_theta,) = continuous_bandpasses(bipolar, ieeg_frequency, (theta_band,), order)
    scalp_features = window_log_power(eeg_theta, eeg_starts, eeg_stops) - window_log_power(
        eeg_broadband, eeg_starts, eeg_stops
    )
    scalp_features = robust_zscore(scalp_features, axis=0)
    depth_log_power = window_log_power(ieeg_theta, ieeg_starts, ieeg_stops)
    depth_z = robust_zscore(depth_log_power, axis=0)
    hippocampal_score = np.median(depth_z, axis=1)
    dominant_pair_index = np.argmax(depth_z, axis=1).astype(np.int64)
    set_size = np.asarray([int(trial["SetSize"]) for trial in eligible], dtype=np.int64)
    labels = stratified_tail_labels(
        hippocampal_score,
        set_size,
        float(selection["lower_label_quantile"]),
        float(selection["upper_label_quantile"]),
        int(selection["minimum_trials_per_session_set_size_stratum"]),
    )
    if set(np.unique(labels[labels >= 0])) != {0, 1}:
        raise ValueError("session did not yield both frozen hippocampal-state classes")

    contact_rows = {contact["name"]: contact for contact in row["hippocampal_contacts"]}
    contact_names = np.asarray([pair[0] for pair in pairs], dtype=str)
    contact_keys = np.asarray(
        [f"{row['subject']}:{name}" for name in contact_names], dtype=str
    )
    contact_x = np.asarray([float(contact_rows[name]["x"]) for name in contact_names])
    contact_y = np.asarray([float(contact_rows[name]["y"]) for name in contact_names])
    contact_z = np.asarray([float(contact_rows[name]["z"]) for name in contact_names])
    hemisphere = np.where(contact_x < 0.0, -1, 1).astype(np.int64)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.stem}.tmp.npz")
    np.savez_compressed(
        temporary_output,
        protocol=np.asarray(PROTOCOL),
        subject=np.asarray(row["subject"]),
        session=np.asarray(row["session"]),
        stem=np.asarray(row["stem"]),
        scalp_channels=np.asarray(channels, dtype=str),
        scalp_features=scalp_features,
        hippocampal_pair_z=depth_z,
        hippocampal_score=hippocampal_score,
        dominant_pair_index=dominant_pair_index,
        labels=labels,
        set_size=set_size,
        trial_number=np.asarray([int(trial["nTrial"]) for trial in eligible], dtype=np.int64),
        original_event_row=np.asarray(eligible_indices, dtype=np.int64),
        pair_left=np.asarray([pair[0] for pair in pairs], dtype=str),
        pair_right=np.asarray([pair[1] for pair in pairs], dtype=str),
        contact_names=contact_names,
        contact_keys=contact_keys,
        contact_x_mni=contact_x,
        contact_y_mni=contact_y,
        contact_z_mni=contact_z,
        hemisphere_code=hemisphere,
        eeg_sampling_frequency=np.asarray(eeg_frequency),
        ieeg_sampling_frequency=np.asarray(ieeg_frequency),
    )
    temporary_output.replace(output)
    return {
        "subject": row["subject"],
        "session": row["session"],
        "stem": row["stem"],
        "output": output.name,
        "eligible_trials": len(eligible),
        "labeled_low_trials": int(np.count_nonzero(labels == 0)),
        "labeled_high_trials": int(np.count_nonzero(labels == 1)),
        "excluded_middle_or_small_stratum_trials": int(np.count_nonzero(labels < 0)),
        "hippocampal_bipolar_pairs": len(pairs),
        "scalp_features": int(scalp_features.shape[1]),
        "finite": bool(
            np.all(np.isfinite(scalp_features))
            and np.all(np.isfinite(depth_z))
            and np.all(np.isfinite(hippocampal_score))
        ),
        "npz_sha256": sha256(output),
    }


def extract_all(
    dataset: Path,
    audit: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    subjects: list[str],
) -> dict[str, Any]:
    session_root = output_root / "sessions"
    selected = [
        row
        for row in audit["sessions"]
        if row["subject"] in subjects and row["working_memory_eligible"]
    ]
    reports = []
    for index, row in enumerate(selected, start=1):
        path = session_root / f"{row['stem']}.npz"
        report = extract_session(dataset, row, config, path)
        reports.append(report)
        print(
            f"[{index}/{len(selected)}] {row['stem']}: "
            f"{report['labeled_low_trials']} low, {report['labeled_high_trials']} high",
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(reports) and all(row["finite"] for row in reports),
        "subjects": subjects,
        "subject_count": len(subjects),
        "session_count": len(reports),
        "eligible_trials": sum(row["eligible_trials"] for row in reports),
        "labeled_low_trials": sum(row["labeled_low_trials"] for row in reports),
        "labeled_high_trials": sum(row["labeled_high_trials"] for row in reports),
        "sessions": reports,
        "versions": {"mne": mne.__version__, "numpy": np.__version__},
        "participants_tsv_read": False,
        "shared_storage_touched": False,
    }
    atomic_json(output_root / "extraction_manifest.json", manifest)
    return manifest


def load_extractions(session_root: Path) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for path in sorted(session_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            item = {name: scalar(np.asarray(archive[name])) for name in archive.files}
        item["path"] = path
        if str(item["protocol"]) != PROTOCOL:
            raise ValueError(f"wrong protocol in {path}")
        sessions.append(item)
    if not sessions:
        raise ValueError("no extracted sessions found")
    return sessions


def contact_catalogue(sessions: list[dict[str, Any]], winsor: list[float]) -> dict[str, Any]:
    contacts: dict[str, dict[str, Any]] = {}
    for session in sessions:
        for index, key in enumerate(session["contact_keys"]):
            record = {
                "contact_key": str(key),
                "subject": str(session["subject"]),
                "name": str(session["contact_names"][index]),
                "x_mni": float(session["contact_x_mni"][index]),
                "y_mni": float(session["contact_y_mni"][index]),
                "z_mni": float(session["contact_z_mni"][index]),
                "hemisphere_code": int(session["hemisphere_code"][index]),
            }
            existing = contacts.get(str(key))
            if existing is not None and existing != record:
                raise ValueError(f"contact metadata changed across sessions: {key}")
            contacts[str(key)] = record
    lower_q, upper_q = (float(value) for value in winsor)
    for hemisphere in (-1, 1):
        subset = [row for row in contacts.values() if row["hemisphere_code"] == hemisphere]
        y = np.asarray([row["y_mni"] for row in subset], dtype=np.float64)
        lower, upper = (float(np.quantile(y, value)) for value in (lower_q, upper_q))
        if upper <= lower:
            raise ValueError("MNI anterior-posterior calibration is degenerate")
        for row in subset:
            row["anterior_coordinate"] = float(
                np.clip((row["y_mni"] - lower) / (upper - lower), 0.0, 1.0)
            )
            row["winsor_y_low"] = lower
            row["winsor_y_high"] = upper
    return contacts


def registered_names(path: Path) -> list[str]:
    rows = read_tsv(path)
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("registered montage names are not unique")
    return names


def closest_area_patch(
    anterior: np.ndarray,
    pd: np.ndarray,
    hemisphere: np.ndarray,
    areas: np.ndarray,
    code: int,
    center: float,
    extent: float,
) -> np.ndarray:
    candidates = np.flatnonzero(hemisphere == code)
    order = candidates[np.lexsort((candidates, pd[candidates], np.abs(anterior[candidates] - center)))]
    target = extent * float(np.sum(areas[candidates]))
    stop = int(np.searchsorted(np.cumsum(areas[order]), target, side="left")) + 1
    return np.sort(order[:stop])


def build_forward_lookup(
    sessions: list[dict[str, Any]],
    config: dict[str, Any],
    hcp_subjects: list[str],
    hcp_forward_root: Path,
    hcp_legacy_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    forward_config = config["forward_predictor"]
    contacts = contact_catalogue(sessions, forward_config["contact_AP_winsor_quantiles"])
    channels = list(config["primary_scalp_channels"])
    extents = [float(value) for value in forward_config["sensitivity_extent_fractions"]]
    keys = sorted(contacts)
    gains = np.empty((len(hcp_subjects), len(keys), len(extents)), dtype=np.float64)
    for subject_index, subject in enumerate(hcp_subjects):
        forward_directory = hcp_forward_root / subject
        legacy_directory = hcp_legacy_root / subject
        leadfield = np.load(
            forward_directory / "hippunfold-hippocampal-fixed-leadfield.npy",
            allow_pickle=False,
        )
        with np.load(
            forward_directory / "hippunfold-hippocampal-source-metadata.npz",
            allow_pickle=False,
        ) as archive:
            metadata = {name: np.asarray(archive[name]) for name in archive.files}
        names = registered_names(legacy_directory / "registered-montage.tsv")
        rows = [names.index(name) for name in channels]
        restricted = np.asarray(leadfield[rows], dtype=np.float64)
        restricted -= restricted.mean(axis=0, keepdims=True)
        anterior, pd, _ = oriented_intrinsic_coordinates(
            metadata["positions_m"],
            metadata["longitudinal_coordinate"],
            metadata["proximal_distal_coordinate"],
            metadata["hemisphere_code"],
            metadata["area_weights_m2"],
            endpoint_decile=0.10,
            minimum_endpoint_separation_m=0.015,
        )
        hemisphere = np.asarray(metadata["hemisphere_code"], dtype=np.int64)
        areas = np.asarray(metadata["area_weights_m2"], dtype=np.float64)
        for contact_index, key in enumerate(keys):
            contact = contacts[key]
            for extent_index, extent in enumerate(extents):
                patch = closest_area_patch(
                    anterior,
                    pd,
                    hemisphere,
                    areas,
                    int(contact["hemisphere_code"]),
                    float(contact["anterior_coordinate"]),
                    extent,
                )
                weights = areas[patch] / np.sum(areas[patch])
                field = restricted[:, patch] @ weights
                gains[subject_index, contact_index, extent_index] = math.sqrt(
                    float(np.mean(field * field))
                )
        print(f"[HCP {subject_index + 1}/{len(hcp_subjects)}] {subject}", flush=True)
    if not np.all(np.isfinite(gains)) or np.any(gains <= 0.0):
        raise ValueError("forward lookup contains invalid gains")
    lookup_rows = []
    for contact_index, key in enumerate(keys):
        row = dict(contacts[key])
        row["gain_by_extent"] = {
            f"{extent:.2f}": {
                "median": float(np.median(gains[:, contact_index, extent_index])),
                "log_median": float(np.median(np.log(gains[:, contact_index, extent_index]))),
                "p05": float(np.quantile(gains[:, contact_index, extent_index], 0.05)),
                "p95": float(np.quantile(gains[:, contact_index, extent_index], 0.95)),
            }
            for extent_index, extent in enumerate(extents)
        }
        lookup_rows.append(row)
    npz_path = output_root / "forward_gain_population.npz"
    temporary_npz = npz_path.with_name(f".{npz_path.stem}.tmp.npz")
    np.savez_compressed(
        temporary_npz,
        hcp_subjects=np.asarray(hcp_subjects, dtype=str),
        contact_keys=np.asarray(keys, dtype=str),
        extents=np.asarray(extents, dtype=np.float64),
        gains=gains,
    )
    temporary_npz.replace(npz_path)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": True,
        "hcp_subject_count": len(hcp_subjects),
        "contact_count": len(keys),
        "primary_scalp_channels": channels,
        "extents": extents,
        "contacts": lookup_rows,
        "npz": npz_path.name,
        "npz_sha256": sha256(npz_path),
        "individual_patient_forward_model": False,
    }
    atomic_json(output_root / "forward_gain_lookup.json", report)
    return report


def fixed_outer_scores(
    features: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    scores = np.full(len(labels), np.nan)
    for heldout in np.unique(patients):
        train = patients != heldout
        test = ~train
        model = fit_shrinkage_lda(features[train], labels[train], shrinkage)
        scores[test] = predict_linear(features[test], model)
    return scores


def circular_shift_null(
    features: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    sessions: np.ndarray,
    shrinkage: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    unique_sessions = np.unique(sessions)
    for replicate in range(replicates):
        shifted = labels.copy()
        for session in unique_sessions:
            indices = np.flatnonzero(sessions == session)
            offset = int(generator.integers(1, len(indices)))
            shifted[indices] = np.roll(labels[indices], offset)
        scores = fixed_outer_scores(features, shifted, patients, shrinkage)
        patient_aucs = [
            roc_auc(shifted[patients == patient], scores[patients == patient])
            for patient in np.unique(patients)
        ]
        null[replicate] = float(np.mean(patient_aucs))
    observed_scores = fixed_outer_scores(features, labels, patients, shrinkage)
    observed = float(
        np.mean(
            [
                roc_auc(labels[patients == patient], observed_scores[patients == patient])
                for patient in np.unique(patients)
            ]
        )
    )
    return {
        "fixed_shrinkage": shrinkage,
        "observed_fixed_shrinkage_macro_patient_auc": observed,
        "replicates": replicates,
        "null_mean": float(np.mean(null)),
        "null_standard_deviation": float(np.std(null)),
        "one_sided_p": float((1 + np.count_nonzero(null >= observed)) / (replicates + 1)),
        "null_p95": float(np.quantile(null, 0.95)),
    }


def write_event_scores(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze_full(
    sessions: list[dict[str, Any]],
    lookup: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    primary_extent = f"{float(config['forward_predictor']['primary_extent_fraction']):.2f}"
    gain_by_extent_contact = {
        str(extent): {
            row["contact_key"]: row["gain_by_extent"][str(extent)]["log_median"]
            for row in lookup["contacts"]
        }
        for extent in (f"{float(value):.2f}" for value in lookup["extents"])
    }
    gain_by_contact = gain_by_extent_contact[primary_extent]
    event_rows: list[dict[str, Any]] = []
    feature_blocks = []
    labels_blocks = []
    patients_blocks = []
    session_blocks = []
    gain_blocks = []
    source_blocks = []
    for item in sessions:
        keep = np.asarray(item["labels"], dtype=np.int64) >= 0
        indices = np.flatnonzero(keep)
        dominant = np.asarray(item["dominant_pair_index"], dtype=np.int64)[keep]
        contact_keys = np.asarray(item["contact_keys"], dtype=str)[dominant]
        gains = np.asarray([gain_by_contact[key] for key in contact_keys], dtype=np.float64)
        features = np.asarray(item["scalp_features"], dtype=np.float64)[keep]
        labels = np.asarray(item["labels"], dtype=np.int64)[keep]
        patient = str(item["subject"])
        session_id = f"{patient}_{item['session']}"
        feature_blocks.append(features)
        labels_blocks.append(labels)
        patients_blocks.append(repeat_string(patient, len(labels)))
        session_blocks.append(repeat_string(session_id, len(labels)))
        gain_blocks.append(gains)
        source_blocks.append(contact_keys)
        for local, original in enumerate(indices):
            event_rows.append(
                {
                    "patient": patient,
                    "session": str(item["session"]),
                    "trial_number": int(np.asarray(item["trial_number"])[original]),
                    "set_size": int(np.asarray(item["set_size"])[original]),
                    "label": int(labels[local]),
                    "hippocampal_score": float(np.asarray(item["hippocampal_score"])[original]),
                    "dominant_contact": str(contact_keys[local]),
                    "predicted_log_gain": float(gains[local]),
                }
            )
    features = np.vstack(feature_blocks)
    labels = np.concatenate(labels_blocks)
    patients = np.concatenate(patients_blocks)
    session_ids = np.concatenate(session_blocks)
    predicted_gain = np.concatenate(gain_blocks)
    source_keys = np.concatenate(source_blocks)
    validation = config["validation"]
    lda_scores, lda_folds = outer_patient_lda(
        features, labels, patients, validation["shrinkage_grid"]
    )
    channel_scores, channel_folds = outer_best_channel(features, labels, patients)
    for index, row in enumerate(event_rows):
        row["heldout_lda_score"] = float(lda_scores[index])
        row["heldout_best_channel_score"] = float(channel_scores[index])

    patient_rows = []
    for patient in np.unique(patients):
        selected = patients == patient
        positive = selected & (labels == 1)
        patient_rows.append(
            {
                "patient": str(patient),
                "events": int(np.count_nonzero(selected)),
                "low_events": int(np.count_nonzero(selected & (labels == 0))),
                "high_events": int(np.count_nonzero(positive)),
                "lda_auc": roc_auc(labels[selected], lda_scores[selected]),
                "best_channel_auc": roc_auc(labels[selected], channel_scores[selected]),
                "median_high_event_predicted_log_gain": float(np.median(predicted_gain[positive])),
                "unique_dominant_contacts": int(len(np.unique(source_keys[positive]))),
            }
        )
    macro = bootstrap_macro_auc(
        [row["lda_auc"] for row in patient_rows],
        int(validation["bootstrap_replicates"]),
        int(validation["master_seed"]),
    )
    baseline_macro = bootstrap_macro_auc(
        [row["best_channel_auc"] for row in patient_rows],
        int(validation["bootstrap_replicates"]),
        int(validation["master_seed"]) + 1,
    )
    association = permutation_spearman(
        [row["median_high_event_predicted_log_gain"] for row in patient_rows],
        [row["lda_auc"] for row in patient_rows],
        int(validation["permutation_replicates"]),
        int(validation["master_seed"]) + 2,
    )
    session_rows = []
    within_rhos = []
    for session_id in np.unique(session_ids):
        selected = session_ids == session_id
        positive = selected & (labels == 1)
        correlation = None
        if np.count_nonzero(positive) >= 6 and len(np.unique(predicted_gain[positive])) >= 2:
            value = float(spearmanr(predicted_gain[positive], lda_scores[positive]).statistic)
            if math.isfinite(value):
                correlation = value
                within_rhos.append(value)
        session_rows.append(
            {
                "session": str(session_id),
                "patient": str(patients[selected][0]),
                "events": int(np.count_nonzero(selected)),
                "auc": roc_auc(labels[selected], lda_scores[selected]),
                "median_high_event_predicted_log_gain": float(np.median(predicted_gain[positive])),
                "unique_high_event_dominant_contacts": int(len(np.unique(source_keys[positive]))),
                "within_high_event_gain_score_spearman": correlation,
            }
        )
    patient_within_rhos = []
    for patient in np.unique(patients):
        values = [
            row["within_high_event_gain_score_spearman"]
            for row in session_rows
            if row["patient"] == patient
            and row["within_high_event_gain_score_spearman"] is not None
        ]
        if values:
            patient_within_rhos.append(
                {"patient": str(patient), "median_session_spearman_rho": float(np.median(values))}
            )

    forward_sensitivity: dict[str, Any] = {}
    for extent_index, extent in enumerate(sorted(gain_by_extent_contact, key=float)):
        contact_map = gain_by_extent_contact[extent]
        event_gain = np.asarray([contact_map[key] for key in source_keys], dtype=np.float64)
        patient_gain = [
            float(np.median(event_gain[(patients == patient) & (labels == 1)]))
            for patient in np.unique(patients)
        ]
        if extent == primary_extent:
            extent_association = association
        else:
            extent_association = permutation_spearman(
                patient_gain,
                [row["lda_auc"] for row in patient_rows],
                int(validation["permutation_replicates"]),
                int(validation["master_seed"]) + 100 + extent_index,
            )
        extent_session_rhos = []
        for session_id in np.unique(session_ids):
            positive = (session_ids == session_id) & (labels == 1)
            if np.count_nonzero(positive) >= 6 and len(np.unique(event_gain[positive])) >= 2:
                value = float(spearmanr(event_gain[positive], lda_scores[positive]).statistic)
                if math.isfinite(value):
                    extent_session_rhos.append(value)
        forward_sensitivity[extent] = {
            "frozen_primary": extent == primary_extent,
            "patient_level_association": extent_association,
            "eligible_within_session_correlations": len(extent_session_rhos),
            "median_within_session_spearman_rho": float(np.median(extent_session_rhos))
            if extent_session_rhos
            else None,
        }
    chosen = float(np.median([row["chosen_shrinkage"] for row in lda_folds]))
    circular = circular_shift_null(
        features,
        labels,
        patients,
        session_ids,
        chosen,
        int(validation["circular_shift_replicates"]),
        int(validation["master_seed"]) + 3,
    )
    event_path = output_root / "heldout_event_scores.csv"
    write_event_scores(event_path, event_rows)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": True,
        "patient_count": len(patient_rows),
        "session_count": len(session_rows),
        "labeled_event_count": len(labels),
        "low_event_count": int(np.count_nonzero(labels == 0)),
        "high_event_count": int(np.count_nonzero(labels == 1)),
        "primary_macro_patient_auc": macro,
        "best_single_channel_macro_patient_auc": baseline_macro,
        "primary_forward_prediction": association,
        "secondary_within_session_prediction": {
            "eligible_sessions": len(within_rhos),
            "median_spearman_rho": float(np.median(within_rhos)) if within_rhos else None,
            "rhos": within_rhos,
            "patient_balanced_medians": patient_within_rhos,
            "median_of_patient_medians": float(
                np.median([row["median_session_spearman_rho"] for row in patient_within_rhos])
            )
            if patient_within_rhos
            else None,
        },
        "forward_extent_sensitivity": forward_sensitivity,
        "circular_shift_negative_control": circular,
        "outer_lda_folds": lda_folds,
        "outer_best_channel_folds": channel_folds,
        "patients": patient_rows,
        "sessions": session_rows,
        "event_scores_csv": event_path.name,
        "event_scores_sha256": sha256(event_path),
        "heldout_ieeg_waveform_entered_scalp_model": False,
        "individual_patient_forward_model": False,
        "physiological_attribution_authorized": False,
    }
    atomic_json(output_root / "analysis_report.json", report)
    return report


def validate_subset(
    sessions: list[dict[str, Any]], lookup: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    feature_blocks = [np.asarray(row["scalp_features"], dtype=np.float64) for row in sessions]
    depth_blocks = [np.asarray(row["hippocampal_score"], dtype=np.float64) for row in sessions]
    labels = [np.asarray(row["labels"], dtype=np.int64) for row in sessions]
    features = np.vstack(feature_blocks)
    primary_extent = f"{float(config['forward_predictor']['primary_extent_fraction']):.2f}"
    primary_log_gains = np.asarray(
        [row["gain_by_extent"][primary_extent]["log_median"] for row in lookup["contacts"]],
        dtype=np.float64,
    )
    anatomy_spreads = np.asarray(
        [
            row["gain_by_extent"][primary_extent]["p95"]
            - row["gain_by_extent"][primary_extent]["p05"]
            for row in lookup["contacts"]
        ],
        dtype=np.float64,
    )
    checks = {
        "all_session_arrays_finite": bool(
            all(np.all(np.isfinite(block)) for block in feature_blocks + depth_blocks)
        ),
        "every_session_has_equal_nonzero_tail_classes": bool(
            all(
                np.count_nonzero(value == 0) == np.count_nonzero(value == 1) > 0
                for value in labels
            )
        ),
        "all_eight_scalp_features_have_variance": bool(
            features.shape[1] == 8 and np.all(np.std(features, axis=0) > 0.05)
        ),
        "every_session_depth_score_has_variance": bool(
            all(float(np.std(value)) > 0.05 for value in depth_blocks)
        ),
        "forward_population_has_50_anatomies": lookup["hcp_subject_count"] == 50,
        "multiple_empirical_contacts_are_represented": lookup["contact_count"] >= 2,
        "primary_forward_predictor_varies_across_contacts": bool(
            np.ptp(primary_log_gains) > 1e-3
        ),
        "anatomical_gain_uncertainty_is_nonzero": bool(np.all(anatomy_spreads > 0.0)),
        "all_contact_AP_coordinates_are_bounded": bool(
            all(0.0 <= row["anterior_coordinate"] <= 1.0 for row in lookup["contacts"])
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": all(checks.values()),
        "mode": "technical_subset",
        "subject_count": len({str(row["subject"]) for row in sessions}),
        "session_count": len(sessions),
        "eligible_trial_count": int(sum(len(value) for value in labels)),
        "labeled_low_count": int(sum(np.count_nonzero(value == 0) for value in labels)),
        "labeled_high_count": int(sum(np.count_nonzero(value == 1) for value in labels)),
        "primary_log_gain_range": float(np.ptp(primary_log_gains)),
        "minimum_primary_anatomical_p05_p95_width": float(np.min(anatomy_spreads)),
        "checks": checks,
        "analysis_run": False,
        "reason": "two-patient subset validates extraction and predictor construction only",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--hcp-forward-root", type=Path, required=True)
    parser.add_argument("--hcp-legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("subset", "full"), required=True)
    parser.add_argument("--stage", choices=("extract", "forward", "analyze", "all"), default="all")
    args = parser.parse_args()
    config = load_json(args.config)
    audit = load_json(args.audit)
    if config.get("protocol") != PROTOCOL or not audit.get("ok"):
        raise SystemExit("protocol or input audit gate failed")
    for relative, expected in config.get("implementation_sha256", {}).items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"frozen implementation hash mismatch: {relative}")
    if sha256(args.audit) != config["input_audit_sha256"]:
        raise SystemExit("hashed input audit differs from frozen config")
    subjects_file = PROJECT / config["hcp_subjects_file"]
    if sha256(subjects_file) != config["hcp_subjects_file_sha256"]:
        raise SystemExit("HCP subject list differs from frozen config")
    hcp_subjects = [line.strip() for line in subjects_file.read_text().splitlines() if line.strip()]
    if len(hcp_subjects) != 50:
        raise SystemExit("expected exactly 50 frozen HCP subjects")
    eligible = list(config["eligible_subjects"])
    if eligible != list(audit["working_memory_eligible_subjects"]):
        raise SystemExit("eligible cohort differs from frozen audit")
    subjects = list(config["minimum_subset_subjects"]) if args.mode == "subset" else eligible
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("extract", "all"):
        manifest = extract_all(args.dataset, audit, config, args.output, subjects)
        if not manifest["ok"]:
            return 2
    sessions = load_extractions(args.output / "sessions")
    if sorted({str(row["subject"]) for row in sessions}) != sorted(subjects):
        raise SystemExit("extracted subject cohort is incomplete or contaminated")
    if args.stage in ("forward", "all"):
        lookup = build_forward_lookup(
            sessions,
            config,
            hcp_subjects,
            args.hcp_forward_root,
            args.hcp_legacy_root,
            args.output,
        )
    else:
        lookup = load_json(args.output / "forward_gain_lookup.json")
    if args.stage in ("analyze", "all"):
        if args.mode == "subset":
            gate = validate_subset(sessions, lookup, config)
            atomic_json(args.output / "subset_gate.json", gate)
            print(json.dumps(gate, indent=2, sort_keys=True))
            if not gate["ok"]:
                return 2
        else:
            report = analyze_full(sessions, lookup, config, args.output)
            print(
                json.dumps(
                    {
                        "ok": report["ok"],
                        "patient_count": report["patient_count"],
                        "session_count": report["session_count"],
                        "labeled_event_count": report["labeled_event_count"],
                        "primary_macro_patient_auc": report["primary_macro_patient_auc"],
                        "primary_forward_prediction": report["primary_forward_prediction"],
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
