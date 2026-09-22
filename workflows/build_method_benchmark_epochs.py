#!/usr/bin/env python3
"""Extract method-benchmark scalp theta epochs while preserving frozen temporal-transfer labels."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.signal import resample_poly

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.working_memory import (  # noqa: E402
    atomic_json,
    continuous_bandpasses,
    read_edf_selected,
    read_tsv,
)


PROTOCOL = "benchmark/method-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def scalar(value: np.ndarray) -> object:
    return value.item() if value.ndim == 0 else value


def rational_resampling(source: float, target: float) -> tuple[int, int]:
    if source <= 0.0 or target <= 0.0:
        raise ValueError("sampling frequencies must be positive")
    ratio = Fraction(target / source).limit_denominator(10000)
    realized = source * ratio.numerator / ratio.denominator
    if abs(realized - target) > 1e-9:
        raise ValueError(f"cannot represent resampling ratio {source:g} -> {target:g} Hz exactly")
    return ratio.numerator, ratio.denominator


def extract_one(
    dataset: Path,
    audit: dict[str, Any],
    temporal_transfer_path: Path,
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    with np.load(temporal_transfer_path, allow_pickle=False) as archive:
        p4 = {name: scalar(np.asarray(archive[name])) for name in archive.files}
    if str(p4["protocol"]) != "openneuro-ds004752/patient-blocked-v1":
        raise ValueError(f"wrong temporal-transfer protocol: {temporal_transfer_path}")
    stem = str(p4["stem"])
    if stem != str(audit["stem"]):
        raise ValueError("temporal-transfer/audit session mismatch")
    eeg_path = dataset / str(audit["eeg_edf"])
    events_path = eeg_path.with_name(f"{stem}_events.tsv")
    channels = list(config["scalp_channels"])
    raw, sampling_frequency = read_edf_selected(eeg_path, channels)
    if abs(sampling_frequency - float(audit["eeg_sampling_frequency"])) > 1e-9:
        raise ValueError("EEG sampling frequency changed after the frozen input audit")
    raw -= raw.mean(axis=0, keepdims=True)
    signal = config["signal"]
    (theta,) = continuous_bandpasses(
        raw,
        sampling_frequency,
        (signal["theta_band_hz"],),
        int(signal["butterworth_order"]),
    )
    events = read_tsv(events_path)
    labels_all = np.asarray(p4["labels"], dtype=np.int64)
    keep = labels_all >= 0
    original = np.asarray(p4["original_event_row"], dtype=np.int64)[keep]
    labels = labels_all[keep]
    blocks = np.asarray(p4["block_ids"], dtype=np.int64)[keep]
    trials = np.asarray(p4["trial_number"], dtype=np.int64)[keep]
    window = [float(value) for value in config["signal"]["analysis_window_seconds"]]
    samples = int(round((window[1] - window[0]) * sampling_frequency))
    if samples < 16:
        raise ValueError("method-benchmark epoch is too short")
    epochs = np.empty((len(original), len(channels), samples), dtype=np.float64)
    for local, event_index in enumerate(original):
        onset = int(events[int(event_index)]["begSample"]) - 1
        left = onset + int(round(window[0] * sampling_frequency))
        right = left + samples
        if left < 0 or right > theta.shape[1]:
            raise ValueError(f"event window exceeds recording: {stem} row {event_index}")
        epochs[local] = theta[:, left:right]
    target_frequency = float(signal["target_sampling_frequency_hz"])
    up, down = rational_resampling(sampling_frequency, target_frequency)
    epochs = resample_poly(epochs, up, down, axis=2, padtype="line")
    expected_samples = int(round((window[1] - window[0]) * target_frequency))
    if epochs.shape[2] != expected_samples:
        raise ValueError("resampled epoch length differs from the frozen target")
    if not np.all(np.isfinite(epochs)) or np.max(np.abs(epochs.sum(axis=1))) > 1e-7 * max(
        float(np.max(np.abs(epochs))), 1.0
    ):
        raise ValueError("non-finite or non-referenced method-benchmark epochs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        protocol=np.asarray(PROTOCOL),
        source_protocol=np.asarray(str(p4["protocol"])),
        subject=np.asarray(str(p4["subject"])),
        session=np.asarray(str(p4["session"])),
        stem=np.asarray(stem),
        scalp_channels=np.asarray(channels, dtype=str),
        theta_epochs=np.asarray(epochs, dtype=np.float32),
        labels=labels,
        block_ids=blocks,
        trial_number=trials,
        original_event_row=original,
        source_sampling_frequency_hz=np.asarray(sampling_frequency),
        target_sampling_frequency_hz=np.asarray(target_frequency),
        temporal_transfer_npz_sha256=np.asarray(sha256(temporal_transfer_path)),
        depth_waveform_included=np.asarray(False),
    )
    temporary.replace(output)
    return {
        "subject": str(p4["subject"]),
        "session": str(p4["session"]),
        "stem": stem,
        "events": len(labels),
        "low_events": int(np.count_nonzero(labels == 0)),
        "high_events": int(np.count_nonzero(labels == 1)),
        "blocks": {str(value): int(np.count_nonzero(blocks == value)) for value in np.unique(blocks)},
        "samples_per_event": epochs.shape[2],
        "output": output.name,
        "sha256": sha256(output),
        "finite": True,
        "depth_waveform_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--temporal_transfer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config = load_json(args.config)
    if config.get("protocol") != PROTOCOL:
        raise SystemExit("unexpected method-benchmark protocol")
    dataset = args.dataset_root
    audit_path = args.input_audit
    audit = load_json(audit_path)
    by_stem = {str(row["stem"]): row for row in audit["sessions"]}
    selected = set(args.subjects or config["eligible_subjects"])
    paths = sorted((args.temporal_transfer_root / "sessions").glob("*.npz"))
    if not paths:
        raise FileNotFoundError("no frozen temporal-transfer sessions")
    reports = []
    args.output.mkdir(parents=True, exist_ok=False)
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            subject = str(np.asarray(archive["subject"]).item())
            stem = str(np.asarray(archive["stem"]).item())
        if subject not in selected:
            continue
        report = extract_one(
            dataset,
            by_stem[stem],
            path,
            config,
            args.output / "sessions" / f"{stem}.npz",
        )
        reports.append(report)
        print(json.dumps({"status": "complete", "stem": stem}), flush=True)
    subjects = sorted({row["subject"] for row in reports})
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": bool(reports)
        and set(subjects) == selected
        and all(row["finite"] and not row["depth_waveform_included"] for row in reports),
        "subjects": subjects,
        "subject_count": len(subjects),
        "session_count": len(reports),
        "event_count": sum(int(row["events"]) for row in reports),
        "sessions": reports,
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "working_memory_input_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "depth_waveform_included": False,
        "participants_tsv_read": False,
        "shared_storage_touched": False,
    }
    atomic_json(args.output / "extraction_manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "sessions"}, indent=2))
    return 0 if manifest["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
