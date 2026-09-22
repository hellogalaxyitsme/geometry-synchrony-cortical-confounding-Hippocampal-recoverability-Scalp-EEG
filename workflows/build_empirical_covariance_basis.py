#!/usr/bin/env python3
"""Build label-blind empirical spatial/temporal covariance shapes for cortical-restriction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.working_memory import continuous_bandpasses, read_edf_selected  # noqa: E402
from theory.recoverability import helmert_reference  # noqa: E402


PROTOCOL = "restriction/empirical-covariance-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_psd(matrix: np.ndarray, ridge_fraction: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, 0.0)
    repaired = (vectors * values[None, :]) @ vectors.T
    trace = float(np.trace(repaired))
    if trace <= 0.0:
        raise ValueError("empirical covariance has zero trace")
    repaired /= trace
    repaired += ridge_fraction * np.eye(len(repaired)) / len(repaired)
    repaired /= float(np.trace(repaired))
    return np.asarray(0.5 * (repaired + repaired.T), dtype=np.float64)


def robust_channel_z(data: np.ndarray) -> np.ndarray:
    center = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - center), axis=1, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.std(data, axis=1, keepdims=True)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.asarray((data - center) / scale, dtype=np.float64)


def session_is_usable(
    retained: int,
    candidate: int,
    minimum_retained_windows: int,
    minimum_retained_fraction: float,
) -> tuple[bool, str | None]:
    """Apply the frozen, label-blind recording-quality gate."""
    if candidate <= 0:
        return False, "no_complete_windows"
    if retained < minimum_retained_windows:
        return False, "too_few_artifact_screened_windows"
    if retained / candidate < minimum_retained_fraction:
        return False, "retained_fraction_below_gate"
    return True, None


def patient_fraction_is_usable(included: int, eligible: int, minimum_fraction: float) -> bool:
    """Return whether the frozen patient-coverage gate is met."""
    return eligible > 0 and included / eligible >= minimum_fraction


def session_covariances(
    path: Path,
    channels: list[str],
    target_frequency: int,
    window_seconds: float,
    rejection_z: float,
    filter_order: int,
    minimum_retained_windows: int,
    minimum_retained_fraction: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object]]:
    data, frequency = read_edf_selected(path, channels)
    data -= data.mean(axis=0, keepdims=True)
    (filtered,) = continuous_bandpasses(data, frequency, ((1.0, 30.0),), filter_order)
    rounded_frequency = int(round(frequency))
    if abs(frequency - rounded_frequency) > 1e-9:
        raise ValueError("cortical-restriction empirical resampling requires integer input frequency")
    divisor = int(np.gcd(rounded_frequency, target_frequency))
    resampled = resample_poly(
        filtered,
        up=target_frequency // divisor,
        down=rounded_frequency // divisor,
        axis=1,
        padtype="line",
    )
    standardized = robust_channel_z(resampled)
    samples = int(round(target_frequency * window_seconds))
    count = standardized.shape[1] // samples
    if count < 10:
        raise ValueError("recording is too short for empirical covariance")
    contrasts = helmert_reference(len(channels))
    spatial = np.zeros((len(channels) - 1, len(channels) - 1), dtype=np.float64)
    temporal = np.zeros((samples, samples), dtype=np.float64)
    retained = 0
    excluded = 0
    for index in range(count):
        window = standardized[:, index * samples : (index + 1) * samples]
        if not np.all(np.isfinite(window)) or float(np.max(np.abs(window))) > rejection_z:
            excluded += 1
            continue
        referenced = contrasts @ window
        spatial += referenced @ referenced.T / samples
        centered = window - window.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
        usable = scale[:, 0] > 1e-12
        if not np.any(usable):
            excluded += 1
            continue
        normalized = centered[usable] / scale[usable]
        temporal += normalized.T @ normalized / int(np.count_nonzero(usable))
        retained += 1
    usable_session, exclusion_reason = session_is_usable(
        retained,
        count,
        minimum_retained_windows,
        minimum_retained_fraction,
    )
    report = {
        "input_frequency_hz": frequency,
        "target_frequency_hz": target_frequency,
        "candidate_windows": count,
        "retained_windows": retained,
        "excluded_windows": excluded,
        "retained_fraction": retained / count,
        "included": usable_session,
        "exclusion_reason": exclusion_reason,
    }
    if not usable_session:
        return None, None, report
    return (
        normalized_psd(spatial / retained, 0.0),
        normalized_psd(temporal / retained, 0.0),
        report,
    )


def retained_rank(covariance: np.ndarray, fraction: float) -> tuple[int, float]:
    values = np.linalg.eigvalsh(covariance)[::-1]
    stop = int(np.searchsorted(np.cumsum(values), fraction * np.sum(values), side="left")) + 1
    return stop, float(np.sum(values[:stop]) / np.sum(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("subset", "full"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if config.get("protocol") != "restriction/ladder-v1" or audit.get("ok") is not True:
        raise SystemExit("cortical-restriction configuration or input audit failed")
    if sha256(args.audit) != config["empirical_covariance"]["input_audit_sha256"]:
        raise SystemExit("empirical covariance input audit changed")
    settings = config["empirical_covariance"]
    channels = list(config["montages"]["clinical8_channels"])
    eligible = list(config["eligible_ds004752_subjects"])
    subjects = list(settings["subset_subjects"]) if args.mode == "subset" else eligible
    rows = [
        row
        for row in audit["sessions"]
        if row.get("working_memory_eligible") and row["subject"] in subjects
    ]
    expected_sessions = 11 if args.mode == "subset" else 62
    if len(rows) != expected_sessions:
        raise SystemExit("empirical covariance session cohort is incomplete")
    by_patient: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {subject: [] for subject in subjects}
    reports = []
    minimum_windows = int(config["gates"]["minimum_empirical_retained_windows_per_session"])
    minimum_fraction = float(config["gates"]["minimum_empirical_session_retained_fraction"])
    for index, row in enumerate(rows, start=1):
        path = args.dataset / row["eeg_edf"]
        spatial, temporal, report = session_covariances(
            path,
            channels,
            int(settings["target_sampling_frequency_hz"]),
            float(settings["window_seconds"]),
            float(settings["artifact_rejection_max_abs_robust_z"]),
            int(settings["butterworth_order"]),
            minimum_windows,
            minimum_fraction,
        )
        if report["included"]:
            assert spatial is not None and temporal is not None
            by_patient[row["subject"]].append((spatial, temporal))
        reports.append({"subject": row["subject"], "session": row["session"], "stem": row["stem"], **report})
        print(
            f"[{index}/{len(rows)}] {row['stem']} retained={report['retained_windows']}/"
            f"{report['candidate_windows']} included={report['included']}",
            flush=True,
        )
    patient_spatial = []
    patient_temporal = []
    minimum_patient_sessions = int(config["gates"]["minimum_empirical_included_sessions_per_patient"])
    included_subjects = []
    excluded_subjects = []
    for subject in subjects:
        values = by_patient[subject]
        if len(values) < minimum_patient_sessions:
            excluded_subjects.append(subject)
            continue
        included_subjects.append(subject)
        patient_spatial.append(np.mean([value[0] for value in values], axis=0))
        patient_temporal.append(np.mean([value[1] for value in values], axis=0))
    minimum_patient_fraction = float(config["gates"]["minimum_empirical_included_patient_fraction"])
    if not patient_fraction_is_usable(len(included_subjects), len(subjects), minimum_patient_fraction):
        raise ValueError("too few patients passed the frozen empirical covariance QC gate")
    included_reports = [row for row in reports if row["included"]]
    excluded_reports = [row for row in reports if not row["included"]]
    exclusion_fraction = len(excluded_reports) / len(reports)
    if exclusion_fraction > float(config["gates"]["maximum_empirical_excluded_session_fraction"]):
        raise ValueError("too many empirical covariance sessions failed the frozen QC gate")
    ridge = float(settings["covariance_ridge_fraction"])
    spatial = normalized_psd(np.mean(patient_spatial, axis=0), ridge)
    temporal = normalized_psd(np.mean(patient_temporal, axis=0), ridge)
    spatial_rank, spatial_retained = retained_rank(spatial, float(config["restrictions"]["covariance_retained_fraction"]))
    temporal_rank, temporal_retained = retained_rank(temporal, float(config["restrictions"]["temporal_retained_fraction"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        protocol=np.asarray(PROTOCOL),
        mode=np.asarray(args.mode),
        channels=np.asarray(channels, dtype=str),
        spatial_covariance=spatial,
        temporal_covariance=temporal,
        target_sampling_frequency_hz=np.asarray(int(settings["target_sampling_frequency_hz"])),
        window_samples=np.asarray(temporal.shape[0]),
        subjects=np.asarray(included_subjects, dtype=str),
    )
    temporary.replace(args.output)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": True,
        "mode": args.mode,
        "eligible_subjects": subjects,
        "eligible_subject_count": len(subjects),
        "subjects": included_subjects,
        "subject_count": len(included_subjects),
        "excluded_subjects": excluded_subjects,
        "included_subject_fraction": len(included_subjects) / len(subjects),
        "session_count": len(rows),
        "included_session_count": len(included_reports),
        "excluded_session_count": len(excluded_reports),
        "excluded_session_fraction": exclusion_fraction,
        "minimum_retained_windows_per_included_session": minimum_windows,
        "minimum_included_sessions_per_patient": minimum_patient_sessions,
        "channels": channels,
        "candidate_windows": sum(int(row["candidate_windows"]) for row in reports),
        "retained_windows": sum(int(row["retained_windows"]) for row in reports),
        "excluded_windows": sum(int(row["excluded_windows"]) for row in reports),
        "minimum_session_retained_fraction": min(float(row["retained_fraction"]) for row in included_reports),
        "spatial_trace": float(np.trace(spatial)),
        "temporal_trace": float(np.trace(temporal)),
        "spatial_minimum_eigenvalue": float(np.linalg.eigvalsh(spatial)[0]),
        "temporal_minimum_eigenvalue": float(np.linalg.eigvalsh(temporal)[0]),
        "spatial_retained_rank_99": spatial_rank,
        "spatial_realized_retained_fraction": spatial_retained,
        "temporal_retained_rank_95": temporal_rank,
        "temporal_realized_retained_fraction": temporal_retained,
        "output": args.output.name,
        "output_sha256": sha256(args.output),
        "sessions": reports,
        "excluded_sessions": [
            {
                "subject": row["subject"],
                "session": row["session"],
                "stem": row["stem"],
                "retained_windows": row["retained_windows"],
                "candidate_windows": row["candidate_windows"],
                "retained_fraction": row["retained_fraction"],
                "exclusion_reason": row["exclusion_reason"],
            }
            for row in excluded_reports
        ],
        "participants_tsv_read": False,
        "task_or_depth_labels_read": False,
        "raw_waveforms_saved": False,
        "shared_storage_touched": False,
    }
    atomic_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "sessions"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
