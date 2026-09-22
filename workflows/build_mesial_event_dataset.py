#!/usr/bin/env python3
"""Build EEG-only Koessler/Ternisien epoch bundles without depth-waveform leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from empirical.mesial_events import load_cell_epochs, preprocess_scalp  # noqa: E402

PROTOCOL = "mesial-events/external-validation-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL: raise ValueError("unexpected mesial-event protocol")
    args.output.mkdir(parents=True, exist_ok=False); rows = []
    total_raw = total_kept = 0
    for network in config["networks"]:
        identifier = str(network["id"]); patient = str(network["patient"])
        directory = args.dataset_root / str(config["data_subdirectory"]) / identifier
        eeg_path = directory / "EEG.mat"; seeg_path = directory / "SEEG.mat"
        eeg_names, eeg, eeg_contract = load_cell_epochs(eeg_path, "EEG", int(config["signal"]["samples_per_epoch"]))
        seeg_names, seeg, seeg_contract = load_cell_epochs(
            seeg_path, "SEEG", int(config["signal"]["samples_per_epoch"]), normalize_scalp_names=False
        )
        if len(eeg) != len(seeg): raise ValueError(f"paired event counts differ: {identifier}")
        processed, keep = preprocess_scalp(
            eeg, float(config["signal"]["sampling_frequency_hz"]), tuple(config["signal"]["bandpass_hz"]),
            int(config["signal"]["butterworth_order"]), float(config["signal"]["artifact_peak_uv"]),
        )
        destination = args.output / f"{identifier}.npz"
        np.savez_compressed(destination, protocol=np.asarray(PROTOCOL), patient=np.asarray(patient), network=np.asarray(identifier),
                            channel_names=np.asarray(eeg_names, dtype=str), eeg_epochs_uv=processed,
                            retained_original_indices=np.flatnonzero(keep), sampling_frequency_hz=np.asarray(float(config["signal"]["sampling_frequency_hz"])),
                            t0_sample=np.asarray(int(config["signal"]["t0_sample"])))
        rows.append({
            "patient": patient, "network": identifier, "scalp_channels": len(eeg_names), "depth_channels": len(seeg_names),
            "raw_events": len(eeg), "retained_events": len(processed), "excluded_events": int(len(eeg) - len(processed)),
            "eeg_contract": eeg_contract, "seeg_contract": seeg_contract,
            "eeg_input_sha256": sha256(eeg_path), "seeg_input_sha256": sha256(seeg_path),
            "bundle": destination.name, "bundle_sha256": sha256(destination),
        }); total_raw += len(eeg); total_kept += len(processed)
        del eeg, seeg, processed
    manifest = {
        "schema_version": 1, "protocol": PROTOCOL, "ok": len(rows) == 9 and len({row["patient"] for row in rows}) == 7,
        "patients": sorted({row["patient"] for row in rows}), "network_count": len(rows), "raw_event_count": total_raw,
        "retained_event_count": total_kept, "networks": rows,
        "depth_waveform_stored": False, "depth_waveform_used_by_scalp_detector": False,
        "auxiliary_non_512_rows_treated_as_trials": False, "participants_tsv_read": False, "shared_storage_touched": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True)); return 0 if manifest["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
