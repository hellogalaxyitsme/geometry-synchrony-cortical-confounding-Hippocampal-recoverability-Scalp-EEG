"""Adapter from two explicitly projected MNE EEG forwards to a bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from .bundle import AnatomicalBundle, write_bundle


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source_metadata(path: Path, hippocampal: bool) -> dict[str, np.ndarray]:
    required = {"positions_m", "directions", "area_weights_m2"}
    if hippocampal:
        required.add("longitudinal_coordinate")
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {
            name: np.asarray(archive[name], dtype=np.float64) for name in required
        }


def _read_fixed_eeg_forward(path: Path) -> dict[str, object]:
    import mne
    from mne.io.constants import FIFF

    forward = mne.read_forward_solution(path, verbose=False)
    picks = mne.pick_types(
        forward["info"], meg=False, eeg=True, ref_meg=False, exclude=[]
    )
    if len(picks) < 2:
        raise ValueError(f"{path} contains fewer than two EEG sensors")
    names = [forward["info"]["ch_names"][index] for index in picks]
    forward = mne.pick_channels_forward(
        forward, include=names, ordered=True, copy=True, verbose=False
    )
    # MNE FIF deliberately serializes an operator derived from an original
    # free-orientation solution in its reversible XYZ form. Reapply the
    # source-space-normal projection on read; the independent source metadata
    # check below then proves that these are the declared directions.
    if forward["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
        forward = mne.convert_forward_solution(
            forward,
            surf_ori=True,
            force_fixed=True,
            use_cps=False,
            copy=False,
            verbose=False,
        )
    if forward["source_ori"] != FIFF.FIFFV_MNE_FIXED_ORI:
        raise ValueError(
            f"{path} could not be projected onto its declared source-space normals"
        )
    if forward["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
        raise ValueError(f"{path} source coordinates must be in the MNE head frame")
    leadfield = np.asarray(forward["sol"]["data"], dtype=np.float64)
    source_positions = np.asarray(forward["source_rr"], dtype=np.float64)
    source_directions = np.asarray(forward["source_nn"], dtype=np.float64)
    if leadfield.shape[1] != int(forward["nsource"]):
        raise ValueError(f"{path} fixed lead field has an unexpected column count")
    electrode_positions = np.asarray(
        [channel["loc"][:3] for channel in forward["info"]["chs"]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(electrode_positions)):
        raise ValueError(f"{path} has non-finite EEG electrode positions")
    return {
        "names": names,
        "leadfield": leadfield,
        "source_positions": source_positions,
        "source_directions": source_directions,
        "electrode_positions": electrode_positions,
    }


def _validate_metadata_against_forward(
    label: str,
    forward: Mapping[str, object],
    metadata: Mapping[str, np.ndarray],
    position_tolerance_m: float,
    direction_tolerance: float,
) -> None:
    positions = np.asarray(metadata["positions_m"])
    directions = np.asarray(metadata["directions"])
    forward_positions = np.asarray(forward["source_positions"])
    forward_directions = np.asarray(forward["source_directions"])
    if positions.shape != forward_positions.shape:
        raise ValueError(f"{label} metadata positions have the wrong shape")
    if directions.shape != forward_directions.shape:
        raise ValueError(f"{label} metadata directions have the wrong shape")
    position_error = float(np.max(np.linalg.norm(positions - forward_positions, axis=1)))
    if position_error > position_tolerance_m:
        raise ValueError(
            f"{label} metadata/forward positions differ by {position_error:g} m"
        )
    metadata_norms = np.linalg.norm(directions, axis=1)
    forward_norms = np.linalg.norm(forward_directions, axis=1)
    if np.any(metadata_norms == 0) or np.any(forward_norms == 0):
        raise ValueError(f"{label} contains a zero source direction")
    cosine = np.sum(directions * forward_directions, axis=1) / (
        metadata_norms * forward_norms
    )
    minimum_cosine = float(np.min(cosine))
    if minimum_cosine < 1.0 - direction_tolerance:
        raise ValueError(
            f"{label} metadata direction disagrees with forward orientation; "
            f"minimum cosine={minimum_cosine:g}"
        )


def _read_noise_covariance(path: Path, sensor_names: list[str]) -> np.ndarray:
    import mne

    covariance = mne.read_cov(path, verbose=False)
    covariance = mne.pick_channels_cov(
        covariance, include=sensor_names, ordered=True, copy=True
    )
    data = np.asarray(covariance["data"], dtype=np.float64)
    if covariance.get("diag", False):
        data = np.diag(data)
    if data.shape != (len(sensor_names), len(sensor_names)):
        raise ValueError("noise covariance does not cover the ordered EEG sensors")
    return 0.5 * (data + data.T)


def import_mne_bundle(
    output_directory: Path,
    hippocampal_forward_path: Path,
    cortical_forward_path: Path,
    hippocampal_metadata_path: Path,
    cortical_metadata_path: Path,
    noise_covariance_path: Path,
    subject_id: str,
    solver: Mapping[str, str],
    provenance: str,
    position_tolerance_m: float = 1e-6,
    direction_tolerance: float = 1e-6,
) -> AnatomicalBundle:
    """Import aligned fixed-orientation MNE operators and validate the result."""

    hippocampal = _read_fixed_eeg_forward(hippocampal_forward_path)
    cortical = _read_fixed_eeg_forward(cortical_forward_path)
    if hippocampal["names"] != cortical["names"]:
        raise ValueError(
            "hippocampal and cortical forward files must have identical ordered EEG channels"
        )
    if not np.allclose(
        hippocampal["electrode_positions"],
        cortical["electrode_positions"],
        rtol=0.0,
        atol=position_tolerance_m,
    ):
        raise ValueError("forward files disagree on EEG electrode positions")

    hippocampal_metadata = _load_source_metadata(
        hippocampal_metadata_path, hippocampal=True
    )
    cortical_metadata = _load_source_metadata(
        cortical_metadata_path, hippocampal=False
    )
    _validate_metadata_against_forward(
        "hippocampal",
        hippocampal,
        hippocampal_metadata,
        position_tolerance_m,
        direction_tolerance,
    )
    _validate_metadata_against_forward(
        "cortical",
        cortical,
        cortical_metadata,
        position_tolerance_m,
        direction_tolerance,
    )
    sensor_names = list(hippocampal["names"])
    noise = _read_noise_covariance(noise_covariance_path, sensor_names)
    input_digests = {
        "hippocampal_forward": _file_digest(hippocampal_forward_path),
        "cortical_forward": _file_digest(cortical_forward_path),
        "hippocampal_metadata": _file_digest(hippocampal_metadata_path),
        "cortical_metadata": _file_digest(cortical_metadata_path),
        "noise_covariance": _file_digest(noise_covariance_path),
    }
    manifest = {
        "schema_version": 1,
        "subject_id": subject_id,
        "coordinate_frame": "mne-head",
        "coordinate_unit": "m",
        "potential_unit": "V",
        "dipole_moment_unit": "A m",
        "leadfield_unit": "V/(A m)",
        "area_unit": "m^2",
        "reference": "common_gauge",
        "sensor_names": sensor_names,
        "solver": dict(solver),
        "provenance": provenance,
        "input_sha256": input_digests,
        "arrays_file": "arrays.npz",
        "arrays_sha256": "0" * 64,
    }
    bundle = AnatomicalBundle(
        manifest=manifest,
        hippocampal_leadfield=np.asarray(hippocampal["leadfield"]),
        cortical_leadfield=np.asarray(cortical["leadfield"]),
        electrode_positions_m=np.asarray(hippocampal["electrode_positions"]),
        hippocampal_positions_m=np.asarray(hippocampal_metadata["positions_m"]),
        cortical_positions_m=np.asarray(cortical_metadata["positions_m"]),
        hippocampal_directions=np.asarray(hippocampal_metadata["directions"]),
        cortical_directions=np.asarray(cortical_metadata["directions"]),
        hippocampal_area_weights_m2=np.asarray(
            hippocampal_metadata["area_weights_m2"]
        ).reshape(-1),
        cortical_area_weights_m2=np.asarray(
            cortical_metadata["area_weights_m2"]
        ).reshape(-1),
        hippocampal_longitudinal_coordinate=np.asarray(
            hippocampal_metadata["longitudinal_coordinate"]
        ).reshape(-1),
        sensor_noise_covariance=noise,
    )
    return write_bundle(output_directory, bundle)
