"""Strict on-disk contract for anatomical EEG lead-field bundles."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from theory.recoverability import helmert_reference


FloatArray = NDArray[np.float64]

ARRAY_NAMES = (
    "hippocampal_leadfield",
    "cortical_leadfield",
    "electrode_positions_m",
    "hippocampal_positions_m",
    "cortical_positions_m",
    "hippocampal_directions",
    "cortical_directions",
    "hippocampal_area_weights_m2",
    "cortical_area_weights_m2",
    "hippocampal_longitudinal_coordinate",
    "sensor_noise_covariance",
)

REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "subject_id",
    "coordinate_frame",
    "coordinate_unit",
    "potential_unit",
    "dipole_moment_unit",
    "leadfield_unit",
    "area_unit",
    "reference",
    "sensor_names",
    "solver",
    "provenance",
    "arrays_file",
    "arrays_sha256",
)


class BundleValidationError(ValueError):
    """Raised when one or more bundle-contract checks fail."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("invalid anatomical bundle: " + "; ".join(self.errors))


@dataclass(frozen=True)
class AnatomicalBundle:
    manifest: dict[str, object]
    hippocampal_leadfield: FloatArray
    cortical_leadfield: FloatArray
    electrode_positions_m: FloatArray
    hippocampal_positions_m: FloatArray
    cortical_positions_m: FloatArray
    hippocampal_directions: FloatArray
    cortical_directions: FloatArray
    hippocampal_area_weights_m2: FloatArray
    cortical_area_weights_m2: FloatArray
    hippocampal_longitudinal_coordinate: FloatArray
    sensor_noise_covariance: FloatArray
    directory: Path | None = None

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["sensor_names"])

    @property
    def number_of_sensors(self) -> int:
        return int(self.hippocampal_leadfield.shape[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deterministic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a byte-reproducible, pickle-free NPZ archive."""

    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(
                filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def _array_mapping(bundle: AnatomicalBundle) -> dict[str, FloatArray]:
    return {name: np.asarray(getattr(bundle, name)) for name in ARRAY_NAMES}


def write_bundle(directory: Path, bundle: AnatomicalBundle) -> AnatomicalBundle:
    """Write arrays and a digest-bearing manifest, then reload and validate."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path = directory / "arrays.npz"
    _deterministic_savez(arrays_path, _array_mapping(bundle))
    manifest = dict(bundle.manifest)
    manifest["arrays_file"] = "arrays.npz"
    manifest["arrays_sha256"] = sha256_file(arrays_path)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return load_bundle(directory)


def _shape_error(
    errors: list[str], name: str, actual: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    if actual != expected:
        errors.append(f"{name} has shape {actual}, expected {expected}")


def validate_bundle(
    bundle: AnatomicalBundle, verify_checksum: bool = True
) -> dict[str, object]:
    """Validate every contract invariant and return an audit report."""

    errors: list[str] = []
    manifest = bundle.manifest
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            errors.append(f"manifest missing {field}")
    if errors:
        raise BundleValidationError(errors)

    if manifest["schema_version"] != 1:
        errors.append("unsupported schema_version")
    expected_units = {
        "coordinate_unit": "m",
        "potential_unit": "V",
        "dipole_moment_unit": "A m",
        "leadfield_unit": "V/(A m)",
        "area_unit": "m^2",
        "reference": "common_gauge",
    }
    for field, expected in expected_units.items():
        if manifest.get(field) != expected:
            errors.append(f"{field} must be {expected!r}")
    if not isinstance(manifest.get("subject_id"), str) or not manifest["subject_id"]:
        errors.append("subject_id must be a nonempty string")
    if not isinstance(manifest.get("coordinate_frame"), str) or not manifest[
        "coordinate_frame"
    ]:
        errors.append("coordinate_frame must be a nonempty string")
    if not isinstance(manifest.get("provenance"), str) or not manifest["provenance"]:
        errors.append("provenance must be a nonempty string")
    solver = manifest.get("solver")
    if not isinstance(solver, dict):
        errors.append("solver must be an object")
    else:
        for field in ("name", "version", "method", "conductivity_model"):
            if not isinstance(solver.get(field), str) or not solver[field]:
                errors.append(f"solver.{field} must be a nonempty string")

    arrays = _array_mapping(bundle)
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number):
            errors.append(f"{name} must be numeric")
        elif not np.all(np.isfinite(array)):
            errors.append(f"{name} contains non-finite values")

    if bundle.hippocampal_leadfield.ndim != 2:
        errors.append("hippocampal_leadfield must be two-dimensional")
    if bundle.cortical_leadfield.ndim != 2:
        errors.append("cortical_leadfield must be two-dimensional")
    if errors:
        raise BundleValidationError(errors)

    sensors, hippocampal_elements = bundle.hippocampal_leadfield.shape
    cortical_sensors, cortical_elements = bundle.cortical_leadfield.shape
    if sensors < 2:
        errors.append("at least two sensors are required")
    if hippocampal_elements < 1 or cortical_elements < 1:
        errors.append("both source spaces must contain at least one element")
    if cortical_sensors != sensors:
        errors.append("hippocampal and cortical lead fields have different sensors")

    _shape_error(errors, "electrode_positions_m", bundle.electrode_positions_m.shape, (sensors, 3))
    _shape_error(
        errors,
        "hippocampal_positions_m",
        bundle.hippocampal_positions_m.shape,
        (hippocampal_elements, 3),
    )
    _shape_error(
        errors,
        "cortical_positions_m",
        bundle.cortical_positions_m.shape,
        (cortical_elements, 3),
    )
    _shape_error(
        errors,
        "hippocampal_directions",
        bundle.hippocampal_directions.shape,
        (hippocampal_elements, 3),
    )
    _shape_error(
        errors,
        "cortical_directions",
        bundle.cortical_directions.shape,
        (cortical_elements, 3),
    )
    _shape_error(
        errors,
        "hippocampal_area_weights_m2",
        bundle.hippocampal_area_weights_m2.shape,
        (hippocampal_elements,),
    )
    _shape_error(
        errors,
        "cortical_area_weights_m2",
        bundle.cortical_area_weights_m2.shape,
        (cortical_elements,),
    )
    _shape_error(
        errors,
        "hippocampal_longitudinal_coordinate",
        bundle.hippocampal_longitudinal_coordinate.shape,
        (hippocampal_elements,),
    )
    _shape_error(
        errors,
        "sensor_noise_covariance",
        bundle.sensor_noise_covariance.shape,
        (sensors, sensors),
    )

    names = manifest.get("sensor_names")
    if not isinstance(names, list) or len(names) != sensors:
        errors.append("sensor_names must be a list matching the sensor count")
    elif (
        any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        errors.append("sensor_names must contain unique nonempty strings")

    for name, directions in (
        ("hippocampal_directions", bundle.hippocampal_directions),
        ("cortical_directions", bundle.cortical_directions),
    ):
        if directions.ndim == 2 and directions.shape[1:] == (3,):
            norms = np.linalg.norm(directions, axis=1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-6):
                errors.append(f"{name} rows must have unit norm")

    for name, weights in (
        ("hippocampal_area_weights_m2", bundle.hippocampal_area_weights_m2),
        ("cortical_area_weights_m2", bundle.cortical_area_weights_m2),
    ):
        if np.any(weights <= 0):
            errors.append(f"{name} must be strictly positive")
    xi = bundle.hippocampal_longitudinal_coordinate
    if np.any(xi < 0.0) or np.any(xi > 1.0):
        errors.append("hippocampal_longitudinal_coordinate must lie in [0, 1]")

    noise = bundle.sensor_noise_covariance
    if noise.shape == (sensors, sensors):
        scale = max(
            np.finfo(np.float64).tiny, float(np.linalg.norm(noise, ord=2))
        )
        if float(np.linalg.norm(noise - noise.T, ord=2)) > 1e-10 * scale:
            errors.append("sensor_noise_covariance must be symmetric")
        else:
            noise = 0.5 * (noise + noise.T)
            raw_eigenvalues = np.linalg.eigvalsh(noise)
            if float(np.min(raw_eigenvalues)) < -1e-10 * scale:
                errors.append("sensor_noise_covariance must be positive semidefinite")
            if sensors >= 2:
                contrasts = helmert_reference(sensors)
                referenced = contrasts @ noise @ contrasts.T
                referenced_scale = max(
                    np.finfo(np.float64).tiny,
                    float(np.linalg.norm(referenced, ord=2)),
                )
                if (
                    float(np.min(np.linalg.eigvalsh(referenced)))
                    <= 1e-12 * referenced_scale
                ):
                    errors.append(
                        "sensor_noise_covariance must be positive definite "
                        "after referencing"
                    )

    digest = str(manifest.get("arrays_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        errors.append("arrays_sha256 must be a lowercase SHA-256 digest")
    if manifest.get("arrays_file") != "arrays.npz":
        errors.append("arrays_file must be 'arrays.npz'")
    if verify_checksum:
        if bundle.directory is None:
            errors.append("cannot verify checksum without a bundle directory")
        else:
            arrays_path = bundle.directory / str(manifest["arrays_file"])
            if not arrays_path.is_file():
                errors.append("arrays file is missing")
            elif sha256_file(arrays_path) != digest:
                errors.append("arrays SHA-256 mismatch")

    if errors:
        raise BundleValidationError(errors)
    return {
        "valid": True,
        "subject_id": manifest["subject_id"],
        "number_of_sensors": sensors,
        "number_of_hippocampal_elements": hippocampal_elements,
        "number_of_cortical_elements": cortical_elements,
        "referenced_dimension": sensors - 1,
        "arrays_sha256": digest,
    }


def load_bundle(directory: Path, verify_checksum: bool = True) -> AnatomicalBundle:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise BundleValidationError(["manifest.json is missing"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise BundleValidationError(["manifest.json must contain an object"])
    arrays_file = manifest.get("arrays_file")
    if arrays_file != "arrays.npz":
        raise BundleValidationError(["arrays_file must be 'arrays.npz'"])
    arrays_path = directory / str(arrays_file)
    if not arrays_path.is_file():
        raise BundleValidationError([f"{arrays_file} is missing"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        missing = [name for name in ARRAY_NAMES if name not in archive.files]
        if missing:
            raise BundleValidationError(
                [f"arrays archive missing {name}" for name in missing]
            )
        arrays = {
            name: np.asarray(archive[name], dtype=np.float64) for name in ARRAY_NAMES
        }
    bundle = AnatomicalBundle(manifest=dict(manifest), directory=directory, **arrays)
    validate_bundle(bundle, verify_checksum=verify_checksum)
    return bundle
