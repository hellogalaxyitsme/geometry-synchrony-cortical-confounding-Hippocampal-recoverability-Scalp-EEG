"""Audit and safely stage New York Head model assets."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import shutil
import uuid
import zipfile


TISSUES = ("air", "bone", "csf", "gray", "skin", "white")
SEGMENTATION_FILES = tuple(f"{name}.nii" for name in TISSUES) + ("license",)
MARKER_NAME = ".nyhead_segmentation_extraction.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_flat_member(info: zipfile.ZipInfo) -> str | None:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        return "invalid member characters"
    path = PurePosixPath(name)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in (".", ".."):
        return "member must be a single relative filename"
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in (0, stat.S_IFREG):
        return "member is not a regular file"
    return None


def audit_segmentation_archive(path: Path) -> dict[str, object]:
    errors: list[str] = []
    members: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            for info in infos:
                reason = _safe_flat_member(info)
                if reason:
                    errors.append(f"{info.filename!r}: {reason}")
                members.append(
                    {
                        "name": info.filename,
                        "uncompressed_bytes": int(info.file_size),
                        "compressed_bytes": int(info.compress_size),
                        "crc32": f"{info.CRC:08x}",
                    }
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        return {"path": str(path), "ok": False, "errors": [repr(error)]}
    if sorted(names) != sorted(SEGMENTATION_FILES):
        errors.append(
            f"expected files {list(SEGMENTATION_FILES)}, found {sorted(names)}"
        )
    if len(names) != len(set(names)) or len(names) != len(set(item.casefold() for item in names)):
        errors.append("duplicate or case-colliding member names")
    nifti_sizes = [
        int(item["uncompressed_bytes"])
        for item in members
        if str(item["name"]).endswith(".nii")
    ]
    if len(nifti_sizes) != 6 or len(set(nifti_sizes)) != 1:
        errors.append("six tissue NIfTIs do not have identical byte sizes")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "archive_bytes": path.stat().st_size,
        "members": members,
        "total_uncompressed_bytes": sum(int(item["uncompressed_bytes"]) for item in members),
        "ok": not errors,
        "errors": errors,
    }


def audit_mat_file(path: Path) -> dict[str, object]:
    import h5py
    import numpy as np

    datasets: list[dict[str, object]] = []
    groups: list[str] = []
    contract: dict[str, object] = {}
    errors: list[str] = []
    with h5py.File(path, "r") as handle:
        root_keys = sorted(handle.keys())

        def visitor(name: str, item: object) -> None:
            if isinstance(item, h5py.Group):
                groups.append(name)
            elif isinstance(item, h5py.Dataset):
                datasets.append(
                    {
                        "path": name,
                        "shape": [int(value) for value in item.shape],
                        "dtype": str(item.dtype),
                        "storage_bytes": int(item.id.get_storage_size()),
                    }
                )

        handle.visititems(visitor)
        required = {
            "sa/clab_electrodes": (231, 1),
            "sa/locs_3D": (6, 231),
            "sa/locs_3D_orig": (3, 231),
            "sa/cortex75K/V_fem": (3, 74382, 231),
            "sa/cortex75K/V_fem_normal": (74382, 231),
            "sa/cortex75K/vc": (3, 74382),
            "sa/cortex75K/normals": (3, 74382),
            "sa/cortex75K/tri": (3, 148756),
            "sa/head/vc": (3, 1082),
            "sa/head/normals": (3, 1082),
            "sa/head/tri": (3, 2160),
            "sa/mni2mri": (4, 4),
            "sa/mri2mni": (4, 4),
            "sa/mri/data": (378, 466, 394),
        }
        missing_required = [name for name in required if name not in handle]
        wrong_shapes = {
            name: {"expected": list(shape), "actual": list(handle[name].shape)}
            for name, shape in required.items()
            if name in handle and tuple(handle[name].shape) != shape
        }
        if missing_required:
            errors.append(f"missing required MAT datasets: {missing_required}")
        if wrong_shapes:
            errors.append(f"required MAT dataset shape mismatches: {wrong_shapes}")

        if not missing_required and not wrong_shapes:
            cortex_vertices = np.asarray(handle["sa/cortex75K/vc"], dtype=np.float64)
            cortex_normals = np.asarray(
                handle["sa/cortex75K/normals"], dtype=np.float64
            )
            cortex_triangles = np.asarray(
                handle["sa/cortex75K/tri"], dtype=np.float64
            )
            head_vertices = np.asarray(handle["sa/head/vc"], dtype=np.float64)
            head_normals = np.asarray(handle["sa/head/normals"], dtype=np.float64)
            head_triangles = np.asarray(handle["sa/head/tri"], dtype=np.float64)
            electrode_locations = np.asarray(handle["sa/locs_3D"], dtype=np.float64)
            transform_mni2mri = np.asarray(handle["sa/mni2mri"], dtype=np.float64)
            transform_mri2mni = np.asarray(handle["sa/mri2mni"], dtype=np.float64)
            numeric_arrays = (
                cortex_vertices,
                cortex_normals,
                cortex_triangles,
                head_vertices,
                head_normals,
                head_triangles,
                electrode_locations,
                transform_mni2mri,
                transform_mri2mni,
            )
            if not all(np.all(np.isfinite(array)) for array in numeric_arrays):
                errors.append("New York Head geometry contains non-finite values")
            cortex_normal_error = float(
                np.max(np.abs(np.linalg.norm(cortex_normals, axis=0) - 1.0))
            )
            head_normal_error = float(
                np.max(np.abs(np.linalg.norm(head_normals, axis=0) - 1.0))
            )
            if cortex_normal_error > 1e-6:
                errors.append(
                    f"cortical normal maximum norm error is {cortex_normal_error}"
                )
            if head_normal_error > 1e-6:
                errors.append(f"head normal maximum norm error is {head_normal_error}")
            for label, triangles, vertices in (
                ("cortex", cortex_triangles, cortex_vertices),
                ("head", head_triangles, head_vertices),
            ):
                if not np.allclose(triangles, np.rint(triangles), rtol=0.0, atol=0.0):
                    errors.append(f"{label} triangle indices are not integers")
                if float(triangles.min()) != 1.0 or float(triangles.max()) > vertices.shape[1]:
                    errors.append(f"{label} triangle indices are outside MATLAB 1-based range")
            inverse_error = float(
                np.max(np.abs(transform_mni2mri @ transform_mri2mni - np.eye(4)))
            )
            if inverse_error > 1e-8:
                errors.append(f"MNI/MRI transform inverse error is {inverse_error}")
            vector_sample = np.asarray(
                handle["sa/cortex75K/V_fem"][:, :64, :16], dtype=np.float64
            )
            normal_sample = np.asarray(
                handle["sa/cortex75K/V_fem_normal"][:64, :16], dtype=np.float64
            )
            if not np.all(np.isfinite(vector_sample)) or not np.all(np.isfinite(normal_sample)):
                errors.append("sampled FEM lead fields contain non-finite values")
            contract = {
                "electrodes": 231,
                "cortical_sources": 74382,
                "vector_leadfield_on_disk_shape": [3, 74382, 231],
                "normal_leadfield_on_disk_shape": [74382, 231],
                "cortical_vertices_coordinate_min": cortex_vertices.min(axis=1).tolist(),
                "cortical_vertices_coordinate_max": cortex_vertices.max(axis=1).tolist(),
                "head_vertices_coordinate_min": head_vertices.min(axis=1).tolist(),
                "head_vertices_coordinate_max": head_vertices.max(axis=1).tolist(),
                "cortical_normal_maximum_norm_error": cortex_normal_error,
                "head_normal_maximum_norm_error": head_normal_error,
                "cortical_triangle_minimum_index": int(cortex_triangles.min()),
                "cortical_triangle_maximum_index": int(cortex_triangles.max()),
                "head_triangle_minimum_index": int(head_triangles.min()),
                "head_triangle_maximum_index": int(head_triangles.max()),
                "mni_mri_transform_inverse_maximum_error": inverse_error,
                "sampled_vector_leadfield_finite": bool(np.all(np.isfinite(vector_sample))),
                "sampled_normal_leadfield_finite": bool(np.all(np.isfinite(normal_sample))),
            }
    public_datasets = [item for item in datasets if not str(item["path"]).startswith("#refs#/")]
    if "sa" not in root_keys:
        errors.append("MAT file has no top-level 'sa' object")
    if not datasets:
        errors.append("MAT file contains no HDF5 datasets")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "format": "MATLAB v7.3 / HDF5",
        "root_keys": root_keys,
        "group_count": len(groups),
        "dataset_count": len(datasets),
        "public_dataset_count": len(public_datasets),
        "public_datasets": public_datasets,
        "all_datasets": datasets,
        "model_contract": contract,
        "ok": not errors,
        "errors": errors,
    }


def extract_segmentation_archive(
    archive_path: Path,
    destination: Path,
    staging_root: Path,
    label: str,
) -> dict[str, object]:
    audit = audit_segmentation_archive(archive_path)
    if not audit["ok"]:
        raise ValueError(f"unsafe New York Head segmentation archive: {audit['errors']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    marker = destination / MARKER_NAME
    if destination.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FileExistsError(f"refusing incomplete destination: {destination}") from error
        if payload.get("archive_sha256") == audit["sha256"] and payload.get("status") == "complete":
            return {"label": label, "status": "already_complete", "destination": str(destination)}
        raise FileExistsError(f"refusing mismatched destination: {destination}")

    stage = staging_root / f"{label}-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                reason = _safe_flat_member(info)
                if reason:
                    raise ValueError(f"unsafe member {info.filename!r}: {reason}")
                output = stage / info.filename
                with archive.open(info, "r") as source, output.open("xb") as target:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                if output.stat().st_size != info.file_size:
                    raise IOError(f"size mismatch for {info.filename}")
        marker_payload = {
            "schema_version": 1,
            "status": "complete",
            "label": label,
            "archive": archive_path.name,
            "archive_sha256": audit["sha256"],
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (stage / MARKER_NAME).write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
        return {"label": label, "status": "extracted", "destination": str(destination)}
    except Exception as error:
        raise RuntimeError(f"extraction failed; staging preserved at {stage}") from error


def audit_extracted_segmentations(root: Path) -> dict[str, object]:
    import nibabel as nib
    import numpy as np

    binary_atol = 1e-6
    slab_depth = 16
    errors: list[str] = []
    warnings: list[str] = []
    tissue_records: dict[str, dict[str, object]] = {}
    reference_shape = None
    reference_affine = None
    occupancy = None
    for tissue in TISSUES:
        path = root / f"{tissue}.nii"
        try:
            image = nib.load(str(path))
        except (OSError, FileNotFoundError) as error:
            errors.append(f"cannot load {tissue} mask: {error}")
            continue
        if len(image.shape) != 3:
            errors.append(f"{tissue} mask is not three-dimensional: {image.shape}")
            continue
        if reference_shape is None:
            reference_shape = tuple(image.shape)
            reference_affine = np.asarray(image.affine, dtype=np.float64)
            occupancy = np.zeros(reference_shape, dtype=np.uint8)
        elif tuple(image.shape) != reference_shape or not np.allclose(
            np.asarray(image.affine), reference_affine, rtol=0.0, atol=1e-6
        ):
            errors.append(f"{tissue} geometry differs from the reference mask")

        nonzero_voxels = 0
        minimum_value = math.inf
        maximum_value = -math.inf
        invalid_voxels = 0
        for start in range(0, image.shape[0], slab_depth):
            stop = min(start + slab_depth, image.shape[0])
            data = np.asarray(image.dataobj[start:stop, :, :], dtype=np.float64)
            finite = np.isfinite(data)
            is_zero = np.isclose(data, 0.0, rtol=0.0, atol=binary_atol)
            is_one = np.isclose(data, 1.0, rtol=0.0, atol=binary_atol)
            valid = finite & (is_zero | is_one)
            invalid_voxels += int(np.count_nonzero(~valid))
            mask = finite & is_one
            nonzero_voxels += int(np.count_nonzero(mask))
            if np.any(finite):
                finite_values = data[finite]
                minimum_value = min(minimum_value, float(finite_values.min()))
                maximum_value = max(maximum_value, float(finite_values.max()))
            if occupancy is not None and tuple(image.shape) == reference_shape:
                occupancy[start:stop, :, :] += mask.astype(np.uint8)
        if invalid_voxels:
            errors.append(
                f"{tissue} mask has {invalid_voxels} non-finite or non-binary "
                f"voxels (absolute tolerance {binary_atol})"
            )
        if nonzero_voxels == 0:
            errors.append(f"{tissue} mask is empty")
        tissue_records[tissue] = {
            "shape": [int(value) for value in image.shape],
            "dtype": str(image.get_data_dtype()),
            "nonzero_voxels": nonzero_voxels,
            "minimum_value": minimum_value,
            "maximum_value": maximum_value,
            "invalid_binary_voxels": invalid_voxels,
            "spatial_zooms": [float(value) for value in image.header.get_zooms()[:3]],
            "spatial_unit": str(image.header.get_xyzt_units()[0]),
        }
    if occupancy is None:
        return {
            "root": str(root),
            "tissues": tissue_records,
            "overlapping_voxels": None,
            "overlap_tolerance_voxels": None,
            "uncovered_voxels": None,
            "ok": False,
            "errors": errors or ["no valid three-dimensional tissue masks"],
            "warnings": warnings,
        }
    overlapping_voxels = int(np.count_nonzero(occupancy > 1))
    uncovered_voxels = int(np.count_nonzero(occupancy == 0))
    overlap_tolerance_voxels = max(2, math.ceil(occupancy.size * 1e-7))
    if overlapping_voxels > overlap_tolerance_voxels:
        errors.append(
            f"tissue masks overlap in {overlapping_voxels} voxels; "
            f"tolerance is {overlap_tolerance_voxels}"
        )
    elif overlapping_voxels:
        warnings.append(
            f"tissue masks overlap in {overlapping_voxels} voxels, within the "
            f"registered tolerance of {overlap_tolerance_voxels}"
        )
    return {
        "root": str(root),
        "tissues": tissue_records,
        "overlapping_voxels": overlapping_voxels,
        "overlap_tolerance_voxels": overlap_tolerance_voxels,
        "uncovered_voxels": uncovered_voxels,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }
