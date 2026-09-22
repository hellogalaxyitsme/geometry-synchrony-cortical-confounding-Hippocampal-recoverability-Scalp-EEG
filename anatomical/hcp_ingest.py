"""Safe ingestion and anatomical QA for HCP-YA structural packages."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Iterable
import uuid
import zipfile


SUBJECT_RE = re.compile(r"^[0-9]{6}$")
ARCHIVE_SUFFIX = "_StructuralRecommended.zip"
MARKER_NAME = ".hcp_structural_extraction.json"


def required_relative_paths(subject: str) -> tuple[str, ...]:
    """Files needed for native-volume and cortical-surface preparation."""

    paths = [
        "T1w/T1w_acpc_dc_restore.nii.gz",
        "T1w/T2w_acpc_dc_restore.nii.gz",
        "T1w/brainmask_fs.nii.gz",
        "T1w/aparc+aseg.nii.gz",
        "T1w/wmparc.nii.gz",
        "T1w/ribbon.nii.gz",
        "T1w/xfms/acpc.mat",
    ]
    for hemisphere in ("L", "R"):
        for surface in ("white", "pial", "midthickness"):
            paths.append(
                f"T1w/Native/{subject}.{hemisphere}.{surface}.native.surf.gii"
            )
    return tuple(paths)


def load_subjects(path: Path) -> list[str]:
    subjects = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    subjects = [subject for subject in subjects if subject]
    invalid = [subject for subject in subjects if SUBJECT_RE.fullmatch(subject) is None]
    if invalid:
        raise ValueError(f"invalid subject IDs: {', '.join(invalid)}")
    if not subjects:
        raise ValueError("subject list is empty")
    if len(subjects) != len(set(subjects)):
        raise ValueError("subject list contains duplicates")
    return subjects


def load_verification_report(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("ok") is not True:
        raise ValueError(f"download verification report is not successful: {path}")
    packages = payload.get("packages")
    if not isinstance(packages, list):
        raise ValueError("download verification report has no package list")
    records: dict[str, dict[str, object]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("invalid package record in verification report")
        subject = str(package.get("subject", ""))
        if SUBJECT_RE.fullmatch(subject) is None or package.get("ok") is not True:
            raise ValueError(f"invalid verified package record for {subject!r}")
        if subject in records:
            raise ValueError(f"duplicate verified package for {subject}")
        records[subject] = package
    return records


def _validate_member_name(name: str, subject: str) -> str | None:
    if not name or "\\" in name or "\x00" in name:
        return "empty, NUL-containing, or backslash-containing member name"
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return "absolute or traversing member path"
    if not path.parts or path.parts[0] != subject:
        return f"member is not rooted below {subject}/"
    return None


def inspect_archive(
    archive_path: Path,
    subject: str,
    verified_package: dict[str, object] | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    if SUBJECT_RE.fullmatch(subject) is None:
        return {"subject": subject, "ok": False, "errors": ["invalid subject ID"]}
    expected_name = f"{subject}{ARCHIVE_SUFFIX}"
    if archive_path.name != expected_name:
        errors.append(f"archive name is not {expected_name}")
    if not archive_path.is_file():
        return {"subject": subject, "ok": False, "errors": ["archive is missing"]}

    archive_bytes = archive_path.stat().st_size
    if verified_package is not None:
        if int(verified_package.get("bytes", -1)) != archive_bytes:
            errors.append("archive byte count disagrees with verified download report")
        if str(verified_package.get("archive", "")) != archive_path.name:
            errors.append("archive name disagrees with verified download report")

    member_names: list[str] = []
    uncompressed_bytes = 0
    compressed_bytes = 0
    unsafe_members: list[str] = []
    non_regular_members: list[str] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            for info in infos:
                member_names.append(info.filename)
                uncompressed_bytes += int(info.file_size)
                compressed_bytes += int(info.compress_size)
                reason = _validate_member_name(info.filename, subject)
                if reason:
                    unsafe_members.append(f"{info.filename!r}: {reason}")
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(unix_mode)
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    non_regular_members.append(info.filename)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        errors.append(f"cannot read ZIP directory: {error}")
        infos = []

    duplicates = sorted(name for name, count in Counter(member_names).items() if count > 1)
    folded_duplicates = sorted(
        name for name, count in Counter(item.casefold() for item in member_names).items()
        if count > 1
    )
    if unsafe_members:
        errors.append(f"unsafe member paths: {unsafe_members[:5]}")
    if non_regular_members:
        errors.append(f"non-regular ZIP members: {non_regular_members[:5]}")
    if duplicates:
        errors.append(f"duplicate ZIP members: {duplicates[:5]}")
    if folded_duplicates:
        errors.append(f"case-folding duplicate ZIP members: {folded_duplicates[:5]}")
    if not member_names:
        errors.append("ZIP contains no members")
    if uncompressed_bytes > 10_000_000_000:
        errors.append("subject archive exceeds the 10 GB uncompressed safety limit")
    if compressed_bytes > 0 and uncompressed_bytes / compressed_bytes > 20.0:
        errors.append("subject archive exceeds the 20:1 expansion-ratio safety limit")

    name_set = set(member_names)
    required = [f"{subject}/{relative}" for relative in required_relative_paths(subject)]
    missing_required = sorted(name for name in required if name not in name_set)
    if missing_required:
        errors.append(f"missing required anatomy files: {missing_required}")

    freesurfer_prefix = f"{subject}/T1w/{subject}/"
    native_surface_prefix = f"{subject}/T1w/Native/"
    return {
        "subject": subject,
        "archive": archive_path.name,
        "archive_bytes": archive_bytes,
        "verified_md5": (
            str(verified_package.get("actual_md5")) if verified_package else None
        ),
        "entries": len(member_names),
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_bytes": uncompressed_bytes,
        "expansion_ratio": (
            uncompressed_bytes / compressed_bytes if compressed_bytes else None
        ),
        "required_files": len(required),
        "missing_required": missing_required,
        "native_gifti_surface_files": sum(
            name.startswith(native_surface_prefix) and name.endswith(".surf.gii")
            for name in member_names
        ),
        "conventional_freesurfer_subject_directory": any(
            name.startswith(freesurfer_prefix) for name in member_names
        ),
        "safe_member_paths": not unsafe_members,
        "regular_members_only": not non_regular_members,
        "unique_member_names": not duplicates and not folded_duplicates,
        "ok": not errors,
        "errors": errors,
    }


def audit_archives(
    archive_root: Path,
    subjects: Iterable[str],
    verified_packages: dict[str, dict[str, object]],
) -> dict[str, object]:
    subject_list = list(subjects)
    records = [
        inspect_archive(
            archive_root / f"{subject}{ARCHIVE_SUFFIX}",
            subject,
            verified_packages.get(subject),
        )
        for subject in subject_list
    ]
    unverified = sorted(set(subject_list).difference(verified_packages))
    extra_verified = sorted(set(verified_packages).difference(subject_list))
    if unverified:
        for record in records:
            if record["subject"] in unverified:
                record["ok"] = False
                record["errors"].append("subject is absent from download verification report")
    failed = [str(record["subject"]) for record in records if not record["ok"]]
    entry_distribution = Counter(str(record["entries"]) for record in records)
    return {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive_root),
        "subjects": len(subject_list),
        "verified_report_subjects": len(verified_packages),
        "unverified_subjects": unverified,
        "extra_verified_subjects": extra_verified,
        "failed_subjects": failed,
        "total_archive_bytes": sum(int(record["archive_bytes"]) for record in records),
        "total_uncompressed_bytes": sum(
            int(record["uncompressed_bytes"]) for record in records
        ),
        "entry_count_distribution": dict(sorted(entry_distribution.items())),
        "subjects_with_conventional_freesurfer_directory": sum(
            bool(record["conventional_freesurfer_subject_directory"])
            for record in records
        ),
        "subjects_with_native_gifti_surfaces": sum(
            int(record["native_gifti_surface_files"]) >= 6 for record in records
        ),
        "bem_ready": False,
        "bem_blocker": (
            "Packages contain cortical GIFTI surfaces but no scalp/outer-skull/"
            "inner-skull BEM surfaces; a pinned meshing environment is required."
        ),
        "ok": not failed and not unverified and not extra_verified,
        "archives": records,
    }


def _marker_matches(
    marker_path: Path,
    archive_path: Path,
    verified_md5: str,
    expected_entries: int,
) -> bool:
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("archive") == archive_path.name
        and payload.get("archive_bytes") == archive_path.stat().st_size
        and payload.get("verified_md5") == verified_md5
        and payload.get("zip_entries") == expected_entries
        and payload.get("status") == "complete"
    )


def extract_subject_archive(
    archive_path: Path,
    destination_root: Path,
    staging_root: Path,
    subject: str,
    verified_md5: str,
    resume: bool = True,
) -> dict[str, object]:
    """Extract one safe ZIP through same-filesystem staging and atomic rename."""

    inspection = inspect_archive(archive_path, subject)
    if not inspection["ok"]:
        raise ValueError(f"archive {subject} failed safety audit: {inspection['errors']}")
    destination_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / subject
    marker = destination / MARKER_NAME
    if destination.exists():
        if resume and destination.is_dir() and _marker_matches(
            marker, archive_path, verified_md5, int(inspection["entries"])
        ):
            return {
                "subject": subject,
                "status": "already_complete",
                "destination": str(destination),
                "uncompressed_bytes": int(inspection["uncompressed_bytes"]),
            }
        raise FileExistsError(
            f"refusing to overwrite unverified or incomplete destination: {destination}"
        )

    stage = staging_root / f"{subject}-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    extracted_bytes = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            for info in infos:
                reason = _validate_member_name(info.filename, subject)
                if reason:
                    raise ValueError(f"unsafe member {info.filename!r}: {reason}")
                output = stage.joinpath(*PurePosixPath(info.filename).parts)
                resolved_parent = output.parent.resolve(strict=False)
                if stage.resolve() not in (resolved_parent, *resolved_parent.parents):
                    raise ValueError(f"member escaped staging directory: {info.filename}")
                if info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, output.open("xb") as target:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                if output.stat().st_size != info.file_size:
                    raise IOError(f"extracted size mismatch for {info.filename}")
                extracted_bytes += info.file_size

        staged_subject = stage / subject
        missing = [
            relative
            for relative in required_relative_paths(subject)
            if not (staged_subject / relative).is_file()
        ]
        if missing:
            raise ValueError(f"staged subject is missing required files: {missing}")
        marker_payload = {
            "schema_version": 1,
            "status": "complete",
            "subject": subject,
            "archive": archive_path.name,
            "archive_bytes": archive_path.stat().st_size,
            "verified_md5": verified_md5,
            "zip_entries": int(inspection["entries"]),
            "uncompressed_bytes": int(inspection["uncompressed_bytes"]),
            "extracted_bytes": extracted_bytes,
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staged_subject / MARKER_NAME).write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged_subject, destination)
        stage.rmdir()
        return {
            "subject": subject,
            "status": "extracted",
            "destination": str(destination),
            "uncompressed_bytes": int(inspection["uncompressed_bytes"]),
        }
    except Exception as error:
        raise RuntimeError(f"extraction failed for {subject}; staging preserved at {stage}") from error


def _nifti_metadata(path: Path) -> tuple[dict[str, object], object]:
    import nibabel as nib
    import numpy as np

    image = nib.load(str(path))
    affine = np.asarray(image.affine, dtype=np.float64)
    spatial_zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    metadata = {
        "path": str(path),
        "shape": [int(value) for value in image.shape],
        "dtype": str(image.get_data_dtype()),
        "spatial_zooms": list(spatial_zooms),
        "affine_determinant": float(np.linalg.det(affine[:3, :3])),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "spatial_unit": str(image.header.get_xyzt_units()[0]),
    }
    return metadata, image


def _gifti_surface_metadata(path: Path) -> tuple[dict[str, object], object, object]:
    import nibabel as nib
    import numpy as np

    image = nib.load(str(path))
    pointset_intent = int(nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"])
    triangle_intent = int(nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"])
    pointsets = [array for array in image.darrays if int(array.intent) == pointset_intent]
    triangles = [array for array in image.darrays if int(array.intent) == triangle_intent]
    if len(pointsets) != 1 or len(triangles) != 1:
        raise ValueError(f"{path} must contain one pointset and one triangle array")
    vertices = np.asarray(pointsets[0].data, dtype=np.float64)
    faces = np.asarray(triangles[0].data, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError(f"invalid surface vertices in {path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"invalid surface faces in {path}")
    if faces.size == 0 or int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError(f"surface indices out of range in {path}")
    degenerate_index_faces = int(
        np.sum(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 0] == faces[:, 2])
        )
    )
    metadata = {
        "path": str(path),
        "vertices": int(vertices.shape[0]),
        "triangles": int(faces.shape[0]),
        "coordinate_min": vertices.min(axis=0).tolist(),
        "coordinate_max": vertices.max(axis=0).tolist(),
        "degenerate_index_triangles": degenerate_index_faces,
    }
    return metadata, vertices, faces


def audit_extracted_subject(subject_root: Path, subject: str) -> dict[str, object]:
    """Validate image geometry, hippocampal labels, and native surface topology."""

    import numpy as np

    errors: list[str] = []
    marker_path = subject_root / MARKER_NAME
    if not marker_path.is_file():
        errors.append(f"missing {MARKER_NAME}")
    missing = [
        relative
        for relative in required_relative_paths(subject)
        if not (subject_root / relative).is_file()
    ]
    if missing:
        errors.append(f"missing required files: {missing}")
        return {"subject": subject, "ok": False, "errors": errors}

    volume_files = {
        "t1w": "T1w/T1w_acpc_dc_restore.nii.gz",
        "t2w": "T1w/T2w_acpc_dc_restore.nii.gz",
        "brainmask": "T1w/brainmask_fs.nii.gz",
        "aparc_aseg": "T1w/aparc+aseg.nii.gz",
        "wmparc": "T1w/wmparc.nii.gz",
        "ribbon": "T1w/ribbon.nii.gz",
    }
    volumes: dict[str, dict[str, object]] = {}
    images: dict[str, object] = {}
    for label, relative in volume_files.items():
        metadata, image = _nifti_metadata(subject_root / relative)
        volumes[label] = metadata
        images[label] = image
        if len(metadata["shape"]) != 3 or any(value <= 0 for value in metadata["shape"]):
            errors.append(f"{label} is not a nonempty 3D image")
        if any(value <= 0.0 or not np.isfinite(value) for value in metadata["spatial_zooms"]):
            errors.append(f"{label} has invalid voxel sizes")
        if not np.isfinite(metadata["affine_determinant"]) or abs(
            float(metadata["affine_determinant"])
        ) < 1e-9:
            errors.append(f"{label} has a singular affine")

    reference_shape = tuple(images["t1w"].shape)
    reference_affine = np.asarray(images["t1w"].affine)
    for label, image in images.items():
        if tuple(image.shape) != reference_shape:
            errors.append(f"{label} shape differs from T1w")
        if not np.allclose(np.asarray(image.affine), reference_affine, rtol=0.0, atol=1e-5):
            errors.append(f"{label} affine differs from T1w")

    segmentation = np.asanyarray(images["aparc_aseg"].dataobj)
    voxel_volume_mm3 = abs(float(np.linalg.det(reference_affine[:3, :3])))
    left_voxels = int(np.count_nonzero(segmentation == 17))
    right_voxels = int(np.count_nonzero(segmentation == 53))
    if left_voxels == 0 or right_voxels == 0:
        errors.append("one or both FreeSurfer hippocampal labels are absent")

    surfaces: dict[str, dict[str, object]] = {}
    for hemisphere in ("L", "R"):
        topology = None
        vertex_count = None
        for surface in ("white", "pial", "midthickness"):
            relative = f"T1w/Native/{subject}.{hemisphere}.{surface}.native.surf.gii"
            try:
                metadata, vertices, faces = _gifti_surface_metadata(
                    subject_root / relative
                )
            except (OSError, ValueError) as error:
                errors.append(str(error))
                continue
            surfaces[f"{hemisphere}.{surface}"] = metadata
            if metadata["degenerate_index_triangles"] != 0:
                errors.append(f"{hemisphere}.{surface} has degenerate index triangles")
            if topology is None:
                topology = faces
                vertex_count = len(vertices)
            elif len(vertices) != vertex_count or not np.array_equal(faces, topology):
                errors.append(f"{hemisphere} native surfaces do not share topology")

    return {
        "subject": subject,
        "root": str(subject_root),
        "ok": not errors,
        "errors": errors,
        "volumes": volumes,
        "hippocampus": {
            "label_source": "T1w/aparc+aseg.nii.gz",
            "left_label": 17,
            "right_label": 53,
            "left_voxels": left_voxels,
            "right_voxels": right_voxels,
            "left_volume_mm3": left_voxels * voxel_volume_mm3,
            "right_volume_mm3": right_voxels * voxel_volume_mm3,
        },
        "native_surfaces": surfaces,
        "conventional_freesurfer_subject_directory": (
            subject_root / "T1w" / subject
        ).is_dir(),
        "bem_surfaces_present": all(
            (subject_root / "bem" / name).is_file()
            for name in ("inner_skull.surf", "outer_skull.surf", "outer_skin.surf")
        ),
    }
