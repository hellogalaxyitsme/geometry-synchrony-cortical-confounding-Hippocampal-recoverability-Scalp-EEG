#!/usr/bin/env python3
"""Independent audit of the source-ensemble retained-power normalization artifacts.

The audit recomputes every saved normalization field of the frozen
``hcp/source-ensembles-v1.1`` result set from the authoritative
forward lead fields and source metadata:

    z_E(q) = sum_{i in E} u_i q_i g_i
    D_E    = sum_{i in E} u_i ||g_i||^2
    R_E(q) = ||z_E(q)||^2 / D_E
    P_E(q) = (A_E / A_full)^2 ||z_E(q)||^2 / D_full

with positive quadrature areas ``a_i``, ``u_i = a_i / A_E`` and unit phases
``q_i``.  The arithmetic is independent of ``phase_locked_metrics`` and
``patch_coherence_metrics``.  Deterministic, patch-coherence and seeded
random-phase rows are all checked; the audit fails closed on missing files or
rows, subject/shape/area/phase defects, hash mismatches and tolerance excess.

Per-subject reports declare container paths such as
``/forward/<subject>/hippunfold-hippocampal-fixed-leadfield.npy`` that do not
exist on the audit host.  Declared inputs rooted at ``/forward/`` are resolved
to ``<forward_root>/<subject>/<basename>`` before hashing, and both the
declared container path and the resolved host path are recorded in the audit
JSON under ``declared_input_resolution``.  A missing or mismatched input still
fails closed.  The post-hoc retained-power addendum is hashed separately as an
auditor input and is never treated as part of the frozen implementation
manifest.

Examples
--------
subset two subjects into temporary outputs::

    python workflows/audit_retained_power_normalization.py \
        --limit 2 --output /tmp/retained_power_audit.json \
        --subject-table /tmp/retained_power_audit_subjects.csv

Full 50-subject audit against the frozen AI-lab roots (defaults come from the
frozen configuration)::

    python workflows/audit_retained_power_normalization.py \
        --output docs/agent-work/retained-power-normalization/audit.json \
        --subject-table docs/agent-work/retained-power-normalization/audit_subjects.csv

Neither output is written unless its path is given explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anatomical.retained_power_audit import (  # noqa: E402
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    FORMULA_VERSION,
    NOT_RECOMPUTED_COLUMNS,
    PROTOCOL,
    AuditError,
    SubjectGeometry,
    SubjectRecorder,
    atomic_write_csv,
    atomic_write_json,
    build_subject_geometry,
    expected_row_counts,
    load_fixed_leadfield,
    load_metadata,
    make_ledgers,
    read_csv_rows,
    resolve_declared_input,
    sha256_file,
    synthetic_checks_passed,
    synthetic_identity_checks,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "source_ensembles_v1_1.json"
RESULT_TABLES = (
    "deterministic_conditions.csv",
    "patch_coherence_conditions.csv",
    "random_phase_conditions.csv",
)
READ_INPUTS = ("fixed_leadfield", "metadata")
ADDENDUM = "hcp_hippunfold_retained_power_normalization_addendum_v1.md"
DOCUMENTATION_MANIFEST_KEYS = ("hcp_hippunfold_ensembles_protocol.md",)
RECOMMENDED_REMOTE_COMMAND = (
    "python workflows/audit_retained_power_normalization.py "
    "--output docs/agent-work/retained-power-normalization/audit.json "
    "--subject-table docs/agent-work/retained-power-normalization/audit_subjects.csv"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: object, project_root: Path) -> Path:
    text = str(value)
    path = Path(text)
    if path.is_absolute() or text.startswith(("/", "\\")):
        return path
    return project_root / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently recompute the retained-power normalization fields of "
            "the frozen source-ensemble ensembles and compare them with every saved "
            "subject row."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "The audit writes nothing unless --output and --subject-table are "
            "given explicitly. Exit status is 0 when every audited row agrees "
            "with the independent recomputation and 2 otherwise."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="frozen source-ensemble configuration providing the default roots",
    )
    parser.add_argument(
        "--forward-root",
        default=None,
        help=(
            "directory holding per-subject fixed lead fields and source "
            "metadata; report-declared container paths "
            "/forward/<subject>/<basename> resolve to "
            "<forward-root>/<subject>/<basename>"
        ),
    )
    parser.add_argument(
        "--result-root",
        default=None,
        help="directory holding per-subject source-ensemble ensemble outputs",
    )
    parser.add_argument(
        "--status-root",
        default=None,
        help="directory holding full50_summary.json and the report-hash registry",
    )
    parser.add_argument(
        "--subjects-file",
        default=None,
        help="cohort file overriding the configured frozen subject list",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="explicit subject identifiers, overriding the cohort file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="audit only the first N subjects (sorted) for a subset check",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path of the machine-readable audit JSON to write",
    )
    parser.add_argument(
        "--subject-table",
        required=True,
        help="path of the subject-level audit CSV to write",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_TOLERANCE,
        help="relative tolerance for saved-versus-recomputed comparisons",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=DEFAULT_ABSOLUTE_TOLERANCE,
        help="absolute tolerance for saved-versus-recomputed comparisons",
    )
    parser.add_argument(
        "--skip-implementation-hashes",
        action="store_true",
        help="do not verify the declared implementation hashes",
    )
    parser.add_argument(
        "--strict-documentation-hash",
        action="store_true",
        help=(
            "treat a declared documentation hash mismatch as a failure; by "
            "default only a mismatch in executed implementation code fails"
        ),
    )
    parser.add_argument(
        "--skip-artifact-hashes",
        action="store_true",
        help="do not verify report/input/output artifact hashes",
    )
    return parser


def _subjects(
    args: argparse.Namespace,
    config: Mapping[str, object],
    project_root: Path,
    warnings: list[str],
) -> list[str]:
    if args.subjects:
        values = [str(value) for value in args.subjects]
        if len(set(values)) != len(values):
            raise AuditError("--subjects contains duplicates")
        warnings.append("subject list supplied explicitly; cohort hash not checked")
        return sorted(values)

    declared = str(config.get("subjects_file", "configs/cohort_subjects.txt"))
    path = (
        _resolve(args.subjects_file, project_root)
        if args.subjects_file
        else _resolve(declared, project_root)
    )
    if not path.is_file():
        raise AuditError(f"missing cohort file: {path}")
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values or len(set(values)) != len(values):
        raise AuditError(f"cohort file has duplicate or empty entries: {path}")
    declared_hash = config.get("subjects_file_sha256")
    if declared_hash and not args.subjects_file:
        actual = sha256_file(path)
        if actual != declared_hash:
            raise AuditError(
                f"frozen cohort file hash changed: {path} ({actual} != {declared_hash})"
            )
    elif args.subjects_file:
        warnings.append("cohort file overridden; declared cohort hash not enforced")
    return sorted(values)


def _implementation_hashes(
    config: Mapping[str, object],
    project_root: Path,
    strict_documentation: bool,
) -> tuple[dict[str, object], list[str], list[str]]:
    declared = dict(config.get("implementation_sha256", {}))  # type: ignore[arg-type]
    entries: dict[str, object] = {}
    code_mismatches: list[str] = []
    documentation_mismatches: list[str] = []
    for relative, expected in sorted(declared.items()):
        path = project_root / relative
        actual = sha256_file(path) if path.is_file() else None
        matches = actual == expected
        kind = (
            "documentation" if relative in DOCUMENTATION_MANIFEST_KEYS else "code"
        )
        entries[relative] = {
            "kind": kind,
            "declared_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            message = f"declared implementation hash mismatch: {relative}"
            if kind == "documentation":
                documentation_mismatches.append(message)
            else:
                code_mismatches.append(message)
    if strict_documentation:
        code_mismatches.extend(documentation_mismatches)
    return entries, code_mismatches, documentation_mismatches


def _addendum_record(project_root: Path) -> dict[str, object]:
    """Describe the retained-power addendum as a distinct auditor input.

    The addendum is a post-hoc implementation-level clarification written
    during the audit.  It is deliberately *not* part of the frozen v1.1
    implementation manifest, so it is hashed and reported separately and never
    compared against the frozen configuration.
    """

    path = project_root / ADDENDUM
    present = path.is_file()
    return {
        "path": ADDENDUM,
        "kind": "auditor_input",
        "present": present,
        "sha256": sha256_file(path) if present else None,
        "part_of_frozen_implementation_manifest": False,
        "note": (
            "post-hoc implementation-level clarification and audit "
            "specification; it documents the implemented retained-power "
            "normalization and does not redefine or alter any frozen v1.1 "
            "output"
        ),
    }


def _verify_report_hashes(
    subject: str,
    directory: Path,
    report: Mapping[str, Any],
    report_hash: str | None,
    forward_root: Path,
    errors: list[str],
) -> dict[str, object]:
    """Verify declared input and output hashes for one subject.

    Declared input paths are resolved from their container form
    (``/forward/<subject>/<basename>``) to ``<forward_root>/<subject>/<basename>``
    before hashing, and both the declared path and the resolved host path are
    recorded.  A missing or hash-mismatched input remains a hard failure.
    """

    record: dict[str, object] = {"report_sha256": report_hash, "outputs": {}, "inputs": {}}
    outputs = dict(report.get("outputs", {}))
    inputs = dict(report.get("inputs", {}))
    for name in RESULT_TABLES:
        if name not in outputs:
            errors.append(f"{subject}: report does not declare output {name}")
    for name, metadata in outputs.items():
        path = directory / str(name)
        if not path.is_file():
            errors.append(f"{subject}: missing declared output {name}")
            record["outputs"][str(name)] = None
            continue
        actual = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        matches = actual["bytes"] == metadata.get("bytes") and actual[
            "sha256"
        ] == metadata.get("sha256")
        if not matches:
            errors.append(f"{subject}: declared output hash mismatch for {name}")
        record["outputs"][str(name)] = {**actual, "matches": matches}
    for name in READ_INPUTS:
        metadata = inputs.get(name)
        if metadata is None:
            record["inputs"][name] = None
            continue
        resolution = resolve_declared_input(subject, metadata.get("path", ""), forward_root)
        path = resolution.host_path
        entry: dict[str, object] = {
            "declared_path": resolution.declared,
            "resolved_host_path": str(path),
            "resolved_from_container": resolution.resolved_from_container,
        }
        if not path.is_file():
            errors.append(f"{subject}: missing report input {name} at {path}")
            entry["present"] = False
            record["inputs"][name] = entry
            continue
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        matches = actual["bytes"] == metadata.get("bytes") and actual[
            "sha256"
        ] == metadata.get("sha256")
        if not matches:
            errors.append(f"{subject}: declared input hash mismatch for {name} at {path}")
        entry.update(actual)
        entry["present"] = True
        entry["matches"] = matches
        record["inputs"][name] = entry
    return record


def _audit_subject(
    subject: str,
    directory: Path,
    forward_root: Path,
    config: Mapping[str, object],
    ledgers: Mapping[str, object],
    relative_tolerance: float,
    absolute_tolerance: float,
    verify_hashes: bool,
) -> tuple[dict[str, object], list[str], dict[str, object]]:
    subject_errors: list[str] = []
    report_path = directory / "report.json"
    marker_path = directory / ".hippunfold_ensembles.json"
    report = _load_json(report_path)
    if report.get("protocol") != PROTOCOL:
        subject_errors.append(f"{subject}: report protocol mismatch")
    if report.get("subject") != subject:
        subject_errors.append(f"{subject}: report subject identity mismatch")
    if report.get("ok") is not True:
        subject_errors.append(f"{subject}: report is not marked OK")
    if not all(dict(report.get("qc", {})).values()):
        subject_errors.append(f"{subject}: report QC contains a false entry")
    report_hash = sha256_file(report_path)
    if marker_path.is_file():
        marker = _load_json(marker_path)
        if marker.get("report_sha256") not in (None, report_hash):
            subject_errors.append(f"{subject}: completion marker report hash mismatch")
        if marker.get("status") != "complete" or marker.get("subject") != subject:
            subject_errors.append(f"{subject}: completion marker is not complete")

    hash_record: dict[str, object] = {}
    if verify_hashes:
        hash_record = _verify_report_hashes(
            subject, directory, report, report_hash, forward_root, subject_errors
        )

    metadata_path = forward_root / subject / "hippunfold-hippocampal-source-metadata.npz"
    leadfield_path = forward_root / subject / "hippunfold-hippocampal-fixed-leadfield.npy"
    metadata = load_metadata(metadata_path)
    areas = metadata["area_weights_m2"]
    sensors = int(config.get("expected_sensors", 339))
    leadfield = load_fixed_leadfield(leadfield_path, sensors, int(len(areas)))
    geometry: SubjectGeometry = build_subject_geometry(
        subject, metadata, leadfield, config
    )
    recorder = SubjectRecorder(
        geometry, config, relative_tolerance, absolute_tolerance, ledgers
    )

    expected = expected_row_counts(config)
    if len(geometry.supports) != expected["supports"]:
        recorder.fail(
            f"reconstructed {len(geometry.supports)} supports, expected "
            f"{expected['supports']}"
        )

    families = (
        ("deterministic", "deterministic_conditions.csv", recorder.audit_deterministic),
        ("patch_coherence", "patch_coherence_conditions.csv", recorder.audit_patch_coherence),
        ("random_phase", "random_phase_conditions.csv", recorder.audit_random_phase),
    )
    for family, filename, method in families:
        rows = read_csv_rows(directory / filename)
        if len(rows) != expected[family]:
            recorder.fail(
                f"{family}: saved row count {len(rows)} != expected {expected[family]}"
            )
        for row in rows:
            if str(row.get("subject", "")) != subject:
                recorder.fail(f"{family}: saved row carries a different subject")
                break
        method(rows)

    audit = recorder.result()
    row = audit.as_row()
    row["report_sha256"] = report_hash
    combined = list(subject_errors) + list(audit.errors)
    row["status"] = "passed" if not combined else "failed"
    row["errors"] = "; ".join(combined)
    row["error_count"] = len(combined)
    return row, combined, hash_record


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = PROJECT_ROOT
    warnings: list[str] = []
    errors: list[str] = []
    config_path = _resolve(args.config, project_root)
    config = _load_json(config_path)
    if config.get("protocol") != PROTOCOL:
        raise AuditError(f"unexpected protocol in {config_path}")

    forward_root = (
        _resolve(args.forward_root, project_root)
        if args.forward_root
        else _resolve(config["hippunfold_forward_root"], project_root)
    )
    result_root = (
        _resolve(args.result_root, project_root)
        if args.result_root
        else _resolve(config["output_root"], project_root)
    )
    status_root = (
        _resolve(args.status_root, project_root)
        if args.status_root
        else result_root
    )
    subjects = _subjects(args, config, project_root, warnings)
    cohort_size = len(subjects)
    if args.limit is not None:
        if args.limit < 1:
            raise AuditError("--limit must be a positive integer")
        if args.limit < len(subjects):
            warnings.append(
                f"--limit audited {args.limit} of {len(subjects)} declared subjects"
            )
        subjects = subjects[: args.limit]

    identity_records = synthetic_identity_checks()
    if not synthetic_checks_passed(identity_records):
        errors.append("synthetic normalization identity checks failed")

    implementation_entries: dict[str, object] = {}
    if not args.skip_implementation_hashes:
        implementation_entries, code_mismatches, documentation_mismatches = (
            _implementation_hashes(config, project_root, args.strict_documentation_hash)
        )
        errors.extend(code_mismatches)
        for message in documentation_mismatches:
            warnings.append(
                f"{message}; the frozen configuration still pins the "
                "pre-revision protocol text and the root owns re-pinning"
            )
    else:
        warnings.append("implementation hash verification skipped by request")

    report_hashes: dict[str, str] = {}
    summary_path = status_root / "source_ensembles_population_report.json"
    if summary_path.is_file():
        try:
            summary = _load_json(summary_path)
        except (AuditError, json.JSONDecodeError) as error:
            warnings.append(f"status registry unreadable: {error}")
        else:
            if summary.get("protocol") == PROTOCOL:
                report_hashes = {
                    str(key): str(value)
                    for key, value in dict(
                        summary.get("subject_report_sha256", {})
                    ).items()
                }
            else:
                warnings.append("status registry protocol mismatch; hashes unused")
    else:
        warnings.append(f"status registry absent: {summary_path}")

    ledgers = make_ledgers(args.relative_tolerance, args.absolute_tolerance)
    subject_rows: list[dict[str, object]] = []
    source_counts: list[int] = []
    sensor_counts: list[int] = []
    resolved_inputs: dict[str, object] = {}
    for subject in subjects:
        directory = result_root / subject
        try:
            row, subject_errors, hash_record = _audit_subject(
                subject,
                directory,
                forward_root,
                config,
                ledgers,
                args.relative_tolerance,
                args.absolute_tolerance,
                not args.skip_artifact_hashes,
            )
        except (AuditError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            row = {
                "subject": subject,
                "status": "error",
                "error_count": 1,
                "errors": f"{type(error).__name__}: {error}",
            }
            subject_errors = [f"{subject}: {type(error).__name__}: {error}"]
            hash_record = {}
        declared_inputs = dict(hash_record.get("inputs", {})) if hash_record else {}
        if declared_inputs:
            resolved_inputs[subject] = declared_inputs
        if report_hashes:
            declared = report_hashes.get(subject)
            if declared is None:
                subject_errors.append(f"{subject}: absent from the report-hash registry")
            elif row.get("report_sha256") not in (None, declared):
                subject_errors.append(f"{subject}: report hash disagrees with registry")
            if subject_errors:
                row["errors"] = "; ".join(subject_errors)
                row["error_count"] = len(subject_errors)
                row["status"] = "failed"
        errors.extend(subject_errors)
        subject_rows.append(row)
        if isinstance(row.get("sources"), int):
            source_counts.append(int(row["sources"]))
        if isinstance(row.get("sensors"), int):
            sensor_counts.append(int(row["sensors"]))
        print(
            json.dumps(
                {
                    "subject": subject,
                    "status": row.get("status"),
                    "error_count": row.get("error_count"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    expected = expected_row_counts(config)
    fields = {name: ledger.as_record() for name, ledger in sorted(ledgers.items())}
    counts = {
        "subjects_available": cohort_size,
        "subjects_selected_in_run": len(subjects),
        "subjects_audited": len(subject_rows),
        "subjects_passed": sum(1 for row in subject_rows if row.get("status") == "passed"),
        "expected_rows": expected,
        "audited_rows": {
            family: int(sum(int(row.get(key, 0)) for row in subject_rows))
            for family, key in (
                ("deterministic", "deterministic_rows"),
                ("patch_coherence", "patch_coherence_rows"),
                ("random_phase", "random_phase_rows"),
            )
        },
        "sensors_minimum": min(sensor_counts) if sensor_counts else None,
        "sensors_maximum": max(sensor_counts) if sensor_counts else None,
        "source_counts_minimum": min(source_counts) if source_counts else None,
        "source_counts_maximum": max(source_counts) if source_counts else None,
    }
    maximum_absolute = max(
        (ledger.maximum_absolute_error for ledger in ledgers.values()), default=0.0
    )
    maximum_relative = max(
        (ledger.maximum_relative_error for ledger in ledgers.values()), default=0.0
    )
    payload = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "ok": not errors,
        "protocol": PROTOCOL,
        "formula_version": FORMULA_VERSION,
        "formula": {
            "sensor_field": "z_E(q) = sum_{i in E} u_i q_i g_i",
            "incoherent_reference": (
                "D_E = sum_{i in E} u_i ||g_i||^2 = tr(G_E diag(u) G_E^T)"
            ),
            "support_normalized_ratio": "R_E(q) = ||z_E(q)||^2 / D_E",
            "whole_sheet_ratio": (
                "P_E(q) = (A_E / A_full)^2 ||z_E(q)||^2 / D_full"
            ),
            "bound": "0 <= R_E(q) <= 1 by weighted Jensen",
            "interpretation": (
                "area-normalized current-density quadrature relative to the "
                "area-weighted incoherent reference; absolute uniform "
                "current-density amplitude cancels"
            ),
            "not_this": [
                (
                    "the generic ratio ||sum_i w_i g_i||^2 / "
                    "sum_i |w_i|^2 ||g_i||^2"
                ),
                (
                    "the independent fixed-magnitude uniform-phase expectation "
                    "sum_i u_i^2 ||g_i||^2"
                ),
            ],
        },
        "generated_at_utc": _now(),
        "auditor": {
            "module": "anatomical/retained_power_audit.py",
            "module_sha256": sha256_file(
                project_root / "anatomical" / "retained_power_audit.py"
            ),
            "script": "workflows/audit_retained_power_normalization.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "roots": {
            "forward_root": str(forward_root),
            "result_root": str(result_root),
            "status_root": str(status_root),
        },
        "tolerances": {
            "relative": args.relative_tolerance,
            "absolute": args.absolute_tolerance,
            "phase_unit": 1e-12,
        },
        "implementation_hashes": implementation_entries,
        "normalization_addendum": _addendum_record(project_root),
        "identity_outcomes": {
            "passed": synthetic_checks_passed(identity_records),
            "checks": identity_records,
        },
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
        "fields": fields,
        "subjects": subject_rows,
        "declared_input_resolution": resolved_inputs,
        "not_recomputed_columns": {
            "columns": list(NOT_RECOMPUTED_COLUMNS),
            "reason": (
                "nuisance-whitened spectral indices require the legacy cortical "
                "lead field and its standardized covariance; they are not part "
                "of the retained-power normalization contract"
            ),
        },
        "limitations": [
            "the audit recomputes normalization arithmetic, not the forward model",
            "hashes are only verified where the artifacts declare them",
            "no physiological or empirical inference is authorized here",
        ],
        "recommended_remote_command": RECOMMENDED_REMOTE_COMMAND,
        "physiological_inference_authorized": False,
        "shared_storage_touched": False,
    }
    atomic_write_json(Path(args.output), payload)
    atomic_write_csv(Path(args.subject_table), subject_rows)
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "status": payload["status"],
                "counts": counts,
                "maximum_absolute_error": maximum_absolute,
                "maximum_relative_error": maximum_relative,
                "error_count": len(errors),
                "warning_count": len(warnings),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
