#!/usr/bin/env python3
"""Validate the public code-only reproducibility release."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifests" / "source_manifest_sha256.csv"
TEXT_SUFFIXES = {".py", ".json", ".csv", ".txt", ".md", ".cff", ".yml"}
SCANNER_FILES = {"validate_release.py", "scan_release.py"}

WORKFLOW_LABELS = re.compile(
    r"priority|pilot|closure|planned|rejected|superseded|quarantine", re.I
)
PLOT_LABELS = re.compile(
    r"matplotlib|seaborn|savefig|pyplot|plt\.|figsize|subplots|figure_dir|pareto", re.I
)
INTERNAL_LABELS = re.compile(
    r"HIPP-|hipp-source|analysis7b|protocol_v2|closure_stage|supersedes|v1_status"
    r"|earlier_version|container_image|mne_image|image_id|runtime_network"
)
NAME_LABELS = re.compile(r"plot|figure|pareto|reserves|^pipeline_", re.I)
SECRET = re.compile(
    r"BEGIN [A-Z ]*PRIVATE KEY|aws_secret_access_key|api[_-]?key\s*[:=]|password\s*[:=]",
    re.I,
)
PRIVATE_PATH = re.compile(r"/DATA/|[A-Za-z]:\\\\Users\\\\")

IMPORT_CHECKS = [
    "theory.recoverability",
    "simulation.phase_models",
    "simulation.synthetic",
    "simulation.montage_information",
    "simulation.head_model_uncertainty",
    "empirical.injection_sources",
    "empirical.injection_benchmark",
    "empirical.calibration",
]


def iter_release_files():
    """Yield shipped files while excluding Git's private metadata."""
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts:
            yield path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(failures: list[str]) -> int:
    if not MANIFEST.exists():
        failures.append("missing SHA-256 manifest")
        return 0
    listed: set[str] = set()
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        rel = row["path"]
        listed.add(rel)
        path = ROOT / rel
        if not path.exists():
            failures.append(f"missing file {rel}")
        elif path.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch {rel}")
        elif sha256(path) != row["sha256"]:
            failures.append(f"hash mismatch {rel}")
    manifest_rel = MANIFEST.relative_to(ROOT).as_posix()
    present = {
        path.relative_to(ROOT).as_posix()
        for path in iter_release_files()
        if path.relative_to(ROOT).as_posix() != manifest_rel
    }
    for rel in sorted(present - listed):
        failures.append(f"unmanifested file {rel}")
    return len(rows)


def check_tree(failures: list[str]) -> int:
    files = list(iter_release_files())
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if WORKFLOW_LABELS.search(rel) or NAME_LABELS.search(path.stem):
            failures.append(f"development-stage or non-analysis name in path {rel}")
        if path.suffix not in TEXT_SUFFIXES or path.name in SCANNER_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        for label, pattern in (
            ("development-stage workflow label", WORKFLOW_LABELS),
            ("plotting or figure-generation code", PLOT_LABELS),
            ("development protocol identifier", INTERNAL_LABELS),
        ):
            if pattern.search(text):
                failures.append(f"{label} in {rel}")
        if PRIVATE_PATH.search(text):
            failures.append(f"absolute private path in {rel}")
        if SECRET.search(text):
            failures.append(f"possible credential in {rel}")
    return len(files)


def check_python(failures: list[str]) -> int:
    modules = 0
    for path in (item for item in iter_release_files() if item.suffix == ".py"):
        modules += 1
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="surrogateescape"))
        except SyntaxError as exc:
            failures.append(f"syntax error in {path.relative_to(ROOT).as_posix()}: {exc}")
    return modules


def check_imports(failures: list[str]) -> tuple[int, list[str]]:
    sys.path.insert(0, str(ROOT))
    imported, skipped = 0, []
    for name in IMPORT_CHECKS:
        try:
            importlib.import_module(name)
            imported += 1
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] in {"scipy", "mne", "sklearn"}:
                skipped.append(f"{name} (optional dependency '{exc.name}' unavailable)")
            else:
                failures.append(f"import failed for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"import failed for {name}: {exc}")
    return imported, skipped


def check_suites(failures: list[str]) -> list[str]:
    ran: list[str] = []
    with tempfile.TemporaryDirectory(prefix="release_validation_") as temporary:
        jobs = [
            ("workflows/check_theorem_stress.py", "theorem_stress_tests.json"),
            ("workflows/check_simulation_smoke.py", "simulation_smoke_tests.json"),
        ]
        for relative, output_name in jobs:
            command = [
                sys.executable,
                str(ROOT / relative),
                "--output",
                str(Path(temporary) / output_name),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if result.returncode:
                failures.append(
                    f"{relative} exited {result.returncode}: {result.stderr.strip()[-300:]}"
                )
            else:
                ran.append(relative)
    return ran


def main() -> None:
    failures: list[str] = []
    hashed = check_manifest(failures)
    files = check_tree(failures)
    modules = check_python(failures)
    imported, skipped = check_imports(failures)
    suites = check_suites(failures)

    print(f"manifest: {hashed} files hashed")
    print(f"tree:     {files} files scanned")
    print(f"python:   {modules} modules parsed, {imported} imported")
    print(f"suites:   {', '.join(suites) if suites else 'none ran'}")
    for note in skipped:
        print(f"SKIP:     {note}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("release validation passed")


if __name__ == "__main__":
    main()
