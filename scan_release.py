#!/usr/bin/env python3
"""Contamination scan for the public release tree.

Reports credentials, absolute private paths, development-stage workflow labels and
superseded-artefact markers. Exits non-zero when any violation is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCANNER_FILES = {"validate_release.py", "scan_release.py"}
TEXT_SUFFIXES = {".py", ".json", ".csv", ".txt", ".md", ".cff", ".yml"}

# Development-stage workstream labels.
WORKFLOW_LABELS = re.compile(r"priority|pilot|closure|planned|rejected|superseded|quarantine", re.I)
# Plotting / figure-generation code and the removed Pareto claim.
PLOT_LABELS = re.compile(r"matplotlib|seaborn|savefig|pyplot|plt\.|figsize|subplots|figure_dir|pareto", re.I)
# Internal protocol identifiers, container tags and development-history bookkeeping.
INTERNAL_LABELS = re.compile(
    r"HIPP-|hipp-source|analysis7b|protocol_v2|closure_stage|supersedes|v1_status|earlier_version"
    r"|container_image|mne_image|image_id|runtime_network",
)
NAME_LABELS = re.compile(r"plot|figure|pareto|reserves|^pipeline_", re.I)
SECRET = re.compile(r"BEGIN [A-Z ]*PRIVATE KEY|aws_secret_access_key|api[_-]?key\s*[:=]|password\s*[:=]", re.I)
PRIVATE_PATH = re.compile(r"/DATA/|[A-Za-z]:\\\\Users\\\\")


def iter_release_files():
    """Yield files shipped in the release, excluding Git's private metadata."""
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts:
            yield path


def main() -> None:
    violations: list[str] = []
    files = list(iter_release_files())
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        parts = path.relative_to(ROOT).parts
        if parts and parts[0] == "results":
            violations.append(f"derived result payload in code-only release: {rel}")
        if rel in {"configs/cohort_subjects.csv", "configs/cohort_subjects.txt"}:
            violations.append(f"cohort identifier payload in code-only release: {rel}")
        if WORKFLOW_LABELS.search(rel) or NAME_LABELS.search(path.stem):
            violations.append(f"label or plot/non-analysis name in path: {rel}")
        if path.suffix not in TEXT_SUFFIXES or path.name in SCANNER_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        for name, pattern in (
            ("workflow label", WORKFLOW_LABELS),
            ("plotting/figure code", PLOT_LABELS),
            ("internal protocol identifier", INTERNAL_LABELS),
        ):
            if pattern.search(text):
                violations.append(f"{name} in content: {rel}")
        if PRIVATE_PATH.search(text):
            violations.append(f"absolute private path: {rel}")
        if SECRET.search(text):
            violations.append(f"possible credential: {rel}")
    large = [p.relative_to(ROOT).as_posix() for p in files if p.stat().st_size > 2_000_000]
    print(f"scanned {len(files)} files")
    if large:
        print("note: files larger than 2 MB (confirm they are intentional):")
        for name in large:
            print(f"  {name}")
    if violations:
        for line in violations:
            print(f"VIOLATION: {line}")
        print(f"{len(violations)} violation(s)")
        sys.exit(1)
    print("no credentials, private paths or development-stage labels found")


if __name__ == "__main__":
    main()
