#!/usr/bin/env python3
"""Deterministic tests for empirical calibration and continuous scanning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from empirical.calibration import (  # noqa: E402
    bids_selected_units,
    edf_selected_physical_dimensions,
    empirical_upper_threshold,
    false_alarms_per_hour,
    nonoverlapping_background_windows,
)


class CalibrationTests(unittest.TestCase):
    def test_background_windows_are_disjoint_and_excluded(self) -> None:
        windows = nonoverlapping_background_windows(100, 10, 10, [(18, 31), (70, 80)])
        self.assertTrue(np.array_equal(windows.starts, [40, 50, 60, 80, 90]))
        self.assertTrue(np.all(windows.starts[1:] >= windows.stops[:-1]))

    def test_threshold_and_rate_use_strict_exceedance(self) -> None:
        scores = np.arange(1200, dtype=np.float64)
        threshold = empirical_upper_threshold(scores, 1.0, 3.0)
        self.assertEqual(threshold, 1199.0)
        count, hours, rate = false_alarms_per_hour(scores, threshold, 3.0)
        self.assertEqual(count, 0)
        self.assertAlmostEqual(hours, 1.0)
        self.assertEqual(rate, 0.0)

    def test_threshold_rejects_empty_data(self) -> None:
        with self.assertRaises(ValueError):
            empirical_upper_threshold([], 1.0, 3.0)

    def test_edf_dimension_parser(self) -> None:
        labels = ["A1", "A2"]
        dimensions = ["uV", "uV"]
        signals = len(labels)
        fixed = bytearray(b" " * 256)
        fixed[252:256] = f"{signals:4d}".encode("ascii")

        def block(values: list[str], width: int) -> bytes:
            return b"".join(value.encode("ascii").ljust(width, b" ") for value in values)

        variable = b"".join(
            (
                block(labels, 16),
                block(["", ""], 80),
                block(dimensions, 8),
                block(["-100", "-100"], 8),
                block(["100", "100"], 8),
                block(["-32768", "-32768"], 8),
                block(["32767", "32767"], 8),
                block(["", ""], 80),
                block(["10", "10"], 8),
                block(["", ""], 32),
            )
        )
        self.assertEqual(len(variable), signals * 256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.edf"
            path.write_bytes(bytes(fixed) + variable + struct.pack("<" + "h" * 20, *([0] * 20)))
            self.assertEqual(edf_selected_physical_dimensions(str(path), labels), {"A1": "uV", "A2": "uV"})

    def test_blank_edf_dimension_requires_bids_unit_provenance(self) -> None:
        labels = ["A1", "A2"]
        signals = len(labels)
        fixed = bytearray(b" " * 256)
        fixed[252:256] = f"{signals:4d}".encode("ascii")

        def block(values: list[str], width: int) -> bytes:
            return b"".join(value.encode("ascii").ljust(width, b" ") for value in values)

        variable = b"".join(
            (
                block(labels, 16), block(["", ""], 80), block(["", ""], 8),
                block(["-100", "-100"], 8), block(["100", "100"], 8),
                block(["-32768", "-32768"], 8), block(["32767", "32767"], 8),
                block(["", ""], 80), block(["10", "10"], 8), block(["", ""], 32),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edf = root / "test.edf"
            edf.write_bytes(bytes(fixed) + variable + struct.pack("<" + "h" * 20, *([0] * 20)))
            channels = root / "test_channels.tsv"
            channels.write_text("name\ttype\tunits\nA1\tEEG\tμV\nA2\tEEG\tμV\n", encoding="utf-8")
            self.assertEqual(edf_selected_physical_dimensions(str(edf), labels), {"A1": "", "A2": ""})
            self.assertEqual(bids_selected_units(str(channels), labels), {"A1": "μV", "A2": "μV"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CalibrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "protocol": "calibration/continuous-false-positive-v1.1",
        "ok": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 2


if __name__ == "__main__":
    raise SystemExit(main())
