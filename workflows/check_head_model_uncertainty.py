#!/usr/bin/env python3
"""Deterministic tests for head-model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from simulation.head_model_uncertainty import radial_shift, rotated_directions  # noqa: E402


class Tests(unittest.TestCase):
    def test_rotation_has_declared_angle(self):
        rng = np.random.default_rng(9); directions = rng.normal(size=(100, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
        rotated = rotated_directions(directions, 17.0, 90); cosines = np.sum(directions * rotated, axis=1)
        self.assertTrue(np.allclose(cosines, np.cos(np.deg2rad(17.0)), atol=1e-12))
        self.assertTrue(np.allclose(np.linalg.norm(rotated, axis=1), 1.0, atol=1e-13))
    def test_radial_shift_is_exact(self):
        rng = np.random.default_rng(91); vertices = rng.normal(size=(200, 3)); vertices *= 80.0 / np.linalg.norm(vertices, axis=1)[:, None]
        shifted = radial_shift(vertices, 0.25)
        self.assertTrue(np.allclose(np.linalg.norm(shifted - vertices, axis=1), 0.25, atol=1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests); result = unittest.TestResult(); suite.run(result)
    report = {"schema_version": 1, "protocol": "uncertainty/head-model-v1", "ok": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": [text for _, text in result.failures], "errors": [text for _, text in result.errors]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
