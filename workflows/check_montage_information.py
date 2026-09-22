#!/usr/bin/env python3
"""Deterministic adversarial tests for montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from simulation.montage_information import (  # noqa: E402
    farthest_order,
    gaussian_information_bits,
    helmert_reference,
    montage_information,
)


class MontageInformationTests(unittest.TestCase):
    def test_helmert_is_orthonormal_and_reference_free(self) -> None:
        matrix = helmert_reference(17)
        self.assertTrue(np.allclose(matrix @ matrix.T, np.eye(16), atol=1e-13))
        self.assertTrue(np.allclose(matrix @ np.ones(17), 0.0, atol=1e-13))

    def test_farthest_order_is_deterministic_and_nested(self) -> None:
        rng = np.random.default_rng(8)
        positions = rng.normal(size=(40, 3))
        first = farthest_order(positions, [0, 3, 7], list(range(40)))
        second = farthest_order(positions, [0, 3, 7], list(range(40)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual(len(set(first)), 40)
        self.assertTrue(set(first[:10]).issubset(first[:20]))

    def test_information_matches_scalar_formula(self) -> None:
        signal = np.diag([2.0, 3.0])
        nuisance = np.diag([4.0, 5.0])
        observed = gaussian_information_bits(signal, nuisance)
        expected = 0.5 * np.sum(np.log2(1.0 + np.asarray([0.5, 0.6])))
        self.assertAlmostEqual(observed, float(expected), places=13)

    def test_nested_physical_measurements_are_monotone(self) -> None:
        rng = np.random.default_rng(80)
        h = rng.normal(size=(12, 3)); c = rng.normal(size=(12, 5))
        signal = h @ h.T; cortical = c @ c.T
        values = [
            montage_information(signal, cortical, list(range(count)), 0.2, 1.0, 0.1)
            for count in (4, 6, 9, 12)
        ]
        self.assertGreaterEqual(min(np.diff(values)), -1e-10)

    def test_common_mode_does_not_change_information(self) -> None:
        rng = np.random.default_rng(800)
        h = rng.normal(size=(8, 2)); c = rng.normal(size=(8, 4))
        base = montage_information(h @ h.T, c @ c.T, list(range(8)), 0.2, 1.0, 0.3)
        h2 = h + np.ones((8, 1)) @ rng.normal(size=(1, 2))
        c2 = c + np.ones((8, 1)) @ rng.normal(size=(1, 4))
        shifted = montage_information(h2 @ h2.T, c2 @ c2.T, list(range(8)), 0.2, 1.0, 0.3)
        self.assertAlmostEqual(base, shifted, places=11)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MontageInformationTests)
    result = unittest.TestResult(); suite.run(result)
    payload = {
        "schema_version": 1,
        "protocol": "montage/physical-value-v1",
        "ok": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": [text for _, text in result.failures],
        "errors": [text for _, text in result.errors],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
