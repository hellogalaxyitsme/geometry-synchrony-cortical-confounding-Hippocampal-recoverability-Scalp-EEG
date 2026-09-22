#!/usr/bin/env python3
"""Adversarial numerical tests for mesial-event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT))
from empirical.mesial_events import fit_matched_filter, normalize_channel, paired_auc, spatial_scores, spearman, window  # noqa: E402


class Tests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_channel("sT3"), "T7"); self.assertEqual(normalize_channel("sFz"), "FZ")
    def test_matched_filter_recovers_injection(self):
        rng = np.random.default_rng(10); epochs = rng.normal(scale=0.2, size=(100, 8, 128)); topography = rng.normal(size=8); topography -= topography.mean()
        epochs[:, :, 60:69] += topography[None, :, None]
        filt, _ = fit_matched_filter(epochs[:60], window(64, 5), (slice(0, 30), slice(98, 128)), 0.1)
        auc = paired_auc(spatial_scores(epochs[60:], filt, window(64, 5)), spatial_scores(epochs[60:], filt, window(20, 5)))
        self.assertGreater(auc, 0.95)
    def test_spearman(self):
        self.assertAlmostEqual(spearman(np.arange(8), np.arange(8) ** 2), 1.0)
        self.assertAlmostEqual(spearman(np.arange(8), -np.arange(8)), -1.0)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests); result = unittest.TestResult(); suite.run(result)
    report = {"schema_version": 1, "protocol": "mesial-events/external-validation-v1", "ok": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": [text for _, text in result.failures], "errors": [text for _, text in result.errors]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["ok"] else 2


if __name__ == "__main__": raise SystemExit(main())
