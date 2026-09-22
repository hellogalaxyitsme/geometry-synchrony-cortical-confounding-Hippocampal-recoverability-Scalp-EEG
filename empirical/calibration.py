"""Leakage-safe calibration utilities for continuous EEG false-alarm audits.

The functions here are deliberately independent of any hippocampal waveform.
They operate on already declared scalp scores, recording intervals, and
patient identifiers.  A threshold is always learned from other patients.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class WindowSet:
    starts: IntArray
    stops: IntArray

    @property
    def samples(self) -> int:
        return int(np.sum(self.stops - self.starts))


def nonoverlapping_background_windows(
    sample_count: int,
    start_sample: int,
    window_samples: int,
    excluded_intervals: Iterable[tuple[int, int]],
) -> WindowSet:
    """Tile a recording with disjoint windows that avoid every exclusion."""
    if sample_count < 1 or window_samples < 1 or not 0 <= start_sample < sample_count:
        raise ValueError("invalid recording/window dimensions")
    intervals = sorted(
        (max(0, int(left)), min(sample_count, int(right)))
        for left, right in excluded_intervals
        if int(right) > int(left)
    )
    starts: list[int] = []
    stops: list[int] = []
    for left in range(start_sample, sample_count - window_samples + 1, window_samples):
        right = left + window_samples
        if any(left < excluded_right and right > excluded_left for excluded_left, excluded_right in intervals):
            continue
        starts.append(left)
        stops.append(right)
    return WindowSet(np.asarray(starts, dtype=np.int64), np.asarray(stops, dtype=np.int64))


def empirical_upper_threshold(scores: ArrayLike, target_false_alarms_per_hour: float, window_seconds: float) -> float:
    """Return a conservative empirical threshold for disjoint scan windows.

    The exceedance probability target is ``rate * window_seconds / 3600``.
    The order statistic uses the finite-sample ``n + 1`` convention and falls
    back to the observed maximum when the requested tail is unresolved.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if (
        len(values) == 0
        or not np.all(np.isfinite(values))
        or target_false_alarms_per_hour <= 0.0
        or window_seconds <= 0.0
    ):
        raise ValueError("invalid threshold calibration inputs")
    exceedance = min(target_false_alarms_per_hour * window_seconds / 3600.0, 1.0)
    ordered = np.sort(values)
    rank = int(np.ceil((1.0 - exceedance) * (len(ordered) + 1))) - 1
    rank = min(max(rank, 0), len(ordered) - 1)
    return float(ordered[rank])


def false_alarms_per_hour(scores: ArrayLike, threshold: float, window_seconds: float) -> tuple[int, float, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)) or window_seconds <= 0.0:
        raise ValueError("invalid false-alarm inputs")
    hours = len(values) * window_seconds / 3600.0
    count = int(np.count_nonzero(values > threshold))
    return count, float(hours), float(count / hours)


def patient_equal_summary(values: ArrayLike) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("patient summary requires finite values")
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def edf_selected_physical_dimensions(path: str, requested: Iterable[str]) -> dict[str, str]:
    """Read only EDF signal labels and physical-dimension fields."""
    names = tuple(str(value) for value in requested)
    if not names or len(set(names)) != len(names):
        raise ValueError("requested EDF labels must be unique")
    with open(path, "rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError("truncated EDF fixed header")
        signals = int(fixed[252:256].decode("ascii").strip())
        variable = stream.read(signals * 256)
    if signals < 1 or len(variable) != signals * 256:
        raise ValueError("invalid EDF signal header")
    offset = 0

    def fields(width: int) -> list[str]:
        nonlocal offset
        block = variable[offset : offset + width * signals]
        offset += width * signals
        return [
            block[index * width : (index + 1) * width].decode("latin-1").strip()
            for index in range(signals)
        ]

    labels = fields(16)
    fields(80)  # transducer
    dimensions = fields(8)
    lookup = {label: dimension for label, dimension in zip(labels, dimensions)}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"EDF physical dimensions missing for {missing}")
    return {name: lookup[name] for name in names}


def bids_selected_units(path: str, requested: Iterable[str]) -> dict[str, str]:
    """Read channel units from a BIDS ``channels.tsv`` sidecar.

    Some valid EDF exports leave their optional physical-dimension text blank
    while retaining calibrated physical minima/maxima. In that case the BIDS
    channel table is the authoritative unit declaration. Duplicate labels,
    absent labels, and missing unit declarations fail closed.
    """
    names = tuple(str(value) for value in requested)
    if not names or len(set(names)) != len(names):
        raise ValueError("requested BIDS labels must be unique")
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or "name" not in rows[0] or "units" not in rows[0]:
        raise ValueError("invalid BIDS channels table")
    labels = [str(row["name"]) for row in rows]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate labels in BIDS channels table")
    lookup = {str(row["name"]): str(row["units"]).strip() for row in rows}
    missing = [name for name in names if name not in lookup or not lookup[name]]
    if missing:
        raise ValueError(f"BIDS channel units missing for {missing}")
    return {name: lookup[name] for name in names}
