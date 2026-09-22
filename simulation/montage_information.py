"""Numerical primitives for the montage physical-montage experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_montage(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    names = [row["name"] for row in rows]
    positions = np.asarray(
        [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in rows],
        dtype=np.float64,
    )
    if len(names) < 2 or len(names) != len(set(names)) or not np.all(np.isfinite(positions)):
        raise ValueError("invalid registered montage")
    return names, positions


def helmert_reference(count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("a referenced montage requires at least two electrodes")
    result = np.zeros((count - 1, count), dtype=np.float64)
    for row in range(1, count):
        scale = np.sqrt(row * (row + 1.0))
        result[row - 1, :row] = 1.0 / scale
        result[row - 1, row] = -row / scale
    return result


def farthest_order(
    positions: np.ndarray, seed_indices: list[int], candidate_indices: list[int]
) -> list[int]:
    """Deterministic spatial-dispersion order with a frozen seed montage."""
    positions = np.asarray(positions, dtype=np.float64)
    center = positions.mean(axis=0)
    directions = positions - center
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("electrode direction is undefined")
    directions /= norms[:, None]
    selected = list(dict.fromkeys(int(value) for value in seed_indices))
    candidates = sorted(set(int(value) for value in candidate_indices) - set(selected))
    if not selected:
        selected.append(candidates.pop(0))
    while candidates:
        selected_directions = directions[np.asarray(selected)]
        scores = []
        for index in candidates:
            distance = np.min(np.linalg.norm(selected_directions - directions[index], axis=1))
            scores.append((float(distance), -index, index))
        chosen = max(scores)[2]
        selected.append(chosen)
        candidates.remove(chosen)
    return selected


def physical_montages(
    names: list[str], positions: np.ndarray, conventional32: list[str], conventional64: list[str],
    inferior_additions: list[str], sizes: tuple[int, ...] = (128, 185, 256),
) -> tuple[dict[str, list[int]], dict[str, object]]:
    lookup = {name: index for index, name in enumerate(names)}
    requested = set(conventional32) | set(conventional64) | set(inferior_additions)
    missing = sorted(requested - set(lookup))
    if missing:
        raise ValueError(f"registered montage lacks frozen sites: {missing}")
    if len(conventional32) != 32 or len(set(conventional32)) != 32:
        raise ValueError("the conventional-32 definition is not exactly 32 unique sites")
    if len(conventional64) != 64 or len(set(conventional64)) != 64:
        raise ValueError("the conventional-64 definition is not exactly 64 unique sites")
    if not set(conventional32).issubset(conventional64):
        raise ValueError("conventional-32 is not nested in conventional-64")
    seed = [lookup[name] for name in conventional64]
    order = farthest_order(positions, seed, list(range(len(names))))
    montages: dict[str, list[int]] = {
        "conventional32": [lookup[name] for name in conventional32],
        "conventional64": seed,
    }
    for size in sizes:
        if size <= 64 or size > len(names):
            raise ValueError("invalid high-density montage size")
        montages[f"hd{size}"] = order[:size]
    montages["full"] = order
    augmented_names = conventional64 + [name for name in inferior_additions if name not in conventional64]
    montages["conventional64_plus_inferior"] = [lookup[name] for name in augmented_names]
    nested = ["conventional32", "conventional64", *[f"hd{s}" for s in sizes], "full"]
    nested_ok = all(set(montages[left]).issubset(montages[right]) for left, right in zip(nested, nested[1:]))
    report = {
        "nested_ladder": nested,
        "exactly_nested": nested_ok,
        "sizes": {name: len(indices) for name, indices in montages.items()},
        "inferior_added_names": [name for name in augmented_names if name not in conventional64],
        "face_neck_policy": "not evaluated: no validated physical face/neck digitization exists in the HCP forward products",
    }
    if not nested_ok:
        raise ValueError("physical montage hierarchy is not nested")
    return montages, report


def referenced_covariance(covariance: np.ndarray, indices: list[int]) -> np.ndarray:
    indices_array = np.asarray(indices, dtype=np.int64)
    reference = helmert_reference(len(indices))
    selected = np.asarray(covariance, dtype=np.float64)[np.ix_(indices_array, indices_array)]
    result = reference @ selected @ reference.T
    return 0.5 * (result + result.T)


def gaussian_information_bits(signal: np.ndarray, nuisance: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if signal.shape != nuisance.shape or signal.ndim != 2 or signal.shape[0] != signal.shape[1]:
        raise ValueError("signal and nuisance covariances must be equally sized square matrices")
    nuisance_values, nuisance_vectors = np.linalg.eigh(0.5 * (nuisance + nuisance.T))
    if nuisance_values[0] <= 0.0:
        raise ValueError("nuisance covariance is not positive definite")
    whitening = (nuisance_vectors / np.sqrt(nuisance_values)[None, :]) @ nuisance_vectors.T
    whitened = whitening @ (0.5 * (signal + signal.T)) @ whitening
    eigenvalues = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues[0] < -tolerance:
        raise ValueError("signal covariance is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return float(0.5 * np.sum(np.log2(1.0 + eigenvalues)))


def montage_information(
    signal_full: np.ndarray,
    cortical_full: np.ndarray,
    indices: list[int],
    signal_scale: float,
    cortical_scale: float,
    noise_variance: float,
) -> float:
    signal = signal_scale * referenced_covariance(signal_full, indices)
    cortical = cortical_scale * referenced_covariance(cortical_full, indices)
    nuisance = cortical + noise_variance * np.eye(len(indices) - 1)
    return gaussian_information_bits(signal, nuisance)
