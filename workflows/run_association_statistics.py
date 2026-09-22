#!/usr/bin/env python3
"""Run the reported template-association statistics from frozen compact inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = "associations/template-recoverability-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(master: int, *parts: object) -> int:
    text = "|".join([str(master), *(str(value) for value in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def auc(positive: Iterable[float], negative: Iterable[float]) -> float:
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0 or not np.all(np.isfinite(np.r_[pos, neg])):
        raise ValueError("invalid AUC inputs")
    ranks = rankdata(np.r_[pos, neg], method="average")
    return float((np.sum(ranks[: len(pos)]) - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def paired_bootstrap(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 10000):
        stop = min(replicates, start + 10000)
        indices = generator.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = np.mean(values[indices], axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def paired_swap_p(values: np.ndarray, replicates: int, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(np.mean(values)))
    generator = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, replicates, 20000):
        size = min(20000, replicates - start)
        signs = generator.integers(0, 2, size=(size, len(values)), dtype=np.int8) * 2 - 1
        estimates = np.abs(np.mean(signs * values[None, :], axis=1))
        extreme += int(np.count_nonzero(estimates >= observed - 1e-15))
    return float((extreme + 1) / (replicates + 1))


def omnibus_patient_block_p(matrix: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    grand = float(np.mean(matrix))
    observed = float(matrix.shape[0] * np.sum((np.mean(matrix, axis=0) - grand) ** 2))
    generator = np.random.default_rng(seed)
    extreme = 0
    for _ in range(replicates):
        permuted = np.vstack([row[generator.permutation(matrix.shape[1])] for row in matrix])
        value = float(matrix.shape[0] * np.sum((np.mean(permuted, axis=0) - grand) ** 2))
        extreme += int(value >= observed - 1e-15)
    return observed, float((extreme + 1) / (replicates + 1))


def patient_equal_sensitivity(scores: np.ndarray, patients: np.ndarray, threshold: float) -> float:
    return float(np.mean([np.mean(scores[patients == patient] > threshold) for patient in np.unique(patients)]))


def patient_equal_threshold(scores: np.ndarray, patients: np.ndarray, target: float) -> float:
    candidates = np.nextafter(np.unique(scores), -np.inf)
    feasible = [float(value) for value in candidates if patient_equal_sensitivity(scores, patients, float(value)) >= target - 1e-15]
    if not feasible:
        raise RuntimeError("no feasible patient-equal threshold")
    return max(feasible)


def patient_equal_threshold_grid(scores: np.ndarray, patients: np.ndarray, targets: np.ndarray) -> dict[float, tuple[float, float]]:
    """Vectorized exact equivalent of repeated patient-equal threshold searches."""
    candidates = np.nextafter(np.unique(scores), -np.inf)
    sensitivities = np.zeros(len(candidates), dtype=np.float64)
    for patient in np.unique(patients):
        patient_scores = scores[patients == patient]
        sensitivities += np.mean(patient_scores[None, :] > candidates[:, None], axis=1)
    sensitivities /= len(np.unique(patients))
    result: dict[float, tuple[float, float]] = {}
    for target in targets:
        feasible = np.flatnonzero(sensitivities >= float(target) - 1e-15)
        if len(feasible) == 0:
            raise RuntimeError("no feasible patient-equal threshold in grid")
        index = int(feasible[-1])
        result[float(target)] = (float(candidates[index]), float(sensitivities[index]))
    return result


def mean_bootstrap_intervals(matrix: np.ndarray, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Patient bootstrap intervals for every column of an n-patient matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = np.empty((replicates, matrix.shape[1]), dtype=np.float32)
    for start in range(0, replicates, 2000):
        stop = min(replicates, start + 2000)
        indices = generator.integers(0, matrix.shape[0], size=(stop - start, matrix.shape[0]))
        samples[start:stop] = np.mean(matrix[indices], axis=1)
    return np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def small_rank(values: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    ranks = starts + (counts + 1.0) / 2.0
    if len(unique) == 1:
        return np.zeros(len(values), dtype=np.float64)
    return ranks[inverse]


def small_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = small_rank(np.asarray(x)), small_rank(np.asarray(y))
    denom = float(np.linalg.norm(rx - np.mean(rx)) * np.linalg.norm(ry - np.mean(ry)))
    return float(np.dot(rx - np.mean(rx), ry - np.mean(ry)) / denom) if denom > 0 else float("nan")


def spearman_bootstrap(x: np.ndarray, y: np.ndarray, replicates: int, seed: int) -> tuple[float, float, int]:
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        indices = generator.integers(0, len(x), size=len(x))
        value = small_spearman(x[indices], y[indices])
        if math.isfinite(value):
            estimates.append(value)
    if len(estimates) < int(0.95 * replicates):
        raise RuntimeError("too many non-finite Spearman bootstrap resamples")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)), len(estimates)


def verify_inputs(config: dict[str, object]) -> dict[str, dict[str, object]]:
    verified: dict[str, dict[str, object]] = {}
    for name, spec in config["inputs"].items():
        path = PROJECT / str(spec["path"])
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
        verified[name] = {"path": str(path.relative_to(PROJECT)), "bytes": path.stat().st_size, "sha256": actual}
    return verified


def method_statistics(config: dict[str, object], output: Path) -> dict[str, object]:
    spec = config["sparse_group"]
    endpoint = str(spec["primary_endpoint"])
    methods = [str(value) for value in spec["methods"]]
    patient_rows = read_csv(PROJECT / config["inputs"]["sparse_group_patient_metrics"]["path"])
    session_rows = read_csv(PROJECT / config["inputs"]["sparse_group_session_scores"]["path"])
    selected_patients = [row for row in patient_rows if row["endpoint"] == endpoint]
    selected_sessions = [row for row in session_rows if row["endpoint"] == endpoint]
    patients = sorted({row["patient"] for row in selected_patients})
    if len(patients) != 36 or len(selected_patients) != 36 * len(methods):
        raise ValueError("unexpected sparse-group patient table cardinality")
    row_lookup = {(row["patient"], row["method"]): row for row in selected_patients}
    outcomes: dict[str, np.ndarray] = {}
    outcomes["cortical_false_fire_fraction"] = np.asarray([
        [float(row_lookup[(patient, method)]["cortical_false_fire_fraction"]) for method in methods]
        for patient in patients
    ])
    outcomes["calibrated_injection_sensitivity"] = np.asarray([
        [float(row_lookup[(patient, method)]["calibrated_injection_sensitivity"]) for method in methods]
        for patient in patients
    ])
    outcomes["sensitivity_minus_false_fire"] = outcomes["calibrated_injection_sensitivity"] - outcomes["cortical_false_fire_fraction"]
    auc_matrix = np.empty((len(patients), len(methods)), dtype=np.float64)
    for patient_index, patient in enumerate(patients):
        for method_index, method in enumerate(methods):
            rows = [row for row in selected_sessions if row["patient"] == patient and row["method"] == method]
            auc_matrix[patient_index, method_index] = auc(
                [float(row["injection_delta_median"]) for row in rows],
                [float(row["response_delta_median"]) for row in rows],
            )
    outcomes["cortical_control_discrimination_auc"] = auc_matrix

    outcome_rows: list[dict[str, object]] = []
    for patient_index, patient in enumerate(patients):
        for method_index, method in enumerate(methods):
            outcome_rows.append({
                "patient": patient,
                "endpoint": endpoint,
                "method": method,
                **{name: float(matrix[patient_index, method_index]) for name, matrix in outcomes.items()},
            })
    write_csv(output / "injection_benchmark_patient_method_outcomes_v1.csv", outcome_rows)

    omnibus_rows: list[dict[str, object]] = []
    for outcome_index, (outcome, matrix) in enumerate(outcomes.items()):
        statistic, p_value = omnibus_patient_block_p(
            matrix, int(spec["omnibus_permutation_replicates"]),
            stable_seed(int(config["master_seed"]), "omnibus", outcome_index, outcome),
        )
        omnibus_rows.append({
            "endpoint": endpoint, "outcome": outcome, "patients": len(patients),
            "methods": len(methods), "statistic_between_method_mean_dispersion": statistic,
            "patient_block_permutation_replicates": int(spec["omnibus_permutation_replicates"]),
            "permutation_p_two_sided": p_value,
        })
    omnibus_adjusted = holm_adjust([float(row["permutation_p_two_sided"]) for row in omnibus_rows])
    for row, adjusted in zip(omnibus_rows, omnibus_adjusted):
        row["holm_adjusted_p_across_4_outcomes"] = adjusted
        row["reject_familywise_0p05"] = int(adjusted <= 0.05)
    write_csv(output / "injection_benchmark_method_omnibus_v1.csv", omnibus_rows)

    directions = {str(row["name"]): str(row["direction"]) for row in spec["declared_outcomes"]}
    focal = str(spec["focal_method"])
    focal_index = methods.index(focal)
    contrast_rows: list[dict[str, object]] = []
    for comparator_spec in spec["declared_comparators"]:
        comparator = str(comparator_spec["method"])
        comparator_index = methods.index(comparator)
        for outcome_index, outcome in enumerate(directions):
            raw = outcomes[outcome][:, focal_index] - outcomes[outcome][:, comparator_index]
            advantage = -raw if directions[outcome] == "lower" else raw
            low, high = paired_bootstrap(
                advantage, int(spec["paired_bootstrap_replicates"]),
                stable_seed(int(config["master_seed"]), "paired_bootstrap", comparator, outcome_index, outcome),
            )
            p_value = paired_swap_p(
                advantage, int(spec["paired_permutation_replicates"]),
                stable_seed(int(config["master_seed"]), "paired_permutation", comparator, outcome_index, outcome),
            )
            contrast_rows.append({
                "endpoint": endpoint, "focal_method": focal, "comparator_method": comparator,
                "comparator_role": comparator_spec["role"], "comparator_selection": comparator_spec["selection"],
                "outcome": outcome, "outcome_preferred_direction": directions[outcome],
                "patients": len(patients), "effect_definition": "positive_favors_fastica_bss",
                "mean_paired_advantage": float(np.mean(advantage)),
                "median_paired_advantage": float(np.median(advantage)),
                "paired_bootstrap_mean_ci95_low": low, "paired_bootstrap_mean_ci95_high": high,
                "paired_bootstrap_replicates": int(spec["paired_bootstrap_replicates"]),
                "paired_swap_permutation_p_two_sided": p_value,
                "paired_swap_permutation_replicates": int(spec["paired_permutation_replicates"]),
            })
    adjusted = holm_adjust([float(row["paired_swap_permutation_p_two_sided"]) for row in contrast_rows])
    for row, value in zip(contrast_rows, adjusted):
        row["holm_adjusted_p_across_8_declared_contrasts"] = value
        row["reject_familywise_0p05"] = int(value <= 0.05)
    write_csv(output / "injection_benchmark_declared_paired_contrasts_v1.csv", contrast_rows)
    return {
        "patients": len(patients), "methods": methods, "endpoint": endpoint,
        "omnibus_rejections": int(sum(int(row["reject_familywise_0p05"]) for row in omnibus_rows)),
        "declared_contrast_rejections": int(sum(int(row["reject_familywise_0p05"]) for row in contrast_rows)),
        "contrast_rows": contrast_rows,
    }


def tradeoff_curves(config: dict[str, object], output: Path) -> dict[str, object]:
    spec = config["sparse_group"]
    methods = [str(value) for value in spec["methods"]]
    rows = read_csv(PROJECT / config["inputs"]["sparse_group_session_scores"]["path"])
    endpoints = sorted({row["endpoint"] for row in rows})
    patients = sorted({row["patient"] for row in rows})
    grid = spec["tradeoff_training_sensitivity_targets"]
    interior = np.arange(float(grid["start"]), float(grid["stop"]) + float(grid["step"]) / 2.0, float(grid["step"]))
    targets = np.r_[0.0, np.round(interior, 10), 1.0] if grid["include_sentinels"] else np.round(interior, 10)
    patient_curve_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for endpoint_index, endpoint in enumerate(endpoints):
        for method_index, method in enumerate(methods):
            selected = [row for row in rows if row["endpoint"] == endpoint and row["method"] == method]
            sensitivity_matrix = np.empty((len(patients), len(targets)), dtype=np.float64)
            false_fire_matrix = np.empty_like(sensitivity_matrix)
            for patient_index, heldout in enumerate(patients):
                training = [row for row in selected if row["patient"] != heldout]
                testing = [row for row in selected if row["patient"] == heldout]
                training_scores = np.asarray([float(row["injection_delta_median"]) for row in training])
                training_patients = np.asarray([row["patient"] for row in training], dtype=str)
                threshold_grid = patient_equal_threshold_grid(
                    training_scores, training_patients, targets[(targets > 0.0) & (targets < 1.0)]
                )
                for target_index, target in enumerate(targets):
                    if target == 0.0:
                        threshold = float("inf")
                        train_sensitivity = 0.0
                    elif target == 1.0:
                        threshold = float("-inf")
                        train_sensitivity = 1.0
                    else:
                        threshold, train_sensitivity = threshold_grid[float(target)]
                    sensitivity = float(np.mean([float(row["injection_delta_median"]) > threshold for row in testing]))
                    false_fire = float(np.mean([float(row["response_delta_median"]) > threshold for row in testing]))
                    sensitivity_matrix[patient_index, target_index] = sensitivity
                    false_fire_matrix[patient_index, target_index] = false_fire
                    patient_curve_rows.append({
                        "patient": heldout, "endpoint": endpoint, "method": method,
                        "training_sensitivity_target": float(target), "threshold": threshold,
                        "attained_training_patient_equal_sensitivity": train_sensitivity,
                        "heldout_injection_sensitivity": sensitivity,
                        "heldout_cortical_false_fire_fraction": false_fire,
                        "heldout_cortical_specificity": 1.0 - false_fire,
                    })
            sensitivity_low, sensitivity_high = mean_bootstrap_intervals(
                sensitivity_matrix, int(spec["curve_bootstrap_replicates"]),
                stable_seed(int(config["master_seed"]), "curve_sensitivity", endpoint_index, method_index),
            )
            false_low, false_high = mean_bootstrap_intervals(
                false_fire_matrix, int(spec["curve_bootstrap_replicates"]),
                stable_seed(int(config["master_seed"]), "curve_false", endpoint_index, method_index),
            )
            for target_index, target in enumerate(targets):
                sensitivity = float(np.mean(sensitivity_matrix[:, target_index]))
                false_fire = float(np.mean(false_fire_matrix[:, target_index]))
                aggregate_rows.append({
                    "endpoint": endpoint, "method": method, "training_sensitivity_target": float(target),
                    "heldout_patient_equal_sensitivity": sensitivity,
                    "heldout_sensitivity_bootstrap_ci95_low": float(sensitivity_low[target_index]),
                    "heldout_sensitivity_bootstrap_ci95_high": float(sensitivity_high[target_index]),
                    "heldout_patient_equal_cortical_false_fire_fraction": false_fire,
                    "heldout_false_fire_bootstrap_ci95_low": float(false_low[target_index]),
                    "heldout_false_fire_bootstrap_ci95_high": float(false_high[target_index]),
                    "heldout_patient_equal_cortical_specificity": 1.0 - false_fire,
                    "heldout_specificity_bootstrap_ci95_low": 1.0 - float(false_high[target_index]),
                    "heldout_specificity_bootstrap_ci95_high": 1.0 - float(false_low[target_index]),
                    "sensitivity_minus_false_fire": sensitivity - false_fire,
                    "patients": len(patients),
                })
            patient_aucs = []
            for patient in patients:
                patient_selected = [row for row in selected if row["patient"] == patient]
                patient_aucs.append(auc(
                    [float(row["injection_delta_median"]) for row in patient_selected],
                    [float(row["response_delta_median"]) for row in patient_selected],
                ))
            low, high = paired_bootstrap(
                np.asarray(patient_aucs), int(spec["paired_bootstrap_replicates"]),
                stable_seed(int(config["master_seed"]), "auc_summary", endpoint_index, method_index),
            )
            summary_rows.append({
                "endpoint": endpoint, "method": method, "patients": len(patients),
                "mean_patient_cortical_control_discrimination_auc": float(np.mean(patient_aucs)),
                "median_patient_cortical_control_discrimination_auc": float(np.median(patient_aucs)),
                "patient_bootstrap_mean_auc_ci95_low": low, "patient_bootstrap_mean_auc_ci95_high": high,
            })
    write_csv(output / "injection_benchmark_tradeoff_patient_curves_v1.csv", patient_curve_rows)

    write_csv(output / "injection_benchmark_tradeoff_curves_v1.csv", aggregate_rows)
    write_csv(output / "injection_benchmark_curve_summary_v1.csv", summary_rows)

    return {
        "patients": len(patients), "endpoints": endpoints, "methods": methods,
        "training_targets": len(targets), "patient_curve_rows": len(patient_curve_rows),
    }


def geometry_intervals(config: dict[str, object], output: Path) -> dict[str, object]:
    spec = config["geometry_associations"]
    report3 = json.loads((PROJECT / config["inputs"]["working_memory_report"]["path"]).read_text(encoding="utf-8"))
    events3 = read_csv(PROJECT / config["inputs"]["working_memory_event_scores"]["path"])
    lookup3 = json.loads((PROJECT / config["inputs"]["working_memory_gain_lookup"]["path"]).read_text(encoding="utf-8"))
    patient_auc = {str(row["patient"]): float(row["lda_auc"]) for row in report3["patients"]}
    patients3 = sorted(patient_auc)
    gain_maps = {
        f"{float(extent):.2f}": {str(row["contact_key"]): float(row["gain_by_extent"][f"{float(extent):.2f}"]["log_median"]) for row in lookup3["contacts"]}
        for extent in spec["working_memory_extent_fractions"]
    }
    interval_rows: list[dict[str, object]] = []
    patient_predictor_rows: list[dict[str, object]] = []
    for extent_index, extent in enumerate(spec["working_memory_extent_fractions"]):
        extent_key = f"{float(extent):.2f}"
        x = []
        y = []
        for patient in patients3:
            contacts = [row["dominant_contact"] for row in events3 if row["patient"] == patient and int(row["label"]) == 1]
            predictor = float(np.median([gain_maps[extent_key][contact] for contact in contacts]))
            x.append(predictor); y.append(patient_auc[patient])
            patient_predictor_rows.append({
                "dataset": "OpenNeuro_ds004752", "patient": patient, "predictor": f"template_log_gain_extent_{extent_key}",
                "predictor_value": predictor, "observed_detectability": patient_auc[patient], "observed_metric": "heldout_lda_auc",
            })
        rho = float(spearmanr(x, y).statistic)
        expected = float(report3["forward_extent_sensitivity"][extent_key]["patient_level_association"]["spearman_rho"])
        if not np.isclose(rho, expected, atol=1e-12, rtol=0.0):
            raise ValueError(f"working-memory association reconstruction failed at {extent_key}")
        low, high, finite = spearman_bootstrap(
            np.asarray(x), np.asarray(y), int(spec["bootstrap_replicates"]),
            stable_seed(int(config["master_seed"]), "working_memory_spearman", extent_index, extent_key),
        )
        interval_rows.append({
            "dataset": "OpenNeuro_ds004752", "predictor": f"template_log_gain_extent_{extent_key}",
            "frozen_primary": int(float(extent) == float(spec["working_memory_primary_extent"])),
            "patients": len(x), "spearman_rho": rho, "patient_bootstrap_ci95_low": low,
            "patient_bootstrap_ci95_high": high, "requested_bootstrap_replicates": int(spec["bootstrap_replicates"]),
            "finite_bootstrap_replicates": finite, "interval_method": spec["interval"],
        })

    rows10 = read_csv(PROJECT / config["inputs"]["mesial_events_patient_metrics"]["path"])
    report10 = json.loads((PROJECT / config["inputs"]["mesial_events_report"]["path"]).read_text(encoding="utf-8"))
    observed10 = np.asarray([float(row["observed_auc"]) for row in rows10])
    report_keys = {"predicted_coherent_bits": "coherent_spearman_rho", "predicted_wave1_bits": "wave1_spearman_rho"}
    for predictor_index, predictor in enumerate(spec["mesial_events_predictors"]):
        predicted = np.asarray([float(row[predictor]) for row in rows10])
        rho = float(spearmanr(predicted, observed10).statistic)
        expected = float(report10["association"][report_keys[predictor]])
        if not np.isclose(rho, expected, atol=1e-12, rtol=0.0):
            raise ValueError(f"mesial-event association reconstruction failed: {predictor}")
        low, high, finite = spearman_bootstrap(
            predicted, observed10, int(spec["bootstrap_replicates"]),
            stable_seed(int(config["master_seed"]), "mesial_events_spearman", predictor_index, predictor),
        )
        interval_rows.append({
            "dataset": "Koessler_Ternisien", "predictor": predictor,
            "frozen_primary": int(predictor == "predicted_coherent_bits"), "patients": len(rows10),
            "spearman_rho": rho, "patient_bootstrap_ci95_low": low, "patient_bootstrap_ci95_high": high,
            "requested_bootstrap_replicates": int(spec["bootstrap_replicates"]),
            "finite_bootstrap_replicates": finite, "interval_method": spec["interval"],
        })
        for row in rows10:
            patient_predictor_rows.append({
                "dataset": "Koessler_Ternisien", "patient": row["patient"], "predictor": predictor,
                "predictor_value": float(row[predictor]), "observed_detectability": float(row["observed_auc"]),
                "observed_metric": "contiguous_holdout_matched_filter_auc",
            })
    write_csv(output / "empirical_geometry_patient_values_v1.csv", patient_predictor_rows)
    write_csv(output / "empirical_geometry_association_intervals_v1.csv", interval_rows)
    return {"association_count": len(interval_rows), "working_memory_patients": len(patients3), "mesial_events_patients": len(rows10)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL or config.get("status") != "frozen_before_execution":
        raise ValueError("unexpected or unfrozen association-statistics protocol")
    verified = verify_inputs(config)
    args.output.mkdir(parents=True, exist_ok=False)
    method_report = method_statistics(config, args.output)
    curve_report = tradeoff_curves(config, args.output)
    geometry_report = geometry_intervals(config, args.output)
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "ok": True,
        "config_sha256": sha256(args.config), "verified_inputs": verified,
        "method_statistics": method_report, "tradeoff_curves": curve_report,
        "geometry_associations": geometry_report, "scope_freeze": config["scope_freeze"],
        "interpretation_guards": {
            "paired_unit": "patient",
            "best_anatomical_method_selection": "comparator fixed from the completed method ranking; not represented as fully prospective",
            "sparse_group_positive": "equal-sensor-energy synthetic hippocampal injection",
            "sparse_group_negative": "real known-cortical stimulation response",
            "geometry_null_worded_as_absence": False,
        },
        "outputs": {},
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            report["outputs"][str(path.relative_to(args.output)).replace("\\", "/")] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "method_statistics": method_report, "tradeoff_curves": curve_report, "geometry_associations": geometry_report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
