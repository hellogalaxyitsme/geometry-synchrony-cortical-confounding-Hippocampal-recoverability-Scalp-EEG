#!/usr/bin/env python3
"""Compact distribution/coupling robustness experiment on frozen forward operators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.linalg import helmert
from scipy.stats import rankdata


PROTOCOL = "robustness/distribution-coupling-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(master: int, *parts: object) -> int:
    text = "|".join([str(master), *(str(value) for value in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("empty output")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def reduced_gram(matrix: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """Gram matrix in an explicit orthonormal zero-sum sensor basis."""
    gram = np.asarray(matrix @ matrix.T, dtype=np.float64)
    return contrast.T @ gram @ contrast


def distribution_sample(name: str, size: tuple[int, ...] | int, generator: np.random.Generator, spec: dict[str, object]) -> np.ndarray:
    if name == "gaussian":
        return generator.standard_normal(size)
    if name == "student_t":
        degrees = float(spec["degrees_of_freedom"])
        return generator.standard_t(degrees, size=size) * math.sqrt((degrees - 2.0) / degrees)
    if name == "bursty_mixture":
        probability = float(spec["burst_probability"])
        quiet = float(spec["quiet_standard_deviation"])
        burst = float(spec["burst_standard_deviation"])
        mask = generator.random(size) < probability
        scales = np.where(mask, burst, quiet)
        variance = probability * burst**2 + (1.0 - probability) * quiet**2
        return generator.standard_normal(size) * scales / math.sqrt(variance)
    raise ValueError(f"unknown distribution: {name}")


def knn_mutual_information_bits(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Kraskov-1 scalar/scalar MI estimator with max-norm neighborhoods."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) <= k + 2:
        raise ValueError("invalid MI inputs")
    points = np.column_stack((x, y))
    distances = cKDTree(points).query(points, k=k + 1, p=np.inf, workers=-1)[0][:, k]
    radii = np.nextafter(distances, 0.0)
    sorted_x = np.sort(x); sorted_y = np.sort(y)
    nx = np.searchsorted(sorted_x, x + radii, side="right") - np.searchsorted(sorted_x, x - radii, side="left") - 1
    ny = np.searchsorted(sorted_y, y + radii, side="right") - np.searchsorted(sorted_y, y - radii, side="left") - 1
    estimate_nats = digamma(k) + digamma(len(x)) - float(np.mean(digamma(nx + 1) + digamma(ny + 1)))
    return float(max(0.0, estimate_nats / math.log(2.0)))


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.r_[positive, negative]
    ranks = rankdata(values, method="average")
    n_pos, n_neg = len(positive), len(negative)
    return float((np.sum(ranks[:n_pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_mean_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    samples = np.mean(values[indices], axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def forward_span(config: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    relative_tolerance = float(config["cortical_rank_relative_tolerance"])
    modes = int(config["hippocampal_sensor_modes"])
    for subject in config["hcp_subjects"]:
        cortical_path = Path(config["cortical_forward_root"]) / str(subject) / "cortical-fixed-leadfield.npy"
        hippocampal_path = Path(config["hippocampal_forward_root"]) / str(subject) / "hippunfold-hippocampal-fixed-leadfield.npy"
        cortical = np.load(cortical_path, mmap_mode="r", allow_pickle=False)
        hippocampal = np.load(hippocampal_path, mmap_mode="r", allow_pickle=False)
        if cortical.shape[0] != 339 or hippocampal.shape[0] != 339:
            raise ValueError(f"unexpected sensor dimension: {subject}")
        contrast = helmert(cortical.shape[0], full=False).T
        if contrast.shape != (339, 338) or not np.allclose(contrast.T @ contrast, np.eye(338), atol=1e-12, rtol=0.0):
            raise RuntimeError("invalid zero-sum sensor contrast basis")
        cortical_gram = reduced_gram(cortical, contrast)
        hippocampal_gram = reduced_gram(hippocampal, contrast)
        c_values, c_vectors = np.linalg.eigh(cortical_gram)
        singular = np.sqrt(np.maximum(c_values, 0.0))
        keep = singular > float(np.max(singular)) * relative_tolerance
        basis = contrast @ c_vectors[:, keep]
        cortical_rank = int(np.count_nonzero(keep))
        h_values, h_vectors = np.linalg.eigh(hippocampal_gram)
        order = np.argsort(h_values)[::-1]
        c_pinv = (basis / np.maximum(c_values[keep], np.finfo(float).tiny)) @ basis.T
        for mode_index in range(modes):
            vector = contrast @ h_vectors[:, order[mode_index]]
            vector /= np.linalg.norm(vector)
            projection = basis @ (basis.T @ vector)
            residual = float(np.linalg.norm(vector - projection))
            rows.append({
                "subject": subject, "sensors": cortical.shape[0], "referenced_sensor_dimension": cortical.shape[0] - 1,
                "cortical_sources": cortical.shape[1], "hippocampal_sources": hippocampal.shape[1],
                "cortical_referenced_rank": cortical_rank, "hippocampal_sensor_mode": mode_index + 1,
                "relative_cortical_projection_residual": residual,
                "minimum_l2_cortical_current_norm_for_unit_sensor_mode": float(math.sqrt(max(0.0, vector @ c_pinv @ vector))),
                "algebraically_in_cortical_span": int(residual <= float(config["projection_relative_tolerance"])),
            })
        del cortical, hippocampal, cortical_gram, hippocampal_gram, contrast
    return rows


def distribution_robustness(config: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reference = config["scalar_reference_model"]
    master = int(config["master_seed"])
    noise_variance = float(reference["sensor_noise_variance"])
    mi_samples = int(reference["mi_samples"])
    k = int(reference["knn_mi_k"])
    replicate_count = int(reference["detection_simulation_replicates"])
    train_windows = int(reference["detection_training_windows_per_class_per_replicate"])
    test_windows = int(reference["detection_testing_windows_per_class_per_replicate"])
    samples_per_window = int(reference["detection_window_samples"])
    fpr_target = float(reference["reference_false_positive_target"])
    bootstrap_replicates = int(reference["replicate_mean_bootstrap_replicates"])
    rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    independent_formula = 0.5 * math.log2(1.0 + 1.0 / (1.0 + noise_variance))
    for distribution_index, (distribution, distribution_spec) in enumerate(config["distributions"].items()):
        for correlation_index, rho in enumerate(config["hippocampal_cortical_correlations"]):
            rho = float(rho)
            generator = np.random.default_rng(stable_seed(master, "mi", distribution_index, correlation_index))
            x = distribution_sample(distribution, mi_samples, generator, distribution_spec)
            v = distribution_sample(distribution, mi_samples, generator, distribution_spec)
            noise = generator.standard_normal(mi_samples) * math.sqrt(noise_variance)
            cortical = rho * x + math.sqrt(1.0 - rho**2) * v
            observation = x + cortical + noise
            empirical_mi = knn_mutual_information_bits(x, observation, k)
            realized_rho = float(np.corrcoef(x, cortical)[0, 1])
            covariance_aware_gaussian = 0.5 * math.log2(1.0 + (1.0 + rho) ** 2 / (1.0 - rho**2 + noise_variance))
            metrics = {"fixed_auc": [], "oracle_auc": [], "fixed_tpr": [], "fixed_fpr": [], "realized_rho": []}
            for replicate in range(replicate_count):
                rg = np.random.default_rng(stable_seed(master, "detection", distribution_index, correlation_index, replicate))
                def window_scores(count: int, positive: bool) -> tuple[np.ndarray, float]:
                    xv = distribution_sample(distribution, (count, samples_per_window), rg, distribution_spec)
                    vv = distribution_sample(distribution, (count, samples_per_window), rg, distribution_spec)
                    ev = rg.standard_normal((count, samples_per_window)) * math.sqrt(noise_variance)
                    if positive:
                        cv = rho * xv + math.sqrt(1.0 - rho**2) * vv
                        signal = xv + cv + ev
                        corr = float(np.corrcoef(xv.ravel(), cv.ravel())[0, 1])
                    else:
                        signal = vv + ev
                        corr = float("nan")
                    return np.mean(signal**2, axis=1), corr
                train_negative, _ = window_scores(train_windows, False)
                test_negative, _ = window_scores(test_windows, False)
                test_positive, realized = window_scores(test_windows, True)
                threshold = float(np.quantile(train_negative, 1.0 - fpr_target))
                fixed_auc = auc(test_positive, test_negative)
                variance_direction = 1.0 if float(np.mean(test_positive)) >= float(np.mean(test_negative)) else -1.0
                oracle_auc = auc(variance_direction * test_positive, variance_direction * test_negative)
                fixed_tpr = float(np.mean(test_positive > threshold)); fixed_fpr = float(np.mean(test_negative > threshold))
                metrics["fixed_auc"].append(fixed_auc); metrics["oracle_auc"].append(oracle_auc)
                metrics["fixed_tpr"].append(fixed_tpr); metrics["fixed_fpr"].append(fixed_fpr); metrics["realized_rho"].append(realized)
                replicate_rows.append({
                    "distribution": distribution, "target_hippocampal_cortical_correlation": rho, "replicate": replicate + 1,
                    "realized_hippocampal_cortical_correlation": realized, "fixed_gaussian_independence_energy_auc": fixed_auc,
                    "variance_direction_oracle_auc": oracle_auc, "fixed_threshold_tpr": fixed_tpr, "fixed_threshold_fpr": fixed_fpr,
                })
            summary: dict[str, object] = {
                "distribution": distribution, "target_hippocampal_cortical_correlation": rho,
                "realized_scalar_hippocampal_cortical_correlation": realized_rho,
                "gaussian_independent_reference_mi_bits": independent_formula,
                "covariance_aware_gaussian_formula_mi_bits": covariance_aware_gaussian,
                "knn_empirical_mi_bits": empirical_mi, "knn_k": k, "mi_samples": mi_samples,
                "detection_replicates": replicate_count, "detection_window_samples": samples_per_window,
            }
            for metric_index, metric in enumerate(("fixed_auc", "oracle_auc", "fixed_tpr", "fixed_fpr")):
                values = np.asarray(metrics[metric], dtype=np.float64)
                low, high = bootstrap_mean_interval(values, bootstrap_replicates, stable_seed(master, "summary", distribution, rho, metric_index))
                summary[f"mean_{metric}"] = float(np.mean(values)); summary[f"mean_{metric}_ci95_low"] = low; summary[f"mean_{metric}_ci95_high"] = high
            summary["mean_detection_realized_correlation"] = float(np.mean(metrics["realized_rho"]))
            rows.append(summary)
    return rows, replicate_rows




def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL or config.get("status") != "frozen_before_execution":
        raise ValueError("unexpected or unfrozen robustness protocol")
    args.output.mkdir(parents=True, exist_ok=False)
    span_rows = forward_span(config)
    if not all(int(row["cortical_referenced_rank"]) == 338 for row in span_rows):
        raise RuntimeError("explicit referenced cortical-rank gate failed")
    if not all(int(row["algebraically_in_cortical_span"]) == 1 for row in span_rows):
        raise RuntimeError("frozen cortical-span gate failed")
    robustness_rows, replicate_rows = distribution_robustness(config)
    write_csv(args.output / "forward_span_results.csv", span_rows)
    write_csv(args.output / "mi_detection_results.csv", robustness_rows)
    write_csv(args.output / "detection_replicates.csv", replicate_rows)
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "ok": True, "config_sha256": sha256(args.config),
        "subjects": config["hcp_subjects"], "forward_span_rows": len(span_rows),
        "maximum_relative_cortical_projection_residual": max(float(row["relative_cortical_projection_residual"]) for row in span_rows),
        "all_cortical_referenced_ranks": sorted({int(row["cortical_referenced_rank"]) for row in span_rows}),
        "distribution_coupling_conditions": len(robustness_rows), "detection_replicate_rows": len(replicate_rows),
        "scope_freeze": config["scope_freeze"],
        "interpretation": {
            "distribution_free_result": "Every tested hippocampal sensor mode lies in the unrestricted cortical sensor span; relabeling the same coefficient sequence therefore reproduces its field independently of coefficient distribution.",
            "model_dependent_result": "Absolute mutual-information and detection values vary with source distribution and hippocampal-cortical coupling; the Gaussian-independent reference numbers are not universal physiological quantities.",
            "physiological_distribution_claim_authorized": False,
        },
        "outputs": {},
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file(): report["outputs"][str(path.relative_to(args.output)).replace("\\", "/")] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
