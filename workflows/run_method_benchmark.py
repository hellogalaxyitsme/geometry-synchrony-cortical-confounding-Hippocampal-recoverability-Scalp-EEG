#!/usr/bin/env python3
"""Run the patient-held-out method-benchmark established-method benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.working_memory import atomic_json, bootstrap_macro_auc, roc_auc  # noqa: E402
from empirical.estimators import (  # noqa: E402
    anatomical_log_power_ratio,
    covariance_factors,
    eloreta_type_operator,
    event_covariances,
    fit_fastica,
    ica_event_features,
    lcmv_operator,
    log_channel_power,
    minimum_norm_operator,
    patient_block_folds,
    select_signed_feature,
    sparse_anatomical_score,
    sparse_group_fista,
)


PROTOCOL = "benchmark/method-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: np.ndarray) -> object:
    return value.item() if value.ndim == 0 else value


def load_epochs(root: Path) -> dict[str, np.ndarray]:
    blocks: dict[str, list[np.ndarray]] = {
        "epochs": [], "labels": [], "patients": [], "sessions": [], "blocks": [], "trials": []
    }
    channels: np.ndarray | None = None
    frequency: float | None = None
    for path in sorted((root / "sessions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            row = {name: scalar(np.asarray(archive[name])) for name in archive.files}
        if str(row["protocol"]) != PROTOCOL or bool(row["depth_waveform_included"]):
            raise ValueError(f"invalid method-benchmark epoch bundle: {path}")
        current_channels = np.asarray(row["scalp_channels"], dtype=str)
        current_frequency = float(row["target_sampling_frequency_hz"])
        if channels is None:
            channels, frequency = current_channels, current_frequency
        elif not np.array_equal(channels, current_channels) or frequency != current_frequency:
            raise ValueError("method-benchmark epoch bundles use inconsistent sampling/montage")
        epochs = np.asarray(row["theta_epochs"], dtype=np.float64)
        labels = np.asarray(row["labels"], dtype=np.int64)
        if epochs.shape[0] != len(labels) or set(np.unique(labels)) - {0, 1}:
            raise ValueError(f"invalid labeled epochs: {path}")
        blocks["epochs"].append(epochs)
        blocks["labels"].append(labels)
        blocks["patients"].append(np.repeat(str(row["subject"]), len(labels)))
        blocks["sessions"].append(np.repeat(str(row["stem"]), len(labels)))
        blocks["blocks"].append(np.asarray(row["block_ids"], dtype=np.int64))
        blocks["trials"].append(np.asarray(row["trial_number"], dtype=np.int64))
    if channels is None or frequency is None:
        raise FileNotFoundError("no method-benchmark epoch bundles")
    result = {name: np.concatenate(values, axis=0) for name, values in blocks.items()}
    result["channels"] = channels
    result["sampling_frequency_hz"] = np.asarray(frequency)
    return result


def load_templates(root: Path, subjects: list[str]) -> list[dict[str, Any]]:
    templates = []
    source_names: np.ndarray | None = None
    for subject in subjects:
        path = root / subject / "source_dictionary.npz"
        report_path = root / subject / "report.json"
        if not path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"missing method-benchmark source dictionary: {subject}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("ok") is not True or report.get("subject") != subject:
            raise ValueError(f"failed method-benchmark source dictionary: {subject}")
        with np.load(path, allow_pickle=False) as archive:
            row = {name: np.asarray(archive[name]) for name in archive.files}
        if str(row["protocol"].item()) != PROTOCOL or str(row["subject"].item()) != subject:
            raise ValueError(f"source dictionary provenance mismatch: {subject}")
        if source_names is None:
            source_names = np.asarray(row["source_names"], dtype=str)
        elif not np.array_equal(source_names, np.asarray(row["source_names"], dtype=str)):
            raise ValueError("source dictionary columns differ across HCP templates")
        templates.append(
            {
                "subject": subject,
                "leadfield": np.asarray(row["leadfield"], dtype=np.float64),
                "reference": np.asarray(row["helmert_reference"], dtype=np.float64),
                "family": np.asarray(row["source_family"], dtype=np.int8),
                "group": np.asarray(row["source_group"], dtype=np.int64),
                "column_norms": np.asarray(row["column_norms"], dtype=np.float64),
                "sha256": sha256(path),
            }
        )
    return templates


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def linear_template_scores(
    method: str,
    parameter: float,
    train_covariance: np.ndarray,
    event_covariance: np.ndarray,
    templates: list[dict[str, Any]],
    depth_exponent: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    scores = []
    diagnostics = []
    for template in templates:
        gain = template["leadfield"]
        family = template["family"]
        cortex = np.flatnonzero(family == 0)
        hippocampus = np.flatnonzero(family == 1)
        if method == "minimum_norm":
            norms = template["column_norms"]
            variances = (norms / np.max(norms)) ** (-2.0 * depth_exponent)
            operator = minimum_norm_operator(gain, train_covariance, parameter, variances)
            detail: dict[str, object] = {}
        elif method == "eloreta_type":
            operator, detail = eloreta_type_operator(gain, train_covariance, parameter)
        elif method == "lcmv":
            operator = lcmv_operator(gain, train_covariance, parameter)
            unit_gain_error = float(np.max(np.abs(np.diag(operator @ gain) - 1.0)))
            noise_power = np.einsum(
                "is,st,it->i", operator, train_covariance, operator, optimize=True
            )
            if np.any(noise_power <= 0.0):
                raise ValueError("LCMV training-noise power is non-positive")
            operator = operator / np.sqrt(noise_power)[:, None]
            detail = {
                "maximum_unit_gain_error_before_NAI_normalization": unit_gain_error,
                "score_normalization": "neural_activity_index_training_covariance",
            }
        else:
            raise ValueError(f"unknown linear inverse method: {method}")
        scores.append(anatomical_log_power_ratio(operator, event_covariance, hippocampus, cortex))
        diagnostics.append({"template": template["subject"], "method": method, "parameter": parameter, **detail})
    return np.median(np.vstack(scores), axis=0), diagnostics


def sparse_template_scores(
    method: str,
    parameter: float,
    event_covariance: np.ndarray,
    templates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    template_scores = []
    diagnostics = []
    sparse = config["methods"][method]
    for template in templates:
        gain = template["leadfield"]
        norms = np.linalg.norm(gain, axis=0)
        design = gain / norms[None, :]
        family = template["family"]
        hippocampus = np.flatnonzero(family == 1)
        cortex = np.flatnonzero(family == 0)
        unique_groups = [np.flatnonzero(template["group"] == value) for value in np.unique(template["group"])]
        values = np.empty(len(event_covariance), dtype=np.float64)
        convergence = []
        iterations = []
        stationarity = []
        for event, covariance in enumerate(event_covariance):
            target = covariance_factors(covariance, int(sparse["covariance_components"]))
            scale = float(np.max(np.abs(design.T @ target)))
            l1 = parameter * max(scale, np.finfo(float).tiny)
            if method == "sparse_l1":
                coefficients, report = sparse_group_fista(
                    design,
                    target,
                    l1,
                    maximum_iterations=int(sparse["maximum_iterations"]),
                    tolerance=float(sparse["tolerance"]),
                )
            elif method == "hierarchical_sparse_group":
                coefficients, report = sparse_group_fista(
                    design,
                    target,
                    float(sparse["l1_fraction_of_group_penalty"]) * l1,
                    groups=unique_groups,
                    group_penalty=l1,
                    maximum_iterations=int(sparse["maximum_iterations"]),
                    tolerance=float(sparse["tolerance"]),
                )
            else:
                raise ValueError(f"unknown sparse method: {method}")
            values[event] = sparse_anatomical_score(coefficients, hippocampus, cortex)
            convergence.append(bool(report["converged"]))
            iterations.append(int(report["iterations"]))
            stationarity.append(float(report["proximal_gradient_stationarity"]))
        template_scores.append(values)
        diagnostics.append(
            {
                "template": template["subject"],
                "method": method,
                "parameter": parameter,
                "converged_fraction": float(np.mean(convergence)),
                "median_iterations": float(np.median(iterations)),
                "maximum_proximal_gradient_stationarity": float(np.max(stationarity)),
                "median_proximal_gradient_stationarity": float(np.median(stationarity)),
            }
        )
    return np.median(np.vstack(template_scores), axis=0), diagnostics


def select_nested_parameter(
    candidates: list[float],
    outer_patients: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    blocks: np.ndarray,
    score_function: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
) -> tuple[float, list[dict[str, object]]]:
    rows = []
    for parameter in candidates:
        aucs = []
        for inner_heldout in outer_patients:
            inner_train = (patients != inner_heldout) & np.isin(patients, outer_patients) & np.isin(blocks, (0, 1))
            inner_test = (patients == inner_heldout) & (blocks == 2)
            if set(np.unique(labels[inner_test])) != {0, 1}:
                raise ValueError(f"inner patient lacks both classes: {inner_heldout}")
            score = score_function(parameter, np.flatnonzero(inner_train), np.flatnonzero(inner_test))
            aucs.append(roc_auc(labels[inner_test], score))
        rows.append({"parameter": parameter, "inner_macro_patient_auc": float(np.mean(aucs)), "inner_patient_aucs": aucs})
    best = max(rows, key=lambda row: (row["inner_macro_patient_auc"], -row["parameter"]))
    return float(best["parameter"]), rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs-root", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("subset", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise SystemExit("unexpected method-benchmark protocol")
    data = load_epochs(args.epochs_root)
    full_subjects = [line.strip() for line in (PROJECT / config["hcp_subjects_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    template_subjects = list(config["subset_hcp_subjects"]) if args.mode == "subset" else full_subjects
    templates = load_templates(args.template_root, template_subjects)
    epochs = np.asarray(data["epochs"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    patients = np.asarray(data["patients"], dtype=str)
    blocks = np.asarray(data["blocks"], dtype=np.int64)
    channels = np.asarray(data["channels"], dtype=str)
    if not np.array_equal(channels, np.asarray(config["scalp_channels"], dtype=str)):
        raise ValueError("epoch montage differs from method-benchmark config")
    reference = templates[0]["reference"]
    reduced_epochs = np.einsum("rs,est->ert", reference, epochs, optimize=True)
    covariances = event_covariances(reduced_epochs)
    channel_features = log_channel_power(epochs)
    folds = patient_block_folds(patients, blocks, (0, 1), (2,))
    method_scores = {
        method: np.full(len(labels), np.nan, dtype=np.float64)
        for method in (
            "best_channel", "fastica_bss", "minimum_norm", "eloreta_type", "lcmv",
            "sparse_l1", "hierarchical_sparse_group",
        )
    }
    fold_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    sparse_score_cache: dict[tuple[str, float], np.ndarray] = {}
    for fold_index, fold in enumerate(folds):
        heldout = str(fold["heldout_patient"])
        train = np.asarray(fold["train_indices"], dtype=np.int64)
        test = np.asarray(fold["test_indices"], dtype=np.int64)
        outer_patients = np.unique(patients[train])
        if set(np.unique(labels[test])) != {0, 1}:
            raise ValueError(f"held-out patient lacks both late classes: {heldout}")

        channel, sign, training_auc = select_signed_feature(channel_features[train], labels[train])
        method_scores["best_channel"][test] = sign * channel_features[test, channel]
        fold_rows.append({
            "heldout_patient": heldout, "method": "best_channel", "parameter": channel,
            "training_auc": training_auc, "heldout_auc": roc_auc(labels[test], method_scores["best_channel"][test]),
        })

        cap = int(config["methods"]["fastica_bss"]["maximum_training_samples"])
        training_samples = reduced_epochs[train].transpose(0, 2, 1).reshape(-1, reduced_epochs.shape[1])
        if len(training_samples) > cap:
            generator = np.random.default_rng(int(config["master_seed"]) + fold_index)
            training_samples = training_samples[np.sort(generator.choice(len(training_samples), cap, replace=False))]
        ica = fit_fastica(
            training_samples,
            int(config["methods"]["fastica_bss"]["components"]),
            seed=int(config["master_seed"]) + fold_index,
            maximum_iterations=int(config["methods"]["fastica_bss"]["maximum_iterations"]),
            tolerance=float(config["methods"]["fastica_bss"]["tolerance"]),
        )
        train_ica = ica_event_features(ica, reduced_epochs[train])
        component, sign, training_auc = select_signed_feature(train_ica, labels[train])
        test_ica = ica_event_features(ica, reduced_epochs[test])
        method_scores["fastica_bss"][test] = sign * test_ica[:, component]
        fold_rows.append({
            "heldout_patient": heldout, "method": "fastica_bss", "parameter": component,
            "training_auc": training_auc, "heldout_auc": roc_auc(labels[test], method_scores["fastica_bss"][test]),
            "ica_converged": ica.converged, "ica_iterations": ica.iterations,
        })

        for method in ("minimum_norm", "eloreta_type", "lcmv"):
            settings = config["methods"][method]
            candidates = [float(value) for value in settings["regularization_grid"]]

            def linear_score(parameter: float, training: np.ndarray, evaluated: np.ndarray) -> np.ndarray:
                training_covariance = np.mean(covariances[training], axis=0)
                score, _ = linear_template_scores(
                    method,
                    parameter,
                    training_covariance,
                    covariances[evaluated],
                    templates,
                    float(config["methods"]["minimum_norm"]["depth_exponent"]),
                )
                return score

            chosen, inner = select_nested_parameter(
                candidates, outer_patients, labels, patients, blocks, linear_score
            )
            score, detail = linear_template_scores(
                method,
                chosen,
                np.mean(covariances[train], axis=0),
                covariances[test],
                templates,
                float(config["methods"]["minimum_norm"]["depth_exponent"]),
            )
            method_scores[method][test] = score
            diagnostics.extend({"heldout_patient": heldout, **row} for row in detail)
            fold_rows.append({
                "heldout_patient": heldout, "method": method, "parameter": chosen,
                "heldout_auc": roc_auc(labels[test], score), "inner_grid": json.dumps(inner, sort_keys=True),
            })

        for method in ("sparse_l1", "hierarchical_sparse_group"):
            candidates = [float(value) for value in config["methods"][method]["penalty_fraction_grid"]]

            def sparse_score(parameter: float, _training: np.ndarray, evaluated: np.ndarray) -> np.ndarray:
                key = (method, parameter)
                if key not in sparse_score_cache:
                    sparse_score_cache[key], detail = sparse_template_scores(
                        method, parameter, covariances, templates, config
                    )
                    diagnostics.extend({"heldout_patient": "all_parameter_cache", **row} for row in detail)
                return sparse_score_cache[key][evaluated]

            chosen, inner = select_nested_parameter(
                candidates, outer_patients, labels, patients, blocks, sparse_score
            )
            score = sparse_score(chosen, train, test)
            method_scores[method][test] = score
            fold_rows.append({
                "heldout_patient": heldout, "method": method, "parameter": chosen,
                "heldout_auc": roc_auc(labels[test], score), "inner_grid": json.dumps(inner, sort_keys=True),
            })
        print(json.dumps({"status": "complete", "heldout_patient": heldout}), flush=True)

    evaluated = np.zeros(len(labels), dtype=bool)
    for fold in folds:
        evaluated[np.asarray(fold["test_indices"], dtype=np.int64)] = True
    event_rows: list[dict[str, object]] = []
    for index in np.flatnonzero(evaluated):
        row: dict[str, object] = {
            "patient": str(patients[index]), "session": str(data["sessions"][index]),
            "trial_number": int(data["trials"][index]), "block": int(blocks[index]), "label": int(labels[index]),
        }
        row.update({f"{method}_score": float(values[index]) for method, values in method_scores.items()})
        event_rows.append(row)
    patient_rows = []
    for patient in np.unique(patients[evaluated]):
        selected = evaluated & (patients == patient)
        row: dict[str, object] = {"patient": str(patient), "events": int(np.count_nonzero(selected))}
        for method, values in method_scores.items():
            row[f"{method}_auc"] = roc_auc(labels[selected], values[selected])
        patient_rows.append(row)
    summary: dict[str, object] = {}
    for method in method_scores:
        summary[method] = bootstrap_macro_auc(
            [float(row[f"{method}_auc"]) for row in patient_rows],
            int(config["bootstrap_replicates"]),
            int(config["master_seed"]) + list(method_scores).index(method),
        )
    if any(not np.all(np.isfinite(values[evaluated])) for values in method_scores.values()):
        raise AssertionError("method-benchmark held-out scores are incomplete")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "event_scores.csv", event_rows)
    write_csv(args.output / "patient_metrics.csv", patient_rows)
    write_csv(args.output / "fold_metrics.csv", fold_rows)
    write_csv(args.output / "operator_diagnostics.csv", diagnostics)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": True,
        "mode": args.mode,
        "patient_count": len(patient_rows),
        "evaluated_event_count": int(np.count_nonzero(evaluated)),
        "template_count": len(templates),
        "template_subjects": template_subjects,
        "methods": list(method_scores),
        "macro_patient_auc": summary,
        "validation": "leave_one_patient_out; train early+middle; test late; nested patient-held-out parameter selection",
        "template_aggregation": "median score across frozen HCP anatomies before AUC",
        "depth_waveform_in_scalp_solution": False,
        "physical_input": "theta-band scalp event covariance; never log-power ratios as voltages",
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                args.output / "event_scores.csv", args.output / "patient_metrics.csv",
                args.output / "fold_metrics.csv", args.output / "operator_diagnostics.csv",
            )
        },
        "physiological_inference_authorized": False,
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "outputs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
