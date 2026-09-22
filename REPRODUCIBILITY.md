# Reproducibility map

This repository contains the exact numerical source code. The manuscript-associated archive
provides the frozen study configuration bundle, cohort manifest, provenance records and
permitted derived outputs. Full regeneration additionally requires the source datasets listed
in `DATA_ACCESS.md`.

Entry points use explicit command-line arguments, typically
`python workflows/<workflow>.py --config <config.json> --output <path>`. Run any entry point
with `--help` for its complete interface.

## Reported analysis to workflow

| Reported analysis | Entry points |
| --- | --- |
| Individual laminar forward models | `workflows/build_hippocampal_forward.py`, `workflows/summarize_hippocampal_forward.py` |
| Geometry and synchrony source ensembles | `workflows/build_source_ensembles.py`, `workflows/summarize_source_ensembles.py`, `workflows/summarize_source_ensemble_key_results.py` |
| Retained-power normalization audit | `workflows/audit_retained_power_normalization.py`, `workflows/check_retained_power_normalization.py` |
| Cortical-restriction ladder | `workflows/build_cortical_restriction_subject.py`, `workflows/build_empirical_covariance_basis.py`, `workflows/summarize_cortical_restriction.py` |
| Working-memory epoch construction | `workflows/build_working_memory_epochs.py` |
| Working-memory seven-method benchmark | `workflows/build_method_benchmark_epochs.py`, `workflows/build_method_benchmark_template.py`, `workflows/run_method_benchmark.py`, `workflows/audit_method_benchmark_results.py` |
| Working-memory temporal transfer | `workflows/build_temporal_transfer_scores.py`, `workflows/check_temporal_transfer.py` |
| Mesial-event within-network detection | `workflows/build_mesial_event_dataset.py`, `workflows/run_mesial_event_analysis.py`, `workflows/audit_mesial_event_results.py` |
| Template-based recoverability associations | `workflows/build_mesial_event_predictions.py`, `workflows/run_association_statistics.py` |
| Empirical injected-energy calibration | `workflows/run_empirical_calibration.py`, `workflows/audit_calibration_results.py` |
| Recorded cortical-stimulation conditional attribution | `workflows/build_stimulation_template_bundle.py`, `workflows/run_stimulation_template.py`, `workflows/run_sparse_group_template.py`, `workflows/summarize_stimulation_control.py` |
| Calibrated simulated-injection benchmark | `workflows/run_injection_benchmark_pairs.py`, `workflows/summarize_injection_benchmark.py`, `workflows/audit_injection_benchmark_results.py` |
| Montage information ladder | `workflows/build_montage_subject.py`, `workflows/summarize_montage_information.py`, `workflows/audit_montage_results.py` |
| Head-model uncertainty | `workflows/build_head_model_subject.py`, `workflows/summarize_head_model_uncertainty.py`, `workflows/audit_head_model_results.py` |
| Distribution and coupling robustness | `workflows/run_distribution_coupling.py` |
| Identifiability and information checks | `workflows/check_theorem_stress.py` |
| Synthetic limiting cases | `workflows/check_simulation_smoke.py` |

## Reproduction boundary

- The public GitHub repository contains code, a synthetic test fixture and the recorded direct
  software dependencies.
- The manuscript-associated archive contains frozen study configurations, cohort membership,
  provenance manifests and redistributable derived outputs.
- Restricted or credentialed source datasets are not redistributed by either mechanism;
  obtain them under their original access and reuse conditions.
- Structural software versions, mesh construction and dataset releases can affect later
  significant digits. Use the archived settings and recorded environment for exact reruns.
