# Recoverability of hippocampal activity from scalp EEG: analysis code

This repository contains the numerical analysis source code and code-level validation tests
for the manuscript

> The role of geometry, synchrony, and cortical confounding in the recoverability of
> hippocampal activity from scalp EEG

The release is intentionally code-only. It does not redistribute source datasets, cohort or
participant manifests, participant-level derived results, manuscript sources, or publication
figure-generation utilities. Frozen study configurations, permitted derived materials and
their provenance are supplied through the archive identified in the manuscript's Data
availability statement and remain subject to the source datasets' access conditions.

## Analysis overview

The code addresses three inference targets:

1. **Expression** - whether a hippocampal source ensemble produces a scalp field.
2. **Detection** - whether a scalp quantity distinguishes a target state from nuisance
   activity.
3. **Attribution** - whether a detected field can be assigned to hippocampus rather than a
   cortical configuration with the same sensor projection.

## Layout

```
anatomical/    forward-model construction, ingestion and numerical operators
empirical/     working-memory, temporal-transfer, mesial-event and control analyses
simulation/    source ensembles, restriction models and montage information
theory/        identifiability and information-theoretic calculations
workflows/     command-line analysis, aggregation, audit and numerical-check entry points
configs/       dependency-light synthetic validation fixture
environment/   recorded direct scientific dependencies
manifests/     SHA-256 manifest for every shipped non-manifest file
```

## Requirements

The recorded analysis environment used Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
h5py 3.16.0, nibabel 5.4.2 and MNE-Python 1.10.2; see `environment/`. A GPU is not required
for the numerical analyses, although regenerating structural derivatives can benefit from
accelerated computing.

## Quick validation

```bash
python validate_release.py
```

This verifies every manifest hash, scans the public tree for credentials and private paths,
parses the Python modules, imports dependency-light components, and runs the theorem and
synthetic limiting-case checks. It does not require restricted source datasets.

## Reproducing the reported analyses

Obtain the frozen configuration/cohort bundle and permitted derived materials from the
archive cited in the manuscript, arrange the source datasets as described in
`DATA_ACCESS.md`, and follow the workflow map in `REPRODUCIBILITY.md`. Each entry point lists
its exact command-line interface under `--help`.

## Citing this release

See `CITATION.cff`.
