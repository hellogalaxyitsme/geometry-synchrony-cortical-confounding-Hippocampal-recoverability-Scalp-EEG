# Source data and archived reproduction inputs

The workflows require four source resources. None is redistributed in this code repository.

1. **Human Connectome Project Young Adult structural data** (T1w/T2w and FreeSurfer
   derivatives), obtained under the HCP data-use terms.
2. **OpenNeuro ds004752**, simultaneous scalp EEG and intracranial EEG during a verbal
   working-memory task, DOI 10.18112/openneuro.ds004752.v1.0.1.
3. **Koessler/Ternisien simultaneous scalp EEG and SEEG mesial-event recordings**, with
   access described in the source publications.
4. **Cortical-stimulation EEG and SEEG resource**, available through its OSF/EBRAINS record,
   DOI 10.17605/OSF.IO/WSGZP, under the record's licence and access conditions.

The separate manuscript-associated archive supplies the frozen configuration files, cohort
manifest, provenance records and subject identifiers permitted for redistribution. Copy that
bundle into the repository root before a full rerun.

## Expected layout

| Prefix | Contents |
| --- | --- |
| `configs/` | archived frozen study configurations and cohort manifest |
| `source_data/hcp_ya/` | HCP Young Adult structural inputs |
| `source_data/ds004752/` | local OpenNeuro ds004752 release |
| `source_data/derivatives/` | structural derivatives generated from source imaging |
| `derivatives/` | intermediate analysis products |
| `results/` | regenerated compact numerical outputs |

Workflow entry points accept explicit `--config`, `--output`, and, where applicable,
`--forward-root`-style arguments. Alternate roots can therefore be supplied without editing
the source code. Missing controlled inputs cause a clear `FileNotFoundError` identifying the
required path.
