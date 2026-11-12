# ADR-0002: Canonical Experiment Output Artifact Schema

- **Status:** Accepted by the decision owner on 2026-11-12; dashboard migration remains open.
- **Decision owner:** Krishna (Team Lead / Integration)
- **Original proposal author:** Ronit Singh
- **Reviewers:** Aditya Rana, Rachit Gupta, Saumya Mishra
- **Related work:** MID-05, MID-06, END-05; GitHub issues #17 and #21

## 1. Context

The repository contains two incompatible artifact contracts:

1. The older flat ExperimentRecord contract in app/utils/schema.py and app/utils/loader.py, using top-level model_name, feature_group, seed, sample_size, and a nested metrics object.
2. The implemented higgs-experiment-v1 envelope used by src/higgs_event_lab/protocol.py, app/loader.py, the experiment exporters, docs/13_EXPERIMENT_PROTOCOL_AND_ARTIFACTS.md, and the committed per-run outputs/*/metrics.json artifacts.

The older draft described contract 1, while the training pipeline and primary artifact loader implement contract 2. Continuing to call both formats canonical would make experiment exports and dashboard interpretation ambiguous.

## 2. Decision

**The canonical schema is higgs-experiment-v1, the nested envelope already used by the training/export pipeline and committed run artifacts.** New runs and new documentation must use this contract. The old flat ExperimentRecord shape is legacy and must not be used for new exports.

### 2.1 Run package

Each run is stored under outputs/<run_id>/ and contains:

- metrics.json: versioned metadata, configuration, provenance and validation/test evaluation.
- test_predictions.csv: held-out test predictions with columns row_index, class_label, and score.

### 2.2 Metrics envelope

Required top-level fields:

| Field | Required content |
|---|---|
| schema_version | Exact string higgs-experiment-v1 |
| run_id | Non-empty unique run identifier |
| model | Object containing name and, where available, family |
| features | Object containing group, positive count, ordered names |
| data | Object containing source and row_count |
| split | Object containing protocol_version, seed, partition fractions and row counts |
| training | Object containing the actual run configuration and measured runtime; never copy timing values between models |
| evaluation | Object with separate validation and test metrics; optional validation-selected operating-point details |

The supported feature groups are low_level, high_level, and all. Metric names and meanings are defined in docs/13_EXPERIMENT_PROTOCOL_AND_ARTIFACTS.md. Validation and test metrics must never be combined or relabelled.

### 2.3 Prediction and curve artifacts

test_predictions.csv must contain:

- row_index: original row index in the source sample.
- class_label: binary ground-truth label (0 or 1).
- score: finite predicted score in [0, 1].

Curve files under outputs/curves/<run_id>.json use the run ID plus roc.fpr, roc.tpr, pr.precision, and pr.recall. Optional calibration coordinates must use the same documented envelope and equal-length arrays.

### 2.4 Provenance and privacy

Provenance paths must be repository-relative or stable source identifiers. Never commit absolute workstation, home-directory, cloud-notebook, temporary-directory or credential-bearing paths. Record source hashes, sample size, sampling method, seed, split version and commit identity where available. Raw HIGGS data and model checkpoints remain excluded from Git.

## 3. Legacy compatibility and migration

- app/loader.py and the higgs-experiment-v1 protocol are the canonical implementation direction.
- app/utils/schema.py and app/utils/loader.py currently implement the older flat contract. They are legacy code, not a second accepted schema.
- Ronit's dashboard consolidation issue #21 owns selecting the single app entry point and either migrating the legacy loader to higgs-experiment-v1 or removing it with regression coverage.
- During migration, the dashboard must not silently reinterpret a flat legacy record as a nested v1 artifact. Invalid or unsupported records must produce an explicit diagnostic/empty state, never fabricated metrics.
- Do not change the schema version for additive optional metadata. Any incompatible change requires a new version and a migration note.

## 4. Consequences

This decision matches the format already emitted by the current experiment pipeline and committed run artifacts. It avoids rewriting measured results merely to fit the superseded flat proposal. It also makes issue #21's remaining dashboard work explicit: consolidate the loaders around this one canonical contract and verify valid, missing, malformed and multiple-run states.

## 5. Verification and release boundary

The schema decision is accepted, but this ADR alone does not prove the dashboard migration or release readiness. Before the release gate closes:

- Run schema and artifact-loader unit tests against valid, incomplete and malformed fixtures.
- Confirm all current dashboard paths consume higgs-experiment-v1.
- Run the full test suite from a clean clone.
- Keep the 250k and 1M model results separate until the T1/T4/T5 reconciliation evidence and Krishna's headline-results decision are recorded.

## 6. Decision history

- 2026-11-12: Krishna selected the existing higgs-experiment-v1 nested contract as canonical. Legacy loader consolidation remains assigned to dashboard issue #21.
