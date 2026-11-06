# Local persistence

SQLite schema version 2 is created through `storage/database.py`. The implemented migration from database version 1 adds feature-schema tags to observation/calibration rows without deleting historical data. Old numerical feature vectors remain available for retention/audit but are excluded from current dashboard and training queries. Startup rejects other unrecognized versions rather than silently overwriting data. `PRAGMA user_version` and `schema_version` identify the database version. Back up before future migrations; no speculative migration is applied.

| Table | Purpose and principal fields |
|---|---|
| observations | Timestamp, context/confidence/source, JSON numeric vector, validity, missing fraction, classification, scores, persistence, method, inference latency and profile version |
| profiles | Context primary key and active model version |
| model_versions | Composite context/version key, context foreign key, status, artifact-relative directory and training metadata |
| calibration_samples | Approved numerical observations with context, timestamp, continuity segment and normal-data provenance |
| events | One episode per anomaly/drift; start/update/end timestamps, context, class, peak score, duration, explanation and version |
| context_history | Context-transition timestamps, source and confidence |
| adaptations | Collection/review/acceptance/rejection/interruption audit, previous/candidate versions, stability and elapsed duration |
| settings | Validated application configuration and local executable-category mappings |
| evaluations | Executed result JSON with mandatory `synthetic`, `real_laptop` or `labelled_local` provenance |
| system_events | Minimal error/rollback audit, excluding raw activity |

Observation time/context, calibration context/time and event time indexes support live queries and retention. Relationships from model versions to profiles are enforced by foreign keys. Event profile versions are retained as historical references even when raw observations expire. Active-version changes are transactional.

## Model files

`<private-data>/profiles/<context-enum-name>/vN/` contains:

- `model.joblib`: fitted RobustScaler, Isolation Forest and score-calibration distributions.
- `temporal.pt`: native trained PyTorch state dict for reproducibility.
- `temporal.onnx`: fixed-shape, CPU inference graph.
- `metadata.json`: feature/schema contract, training/validation counts, baseline aggregate statistics, provenance, losses, thresholds, parity evidence, file sizes and SHA-256 digests.

Artifacts are written through temporary files before database registration. Existing version directories are never overwritten; orphaned directories from interrupted writes cause version numbers to advance. Loading checks schema and checksums before deserializing. SHA-256 detects accidental corruption, not a malicious local actor who can rewrite both artifacts and metadata. Joblib is for self-generated trusted local data only. Do not import downloaded model files.

## Retention and deletion

Hourly retention removes expired observations, calibration rows, events, context history, system events and evaluation records. Model versions and adaptation audit metadata remain available for rollback until explicit deletion. Observation deletion retains models; full deletion also removes profiles, settings and audit history. The verified model directory must be exactly the application's `profiles` directory before recursive deletion is allowed. SQLite secure_delete, WAL checkpoint/truncation and VACUUM reduce logical remnants; SSD/backup forensic erasure is not guaranteed.

Model and SQLite files use the local account's inherited permissions; encryption at rest is not provided. Use OS account isolation and disk encryption if required. Backups are the user's responsibility. Never share an active data directory across multiple machines or network filesystems.
