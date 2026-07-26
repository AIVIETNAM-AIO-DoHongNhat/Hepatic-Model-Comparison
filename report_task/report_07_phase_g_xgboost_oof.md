# Progress report 07 — Phase G: XGBoost OOF recreation

## Status

OOF generation completed; structural validation passed.

## Artifact

- `results/diagnostics/oof_xgboost_reproduction_attempt.csv`

## Generation protocol

- Frozen config: 400 estimators, max depth 2, learning rate 0.15, seed 42.
- Shared processed features and official `fold_id.csv`.
- Each model was fit on four folds and predicted only the held-out validation fold.
- No tuning and no test-set use.

## Validation

- Rows/unique IDs: 9,600 / 9,600.
- Fold counts: 1,920 each.
- Fold alignment mismatches: 0.
- Class order: C, CL, D.
- Maximum probability row-sum error: `8.731149137020111e-08`.
- Accepted row-sum tolerance: `1e-6`, appropriate for float32 probabilities.
- Structural status: PASS.

## Important qualification

Structural validity does not establish reproduction of the accepted final Track C scores. That is decided by the separate reproduction gate.

After the gate failed, this file was quarantined as a diagnostic attempt and is not approved for calibration.

## Next step

Compare per-fold metrics against `scores_track_c.csv`.
