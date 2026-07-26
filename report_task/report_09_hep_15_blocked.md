# Progress report 09 — HEP-15 calibration

## Status

Blocked by the failed reproduction gate.

## Available input

The diagnostic OOF attempt at `results/diagnostics/oof_xgboost_reproduction_attempt.csv` is structurally valid, but it is not accepted as the frozen final XGBoost OOF because its fold metrics do not reproduce the final score file.

## Artifacts intentionally not created

- Final HEP-15 calibration metrics.
- Final reliability-bin tables.
- Final reliability diagram.

## Safe resolution

Use OOF probabilities exported from the exact original Track C environment, or recreate that environment from exact package versions and rerun the gate. Calibration must remain blocked until the gate passes.

## Next step

Complete the statistics notebook with HEP-15 visibly marked as gated/blocked, then run final non-calibration validation.
