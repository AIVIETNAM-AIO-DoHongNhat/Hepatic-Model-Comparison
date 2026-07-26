# Progress report 14 — Failed OOF diagnostic cleanup

## Status

Completed.

## Cleanup

- Moved the structurally valid but non-reproducing file out of the final-results namespace.
- New path: `results/diagnostics/oof_xgboost_reproduction_attempt.csv`.
- Added machine-readable metadata with `REPRODUCTION_FAILED` and `approved_for_calibration: false`.
- Updated code, notebook, validation metadata, and reports to use diagnostic wording/path.
- Preserved all reproduction evidence.

## Guardrail

The diagnostic attempt must not be used for final ECE, Brier score, reliability bins, or reliability diagrams.

## Next step

Audit the phase-report collection and create a canonical index without deleting required history.
