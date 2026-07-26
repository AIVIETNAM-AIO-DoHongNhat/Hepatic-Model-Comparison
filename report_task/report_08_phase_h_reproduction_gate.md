# Progress report 08 — Phase H: XGBoost reproduction gate

## Status

FAIL. Zero of five folds reproduced all metrics within `1e-5`.

## Evidence

- Maximum absolute metric difference: `0.016901819109425342`.
- Detailed fold/metric deltas: `results/validation/xgboost_oof_reproduction.csv`.
- Gate record: `results/validation/xgboost_oof_reproduction_gate.json`.

## Reproduction environment

- Python 3.13.3
- NumPy 2.2.6
- pandas 2.3.0
- SciPy 1.17.0
- scikit-learn 1.8.0
- XGBoost 3.3.0

The source notebook records Python 3.9.18 but does not freeze the original XGBoost/scikit-learn versions. Environment drift is therefore a plausible cause, but the gate result remains FAIL regardless of cause.

## Guardrail action

- Did not alter `scores_track_c.csv`.
- Did not create final calibration metrics or figures.
- Stopped HEP-15 as required.

## Next step

Obtain the exact Track C training environment/version lock or have the Track C owner export the original validation-fold probabilities.
