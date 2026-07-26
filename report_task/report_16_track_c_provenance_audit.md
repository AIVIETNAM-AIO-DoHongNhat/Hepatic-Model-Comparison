# Progress report 16 — Track C provenance audit

## Status

Completed read-only; no training or retuning was run.

## Verdict

Route 3: exact provenance cannot be recovered from the repository. Request the original validation-only OOF and full environment/config evidence from the Track C owner.

## Key findings

- Score, Track C notebook, processed data, labels, and folds match commit `aa1ce1a` by Git blob.
- Notebook execution counts are sequential and its displayed final scores match the CSV to displayed precision.
- Only six XGBoost arguments are explicit; objective and many prediction-affecting defaults are not frozen.
- Notebook records Python 3.9.18 but no package versions or lock file.
- Saved XGBoost/RF `.pkl` files are older baseline artifacts.
- Environment drift is POSSIBLE, not confirmed.
- Data/fold mismatch and CSV rounding are unsupported causes.

## Detailed evidence

- `report/track_c_provenance_audit.md`
- `results/diagnostics/track_c_provenance_audit.json`

## Next step

Run non-training cleanup validation and prepare the final review handoff.
