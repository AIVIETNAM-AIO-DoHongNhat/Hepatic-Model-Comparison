# Progress report 10 — Phase J: Statistics notebook

## Status

Completed and smoke-tested.

## Notebook contents

- Final score loading and validation.
- Performance summary.
- Five HEP-13 comparisons and confidence intervals.
- Bonferroni results.
- HEP-14 Shapiro-Wilk checks and Wilcoxon sensitivity analysis.
- XGBoost OOF structural validation.
- Reproduction gate display.
- Conditional calibration section that refuses to report HEP-15 when the gate fails.
- Required methodological limitations.

## Validation

- Every code cell executed sequentially from the repository root.
- Smoke-test status: PASS.
- The notebook correctly printed that HEP-15 is blocked.
- No absolute path is stored in the notebook.

## Files changed

- `notebooks/08_statistics.ipynb`
- `scripts/smoke_notebook.py`

## Next step

Run the complete final validation suite and inspect the Git diff.
