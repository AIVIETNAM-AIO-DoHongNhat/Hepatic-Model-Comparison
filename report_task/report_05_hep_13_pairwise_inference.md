# Progress report 05 — HEP-13 paired inference

## Status

Completed for the approved five-comparison family.

## Output

- `tables/hep13_pairwise_comparisons.csv`

## Findings

- All five mean differences (`XGBoost - comparator`) are negative.
- All five Student-t 95% confidence intervals lie entirely below zero.
- All five paired t-test p-values remain significant after Bonferroni correction.
- Adjusted alpha is `0.01`; family size is `5`.

These results support lower fold-level log loss for XGBoost under the paired-t analysis. They do not remove the post-result protocol, same-fold tuning/evaluation, or overlapping-training-set limitations.

This is not the final cross-method conclusion. HEP-14 controls the final wording: the Logistic Regression comparison switches to Wilcoxon after Shapiro rejection and is not significant after Bonferroni correction.

## Next step

Run assumption checks and Wilcoxon sensitivity analysis for HEP-14.
