# Progress report 06 — HEP-14 assumptions and sensitivity

## Status

Completed.

## Output

- `tables/hep14_assumption_checks.csv`

## Findings

- Shapiro-Wilk did not reject normality for Random Forest, Decision Tree, KNN, or Naive Bayes differences.
- Shapiro-Wilk rejected normality for Logistic Regression (`p ≈ 0.01423`).
- Logistic Regression therefore uses Wilcoxon as the protocol-defined primary test.
- Its Wilcoxon raw p-value is `0.0625`; Bonferroni-adjusted p-value is `0.3125`, so evidence is insufficient under the primary-test protocol.
- The other four primary paired-t tests remain significant after Bonferroni correction.

## Required caution

- With only five differences, Shapiro-Wilk has very low power.
- With `n=5`, two-sided exact Wilcoxon has a minimum attainable p-value of `0.0625` when all five non-zero differences have the same sign; it therefore cannot reach adjusted alpha `0.01`.
- CV training sets overlap, so fold-level p-values are not based on fully independent experiments.
- A non-significant result does not establish equivalence.

## Next step

Recreate XGBoost OOF probabilities and enforce the reproduction gate before HEP-15.
