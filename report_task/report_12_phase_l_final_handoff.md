# Progress report 12 — Phase L: Final handoff

## Executive status

- HEP-7: complete with tests.
- HEP-9: complete with tests.
- HEP-25: complete in tracked documentation.
- HEP-13: complete.
- HEP-14: complete.
- HEP-15: blocked correctly after reproduction gate failure.

## Final validation

- Unit tests: 12/12 passed.
- Score validation: PASS.
- OOF structural validation: PASS.
- XGBoost frozen-score reproduction: FAIL, 0/5 folds within tolerance across all metrics.
- Notebook smoke: PASS.
- Git diff check: no whitespace errors; only Windows LF/CRLF conversion warnings.

## Statistical summary

The HEP-13 paired-t analysis supports lower XGBoost log loss against all five comparators after Bonferroni correction. Under the HEP-14 protocol-defined primary tests, Logistic Regression switches to Wilcoxon after Shapiro rejection and is not significant after correction; the other four comparisons remain significant.

Paired t-based inference and Wilcoxon differ because of their assumptions and sensitivity. With only five paired folds, Shapiro-Wilk has very low power and two-sided exact Wilcoxon has minimum attainable p-value `0.0625` for five same-direction non-zero differences. It cannot reach Bonferroni adjusted alpha `0.01`; this limitation must not be presented as equivalence. Fold training sets also overlap.

## HEP-15 blocker

The recreated OOF is structurally valid but does not reproduce the accepted Track C metrics. No final calibration artifact was created. Exact original package versions or original OOF probabilities are required.

## Git action

- No commit.
- No push.
- No merge.

## Proposed commit message after review

`feat(statistics): complete HEP-7/9/13/14 and add gated XGBoost OOF workflow`
