# Progress report 13 — Statistical wording and HEP-25 cleanup

## Status

Completed.

## Corrections

- Made HEP-14 primary-test results authoritative for final wording.
- Restricted the final significant set to Random Forest, Decision Tree, KNN, and Naive Bayes.
- Documented the Logistic Regression Shapiro and Wilcoxon results without claiming equivalence.
- Explained the paired-t versus Wilcoxon divergence.
- Added the exact-Wilcoxon resolution limitation for five folds and overlapping-fold caution.
- Set HEP-25 status to `COMPLETE VIA LEADER-APPROVED POST-RESULT PROTOCOL AMENDMENT`.
- Preserved the ignored historical task unchanged.

## Files updated

- `README.md`
- `notebooks/08_statistics.ipynb`
- `report_task/report_05_hep_13_pairwise_inference.md`
- `report_task/report_06_hep_14_assumptions.md`
- `report_task/report_12_phase_l_final_handoff.md`

## Next step

Quarantine the failed OOF reproduction attempt and add explicit diagnostic metadata.
