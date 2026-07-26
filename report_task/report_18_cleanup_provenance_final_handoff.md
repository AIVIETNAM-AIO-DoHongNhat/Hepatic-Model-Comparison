# Progress report 18 — Cleanup and provenance final handoff

## Executive summary

- Statistical wording now follows the HEP-14 primary-test protocol.
- HEP-25 is explicitly complete via a leader-approved post-result amendment.
- The failed OOF attempt is quarantined and machine-labelled as not approved for calibration.
- All sequential reports are retained under the user-mandated reporting policy and indexed by `report_task/README.md`.
- Track C exact environment/full implicit configuration cannot be recovered; HEP-15 remains blocked under Route 3.

## Final statistical conclusion

There is corrected-protocol evidence that XGBoost has lower fold-level log loss than Random Forest, Decision Tree, KNN, and Naive Bayes. For Logistic Regression, Shapiro-Wilk p is approximately 0.01423, the primary test is exact two-sided Wilcoxon, raw p is 0.0625, and Bonferroni-adjusted p is 0.3125. Evidence is insufficient under the primary protocol; this does not establish equivalence.

With five non-zero same-direction differences, exact two-sided Wilcoxon cannot produce p below 0.0625, so it cannot reach adjusted alpha 0.01. Shapiro has low power at n=5, and overlapping CV training sets further limit inference.

## HEP-25 status

`COMPLETE VIA LEADER-APPROVED POST-RESULT PROTOCOL AMENDMENT`.

The historical task retains its original 10-fold/prior-declaration language. The leader later approved final shared 5-fold analysis and five XGBoost-reference comparisons after preliminary results; this is not preregistration.

## OOF cleanup

- Diagnostic attempt: `results/diagnostics/oof_xgboost_reproduction_attempt.csv`.
- Metadata status: `REPRODUCTION_FAILED`.
- `approved_for_calibration`: false.
- Root-level final OOF path: absent.
- Final calibration metrics/figure: absent.

## Track C provenance verdict

Route 3 — exact provenance is not recoverable from repository evidence. Package/default drift is POSSIBLE, not confirmed. Score/notebook/data/fold Git alignment is confirmed; baseline model pickles are confirmed stale relative to the tuned commit.

## Validation

- 12/12 unit tests passed.
- Score validator passed.
- Notebook smoke passed and displayed HEP-15 blocked.
- Diagnostic guard passed.
- `git diff --check` reported no whitespace errors, only LF/CRLF warnings.

## Proposed commit split

### Approved statistics scope

Suggested message:

`feat(statistics): finalize HEP-7/9/25/13/14 analysis`

Include statistics/calibration utilities and tests, README protocol/conclusions, `scores_all.csv`, HEP-13/14 tables, score validation, analysis script, statistics notebook, and non-diagnostic progress reports.

### Failed diagnostic evidence — exclude from approved-ticket commit

- `results/diagnostics/oof_xgboost_reproduction_attempt.csv`
- `results/diagnostics/oof_xgboost_reproduction_attempt.metadata.json`
- `results/validation/oof_xgboost_validation.json`
- `results/validation/xgboost_oof_reproduction.csv`
- `results/validation/xgboost_oof_reproduction_gate.json`
- Diagnostic-only phase reports 07, 08, 09, 14 and 16
- `report/track_c_provenance_audit.md`
- `results/diagnostics/track_c_provenance_audit.json`

Preserve these files in the working tree; if the team wants repository-level provenance, commit them separately as diagnostics rather than mixing them into the approved HEP-7/9/25/13/14 commit.

## Git action

No commit, push, merge, retraining, retuning, or final calibration was performed.
