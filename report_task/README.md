# P4 progress-report index

The user-requested reporting policy requires one sequential report after each completed work item. For that reason, phase reports are retained as an audit trail rather than deleted or collapsed.

## Canonical reports

- Latest implementation handoff before cleanup: `report_12_phase_l_final_handoff.md`.
- Latest cleanup/provenance handoff: use the highest-numbered `final_handoff` report.
- Machine-readable evidence remains under `results/validation/`, `results/diagnostics/`, and `tables/`.

## Phase evidence

| Report | Purpose |
|---|---|
| 01 | Protocol formalization |
| 02 | HEP-7 implementation/tests |
| 03 | HEP-9 implementation/tests |
| 04 | Final score aggregation |
| 05 | HEP-13 paired-t analysis |
| 06 | HEP-14 assumptions/primary tests |
| 07 | Diagnostic OOF creation |
| 08 | Reproduction gate failure |
| 09 | HEP-15 blocked decision |
| 10 | Statistics notebook |
| 11 | Final validation before review |
| 12 | Approved implementation handoff |
| 13 | Statistical wording and HEP-25 cleanup |
| 14 | Failed OOF quarantine |
| 15 | Report collection audit/index decision |
| 16 | Track C provenance audit |
| 17 | Cleanup validation |
| 18 | Canonical cleanup/provenance final handoff |
| 19 | Commit 1 approved statistics scope |
| 20 | Commit 2 HEP-15 diagnostic scope |

## Retention decision

- **Keep:** all numbered reports, because they satisfy the explicit per-task reporting policy and preserve review history.
- **Do not merge/delete:** intermediate reports; later reports may correct interpretation but should not erase the earlier evidence trail.
- **Canonical navigation:** this index plus the latest final handoff prevents users from treating every phase report as current guidance.
