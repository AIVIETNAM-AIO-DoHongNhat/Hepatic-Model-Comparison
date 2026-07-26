# Progress report 03 — HEP-9 calibration module

## Status

Completed with synthetic test evidence.

## Work completed

- Added multiclass Brier score, top-label ECE, and macro classwise ECE.
- Added top-label and classwise reliability-bin tables with empty-bin retention.
- Added a reliability-diagram plotting function with sample-count annotations.
- Enforced fixed class order `C, CL, D` and strict probability validation.
- Retained backward-compatible calibration wrappers used by earlier documentation.
- Added five synthetic tests covering perfect, calibrated-like, overconfident-wrong, invalid, and plotting cases.

## Files changed

- `src/stats.py`
- `tests/test_calibration.py`

## Validation

- Combined suite result at completion: 12 tests passed.

## Next step

Create and validate the final combined score table.
