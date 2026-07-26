# Track C provenance audit for HEP-15

## Verdict

The accepted `results/scores_track_c.csv` has strong Git/notebook provenance, but the exact execution environment and full XGBoost parameter state are not recoverable from the repository. The safe HEP-15 unlock path is **Route 3 — provenance cannot be fully recovered**.

## Git provenance

- Authoritative score/notebook commit: `aa1ce1a7a4f93a6d5cd75533d267efb791375603`, authored 2026-07-24 22:35 +07:00.
- Commit subject: `feat(track-c): tune & freeze RF + XGBoost hyperparameters (HEP-23)`.
- That commit changed the Track C notebook, score CSV, README, tuning figures, and model figures; it did not change `data/` or `src/`.
- Current Track C score, notebook, processed features, labels, and fold artifacts have the same Git blobs as commit `aa1ce1a`.
- `results/models/Random_Forest.pkl` and `results/models/XGBOOST.pkl` come from earlier baseline commit `bde56f3`; they are not authoritative frozen Track C artifacts.

## Notebook execution evidence

Saved execution counts are monotonic from 1 through 13. Relevant cells executed in order:

1. Load processed data/folds: execution 3.
2. Define grid search: execution 4.
3. RF tuning: execution 5.
4. XGBoost tuning: execution 6.
5. Construct final models: execution 7.
6. Run final CV: execution 8.
7. Write `scores_track_c.csv`: execution 9.

The saved final-CV output agrees with the committed score CSV to the displayed precision. This is strong evidence against an out-of-order saved notebook. It does not recover the missing package versions or implicit XGBoost defaults.

## XGBoost parameter provenance

| Parameter | Repository evidence |
|---|---|
| `n_estimators` | Explicit: `400` |
| `max_depth` | Explicit: `2` |
| `learning_rate` | Explicit: `0.15` |
| `random_state` / seed | Explicit: `42` |
| `eval_metric` | Explicit: `mlogloss` |
| `n_jobs` | Explicit: `-1` |
| `objective` | Not explicit; wrapper behavior/default is version-dependent |
| `tree_method` | Not explicit |
| `subsample` | Not explicit |
| `colsample_bytree` | Not explicit |
| `min_child_weight` | Not explicit |
| `gamma` | Not explicit |
| `reg_alpha` | Not explicit |
| `reg_lambda` | Not explicit |
| `base_score` | Not explicit |
| Other prediction-affecting parameters | Not captured as a fitted-model dump or config manifest |

The notebook freezes the three tuned hyperparameters and seed-related arguments, but not the full model parameter state.

## Data and preprocessing provenance

- Input: `data/processed/train_X.csv`, 9,600 × 31, all `float64`, no missing values.
- Labels: `data/processed/train_y.csv`, mapping `0=C`, `1=CL`, `2=D`.
- Fold source: `data/interim/fold_id.csv`, five folds of 1,920 rows each.
- Feature order is the CSV column order and matches `feature_list.csv`.
- Numerical missing values were median-imputed from the 9,600-row training file; categorical missing values used `Unknown`.
- Transformations: selected `log1p` features plus Bilirubin/Albumin ratio, `OrdinalEncoder`, then `RobustScaler` for numerical columns.
- Encoders/scaler were fit once before CV and reused by Track C. This creates preprocessing leakage across CV folds, but both the accepted notebook and reproduction attempt consume the same frozen processed CSV, so it does not explain their mismatch.
- `metrics.compute_metrics` passes raw `predict_proba` output to scikit-learn `log_loss`; no explicit project-level probability clipping is applied.

Relevant SHA-256 values:

- `train_X.csv`: `17EECFDECE02A5261D42CED2FA0FAA46CFF08F9DC43CF63B9DFA70CE53329B9`.
- `train_y.csv`: `D72AAD5E5AD10397DFB67A90712838DCB3EB51CA72955E6C8FB79758DB717982`.
- `fold_id.csv`: `A75DC5286E5D2C82E4F21D54982667F706DA747BC71026F8B08645140F9A15E2`.

## Environment evidence

- Notebook metadata: Python `3.9.18`, kernelspec display name `dynamic`.
- Repository requirements are lower bounds, including `xgboost>=2.0.0` and `scikit-learn>=1.6.1`; they are not a lock file.
- No conda environment, pip freeze, poetry/Pipfile lock, CI environment capture, or saved package-version output was found.
- Diagnostic reproduction environment: Python 3.13.3, XGBoost 3.3.0, scikit-learn 1.8.0, NumPy 2.2.6, pandas 2.3.0, SciPy 1.17.0.
- Therefore environment drift is possible, but not confirmed as the causal mechanism.

## Reproduction differences

| Fold | Δ log loss (actual−expected) | Δ accuracy | Δ macro-F1 |
|---:|---:|---:|---:|
| 0 | -0.002307974 | -0.001041667 | +0.005138837 |
| 1 | +0.000089209 | -0.000520833 | -0.010915011 |
| 2 | +0.000794840 | +0.001562500 | +0.016487257 |
| 3 | +0.000737142 | 0.000000000 | +0.016901819 |
| 4 | +0.000637889 | +0.004687500 | +0.003741365 |

The mismatch is not a constant offset and is much larger than CSV rounding. Zero of five folds passed the all-metric `1e-5` gate.

## Root-cause confidence

| Finding | Confidence | Basis |
|---|---|---|
| Exact original package versions/full implicit XGBoost config are absent | CONFIRMED | No lock/version output/model dump in repo |
| Baseline `.pkl` files are not the final tuned models | CONFIRMED | Git history predates `aa1ce1a` |
| Current reproduction does not match final scores | CONFIRMED | Per-fold gate evidence |
| Accepted score and notebook/data/fold artifacts belong to the same Git state | CONFIRMED | Identical Git blobs at HEAD and `aa1ce1a` |
| Score CSV likely came from the saved sequential notebook run | HIGHLY_LIKELY | Monotonic execution counts and matching displayed output |
| Package/default drift caused the mismatch | POSSIBLE | Environments differ, but original package versions are unknown |
| Parallelism or platform-level nondeterminism contributed | POSSIBLE | `n_jobs=-1`; no same-environment repeat evidence |
| Current data/fold artifact mismatch caused the failure | UNSUPPORTED | Git blobs and positional alignment match |
| CSV rounding caused the failure | UNSUPPORTED | Differences greatly exceed rounding precision |

## Recommended HEP-15 route

**Route 3 — provenance cannot be recovered exactly from this repository.** Do not guess/install successive XGBoost versions until one matches, because that would select an environment post hoc against the target scores.

### Ready-to-send request to Track C owner

Please export validation-only OOF probabilities from the exact environment that created final `results/scores_track_c.csv` using schema:

`id,fold,model,y_true,prob_C,prob_CL,prob_D`

Please also provide:

1. Python, XGBoost, scikit-learn, NumPy, pandas and SciPy versions (`pip freeze` preferred).
2. Full fitted XGBoost config (`model.get_xgb_params()` plus `model.get_booster().save_config()`).
3. Exact preprocessing/data artifact hashes and ordered feature list.
4. Exact shared fold artifact/hash.
5. Per-fold log loss, accuracy and macro-F1 recomputed directly from the exported OOF.
6. Confirmation that every OOF row was predicted only by a model trained without that row's fold.

HEP-15 should resume only after this OOF reproduces the accepted Track C score file within the agreed tolerance.
