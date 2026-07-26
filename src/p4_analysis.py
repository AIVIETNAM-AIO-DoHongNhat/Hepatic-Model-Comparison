"""Reproducible P4 score, inference, OOF, and calibration workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats

REFERENCE_MODEL = "XGBoost"
COMPARATORS = (
    "Random Forest",
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
)
EXPECTED_MODELS = (REFERENCE_MODEL, *COMPARATORS)
EXPECTED_FOLDS = tuple(range(5))
SCORE_COLUMNS = ("model", "fold", "log_loss", "accuracy", "macro_f1")
ALPHA = 0.05


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_scores(scores: pd.DataFrame) -> dict:
    """Validate the accepted final six-model, five-fold score family."""
    errors: list[str] = []
    if tuple(scores.columns) != SCORE_COLUMNS:
        errors.append(f"Schema must be {SCORE_COLUMNS}, got {tuple(scores.columns)}.")
    if len(scores) != 30:
        errors.append(f"Expected 30 rows, got {len(scores)}.")
    if set(scores.get("model", [])) != set(EXPECTED_MODELS):
        errors.append("Model set does not match the six accepted final models.")
    if scores.duplicated(["model", "fold"]).any():
        errors.append("Duplicate model × fold rows found.")
    for model in EXPECTED_MODELS:
        folds = set(scores.loc[scores.get("model") == model, "fold"].astype(int))
        if folds != set(EXPECTED_FOLDS):
            errors.append(f"{model} folds are {sorted(folds)}, expected {list(EXPECTED_FOLDS)}.")
    numeric_columns = ["fold", "log_loss", "accuracy", "macro_f1"]
    numeric = scores[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        errors.append("Score table contains NaN or infinite numeric values.")
    if (numeric["log_loss"] < 0).any():
        errors.append("Log loss must be non-negative.")
    for metric in ("accuracy", "macro_f1"):
        if ((numeric[metric] < 0) | (numeric[metric] > 1)).any():
            errors.append(f"{metric} must lie within [0, 1].")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "rows": int(len(scores)),
        "models": sorted(set(scores.get("model", []))),
        "n_models": int(scores.get("model", pd.Series(dtype=str)).nunique()),
        "folds": sorted(numeric["fold"].dropna().astype(int).unique().tolist()),
        "duplicate_model_fold_rows": int(scores.duplicated(["model", "fold"]).sum()),
        "errors": errors,
    }
    if errors:
        raise ValueError("Score validation failed: " + " ".join(errors))
    return report


def build_scores_all(project_root: Path) -> tuple[pd.DataFrame, dict]:
    inputs = [project_root / "results" / f"scores_track_{track}.csv" for track in "abc"]
    scores = pd.concat([pd.read_csv(path) for path in inputs], ignore_index=True)
    scores["fold"] = scores["fold"].astype(int)
    report = validate_scores(scores)
    output = project_root / "results" / "scores_all.csv"
    scores.to_csv(output, index=False)
    report.update(
        {
            "output": output.relative_to(project_root).as_posix(),
            "inputs": [path.relative_to(project_root).as_posix() for path in inputs],
        }
    )
    _write_json(project_root / "results" / "validation" / "scores_all_validation.json", report)
    return scores, report


def _paired_vectors(scores: pd.DataFrame, comparator: str) -> tuple[np.ndarray, np.ndarray]:
    reference = (
        scores[scores["model"] == REFERENCE_MODEL].set_index("fold").loc[list(EXPECTED_FOLDS), "log_loss"]
    )
    compared = scores[scores["model"] == comparator].set_index("fold").loc[list(EXPECTED_FOLDS), "log_loss"]
    return reference.to_numpy(dtype=float), compared.to_numpy(dtype=float)


def run_hep13(scores: pd.DataFrame) -> pd.DataFrame:
    validate_scores(scores)
    rows = []
    raw_p_values = []
    for comparator in COMPARATORS:
        reference, compared = _paired_vectors(scores, comparator)
        diff = reference - compared
        t_result = stats.paired_ttest(reference, compared)
        ci = stats.ci_diff_t(reference, compared)
        raw_p_values.append(t_result.p_value)
        rows.append(
            {
                "reference_model": REFERENCE_MODEL,
                "comparator_model": comparator,
                "n_folds": len(diff),
                "mean_log_loss_reference": reference.mean(),
                "mean_log_loss_comparator": compared.mean(),
                "mean_difference": diff.mean(),
                "std_difference": diff.std(ddof=1),
                "ci_95_low": ci.low,
                "ci_95_high": ci.high,
                "t_statistic": t_result.statistic,
                "p_value_raw": t_result.p_value,
            }
        )
    correction = stats.bonferroni_correction(raw_p_values, alpha=ALPHA)
    for index, row in enumerate(rows):
        adjusted = float(correction.adjusted_p_values[index])
        significant = bool(correction.reject[index])
        mean_diff = row["mean_difference"]
        row.update(
            {
                "p_value_bonferroni": adjusted,
                "adjusted_alpha": correction.adjusted_alpha,
                "significant_bonferroni": significant,
                "effect_direction": (
                    "XGBoost_lower_log_loss"
                    if mean_diff < 0
                    else "comparator_lower_log_loss"
                    if mean_diff > 0
                    else "no_mean_difference"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_hep14(scores: pd.DataFrame) -> pd.DataFrame:
    validate_scores(scores)
    rows = []
    primary_p_values = []
    for comparator in COMPARATORS:
        reference, compared = _paired_vectors(scores, comparator)
        differences = reference - compared
        shapiro = stats.shapiro_test(differences)
        t_result = stats.paired_ttest(reference, compared)
        wilcoxon = stats.wilcoxon_test(reference, compared)
        normal = shapiro.p_value >= ALPHA
        primary_test = "paired_t_test" if normal else "wilcoxon"
        primary_p = t_result.p_value if normal else wilcoxon.p_value
        primary_p_values.append(primary_p)
        rows.append(
            {
                "reference_model": REFERENCE_MODEL,
                "comparator_model": comparator,
                "shapiro_statistic": shapiro.statistic,
                "shapiro_p_value": shapiro.p_value,
                "normality_assumption": "not_rejected_low_power_n5" if normal else "rejected",
                "t_test_p_value": t_result.p_value,
                "wilcoxon_statistic": wilcoxon.statistic,
                "wilcoxon_p_value": wilcoxon.p_value,
                "primary_test": primary_test,
                "primary_p_value_raw": primary_p,
                "mean_difference": differences.mean(),
            }
        )
    correction = stats.bonferroni_correction(primary_p_values, alpha=ALPHA)
    for index, row in enumerate(rows):
        adjusted = float(correction.adjusted_p_values[index])
        significant = bool(correction.reject[index])
        if significant and row["mean_difference"] < 0:
            interpretation = "Evidence that XGBoost has lower fold-level log loss after Bonferroni correction."
        elif significant and row["mean_difference"] > 0:
            interpretation = "Evidence that the comparator has lower fold-level log loss after Bonferroni correction."
        else:
            interpretation = "Insufficient evidence of a difference; this does not establish equivalence."
        row["primary_p_value_adjusted"] = adjusted
        row["interpretation"] = interpretation
    return pd.DataFrame(rows)


def write_inference_outputs(project_root: Path, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = project_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    hep13 = run_hep13(scores)
    hep14 = run_hep14(scores)
    hep13.to_csv(tables / "hep13_pairwise_comparisons.csv", index=False)
    hep14.to_csv(tables / "hep14_assumption_checks.csv", index=False)
    return hep13, hep14


def validate_oof_xgboost(oof: pd.DataFrame, official_folds: pd.DataFrame) -> dict:
    """Validate the frozen XGBoost OOF artifact and official fold alignment."""
    expected_columns = ("id", "fold", "model", "y_true", "prob_C", "prob_CL", "prob_D")
    errors: list[str] = []
    if tuple(oof.columns) != expected_columns:
        errors.append(f"Schema must be {expected_columns}, got {tuple(oof.columns)}.")
    if len(oof) != 9_600:
        errors.append(f"Expected 9600 rows, got {len(oof)}.")
    if oof["id"].nunique() != 9_600 or oof["id"].duplicated().any():
        errors.append("OOF must contain exactly 9600 unique IDs.")
    if set(oof["fold"].astype(int)) != set(EXPECTED_FOLDS):
        errors.append("OOF folds must be exactly 0..4.")
    fold_counts = oof.groupby("fold").size().sort_index().to_dict()
    if any(fold_counts.get(fold, 0) != 1_920 for fold in EXPECTED_FOLDS):
        errors.append(f"Each fold must contain 1920 rows; got {fold_counts}.")
    if set(oof["model"]) != {REFERENCE_MODEL}:
        errors.append("OOF model column must contain only XGBoost.")
    if not set(oof["y_true"]).issubset(set(stats.CLASS_ORDER)):
        errors.append("OOF y_true must use C, CL, D labels.")
    probabilities = oof[["prob_C", "prob_CL", "prob_D"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        errors.append("OOF probabilities contain NaN or Inf.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        errors.append("OOF probabilities must lie within [0, 1].")
    row_sum_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if row_sum_error > 1e-6:
        errors.append(f"OOF probability row sums exceed tolerance: {row_sum_error}.")
    official = official_folds.rename(columns={"fold_id": "official_fold"})
    aligned = oof[["id", "fold"]].merge(official, on="id", how="outer", indicator=True)
    fold_mismatches = int(
        ((aligned["_merge"] != "both") | (aligned["fold"] != aligned["official_fold"])).sum()
    )
    if fold_mismatches:
        errors.append(f"OOF has {fold_mismatches} ID/fold alignment mismatches.")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "rows": int(len(oof)),
        "unique_ids": int(oof["id"].nunique()),
        "fold_counts": {str(int(key)): int(value) for key, value in fold_counts.items()},
        "probability_row_sum_max_abs_error": row_sum_error,
        "probability_row_sum_tolerance": 1e-6,
        "fold_alignment_mismatches": fold_mismatches,
        "class_order": list(stats.CLASS_ORDER),
        "errors": errors,
    }
    if errors:
        raise ValueError("OOF validation failed: " + " ".join(errors))
    return report
