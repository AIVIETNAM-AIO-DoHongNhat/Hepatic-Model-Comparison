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


def reproduce_xgboost_oof(project_root: Path, tolerance: float = 1e-5) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Recreate validation-only OOF probabilities for the frozen XGBoost config."""
    from xgboost import XGBClassifier

    from . import data, metrics

    x, y, _ = data.load_processed()
    fold_series = data.load_folds()
    official_folds = pd.read_csv(project_root / "data" / "interim" / "fold_id.csv")
    if not (len(x) == len(y) == len(fold_series) == len(official_folds) == 9_600):
        raise ValueError("Processed data, labels, and official folds must all contain 9600 aligned rows.")

    probabilities = np.full((len(x), 3), np.nan, dtype=float)
    assigned = np.zeros(len(x), dtype=bool)
    for fold, train_index, valid_index in data.iter_folds(fold_series):
        if assigned[valid_index].any():
            raise RuntimeError(f"Fold {fold} attempts to overwrite existing OOF rows.")
        model = XGBClassifier(
            n_estimators=400,
            max_depth=2,
            learning_rate=0.15,
            random_state=data.RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        model.fit(x.iloc[train_index], y.iloc[train_index])
        if not np.array_equal(model.classes_, np.array(data.LABELS)):
            raise RuntimeError(f"Unexpected XGBoost class order: {model.classes_}.")
        probabilities[valid_index] = model.predict_proba(x.iloc[valid_index])
        assigned[valid_index] = True
    if not assigned.all():
        raise RuntimeError("Not every training row received exactly one validation-fold prediction.")

    class_names = np.asarray(data.CLASS_ORDER)
    oof = pd.DataFrame(
        {
            "id": official_folds["id"].to_numpy(),
            "fold": fold_series.to_numpy(dtype=int),
            "model": REFERENCE_MODEL,
            "y_true": class_names[y.to_numpy(dtype=int)],
            "prob_C": probabilities[:, 0],
            "prob_CL": probabilities[:, 1],
            "prob_D": probabilities[:, 2],
        }
    )
    validation = validate_oof_xgboost(oof, official_folds)
    output = project_root / "results" / "diagnostics" / "oof_xgboost_reproduction_attempt.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output, index=False)
    validation.update(
        {
            "output": output.relative_to(project_root).as_posix(),
            "frozen_config": {
                "n_estimators": 400,
                "max_depth": 2,
                "learning_rate": 0.15,
                "random_state": data.RANDOM_STATE,
                "eval_metric": "mlogloss",
                "n_jobs": -1,
            },
            "validation_only_generation": True,
            "fold_source": "data/interim/fold_id.csv",
        }
    )
    _write_json(project_root / "results" / "validation" / "oof_xgboost_validation.json", validation)

    expected = pd.read_csv(project_root / "results" / "scores_track_c.csv")
    expected = expected[expected["model"] == REFERENCE_MODEL].set_index("fold")
    y_numeric = pd.Series(oof["y_true"]).map(data.LABEL_MAP).to_numpy(dtype=int)
    reproduction_rows = []
    for fold in EXPECTED_FOLDS:
        mask = oof["fold"].to_numpy(dtype=int) == fold
        actual = metrics.compute_metrics(y_numeric[mask], probabilities[mask])
        row = {"fold": fold}
        for metric_name in metrics.METRIC_NAMES:
            expected_value = float(expected.loc[fold, metric_name])
            actual_value = float(actual[metric_name])
            row[f"expected_{metric_name}"] = expected_value
            row[f"actual_{metric_name}"] = actual_value
            row[f"abs_diff_{metric_name}"] = abs(actual_value - expected_value)
        reproduction_rows.append(row)
    reproduction = pd.DataFrame(reproduction_rows)
    diff_columns = [column for column in reproduction if column.startswith("abs_diff_")]
    max_abs_difference = float(reproduction[diff_columns].to_numpy().max())
    gate_pass = bool(max_abs_difference <= tolerance)
    reproduction["within_tolerance"] = reproduction[diff_columns].max(axis=1) <= tolerance
    reproduction.to_csv(project_root / "results" / "validation" / "xgboost_oof_reproduction.csv", index=False)
    gate = {
        "status": "PASS" if gate_pass else "FAIL",
        "tolerance": tolerance,
        "max_absolute_metric_difference": max_abs_difference,
        "folds_passed": int(reproduction["within_tolerance"].sum()),
        "folds_total": len(EXPECTED_FOLDS),
    }
    _write_json(project_root / "results" / "validation" / "xgboost_oof_reproduction_gate.json", gate)
    return oof, gate, reproduction


def run_hep15_calibration(project_root: Path, n_bins: int = 10) -> dict:
    """Create final calibration artifacts only after a passed reproduction gate."""
    gate_path = project_root / "results" / "validation" / "xgboost_oof_reproduction_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("HEP-15 calibration is blocked because the reproduction gate did not pass.")
    oof = pd.read_csv(
        project_root / "results" / "diagnostics" / "oof_xgboost_reproduction_attempt.csv"
    )
    official_folds = pd.read_csv(project_root / "data" / "interim" / "fold_id.csv")
    validate_oof_xgboost(oof, official_folds)
    label_map = {name: index for index, name in enumerate(stats.CLASS_ORDER)}
    y = oof["y_true"].map(label_map).to_numpy(dtype=int)
    proba = oof[["prob_C", "prob_CL", "prob_D"]].to_numpy(dtype=float)
    metrics_payload = {
        "model": REFERENCE_MODEL,
        "n_samples": len(oof),
        "class_order": ",".join(stats.CLASS_ORDER),
        "n_bins": n_bins,
        "binning_strategy": "uniform",
        "multiclass_brier_score": stats.brier_multiclass(y, proba),
        "top_label_ece": stats.top_label_ece(y, proba, n_bins=n_bins),
        "macro_classwise_ece": stats.macro_classwise_ece(y, proba, n_bins=n_bins),
    }
    tables = project_root / "tables"
    figures = project_root / "figures" / "calibration"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics_payload]).to_csv(tables / "hep15_calibration_metrics.csv", index=False)
    figure, _, top_bins, class_bins = stats.plot_reliability_diagram(y, proba, n_bins=n_bins)
    top_bins.to_csv(tables / "hep15_top_label_reliability_bins.csv", index=False)
    class_bins.to_csv(tables / "hep15_classwise_reliability_bins.csv", index=False)
    figure.savefig(figures / "xgboost_reliability_diagram.png", dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    _write_json(project_root / "results" / "validation" / "hep15_calibration_metrics.json", metrics_payload)
    return metrics_payload
