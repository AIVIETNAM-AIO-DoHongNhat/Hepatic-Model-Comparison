"""Leakage-safe evaluation pipeline for Track A (KNN + Logistic Regression)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, RobustScaler

from .data import CLASS_ORDER, LABEL_MAP, LABELS, RANDOM_STATE, load_raw
from .metrics import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = PROJECT_ROOT / "tables"
TARGET_COL = "Status"
ID_COL = "id"

CAT_COLS = [
    "Treatment_Assignment",
    "Patient_Sex",
    "Ascites_Indicator",
    "Liver_Enlargement",
    "Spider_Angioma",
]
FLAG_FEATURES = [
    "Cholesterol_Level",
    "Triglyceride_Level",
    "Copper_Level",
    "Alkaline_Phosphatase",
    "AST_Level",
    "Platelet_Count",
    "Prothrombin_Time",
    "Treatment_Assignment",
    "Ascites_Indicator",
    "Liver_Enlargement",
    "Spider_Angioma",
    "Edema_Status",
]
LOG_TRANSFORM_COLS = [
    "Bilirubin_Level",
    "Cholesterol_Level",
    "Copper_Level",
    "Alkaline_Phosphatase",
    "AST_Level",
    "Triglyceride_Level",
]

KNN_GRID = {"model__n_neighbors": [3, 5, 11, 21, 31]}
LOGREG_GRID = {"model__C": [0.1, 1.0, 10.0]}


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature construction without learning from the data."""
    df = frame.copy()
    for feature in FLAG_FEATURES:
        if feature in df.columns:
            df[f"is_missing_{feature}"] = df[feature].isna().astype(float)

    df["Patient_Age_Years"] = df["Patient_Age_Days"] / 365.25
    df["Bilirubin_Albumin_Ratio"] = df["Bilirubin_Level"] / (df["Albumin_Level"] + 1e-6)
    for column in LOG_TRANSFORM_COLS:
        df[column] = np.log1p(df[column])
    df["Bilirubin_Albumin_Ratio"] = np.log1p(df["Bilirubin_Albumin_Ratio"])
    return df.drop(columns=[ID_COL, "Edema_Status"], errors="ignore")


def build_preprocessor(scale: bool = True) -> Pipeline:
    """Build preprocessing whose learned state is fitted inside each CV fold."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", RobustScaler()))

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                make_column_selector(dtype_include=np.number),
            ),
            ("categorical", categorical, CAT_COLS),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            (
                "features",
                FunctionTransformer(engineer_features, validate=False),
            ),
            ("columns", columns),
        ]
    )


def build_knn_pipeline(n_neighbors: int = 5, scale: bool = True) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale=scale)),
            (
                "model",
                KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform"),
            ),
        ]
    )


def build_logreg_pipeline(c_value: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale=True)),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load raw training features, encoded target, and shared folds aligned by id."""
    train_df, _ = load_raw()
    fold_df = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "fold_id.csv")
    if fold_df[ID_COL].duplicated().any():
        raise ValueError("fold_id.csv contains duplicate ids")

    aligned = train_df[[ID_COL]].merge(fold_df, on=ID_COL, how="left", validate="1:1")
    if aligned["fold_id"].isna().any():
        raise ValueError("Some training ids are missing from fold_id.csv")

    y = train_df[TARGET_COL].map(LABEL_MAP)
    if y.isna().any():
        raise ValueError(f"Unexpected target value; expected {CLASS_ORDER}")
    X = train_df.drop(columns=[TARGET_COL])
    return X, y.astype(int), aligned["fold_id"].astype(int)


def validate_fold_assignments(fold_ids: pd.Series, expected_folds: int = 10) -> None:
    values = np.asarray(fold_ids, dtype=int)
    expected = set(range(expected_folds))
    actual = set(np.unique(values).tolist())
    if actual != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, found {sorted(actual)}")
    if len(values) == 0:
        raise ValueError("Fold assignment is empty")


def _predict_proba_checked(estimator: Pipeline, X_valid: pd.DataFrame) -> np.ndarray:
    classes = estimator.named_steps["model"].classes_.tolist()
    if classes != LABELS:
        raise ValueError(f"Probability columns are {classes}, expected {LABELS}")
    proba = np.asarray(estimator.predict_proba(X_valid), dtype=float)
    if proba.shape != (len(X_valid), len(LABELS)):
        raise ValueError(f"Unexpected probability shape: {proba.shape}")
    if not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Probability rows do not sum to one")
    return proba


def _score_row(model_name: str, fold: int, y_true: pd.Series, proba: np.ndarray):
    return {"model": model_name, "fold": fold, **compute_metrics(y_true, proba)}


def evaluate_track_a(
    X: pd.DataFrame,
    y: pd.Series,
    fold_ids: pd.Series,
    inner_splits: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested-CV evaluation of tuned KNN and Logistic Regression."""
    validate_fold_assignments(fold_ids)
    score_rows = []
    tuning_rows = []

    for fold in sorted(pd.unique(fold_ids)):
        train_idx = np.flatnonzero(np.asarray(fold_ids) != fold)
        valid_idx = np.flatnonzero(np.asarray(fold_ids) == fold)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=RANDOM_STATE + int(fold),
        )
        specs = [
            ("KNN", build_knn_pipeline(scale=True), KNN_GRID),
            ("Logistic_Regression", build_logreg_pipeline(), LOGREG_GRID),
        ]
        for model_name, pipeline, param_grid in specs:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring="neg_log_loss",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            proba = _predict_proba_checked(search.best_estimator_, X_valid)
            score_rows.append(_score_row(model_name, int(fold), y_valid, proba))
            tuning_rows.append(
                {
                    "model": model_name,
                    "outer_fold": int(fold),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    "inner_log_loss": -float(search.best_score_),
                }
            )

    return pd.DataFrame(score_rows), pd.DataFrame(tuning_rows)


def run_scaling_ablation(
    X: pd.DataFrame, y: pd.Series, fold_ids: pd.Series, n_neighbors: int = 5
) -> pd.DataFrame:
    """Compare KNN with and without scaling, holding all other choices fixed."""
    validate_fold_assignments(fold_ids)
    rows = []
    for fold in sorted(pd.unique(fold_ids)):
        train_idx = np.flatnonzero(np.asarray(fold_ids) != fold)
        valid_idx = np.flatnonzero(np.asarray(fold_ids) == fold)
        for label, scale in [("KNN_scaled", True), ("KNN_unscaled", False)]:
            estimator = build_knn_pipeline(n_neighbors=n_neighbors, scale=scale)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = _predict_proba_checked(estimator, X.iloc[valid_idx])
            row = _score_row(label, int(fold), y.iloc[valid_idx], proba)
            row["zero_probability_rate"] = float(np.mean(proba == 0.0))
            row["mean_max_probability"] = float(np.mean(proba.max(axis=1)))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "log_loss",
        "accuracy",
        "macro_f1",
        "zero_probability_rate",
        "mean_max_probability",
    ]
    summary = ablation.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    scaled_loss = summary.loc[summary["model"] == "KNN_scaled", "log_loss_mean"].iloc[0]
    summary["log_loss_delta_vs_scaled"] = summary["log_loss_mean"] - scaled_loss
    return summary


def run_experiment() -> dict[str, pd.DataFrame]:
    X, y, fold_ids = load_training_data()
    scores, tuning = evaluate_track_a(X, y, fold_ids)
    ablation = run_scaling_ablation(X, y, fold_ids)
    ablation_summary = summarize_ablation(ablation)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "scores_track_a.csv", index=False)
    tuning.to_csv(RESULTS_DIR / "tuning_track_a.csv", index=False)
    ablation.to_csv(RESULTS_DIR / "ablation_scaling_track_a.csv", index=False)
    ablation_summary.to_csv(TABLES_DIR / "track_a_scaling_ablation.csv", index=False)
    return {
        "scores": scores,
        "tuning": tuning,
        "ablation": ablation,
        "ablation_summary": ablation_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    outputs = run_experiment()
    metrics = ["log_loss", "accuracy", "macro_f1"]
    print(outputs["scores"].groupby("model")[metrics].agg(["mean", "std"]))
    print("\nScaling ablation:\n", outputs["ablation_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
