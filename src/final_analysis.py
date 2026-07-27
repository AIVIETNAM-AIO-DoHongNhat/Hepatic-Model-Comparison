"""Quy trình phân tích cuối, tái lập được: bảng điểm, kiểm định, OOF và calibration.

`stats.py` giữ các phép toán thuần (t-test, Wilcoxon, Bonferroni, ECE); module này biết
file nào nằm ở đâu, mô hình nào là tham chiếu, thứ tự kiểm tra ra sao và ghi artifact ra đĩa.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
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
OOF_COLUMNS = ("id", "fold", "model", "y_true", "prob_C", "prob_CL", "prob_D")
ALPHA = 0.05

# Hyperparameter đã freeze ở notebooks/06_model_track_c.ipynb (mục 5.3), chọn theo
# log loss trung bình thấp nhất trên 5 fold dùng chung.
XGBOOST_FROZEN_PARAMS = {"n_estimators": 400, "max_depth": 2, "learning_rate": 0.15}

# Hyperparameter đã freeze của cả 6 mô hình, kèm nơi đã chốt. Đây là bản khai báo duy nhất
# dùng cho việc đo runtime; các notebook track vẫn tự khai báo model của mình như cũ.
FROZEN_MODEL_SPECS = {
    "XGBoost": ("notebooks/06 mục 5.3", XGBOOST_FROZEN_PARAMS),
    "Random Forest": ("notebooks/06 mục 5.3", {"n_estimators": 600, "max_depth": 14}),
    "Logistic Regression": ("notebooks/04 mục 3.5", {"max_iter": 1000}),
    "Decision Tree": ("notebooks/05 mục 4.3", {"max_depth": 3}),
    "KNN": ("notebooks/04 mục 3.3", {"n_neighbors": 51}),
    "Naive Bayes": ("mặc định, không có tham số cần tune", {}),
}

RUNTIME_TABLE = "results/runtime.csv"

# Commit đã xác lập provenance cho scores_track_c.csv (xem report/track_c_provenance_audit.md).
TRACK_C_PROVENANCE_COMMIT = "aa1ce1a7a4f93a6d5cd75533d267efb791375603"

# Môi trường authoritative: kernel "dynamic" của notebook 06 (Python 3.9.18).
AUTHORITATIVE_CONDA_ENV = "dynamic"
AUTHORITATIVE_PYTHON = "3.9.18"

# Các file đầu vào cần chốt hash để bên nhận verify được provenance.
PROVENANCE_INPUTS = (
    "data/processed/train_X.csv",
    "data/processed/train_y.csv",
    "data/interim/fold_id.csv",
    "data/processed/feature_list.csv",
    "data/processed/scaler.pkl",
    "data/processed/cat_encoder.pkl",
    "data/processed/label_encoder.pkl",
    "results/scores_track_c.csv",
)

# Đường dẫn artifact authoritative (nguồn duy nhất cho phân tích calibration).
AUTHORITATIVE_OOF = "results/oof/oof_xgboost_track_c.csv"
AUTHORITATIVE_GATE = "results/oof/oof_xgboost_track_c.gate.json"
AUTHORITATIVE_MANIFEST = "results/oof/oof_xgboost_track_c.manifest.json"
AUTHORITATIVE_PER_FOLD = "results/oof/oof_xgboost_track_c.per_fold_metrics.csv"

# Đầu ra của phần kiểm định (mục 7.4 - 7.5 của notebook 08).
PAIRWISE_TABLE = "pairwise_comparisons.csv"
ASSUMPTION_TABLE = "assumption_checks.csv"

# Đầu ra của phân tích calibration (mô hình có log loss thấp nhất).
CALIBRATION_METRICS_TABLE = "tables/calibration_metrics.csv"
CALIBRATION_TOP_LABEL_BINS_TABLE = "tables/calibration_top_label_reliability_bins.csv"
CALIBRATION_CLASSWISE_BINS_TABLE = "tables/calibration_classwise_reliability_bins.csv"
CALIBRATION_SENSITIVITY_TABLE = "tables/calibration_ece_bin_sensitivity.csv"
CALIBRATION_FIGURE = "figures/calibration/xgboost_reliability_diagram.png"
CALIBRATION_METRICS_JSON = "results/validation/calibration_metrics.json"

# ECE phụ thuộc cách chia bin, nên luôn báo cáo kèm bảng độ nhạy.
CALIBRATION_BIN_GRID = (5, 10, 15, 20)

# Đường dẫn diagnostic cũ (giữ nguyên để không xê dịch artifact đã commit).
DIAGNOSTIC_OOF = "results/diagnostics/oof_xgboost_reproduction_attempt.csv"
DIAGNOSTIC_VALIDATION = "results/validation/oof_xgboost_validation.json"
DIAGNOSTIC_PER_FOLD = "results/validation/xgboost_oof_reproduction.csv"
DIAGNOSTIC_GATE = "results/validation/xgboost_oof_reproduction_gate.json"

# sklearn >= 1.6 cảnh báo vì xgboost trả xác suất float32 (tổng hàng lệch ~1e-7).
# Không normalize lại: làm vậy sẽ đổi số so với scores_track_c.csv.
KNOWN_BENIGN_WARNINGS = (
    "sklearn.metrics log_loss: 'The y_pred values do not sum to one.' — xgboost trả "
    "float32 nên tổng xác suất mỗi hàng lệch ~8.7e-08, dưới ngưỡng 1e-6 của "
    "validate_oof_xgboost() và không ảnh hưởng metric. Cố ý KHÔNG normalize lại.",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    """SHA-256 của một file, đọc theo khối để không nạp cả file vào RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pip_freeze() -> list[str]:
    """`pip freeze` của interpreter đang chạy. Trả về [] nếu pip không gọi được."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _environment_manifest() -> dict:
    """Chụp lại môi trường đang chạy để bên nhận đối chiếu được."""
    import sklearn
    import scipy
    import xgboost

    running_python = platform.python_version()
    conda_env = Path(sys.prefix).name
    return {
        "python": running_python,
        "interpreter": sys.executable,
        "conda_env": conda_env,
        "platform": platform.platform(),
        "key_packages": {
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "matches_authoritative_env": bool(
            conda_env == AUTHORITATIVE_CONDA_ENV and running_python == AUTHORITATIVE_PYTHON
        ),
        "expected_authoritative_env": {
            "conda_env": AUTHORITATIVE_CONDA_ENV,
            "python": AUTHORITATIVE_PYTHON,
        },
        "pip_freeze": _pip_freeze(),
    }


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


def run_pairwise_comparisons(scores: pd.DataFrame) -> pd.DataFrame:
    """Kiểm định ghép cặp + khoảng tin cậy 95% + hiệu chỉnh Bonferroni.

    Với mỗi mô hình so sánh: paired t-test trên chênh lệch log loss theo từng fold,
    CI 95% dựa trên phân phối t, rồi hiệu chỉnh Bonferroni cho họ 5 phép so sánh.
    """
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


def run_assumption_checks(scores: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra giả định chuẩn (Shapiro-Wilk) và phân tích độ nhạy Wilcoxon.

    Shapiro-Wilk trên chênh lệch từng cặp quyết định kiểm định chính: p >= alpha
    dùng paired t-test, ngược lại dùng Wilcoxon signed-rank hai phía. Bảng trả về
    báo cáo cả hai để thấy kết luận nhạy thế nào với lựa chọn kiểm định.
    """
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
    pairwise = run_pairwise_comparisons(scores)
    assumptions = run_assumption_checks(scores)
    pairwise.to_csv(tables / PAIRWISE_TABLE, index=False)
    assumptions.to_csv(tables / ASSUMPTION_TABLE, index=False)
    return pairwise, assumptions


def _build_frozen_model(name: str):
    """Dựng một mô hình theo đúng cấu hình đã freeze trong FROZEN_MODEL_SPECS."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    from . import data

    params = dict(FROZEN_MODEL_SPECS[name][1])
    seed = {"random_state": data.RANDOM_STATE}
    if name == "XGBoost":
        return XGBClassifier(**params, **seed, eval_metric="mlogloss", n_jobs=-1)
    if name == "Random Forest":
        return RandomForestClassifier(**params, **seed, n_jobs=-1)
    if name == "Logistic Regression":
        return LogisticRegression(**params, **seed)
    if name == "Decision Tree":
        return DecisionTreeClassifier(**params, **seed)
    if name == "KNN":
        return KNeighborsClassifier(**params)  # KNN không có yếu tố ngẫu nhiên
    if name == "Naive Bayes":
        return GaussianNB(**params)
    raise ValueError(f"Chưa khai báo cách dựng mô hình {name!r}.")


def measure_runtime(project_root: Path) -> pd.DataFrame:
    """Đo thời gian huấn luyện và suy luận của 6 mô hình đã freeze trên 5 fold dùng chung.

    Ghi ra results/runtime.csv. **Không** ghi đè bất kỳ file điểm nào: scores_track_*.csv
    nằm trong danh sách chốt hash provenance, thêm cột vào đó sẽ làm lệch manifest.

    Mỗi fold vẫn tính lại log loss và đối chiếu với results/scores_all.csv để chứng minh
    lần chạy đo giờ này tái tạo đúng con số đã báo cáo, chứ không phải một lần chạy khác.
    """
    from sklearn.base import clone

    from . import data, metrics

    x, y, _ = data.load_processed()
    fold_series = data.load_folds()
    expected = pd.read_csv(project_root / "results" / "scores_all.csv").set_index(["model", "fold"])

    rows = []
    for name in EXPECTED_MODELS:
        base_model = _build_frozen_model(name)
        for fold, train_index, valid_index in data.iter_folds(fold_series):
            model = clone(base_model)

            start = time.perf_counter()
            model.fit(x.iloc[train_index], y.iloc[train_index])
            fit_seconds = time.perf_counter() - start

            start = time.perf_counter()
            proba = model.predict_proba(x.iloc[valid_index])
            predict_seconds = time.perf_counter() - start

            log_loss_value = metrics.compute_metrics(y.iloc[valid_index], proba)["log_loss"]
            rows.append(
                {
                    "model": name,
                    "fold": int(fold),
                    "n_train": len(train_index),
                    "n_valid": len(valid_index),
                    "fit_seconds": round(fit_seconds, 4),
                    "predict_seconds": round(predict_seconds, 4),
                    "log_loss_khop_scores": bool(
                        np.isclose(log_loss_value, expected.loc[(name, int(fold)), "log_loss"], atol=1e-9)
                    ),
                }
            )

    runtime = pd.DataFrame(rows)
    output = project_root / RUNTIME_TABLE
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime.to_csv(output, index=False)
    return runtime


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


def reproduce_xgboost_oof(
    project_root: Path, tolerance: float = 1e-5, authoritative: bool = False
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Recreate validation-only OOF probabilities for the frozen XGBoost config.

    authoritative=False (mặc định): ghi ra đường dẫn diagnostic như trước, không
    đổi hành vi cũ. authoritative=True: ghi ra results/oof/ kèm manifest provenance
    đầy đủ (pip freeze, get_xgb_params, save_config, hash, feature list, kiểm tra
    leakage, metric từng fold tính lại từ OOF) — đây là nguồn duy nhất cho phân tích
    calibration.

    Mỗi dòng OOF luôn được dự đoán bởi model KHÔNG train trên fold chứa dòng đó:
    data.iter_folds() cấp train_index = (fold != f), và mask `assigned` chứng minh
    mỗi dòng được gán đúng một lần.
    """
    from xgboost import XGBClassifier

    from . import data, metrics

    x, y, _ = data.load_processed()
    fold_series = data.load_folds()
    official_folds = pd.read_csv(project_root / "data" / "interim" / "fold_id.csv")
    if not (len(x) == len(y) == len(fold_series) == len(official_folds) == 9_600):
        raise ValueError("Processed data, labels, and official folds must all contain 9600 aligned rows.")

    all_ids = official_folds["id"].to_numpy()
    probabilities = np.full((len(x), 3), np.nan, dtype=float)
    assigned = np.zeros(len(x), dtype=bool)
    xgb_params_per_fold: dict[str, dict] = {}
    booster_config_per_fold: dict[str, dict] = {}
    leakage_per_fold = []
    for fold, train_index, valid_index in data.iter_folds(fold_series):
        if assigned[valid_index].any():
            raise RuntimeError(f"Fold {fold} attempts to overwrite existing OOF rows.")
        model = XGBClassifier(
            **XGBOOST_FROZEN_PARAMS,
            random_state=data.RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        model.fit(x.iloc[train_index], y.iloc[train_index])
        if not np.array_equal(model.classes_, np.array(data.LABELS)):
            raise RuntimeError(f"Unexpected XGBoost class order: {model.classes_}.")
        probabilities[valid_index] = model.predict_proba(x.iloc[valid_index])
        assigned[valid_index] = True

        # Bằng chứng tường minh cho cam kết "không leakage": id train và id valid
        # của fold này phải rời nhau hoàn toàn.
        train_ids = set(all_ids[train_index].tolist())
        valid_ids = set(all_ids[valid_index].tolist())
        overlap = train_ids & valid_ids
        if overlap:
            raise RuntimeError(
                f"Fold {fold} leakage: {len(overlap)} id nằm ở cả tập train và tập valid."
            )
        leakage_per_fold.append(
            {
                "fold": int(fold),
                "n_train": len(train_ids),
                "n_valid": len(valid_ids),
                "n_train_valid_overlap": 0,
                "trained_without_validation_fold": True,
            }
        )

        # Chụp toàn bộ cấu hình model, kể cả các default ngầm không ghi trong notebook.
        # Ở xgboost 2.1.x, learner.gradient_booster.updater là LIST -> tree param nằm ở
        # learner.gradient_booster.tree_train_param.
        xgb_params_per_fold[f"fold_{int(fold)}"] = model.get_xgb_params()
        booster_config_per_fold[f"fold_{int(fold)}"] = json.loads(
            model.get_booster().save_config()
        )
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
    output = project_root / (AUTHORITATIVE_OOF if authoritative else DIAGNOSTIC_OOF)
    output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output, index=False)
    frozen_config = {
        **XGBOOST_FROZEN_PARAMS,
        "random_state": data.RANDOM_STATE,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
    }
    validation.update(
        {
            "output": output.relative_to(project_root).as_posix(),
            "frozen_config": frozen_config,
            "validation_only_generation": True,
            "fold_source": "data/interim/fold_id.csv",
        }
    )
    if not authoritative:
        _write_json(project_root / DIAGNOSTIC_VALIDATION, validation)

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
    reproduction.to_csv(
        project_root / (AUTHORITATIVE_PER_FOLD if authoritative else DIAGNOSTIC_PER_FOLD),
        index=False,
    )
    gate = {
        "status": "PASS" if gate_pass else "FAIL",
        "tolerance": tolerance,
        "max_absolute_metric_difference": max_abs_difference,
        "folds_passed": int(reproduction["within_tolerance"].sum()),
        "folds_total": len(EXPECTED_FOLDS),
    }
    _write_json(project_root / (AUTHORITATIVE_GATE if authoritative else DIAGNOSTIC_GATE), gate)

    if authoritative:
        feature_list = pd.read_csv(
            project_root / "data" / "processed" / "feature_list.csv"
        )["feature"].tolist()
        ordered_features = list(x.columns)
        distinct_params = {json.dumps(v, sort_keys=True, default=str) for v in xgb_params_per_fold.values()}
        distinct_configs = {json.dumps(v, sort_keys=True) for v in booster_config_per_fold.values()}
        manifest = {
            "artifact": AUTHORITATIVE_OOF,
            "status": "AUTHORITATIVE",
            "approved_for_calibration": gate_pass,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reference_score_file": "results/scores_track_c.csv",
            "track_c_provenance_commit": TRACK_C_PROVENANCE_COMMIT,
            "environment": _environment_manifest(),
            "model": {
                "name": REFERENCE_MODEL,
                "frozen_config": frozen_config,
                "get_xgb_params": xgb_params_per_fold,
                "get_xgb_params_identical_across_folds": len(distinct_params) == 1,
                "booster_save_config": booster_config_per_fold,
                "booster_save_config_identical_across_folds": len(distinct_configs) == 1,
            },
            "data": {
                "n_rows": int(len(x)),
                "n_features": int(x.shape[1]),
                "ordered_features": ordered_features,
                "ordered_features_match_feature_list_csv": ordered_features == feature_list,
                "fold_source": "data/interim/fold_id.csv",
                "class_order": list(data.CLASS_ORDER),
                "label_map": dict(data.LABEL_MAP),
                "random_state": data.RANDOM_STATE,
                "n_folds": data.N_FOLDS,
                "sha256": {
                    relative: _sha256(project_root / relative) for relative in PROVENANCE_INPUTS
                },
            },
            "oof": {
                "schema": list(OOF_COLUMNS),
                "validation": validation,
                "leakage_check": {
                    "per_fold": leakage_per_fold,
                    "every_row_predicted_exactly_once": bool(assigned.all()),
                    "guarantee": (
                        "Mỗi dòng chỉ được dự đoán bởi model train trên (fold != fold của "
                        "dòng đó); id train và id valid rời nhau ở cả 5 fold."
                    ),
                },
            },
            "reproduction": {
                "tolerance": tolerance,
                "gate": gate,
                "per_fold": reproduction.to_dict(orient="records"),
                "metric_source": "src.metrics.compute_metrics (đúng hàm đã sinh scores_track_c.csv)",
            },
            "known_benign_warnings": list(KNOWN_BENIGN_WARNINGS),
        }
        _write_json(project_root / AUTHORITATIVE_MANIFEST, manifest)
    return oof, gate, reproduction


def run_calibration_analysis(project_root: Path, n_bins: int = 10) -> dict:
    """Phân tích calibration của mô hình có log loss thấp nhất (XGBoost).

    Sinh reliability diagram + ECE, chỉ chạy sau khi gate tái tạo đã PASS.

    Chỉ đọc OOF authoritative ở results/oof/ — KHÔNG đọc file
    results/diagnostics/oof_xgboost_reproduction_attempt.csv (bản thử đã bị thay thế,
    sinh ra từ môi trường sai). Xem report/track_c_provenance_audit.md.

    n_bins là số bin chính đưa vào báo cáo; ngoài ra luôn ghi thêm bảng độ nhạy qua
    CALIBRATION_BIN_GRID để cho thấy kết luận không phụ thuộc lựa chọn bin.
    """
    gate_path = project_root / AUTHORITATIVE_GATE
    if not gate_path.exists():
        raise RuntimeError(
            "Phân tích calibration bị chặn: chưa có OOF authoritative. Chạy trước:\n"
            "  python scripts/run_final_analysis.py oof-authoritative --project-root ."
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError(
            "Phân tích calibration bị chặn vì gate tái tạo chưa PASS."
        )
    oof = pd.read_csv(project_root / AUTHORITATIVE_OOF)
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
    pd.DataFrame([metrics_payload]).to_csv(project_root / CALIBRATION_METRICS_TABLE, index=False)

    # Độ nhạy theo số bin: ECE là chỉ số phụ thuộc cách chia bin, nên báo cáo một con
    # số duy nhất là chưa đủ thuyết phục.
    sensitivity = pd.DataFrame(
        [
            {
                "n_bins": bins,
                "top_label_ece": stats.top_label_ece(y, proba, n_bins=bins),
                "macro_classwise_ece": stats.macro_classwise_ece(y, proba, n_bins=bins),
            }
            for bins in CALIBRATION_BIN_GRID
        ]
    )
    sensitivity.to_csv(project_root / CALIBRATION_SENSITIVITY_TABLE, index=False)

    figure, _, top_bins, class_bins = stats.plot_reliability_diagram(y, proba, n_bins=n_bins)
    top_bins.to_csv(project_root / CALIBRATION_TOP_LABEL_BINS_TABLE, index=False)
    class_bins.to_csv(project_root / CALIBRATION_CLASSWISE_BINS_TABLE, index=False)
    figure.savefig(project_root / CALIBRATION_FIGURE, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    metrics_payload["bin_sensitivity"] = sensitivity.to_dict(orient="records")
    _write_json(project_root / CALIBRATION_METRICS_JSON, metrics_payload)
    return metrics_payload
