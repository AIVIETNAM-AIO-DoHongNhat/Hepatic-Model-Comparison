import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, f1_score

from .data import LABELS

# Ba chỉ số dùng chung cho mọi mô hình
METRIC_NAMES = ["log_loss", "accuracy", "macro_f1"]


def compute_metrics(y_true, y_proba, labels=LABELS):
    """Tính 3 chỉ số giống hệt nhau cho mọi mô hình. Trả về dict
    {'log_loss', 'accuracy', 'macro_f1'}.

    y_proba: ma trận (n, 3), cột theo thứ tự [C, CL, D]. Luôn truyền labels=[0,1,2]
    để không lỗi khi một fold thiếu lớp hiếm (CL chỉ ~2.6%).
    """
    y_proba = np.asarray(y_proba)
    y_pred = y_proba.argmax(axis=1)
    return {
        "log_loss": log_loss(y_true, y_proba, labels=labels),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro"),
    }


def score_cv(model_name, fold_ids, y_true_per_fold, y_proba_per_fold):
    """Gom điểm từng fold thành bảng (một hàng mỗi fold).

    Trả về DataFrame các cột [model, fold, log_loss, accuracy, macro_f1] —
    chính là bảng điểm ghép cặp để ghi ra results/scores_*.csv và cho stats.py dùng.
    """
    rows = []
    for fold, y_true, y_proba in zip(fold_ids, y_true_per_fold, y_proba_per_fold):
        metrics = compute_metrics(y_true, y_proba)
        rows.append({"model": model_name, "fold": fold, **metrics})
    return pd.DataFrame(rows)


def summarize_scores(scores):
    """Tổng hợp trung bình và độ lệch chuẩn mỗi chỉ số theo từng mô hình.

    scores: DataFrame trả về từ score_cv. Trả về DataFrame gộp theo 'model'.
    """
    return scores.groupby("model")[METRIC_NAMES].agg(["mean", "std"])
