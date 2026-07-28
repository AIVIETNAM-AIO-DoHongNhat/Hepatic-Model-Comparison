import numpy as np
import pandas as pd

from .data import RAW_DIR, PROJECT_ROOT, CLASS_ORDER

SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_COLS = [f"Status_{c}" for c in CLASS_ORDER]  # ["Status_C", "Status_CL", "Status_D"]


def load_test_ids():
    """Đọc cột 'id' của tập test thô, căn khớp theo vị trí với X_test từ load_processed()."""
    return pd.read_csv(RAW_DIR / "aio26_test.csv")["id"]


def make_submission(test_ids, y_proba, path=None):
    """Tạo DataFrame nộp bài đúng định dạng cuộc thi và lưu ra CSV.

    test_ids: id của từng dòng test, căn khớp theo vị trí với y_proba (dùng load_test_ids()).
    y_proba: ma trận (n, 3), cột theo thứ tự CLASS_ORDER = [C, CL, D].
    path: nơi lưu file; mặc định (None) lưu vào submissions/submission.csv.
    Truyền path=False để chỉ trả về DataFrame, không lưu file.

    Kiểm tra mỗi hàng có tổng xác suất ~1 trước khi lưu, để lỗi bị bắt ở đây thay vì
    khi nộp lên hệ thống chấm điểm.
    """
    test_ids = np.asarray(test_ids)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2 or y_proba.shape[1] != len(SUBMISSION_COLS):
        raise ValueError(
            f"y_proba phải là ma trận 2 chiều có {len(SUBMISSION_COLS)} cột "
            f"(thứ tự {CLASS_ORDER}); nhận được shape={y_proba.shape}"
        )
    if len(test_ids) != len(y_proba):
        raise ValueError(
            f"Số id ({len(test_ids)}) không khớp số dòng xác suất ({len(y_proba)})"
        )
    if len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("Cột id chứa giá trị trùng lặp")
    if not np.isfinite(y_proba).all():
        raise ValueError("y_proba chứa NaN hoặc giá trị vô hạn")
    if np.any(y_proba < 0) or np.any(y_proba > 1):
        raise ValueError("Mọi xác suất phải nằm trong đoạn [0, 1]")

    row_sums = y_proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        bad_idx = np.where(~np.isclose(row_sums, 1.0, atol=1e-3))[0]
        raise ValueError(
            f"Tổng xác suất khác 1 ở {len(bad_idx)} dòng, ví dụ index {bad_idx[:5].tolist()}"
        )

    submission_df = pd.DataFrame({"id": test_ids})
    submission_df[SUBMISSION_COLS] = y_proba

    if path is False:
        return submission_df

    if path is None:
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        path = SUBMISSION_DIR / "submission.csv"
    submission_df.to_csv(path, index=False)
    return submission_df
