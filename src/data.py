import pickle

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT/"data"
RAW_DIR     = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROC_DIR    = DATA_DIR / "processed"

RANDOM_STATE = 42
N_FOLDS      = 5
TARGET_COL   = "Status"
CLASS_ORDER  = ["C", "CL", "D"] # thứ tự cột xác suất
LABELS       = [0, 1, 2] # dùng cho log_loss/f1_score
LABEL_MAP    = {"C": 0, "CL": 1, "D": 2}

def load_raw():
    """Đọc dữ liệu thô của cuộc thi. Trả về (train_df, test_df)."""
    train_df = pd.read_csv(RAW_DIR / "aio26_train.csv")
    test_df = pd.read_csv(RAW_DIR / "aio26_test.csv")
    return train_df, test_df


def load_processed():
    """Đọc dữ liệu đã tiền xử lý, sẵn sàng cho mô hình. Trả về (X_train, y_train, X_test).

    y_train là Status đã mã hóa số (0=C, 1=CL, 2=D). Các hàng của X_train,
    y_train và load_folds() căn khớp theo vị trí (cùng thứ tự ở mọi nơi).
    """
    X_train = pd.read_csv(PROC_DIR / "train_X.csv")
    y_train = pd.read_csv(PROC_DIR / "train_y.csv")[TARGET_COL].astype(int)
    X_test = pd.read_csv(PROC_DIR / "test_X.csv")
    return X_train, y_train, X_test


def load_processed_unscaled():
    """Đọc bản KHÔNG scaling của dữ liệu đã tiền xử lý. Trả về (X_train, X_test).

    Giống hệt load_processed() ở mọi bước (impute, log1p, encode, thứ tự cột) trừ
    một điều: bỏ qua RobustScaler. Dùng cho ablation đo ảnh hưởng của scaling lên
    các mô hình dựa trên khoảng cách như KNN. Nhãn y dùng chung với load_processed()
    vì mã hóa nhãn không phụ thuộc scaling.
    """
    X_train = pd.read_csv(PROC_DIR / "train_X_unscaled.csv")
    X_test = pd.read_csv(PROC_DIR / "test_X_unscaled.csv")
    return X_train, X_test


def load_folds():
    """Đọc mã fold dùng chung cho cross-validation (0..N_FOLDS-1) dưới dạng Series.
    Căn khớp theo vị trí với X_train từ load_processed(), nhờ đó mọi mô hình đều
    được chấm trên cùng bộ fold (điều kiện tiên quyết để so sánh thống kê ghép cặp).
    """
    return pd.read_csv(INTERIM_DIR / "fold_id.csv")["fold_id"].astype(int)


def load_feature_list():
    """Trả về danh sách tên 31 cột đặc trưng đã tiền xử lý."""
    return pd.read_csv(PROC_DIR / "feature_list.csv")["feature"].tolist()


def load_transformers():
    """Đọc các bộ tiền xử lý đã fit. Trả về dict với các khóa
    'label_encoder', 'cat_encoder', 'scaler' (dùng để biến đổi dữ liệu test thô).
    """
    names = {"label_encoder": "label_encoder.pkl",
             "cat_encoder": "cat_encoder.pkl",
             "scaler": "scaler.pkl"}
    transformers = {}
    for key, fname in names.items():
        with open(PROC_DIR / fname, "rb") as f:
            transformers[key] = pickle.load(f)
    return transformers


def iter_folds(fold_id):
    """Sinh (fold, train_idx, valid_idx) dựa trên fold_id dùng chung (không chia lại).
    Truyền vào Series từ load_folds(); chỉ số theo vị trí (dùng với .iloc).
    """
    fold_values = np.asarray(fold_id)
    for fold in range(N_FOLDS):
        valid_idx = np.where(fold_values == fold)[0]
        train_idx = np.where(fold_values != fold)[0]
        yield fold, train_idx, valid_idx
