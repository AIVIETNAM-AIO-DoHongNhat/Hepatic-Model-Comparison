# Đánh Giá Ý Nghĩa Thống Kê Khi So Sánh Mô Hình Học Máy

> **Dự đoán Kết cục Bệnh nhân Gan AIO26 Challenge - Hepatic Model Comparision** 

## 1. Tổng quan

Bài toán dự đoán là ước lượng,
với mỗi bệnh nhân gan, xác suất của ba kết cục có thể xảy ra:

| Mã | Ý nghĩa |
|------|----------------------|
| `C`  | Còn sống (censored)  |
| `CL` | Censored - đã ghép gan |
| `D`  | Tử vong              |

Chủ đề nghiên cứu của dự án là **phương pháp luận so sánh mô hình**:
sau khi huấn luyện nhiều họ mô hình, ta đánh giá xem chênh lệch hiệu năng quan sát được có
*thực sự có ý nghĩa thống kê* (kiểm định ghép cặp, khoảng tin cậy, hiệu chỉnh so sánh bội) hay
chỉ là nhiễu từ một cách chia dữ liệu cụ thể.

## 2. Phát biểu bài toán

- **Loại:** phân loại đa lớp (3 lớp), đầu ra là xác suất.
- **Biến mục tiêu:** `Status ∈ {C, CL, D}`.
- **Định dạng nộp bài:** xác suất theo từng bệnh nhân `id, Status_C, Status_CL, Status_D`.
- **Chỉ số đánh giá:** **log loss** đa lớp (cross-entropy).
- **Thách thức chính:** mất cân bằng lớp nghiêm trọng (`C` ≈ 67.9%, `D` ≈ 29.4%, `CL` ≈ 2.6%)
  và thiếu dữ liệu nặng, không ngẫu nhiên ở các chỉ số xét nghiệm.

## 3. Bộ dữ liệu

Do ban tổ chức cung cấp (không đưa vào quản lý phiên bản - xem [`.gitignore`](.gitignore)).

| File | Kích thước | Ghi chú |
|------|-------|-------|
| `aio26_train.csv` | 12,000 × 21 | 18 đặc trưng + `id` + `Status` |
| `aio26_test.csv`  | 10,000 × 20 | đặc trưng + `id` (không nhãn) |
| `aio26_sample-submission.csv` | 10,000 × 4 | định dạng đầu ra mong đợi |

**Nhóm đặc trưng:** nhân khẩu học (`Patient_Age_Days`, `Patient_Sex`), theo dõi
(`Followup_Days`, `Treatment_Assignment`, `Clinical_Stage`), dấu hiệu lâm sàng nhị phân
(`Ascites_Indicator`, `Liver_Enlargement`, `Spider_Angioma`, `Edema_Status`), và các chỉ số
xét nghiệm (`Bilirubin_Level`, `Albumin_Level`, `Copper_Level`, `AST_Level`, `Platelet_Count`,
`Prothrombin_Time`, …).

Các phát hiện chính từ EDA (xem [`eda_figures/`](eda_figures/)):
- `Edema_Status` thiếu ~92%, `Cholesterol`/`Triglyceride` ~55%, và 7 chỉ số xét nghiệm cùng thiếu ~43%.
- Nhiều đặc trưng xét nghiệm lệch phải mạnh → cần biến đổi `log1p`.
- Không có drift phân phối đáng kể giữa train/test → cross-validation tiêu chuẩn là an toàn.

## 4. Cấu trúc dự án

```
Hepatic-Model-Comparison/
├── data/
│   ├── raw/            # CSV gốc của cuộc thi (git-ignored)
│   ├── interim/        # đã làm sạch + impute + cờ missing + fold_id
│   └── processed/      # đã encode + log-transform + scale; X / y / encoders
├── notebooks/
│   ├── 01_eda.ipynb            # phân tích khám phá dữ liệu       ✅
│   ├── 02_preprocesing.ipynb   # pipeline raw → interim → processed ✅
│   ├── 03_model_track_a.ipynb  # họ mô hình A (dự kiến)            ⬜
│   ├── 04_model_track_b.ipynb  # họ mô hình B (dự kiến)            ⬜
│   ├── 05_model_track_c.ipynb  # họ mô hình C: RF + XGBoost (dự kiến) ⬜
│   ├── 06_statistics.ipynb     # kiểm định ý nghĩa (dự kiến)       ⬜
│   └── 07_report_assets.ipynb  # hình/bảng cho báo cáo (dự kiến)   ⬜
├── src/                # module dùng chung (import: from src import data, metrics, stats, submission)
│   ├── __init__.py     # đánh dấu package, re-export hằng số
│   ├── data.py         # loaders + hằng số dùng chung                         ✅
│   ├── metrics.py      # log_loss / accuracy / macro-F1 thống nhất            ✅
│   ├── stats.py        # kiểm định ghép cặp, CI, Bonferroni, calibration      ✅
│   └── submission.py   # tạo file nộp bài đúng định dạng cuộc thi             ✅
├── eda_figures/        # hình EDA (theme sáng cho nghiên cứu)
├── results/            # điểm mô hình theo fold (scores_track_a/b/c.csv, scores_all.csv)
├── tables/             # bảng số liệu cho báo cáo (xuất từ 07_report_assets.ipynb)
├── submissions/        # file nộp Kaggle (ghi bởi src/submission.py, mặc định submission.csv)
├── report/             # báo cáo LaTeX (template AI CONQUER 2026)
├── dataset/            # notebook baseline tham khảo của cuộc thi
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Pipeline dữ liệu

Tiền xử lý theo thiết kế ba tầng `raw → interim → processed` để mọi bước đều kiểm tra được và
tái lập được:

| Tầng | Vị trí | Nội dung |
|-------|----------|----------|
| **Raw** | `data/raw/` | CSV gốc, giữ nguyên |
| **Interim** | `data/interim/` | Làm sạch + impute median + cờ missing nhị phân + fold ID |
| **Processed** | `data/processed/` | Đặc trưng đã encode + `log1p` + scale, tách thành `X` / `y` / meta |

Điểm nổi bật:
- `Patient_Age_Years = Patient_Age_Days / 365.25`.
- Biến trạng thái thiếu dữ liệu thành tín hiệu qua các cột cờ `*_flag` trước khi impute.
- `log1p` cho đặc trưng lệch phải (skew > 1.5); `OrdinalEncoder` cho biến phân loại; scale cho biến số.
- **Stratified K-Fold** ID được gán một lần (`fold_id.csv`) và dùng lại ở mọi nơi, để mọi mô hình
  được đánh giá trên cùng bộ fold - điều kiện tiên quyết cho so sánh thống kê ghép cặp công bằng.

## 6. Bắt đầu

### Yêu cầu
- Python 3.9+
- Các file CSV của cuộc thi đặt trong `data/raw/` (không phân phối kèm repo này).

### Cài đặt
```bash
git clone <đường-dẫn-repo-của-bạn>
cd Hepatic-Model-Comparison

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r requirements.txt
```

### Chạy pipeline
Chạy các notebook theo thứ tự từ thư mục `notebooks/` (đường dẫn là tương đối so với thư mục này):
```bash
jupyter lab
```
1. `01_eda.ipynb` - phân tích khám phá, tạo lại `eda_figures/`.
2. `02_preprocesing.ipynb` - dựng `data/interim/` và `data/processed/`.
3. `03_model_track_a.ipynb` / `04_model_track_b.ipynb` / `05_model_track_c.ipynb` - huấn luyện các họ mô hình (track C: Random Forest + XGBoost), ghi `results/scores_track_*.csv`.
4. `06_statistics.ipynb` - kiểm định ý nghĩa trên `results/scores_all.csv`.
5. `07_report_assets.ipynb` - xuất hình/bảng cuối cho báo cáo.

## 7. Phương pháp luận - so sánh mô hình

Tầng thống kê (notebook 06, `src/stats.py`) là đóng góp nghiên cứu cốt lõi:
- **Chỉ số thống nhất** tính giống hệt nhau cho mọi mô hình (`src/metrics.py`) - log loss, accuracy, macro-F1.
- **Dùng chung fold** để điểm theo fold được ghép cặp giữa các mô hình.
- **Kiểm định ghép cặp** (ví dụ paired *t*-test / Wilcoxon) trên chênh lệch theo từng fold.
- **Khoảng tin cậy** cho chênh lệch hiệu năng trung bình.
- **Hiệu chỉnh so sánh bội** (Bonferroni) khi so nhiều mô hình cùng lúc.
- **Kiểm tra calibration**, vì chỉ số log loss thưởng cho xác suất được hiệu chỉnh tốt.

## 8. Sử dụng module dùng chung (`src/`)

Bốn module trong `src/` được viết dưới dạng **package** để notebook `03`–`06` dùng chung
một cách load, một bộ chỉ số, một bộ fold và một cách nộp bài — bảo đảm mọi mô hình được
so sánh công bằng.

> **Import & môi trường:** luôn import bằng `from src import data, metrics, stats, submission`
> (chạy từ thư mục gốc dự án hoặc thêm gốc vào `sys.path`). Cần môi trường có `scikit-learn` để
> đọc các encoder `.pkl` — chạy notebook bằng đúng env đã cài `requirements.txt`.

**`data.py` — nạp dữ liệu + hằng số dùng chung**

| Hàm | Trả về |
|-----|--------|
| `load_processed()` | `(X_train, y_train, X_test)` — đặc trưng đã sẵn sàng, `y` mã hóa `{C:0, CL:1, D:2}` |
| `load_folds()` | Series `fold_id` (0–4), căn khớp theo vị trí với `X_train` |
| `iter_folds(fold_id)` | sinh `(fold, train_idx, valid_idx)` — chỉ số dùng với `.iloc` |
| `load_feature_list()` | danh sách 31 tên cột đặc trưng |
| `load_transformers()` | dict `{label_encoder, cat_encoder, scaler}` đã fit |
| `load_raw()` | `(train_df, test_df)` CSV gốc |

Hằng số: `RANDOM_STATE=42`, `N_FOLDS=5`, `LABELS=[0,1,2]`, `LABEL_MAP`, `CLASS_ORDER`.

**`metrics.py` — chỉ số thống nhất** (luôn dùng `labels=[0,1,2]` để không lỗi khi fold thiếu lớp hiếm `CL`)

- `compute_metrics(y_true, y_proba)` → `{'log_loss', 'accuracy', 'macro_f1'}`
- `score_cv(model_name, fold_ids, y_true_per_fold, y_proba_per_fold)` → DataFrame điểm theo fold
- `summarize_scores(scores)` → trung bình ± độ lệch chuẩn mỗi chỉ số theo mô hình

**`stats.py` — tầng thống kê** (điểm theo fold đã ghép cặp; với log loss thì nhỏ hơn = tốt hơn)

- `paired_ttest`, `wilcoxon_test` — kiểm định ghép cặp trên chênh lệch theo fold
- `ci_diff_t`, `ci_diff_bootstrap` — khoảng tin cậy cho chênh lệch trung bình (tham số & bootstrap)
- `bonferroni_correction(pvalues, alpha)` — hiệu chỉnh so sánh bội
- `brier_multiclass`, `expected_calibration_error`, `reliability_data` — kiểm tra calibration

**`submission.py` — tạo file nộp bài thống nhất** (đúng định dạng `aio26_sample-submission.csv`)

- `load_test_ids()` — cột `id` của test thô, căn khớp theo vị trí với `X_test` từ `load_processed()`
- `make_submission(test_ids, y_proba, path=None)` — kiểm tra `y_proba` đủ 3 cột đúng thứ tự
  `CLASS_ORDER` và mỗi hàng tổng xác suất ≈ 1, rồi ghi CSV `id, Status_C, Status_CL, Status_D`
  (mặc định `submissions/submission.csv`; `path=False` để chỉ lấy DataFrame, không ghi file)

**Ví dụ trong notebook mô hình (03/04/05):**
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src import data, metrics

X, y, _ = data.load_processed()
fold = data.load_folds()

rows = []
for f, tr, va in data.iter_folds(fold):
    model = LogisticRegression(max_iter=5000).fit(X.iloc[tr], y.iloc[tr])
    proba = model.predict_proba(X.iloc[va])
    rows.append({"model": "track_a", "fold": f, **metrics.compute_metrics(y.iloc[va], proba)})

scores = pd.DataFrame(rows)
scores.to_csv("results/scores_track_a.csv", index=False)
```

**Ví dụ trong notebook thống kê (06):**
```python
from src import stats

a = scores_track_a["log_loss"].values   # đã ghép cặp theo fold
b = scores_track_b["log_loss"].values
print(stats.paired_ttest(a, b))         # (t, p)
print(stats.ci_diff_t(a, b))            # (mean_diff, low, high); không chứa 0 => có ý nghĩa
reject, p_adj = stats.bonferroni_correction([p1, p2, p3])
```

**Ví dụ tạo file nộp bài:**
```python
from src import data, submission

X_train, y_train, X_test = data.load_processed()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)  # cột theo thứ tự CLASS_ORDER = [C, CL, D]

submission.make_submission(submission.load_test_ids(), y_proba)  # -> submissions/submission.csv
```

## 9. Trạng thái & lộ trình

- [x] Nạp và mô tả bộ dữ liệu
- [x] Phân tích khám phá dữ liệu (5 hình)
- [x] Pipeline tiền xử lý `raw → interim → processed`
- [x] Hàm dùng chung trong `src/` (`data.py`, `metrics.py`, `stats.py`, `submission.py`)
- [ ] Mô hình track A
- [ ] Mô hình track B
- [ ] Mô hình track C (Random Forest + XGBoost)
- [ ] Phân tích ý nghĩa thống kê
- [ ] Hình/bảng báo cáo + báo cáo LaTeX cuối
- [ ] File nộp Kaggle
