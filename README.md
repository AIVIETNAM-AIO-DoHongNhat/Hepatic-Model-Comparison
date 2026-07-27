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

Các phát hiện chính từ EDA (xem [`figures/eda/`](figures/eda/)):
- `Edema_Status` thiếu ~92%, `Cholesterol`/`Triglyceride` ~55%, và 7 chỉ số xét nghiệm cùng thiếu ~43%.
- Nhiều đặc trưng xét nghiệm lệch phải mạnh → cần biến đổi `log1p`.
- Không có drift phân phối đáng kể giữa train/test → cross-validation tiêu chuẩn là an toàn.

## 4. Cấu trúc dự án

```
Hepatic-Model-Comparison/
├── data/
│   ├── raw/            # CSV gốc của cuộc thi (đang commit trong repo)
│   ├── interim/        # đã làm sạch + impute + cờ missing + fold_id
│   └── processed/      # đã encode + log-transform + scale; X / y / encoders
│                       # kèm bản *_unscaled.csv (bỏ scaling) cho ablation KNN
├── notebooks/
│   ├── 01_eda.ipynb              # phân tích khám phá dữ liệu       ✅
│   ├── 02_preprocesing.ipynb     # pipeline raw → interim → processed ✅
│   ├── 03_baseline_models.ipynb  # baseline nhanh 6 model (1 lần train/val), tham khảo ✅
│   ├── 04_model_track_a.ipynb    # họ mô hình A: KNN + Logistic Regression ✅
│   ├── 05_model_track_b.ipynb    # họ mô hình B: Decision Tree + Naive Bayes ✅
│   ├── 06_model_track_c.ipynb    # họ mô hình C: RF + XGBoost (đã tune & freeze) ✅
│   ├── 07_report_assets.ipynb    # hình/bảng cuối cho báo cáo        ✅
│   └── 08_statistics.ipynb       # kiểm định ý nghĩa + calibration   ✅
├── src/                # module dùng chung (import: from src import data, metrics, stats, submission)
│   ├── __init__.py     # đánh dấu package, re-export hằng số
│   ├── data.py         # loaders + hằng số dùng chung                         ✅
│   ├── metrics.py      # log_loss / accuracy / macro-F1 thống nhất            ✅
│   ├── stats.py        # kiểm định ghép cặp, CI, Bonferroni, calibration      ✅
│   ├── final_analysis.py # quy trình phân tích cuối: validate, gate, calibration ✅
│   └── submission.py   # tạo file nộp bài đúng định dạng cuộc thi             ✅
├── scripts/            # CLI: run_final_analysis.py, smoke_notebook.py, reproduce_all.py
├── tests/              # unittest cho src/stats.py (chạy: python -m unittest discover -s tests -t .)
├── figures/            # hình vẽ, mỗi track/notebook một thư mục con (eda/, model_track_b/, ...)
├── results/            # điểm mô hình theo fold (scores_track_a/b/c.csv, scores_all.csv, runtime.csv)
│   └── oof/            # OOF authoritative của XGBoost final + manifest provenance (nguồn cho calibration) ✅
├── tables/             # bảng số liệu cho báo cáo (xuất từ 07_report_assets.ipynb, kèm bản .tex)
├── submissions/        # file nộp Kaggle (ghi bởi src/submission.py, mặc định submission.csv)
├── report/             # báo cáo LaTeX (template AI CONQUER 2026) + track_c_provenance_audit.md
├── dataset/            # notebook baseline tham khảo của cuộc thi
├── requirements.txt      # dependency (pin 5 package quyết định con số)
├── requirements.lock.txt # pip freeze của env authoritative "dynamic" — bản ghi provenance ✅
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
- Python 3.9.18 (bản của env authoritative `dynamic`; bản khác sẽ trượt gate tái tạo)
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
1. `01_eda.ipynb` - phân tích khám phá, tạo lại `figures/eda/`.
2. `02_preprocesing.ipynb` - dựng `data/interim/` và `data/processed/`.
3. `03_baseline_models.ipynb` - baseline nhanh 6 model (1 lần train/val), tham khảo trước khi vào track chính thức.
4. `04_model_track_a.ipynb` / `05_model_track_b.ipynb` / `06_model_track_c.ipynb` - huấn luyện các họ mô hình, mỗi track tune hyperparameter bằng grid/line search trên fold dùng chung rồi **freeze** trước khi ghi `results/scores_track_*.csv` (track C: RF + XGBoost).
5. `07_report_assets.ipynb` - xuất 4 hình (`figures/report/`) và 3 bảng (`tables/report_*.csv|.tex`) cho báo cáo.
6. `08_statistics.ipynb` - kiểm định ý nghĩa trên `results/scores_all.csv`.

### 6.1. Kiểm tra tính tái lập

```bash
# Chỉ đọc, an toàn chạy bất cứ lúc nào: đối chiếu artifact với bản ghi provenance.
python scripts/reproduce_all.py

# Chạy lại toàn bộ pipeline (notebook 02 -> 06 + mọi phase phân tích) rồi so từng byte
# với artifact đang có. GHI ĐÈ data/processed và results/.
python scripts/reproduce_all.py --full
```

Chế độ `--check` verify: môi trường khớp env authoritative, 8 hash đầu vào khớp manifest, gate tái
tạo `PASS` 5/5 fold, `scores_all.csv` khớp 3 file track, OOF hợp lệ, và các số calibration khớp nhau
giữa JSON và CSV.

Đo thời gian huấn luyện/suy luận của 6 mô hình đã freeze (ghi ra `results/runtime.csv`, **không**
đụng file điểm nào):

```bash
python scripts/run_final_analysis.py runtime --project-root .
```

## 7. Phương pháp luận - so sánh mô hình

Tầng thống kê (notebook 08, `src/stats.py`) là đóng góp nghiên cứu cốt lõi:
- **Chỉ số thống nhất** tính giống hệt nhau cho mọi mô hình (`src/metrics.py`) - log loss, accuracy, macro-F1.
- **Dùng chung fold** để điểm theo fold được ghép cặp giữa các mô hình.
- **Kiểm định ghép cặp** (ví dụ paired *t*-test / Wilcoxon) trên chênh lệch theo từng fold.
- **Khoảng tin cậy** cho chênh lệch hiệu năng trung bình.
- **Hiệu chỉnh so sánh bội** (Bonferroni) khi so nhiều mô hình cùng lúc.
- **Kiểm tra calibration**, vì chỉ số log loss thưởng cho xác suất được hiệu chỉnh tốt.

### 7.1. Protocol amendment cho phân tích cuối

Phân tích cuối sử dụng **5 fold cố định dùng chung**, thay cho yêu cầu 10-fold trong
task ban đầu. Fold chính thức được lấy từ `data/interim/fold_id.csv`; mọi score dùng
cho kiểm định phải được ghép cặp theo cùng `fold_id`.

Protocol so sánh nhiều mô hình đã được leader phê duyệt sau khi có kết quả
(`leader-approved post-result protocol`), do đó **không được mô tả là preregistered**:

- Mô hình tham chiếu: **XGBoost**, có mean final 5-fold log loss thấp nhất (`0.381948`).
- Năm comparator: Random Forest, Logistic Regression, Decision Tree, KNN và Naive Bayes.
- Metric chính: fold-level multiclass log loss; log loss thấp hơn là tốt hơn.
- Chênh lệch ghép cặp: `difference = log_loss_XGBoost - log_loss_comparator`.
  `difference < 0` nghĩa là XGBoost tốt hơn; `difference > 0` nghĩa là comparator tốt hơn.
- Bonferroni family size: `5`; alpha tổng thể: `0.05`; adjusted alpha: `0.01`.
- Adjusted p-value: `min(raw_p * 5, 1.0)`.

Giới hạn phương pháp luận: lựa chọn siêu tham số và đánh giá cuối sử dụng cùng bộ
fixed folds, không phải nested cross-validation. Vì vậy score và p-value có thể lạc
quan, và mọi kết luận suy diễn phải được diễn giải thận trọng.

**Trạng thái protocol:** `COMPLETE VIA LEADER-APPROVED POST-RESULT PROTOCOL AMENDMENT`.
Task lịch sử yêu cầu 10-fold và lựa chọn protocol trước khi xem kết quả; sau khi có
preliminary results, leader đã sửa protocol cuối thành shared 5-fold và năm phép so
sánh lấy XGBoost làm reference. Amendment này được phê duyệt nhưng không hồi tố thành
preregistration; source task được giữ nguyên để bảo toàn lịch sử yêu cầu.

### 7.2. Kết luận thống kê theo primary-test protocol

Kết luận cuối được điều khiển bởi **primary test** (Shapiro-Wilk chọn giữa paired
*t*-test và Wilcoxon), không chỉ bởi paired *t*-test đơn thuần:

- Có bằng chứng XGBoost có fold-level log loss thấp hơn Random Forest, Decision Tree,
  KNN và Naive Bayes sau Bonferroni correction.
- Với Logistic Regression, Shapiro–Wilk cho `p ≈ 0.01423`, nên primary test là
  two-sided exact Wilcoxon. Raw `p = 0.0625`, Bonferroni-adjusted `p = 0.3125`:
  chưa đủ bằng chứng kết luận XGBoost tốt hơn Logistic Regression theo protocol,
  đồng thời không được diễn giải là hai mô hình tương đương.

Paired *t*-test và t-based CI có thể khác Wilcoxon vì dựa trên giả định và độ nhạy
khác nhau. Chỉ có năm paired fold observations; Shapiro–Wilk có power rất thấp, còn
two-sided exact Wilcoxon với `n=5` có độ phân giải p-value thô: ngay cả khi cả năm
differences cùng chiều, p-value nhỏ nhất vẫn là `0.0625`. Vì adjusted alpha là `0.01`,
Wilcoxon không đủ statistical resolution để đạt significance trong family này.
Ngoài ra, training samples giữa các folds overlap nên fold-level inference cần được
diễn giải thận trọng.

## 8. Sử dụng module dùng chung (`src/`)

Bốn module trong `src/` được viết dưới dạng **package** để các notebook mô hình (`04`, `05`,
`06`) và notebook thống kê (`08`) dùng chung một cách load, một bộ chỉ số, một bộ fold và một
cách nộp bài — bảo đảm mọi mô hình được so sánh công bằng.

> **Import & môi trường:** luôn import bằng `from src import data, metrics, stats, submission`
> (chạy từ thư mục gốc dự án hoặc thêm gốc vào `sys.path`). Cần môi trường có `scikit-learn` để
> đọc các encoder `.pkl` — chạy notebook bằng đúng env đã cài `requirements.txt`.
>
> ⚠️ **Tái tạo con số Track C thì PHẢI dùng env authoritative `dynamic`** (Python 3.9.18,
> xgboost 2.1.4, scikit-learn 1.6.1, numpy 1.26.4, pandas 2.3.3, scipy 1.13.1), đã pin ở
> [`requirements.lock.txt`](requirements.lock.txt). `requirements.txt` chỉ là lower bound, không
> phải lock file. Chạy sai env khiến metric lệch tới ~1.7e-2 — đúng cái đã làm phân tích calibration
> bị chẩn đoán sai là "mất provenance" (xem [report/track_c_provenance_audit.md](report/track_c_provenance_audit.md)).

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

- `paired_ttest`, `wilcoxon_test`, `shapiro_test` — structured results cho kiểm định ghép cặp/giả định
- `ci_diff_t`, `ci_diff_bootstrap` — structured confidence intervals cho chênh lệch trung bình
- `bonferroni_correction(pvalues, alpha)` — structured Bonferroni result gồm family size và adjusted alpha
- `brier_multiclass`, `top_label_ece`, `macro_classwise_ece` — calibration metrics đa lớp
- `top_label_reliability_bins`, `classwise_reliability_bins`, `plot_reliability_diagram` — dữ liệu bin và biểu đồ calibration

**`submission.py` — tạo file nộp bài thống nhất** (đúng định dạng `aio26_sample-submission.csv`)

- `load_test_ids()` — cột `id` của test thô, căn khớp theo vị trí với `X_test` từ `load_processed()`
- `make_submission(test_ids, y_proba, path=None)` — kiểm tra `y_proba` đủ 3 cột đúng thứ tự
  `CLASS_ORDER` và mỗi hàng tổng xác suất ≈ 1, rồi ghi CSV `id, Status_C, Status_CL, Status_D`
  (mặc định `submissions/submission.csv`; `path=False` để chỉ lấy DataFrame, không ghi file)

**Ví dụ trong notebook mô hình (04/05/06):**
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

**Ví dụ trong notebook thống kê (08):**
```python
from src import stats

a = scores_track_a["log_loss"].values   # đã ghép cặp theo fold
b = scores_track_b["log_loss"].values
t_result = stats.paired_ttest(a, b)
ci = stats.ci_diff_t(a, b)
print(t_result.statistic, t_result.p_value, t_result.mean_difference)
print(ci.low, ci.high)                  # không chứa 0 => có bằng chứng về chênh lệch
correction = stats.bonferroni_correction([p1, p2, p3])
print(correction.reject, correction.adjusted_p_values)
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
- [x] Mô hình track A (KNN + Logistic Regression, 5-fold CV)
- [x] Mô hình track B (Decision Tree + Naive Bayes, tune `max_depth`)
- [x] Mô hình track C (Random Forest + XGBoost, tune & freeze hyperparameter)
- [x] OOF authoritative của XGBoost final + manifest provenance (gate tái tạo PASS 5/5 fold)
- [x] Phân tích ý nghĩa thống kê (kiểm định ghép cặp, kiểm tra giả định, Bonferroni)
- [x] Phân tích calibration mô hình tốt nhất — reliability diagram + ECE
- [x] Hình/bảng báo cáo (`07_report_assets.ipynb`)
- [ ] Báo cáo LaTeX cuối
- [ ] File nộp Kaggle

### P4 — OOF authoritative & phân tích calibration

`results/oof/oof_xgboost_track_c.csv` là **nguồn duy nhất** cho phân tích calibration. Sinh lại bằng
đúng interpreter của env `dynamic`:

```bash
C:\Users\ADMIN\miniconda3\envs\dynamic\python.exe scripts/run_final_analysis.py oof-authoritative --project-root .
```

Kết quả: gate `PASS`, 5/5 fold, sai lệch metric tối đa `5.55e-17` so với `results/scores_track_c.csv`.
Kèm `oof_xgboost_track_c.manifest.json` chứa `pip freeze`, `get_xgb_params()`,
`get_booster().save_config()` từng fold, SHA-256 của 8 file đầu vào, ordered feature list, kiểm tra
chống leakage và metric từng fold tính lại từ OOF.

Chạy phân tích calibration (chỉ chạy được sau khi gate PASS, nếu không sẽ raise `RuntimeError`):

```bash
C:\Users\ADMIN\miniconda3\envs\dynamic\python.exe scripts/run_final_analysis.py calibration --project-root .
```

Kết quả cho XGBoost (mô hình log loss thấp nhất): Brier `0.216205`, top-label ECE `0.011651`,
macro classwise ECE `0.008474` — mô hình hiệu chỉnh tốt. Diễn giải đầy đủ kèm **6 hạn chế phương
pháp luận** (leakage tiền xử lý, lớp CL quá hiếm, độ nhạy số bin, …) nằm ở
[report/calibration_analysis.md](report/calibration_analysis.md); hình ở
`figures/calibration/xgboost_reliability_diagram.png`.

> `results/diagnostics/oof_xgboost_reproduction_attempt.csv` là bản thử **đã bị thay thế**
> (`SUPERSEDED`), sinh ra từ môi trường sai. Giữ lại làm bằng chứng, **không dùng cho số nào trong
> báo cáo**.
