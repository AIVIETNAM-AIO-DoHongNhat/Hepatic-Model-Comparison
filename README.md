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
│   ├── 05_statistics.ipynb     # kiểm định ý nghĩa (dự kiến)       ⬜
│   └── 06_report_assets.ipynb  # hình/bảng cho báo cáo (dự kiến)   ⬜
├── src/
│   ├── data.py         # hàm load + làm sạch dùng chung  (cần viết)
│   ├── metrics.py      # log_loss / accuracy / f1 thống nhất (cần viết)
│   └── stats.py        # kiểm định ghép cặp, CI, Bonferroni, calibration (cần viết)
├── eda_figures/        # hình EDA (theme sáng cho nghiên cứu)
├── results/            # điểm mô hình theo fold (scores_track_a/b.csv, scores_all.csv)
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
3. `03_model_track_a.ipynb` / `04_model_track_b.ipynb` - huấn luyện các họ mô hình, ghi `results/scores_track_*.csv`.
4. `05_statistics.ipynb` - kiểm định ý nghĩa trên `results/scores_all.csv`.
5. `06_report_assets.ipynb` - xuất hình/bảng cuối cho báo cáo.

## 7. Phương pháp luận - so sánh mô hình

Tầng thống kê (notebook 05, `src/stats.py`) là đóng góp nghiên cứu cốt lõi:
- **Chỉ số thống nhất** tính giống hệt nhau cho mọi mô hình (`src/metrics.py`) - log loss, accuracy, macro-F1.
- **Dùng chung fold** để điểm theo fold được ghép cặp giữa các mô hình.
- **Kiểm định ghép cặp** (ví dụ paired *t*-test / Wilcoxon) trên chênh lệch theo từng fold.
- **Khoảng tin cậy** cho chênh lệch hiệu năng trung bình.
- **Hiệu chỉnh so sánh bội** (Bonferroni) khi so nhiều mô hình cùng lúc.
- **Kiểm tra calibration**, vì chỉ số log loss thưởng cho xác suất được hiệu chỉnh tốt.

## 8. Trạng thái & lộ trình

- [x] Nạp và mô tả bộ dữ liệu
- [x] Phân tích khám phá dữ liệu (5 hình)
- [x] Pipeline tiền xử lý `raw → interim → processed`
- [ ] Hàm dùng chung trong `src/` (`data.py`, `metrics.py`, `stats.py`)
- [ ] Mô hình track A
- [ ] Mô hình track B
- [ ] Phân tích ý nghĩa thống kê
- [ ] Hình/bảng báo cáo + báo cáo LaTeX cuối
- [ ] File nộp Kaggle

## 9. Công nghệ sử dụng

`pandas` · `numpy` · `scikit-learn` · `scipy` · `matplotlib` · `seaborn`
(xem [`requirements.txt`](requirements.txt) để biết phiên bản tối thiểu đã ghim).

## 10. Lời cảm ơn

Thực hiện cho **AI CONQUER 2026 (AIO26)**, AI VIET NAM. Bộ dữ liệu © ban tổ chức cuộc thi;
sử dụng theo điều khoản của cuộc thi và không phân phối lại.
