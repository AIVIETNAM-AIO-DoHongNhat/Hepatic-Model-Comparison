# Đánh Giá Ý Nghĩa Thống Kê Khi So Sánh Mô Hình Học Máy

<p align="center">
  <img src="figures/overview/header.jpg" width="600" alt="Hepatic Model Comparison">
</p>

> **AIO26 Challenge — Dự đoán kết cục bệnh nhân gan**

## Dự án này làm gì

Bài toán nền: với mỗi bệnh nhân gan, dự đoán xác suất của ba kết cục `C` (còn sống),
`CL` (đã ghép gan), `D` (tử vong). Chỉ số đánh giá là **log loss** đa lớp.

Nhưng câu hỏi nghiên cứu không phải "mô hình nào điểm cao nhất", mà là:

> Khi mô hình A hơn mô hình B vài phần nghìn log loss, đó là **chênh lệch thật** hay chỉ là
> nhiễu từ một cách chia dữ liệu cụ thể?

Để trả lời, 6 mô hình được huấn luyện trên **cùng một bộ 5 fold**, rồi so sánh bằng kiểm định
ghép cặp, khoảng tin cậy và hiệu chỉnh Bonferroni — thay vì chỉ nhìn hai con số trung bình.

## Kết quả

| Mô hình | Log loss ↓ | Accuracy ↑ | Macro-F1 ↑ |
|---|---|---|---|
| **XGBoost** | **0.382 ± 0.007** | **0.850 ± 0.006** | **0.640 ± 0.032** |
| Random Forest | 0.406 ± 0.006 | 0.848 ± 0.007 | 0.569 ± 0.008 |
| Logistic Regression | 0.441 ± 0.009 | 0.833 ± 0.005 | 0.558 ± 0.014 |
| Decision Tree | 0.474 ± 0.011 | 0.830 ± 0.007 | 0.529 ± 0.008 |
| KNN | 0.660 ± 0.055 | 0.814 ± 0.009 | 0.515 ± 0.010 |
| Naive Bayes | 2.211 ± 0.217 | 0.589 ± 0.030 | 0.466 ± 0.019 |

Lấy XGBoost làm mốc so với 5 mô hình còn lại: **4/5 so sánh đạt ngưỡng** sau Bonferroni.
Riêng Logistic Regression **chưa đủ bằng chứng** — không phải vì hai mô hình tương đương, mà
vì primary test phải chuyển sang Wilcoxon, và với `n = 5` thì p-value nhỏ nhất kiểm định này
trả về được đã vượt ngưỡng từ trước khi nhìn dữ liệu.

Báo cáo đầy đủ: [report/Research_Template_Report.pdf](report/Research_Template_Report.pdf) ·
Chi tiết từng ticket: [PLAN.md](PLAN.md)

## Chạy thử

Cần Python 3.9.18 và các CSV cuộc thi đặt trong `data/raw/` (không phân phối kèm repo).

```bash
python -m venv .venv && .venv\Scripts\activate   # Unix: source .venv/bin/activate
pip install -r requirements.txt

python scripts/reproduce_all.py   # chỉ đọc: đối chiếu artifact với bản ghi provenance
jupyter lab                       # rồi chạy notebooks/ theo thứ tự 01 → 08
```

> ⚠️ Muốn tái tạo đúng con số Track C thì phải dùng env authoritative `dynamic`, pin đầy đủ ở
> [requirements.lock.txt](requirements.lock.txt). Chạy sai env khiến metric lệch tới ~1.7e-2.

## Cấu trúc

```
├── data/          raw → interim → processed
├── notebooks/     01_eda → 08_statistics
├── src/           package dùng chung: data, metrics, stats, final_analysis, submission
├── scripts/       CLI: reproduce_all.py, run_final_analysis.py, smoke_notebook.py
├── figures/       eda/, model_track_*/, calibration/, report/
├── results/       điểm theo fold, runtime, và OOF authoritative + manifest
├── tables/        bảng số liệu (.csv kèm bản .tex)
└── report/        báo cáo LaTeX + ghi chú phân tích
```

Điểm mấu chốt của thiết kế: fold được gán **một lần** vào `data/interim/fold_id.csv` và dùng
lại ở mọi nơi, nên mọi mô hình đều được chấm trên đúng cùng những bệnh nhân — điều kiện tiên
quyết để kiểm định ghép cặp có nghĩa.

## Trạng thái

Đã xong toàn bộ: EDA → tiền xử lý → 3 track mô hình → OOF authoritative (gate `PASS` 5/5 fold)
→ kiểm định thống kê → calibration → báo cáo LaTeX. Còn lại: nộp file lên Kaggle.
