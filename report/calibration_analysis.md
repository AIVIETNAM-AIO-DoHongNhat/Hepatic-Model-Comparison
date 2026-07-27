# Phân tích calibration — mô hình có log loss thấp nhất

## Tóm tắt

Mô hình được phân tích: **XGBoost** (`n_estimators=400`, `max_depth=2`, `learning_rate=0.15`),
mean 5-fold log loss `0.381948` — thấp nhất trong 6 mô hình của dự án.

| Chỉ số | Giá trị |
|---|---|
| Multiclass Brier score | **0.216205** |
| Top-label ECE (10 bin đều) | **0.011651** |
| Macro classwise ECE (10 bin đều) | **0.008474** |
| Số mẫu | 9.600 (OOF) |

**Kết luận: mô hình hiệu chỉnh tốt.** Sai lệch giữa confidence và độ chính xác thực tế trung bình
chỉ ~1,2%. Ở vùng confidence cao — nơi chứa 56% dữ liệu — sai lệch gần như bằng 0.

Một phát hiện có hệ thống: **mô hình overconfident ở mọi bin có dữ liệu**, không có bin nào
underconfident. Xem mục [Diễn giải](#diễn-giải).

## Nguồn dữ liệu và cách tái tạo

Xác suất lấy từ `results/oof/oof_xgboost_track_c.csv` — OOF authoritative, mỗi dòng được dự đoán
bởi model **không** train trên fold chứa dòng đó. Gate tái tạo đã `PASS` 5/5 fold (sai lệch tối đa
`5.55e-17` so với `results/scores_track_c.csv`). Chi tiết provenance:
[track_c_provenance_audit.md](track_c_provenance_audit.md).

Hàm `run_calibration_analysis()` raise `RuntimeError` nếu gate chưa `PASS`, nên không thể sinh ra
con số calibration từ xác suất chưa được kiểm chứng.

```
C:\Users\ADMIN\miniconda3\envs\dynamic\python.exe scripts/run_final_analysis.py calibration --project-root .
```

## Kết quả chi tiết

### Top-label reliability (10 bin đều)

| bin | khoảng | n | % dữ liệu | confidence TB | accuracy quan sát | lệch (obs − conf) |
|---|---|---:|---:|---:|---:|---:|
| 0–2 | 0.0–0.3 | 0 | 0% | — | — | — |
| 3 | 0.3–0.4 | 13 | 0.1% | 0.3773 | 0.3077 | −0.0696 |
| 4 | 0.4–0.5 | 136 | 1.4% | 0.4693 | 0.4118 | −0.0576 |
| 5 | 0.5–0.6 | 644 | 6.7% | 0.5484 | 0.5047 | −0.0437 |
| 6 | 0.6–0.7 | 779 | 8.1% | 0.6514 | 0.6187 | −0.0326 |
| 7 | 0.7–0.8 | 934 | 9.7% | 0.7531 | 0.7484 | −0.0047 |
| 8 | 0.8–0.9 | 1.678 | 17.5% | 0.8563 | 0.8296 | −0.0267 |
| 9 | 0.9–1.0 | **5.416** | **56.4%** | 0.9604 | 0.9603 | **−0.0001** |

### ECE theo từng lớp (10 bin đều)

| Lớp | Tỉ lệ trong dữ liệu | Classwise ECE | Bin đông nhất |
|---|---:|---:|---|
| C | 67,9% (6.521) | 0.011932 | `[0.9, 1.0)` — 4.230 mẫu (44,1%) |
| CL | **2,65% (254)** | **0.003835** | `[0.0, 0.1)` — **9.073 mẫu (94,5%)** |
| D | 29,4% (2.825) | 0.009655 | `[0.0, 0.1)` — 4.631 mẫu (48,2%) |

### Độ nhạy theo số bin

ECE là chỉ số phụ thuộc cách chia bin, nên báo cáo một con số duy nhất là chưa đủ:

| n_bins | Top-label ECE | Macro classwise ECE |
|---:|---:|---:|
| 5 | 0.011651 | 0.006569 |
| **10** | **0.011651** | **0.008474** |
| 15 | 0.013832 | 0.009432 |
| 20 | 0.014439 | 0.010603 |

Kết luận "hiệu chỉnh tốt" **không phụ thuộc** lựa chọn bin: ECE dao động 0.0117–0.0144, đều nhỏ.

## Diễn giải

**Mô hình overconfident một cách nhất quán.** Ở cả 7 bin có dữ liệu, accuracy quan sát được luôn
**thấp hơn** confidence — không có bin nào ngược lại. Nghĩa là mô hình hơi tự tin quá mức ở mọi
mức, chứ không phải lúc quá tự tin lúc quá dè dặt.

Đây cũng là lý do toán học khiến top-label ECE ở 5 bin **bằng đúng** ở 10 bin (`0.011651`): khi gộp
các bin có cùng dấu lệch, tổng có trọng số của trị tuyệt đối không đổi. Đây là hệ quả, **không phải
lỗi tính toán** — đã kiểm tra lại dấu của từng bin để xác nhận.

**Mức độ overconfident rất nhỏ ở nơi quan trọng nhất.** Bin `[0.9, 1.0)` chứa 56,4% dữ liệu và chỉ
lệch `0.0001`. Các sai lệch lớn (0.03–0.07) đều nằm ở các bin thưa: bin `[0.3, 0.4)` chỉ có **13
mẫu**. Với 13 mẫu, sai số chuẩn của một tỉ lệ đã vào khoảng ±13%, nên khoảng lệch 7% ở đó **chủ yếu
là nhiễu lấy mẫu**, không phải bằng chứng mô hình lệch. Đó là lý do hình reliability diagram kèm
histogram số mẫu — nếu chỉ nhìn đường cong, người đọc sẽ hiểu sai rằng mô hình tệ ở vùng
confidence thấp.

**Brier score 0.216** cần đọc trong bối cảnh bài toán 3 lớp mất cân bằng, không so trực tiếp được
với bài toán nhị phân.

Hình: `figures/calibration/xgboost_reliability_diagram.png`.

## Hạn chế

Phải nêu kèm khi trích dẫn bất kỳ con số nào ở trên.

1. **Leakage tiền xử lý.** Encoder và scaler được fit **một lần trước khi** chia CV, rồi dùng lại
   cho mọi fold. Thông tin từ fold validation đã rò rỉ vào bước tiền xử lý. Vì vậy calibration đo
   trên OOF này **có thể lạc quan hơn** thực tế. Chi tiết:
   [track_c_provenance_audit.md](track_c_provenance_audit.md).

2. **Không phải nested CV.** Hyperparameter được chọn và mô hình được đánh giá trên **cùng một bộ
   5 fold cố định**. Xem README §7.1.

3. **Lớp CL quá hiếm để kết luận.** CL chỉ có 254/9.600 mẫu (2,65%). Classwise ECE của CL là
   `0.003835` — nhìn thì tốt nhất trong 3 lớp, nhưng **94,5% toàn bộ dữ liệu rơi vào đúng một bin
   `[0.0, 0.1)`**. Con số đó chỉ nói lên "mô hình gán xác suất CL rất thấp cho hầu hết bệnh nhân, và
   điều đó đúng" — nó **không** chứng minh mô hình hiệu chỉnh tốt khi thật sự dự đoán CL. Không
   được dùng con số này để khẳng định mô hình đáng tin với lớp hiếm.

4. **Bin 0.0–0.3 trống là do cấu trúc, không phải thiếu dữ liệu.** Với 3 lớp, xác suất lớn nhất
   luôn ≥ 1/3, nên top-label confidence không bao giờ xuống dưới 0,333.

5. **ECE phụ thuộc cách chia bin.** Con số chính dùng 10 bin đều; xem bảng độ nhạy ở trên.

6. **Đây là calibration trên OOF của tập train, không phải tập test cuộc thi.** Không có kết luận
   nào về calibration trên tập test được rút ra từ đây. Ngoài ra, không có calibrator (Platt/
   isotonic) nào được fit — đây là đo đạc, không phải hiệu chỉnh.

## Artifact

| File | Nội dung |
|---|---|
| `tables/calibration_metrics.csv` | 3 chỉ số chính |
| `tables/calibration_top_label_reliability_bins.csv` | 10 bin top-label (giữ cả bin trống) |
| `tables/calibration_classwise_reliability_bins.csv` | 30 dòng (3 lớp × 10 bin) |
| `tables/calibration_ece_bin_sensitivity.csv` | ECE ở 5/10/15/20 bin |
| `figures/calibration/xgboost_reliability_diagram.png` | Reliability diagram + histogram số mẫu |
| `results/validation/calibration_metrics.json` | toàn bộ chỉ số dạng JSON |
