# Sprint 1

| Ticket | Người | Est | Phụ thuộc | Definition of Done | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| **HEP-2** · Chốt schema CSV + cấu trúc repo | Cả nhóm | 0.5d | — | Schema + cấu trúc thư mục commit, cả nhóm đồng ý | ✅ `README.md` §4 |
| **HEP-1** · Load data + báo cáo chất lượng | P1 | 1d | — | Notebook: tỉ lệ NA từng cột, kiểu dữ liệu, phân bố 3 lớp `Status` | ✅ `notebooks/01_eda.ipynb` |
| **HEP-3** · Làm sạch + encode + 2 bản scaling | P1 | 1.5d | HEP-1 | Train sạch lưu ra; bản có/không scaling; log mọi quyết định | ✅ `data/processed/train_X.csv` + `train_X_unscaled.csv`, notebook 02 §2.6 |
| **HEP-4** · Cố định `fold_id` dùng chung | P1 | 0.5d | HEP-3 | `fold_id.csv` commit; mọi người dùng chung | ✅ `data/interim/fold_id.csv`, `data.iter_folds()` |
| **HEP-21** · Khung pipeline Model C | P1 | 1d | HEP-2, HEP-4 | RF + XGBoost chạy end-to-end 1 fold, đúng schema; `xgboost` đã pin | ✅ notebook 06; `requirements.txt` pin `xgboost==2.1.4` |
| **HEP-5** · Khung pipeline Model A | P2 | 1d | HEP-2, HEP-4 | KNN + LogReg xuất `predict_proba`, tính log loss 1 fold | ✅ notebook 04 |
| **HEP-6** · Khung pipeline Model B | P3 | 1d | HEP-2, HEP-4 | Decision Tree + Naive Bayes chạy 1 fold, đúng format | ✅ notebook 05 |
| **HEP-7** · Module thống kê (scaffold) | P4 | 1.5d | HEP-2 | Hàm paired t-test, CI, Wilcoxon, Bonferroni chạy đúng trên dữ liệu giả | ✅ `src/stats.py`, `tests/test_stats.py` (7 test) |
| **HEP-9** · Module calibration (scaffold) | P4 | 0.5d | HEP-2 | Hàm reliability diagram + ECE chạy trên dữ liệu giả | ✅ `src/stats.py`, `tests/test_calibration.py` (5 test) |
| **HEP-25** · Chốt chiến lược so sánh nhiều mô hình | P4 | 0.5d | HEP-2 | Ghi vào README: so 15 cặp hay 5 cặp so với mô hình tốt nhất **khai báo trước khi nhìn kết quả** | ⚠️ `README.md` §7.1 — chiến lược 5 cặp có ghi, nhưng chốt **sau** khi có kết quả. Không sửa hồi tố được; đã khai báo trung thực là post-result |
| **HEP-8** · Pipeline submission + nộp thử | P5 | 1d | HEP-2 | 1 submission hợp lệ (`id,Status_C,Status_CL,Status_D`, tổng = 1) được Kaggle chấp nhận; khung report dựng sẵn | ⚠️ `submissions/submission.csv` hợp lệ và tái tạo được đúng bằng param đã freeze. Chưa có bằng chứng Kaggle chấp nhận; `report/` chưa có `.tex` của nhóm |

# Sprint 2

| Ticket | Người | Est | Phụ thuộc | Definition of Done | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| **HEP-22** · Chạy đủ 5-fold Model C | P1 | 1.5d | HEP-21 | `scores_track_c.csv` đủ **10 dòng** (2 mô hình × 5 fold); ghi runtime từng mô hình | ✅ `results/scores_track_c.csv`, `results/runtime.csv` (30 dòng) |
| **HEP-23** · Siêu tham số + feature importance | P1 | 1d | HEP-22 | Chốt `n_estimators`/`max_depth`/`learning_rate`; biểu đồ feature importance RF & XGB | ✅ notebook 06 §5.3; `figures/model_track_c/` |
| **HEP-10** · Chạy đủ 5-fold Model A | P2 | 1.5d | HEP-5 | File log loss (+acc, F1) từng fold cho KNN & LogReg; ghi lại `n_neighbors` đã chọn | ✅ `results/scores_track_a.csv`; `k=51` chọn ở notebook 04 §3.3 |
| **HEP-12** · Ablation ảnh hưởng scaling (KNN) | P2 | 1d | HEP-3 | So log loss KNN có/không scaling; bảng + nhận xét | ✅ `tables/knn_scaling_ablation.csv`, notebook 04 §3.4 |
| **HEP-11** · Chạy đủ 5-fold Model B | P3 | 1.5d | HEP-6 | Tương tự; xử lý xác suất 0 của Decision Tree, ghi rõ cách xử lý; ghi lại `max_depth` đã chọn | ✅ `results/scores_track_b.csv`; không clip, xử lý bằng `max_depth=3`, giải thích ở notebook 05 §4.11 |
| **HEP-24** · So sánh cây đơn vs cây tổ hợp | P3 | 1d | HEP-11, HEP-22 | Phân tích vì sao xác suất cây đơn overconfident, log loss chênh ra sao | ✅ `tables/tree_vs_ensemble.csv`, notebook 05 §4.11 |
| **HEP-13** · Chạy t-test + CI + hiệu chỉnh | P4 | 2d | HEP-10, HEP-11, HEP-22, HEP-25 | Bảng p-value & CI theo chiến lược đã chốt | ✅ `tables/pairwise_comparisons.csv` |
| **HEP-14** · Kiểm tra giả định + diễn giải | P4 | 1d | HEP-13 | Shapiro–Wilk từng cặp; Wilcoxon khi vi phạm; ghi chú diễn giải | ✅ `tables/assumption_checks.csv`, notebook 08 §7.5 |
| **HEP-15** · Phân tích calibration mô hình tốt nhất | P4 | 1d | HEP-13 | Reliability diagram + ECE cho mô hình log loss thấp nhất | ✅ `tables/calibration_metrics.csv`, `figures/calibration/`, `report/calibration_analysis.md` |
| **HEP-16** · Sản xuất toàn bộ biểu đồ + bảng | P5 | 1.5d | HEP-13 | Box plot log loss, heatmap p-value, bảng tổng hợp; đóng gói đầy đủ vào `figures/` + `tables/`, đặt tên rõ | ✅ 2 hình trong `figures/report/`, `tables/report_*.csv\|.tex`, notebook 07 |
| **HEP-26** · Provenance môi trường + gate tái tạo OOF | P1, P4 | 1.5d | HEP-22 | `requirements.lock.txt` ghi env authoritative; `results/oof/` có OOF + manifest chốt SHA-256 của 8 file đầu vào; gate tái tạo `PASS` 5/5 fold | ✅ `results/oof/*.manifest.json`, `report/track_c_provenance_audit.md` |
| **HEP-27** · Công cụ kiểm tra tái lập | P1 | 1d | HEP-26 | Chạy được mọi notebook không cần Jupyter; một lệnh verify artifact, một lệnh chạy lại toàn pipeline rồi so kết quả | ✅ `scripts/smoke_notebook.py`, `scripts/reproduce_all.py` |
| **HEP-19** · Chạy tái lập toàn bộ | Cả nhóm *(P1)* | 0.5d | Mọi ticket trên | Clone sạch → chạy lại ra đúng số; ghi Python version, seed, library versions | ⚠️ `reproduce_all.py --check` PASS 8/8; `--full` chạy lại notebook 02→06 và tái tạo đúng số trong dung sai `1e-12`. Thao tác clone ra thư mục sạch chưa thực hiện |

# Sprint 3 — Dọn dư thừa

| Ticket | Người | Est | Phụ thuộc | Definition of Done | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| **HEP-28** · Xử lý code chết trong `src/` | P4 | 0.5d | — | Quyết định cho từng mục: `src/submission.py` (không nơi nào gọi — nhưng 3 file trong `submissions/` tái tạo được đúng bằng param đã freeze, nên **nối vào pipeline** hợp lý hơn là xoá), `stats.expected_calibration_error`, `stats.reliability_data`, `metrics.score_cv`, `data.load_raw`, `data.load_transformers`, `data.load_feature_list` | ❌ |
| **HEP-29** · Gộp hằng số khai báo trùng | P4 | 0.5d | — | `CLASS_ORDER` về một nguồn (đang là list ở `data.py`, tuple ở `stats.py`); `LABEL_MAP` không dựng lại inline trong `run_calibration_analysis` | ❌ |
| **HEP-30** · Bỏ nhánh artifact diagnostic song song | P4 | 0.5d | HEP-26 | Bỏ hoặc khoá phase `oof` (không authoritative) — nó sinh lại đúng file đã bị đánh dấu `SUPERSEDED` | ❌ |
| **HEP-31** · Bản sao dữ liệu và `.pkl` baseline | Cả nhóm | 0.5d | — | Quyết định về `dataset/` (byte-identical với `data/raw/`) và `results/models/*.pkl` 25MB (từ commit baseline, **không** phải artifact Track C đã freeze) | ❌ |

# Ghi chú — phần cố ý chưa có ticket

Ghi lại để khoảng trống là có chủ đích, không phải bỏ sót:

- **HEP-25** không sửa hồi tố được. Protocol đã chốt sau khi có kết quả; chạy lại rồi freeze lại
  không khôi phục được tính preregistered vì vấn đề nằm ở thứ tự thông tin. `README.md` §7.1 đã khai
  báo trung thực — đó là cách xử lý đúng.
- **HEP-17 / HEP-8** cần thao tác người: nộp lên Kaggle rồi ghi lại thứ hạng public LB.
- **Báo cáo LaTeX cuối**: `report/` mới có template BTC + 2 file `.md`, chưa có `.tex` của nhóm.
- **Test cho `src/final_analysis.py`**: 12 test hiện chỉ phủ `src/stats.py`.
- **Artifact chưa được commit**: nhiều đường dẫn đang untracked, gồm cả `results/oof/`,
  `requirements.lock.txt`, `scripts/reproduce_all.py`, toàn bộ `tables/` và `figures/report/`.
- **Random Forest lệch 1 ULP giữa các lần chạy** (`n_jobs=-1`, cộng dấu phẩy động không kết hợp).
  Đã ghi trong `report/track_c_provenance_audit.md`; nếu cần byte-identical thì phải đặt `n_jobs=1`,
  đổi lại là chậm hơn nhiều.

# Ticket đã xoá

- **HEP-18** · Tinh chỉnh nhẹ siêu tham số *(optional)* — trùng việc đã nằm trong HEP-10 (chọn `k`
  cho KNN, notebook 04 §3.3) và HEP-11 (chọn `max_depth` cho Tree, notebook 05 §4.3). Yêu cầu
  "ghi lại lựa chọn" đã chuyển vào DoD của hai ticket đó.
- **HEP-20** · Đóng gói figure/table cho tuần 3 — trùng HEP-16, cùng người P5, và trên thực tế cùng
  do `07_report_assets.ipynb` làm trong một lần. Yêu cầu "đầy đủ, đặt tên rõ" đã chuyển vào DoD của
  HEP-16.
