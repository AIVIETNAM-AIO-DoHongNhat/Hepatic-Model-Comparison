# Track C provenance audit — nguồn dữ liệu cho phân tích calibration

> **Bản cập nhật 2026-07-26 — kết luận cũ đã bị phản chứng.** Bản audit đầu tiên kết luận
> "Route 3 — không thể phục hồi provenance" và đề xuất đi xin OOF từ chủ Track C.
> Kết luận đó **SAI**. Nguyên nhân thật là **chạy sai môi trường Python**, và môi trường
> authoritative vẫn còn trên máy. Xem mục [Verdict](#verdict) và
> [Diagnostic đã bị thay thế](#diagnostic-đã-bị-thay-thế-chạy-sai-môi-trường).

## Verdict

`results/scores_track_c.csv` có provenance Git/notebook mạnh, **và môi trường thực thi đã được
phục hồi thành công**. Môi trường authoritative là conda env **`dynamic`** — chính là kernel mà
`notebooks/06_model_track_c.ipynb` đã ghi lại, với `language_info.version` = **3.9.18**.

Chạy lại `reproduce_xgboost_oof()` trong env đó tái tạo `results/scores_track_c.csv` **khớp
tuyệt đối**: sai lệch metric lớn nhất **5.551115123125783e-17**, pass **5/5 fold** ở ngưỡng `1e-5`.

Phân tích calibration **đã được mở khóa**. Artifact authoritative:

| File | Nội dung |
|---|---|
| `results/oof/oof_xgboost_track_c.csv` | OOF 9.600 dòng, schema `id,fold,model,y_true,prob_C,prob_CL,prob_D` |
| `results/oof/oof_xgboost_track_c.manifest.json` | manifest provenance đầy đủ (xem dưới) |
| `results/oof/oof_xgboost_track_c.gate.json` | gate tái tạo, `status: PASS`, 5/5 fold |
| `results/oof/oof_xgboost_track_c.per_fold_metrics.csv` | metric từng fold, expected vs actual |
| `requirements.lock.txt` | `pip freeze` của env authoritative (160 package) |

Lệnh tái tạo (**bắt buộc** dùng interpreter của env `dynamic`):

```
C:\Users\ADMIN\miniconda3\envs\dynamic\python.exe scripts/run_final_analysis.py oof-authoritative --project-root .
```

Phase này in cảnh báo nếu bị chạy sai env, và `run_calibration_analysis()` vẫn chặn cứng bằng
`RuntimeError` cho tới khi gate `PASS`.

> Ghi chú: lúc audit được thực hiện, script mang tên `scripts/run_p4_analysis.py`; sau đó đổi tên
> thành `run_final_analysis.py` (nội dung không đổi). Lệnh ở trên đã cập nhật theo tên mới.

## Vì sao đây không phải "chọn env hậu kiểm theo target score"

Bản audit đầu đã cảnh báo rất đúng: không được cài lần lượt từng version XGBoost cho tới khi có
version khớp số, vì như vậy là chọn môi trường hậu kiểm theo target. Quy trình lần này **không**
làm thế:

1. Env được xác định **từ metadata của notebook** (`kernelspec.display_name` = `dynamic`,
   `language_info.version` = `3.9.18`), **trước khi** so sánh bất kỳ metric nào.
2. Env `dynamic` trên máy đúng là Python **3.9.18** — trùng khớp độc lập.
3. Chỉ có **một** env được thử. Không có vòng lặp dò version.

Bằng chứng củng cố: version trong env khớp bound của `requirements.txt`, cho thấy
`requirements.txt` được sinh ra từ chính env này.

| Package | Bound trong requirements.txt | Trong env `dynamic` | Khớp |
|---|---|---|---|
| numpy | `>=1.26.4` | 1.26.4 | bằng đúng bound |
| pandas | `>=2.3.3` | 2.3.3 | bằng đúng bound |
| scipy | `>=1.13.1` | 1.13.1 | bằng đúng bound |
| scikit-learn | `>=1.6.1` | 1.6.1 | bằng đúng bound |
| Jinja2 | `>=3.1.6` | 3.1.6 | bằng đúng bound |
| xgboost | `>=2.0.0` | 2.1.4 | thỏa |
| matplotlib | `>=3.9.4` | **3.9.2** | **thấp hơn bound** |

Ghi nhận trung thực: matplotlib **không** khớp (3.9.2 < 3.9.4), nên `requirements.txt` không phải
bản chụp hoàn hảo của env. matplotlib chỉ dùng vẽ hình, không tham gia tính metric của model,
nên không ảnh hưởng kết luận. 6/7 dependency còn lại khớp, cộng với Python 3.9.18 trùng khớp
chính xác, là bằng chứng đủ mạnh.

## Git provenance

- Commit authoritative của score/notebook: `aa1ce1a7a4f93a6d5cd75533d267efb791375603`, authored 2026-07-24 22:35 +07:00.
- Commit subject: `feat(track-c): tune & freeze RF + XGBoost hyperparameters (HEP-23)`.
- Commit đó đổi notebook Track C, file score CSV, README, hình tuning và hình model; **không** đổi `data/` hay `src/`.
- Score Track C, notebook, processed features, labels và fold artifact hiện tại có cùng Git blob với commit `aa1ce1a`.
- `results/models/Random_Forest.pkl` và `results/models/XGBOOST.pkl` đến từ commit baseline `bde56f3` trước đó; **không** phải artifact Track C đã freeze.

## Notebook execution evidence

Execution count đã lưu tăng đơn điệu từ 1 đến 13. Các cell liên quan chạy theo thứ tự:

1. Nạp processed data/folds: execution 3.
2. Định nghĩa grid search: execution 4.
3. Tuning RF: execution 5.
4. Tuning XGBoost: execution 6.
5. Khai báo model final: execution 7.
6. Chạy CV final: execution 8.
7. Ghi `scores_track_c.csv`: execution 9.

Output CV final đã lưu khớp với score CSV đã commit tới độ chính xác hiển thị. Đây là bằng chứng
mạnh cho việc notebook không bị chạy lệch thứ tự.

## XGBoost parameter provenance

Bản audit đầu liệt kê nhiều param là "Not explicit". **Toàn bộ số đó giờ đã được chụp lại** vào
`results/oof/oof_xgboost_track_c.manifest.json` qua `model.get_xgb_params()` và
`model.get_booster().save_config()` của model đã fit, cho từng fold. Manifest xác nhận cấu hình
**giống nhau ở cả 5 fold** (`booster_save_config_identical_across_folds: true`).

| Parameter | Trạng thái cũ | Giá trị thật (từ save_config) |
|---|---|---|
| `n_estimators` | Explicit: `400` | 400 |
| `max_depth` | Explicit: `2` | `2` |
| `learning_rate` | Explicit: `0.15` | `eta` = `0.150000006` (float32) |
| `random_state` | Explicit: `42` | 42 |
| `eval_metric` | Explicit: `mlogloss` | `mlogloss` |
| `n_jobs` | Explicit: `-1` | -1 |
| `objective` | Not explicit | **`multi:softprob`**, `num_class` = 3 |
| `tree_method` | Not explicit | **`grow_quantile_histmaker`** (hist) |
| `subsample` | Not explicit | **`1`** |
| `colsample_bytree` | Not explicit | **`1`** |
| `min_child_weight` | Not explicit | **`1`** |
| `gamma` | Not explicit | **`min_split_loss` = `0`** |
| `reg_alpha` | Not explicit | **`0`** |
| `reg_lambda` | Not explicit | **`1`** |
| `base_score` | Not explicit | **`5E-1`** |
| `max_bin` / `grow_policy` | Not captured | **`256`** / **`depthwise`** |
| Cấu hình đầy đủ | Not captured | `manifest.model.booster_save_config` (JSON đầy đủ, mỗi fold) |

> Lưu ý cho người đọc code: ở xgboost 2.1.x, `learner.gradient_booster.updater` là một **list**.
> Tree param phải đọc từ `learner.gradient_booster.tree_train_param`; index `updater` bằng string
> sẽ raise `TypeError`.

## Data and preprocessing provenance

- Input: `data/processed/train_X.csv`, 9.600 × 31, toàn bộ `float64`, không có missing.
- Labels: `data/processed/train_y.csv`, mapping `0=C`, `1=CL`, `2=D`.
- Fold source: `data/interim/fold_id.csv`, 5 fold mỗi fold 1.920 dòng.
- Thứ tự feature là thứ tự cột CSV và khớp `feature_list.csv` (manifest verify lại: `ordered_features_match_feature_list_csv: true`).
- Missing dạng số được impute bằng median tính từ file train 9.600 dòng; missing dạng phân loại dùng `Unknown`.
- Biến đổi: một số feature `log1p` cộng thêm tỉ số Bilirubin/Albumin, `OrdinalEncoder`, rồi `RobustScaler` cho cột số.
- **Encoder/scaler được fit MỘT LẦN trước khi CV và dùng lại cho Track C. Đây là leakage tiền xử lý xuyên fold.** Hạn chế này **vẫn còn nguyên**, không được sửa bởi lần mở khóa này, và **đã được nêu trong [calibration_analysis.md](calibration_analysis.md)**. Nó không giải thích được chênh lệch giữa notebook và lần tái tạo, vì cả hai đều đọc cùng file processed CSV đã đóng băng.
- `metrics.compute_metrics` truyền thẳng output `predict_proba` vào `log_loss` của scikit-learn; không clip xác suất ở tầng project.

Giá trị SHA-256 (manifest verify lại 8/8 file khớp):

- `train_X.csv`: `17eecfdece02a5261d42ced2fa0faa46cff08f9dc43cf63b9dfa70ce53329b9b`
- `train_y.csv`: `d72aad5e5ad10397dfb67a90712838dcb3eb51ca72955e6c8fb79758db717982`
- `fold_id.csv`: `a75dc5286e5d2c82e4f21d54982667f706da747bc71026f8b08645140f9a15e2`
- `feature_list.csv`: `323e37fa57eaa0503f5b8ca3a5e0457192ab29b29139ae7107903cd50f815bc4`
- `scaler.pkl`: `9c2eb908a2d7c0702d1a5d96a1791fbd355bf1e1fd4d539e409024de5c0e4830`
- `cat_encoder.pkl`: `045c32afa9d9f61545f9c1be2afb301e9883b673242c66f4c7b9b17f5e8992af`
- `label_encoder.pkl`: `0f7efe1ef839a0aebf48867420ef2b579459bf77c15257e941f0bc029314c939`
- `scores_track_c.csv`: `e8cb6879975ccc3dbc78582f0519e23bdeb2540b41fa617b32665808c4299e5f`

> **Sửa lỗi trong bản audit trước:** hash của `train_X.csv` từng được ghi là
> `17EECFDECE02A5261D42CED2FA0FAA46CFF08F9DC43CF63B9DFA70CE53329B9` — chỉ **63 ký tự hex**, tức
> digest 64 ký tự bị cắt mất chữ `b` cuối. Ai verify theo giá trị cũ sẽ fail oan.

### Sinh lại 3 encoder trong env authoritative (2026-07-27)

Ba file `.pkl` bản trước được pickle bởi **scikit-learn 1.9.0 + numpy 2.x**, không phải env
authoritative. Hệ quả: `scaler.pkl` **không mở được** trong chính env `dynamic`
(`ModuleNotFoundError: No module named 'numpy._core.numeric'` — định dạng pickle của numpy ≥ 2.0),
nên `src/data.load_transformers()` sẽ crash trong đúng môi trường mà mọi thứ khác bắt buộc dùng.

Đã chạy lại `notebooks/02_preprocesing.ipynb` bằng interpreter của env `dynamic` để sinh lại 3 file
(nay mang tag `1.6.1` và đọc được), rồi chạy lại `oof-authoritative` để manifest ghi hash mới. Ba
hash ở trên là giá trị sau lần sinh lại.

Bằng chứng việc này **không đổi bất kỳ con số nào**:

- Toàn bộ CSV trong `data/interim/` và `data/processed/` giữ nguyên **từng byte** (`git status` sạch).
- `results/oof/oof_xgboost_track_c.csv` và `per_fold_metrics.csv` giống hệt bản trước (cùng md5).
- Gate vẫn `PASS` 5/5 fold, sai lệch metric tối đa `5.551115123125783e-17`.
- 4 bảng `tables/calibration_*.csv` giống hệt bản trước.
- So khớp manifest cũ/mới: chỉ khác đúng **3 hash `.pkl` + `generated_at`**; `pip_freeze`,
  `get_xgb_params`, `booster_save_config` và metric từng fold không đổi.

Lý do 3 file này không ảnh hưởng số: `reproduce_xgboost_oof()` chỉ đọc CSV đã đóng băng qua
`data.load_processed()`, không nạp encoder. Chúng nằm trong `PROVENANCE_INPUTS` để chốt hash, không
nằm trên đường tính toán.

## Environment evidence

- Notebook metadata: Python `3.9.18`, kernelspec display name `dynamic`.
- **Env `dynamic` vẫn tồn tại** tại `C:\Users\ADMIN\miniconda3\envs\dynamic`: Python **3.9.18**, xgboost **2.1.4**, scikit-learn **1.6.1**, numpy **1.26.4**, pandas **2.3.3**, scipy **1.13.1**.
- Env này giờ đã được pin vào `requirements.lock.txt` (160 package), kèm cả trong `manifest.environment.pip_freeze`.
- `requirements.txt` chỉ là lower bound, không phải lock file — xem bảng so khớp ở trên.
- Môi trường của lần tái tạo bị fail: Python 3.13.3, XGBoost 3.3.0, scikit-learn 1.8.0, NumPy 2.2.6, pandas 2.3.0, SciPy 1.17.0.

## Diagnostic đã bị thay thế (chạy sai môi trường)

Bảng dưới là kết quả **cũ**, sinh ra trên Python 3.13.3 / XGBoost 3.3.0. Giữ lại làm bằng chứng
cho hiện tượng environment drift. **Không dùng cho bất kỳ số nào trong báo cáo.** File tương ứng
`results/diagnostics/oof_xgboost_reproduction_attempt.csv` đã được đánh dấu `SUPERSEDED` trong
metadata JSON của nó.

| Fold | Δ log loss (actual−expected) | Δ accuracy | Δ macro-F1 |
|---:|---:|---:|---:|
| 0 | -0.002307974 | -0.001041667 | +0.005138837 |
| 1 | +0.000089209 | -0.000520833 | -0.010915011 |
| 2 | +0.000794840 | +0.001562500 | +0.016487257 |
| 3 | +0.000737142 | 0.000000000 | +0.016901819 |
| 4 | +0.000637889 | +0.004687500 | +0.003741365 |

Chênh lệch không phải offset hằng số và lớn hơn nhiều so với sai số làm tròn CSV. 0/5 fold pass gate `1e-5`.

## Kết quả tái tạo authoritative (env `dynamic`)

| Fold | abs Δ log loss | abs Δ accuracy | abs Δ macro-F1 | Trong ngưỡng |
|---:|---:|---:|---:|:--:|
| 0 | 0.000000e+00 | 0.0 | 0.0 | ✅ |
| 1 | 0.000000e+00 | 0.0 | 0.0 | ✅ |
| 2 | 5.551115e-17 | 0.0 | 0.0 | ✅ |
| 3 | 0.000000e+00 | 0.0 | 0.0 | ✅ |
| 4 | 0.000000e+00 | 0.0 | 0.0 | ✅ |

Giá trị 5.55e-17 duy nhất là nhiễu làm tròn float64 (ULP) ở một giá trị log loss — nhỏ hơn ngưỡng
gate `1e-5` khoảng 11 bậc độ lớn. Nói cách khác: 14/15 ô metric khớp **bit-for-bit**.

## Root-cause confidence (đã cập nhật)

| Finding | Confidence | Basis |
|---|---|---|
| **Package/default drift gây ra chênh lệch** | **CONFIRMED** | Cùng code, cùng data, đổi env → từ 0/5 fold pass sang 5/5 fold khớp tuyệt đối |
| **Version package gốc + full config phục hồi được** | **RESOLVED** | Phục hồi từ metadata notebook + env `dynamic` trên máy; đã pin ở `requirements.lock.txt` và chụp vào manifest |
| Baseline `.pkl` không phải model tuned final | CONFIRMED | Git history có trước `aa1ce1a` |
| Score đã nhận và notebook/data/fold thuộc cùng một Git state | CONFIRMED | Git blob giống nhau ở HEAD và `aa1ce1a` |
| Score CSV đến từ lần chạy notebook tuần tự đã lưu | **CONFIRMED** (nâng từ HIGHLY_LIKELY) | Execution count đơn điệu + output khớp, nay cộng thêm tái tạo khớp tuyệt đối |
| Song song / nondeterminism ở tầng platform có góp phần | **UNSUPPORTED** (hạ từ POSSIBLE) | `n_jobs=-1` vẫn cho kết quả khớp tuyệt đối trong đúng env |
| Data/fold artifact hiện tại gây fail | UNSUPPORTED | Git blob và alignment theo vị trí đều khớp |
| Làm tròn CSV gây fail | UNSUPPORTED | Chênh lệch lớn hơn nhiều độ chính xác làm tròn |
| Leakage tiền xử lý trước CV | **CONFIRMED, CHƯA SỬA** | Encoder/scaler fit trước CV; đã nêu trong `calibration_analysis.md` |

## Bài học rút ra

Notebook chỉ lưu kernel **name** chứ không lưu version package. Chừng đó là không đủ để tái tạo,
và khoảng trống đó đã bị chẩn đoán sai thành "mất provenance", suýt dẫn tới việc đi xin artifact
bên ngoài trong khi môi trường đúng vẫn nằm ngay trên máy. `requirements.lock.txt` cùng
`manifest.environment` khép lại khoảng trống này. Với mọi artifact số về sau: **chốt env trước,
rồi mới điều tra chênh lệch số.**

## Giới hạn tái lập: Random Forest lệch 1 ULP giữa các lần chạy (2026-07-27)

Chạy `scripts/reproduce_all.py --full` (chạy lại notebook 02 → 06 rồi sinh lại toàn bộ artifact) cho
kết quả: **tracks A và B tái tạo byte-identical**, riêng `results/scores_track_c.csv` lệch ở **đúng
một ô**:

| | log loss của Random Forest, fold 3 |
|---|---|
| Bản đã chốt | `0.40094091193670306` |
| Lần chạy lại | `0.4009409119367031` |
| Chênh lệch | `1.11e-16` |

Nguyên nhân: Random Forest chạy `n_jobs=-1` cộng xác suất của 600 cây theo thứ tự luồng do OS quyết
định. Phép cộng dấu phẩy động **không có tính kết hợp**, nên `(a+b)+c` và `a+(b+c)` có thể khác nhau
ở chữ số cuối cùng biểu diễn được của kiểu double. Đây là giới hạn của phần cứng/thư viện, không
phải lỗi pipeline, và **không** ảnh hưởng bất kỳ kết luận nào: nó nhỏ hơn 11 bậc so với dung sai
`1e-5` của gate tái tạo OOF, và nhỏ hơn 14 bậc so với chênh lệch log loss nhỏ nhất giữa hai mô hình
được so sánh (`0.024` giữa XGBoost và Random Forest).

Hệ quả về cách kiểm tra: `reproduce_all.py` so các bảng số **theo giá trị** với ngưỡng `1e-12`, không
so từng byte. Muốn byte-identical thật thì phải đặt `n_jobs=1` cho Random Forest - đổi lại là chậm
hơn nhiều, và sẽ làm lệch `scores_track_c.csv` đang được chốt hash.

File `results/scores_track_c.csv` sau lần chạy thử này đã được khôi phục về **bản đã chốt**
(SHA-256 `e8cb6879…`, đúng giá trị ghi ở mục trên), rồi sinh lại manifest và các bảng dẫn xuất để
toàn bộ artifact nhất quán với nó.

## Trạng thái phân tích calibration

Gate reproduction đã `PASS`. `run_calibration_analysis()` đọc OOF authoritative và đã chạy xong:
ECE, Brier score và reliability diagram nằm ở [calibration_analysis.md](calibration_analysis.md),
trong đó hạn chế leakage tiền xử lý ở mục
[Data and preprocessing provenance](#data-and-preprocessing-provenance) được nêu lại đầy đủ.
