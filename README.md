# Statistical-Significance-Assessment-in-Machine-Learning-Model-Comparison
hepatic-model-comparison/
├── data/
│   ├── raw/              # dữ liệu gốc cuộc thi — KHÔNG commit (xem .gitignore)
│   └── processed/        # train sạch, 2 bản scaling, fold_id.csv
├── notebooks/
│   ├── 01_eda.ipynb              # P1
│   ├── 02_preprocessing.ipynb    # P1
│   ├── 03_model_track_a.ipynb    # P2
│   ├── 04_model_track_b.ipynb    # P3
│   ├── 05_statistics.ipynb       # P4
│   └── 06_report_assets.ipynb    # P5
├── src/
│   ├── data.py          # hàm load + làm sạch dùng chung
│   ├── metrics.py       # hàm tính log_loss, accuracy, f1 THỐNG NHẤT
│   └── stats.py         # t-test, CI, Bonferroni, calibration
├── results/
│   ├── scores_track_a.csv    # P2 xuất
│   ├── scores_track_b.csv    # P3 xuất
│   └── scores_all.csv        # gộp, P4 dùng
├── figures/             # biểu đồ cho report (HEP-16)
├── tables/              # bảng cho report (HEP-20)
├── submissions/         # file nộp Kaggle
├── report/              # LaTex + PDF
├── requirements.txt     # packets
├── .gitignore
└── README.md