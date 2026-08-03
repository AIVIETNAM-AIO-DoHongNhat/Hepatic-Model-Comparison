"""Kiểm tra tính tái lập của toàn bộ pipeline.

Hai chế độ:

    --check  (mặc định)  chỉ ĐỌC. Đối chiếu artifact đang có với bản ghi provenance:
                         môi trường, 8 hash đầu vào, gate tái tạo, bảng điểm, OOF,
                         và các số calibration. Không ghi gì, an toàn chạy bất cứ lúc nào.

    --full               chạy lại pipeline từ notebook 02 đến 06 rồi sinh lại toàn bộ
                         artifact phân tích, sau đó so từng file với bản trước khi chạy.
                         GHI ĐÈ dữ liệu processed và results/ - chỉ dùng khi thật sự muốn
                         trả lời câu hỏi "clone sạch chạy lại có ra đúng số không".

Chạy bằng đúng interpreter của env authoritative, nếu không mọi thứ sẽ FAIL ngay ở bước
đầu tiên và đó là hành vi đúng.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "02_preprocesing.ipynb",
    "04_model_track_a.ipynb",
    "05_model_track_b.ipynb",
    "06_model_track_c.ipynb",
]
PHASES = ["scores", "inference", "oof-authoritative", "calibration"]

# Ngưỡng dung sai khi so lại các bảng số sau một lần chạy lại.
#
# Vì sao KHÔNG so từng byte: Random Forest chạy `n_jobs=-1` cộng xác suất của 600 cây theo
# thứ tự luồng do OS quyết định. Phép cộng dấu phẩy động không có tính kết hợp, nên tổng có
# thể lệch đúng 1 ULP giữa hai lần chạy. Thực đo: log loss của RF ở fold 3 ra
# 0.40094091193670306 và 0.4009409119367031 - lệch 1.1e-16, tức chữ số cuối cùng biểu diễn
# được của kiểu double. Đòi byte-identical ở đây là đòi một thứ phần cứng không hứa;
# ngưỡng dưới đây vẫn chặt hơn 4 bậc so với dung sai 1e-5 của gate tái tạo OOF.
NUMERIC_TOLERANCE = 1e-12

# Các file phải khớp lại sau khi chạy lại: bảng số so theo giá trị (dung sai ở trên),
# file khác so bằng hash. Đây chính là định nghĩa "tái lập được" của dự án này.
INVARIANT_FILES = [
    "results/scores_track_a.csv",
    "results/scores_track_b.csv",
    "results/scores_track_c.csv",
    "results/scores_all.csv",
    "results/oof/oof_xgboost_track_c.csv",
    "results/oof/oof_xgboost_track_c.per_fold_metrics.csv",
    "tables/pairwise_comparisons.csv",
    "tables/assumption_checks.csv",
    "tables/calibration_metrics.csv",
    "tables/calibration_ece_bin_sensitivity.csv",
    "data/processed/train_X.csv",
    "data/processed/train_y.csv",
    "data/interim/fold_id.csv",
]


class Report:
    """Gom kết quả kiểm tra rồi in một bảng PASS/FAIL duy nhất ở cuối."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))

    def print_and_exit(self) -> None:
        width = max(len(name) for name, _, _ in self.rows)
        print()
        for name, ok, detail in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
        failed = [name for name, ok, _ in self.rows if not ok]
        print()
        if failed:
            print(f"{len(failed)}/{len(self.rows)} kiem tra FAIL: {', '.join(failed)}")
            sys.exit(1)
        print(f"Tat ca {len(self.rows)} kiem tra PASS.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_snapshot(path: Path, before: bytes) -> tuple[bool, float]:
    """So một file với bản chụp trước khi chạy lại.

    Trả về (khớp, lệch số học lớn nhất). CSV được so theo GIÁ TRỊ với NUMERIC_TOLERANCE;
    file khác so nguyên byte. Cột chữ và hình dạng bảng vẫn phải trùng tuyệt đối.
    """
    import io

    import numpy as np
    import pandas as pd

    after = path.read_bytes()
    if after == before:
        return True, 0.0
    if path.suffix != ".csv":
        return False, float("nan")

    old = pd.read_csv(io.BytesIO(before))
    new = pd.read_csv(path)
    if old.shape != new.shape or list(old.columns) != list(new.columns):
        return False, float("nan")

    # Cẩn thận: pandas coi bool LÀ kiểu số, nên `is_numeric_dtype` một mình sẽ kéo cả
    # significant_bonferroni vào nhóm so theo dung sai rồi ném TypeError khi trừ hai cột bool.
    # Bool phải so bằng nhau tuyệt đối: cờ ý nghĩa thống kê mà đổi giá trị thì đó là đổi
    # kết luận, không phải nhiễu dấu phẩy động.
    numeric = [
        c for c in old.columns
        if pd.api.types.is_numeric_dtype(old[c]) and old[c].dtype != bool
    ]
    others = [c for c in old.columns if c not in numeric]
    if not old[others].equals(new[others]):
        return False, float("nan")
    if not numeric:
        return False, float("nan")

    delta = float(np.nanmax((old[numeric] - new[numeric]).abs().to_numpy()))
    return delta <= NUMERIC_TOLERANCE, delta


def check(project_root: Path, report: Report) -> None:
    """Đối chiếu artifact hiện có với bản ghi provenance. Chỉ đọc."""
    import pandas as pd

    from src import final_analysis as fa

    env = fa._environment_manifest()
    report.add(
        "moi truong khop env authoritative",
        env["matches_authoritative_env"],
        f"{env['conda_env']} / Python {env['python']} / xgboost {env['key_packages']['xgboost']}",
    )

    manifest_path = project_root / fa.AUTHORITATIVE_MANIFEST
    if not manifest_path.exists():
        report.add("manifest provenance ton tai", False, str(manifest_path))
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mismatched = [
        relative
        for relative, expected in manifest["data"]["sha256"].items()
        if sha256(project_root / relative) != expected
    ]
    report.add(
        "8 hash dau vao khop manifest",
        not mismatched,
        f"{len(manifest['data']['sha256']) - len(mismatched)}/{len(manifest['data']['sha256'])} file",
    )

    gate = json.loads((project_root / fa.AUTHORITATIVE_GATE).read_text(encoding="utf-8"))
    report.add(
        "gate tai tao OOF",
        gate["status"] == "PASS" and gate["folds_passed"] == gate["folds_total"],
        f"{gate['status']} {gate['folds_passed']}/{gate['folds_total']} fold, "
        f"lech toi da {gate['max_absolute_metric_difference']:.3e}",
    )

    scores = pd.read_csv(project_root / "results" / "scores_all.csv")
    try:
        fa.validate_scores(scores)
        report.add("scores_all.csv hop le", True, f"{len(scores)} dong / 6 mo hinh / 5 fold")
    except ValueError as error:
        report.add("scores_all.csv hop le", False, str(error)[:70])

    tracks = pd.concat(
        [pd.read_csv(project_root / "results" / f"scores_track_{t}.csv") for t in "abc"],
        ignore_index=True,
    )
    tracks["fold"] = tracks["fold"].astype(int)
    merged = scores.merge(tracks, on=["model", "fold"], suffixes=("_all", "_track"))
    consistent = len(merged) == len(scores) and all(
        (merged[f"{column}_all"] - merged[f"{column}_track"]).abs().max() < 1e-12
        for column in ("log_loss", "accuracy", "macro_f1")
    )
    report.add("scores_all khop 3 file track", consistent, f"{len(merged)} cap dong")

    oof = pd.read_csv(project_root / fa.AUTHORITATIVE_OOF)
    folds = pd.read_csv(project_root / "data" / "interim" / "fold_id.csv")
    try:
        validation = fa.validate_oof_xgboost(oof, folds)
        report.add("OOF authoritative hop le", True,
                   f"{validation['rows']} dong, lech tong xac suat {validation['probability_row_sum_max_abs_error']:.1e}")
    except ValueError as error:
        report.add("OOF authoritative hop le", False, str(error)[:70])

    metrics_json = project_root / fa.CALIBRATION_METRICS_JSON
    metrics_csv = project_root / fa.CALIBRATION_METRICS_TABLE
    if metrics_json.exists() and metrics_csv.exists():
        stored = json.loads(metrics_json.read_text(encoding="utf-8"))
        table = pd.read_csv(metrics_csv).iloc[0]
        same = all(
            abs(float(stored[key]) - float(table[key])) < 1e-12
            for key in ("multiclass_brier_score", "top_label_ece", "macro_classwise_ece")
        )
        report.add("so calibration khop giua JSON va CSV", same,
                   f"ECE {table['top_label_ece']:.4f}, Brier {table['multiclass_brier_score']:.4f}")
    else:
        report.add("so calibration khop giua JSON va CSV", False, "thieu artifact calibration")

    missing = [name for name in INVARIANT_FILES if not (project_root / name).exists()]
    report.add("du artifact bat bien", not missing, f"{len(INVARIANT_FILES) - len(missing)}/{len(INVARIANT_FILES)} file")


def run(command: list[str], project_root: Path) -> tuple[bool, str]:
    completed = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
    tail = (completed.stdout or completed.stderr or "").strip().splitlines()
    return completed.returncode == 0, (tail[-1][:70] if tail else "")


def full(project_root: Path, report: Report) -> None:
    """Chạy lại toàn bộ pipeline rồi so từng file bất biến với bản trước khi chạy."""
    before = {
        name: (project_root / name).read_bytes()
        for name in INVARIANT_FILES
        if (project_root / name).exists()
    }
    print(f"Da chup {len(before)} file bat bien truoc khi chay lai.")

    for notebook in NOTEBOOKS:
        ok, tail = run(
            [sys.executable, "scripts/smoke_notebook.py", f"notebooks/{notebook}"], project_root
        )
        report.add(f"chay lai {notebook}", ok, tail)
        if not ok:
            return

    for phase in PHASES:
        ok, tail = run(
            [sys.executable, "scripts/run_final_analysis.py", phase, "--project-root", "."],
            project_root,
        )
        report.add(f"phase {phase}", ok, tail)
        if not ok:
            return

    changed, worst = [], 0.0
    for name, snapshot in before.items():
        matched, delta = compare_snapshot(project_root / name, snapshot)
        if not matched:
            changed.append(name)
        elif delta == delta:  # bỏ qua NaN
            worst = max(worst, delta)
    report.add(
        "moi file bat bien tai tao lai duoc",
        not changed,
        f"{len(before)} file, lech so hoc toi da {worst:.2e} (nguong {NUMERIC_TOLERANCE:.0e})"
        if not changed
        else f"KHONG TAI TAO DUOC: {', '.join(changed)}",
    )

    if worst > 0:
        # Chạy lại luôn để lại artifact mới lệch vài ULP so với bản đã chốt. Chúng đúng về
        # mặt số học nhưng khác hash, nên report/track_c_provenance_audit.md sẽ không còn
        # verify được. Nhắc rõ ở đây vì rất dễ quên.
        print(
            "\nLUU Y: lan chay nay da ghi de artifact bang ban moi (lech <= "
            f"{worst:.1e}, van dat nguong). Muon giu nguyen hash da chot trong "
            "report/track_c_provenance_audit.md thi khoi phuc lai:\n"
            "  git checkout -- results/scores_track_c.csv\n"
            "  python scripts/run_final_analysis.py scores --project-root .\n"
            "  python scripts/run_final_analysis.py inference --project-root .\n"
            "  python scripts/run_final_analysis.py oof-authoritative --project-root .\n"
            "  python scripts/run_final_analysis.py calibration --project-root ."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="Chay lai toan bo pipeline (GHI DE artifact) thay vi chi kiem tra.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))

    print(f"Project root : {project_root}")
    print(f"Interpreter  : {sys.executable}")
    print(f"Che do       : {'--full (ghi de)' if args.full else '--check (chi doc)'}")

    report = Report()
    if args.full:
        full(project_root, report)
    check(project_root, report)
    report.print_and_exit()


if __name__ == "__main__":
    main()
