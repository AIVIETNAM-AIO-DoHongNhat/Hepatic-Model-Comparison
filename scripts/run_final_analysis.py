"""CLI sinh lại các artifact của phân tích cuối một cách tái lập được."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("scores", "inference", "oof", "oof-authoritative", "calibration", "runtime"),
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))

    from src import final_analysis

    if args.phase == "scores":
        _, report = final_analysis.build_scores_all(project_root)
        print(report)
    elif args.phase == "inference":
        scores = __import__("pandas").read_csv(project_root / "results" / "scores_all.csv")
        pairwise, assumptions = final_analysis.write_inference_outputs(project_root, scores)
        # Console Windows mặc định cp1252 nên giữ log ở ASCII.
        print("pairwise comparison rows:", len(pairwise))
        print("assumption check rows:", len(assumptions))
    elif args.phase == "oof":
        _, gate, reproduction = final_analysis.reproduce_xgboost_oof(project_root)
        print(gate)
        print(reproduction.to_string(index=False))
    elif args.phase == "oof-authoritative":
        # Phải chạy bằng interpreter của env "dynamic" (Python 3.9.18), xem
        # requirements.lock.txt. Env khác sẽ không tái tạo được scores_track_c.csv.
        environment = final_analysis._environment_manifest()
        print("Environment:", environment["conda_env"], "| Python", environment["python"])
        print("Key packages:", environment["key_packages"])
        if not environment["matches_authoritative_env"]:
            print(
                "\nCẢNH BÁO: đây KHÔNG phải env authoritative "
                f"({final_analysis.AUTHORITATIVE_CONDA_ENV} / Python "
                f"{final_analysis.AUTHORITATIVE_PYTHON}). Gate gần như chắc chắn sẽ FAIL."
            )
        _, gate, reproduction = final_analysis.reproduce_xgboost_oof(
            project_root, authoritative=True
        )
        print("\n" + reproduction.to_string(index=False))
        print("\nGate:", gate)
        print("OOF     :", final_analysis.AUTHORITATIVE_OOF)
        print("Manifest:", final_analysis.AUTHORITATIVE_MANIFEST)
    elif args.phase == "runtime":
        runtime = final_analysis.measure_runtime(project_root)
        summary = runtime.groupby("model")[["fit_seconds", "predict_seconds"]].agg(["mean", "std"])
        summary = summary.reindex(summary[("fit_seconds", "mean")].sort_values().index)
        print(summary.round(4).to_string())
        print("\nrows:", len(runtime), "| output:", final_analysis.RUNTIME_TABLE)
        print("log loss reproduced the reported scores on every fold:",
              bool(runtime["log_loss_khop_scores"].all()))
    else:
        print(final_analysis.run_calibration_analysis(project_root))


if __name__ == "__main__":
    main()
