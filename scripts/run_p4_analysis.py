"""Command-line entry point for reproducible P4 artifacts."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("scores", "inference", "oof", "calibration"))
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))

    from src import p4_analysis

    if args.phase == "scores":
        _, report = p4_analysis.build_scores_all(project_root)
        print(report)
    elif args.phase == "inference":
        scores = __import__("pandas").read_csv(project_root / "results" / "scores_all.csv")
        hep13, hep14 = p4_analysis.write_inference_outputs(project_root, scores)
        print("HEP-13 rows:", len(hep13))
        print("HEP-14 rows:", len(hep14))
    elif args.phase == "oof":
        _, gate, reproduction = p4_analysis.reproduce_xgboost_oof(project_root)
        print(gate)
        print(reproduction.to_string(index=False))
    else:
        print(p4_analysis.run_hep15_calibration(project_root))


if __name__ == "__main__":
    main()
