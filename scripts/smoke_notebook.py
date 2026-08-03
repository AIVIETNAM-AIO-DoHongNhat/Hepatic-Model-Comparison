"""Chạy tuần tự các cell code của notebook mà không cần cài Jupyter."""

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Gốc dự án để thêm vào sys.path; mặc định là thư mục cha của notebook.",
    )
    args = parser.parse_args()
    # Notebook in tiếng Việt; console Windows mặc định cp1252 sẽ ném UnicodeEncodeError
    # và biến một notebook chạy tốt thành "fail" giả.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # Chạy không có màn hình -> không mở cửa sổ figure.
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook_path = args.notebook.resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_smoke__"}
    project_root = (args.project_root or notebook_path.parent.parent).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Phải chdir vào chính thư mục chứa notebook: mọi notebook trong repo đều xác định
    # gốc dự án bằng `Path.cwd().parent`, nên chdir ra gốc repo sẽ khiến đường dẫn lệch
    # lên một cấp và cell đầu tiên đã hỏng.
    os.chdir(notebook_path.parent)
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            exec(compile(source, f"{args.notebook.name}:cell-{index}", "exec"), namespace)
        except Exception as exc:
            raise RuntimeError(f"Notebook smoke test failed at code cell {index}.") from exc
    print(f"PASS: executed all code cells in {args.notebook}")


if __name__ == "__main__":
    main()
