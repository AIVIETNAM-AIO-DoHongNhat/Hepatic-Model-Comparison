"""Execute notebook code cells sequentially without requiring Jupyter packages."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_smoke__"}
    os.chdir(args.project_root.resolve())
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
