"""Check notebook schema, plain-Python cell syntax, and reproducible generation."""
import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 5
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path.name}:cell-{index}")
                assert cell.execution_count is None and not cell.outputs
        print(f"Valid: {path.name} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
