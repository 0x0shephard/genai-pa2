import ast
import json
from pathlib import Path

from alignment.utils import ROOT


def test_notebook_python_cells_are_valid_and_outputs_empty():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 5
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        code = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert 'REPO_URL = "https://github.com/0x0shephard/genai-pa2.git"' in code
        assert "GIT_REF" in code and "rev-parse" in code
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                assert cell["outputs"] == [] and cell["execution_count"] is None


def test_main_trainers_do_not_use_prohibited_libraries():
    for file in ("alignment/core.py", "alignment/train.py"):
        tree = ast.parse((ROOT / file).read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("trl", "peft")
                assert not any(alias.name.endswith("Trainer") for alias in node.names)
