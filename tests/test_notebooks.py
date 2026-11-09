import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import nbformat


def test_notebooks_schema_python_and_embedded_sources():
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    # Embedded project snapshots intentionally exclude the notebooks themselves.
    if not notebooks:
        return  # Allows the embedded package to run its numerical tests in Colab.
    assert len(notebooks) == 4
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.execution_count is None and not cell.outputs
        setup = next(cell.source for cell in notebook.cells if "base64.b64decode" in cell.source)
        tree = ast.parse(setup)
        expression = next(node.value.args[0] for node in tree.body if isinstance(node, ast.Assign)
                          and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                          and node.value.func.attr == "b64decode")
        archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(expression))))
        for module in (root/"freqblender").glob("*.py"):
            assert archive.read("freqblender/"+module.name) == module.read_bytes()
