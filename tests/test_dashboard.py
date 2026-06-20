"""Integrity checks for the executed comparison dashboard."""

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/model_comparison_dashboard.ipynb")


def test_dashboard_exists_and_contains_required_sections() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    required_sections = [
        "Dataset Summary",
        "Overall Test Performance",
        "Per-Class Performance",
        "Confusion Matrices",
        "Efficiency Comparison",
        "Deployment Recommendation",
        "Final Conclusion",
    ]
    for section in required_sections:
        assert section in markdown


def test_dashboard_executed_without_cell_errors() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []
