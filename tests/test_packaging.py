"""Checks for reproducible packaging and final submission documents."""

from pathlib import Path


def test_requirements_are_pinned() -> None:
    for filename in ("requirements.txt", "requirements-api.txt"):
        lines = [
            line.strip()
            for line in Path(filename).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        assert lines
        assert all("==" in line for line in lines)


def test_docker_packaging_files_exist() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert Path(".dockerignore").is_file()
    assert "HEALTHCHECK" in dockerfile
    assert "src.api.main:app" in dockerfile
    assert "EXPOSE 8000" in dockerfile


def test_readme_contains_required_documentation_topics() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    required_sections = [
        "## Architecture",
        "## Dataset Building And Curation",
        "## Preprocessing Pipeline",
        "## Model A: Classical ML",
        "## Model B: DistilBERT",
        "## Results And Comparison",
        "## Installation",
        "## FastAPI Usage",
    ]

    for section in required_sections:
        assert section in readme

    assert '"text": "The product quality is excellent and delivery was fast."' in readme
