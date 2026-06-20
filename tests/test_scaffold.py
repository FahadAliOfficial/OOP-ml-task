"""Scaffold checks for the project structure."""

from importlib import import_module
from pathlib import Path


REQUIRED_DIRECTORIES = [
    "src/data",
    "src/preprocessing",
    "src/models",
    "src/evaluation",
    "src/api",
    "src/config",
    "src/utils",
    "tests",
    "notebooks",
    "data/raw",
    "data/interim",
    "data/processed",
    "artifacts/ml",
    "artifacts/dl",
    "artifacts/evaluation",
    "logs",
]

REQUIRED_CONFIGS = [
    "src/config/data.yaml",
    "src/config/ml.yaml",
    "src/config/dl.yaml",
    "src/config/api.yaml",
]

IMPORTABLE_MODULES = [
    "src.data.build_dataset",
    "src.preprocessing.clean_text",
    "src.preprocessing.validate_dataset",
    "src.models.train_ml",
    "src.models.train_dl",
    "src.models.inference",
    "src.evaluation.evaluate_models",
    "src.api.main",
    "src.api.schemas",
    "src.api.database",
    "src.utils.config",
    "src.utils.logging",
]


def test_required_directories_exist() -> None:
    for directory in REQUIRED_DIRECTORIES:
        assert Path(directory).is_dir(), f"Missing directory: {directory}"


def test_required_configs_exist() -> None:
    for config in REQUIRED_CONFIGS:
        assert Path(config).is_file(), f"Missing config: {config}"


def test_scaffold_modules_import() -> None:
    for module_name in IMPORTABLE_MODULES:
        import_module(module_name)
