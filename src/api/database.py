"""SQLite logging helpers for prediction requests."""

from pathlib import Path
import re
import sqlite3
from typing import Any


CREATE_LOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    input_text TEXT NOT NULL,
    model_used TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL,
    inference_time_ms REAL NOT NULL
);
"""


TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_table_name(table_name: str) -> str:
    """Validate a configured SQLite identifier before interpolation."""
    if not TABLE_NAME_PATTERN.fullmatch(table_name):
        raise ValueError(f"Invalid SQLite table name: {table_name}")
    return table_name


def initialize_database(
    sqlite_path: str | Path,
    table_name: str = "prediction_logs",
) -> None:
    """Create the prediction log table if it does not already exist."""
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_name = validate_table_name(table_name)
    create_sql = CREATE_LOG_TABLE_SQL.replace("prediction_logs", table_name)
    with sqlite3.connect(path) as connection:
        connection.execute(create_sql)


def log_prediction(
    sqlite_path: str | Path,
    *,
    input_text: str,
    model_used: str,
    prediction: str,
    confidence: float | None,
    inference_time_ms: float,
    table_name: str = "prediction_logs",
) -> int:
    """Insert a prediction audit row and return its generated identifier."""
    path = Path(sqlite_path)
    table_name = validate_table_name(table_name)
    initialize_database(path, table_name)
    insert_sql = f"""
        INSERT INTO {table_name}
            (input_text, model_used, prediction, confidence, inference_time_ms)
        VALUES (?, ?, ?, ?, ?)
    """
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            insert_sql,
            (input_text, model_used, prediction, confidence, inference_time_ms),
        )
        return int(cursor.lastrowid)


def fetch_prediction_logs(
    sqlite_path: str | Path,
    table_name: str = "prediction_logs",
) -> list[dict[str, Any]]:
    """Return prediction rows for tests and operational inspection."""
    path = Path(sqlite_path)
    table_name = validate_table_name(table_name)
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT * FROM {table_name} ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]
