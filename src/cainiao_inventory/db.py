from __future__ import annotations

from pathlib import Path

import duckdb

from .config import PipelineConfig
from .schema import CONFIG_COLUMNS, ITEM_COLUMNS, STORE_COLUMNS, duckdb_column_map


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _render_sql(template: str, config: PipelineConfig) -> str:
    replacements = {
        "{{ITEM_FEATURE_PATH}}": _sql_path(config.raw["item_feature"]),
        "{{ITEM_STORE_FEATURE_PATH}}": _sql_path(config.raw["item_store_feature"]),
        "{{CONFIG_PATH}}": _sql_path(config.raw["config"]),
        "{{SAMPLE_SUBMISSION_PATH}}": _sql_path(config.raw["sample_submission"]),
        "{{ITEM_COLUMN_MAP}}": duckdb_column_map(ITEM_COLUMNS),
        "{{STORE_COLUMN_MAP}}": duckdb_column_map(STORE_COLUMNS),
        "{{CONFIG_COLUMN_MAP}}": duckdb_column_map(CONFIG_COLUMNS),
        "{{HISTORY_PRECEDING}}": str(config.backtest["history_days"] - 1),
        "{{HORIZON_DAYS}}": str(config.backtest["horizon_days"]),
        "{{HORIZON_FOLLOWING_START}}": "1",
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def connect(config: PipelineConfig, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    database = config.outputs["database"]
    database.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(database), read_only=read_only)


def execute_sql_file(
    connection: duckdb.DuckDBPyConnection,
    sql_file: Path,
    config: PipelineConfig,
) -> None:
    sql = _render_sql(sql_file.read_text(encoding="utf-8"), config)
    statements = connection.extract_statements(sql)
    for statement_number, statement in enumerate(statements, start=1):
        try:
            connection.execute(statement)
        except Exception as exc:
            raise RuntimeError(
                f"SQL failed in {sql_file.name}, statement {statement_number}"
            ) from exc


def build_database(config: PipelineConfig) -> None:
    sql_dir = config.root / "sql"
    scripts = [
        "01_create_raw_views.sql",
        "02_create_clean_tables.sql",
        "03_build_daily_panel.sql",
        "04_create_audit_views.sql",
    ]
    with connect(config) as connection:
        connection.execute("SET threads = 4")
        connection.execute("SET preserve_insertion_order = false")
        for filename in scripts:
            execute_sql_file(connection, sql_dir / filename, config)
        connection.execute("CHECKPOINT")
