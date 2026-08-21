from __future__ import annotations

import logging
from pathlib import Path

from .audit import write_audit_report, write_run_metadata
from .backtest import build_backtest_artifacts
from .config import load_config
from .db import build_database
from .extract import ensure_raw_data
from .feature_engineering import build_feature_datasets

LOGGER = logging.getLogger(__name__)


def run_pipeline(config_path: str | Path) -> None:
    config = load_config(config_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("1/5 Extracting and validating raw files")
    ensure_raw_data(config)
    LOGGER.info("2/5 Building DuckDB raw, clean, panel and audit layers")
    build_database(config)
    LOGGER.info("3/5 Building leakage-safe feature snapshots")
    samples = build_feature_datasets(config)
    LOGGER.info("4/5 Building rolling backtest folds")
    manifest = build_backtest_artifacts(samples, config)
    LOGGER.info("5/5 Writing audit report and run metadata")
    write_audit_report(config)
    write_run_metadata(config, samples, manifest)
    LOGGER.info("Pipeline complete: %s", config.root)
