from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    raw: dict[str, Path]
    outputs: dict[str, Path]
    backtest: dict[str, int]
    modeling: dict[str, Any]
    allocation: dict[str, Any]
    random_seed: int


def _resolve(root: Path, value: str) -> Path:
    return (root / value).resolve()


def load_config(config_path: str | Path) -> PipelineConfig:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)

    root = config_file.parent.parent
    data = payload["data"]
    raw = {
        "archive": _resolve(root, data["archive"]),
        "extracted_dir": _resolve(root, data["extracted_dir"]),
        "item_feature": _resolve(root, data["item_feature"]),
        "item_store_feature": _resolve(root, data["item_store_feature"]),
        "config": _resolve(root, data["config"]),
        "sample_submission": _resolve(root, data["sample_submission"]),
    }
    outputs = {
        "database": _resolve(root, data["database"]),
        "feature_dataset": _resolve(root, data["feature_dataset"]),
        "inference_dataset": _resolve(root, data["inference_dataset"]),
        "fold_assignments": _resolve(root, data["fold_assignments"]),
        "fold_manifest": _resolve(root, data["fold_manifest"]),
        "intermittent_baselines": _resolve(root, data["intermittent_baselines"]),
        "model_predictions": _resolve(root, data["model_predictions"]),
        "inventory_recommendations": _resolve(root, data["inventory_recommendations"]),
        "coordinated_allocation": _resolve(root, data["coordinated_allocation"]),
        "stochastic_allocation": _resolve(root, data["stochastic_allocation"]),
        "audit_report": root / "reports" / "initial_data_audit.md",
        "audit_tables": root / "reports" / "tables",
        "run_metadata": root / "reports" / "run_metadata.json",
    }
    return PipelineConfig(
        root=root,
        raw=raw,
        outputs=outputs,
        backtest={key: int(value) for key, value in payload["backtest"].items()},
        modeling=payload.get("modeling", {}),
        allocation=payload.get("allocation", {}),
        random_seed=int(payload["project"]["random_seed"]),
    )
