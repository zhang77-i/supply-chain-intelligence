from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import PipelineConfig


def make_fold_manifest(
    samples: pd.DataFrame,
    validation_folds: int,
    minimum_training_cutoffs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = samples.loc[samples["cutoff_kind"].eq("backtest")].copy()
    cutoffs = sorted(pd.to_datetime(labeled["cutoff_date"]).dt.date.unique())
    eligible = cutoffs[minimum_training_cutoffs:]
    validation_dates = eligible[-validation_folds:]

    manifests: list[dict] = []
    assignments: list[pd.DataFrame] = []
    for fold_number, validation_date in enumerate(validation_dates, start=1):
        cutoff_dates = pd.to_datetime(labeled["cutoff_date"]).dt.date
        train_mask = cutoff_dates < validation_date
        validation_mask = cutoff_dates == validation_date
        train = labeled.loc[train_mask, ["sample_id", "cutoff_date"]].copy()
        valid = labeled.loc[validation_mask, ["sample_id", "cutoff_date"]].copy()
        if train.empty or valid.empty:
            continue

        fold_id = f"fold_{fold_number:02d}"
        train["fold_id"] = fold_id
        train["split"] = "train"
        valid["fold_id"] = fold_id
        valid["split"] = "validation"
        assignments.extend([train, valid])
        manifests.append(
            {
                "fold_id": fold_id,
                "train_cutoff_min": train["cutoff_date"].min(),
                "train_cutoff_max": train["cutoff_date"].max(),
                "validation_cutoff": pd.Timestamp(validation_date),
                "train_rows": len(train),
                "validation_rows": len(valid),
                "train_cutoffs": train["cutoff_date"].nunique(),
            }
        )

    manifest = pd.DataFrame(manifests)
    assignment = (
        pd.concat(assignments, ignore_index=True)
        if assignments
        else pd.DataFrame(columns=["sample_id", "cutoff_date", "fold_id", "split"])
    )
    return manifest, assignment


def validate_no_time_leakage(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    for row in manifest.itertuples(index=False):
        fold = assignments.loc[assignments["fold_id"].eq(row.fold_id)]
        train_max = pd.to_datetime(
            fold.loc[fold["split"].eq("train"), "cutoff_date"]
        ).max()
        validation_min = pd.to_datetime(
            fold.loc[fold["split"].eq("validation"), "cutoff_date"]
        ).min()
        if not train_max < validation_min:
            raise ValueError(
                f"Time leakage in {row.fold_id}: train={train_max}, validation={validation_min}"
            )


def build_backtest_artifacts(samples: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    manifest, assignments = make_fold_manifest(
        samples,
        validation_folds=config.backtest["validation_folds"],
        minimum_training_cutoffs=config.backtest["minimum_training_cutoffs"],
    )
    validate_no_time_leakage(manifest, assignments)

    config.outputs["fold_manifest"].parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.outputs["fold_manifest"], index=False)
    assignments.to_parquet(config.outputs["fold_assignments"], index=False)

    definitions = [
        {
            "fold_id": row.fold_id,
            "train_cutoff_min": str(row.train_cutoff_min.date()),
            "train_cutoff_max": str(row.train_cutoff_max.date()),
            "validation_cutoff": str(row.validation_cutoff.date()),
            "train_rows": int(row.train_rows),
            "validation_rows": int(row.validation_rows),
            "train_cutoffs": int(row.train_cutoffs),
        }
        for row in manifest.itertuples(index=False)
    ]
    definitions_path = config.outputs["fold_manifest"].with_suffix(".json")
    definitions_path.write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
