import pandas as pd

from cainiao_inventory.backtest import make_fold_manifest, validate_no_time_leakage


def test_rolling_folds_keep_validation_after_training() -> None:
    cutoffs = pd.date_range("2024-01-01", periods=10, freq="14D")
    samples = pd.DataFrame(
        {
            "sample_id": range(20),
            "cutoff_date": cutoffs.repeat(2),
            "cutoff_kind": "backtest",
        }
    )
    manifest, assignments = make_fold_manifest(
        samples,
        validation_folds=3,
        minimum_training_cutoffs=4,
    )
    validate_no_time_leakage(manifest, assignments)
    assert len(manifest) == 3
    assert (manifest["train_cutoff_max"] < manifest["validation_cutoff"]).all()
