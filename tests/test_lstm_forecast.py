import numpy as np
import pandas as pd
import pytest

from cainiao_inventory.lstm_forecast import (
    build_sequences,
    regression_metrics,
    rolling_train_validation_split,
    train_lstm_baseline,
)


def daily_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rows = []
    for item_id, offset in [(101, 0.0), (202, 2.0)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "item_id": item_id,
                    "store_code": "all",
                    "demand": float(index % 7) + offset,
                }
            )
    return pd.DataFrame(rows)


def test_sequence_windows_and_rolling_split_have_no_leakage() -> None:
    cutoffs = pd.to_datetime(["2024-01-28", "2024-02-11", "2024-02-25"])
    sequences = build_sequences(
        daily_panel(), cutoffs, lookback_days=14, horizon_days=7
    )
    train, validation = rolling_train_validation_split(
        sequences, "2024-02-25"
    )

    assert train.x.shape == (4, 14)
    assert validation.x.shape == (2, 14)
    assert (train.target_ends <= np.datetime64("2024-02-25")).all()
    assert (train.cutoffs < np.datetime64("2024-02-25")).all()
    assert (validation.cutoffs == np.datetime64("2024-02-25")).all()
    expected = daily_panel().query("item_id == 101").set_index("date").loc[
        "2024-02-26":"2024-03-03", "demand"
    ].sum()
    assert validation.y[0] == pytest.approx(expected)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same non-empty shape"):
        regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_lstm_training_smoke() -> None:
    pytest.importorskip("torch")
    cutoffs = pd.to_datetime(
        ["2024-01-28", "2024-02-11", "2024-02-25", "2024-03-10"]
    )
    sequences = build_sequences(
        daily_panel(), cutoffs, lookback_days=14, horizon_days=7
    )
    train, validation = rolling_train_validation_split(
        sequences, "2024-03-10"
    )
    _, prediction, metadata = train_lstm_baseline(
        train,
        validation,
        hidden_size=4,
        epochs=2,
        batch_size=4,
        random_seed=7,
    )

    assert prediction.shape == validation.y.shape
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()
    assert metadata["training_rows"] == len(train)
    assert len(metadata["training_loss"]) == 2
