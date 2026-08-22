"""Leakage-safe sequence construction and an optional PyTorch LSTM baseline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDataset:
    """Daily histories and horizon totals with their temporal audit fields."""

    x: np.ndarray
    y: np.ndarray
    cutoffs: np.ndarray
    target_ends: np.ndarray
    item_ids: np.ndarray
    store_codes: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "SequenceDataset":
        selected = np.asarray(mask, dtype=bool)
        return SequenceDataset(
            x=self.x[selected],
            y=self.y[selected],
            cutoffs=self.cutoffs[selected],
            target_ends=self.target_ends[selected],
            item_ids=self.item_ids[selected],
            store_codes=self.store_codes[selected],
        )


def build_sequences(
    daily: pd.DataFrame,
    cutoffs: list[pd.Timestamp] | pd.DatetimeIndex,
    *,
    lookback_days: int,
    horizon_days: int,
) -> SequenceDataset:
    """Build fixed windows using only history through each explicit cutoff.

    A row is emitted only when the complete lookback and future label horizon
    are present. Reindexing makes missing calendar days explicit zero-demand
    days, matching the repository's DuckDB daily-panel convention.
    """
    required = {"date", "item_id", "store_code", "demand"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily data is missing columns: {sorted(missing)}")
    if lookback_days <= 0 or horizon_days <= 0:
        raise ValueError("lookback_days and horizon_days must be positive")

    requested_cutoffs = pd.DatetimeIndex(pd.to_datetime(cutoffs)).normalize()
    if requested_cutoffs.empty:
        raise ValueError("at least one cutoff is required")

    histories: list[np.ndarray] = []
    targets: list[float] = []
    sample_cutoffs: list[np.datetime64] = []
    target_ends: list[np.datetime64] = []
    item_ids: list[Any] = []
    store_codes: list[str] = []

    prepared = daily.loc[:, list(required)].copy()
    prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
    prepared["demand"] = pd.to_numeric(
        prepared["demand"], errors="raise"
    ).clip(lower=0)

    grouped = prepared.groupby(
        ["item_id", "store_code"], observed=True, sort=True
    )
    for (item_id, store_code), group in grouped:
        series = (
            group.groupby("date", observed=True)["demand"]
            .sum()
            .sort_index()
        )
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
        series = series.reindex(full_index, fill_value=0.0)
        for cutoff in requested_cutoffs:
            history_start = cutoff - pd.Timedelta(days=lookback_days - 1)
            target_end = cutoff + pd.Timedelta(days=horizon_days)
            if history_start < series.index.min() or target_end > series.index.max():
                continue
            history = series.loc[history_start:cutoff].to_numpy(dtype=np.float32)
            future = series.loc[
                cutoff + pd.Timedelta(days=1) : target_end
            ].to_numpy(dtype=np.float32)
            if len(history) != lookback_days or len(future) != horizon_days:
                continue
            histories.append(history)
            targets.append(float(future.sum()))
            sample_cutoffs.append(cutoff.to_datetime64())
            target_ends.append(target_end.to_datetime64())
            item_ids.append(item_id)
            store_codes.append(str(store_code))

    if not histories:
        raise ValueError("no complete sequences could be constructed")
    return SequenceDataset(
        x=np.stack(histories).astype(np.float32, copy=False),
        y=np.asarray(targets, dtype=np.float32),
        cutoffs=np.asarray(sample_cutoffs, dtype="datetime64[D]"),
        target_ends=np.asarray(target_ends, dtype="datetime64[D]"),
        item_ids=np.asarray(item_ids),
        store_codes=np.asarray(store_codes, dtype=object),
    )


def rolling_train_validation_split(
    sequences: SequenceDataset,
    validation_cutoff: pd.Timestamp | str,
) -> tuple[SequenceDataset, SequenceDataset]:
    """Split so every training label is observable at validation time."""
    boundary = np.datetime64(pd.Timestamp(validation_cutoff).normalize(), "D")
    train_mask = (sequences.cutoffs < boundary) & (
        sequences.target_ends <= boundary
    )
    validation_mask = sequences.cutoffs == boundary
    train = sequences.subset(train_mask)
    validation = sequences.subset(validation_mask)
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("rolling split produced an empty train or validation set")
    return train, validation


def regression_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    if truth.shape != forecast.shape or truth.size == 0:
        raise ValueError("actual and prediction must have the same non-empty shape")
    absolute_error = np.abs(truth - forecast)
    return {
        "mae": float(absolute_error.mean()),
        "wape": float(absolute_error.sum() / max(np.abs(truth).sum(), 1e-12)),
    }


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is optional; install requirements-deep-learning.txt"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def train_lstm_baseline(
    train: SequenceDataset,
    validation: SequenceDataset,
    *,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Train a compact global univariate LSTM and predict horizon totals."""
    if min(hidden_size, num_layers, epochs, batch_size) <= 0:
        raise ValueError("model sizes, epochs and batch_size must be positive")
    if not 0 <= dropout < 1 or learning_rate <= 0:
        raise ValueError("dropout and learning_rate are outside valid ranges")
    if train.x.ndim != 2 or validation.x.shape[1:] != train.x.shape[1:]:
        raise ValueError("train and validation windows must share a 2D shape")

    torch, nn, DataLoader, TensorDataset = _require_torch()
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.set_num_threads(1)

    class DemandLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            head_size = max(hidden_size // 2, 4)
            self.output = nn.Sequential(
                nn.Linear(hidden_size, head_size),
                nn.ReLU(),
                nn.Linear(head_size, 1),
                nn.Softplus(),
            )

        def forward(self, values: Any) -> Any:
            encoded, _ = self.recurrent(values)
            return self.output(encoded[:, -1, :]).squeeze(-1)

    train_scale = np.maximum(train.x.mean(axis=1), 1.0).astype(np.float32)
    validation_scale = np.maximum(
        validation.x.mean(axis=1), 1.0
    ).astype(np.float32)
    log_train_x = np.log1p(train.x / train_scale[:, None]).astype(np.float32)
    input_mean = float(log_train_x.mean())
    input_std = max(float(log_train_x.std()), 1e-6)

    def transform_x(values: np.ndarray, scale: np.ndarray) -> Any:
        logged = np.log1p(values / scale[:, None]).astype(np.float32)
        normalized = (logged - input_mean) / input_std
        return torch.from_numpy(normalized[:, :, None])

    train_x = transform_x(train.x, train_scale)
    train_y = torch.from_numpy(
        np.log1p(train.y / train_scale).astype(np.float32)
    )
    generator = torch.Generator().manual_seed(random_seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train)),
        shuffle=True,
        generator=generator,
    )
    model = DemandLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.SmoothL1Loss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_rows = len(batch_y)
            total_loss += float(loss.detach()) * batch_rows
            seen += batch_rows
        losses.append(total_loss / max(seen, 1))

    model.eval()
    with torch.no_grad():
        predicted_log = model(
            transform_x(validation.x, validation_scale)
        ).cpu().numpy()
    prediction = np.clip(
        np.expm1(predicted_log) * validation_scale,
        0.0,
        None,
    )
    metadata: dict[str, Any] = {
        "architecture": "global univariate LSTM",
        "lookback_days": int(train.x.shape[1]),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout if num_layers > 1 else 0.0,
        "epochs": epochs,
        "batch_size": min(batch_size, len(train)),
        "learning_rate": learning_rate,
        "random_seed": random_seed,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "input_log_mean": input_mean,
        "input_log_std": input_std,
        "window_scaling": "divide by max(history mean, 1.0)",
        "target_scaling": "log1p(horizon total / window scale)",
        "training_loss": losses,
    }
    return model, prediction.astype(np.float32), metadata
