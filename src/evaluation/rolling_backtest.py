import pandas as pd


def rolling_split(df, date_col, train_days, horizon_days, step_days):
    dates = sorted(df[date_col].unique())
    splits = []

    start = train_days
    while start + horizon_days <= len(dates):
        train_dates = dates[:start]
        valid_dates = dates[start:start + horizon_days]

        splits.append((
            df[df[date_col].isin(train_dates)],
            df[df[date_col].isin(valid_dates)]
        ))

        start += step_days

    return splits
