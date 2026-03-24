import pandas as pd
import numpy as np

HORIZON = 20
EPSILON = 1e-8
RETURN_CLIP = 0.5

BASE_FEATURES = [
    "return_t-1","return_t-2","return_t-5",
    "squared_return_t-1","squared_return_t-5",
    "rolling_vol_5","rolling_vol_20","rolling_vol_60",
    "volume_change","abs_return_t-1","volatility_zscore_60",
]

def build_vol_features(df: pd.DataFrame, stock_id: str):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["daily_return"] = df["Close"].pct_change()
    df.loc[np.abs(df["daily_return"]) > RETURN_CLIP, "daily_return"] = np.nan

    future_sq_returns = (
        df["daily_return"].shift(-1).pow(2).rolling(HORIZON).sum()
    )

    df["realized_var_20"] = future_sq_returns + EPSILON
    df["log_realized_var_20"] = np.log(df["realized_var_20"])
    df["lag_log_var_1"] = df["log_realized_var_20"].shift(1)

    return df