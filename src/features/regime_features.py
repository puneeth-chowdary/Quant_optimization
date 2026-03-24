import numpy as np
import pandas as pd

REGIME_FEATURES = [
    "rolling_vol_20",
    "rolling_vol_60",
    "volatility_zscore_60",
    "lag_log_var_1",
    "return_t-1",
    "return_t-5",
    "abs_return_t-1",
    "squared_return_t-1",
]

def build_regime_features(df: pd.DataFrame):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    close = df["Close"]
    log_ret = np.log(close / close.shift(1))

    if "return_t-1" not in df.columns:
        df["return_t-1"] = close.pct_change()

    if "return_t-5" not in df.columns:
        df["return_t-5"] = close.pct_change(5)

    if "abs_return_t-1" not in df.columns:
        df["abs_return_t-1"] = df["return_t-1"].abs()

    if "squared_return_t-1" not in df.columns:
        df["squared_return_t-1"] = df["return_t-1"] ** 2

    if "rolling_vol_20" not in df.columns:
        df["rolling_vol_20"] = log_ret.rolling(20).std()

    if "rolling_vol_60" not in df.columns:
        df["rolling_vol_60"] = log_ret.rolling(60).std()

    if "volatility_zscore_60" not in df.columns:
        vol = df["rolling_vol_60"]
        df["volatility_zscore_60"] = (
            (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
        )

    if "lag_log_var_1" not in df.columns:
        rv = log_ret.pow(2).rolling(20).sum()
        log_rv = np.log(rv + 1e-8)
        df["lag_log_var_1"] = log_rv.shift(1)

    return df[["Date"] + REGIME_FEATURES].dropna()