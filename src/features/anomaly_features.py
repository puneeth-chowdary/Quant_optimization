import pandas as pd

ANOM_FEATURES = [
    "correlation_zscore",
    "spread_from_rolling_mean",
    "volatility_zscore_60",
    "return_Zscore_20",
    "volume_zscore_20",
]

def prepare_anomaly_dataframe(stock_df: pd.DataFrame, regime_df: pd.DataFrame, stock_id: str):
    df = stock_df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["stock_id"] = stock_id

    df = df[["Date", "stock_id"] + ANOM_FEATURES]

    df = df.merge(regime_df, on="Date", how="left")
    df = df.dropna(subset=["Regime"] + ANOM_FEATURES)

    df = df.sort_values(["stock_id", "Date"]).reset_index(drop=True)

    return df