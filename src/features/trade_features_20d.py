import numpy as np
import pandas as pd


# ======================================================
# FEATURE LIST
# ======================================================

BASE_FEATURES = [
    "rolling_vol_20","rolling_vol_5","rolling_vol_60",
    "volatility_zscore_60","abs_return_t-1","return_t-5",
    "return_Zscore_20","spread_from_rolling_mean",
    "drawdown_20","drawdown_60","volume_zscore_20",
    "volume_change","rolling_correlation_with_index",
    "correlation_zscore",
]

NEW_FEATURES = [
    "momentum_5","momentum_20","momentum_60",
    "trend_slope_20","trend_slope_60",
    "relative_strength_20","idiosyncratic_vol_20",
]

ALL_FEATURES = BASE_FEATURES + NEW_FEATURES


# ======================================================
# HELPER: Rolling Log-Slope
# ======================================================

def rolling_slope(series, window):
    y = np.log(series.values)
    x = np.arange(window)
    slopes = np.full(len(series), np.nan)

    for i in range(window, len(series)):
        yy = y[i-window:i]
        if np.any(np.isnan(yy)):
            continue
        slopes[i] = np.polyfit(x, yy, 1)[0]

    return slopes


# ======================================================
# MAIN FEATURE BUILDER
# ======================================================

def build_trade_features_20d(df, index_df, regime_df, regime_col="Regime"):

    df = df.copy()

    # -----------------------------
    # Date handling
    # -----------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    index_df = index_df.copy()
    index_df["Date"] = pd.to_datetime(index_df["Date"])

    regime_df = regime_df.copy()
    #regime_df["Date"] = pd.to_datetime(regime_df["Date"])

    # -----------------------------
    # Merge index
    # -----------------------------
    df = df.merge(
        index_df[["Date","Close","index_ret"]],
        on="Date",
        how="left",
        suffixes=("","_index")
    )

    # -----------------------------
    # Returns
    # -----------------------------
    df["ret"] = df["Close"].pct_change()

    # =====================================================
    # BASE FEATURES
    # =====================================================

    # Rolling Volatility
    df["rolling_vol_5"]  = df["ret"].rolling(5).std()
    df["rolling_vol_20"] = df["ret"].rolling(20).std()
    df["rolling_vol_60"] = df["ret"].rolling(60).std()

    # Volatility Z-score (60)
    vol_mean_60 = df["rolling_vol_60"].rolling(60).mean()
    vol_std_60  = df["rolling_vol_60"].rolling(60).std()
    df["volatility_zscore_60"] = (
        (df["rolling_vol_60"] - vol_mean_60) / vol_std_60
    )

    # Absolute Return t-1
    df["abs_return_t-1"] = df["ret"].shift(1).abs()

    # Return t-5
    df["return_t-5"] = df["ret"].shift(5)

    # Return Z-score (20)
    ret_mean_20 = df["ret"].rolling(20).mean()
    ret_std_20  = df["ret"].rolling(20).std()
    df["return_Zscore_20"] = (df["ret"] - ret_mean_20) / ret_std_20

    # Spread from Rolling Mean
    rolling_mean_20 = df["Close"].rolling(20).mean()
    df["spread_from_rolling_mean"] = (
        (df["Close"] - rolling_mean_20) / rolling_mean_20
    )

    # Drawdowns
    rolling_max_20 = df["Close"].rolling(20).max()
    df["drawdown_20"] = (df["Close"] - rolling_max_20) / rolling_max_20

    rolling_max_60 = df["Close"].rolling(60).max()
    df["drawdown_60"] = (df["Close"] - rolling_max_60) / rolling_max_60

    # Volume Features
    if "Volume" in df.columns:
        vol_mean_20 = df["Volume"].rolling(20).mean()
        vol_std_20  = df["Volume"].rolling(20).std()

        df["volume_zscore_20"] = (
            (df["Volume"] - vol_mean_20) / vol_std_20
        )
        df["volume_change"] = df["Volume"].pct_change()
    else:
        df["volume_zscore_20"] = np.nan
        df["volume_change"] = np.nan

    # Rolling Correlation With Index
    df["rolling_correlation_with_index"] = (
        df["ret"].rolling(20).corr(df["index_ret"])
    )

    # Correlation Z-score
    corr_mean_60 = df["rolling_correlation_with_index"].rolling(60).mean()
    corr_std_60  = df["rolling_correlation_with_index"].rolling(60).std()

    df["correlation_zscore"] = (
        (df["rolling_correlation_with_index"] - corr_mean_60)
        / corr_std_60
    )

    # =====================================================
    # NEW FEATURES
    # =====================================================

    df["momentum_5"]  = df["Close"]/df["Close"].shift(5) - 1
    df["momentum_20"] = df["Close"]/df["Close"].shift(20) - 1
    df["momentum_60"] = df["Close"]/df["Close"].shift(60) - 1

    df["trend_slope_20"] = rolling_slope(df["Close"],20)
    df["trend_slope_60"] = rolling_slope(df["Close"],60)

    df["relative_strength_20"] = df["momentum_20"] - (
        df["Close_index"]/df["Close_index"].shift(20) - 1
    )

    cov = df["ret"].rolling(20).cov(df["index_ret"])
    var = df["index_ret"].rolling(20).var()
    beta = cov / var

    residual = df["ret"] - beta * df["index_ret"]
    df["idiosyncratic_vol_20"] = residual.rolling(20).std()

    # =====================================================
    # Merge Regime
    # =====================================================

    if regime_col not in regime_df.columns:
        raise ValueError(
            f"{regime_col} not found in regime_df. "
            f"Available columns: {regime_df.columns}"
        )

    # Merge regime safely
    regime_df_merge = regime_df.reset_index()

    if regime_col not in regime_df_merge.columns:
        raise ValueError(
            f"{regime_col} not found. Available columns: {regime_df_merge.columns}"
        )

    df = df.merge(
        regime_df_merge[["Date", regime_col]],
        on="Date",
        how="left"  
    )

    return df