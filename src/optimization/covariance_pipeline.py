import pandas as pd

from src.inference.predict_vol import predict_vol_from_df
from src.inference.predict_regime import predict_regime_from_df
from src.optimization.covariance_matrix import build_dynamic_covariance


# ============================================
# RETURNS PANEL
# ============================================
def build_returns_panel(stock_dfs):
    returns = []

    for stock, df in stock_dfs.items():

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        df["ret"] = df["Close"].pct_change()
        df = df[["Date", "ret"]].dropna()
        df["stock"] = stock

        returns.append(df)

    panel = pd.concat(returns)

    returns_df = panel.pivot_table(
        index="Date",
        columns="stock",
        values="ret",
        aggfunc="mean"
    )

    return returns_df.sort_index()


# ============================================
# VOL VECTOR
# ============================================
def build_vol_vector(stock_dfs):
    vols = {}

    for stock, df in stock_dfs.items():
        vol = predict_vol_from_df(df, stock_id=stock)
        vols[stock] = vol[-1]

    return pd.Series(vols)


# ============================================
# MARKET REGIME DF
# ============================================
def build_market_regime_df(stock_dfs):
    feats = []

    for df in stock_dfs.values():
        reg = predict_regime_from_df(df)
        feats.append(reg)

    combined=pd.concat(feats)
    numveric_cols=["state","P_state_0","P_state_1","P_state_2","P_state_3"]
    market_df=(
        combined.groupby("Date")[numveric_cols]
        .mean()
        .sort_index()
    )
    return market_df


# ============================================
# FULL COVARIANCE PIPELINE
# ============================================
def compute_covariance_from_stocks(
    stock_dfs,
    current_date
):
    """
    stock_dfs: dict {stock: DataFrame}
    current_date: date for covariance
    """

    returns_df = build_returns_panel(stock_dfs)
    vol_vec = build_vol_vector(stock_dfs)
    regime_df = build_market_regime_df(stock_dfs)

    Sigma, stocks = build_dynamic_covariance(
        current_date=current_date,
        vol_forecasts=vol_vec,
        returns_df=returns_df,
        regime_df=regime_df
    )

    return Sigma, stocks
