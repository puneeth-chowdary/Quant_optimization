import pandas as pd

from src.inference.predict_trade_20d import predict_trade_20d
from src.inference.predict_regime import predict_regime_from_df
from src.optimization.covariance_pipeline import compute_covariance_from_stocks
from src.optimization.mpt_optimizer import build_portfolio
import yfinance as yf
import pandas as pd
from typing import Dict, List
# ======================================================
# BUILD index df(live)
# ======================================================
from typing import Dict
import pandas as pd
import yfinance as yf


def retrieve_nifty200_data(days: int = 300) -> pd.DataFrame:
    """
    Retrieves historical daily data for NIFTY 200 index.
    Returns: DataFrame with Date, Open, Close, index_ret
    """

    ticker = "^CNX200"
    print("Retrieving", ticker)

    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="column"
    )

    if df.empty:
        raise RuntimeError("Failed to retrieve NIFTY 200 data")

    # -------- flatten columns if MultiIndex --------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    out = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"]),
        "Close": df["Close"].astype(float),
        "Open": df["Open"].astype(float),
    })

    # Ensure chronological order
    out = out.sort_values("Date").reset_index(drop=True)

    # -------- ADD INDEX RETURN COLUMN --------
    out["index_ret"] = out["Close"].pct_change()

    return out
# ======================================================
# BUILD ALPHA DF (live trade model)
# ======================================================
def build_alpha_df(stock_dfs, index_df, regime_df):
    """
    Builds alpha dataframe for all stocks.
    """

    # 1️⃣ Get predictions for all stocks at once
    preds = predict_trade_20d(
        stock_dfs=stock_dfs,
        index_df=index_df,
        regime_df=regime_df
    )

    # 2️⃣ Build rows
    rows = []
    stock_list = list(stock_dfs.keys())

    for i, stock in enumerate(stock_list):

        df = stock_dfs[stock]

        rows.append({
            "Date": pd.to_datetime(df["Date"]).iloc[-1],
            "stock_id": stock,
            "pred_future_20d_return": preds[i]
        })

    return pd.DataFrame(rows)


# ======================================================
# BUILD MARKET REGIME DF (live)
# ======================================================
import pandas as pd


def build_regime_df(stock_dfs):
    """
    Build aggregated market regime dataframe from individual stock regime predictions.

    Returns
    -------
    regime_df : pd.DataFrame
        DatetimeIndex
        Columns:
            - state  (int: 0= Bull, 1=Bear, 2=Crisis, 3=Recovery)
            - Regime (string label)
    """

    feats = []

    # ------------------------------------------
    # Collect regime predictions per stock
    # ------------------------------------------
    for df in stock_dfs.values():
        reg = predict_regime_from_df(df)

        if "Date" not in reg.columns or "Regime" not in reg.columns:
            raise ValueError(
                "predict_regime_from_df must return columns ['Date','Regime']"
            )

        reg = reg[["Date", "Regime"]].copy()
        reg["Date"] = pd.to_datetime(reg["Date"])

        feats.append(reg)

    if not feats:
        raise ValueError("No regime predictions generated from stock_dfs.")

    # ------------------------------------------
    # Concatenate all stock regimes
    # ------------------------------------------
    df_all = pd.concat(feats, ignore_index=True)

    # ------------------------------------------
    # Map string regime → numeric state
    # ------------------------------------------
    state_map = {
        "Bull": 0,
        "Bear": 1,
        "Crisis": 2,
        "Recovery": 3
    }

    df_all["state"] = df_all["Regime"].map(state_map)

    if df_all["state"].isna().any():
        missing_vals = df_all[df_all["state"].isna()]["Regime"].unique()
        raise ValueError(
            f"Unknown regime labels detected: {missing_vals}"
        )

    # ------------------------------------------
    # Aggregate cross-sectionally by Date
    # (Mean + Round = smooth ordinal regime)
    # ------------------------------------------
    regime_df = (
        df_all
        .groupby("Date")["state"]
        .mean()
        .round()
        .astype(int)
        .to_frame()
    )

    # ------------------------------------------
    # Convert back to label for reporting
    # ------------------------------------------
    reverse_map = {v: k for k, v in state_map.items()}
    regime_df["Regime"] = regime_df["state"].map(reverse_map)

    # ------------------------------------------
    # Ensure proper time-series structure
    # ------------------------------------------
    regime_df.index = pd.to_datetime(regime_df.index)
    regime_df = regime_df.sort_index()

    return regime_df

# ======================================================
# FULL MPT PIPELINE
# ======================================================
def compute_portfolio_weights(
    stock_dfs,
    current_date
):
    """
    stock_dfs: dict {stock: DataFrame}
    current_date: portfolio date
    """

    current_date = pd.to_datetime(current_date)

    # 1️⃣ Covariance (vol + regime + returns)
    Sigma, cov_stocks = compute_covariance_from_stocks(
        stock_dfs=stock_dfs,
        current_date=current_date
    )

    # 2️⃣ Alpha (trade model)
   

    # 3️⃣ Regime DF
    regime_df = build_regime_df(stock_dfs)
    #regime_df = regime_df.set_index("Date")
  # 2️⃣ Alpha (trade model)
    index_df_retreived = retrieve_nifty200_data(days=300)
    alpha_df = build_alpha_df(stock_dfs,index_df_retreived,regime_df)

    # 4️⃣ Portfolio optimization
    weights = build_portfolio(
        current_date=current_date,
        alpha_df=alpha_df,
        regime_df=regime_df,
        Sigma=Sigma,
        cov_stocks=cov_stocks
    )

    return weights