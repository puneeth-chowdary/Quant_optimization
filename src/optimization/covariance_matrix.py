import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import LedoitWolf

# ============================================
# CONFIG
# ============================================
DATA_FOLDER = Path("/Users/reddeppakollu/Datasets")
LOOKBACK = 252
RIDGE_EPS = 1e-4
N_REGIMES = 4


# ============================================
# LOAD RETURNS PANEL
# ============================================
def load_returns_panel(data_folder=DATA_FOLDER):

    dfs = []

    for file in data_folder.glob("*.csv"):
        if file.name == "NIFTY 200.csv":
            continue

        df = pd.read_csv(file)

        if "Close" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        stock = file.stem.replace("features_", "")
        df["return"] = df["Close"].pct_change()

        df = df[["Date", "return"]].dropna()
        df["stock"] = stock

        dfs.append(df)

    panel = pd.concat(dfs)

    pivot = panel.pivot_table(
        index="Date",
        columns="stock",
        values="return",
        aggfunc="mean"
    )

    return pivot.sort_index()


# ============================================
# LOAD REGIME
# ============================================
def load_regime_df(data_folder=DATA_FOLDER):

    regime_df = pd.read_csv(
        data_folder / "market_regimes_hmm_4state.csv",
        parse_dates=["Date"],
        index_col="Date"
    )

    return regime_df.sort_index()


# ============================================
# BUILD DYNAMIC COVARIANCE
# ============================================
def build_dynamic_covariance(
    current_date,
    vol_forecasts,
    returns_df,
    regime_df,
    lookback=252,
    ridge_eps=1e-4,
    n_regimes=4
):
    """
    Build covariance using:
    - live vol forecasts
    - live regime probabilities
    - live returns panel
    """

    current_date = pd.to_datetime(current_date)

    returns_df = returns_df.copy()
    regime_df = regime_df.copy()

    returns_df.index = pd.to_datetime(returns_df.index)
    regime_df.index = pd.to_datetime(regime_df.index)

    # 1️⃣ Rolling window
    window_returns = returns_df.loc[:current_date]

    # use as much as available up to lookback
    window_returns = window_returns.tail(lookback)

    available = len(window_returns)

    if available < 40:
        raise ValueError(
            f"Not enough history ({available} days). "
            "Need at least 40 aligned return observations."
        )
    window_regimes = regime_df.reindex(window_returns.index).ffill()

    # 2️⃣ Filter sparse assets
    min_obs = int(0.6 * available)
    window_returns = window_returns.dropna(axis=1, thresh=min_obs)

    window_returns = window_returns.fillna(0.0)

    stocks = window_returns.columns
    n_assets = len(stocks)

    # 3️⃣ Regime probabilities at date
    if current_date not in regime_df.index:
        regime_row = regime_df.loc[:current_date].iloc[-1]
    else:
        regime_row = regime_df.loc[current_date]

    probs = regime_row[[f"P_state_{k}" for k in range(n_regimes)]].values
    probs = probs / probs.sum()

    # 4️⃣ Regime covariances
    regime_covs = {}

    for k in range(n_regimes):

        mask = window_regimes["state"] == k
        regime_returns = window_returns[mask]

        if regime_returns.shape[0] < 15:
            continue

        lw = LedoitWolf()
        lw.fit(regime_returns.values)

        regime_covs[k] = lw.covariance_

    # 5️⃣ Blend
    if len(regime_covs) == 0:
        lw = LedoitWolf()
        lw.fit(window_returns.values)
        Sigma = lw.covariance_
    else:
        Sigma = np.zeros((n_assets, n_assets))
        for k, cov in regime_covs.items():
            Sigma += probs[k] * cov

    # 6️⃣ Replace diagonal with ML vol
    vol_vec = vol_forecasts.reindex(stocks).fillna(
        vol_forecasts.mean()
    ).values

    D = np.diag(vol_vec)

    std = np.sqrt(np.diag(Sigma))
    std[std == 0] = 1e-8

    corr = Sigma / np.outer(std, std)
    corr = np.nan_to_num(corr)

    Sigma = D @ corr @ D

    # 7️⃣ Ridge
    Sigma += ridge_eps * np.eye(n_assets)

    return Sigma, list(stocks)