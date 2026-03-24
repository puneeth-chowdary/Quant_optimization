import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ======================================================
# RISK MAP (regime → risk aversion)
# ======================================================
RISK_MAP = {
    0: 2.0,    # Bull
    1: 4.0,    # Recovery
    2: 8.0,    # Bear
    3: 15.0    # Crisis
}

# print(hf_chat("You are an expert financial analyst. what is MPT optimization?, so the results of the mpt are here JSW:0.200000 ,icici-bank:0.200000,larsen-toubro:0.173049,reliance-industries:0.148627,Orient green:0.090872,tata-consultancy-services:0.065405,infosys:0.050759, state-bank-of-india:0.049441, BIRLASOFT:0.015844 heritage: 0.006003,analysis this and explain to the user"))
# ======================================================
# MEAN–VARIANCE OPTIMIZER
# ======================================================
def mean_variance_opt(
    mu,
    Sigma,
    risk_aversion,
    long_only=True,
    max_weight=0.05
):
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    
    n = len(mu)
    max_weight = min(0.30, max(2.0 / n, 0.08))
    def objective(w):
        return -(w @ mu - 0.5 * risk_aversion * w @ Sigma @ w)

    def grad(w):
        return -(mu - risk_aversion * Sigma @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if long_only:
        bounds = [(0.0, max_weight)] * n
    else:
        bounds = [(-max_weight, max_weight)] * n

    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9}
    )

    if not result.success:
        raise ValueError(f"Optimizer failed: {result.message}")

    return result.x


# ======================================================
# PORTFOLIO ENGINE
# ======================================================
def build_portfolio(
    current_date,
    alpha_df,
    regime_df,
    Sigma,
    cov_stocks
):
    """
    Parameters
    ----------
    current_date : date
    alpha_df : DataFrame with Date, stock_id, pred_future_20d_return
    regime_df : DataFrame indexed by Date with column 'state'
    Sigma : covariance matrix (NxN)
    cov_stocks : list of stocks matching Sigma order
    """

    current_date = pd.to_datetime(current_date)

    # 1️⃣ Regime state
    if current_date not in regime_df.index:
        regime_state = regime_df.loc[:current_date].iloc[-1]["state"]
    else:
        regime_state = regime_df.loc[current_date, "state"]

    risk_aversion = RISK_MAP[int(regime_state)]

    # 2️⃣ Alpha slice
    alpha_slice = (
        alpha_df.loc[alpha_df["Date"] == current_date,
                     ["stock_id", "pred_future_20d_return"]]
        .dropna()
        .set_index("stock_id")
    )

    # 3️⃣ Align to covariance ordering
    mu = (
        alpha_slice
        .reindex(cov_stocks)
        .fillna(0.0)
        .iloc[:, 0]
        .values
    )

    # 4️⃣ Optimize
    weights = mean_variance_opt(
        mu=mu,
        Sigma=Sigma,
        risk_aversion=risk_aversion,
        long_only=True,
        max_weight=0.04
    )

    return pd.Series(weights, index=cov_stocks)