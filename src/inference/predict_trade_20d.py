import joblib
import numpy as np
import pandas as pd

from src.features.trade_features_20d import build_trade_features_20d

MODEL_PATH   = "artifacts/trade_model_20d.pkl"
FEAT_PATH    = "artifacts/trade_features_20d.pkl"
REGIME_PATH  = "artifacts/trade_regime_encoder.pkl"


def load_trade_model():
    return joblib.load(MODEL_PATH)

def load_features():
    return joblib.load(FEAT_PATH)

def load_regime_encoder():
    return joblib.load(REGIME_PATH)


def predict_trade_20d(stock_dfs, index_df, regime_df):
    """
    stock_dfs: dict {stock_id: DataFrame}
    Returns: np.array of predictions (N stocks)
    """

    model = load_trade_model()
    FEATURES = load_features()
    regime_encoder = load_regime_encoder()

    rows = []

    for stock_id, df in stock_dfs.items():

        feat = build_trade_features_20d(df, index_df, regime_df)

        feat = feat.dropna(subset=FEATURES[:-1])  # exclude regime
        if len(feat) == 0:
            rows.append(np.nan)
            continue

        last = feat.iloc[-1].copy()

        # encode regime
        regime_val = last["Regime"]
        inv_map = {v:k for k,v in regime_encoder.items()}
        last["Regime"] = inv_map.get(regime_val, 0)

        X = last[FEATURES].values.reshape(1,-1)
        pred = model.predict(X)[0]

        rows.append(pred)

    return np.array(rows)