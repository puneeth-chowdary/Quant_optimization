import joblib
import numpy as np
import pandas as pd

from src.features.vol_features import build_vol_features

MODEL_PATH = "artifacts/vol_weights.pkl"
FEATURE_PATH = "artifacts/vol_feature_list.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def load_feature_list():
    return joblib.load(FEATURE_PATH)


# ======================================================
# PREDICT FROM DATAFRAME (LIVE / API)
# ======================================================
def predict_vol_from_df(df: pd.DataFrame, stock_id: str):
    """
    Predict volatility from raw OHLC dataframe.
    Requires columns: Date, Close
    """

    raw = df.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])

    feat = build_vol_features(raw, stock_id)
    feat = feat.dropna(subset=["lag_log_var_1"])

    model = load_model()
    feature_list = load_feature_list()

    X = feat.reindex(columns=feature_list, fill_value=0.0)

    preds_log = model.predict(X)
    preds_var = np.exp(preds_log)
    preds_vol = np.sqrt(preds_var)

    return preds_vol


# ======================================================
# PREDICT FROM CSV (legacy)
# ======================================================
def predict_vol_from_csv(csv_path: str, stock_id: str):
    raw = pd.read_csv(csv_path)
    return predict_vol_from_df(raw, stock_id)