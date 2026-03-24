import joblib
import pandas as pd

from src.features.regime_features import build_regime_features, REGIME_FEATURES

MODEL_PATH = "artifacts/regime_hmm.pkl"
SCALER_PATH = "artifacts/regime_scaler.pkl"
LABELS_PATH = "artifacts/regime_labels.pkl"

def load_regime_model():
    return joblib.load(MODEL_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)

def load_labels():
    return joblib.load(LABELS_PATH)

def predict_regime_from_df(df: pd.DataFrame):

    feat = build_regime_features(df)
    X = feat[REGIME_FEATURES].values

    scaler = load_scaler()
    hmm_model = load_regime_model()
    labels = load_labels()

    X_scaled = scaler.transform(X)

    states = hmm_model.predict(X_scaled)
    probs  = hmm_model.predict_proba(X_scaled)

    regimes = [labels[s] for s in states]

    out = feat.copy()
    out["state"] = states
    out["Regime"] = regimes

    for k in range(hmm_model.n_components):
        out[f"P_state_{k}"] = probs[:, k]

    return out