import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.anomaly_model import build_anomaly_model
from src.features.anomaly_features import ANOM_FEATURES

def predict_anomaly(df: pd.DataFrame):
    df = df.copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[ANOM_FEATURES])

    iso = build_anomaly_model()
    iso.fit(X)

    scores = -iso.score_samples(X)

    df["anomaly_score"] = scores
    return df