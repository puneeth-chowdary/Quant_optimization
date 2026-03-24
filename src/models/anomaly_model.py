from sklearn.ensemble import IsolationForest

def build_anomaly_model(
    n_estimators=150,
    contamination=0.0075,
    random_state=42
):
    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=1  # avoid nested parallelism
    )