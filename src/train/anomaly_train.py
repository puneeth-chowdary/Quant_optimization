import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import StandardScaler

from src.models.anomaly_model import build_anomaly_model
from src.features.anomaly_features import ANOM_FEATURES

DATA_FOLDER = Path("/Users/reddeppakollu/Datasets")

CONTAM = 0.0075
N_ESTIMATORS = 150
N_CORES = multiprocessing.cpu_count() - 1

# =========================
# LOAD REGIMES
# =========================
regime_path = DATA_FOLDER / "market_regimes_hmm_4state.csv"

market_regimes = pd.read_csv(
    regime_path,
    parse_dates=["Date"],
    usecols=["Date", "Regime"]
)

# =========================
# LOAD STOCK DATA
# =========================
dfs = []

for file in DATA_FOLDER.glob("*.csv"):

    if file.name in ["NIFTY 200.csv", "market_regimes_hmm_4state.csv"]:
        continue

    df = pd.read_csv(file)

    if not all(col in df.columns for col in ANOM_FEATURES):
        continue

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    stock_id = file.stem.replace("features_", "")
    df["stock_id"] = stock_id

    dfs.append(df[["Date", "stock_id"] + ANOM_FEATURES])

data = pd.concat(dfs, ignore_index=True)

# =========================
# MERGE REGIMES
# =========================
data = data.merge(market_regimes, on="Date", how="left")
data.dropna(subset=["Regime"] + ANOM_FEATURES, inplace=True)
data.sort_values(["stock_id", "Date"], inplace=True)
data.reset_index(drop=True, inplace=True)

# =========================
# PER-STOCK PROCESS
# =========================
def process_stock(stock, sdf):

    sdf = sdf.copy()
    n_total = len(sdf)

    scaler = StandardScaler()
    sdf[ANOM_FEATURES] = scaler.fit_transform(sdf[ANOM_FEATURES])
    sdf[ANOM_FEATURES] = sdf[ANOM_FEATURES].astype(np.float32)

    anomaly_scores = np.full(n_total, np.nan, dtype=np.float32)

    for regime, rdf in sdf.groupby("Regime", sort=False):

        if len(rdf) < 100:
            continue

        pos_idx = sdf.index.get_indexer(rdf.index)
        X = rdf[ANOM_FEATURES].values

        iso = build_anomaly_model(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAM
        )

        iso.fit(X)
        scores = -iso.score_samples(X)

        anomaly_scores[pos_idx] = scores

    sdf["anomaly_score"] = anomaly_scores
    return sdf

# =========================
# PARALLEL
# =========================
grouped = list(data.groupby("stock_id", sort=False))

results = Parallel(n_jobs=N_CORES)(
    delayed(process_stock)(stock, sdf)
    for stock, sdf in grouped
)

result = pd.concat(results, ignore_index=True)

# =========================
# Z-SCORE
# =========================
result["anomaly_z"] = (
    result.groupby("stock_id")["anomaly_score"]
    .transform(lambda x: (x - x.mean()) / x.std())
)

result["anomaly_flag"] = result["anomaly_z"] > 3.0

# =========================
# SAVE
# =========================
out_path = DATA_FOLDER / "stock_regime_anomaly_scores.csv"
result.to_csv(out_path, index=False)

print(f"Saved: {out_path}")

if __name__ == "__main__":
    pass