import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.features.regime_features import build_regime_features, REGIME_FEATURES
from src.models.regime_model import build_hmm_model

DATA_FOLDER = Path("/Users/reddeppakollu/Datasets")

dfs = []

# LOAD STOCK FILES
for file in DATA_FOLDER.glob("*.csv"):

    if file.name == "NIFTY 200.csv":
        continue

    df = pd.read_csv(file)

    if "Close" not in df.columns:
        continue

    feat = build_regime_features(df)

    if len(feat) > 50:
        dfs.append(feat)

if len(dfs) == 0:
    raise ValueError("No valid stock files with regime features found")

data = pd.concat(dfs, ignore_index=True)

# MARKET AGGREGATION
market_df = (
    data.groupby("Date")[REGIME_FEATURES]
    .mean()
    .sort_index()
)

# SPLIT
train_mask = market_df.index < "2015-01-01"
test_mask  = market_df.index >= "2015-01-01"

X_train = market_df.loc[train_mask].values
X_test  = market_df.loc[test_mask].values

# SCALE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================
# HMM WITH KMEANS INIT
# =========================
n_states = 4

kmeans = KMeans(n_clusters=n_states, random_state=42).fit(X_train_scaled)

hmm_model = build_hmm_model(n_states)
hmm_model.means_ = kmeans.cluster_centers_

hmm_model.fit(X_train_scaled)

# STATES
train_states = hmm_model.predict(X_train_scaled)
test_states  = hmm_model.predict(X_test_scaled)

# diagnostic
state_counts = np.bincount(train_states.astype(int), minlength=n_states)
print("\nState counts (train):", state_counts)

market_df.loc[train_mask, "state"] = train_states
market_df.loc[test_mask,  "state"] = test_states

# PROBABILITIES
train_probs = hmm_model.predict_proba(X_train_scaled)
test_probs  = hmm_model.predict_proba(X_test_scaled)

for k in range(n_states):
    market_df.loc[train_mask, f"P_state_{k}"] = train_probs[:, k]
    market_df.loc[test_mask,  f"P_state_{k}"] = test_probs[:, k]

# =========================
# IMPROVED LABELING
# =========================
state_means = pd.DataFrame(
    scaler.inverse_transform(hmm_model.means_),
    columns=REGIME_FEATURES
)

labels = {}

# Crisis = highest vol
crisis_state = state_means["rolling_vol_20"].idxmax()
labels[crisis_state] = "Crisis"

remaining = state_means.drop(index=[crisis_state])

# Bull = highest return
bull_state = remaining["return_t-1"].idxmax()
labels[bull_state] = "Bull"

remaining = remaining.drop(index=[bull_state])

# Bear = lowest return
bear_state = remaining["return_t-1"].idxmin()
labels[bear_state] = "Bear"

remaining = remaining.drop(index=[bear_state])

# Recovery = last
recovery_state = remaining.index[0]
labels[recovery_state] = "Recovery"

market_df["Regime"] = market_df["state"].map(labels)

print("\nSTATE MEANS:\n", state_means[["rolling_vol_20","return_t-1"]])
print("\nSTATE LABELS:", labels)

# SAVE ARTIFACTS
joblib.dump(hmm_model, "artifacts/regime_hmm.pkl")
joblib.dump(scaler, "artifacts/regime_scaler.pkl")
joblib.dump(labels, "artifacts/regime_labels.pkl")

# SAVE CSV
out = DATA_FOLDER / "market_regimes_hmm_4state.csv"
market_df.to_csv(out)

print("Saved:", out)

if __name__ == "__main__":
    pass