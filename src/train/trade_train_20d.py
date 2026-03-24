import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.models.trade_model_20d import build_trade_model_20d
from src.features.trade_features_20d import ALL_FEATURES, build_trade_features_20d

DATA_FOLDER = Path("/Users/reddeppakollu/Datasets")
INDEX_FILE = "NIFTY 200.csv"
REGIME_FILE = "market_regimes_hmm_4state.csv"
REGIME_COL = "Regime"

# LOAD INDEX
index_df = pd.read_csv(DATA_FOLDER/INDEX_FILE)
index_df["Date"] = pd.to_datetime(index_df["Date"])
index_df = index_df.sort_values("Date")
index_df["index_ret"] = index_df["Close"].pct_change()

# LOAD REGIME
regime_df = pd.read_csv(DATA_FOLDER/REGIME_FILE)
regime_df["Date"] = pd.to_datetime(regime_df["Date"])

dfs = []

required_cols = ["Date","Close"] + ALL_FEATURES

for file in DATA_FOLDER.glob("*.csv"):

    if file.name in [INDEX_FILE, REGIME_FILE]:
        continue

    df = pd.read_csv(file)

    if "Close" not in df.columns:
        continue

    stock_id = file.stem.replace("features_","")

    df = build_trade_features_20d(df, index_df, regime_df, REGIME_COL)

    df["future_20d_return"] = df["Close"].shift(-20)/df["Close"] - 1
    df["stock_id"] = stock_id

    keep = ["Date","stock_id","future_20d_return",REGIME_COL] + ALL_FEATURES
    dfs.append(df[keep])

data = pd.concat(dfs, ignore_index=True)

# CLEAN
data[ALL_FEATURES] = data[ALL_FEATURES].replace([np.inf,-np.inf],np.nan)
data = data.dropna(subset=ALL_FEATURES+["future_20d_return",REGIME_COL])

# ENCODE REGIME
data[REGIME_COL] = data[REGIME_COL].astype("category")
regime_encoder = dict(enumerate(data[REGIME_COL].cat.categories))
data[REGIME_COL] = data[REGIME_COL].cat.codes

FEATURES = ALL_FEATURES + [REGIME_COL]

train_mask = data["Date"] < "2015-01-01"
X_train = data.loc[train_mask, FEATURES].values
y_train = data.loc[train_mask,"future_20d_return"].values

model = build_trade_model_20d()
model.fit(X_train, y_train)

# SAVE
joblib.dump(model, "artifacts/trade_model_20d.pkl")
joblib.dump(FEATURES, "artifacts/trade_features_20d.pkl")
joblib.dump(regime_encoder, "artifacts/trade_regime_encoder.pkl")

print("Saved trade_model_20d artifacts")