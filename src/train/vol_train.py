import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.vol_model import build_vol_model

# -----------------------------
# CONFIG
# -----------------------------
HORIZON = 20
EPSILON = 1e-8
RETURN_CLIP = 0.5

DATA_FOLDER = Path("/Users/reddeppakollu/Datasets")

BASE_FEATURES = [
    "return_t-1","return_t-2","return_t-5",
    "squared_return_t-1","squared_return_t-5",
    "rolling_vol_5","rolling_vol_20","rolling_vol_60",
    "volume_change","abs_return_t-1","volatility_zscore_60",
]

dfs = []

# -----------------------------
# LOAD DATA
# -----------------------------
for file in DATA_FOLDER.glob("*.csv"):

    if file.name == "NIFTY 200.csv":
        continue

    df = pd.read_csv(file)

    if "Close" not in df.columns:
        continue

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    stock_id = file.stem.replace("features_", "")
    df["stock_id"] = stock_id

    df["daily_return"] = df["Close"].pct_change()
    df.loc[np.abs(df["daily_return"]) > RETURN_CLIP, "daily_return"] = np.nan

    future_sq_returns = (
        df["daily_return"].shift(-1).pow(2).rolling(HORIZON).sum()
    )

    df["realized_var_20"] = future_sq_returns + EPSILON
    df["log_realized_var_20"] = np.log(df["realized_var_20"])
    df["lag_log_var_1"] = df["log_realized_var_20"].shift(1)

    df = df.dropna(subset=[
        "realized_var_20","log_realized_var_20","lag_log_var_1"
    ])

    dfs.append(df)

if len(dfs) == 0:
    raise ValueError("No valid stock files found")

data = pd.concat(dfs, ignore_index=True)

# -----------------------------
# FEATURES
# -----------------------------
data = pd.get_dummies(data, columns=["stock_id"], drop_first=True)
DUMMY_COLS = [c for c in data.columns if c.startswith("stock_id_")]
FEATURES = BASE_FEATURES + ["lag_log_var_1"] + DUMMY_COLS

cols = ["Date","realized_var_20","log_realized_var_20"] + FEATURES
data = data[cols].replace([np.inf,-np.inf],np.nan).dropna()
joblib.dump(FEATURES, "artifacts/vol_feature_list.pkl")
# -----------------------------
# SPLIT
# -----------------------------
train_mask = data["Date"] < "2015-01-01"
val_mask   = (data["Date"] >= "2015-01-01") & (data["Date"] < "2018-01-01")
test_mask  = data["Date"] >= "2018-01-01"

X_train = data.loc[train_mask, FEATURES]
y_train = data.loc[train_mask, "log_realized_var_20"]
w_train = data.loc[train_mask, "realized_var_20"]

X_val = data.loc[val_mask, FEATURES]
y_val = data.loc[val_mask, "log_realized_var_20"]
y_val_var = data.loc[val_mask, "realized_var_20"]

X_test = data.loc[test_mask, FEATURES]
y_test = data.loc[test_mask, "log_realized_var_20"]
y_test_var = data.loc[test_mask, "realized_var_20"]

print("Train:",len(X_train),"Val:",len(X_val),"Test:",len(X_test))

# -----------------------------
# TRAIN
# -----------------------------
model = build_vol_model()

model.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_val,y_val)],
    callbacks=[]
)

# -----------------------------
# SAVE WEIGHTS
# -----------------------------
joblib.dump(model, "artifacts/vol_weights.pkl")
print("Saved → artifacts/vol_weights.pkl")

# -----------------------------
# METRICS
# -----------------------------
def qlike(y_true_var,y_pred_var):
    ratio = y_true_var/y_pred_var
    return np.mean(ratio-np.log(ratio)-1)

def evaluate(name,X,y_log,y_var):
    preds_log = model.predict(X)
    preds_var = np.exp(preds_log)

    print("\n",name)
    print("MSE:",mean_squared_error(y_log,preds_log))
    print("MAE:",mean_absolute_error(np.sqrt(y_var),np.sqrt(preds_var)))
    print("QLIKE:",qlike(y_var,preds_var))

evaluate("Validation",X_val,y_val,y_val_var)
evaluate("Test",X_test,y_test,y_test_var)

if __name__ == "__main__":
    pass