from sklearn.ensemble import HistGradientBoostingRegressor

def build_trade_model_20d():
    return HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=4,
        learning_rate=0.04,
        l2_regularization=1.0,
        random_state=42
    )