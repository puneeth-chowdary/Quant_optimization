# portfolio_data.py
# In-memory portfolio maps shared across Python modules

NSE_MAP = {}
QTY_MAP = {}


def update_maps(stocks):
    """
    Update NSE_MAP and QTY_MAP from frontend stock list.

    stocks = [
        {"symbol": "RELIANCE.NS", "qty": 10},
        {"symbol": "TCS.NS", "qty": 5}
    ]
    """
    NSE_MAP.clear()
    QTY_MAP.clear()

    if not stocks:
        return

    for item in stocks:
        symbol_full = str(item.get("symbol", "")).strip()
        qty = int(item.get("qty", 0))

        if not symbol_full or qty <= 0:
            continue

        # xyz.NS → xyz
        base = symbol_full.split(".")[0].lower()

        NSE_MAP[base] = symbol_full
        QTY_MAP[base] = qty


def reset_maps():
    """Clear all portfolio data (called on browser refresh)."""
    NSE_MAP.clear()
    QTY_MAP.clear()