import os
from openai import OpenAI
from dotenv import load_dotenv
from test1 import compute_weights_from_data
import json 
import pandas as pd
load_dotenv()
#-----
import requests

def get_live_portfolio():
    url = "http://127.0.0.1:5001/get-portfolio"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()
    return data["NSE_MAP"], data["QTY_MAP"]
#--------
#weights=compute_weights_from_data()
def series_to_json(series: pd.Series) -> str:
  
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    result_dict = series.to_dict()
    return json.dumps(result_dict, indent=4)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def generate_response(weights: pd.Series = None, QTY_MAP: dict = None) -> str:
    # Fetch live portfolio ONLY when function is called
    NSE_MAP_live, QTY_MAP_live = get_live_portfolio()

    if weights is None:
        weights = compute_weights_from_data()

    if QTY_MAP is None:
        QTY_MAP = QTY_MAP_live

    response = client.chat.completions.create(
        model="groq/compound",
        messages = [
    {
        "role": "system",
        "content": (
            "You are a senior financial analyst specializing in Modern Portfolio Theory, "
            "portfolio optimization, and risk minimization. Provide concise, quantitative, "
            "actionable portfolio recommendations."
        )
    },
    {
        "role": "user",
        "content": f"""
Analyze the following optimized portfolio from an MPT perspective of Indian stock market equity.

Optimized target weights (fraction of total portfolio value):
{series_to_json(weights)}

Current asset quantities held:
{QTY_MAP}

Tasks:
1. Compare target weights vs current holdings and identify over- and under-allocated assets.
2. Estimate implied current portfolio weights from quantities (assume market prices if needed).
3. Recommend specific rebalancing trades (buy/sell and relative magnitude).
4. Assess diversification, concentration risk, and asset class balance.
5. Highlight any risk or efficiency concerns relative to MPT principles.

Output requirements:
- Be concise and quantitative.
- Use bullet points.
- Provide clear rebalance actions.
- Avoid generic explanations of MPT.
"""
    }
]
    )

    return response.choices[0].message.content