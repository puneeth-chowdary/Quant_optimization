import os
from openai import OpenAI
from dotenv import load_dotenv
from test1 import compute_weights_from_data
import json
import pandas as pd
import requests

load_dotenv()

# --------------------
# Fetch live portfolio
# --------------------

def get_live_portfolio():
    url = "http://127.0.0.1:5001/get-portfolio"

    r = requests.get(url, timeout=5)
    r.raise_for_status()

    data = r.json()

    return data["NSE_MAP"], data["QTY_MAP"]


# --------------------
# Convert Series -> JSON
# --------------------

# def series_to_json(series: pd.Series) -> str:

#     if not isinstance(series, pd.Series):
#         raise TypeError("Input must be a pandas Series")

#     # Convert numpy types safely
#     result_dict = {
#         str(k): float(v)
#         for k, v in series.to_dict().items()
#     }

#     # Removed indent=4 to reduce payload size
#     return json.dumps(result_dict)

def series_to_json(series: pd.Series) -> str:

    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")

    result_dict = {
        str(k): round(float(v), 4)
        for k, v in series.to_dict().items()
    }

    return json.dumps(
        result_dict,
        separators=(",", ":")
    )
# --------------------
# OpenAI / Groq Client
# --------------------

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# --------------------
# Generate Response
# --------------------

def generate_response(
    weights: pd.Series = None,
    QTY_MAP: dict = None
) -> str:

    # Fetch live portfolio ONLY when function is called
    NSE_MAP_live, QTY_MAP_live = get_live_portfolio()

    if weights is None:
        weights = compute_weights_from_data()

    if QTY_MAP is None:
        QTY_MAP = QTY_MAP_live

    # Compact JSON formatting
    #weights_json = series_to_json(weights)

    #qty_json = json.dumps(QTY_MAP)
    #-------
  # Keep only top allocations
    TOP_N = 15

    weights = (
         weights
        .sort_values(ascending=False)
        .head(TOP_N)
    )

    weights_json = series_to_json(weights)

    filtered_qty = {
         k: QTY_MAP[k]
        for k in weights.index
        if k in QTY_MAP
    }

    qty_json = json.dumps(
        filtered_qty,
        separators=(",", ":")
    )
    #---now
    # ---- Payload protection ----
    MAX_CHARS = 12000

    combined_data = weights_json + qty_json

    if len(combined_data) > MAX_CHARS:
        combined_data = combined_data[:MAX_CHARS]

    prompt = f"""
Analyze the following optimized portfolio from an MPT perspective of Indian stock market equity.

Optimized target weights (fraction of total portfolio value):
{weights_json}

Current asset quantities held:
{qty_json}

Tasks:
*use current market prices to estimate current portfolio weights from quantities*
1. Compare target weights vs current holdings and identify over- and under-allocated assets.
2. Estimate implied current portfolio weights from quantities (assume market prices if needed).
3. Recommend specific rebalancing trades (buy/sell and relative magnitude).
4. Assess diversification, concentration risk, and asset class balance.
5. Highlight any risk or efficiency concerns relative to MPT principles.

Output requirements:
-BE CLEAR AND ACTIONABLE
- Be concise and quantitative.
- Use bullet points.
- Provide clear rebalance actions.
- Avoid generic explanations of MPT.
Output format rules:
- Output ONLY plain text.
- No markdown.
- No asterisks (*)
- No hashtags (#)
- No JSON.
- No tables.
- No introductory sentences.
- No concluding sentences.
- No filler language.
- No blank sections.
- Use short bullet points starting with "-".
- Keep each bullet under 25 words.
- Use precise quantitative language.
- Mention only actionable insights.
"""

    # Final safety trim
    prompt = prompt[:15000]

    response = client.chat.completions.create(
        model= "openai/gpt-oss-120b",#"llama-3.3-70b-versatile"--2/5 very small response no explanation ,#"openai/gpt-oss-120b"--good 3.75/5 ,#"qwen/qwen3-32b"-pretty bad 2.5/5,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior financial analyst specializing in "
                    "Modern Portfolio Theory, portfolio optimization, "
                    "and risk minimization. Provide concise, quantitative, "
                    "actionable portfolio recommendations."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

   # return response.choices[0].message.content
    content = response.choices[0].message.content

# --------------------
# Clean Output
# --------------------

    content = (
        content
        .replace("*", "")
        .replace("#", "")
        .replace("```", "")
    )

# Remove excessive blank lines
    lines = [line.strip() for line in content.splitlines()]

    cleaned_lines = []
    for line in lines:
        if line:
            cleaned_lines.append(line)

    cleaned_content = "\n".join(cleaned_lines)

    return cleaned_content