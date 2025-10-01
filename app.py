
import os, json, streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd

def configure_genai():
    sa_dict = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
    sa_dict = dict(sa_dict)  # convert AttrDict → normal dict

    sa_path = "/tmp/sa.json"
    with open(sa_path, "w") as f:
        json.dump(sa_dict, f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    genai.configure()
    st.write("✅ ADC configured successfully")

# Call once at startup
configure_genai()

# Create model after credentials are set
model = genai.GenerativeModel("gemini-2.5-pro")




def get_sector(ticker):
    """获取股票所属行业"""
    try:
        return yf.Ticker(ticker).info.get('sector', 'Other')
    except Exception:
        return 'Other'



def get_metrics(tickers, weights):
    """计算组合的回报、风险、最大回撤"""
    data = yf.download(tickers, period="6mo", auto_adjust=True, progress=False)

    # 取 Close 层
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs('Close', level=0, axis=1)
    elif 'Close' in data.columns:
        data = data['Close']
    else:
        raise ValueError(f"No price data for {tickers}")

    # 确保是 DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    daily_ret = data.pct_change().dropna()
    port_ret = (daily_ret * weights).sum(axis=1)

    avg_return = port_ret.mean() * 252
    risk = port_ret.std() * np.sqrt(252)

    cum_ret = (1 + port_ret).cumprod()
    roll_max = cum_ret.cummax()
    drawdown = ((cum_ret / roll_max - 1).min()) * 100

    return avg_return, risk, abs(drawdown)


# ------------------- UI -------------------
st.title("AI Portfolio Advisor")

with st.form("form"):
    age = st.slider("Age", 18, 100, 35)
    risk_profile = st.selectbox("Risk Profile", ["Averse", "Aggressive"])
    portfolio = st.text_area(
        "Portfolio (ticker %weight, one per line)",
        placeholder="MSFT 5\nNVDA 5\nBND 20"
    )
    stock = st.text_input("Stock to Add (full ticker, e.g., AAPL)")
    submit = st.form_submit_button("Assess Fit")

# ------------------- Logic -------------------
if submit:
    # Parse portfolio
    port_dict = {}
    for line in portfolio.splitlines():
        try:
            t, w = line.split()
            port_dict[t.strip().upper()] = float(w.strip('%')) / 100
        except Exception:
            continue

    tickers = list(port_dict.keys())
    weights = np.array(list(port_dict.values()))

    if not np.isclose(sum(weights), 1.0, atol=0.01):
        st.error("Weights must sum to ~100%")
        st.stop()

    # Before metrics
    r_before, risk_before, dd_before = get_metrics(tickers, weights)

    # Add new stock
    new_weight = 0.05
    weights = weights * (1 - new_weight)
    tickers_new = tickers + [stock.upper()]
    weights_new = np.append(weights, new_weight)

    # After metrics
    r_after, risk_after, dd_after = get_metrics(tickers_new, weights_new)

    # Calculate deltas
    delta = {
        "return_change": round((r_after - r_before) / abs(r_before) * 100, 2),
        "risk_change": round((risk_after - risk_before) / risk_before * 100, 2),
        "drawdown_change": round((dd_after - dd_before) / dd_before * 100, 2)
    }

    # ------------------- Charts -------------------
    # Sector allocation (before/after)
    sectors_before = {}
    for t, w in port_dict.items():
        sec = get_sector(t)
        sectors_before[sec] = sectors_before.get(sec, 0) + w

    sectors_after = sectors_before.copy()
    sec_new = get_sector(stock.upper())
    sectors_after[sec_new] = sectors_after.get(sec_new, 0) + new_weight

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flatten()
    ax[0].pie(sectors_before.values(), labels=sectors_before.keys(), autopct='%1.1f%%')
    ax[0].set_title("Before")
    ax[1].pie(sectors_after.values(), labels=sectors_after.keys(), autopct='%1.1f%%')
    ax[1].set_title("After")
    st.pyplot(fig)

    # Bar chart (fixed order)
    fig2, ax2 = plt.subplots()
    keys = ["return_change", "risk_change", "drawdown_change"]
    ax2.bar(["Return", "Risk", "Drawdown"], [delta[k] for k in keys])
    ax2.set_title("Portfolio Impact (%)")
    st.pyplot(fig2)

    # ------------------- LLM Recommendation -------------------
    prompt = f"""
    You are a CFA-level advisor. User: {age} years old, risk profile {risk_profile}.
    Portfolio before: {port_dict}. Added stock: {stock.upper()} (5%).
    Metrics changes: {json.dumps(delta)}. Criteria: EPS>10%, P/E<30, D/E<1, ROE>15%.
    Write a short recommendation explaining the trade-off, suggest an alternative if diversification is poor.
    """
    response = model.generate_content(prompt)
    st.write(response.text)

    st.json(delta)
    st.caption("Demo only — not financial advice.")
