
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
    #st.write("✅ ADC configured successfully")

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

    sectors_before = {}
    for t, w in port_dict.items():
        sec = get_sector(t)
        sectors_before[sec] = sectors_before.get(sec, 0) + w

    sectors_after = sectors_before.copy()
    sec_new = get_sector(stock.upper())
    sectors_after[sec_new] = sectors_after.get(sec_new, 0) + new_weight

    # Define palettes
 
    
    pie_colors = ['#4da6ff', '#6666ff', '#9933ff', '#cc0000',
                '#ffb84d', '#ff704d', '#ff4d4d', '#ff66b3']


    bar_colors = ['#6666ff', '#9933ff', '#cc0000']  # Return, Risk, Drawdown

    # Pie charts
    fig, ax = plt.subplots(1, 2, figsize=(11, 7), facecolor="white")
    #plt.subplots_adjust(wspace=0.4)  # add spacing between pies
    plt.subplots_adjust(wspace=0.4, top=1.2) 
    ax = ax.flatten()

    # Global title for both charts
    fig.suptitle(
        "Sector Concentration Impact",
        fontsize=16, fontweight="bold", color="black"
    )

    ax[0].pie(
        sectors_before.values(),
        labels=sectors_before.keys(),
        autopct='%1.1f%%',
        colors=pie_colors[:len(sectors_before)],
        textprops={'color': 'black', 'fontsize': 10, 'fontweight': 'bold'}
    )
    ax[0].set_title("Before", color="black", fontsize=14, fontweight="bold")

    ax[1].pie(
        sectors_after.values(),
        labels=sectors_after.keys(),
        autopct='%1.1f%%',
        colors=pie_colors[:len(sectors_after)],
        textprops={'color': 'black', 'fontsize': 10, 'fontweight': 'bold'}
    )
    ax[1].set_title("After", color="black", fontsize=14, fontweight="bold")

    st.pyplot(fig)

    # Bar chart (fixed order)
    fig2, ax2 = plt.subplots(facecolor="white")

    keys   = ["return_change", "risk_change", "drawdown_change"]
    labels = ["Return", "Risk\n(Std Dev)", "Drawdown\n(MaxDD–Worst Drop)"]

    # make sure all values are finite numbers (NaN/None → 0)
    vals = []
    for k in keys:
        v = delta.get(k, 0)
        try:
            v = float(v)
        except:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        vals.append(v)

    x = np.arange(len(labels))
    ax2.bar(x, vals, color=['#6666ff', '#9933ff', '#cc0000'])  # your palette

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, ha="center")
    ax2.set_title("Portfolio Impact (%)", color="#333")
    ax2.tick_params(colors="#333")
    fig2.subplots_adjust(bottom=0.2)  # room for multi-line labels

    st.pyplot(fig2)





# ------------------- LLM Recommendation -------------------
    prompt = f"""
    You are a CFA-level advisor. User: {age} years old, risk profile {risk_profile}.
    Portfolio before: {port_dict}. Added stock: {stock.upper()} (5%).

    Metrics changes (note carefully):
    - return_change = % change in expected return (positive = better, negative = worse).
    - risk_change = % change in volatility (positive = more daily swings).
    - drawdown_change = % change in maximum drawdown 
    (negative = improved resilience, positive = worse max loss).

    Metrics changes: {json.dumps(delta)}

    Criteria: EPS>10% (growth), P/E<30 (value), D/E<1 (health), ROE>15% (quality).

    **Output format:**
    1. Executive Summary — one sentence (30–40 words) clearly state if adding {stock.upper()} is advisable or not, based on return_change, risk_change, drawdown_change, and sector concentration.
    2. Detailed Analysis — explain trade-off in plain English (risk = daily swings, drawdown = worst-case loss during 6 months).
    3. Alternative Suggestion — only if diversification is poor, suggest one stock/ETF and why.
    """
    response = model.generate_content(prompt)
    st.write(response.text)
    
    st.json(delta)
    st.caption("Demo only — not financial advice.")