import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Options Pricing Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================
# STYLE
# =========================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020617, #0f172a);
}

[data-testid="stSidebar"] {
    background-color: #020617;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label, span, div {
    color: #e5e7eb;
}

.metric-card {
    background: linear-gradient(135deg, #111827, #1e293b);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #334155;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 6px;
}

.metric-value {
    color: #38bdf8;
    font-size: 30px;
    font-weight: 700;
}

.info-box {
    background: #0f172a;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #334155;
    margin-bottom: 18px;
}

.formula-box {
    background: #020617;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1e40af;
    color: #dbeafe;
    margin-bottom: 18px;
}

.small-text {
    color: #94a3b8;
    font-size: 14px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #111827;
    border-radius: 12px;
    padding: 10px 18px;
    color: white;
}

.stTabs [aria-selected="true"] {
    background-color: #2563eb;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}

.stDataFrame {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# BLACK SCHOLES FUNCTIONS
# =========================

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def black_scholes(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    put = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)

    return call, put, d1, d2

def greeks(S, K, T, r, sigma, q=0.0):
    result = black_scholes(S, K, T, r, sigma, q)

    if result is None:
        return None

    call, put, d1, d2 = result

    call_delta = math.exp(-q * T) * norm_cdf(d1)
    put_delta = math.exp(-q * T) * (norm_cdf(d1) - 1)

    gamma = math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) / 100

    call_theta = (
        -(S * math.exp(-q * T) * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * norm_cdf(d2)
        + q * S * math.exp(-q * T) * norm_cdf(d1)
    ) / 365

    put_theta = (
        -(S * math.exp(-q * T) * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        + r * K * math.exp(-r * T) * norm_cdf(-d2)
        - q * S * math.exp(-q * T) * norm_cdf(-d1)
    ) / 365

    call_rho = K * T * math.exp(-r * T) * norm_cdf(d2) / 100
    put_rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100

    return {
        "Call Delta": call_delta,
        "Put Delta": put_delta,
        "Gamma": gamma,
        "Vega": vega,
        "Call Theta": call_theta,
        "Put Theta": put_theta,
        "Call Rho": call_rho,
        "Put Rho": put_rho
    }

def implied_volatility(market_price, S, K, T, r, q, option_type):
    low = 0.0001
    high = 5.0

    for _ in range(100):
        mid = (low + high) / 2
        call, put, _, _ = black_scholes(S, K, T, r, mid, q)

        model_price = call if option_type == "Call" else put

        if abs(model_price - market_price) < 1e-6:
            return mid

        if model_price > market_price:
            high = mid
        else:
            low = mid

    return mid

def plotly_layout(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"),
        margin=dict(l=40, r=40, t=70, b=40),
        height=520
    )
    return fig

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.title("Options Pricing Dashboard")
st.sidebar.caption("Derivatives analytics terminal")

st.sidebar.divider()

S = st.sidebar.number_input("Spot Price", min_value=0.01, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity, in years", min_value=0.01, value=1.0, step=0.05)
r = st.sidebar.number_input("Risk Free Rate", value=0.03, step=0.005, format="%.4f")
sigma = st.sidebar.number_input("Volatility", min_value=0.001, value=0.20, step=0.01, format="%.4f")
q = st.sidebar.number_input("Dividend Yield", value=0.00, step=0.005, format="%.4f")

st.sidebar.divider()
st.sidebar.caption("Inputs are expressed in decimals. Example: 20 percent volatility equals 0.20.")

st.sidebar.markdown("""
<div style="margin-top:8px; padding-top:8px; border-top:1px solid #1e293b; text-align:center;">
    <span style="font-size:12px; color:#64748b;">
        Designed and developed by
    </span><br>
    <span style="font-size:13px; color:#e2e8f0; font-weight:500;">
        Aurélien Challet
    </span>
</div>
""", unsafe_allow_html=True)
# =========================
# HEADER
# =========================

st.title("Options Pricing Dashboard")
st.markdown(
    "<p class='small-text'>An interactive dashboard for Black Scholes options pricing, Greeks, implied volatility, payoff analysis, scenario grids and volatility skew.</p>",
    unsafe_allow_html=True
)

result = black_scholes(S, K, T, r, sigma, q)
greeks_result = greeks(S, K, T, r, sigma, q)

if result is None:
    st.error("Please check your inputs.")
    st.stop()

call_price, put_price, d1, d2 = result

# =========================
# TOP METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Call Price</div>
        <div class="metric-value">{call_price:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Put Price</div>
        <div class="metric-value">{put_price:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.empty()

with col4:
    st.empty()

st.divider()

# =========================
# TABS
# =========================

tab1, tab2, tab3, tab5, tab4, tab6 = st.tabs([
    "Black Scholes",
    "Greeks",
    "Payoff",
    "Implied Volatility",
    "Scenario Grid",
    "Volatility Skew"
])

# =========================
# BLACK SCHOLES TAB
# =========================

with tab1:

    left_col, right_col = st.columns([1.15, 1])

    with left_col:
        st.header("Black-Scholes Model")

        st.markdown("""
        <div class="info-box">
        The Black–Scholes model prices European options by assuming that the underlying asset price follows a geometric Brownian motion.
        It uses spot price, strike price, time to maturity, volatility, risk-free rate and dividend yield to estimate theoretical call and put prices.
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)")
        st.latex(r"P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)")
        st.latex(r"d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")

    with right_col:
        st.header("Model Inputs")

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Spot Price</div>
                <div class="metric-value">{S}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Maturity</div>
                <div class="metric-value">{T}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Risk-Free Rate</div>
                <div class="metric-value">{r:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

        with input_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Strike</div>
                <div class="metric-value">{K}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Volatility</div>
                <div class="metric-value">{sigma:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Dividend Yield</div>
                <div class="metric-value">{q:.2%}</div>
            </div>
            """, unsafe_allow_html=True)


# =========================
# GREEKS TAB
# =========================

with tab2:
    st.header("Greeks Overview")

    st.markdown("""
    <style>
    .greek-card {
        background: linear-gradient(135deg, #111827, #1e293b);
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 6px;
    }

    .greek-title {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 2px;
    }

    .greek-value {
        color: #38bdf8;
        font-size: 20px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Call Option")

        st.markdown(f"""
        <div class="greek-card">
            <div class="greek-title">Δ Call Delta</div>
            <div class="greek-value">{greeks_result["Call Delta"]:.4f}</div>
        </div>

        <div class="greek-card">
            <div class="greek-title">Γ Gamma</div>
            <div class="greek-value">{greeks_result["Gamma"]:.4f}</div>
        </div>

        <div class="greek-card">
            <div class="greek-title">ν Vega</div>
            <div class="greek-value">{greeks_result["Vega"]:.4f}</div>
        </div>

        <div class="greek-card">
            <div class="greek-title">Θ Call Theta</div>
            <div class="greek-value">{greeks_result["Call Theta"]:.4f}</div>
        </div>

        <div class="greek-card">
            <div class="greek-title">ρ Call Rho</div>
            <div class="greek-value">{greeks_result["Call Rho"]:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### Put Option")

        st.markdown(f"""
        <div class="greek-card">
            <div class="greek-title">Δ Put Delta</div>
            <div class="greek-value">{greeks_result["Put Delta"]:.4f}</div>
        </div>

        <div class="greek-card">
            <div class="greek-title">Γ Gamma</div>
            <div class="greek-value">{greeks_result["Gamma"]:.4f}</div>
        </div>

        <div class="greek-card">
            <div class="greek-title">ν Vega</div>
            <div class="greek-value">{greeks_result["Vega"]:.4f}</div>
        </div>

        <div class="greek-card">
            <div class="greek-title">Θ Put Theta</div>
            <div class="greek-value">{greeks_result["Put Theta"]:.4f}</div>
        </div>

        <div class="greek-card">
            <div class="greek-title">ρ Put Rho</div>
            <div class="greek-value">{greeks_result["Put Rho"]:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
# =========================
# PAYOFF TAB
# =========================

with tab3:

    S_range = np.linspace(0, 2 * K, 300)

    # Payoff
    call_payoff = np.maximum(S_range - K, 0)
    put_payoff = np.maximum(K - S_range, 0)

    # Profit (payoff - premium)
    call_profit = call_payoff - call_price
    put_profit = put_payoff - put_price

    fig = go.Figure()

    # Call
    fig.add_trace(go.Scatter(
        x=S_range,
        y=call_profit,
        mode="lines",
        name="Call Profit",
        line=dict(width=3)
    ))

    # Put
    fig.add_trace(go.Scatter(
        x=S_range,
        y=put_profit,
        mode="lines",
        name="Put Profit",
        line=dict(width=3)
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white")

    # Strike
    fig.add_vline(x=K, line_dash="dash", line_color="#facc15")

    # Layout
    fig.update_layout(
        title=dict(
            text="Payoff and Profit Analysis",
            font=dict(color="white", size=22),
            x=0.02
        ),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=520
    )

    fig.update_xaxes(
        title="Underlying Price at Maturity",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        gridcolor="rgba(255,255,255,0.20)"
    )

    fig.update_yaxes(
        title="Profit and Loss",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        gridcolor="rgba(255,255,255,0.20)"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# SCENARIO GRID TAB
# =========================

with tab4:
    st.header("Scenario Grid")

    selected_option = st.radio(
        "Option Type",
        ["Call", "Put"],
        horizontal=True,
        key="scenario_option_type"
    )

    spot_grid = np.linspace(0.5 * S, 1.5 * S, 30)
    vol_grid = np.linspace(max(0.01, sigma * 0.5), sigma * 1.8, 30)

    matrix = []

    for vol in vol_grid:
        row = []
        for spot in spot_grid:
            c, p, _, _ = black_scholes(spot, K, T, r, vol, q)
            row.append(c if selected_option == "Call" else p)
        matrix.append(row)

    heatmap_df = pd.DataFrame(
        matrix,
        index=np.round(vol_grid, 4),
        columns=np.round(spot_grid, 2)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale="Turbo",
            colorbar=dict(
                title=dict(
                    text="Option Price",
                    font=dict(color="white")
                ),
                tickfont=dict(color="white")
            ),
            hovertemplate=
            "Spot Price: %{x:.2f}<br>" +
            "Volatility: %{y:.2%}<br>" +
            "Option Price: %{z:.4f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{selected_option} Price Sensitivity Grid",
            font=dict(color="white", size=22),
            x=0.02
        ),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="white",
            font_size=16,
            font_color="black"
        ),
        margin=dict(l=50, r=40, t=50, b=40),
        height=520
    )

    fig.update_xaxes(
        title="Spot Price",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        gridcolor="rgba(255,255,255,0.20)"
    )

    fig.update_yaxes(
        title="Volatility",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
        gridcolor="rgba(255,255,255,0.20)",
        tickformat=".0%"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# IMPLIED VOL TAB
# =========================

with tab5:
    st.header("Implied Volatility Solver")

    st.markdown("""
    <div class="info-box">
    Enter the observed market price of the option to estimate its implied volatility.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        option_type_iv = st.radio("Option Type for IV", ["Call", "Put"], horizontal=True)
        market_price = st.number_input("Market Option Price", min_value=0.01, value=float(call_price if option_type_iv == "Call" else put_price), step=0.1)

    with col_b:
        iv = implied_volatility(market_price, S, K, T, r, q, option_type_iv)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Implied Volatility</div>
            <div class="metric-value">{iv:.2%}</div>
        </div>
        """, unsafe_allow_html=True)



# =========================
# VOLATILITY SKEW TAB
# =========================

with tab6:
    st.markdown("""
    ### Volatility Skew
    The Black–Scholes model assumes constant volatility across strikes.

    In reality, implied volatility varies with strike. 
    Out-of-the-money puts typically exhibit higher implied volatility than calls, 
    reflecting the market’s demand for downside protection.

    For illustration purposes, the chart below shows a typical volatility skew shape.
    """)

    # Strike range
    strikes = np.linspace(0.7 * K, 1.3 * K, 100)

    # Stylised skew (example, not real data)
    base_vol = sigma
    moneyness = strikes / K

    

    skew = base_vol \
    + 0.35 * (moneyness - 1) ** 2 \
    + 0.10 * np.maximum(1 - moneyness, 0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strikes,
        y=skew,
        mode="lines",
        name="Implied Volatility",
        line=dict(width=3)
    ))

    # ATM line
    fig.add_vline(x=K, line_dash="dash", line_color="#facc15")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=20, b=40),
        height=450
    )

    fig.update_xaxes(
        title="Strike",
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    )

    fig.update_yaxes(
        title="Implied Volatility",
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

    

