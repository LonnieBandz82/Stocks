import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import stock_core as sc

st.set_page_config(page_title="Stock Edge", page_icon="📈", layout="centered")
st.title("📈 Stock Edge")

with st.expander("Settings", expanded=False):
    raw = st.text_input("Watchlist (comma separated)", "AAPL, MSFT, NVDA, AMZN, GOOGL, TSLA, SPY")
    account = st.number_input("Account size ($)", 1000, 10_000_000, 10_000, step=1000)
    risk = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25) / 100
tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]


@st.cache_data(ttl=900, show_spinner=False)
def cached_load(t, period="5y"):
    return sc.load(t, period)


tab1, tab2, tab3 = st.tabs(["Scanner", "Chart", "Backtest"])

with tab1:
    with st.spinner("Scanning..."):
        rows = sc.scan(tickers, account, risk, loader=cached_load)
    if rows:
        st.dataframe([{k: v for k, v in r.items() if k != "Prev"} for r in rows],
                     hide_index=True, use_container_width=True)
        st.caption("Shares = size so a 2×ATR stop loses ≈ your risk %. Signals are research, not advice.")
    else:
        st.warning("No data returned.")

with tab2:
    t = st.selectbox("Ticker", tickers, key="chart_t")
    days = st.select_slider("Days shown", [60, 120, 250, 500], 250)
    df = sc.add_indicators(cached_load(t))
    s = sc.score(df)
    buys = df[(s >= 3) & (s.shift() < 3)].tail(days)
    v = df.tail(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close,
                                 name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v.SMA50, name="SMA50", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v.SMA200, name="SMA200", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=buys.index, y=buys.Low * 0.98, mode="markers", name="New BUY",
                             marker=dict(symbol="triangle-up", size=11, color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v.RSI, name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    t = st.selectbox("Ticker", tickers, key="bt_t")
    thr = st.slider("Entry score (higher = pickier)", 1, 4, 3)
    df = sc.add_indicators(cached_load(t))
    b = sc.backtest(df, thr)
    c1, c2 = st.columns(2)
    c1.metric("Strategy", f"{b['return']:.0%}", f"{b['return'] - b['bh_return']:+.0%} vs hold")
    c2.metric("Buy & hold", f"{b['bh_return']:.0%}")
    c1.metric("Max drawdown", f"{b['max_dd']:.0%}")
    c2.metric("Sharpe", f"{b['sharpe']:.2f}", f"hold {b['bh_sharpe']:.2f}", delta_color="off")
    st.line_chart({"Strategy": b["equity"], "Buy & hold": b["buy_hold"]})

    st.subheader("Walk-forward test (out-of-sample)")
    st.caption("Tunes the entry score on 2 years of past data, then trades it on the next unseen "
               "6 months, repeatedly. This is the honest test.")
    if st.button("Run walk-forward"):
        w = sc.walk_forward(sc.add_indicators(cached_load(t, "10y")))
        if not w:
            st.warning("Not enough history.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Out-of-sample", f"{w['return']:.0%}")
            c2.metric("Buy & hold (same dates)", f"{w['bh_return']:.0%}")
            st.line_chart({"Walk-forward": w["equity"], "Buy & hold": w["buy_hold"]})
            if w["sharpe"] > w["bh_sharpe"] and w["return"] > 0:
                st.success("Held up out-of-sample on risk-adjusted terms. Still no guarantee.")
            else:
                st.error("No proven edge: it did not beat buy & hold risk-adjusted on unseen data.")
