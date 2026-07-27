import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import time
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Crypto Analyzer", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main { background-color: #0e1117; }
div[data-testid="stSidebar"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# ---------- Data helpers ----------
COINGECKO = "https://api.coingecko.com/api/v3"
_cache = {}

def cached_get(url, params=None, ttl=60):
    key = url + str(params)
    if key in _cache and time.time() - _cache[key][1] < ttl:
        return _cache[key][0]
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        _cache[key] = (data, time.time())
        return data
    except:
        return None

def search_coins(q):
    data = cached_get(f"{COINGECKO}/search", {"query": q})
    if not data: return []
    return [{"id": c["id"], "symbol": c["symbol"].upper(), "name": c["name"]} for c in data.get("coins", [])[:12]]

def get_top_coins(n=50):
    data = cached_get(f"{COINGECKO}/coins/markets", {
        "vs_currency": "usd", "order": "market_cap_desc", "per_page": n, "page": 1
    })
    if not data:
        return [
            {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin", "current_price": 65000, "price_change_percentage_24h": 1.2},
            {"id": "ethereum", "symbol": "ETH", "name": "Ethereum", "current_price": 3400, "price_change_percentage_24h": -0.5},
            {"id": "solana", "symbol": "SOL", "name": "Solana", "current_price": 145, "price_change_percentage_24h": 3.1},
        ]
    return data

def get_ohlcv(coin_id, days=90):
    data = cached_get(f"{COINGECKO}/coins/{coin_id}/market_chart", {
        "vs_currency": "usd", "days": days
    })
    if not data or "prices" not in data:
        # sample data
        np.random.seed(hash(coin_id) % 2**32)
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        price = 100.0
        rows = []
        for _ in dates:
            change = np.random.normal(0, 0.02)
            o = price
            c = price * (1 + change)
            h = max(o, c) * (1 + abs(np.random.normal(0, 0.01)))
            l = min(o, c) * (1 - abs(np.random.normal(0, 0.01)))
            rows.append([o, h, l, c, abs(np.random.normal(1e6, 3e5))])
            price = c
        return pd.DataFrame(rows, index=dates, columns=["open", "high", "low", "close", "volume"])

    prices = data["prices"]
    volumes = data.get("total_volumes", [])
    df = pd.DataFrame(prices, columns=["ts", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts")
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    if volumes:
        vdf = pd.DataFrame(volumes, columns=["ts", "volume"])
        vdf["ts"] = pd.to_datetime(vdf["ts"], unit="ms")
        df = df.join(vdf.set_index("ts"), how="left")
    df["volume"] = df.get("volume", 0).fillna(0)
    return df[["open", "high", "low", "close", "volume"]].dropna()

def add_indicators(df):
    if df is None or len(df) < 30:
        return df
    out = df.copy()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["ema_12"] = out["close"].ewm(span=12).mean()
    out["ema_26"] = out["close"].ewm(span=26).mean()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    out["rsi"] = 100 - (100 / (1 + rs))
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["bb_mid"] = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_high"] = out["bb_mid"] + 2 * std
    out["bb_low"] = out["bb_mid"] - 2 * std
    return out

def get_signals(df):
    if df is None or len(df) < 50:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    s = {}
    rsi = last.get("rsi")
    if pd.notna(rsi):
        s["rsi"] = f"Overbought ({rsi:.1f})" if rsi > 70 else f"Oversold ({rsi:.1f})" if rsi < 30 else f"Neutral ({rsi:.1f})"
    if pd.notna(last.get("macd")):
        if last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
            s["macd"] = "Bullish crossover"
        elif last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
            s["macd"] = "Bearish crossover"
        else:
            s["macd"] = "Bullish" if last["macd"] > last["macd_signal"] else "Bearish"
    if pd.notna(last.get("ema_12")):
        s["trend"] = "Uptrend" if last["ema_12"] > last["ema_26"] else "Downtrend"
    return s

# ---------- Session ----------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["bitcoin", "ethereum", "solana"]
if "selected" not in st.session_state:
    st.session_state.selected = "bitcoin"
if "chat" not in st.session_state:
    st.session_state.chat = []
if "drawings" not in st.session_state:
    st.session_state.drawings = []

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📈 Crypto Analyzer")
    st.caption("Charts • Indicators • Chat")
    st.markdown("---")
    q = st.text_input("Search coin", placeholder="bitcoin, sol, pepe...")
    if q:
        for c in search_coins(q):
            if st.button(f"➕ {c['symbol']} – {c['name']}", key=f"add_{c['id']}"):
                if c["id"] not in st.session_state.watchlist:
                    st.session_state.watchlist.append(c["id"])
                    st.rerun()
    st.markdown("---")
    st.subheader("Watchlist")
    tops = {c["id"]: c for c in get_top_coins(80)}
    for cid in st.session_state.watchlist:
        info = tops.get(cid, {"symbol": cid[:5].upper(), "price_change_percentage_24h": None})
        chg = info.get("price_change_percentage_24h")
        label = f"{info.get('symbol', cid)} ({chg:+.1f}%)" if chg is not None else info.get("symbol", cid)
        cols = st.columns([4, 1])
        if cols[0].button(label, key=f"sel_{cid}", use_container_width=True):
            st.session_state.selected = cid
            st.rerun()
        if cols[1].button("✕", key=f"rm_{cid}"):
            st.session_state.watchlist.remove(cid)
            st.rerun()
    st.markdown("---")
    days = st.select_slider("Timeframe", options=[7, 14, 30, 90, 180, 365], value=90, format_func=lambda x: f"{x}d")
    show_sma = st.checkbox("SMA 20/50", True)
    show_ema = st.checkbox("EMA 12/26", True)
    show_bb = st.checkbox("Bollinger", True)
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_vol = st.checkbox("Volume", True)

# ---------- Main ----------
coin = st.session_state.selected
info = next((c for c in get_top_coins(100) if c["id"] == coin), {"name": coin, "symbol": coin[:5].upper()})
name = info.get("name", coin)
symbol = info.get("symbol", coin[:5].upper())

st.header(f"{name} ({symbol})")

c1, c2, c3 = st.columns(3)
price = info.get("current_price")
c1.metric("Price", f"\( {price:,.4f}" if price and price < 1 else f" \){price:,.2f}" if price else "—")
chg = info.get("price_change_percentage_24h")
c2.metric("24h", f"{chg:+.2f}%" if chg is not None else "—")
c3.metric("Coin", symbol)

df = add_indicators(get_ohlcv(coin, days))
signals = get_signals(df)

if df is not None and not df.empty:
    rows = 1 + sum([show_vol, show_rsi, show_macd])
    heights = [0.55] + [0.15] * (rows - 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)

    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)

    if show_sma:
        if "sma_20" in df: fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA20", line=dict(width=1.2, color="#42a5f5")), row=1, col=1)
        if "sma_50" in df: fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA50", line=dict(width=1.2, color="#ab47bc")), row=1, col=1)
    if show_ema:
        if "ema_12" in df: fig.add_trace(go.Scatter(x=df.index, y=df["ema_12"], name="EMA12", line=dict(width=1, color="#ffca28")), row=1, col=1)
        if "ema_26" in df: fig.add_trace(go.Scatter(x=df.index, y=df["ema_26"], name="EMA26", line=dict(width=1, color="#ff7043")), row=1, col=1)
    if show_bb and "bb_high" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"], name="BB High", line=dict(width=1, color="rgba(100,181,246,0.5)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"], name="BB Low", line=dict(width=1, color="rgba(100,181,246,0.5)", dash="dot"), fill="tonexty", fillcolor="rgba(100,181,246,0.08)"), row=1, col=1)

    for d in st.session_state.drawings:
        if d["type"] == "hline":
            fig.add_hline(y=d["y"], line_dash="dash", line_color="#ffeb3b", annotation_text=d.get("label", ""), row=1, col=1)

    r = 2
    if show_vol:
        colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.6), row=r, col=1)
        r += 1
    if show_rsi and "rsi" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#7e57c2")), row=r, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.5, row=r, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=r, col=1)
        r += 1
    if show_macd and "macd" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#29b6f6")), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#ef5350")), row=r, col=1)
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Hist", marker_color=colors, opacity=0.5), row=r, col=1)

    fig.update_layout(height=680, template="plotly_dark", xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02), margin=dict(l=40, r=40, t=30, b=30), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("✏️ Drawing tools"):
        price_lvl = st.number_input("Price level", value=float(df["close"].iloc[-1]), format="%.4f")
        label = st.text_input("Label", "Support / Resistance")
        if st.button("Add horizontal line"):
            st.session_state.drawings.append({"type": "hline", "y": price_lvl, "label": label})
            st.rerun()
        if st.button("Clear drawings"):
            st.session_state.drawings = []
            st.rerun()

    if signals:
        st.subheader("📌 Signals")
        cols = st.columns(len(signals))
        for i, (k, v) in enumerate(signals.items()):
            cols[i].info(f"**{k.upper()}**\n\n{v}")
else:
    st.warning("No data available right now.")

# ---------- Chat ----------
st.markdown("---")
st.subheader("💬 Analysis Chat")
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about RSI, trend, MACD, support...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    p = prompt.lower()
    if "rsi" in p:
        reply = f"**{symbol} RSI** → {signals.get('rsi', 'not ready')}"
    elif "macd" in p:
        reply = f"**{symbol} MACD** → {signals.get('macd', 'not ready')}"
    elif "trend" in p or "bull" in p or "bear" in p:
        reply = f"**{symbol} trend** → {signals.get('trend', 'not ready')}"
    elif "summary" in p or "analyze" in p:
        reply = f"**Quick take on {name}**\n" + "\n".join([f"- **{k.upper()}**: {v}" for k, v in signals.items()])
    else:
        reply = f"I can answer about **RSI, MACD, trend** or give a **summary** of {symbol}."
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()
