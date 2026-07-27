"""
Technical indicators using the `ta` library.
Easy to extend with more later.
"""

import pandas as pd
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a solid set of indicators to an OHLCV DataFrame."""
    if df is None or df.empty:
        return df

    out = df.copy()

    # Moving averages
    out["sma_20"] = SMAIndicator(close=out["close"], window=20).sma_indicator()
    out["sma_50"] = SMAIndicator(close=out["close"], window=50).sma_indicator()
    out["ema_12"] = EMAIndicator(close=out["close"], window=12).ema_indicator()
    out["ema_26"] = EMAIndicator(close=out["close"], window=26).ema_indicator()
    out["ema_50"] = EMAIndicator(close=out["close"], window=50).ema_indicator()

    # RSI
    out["rsi"] = RSIIndicator(close=out["close"], window=14).rsi()

    # MACD
    macd = MACD(close=out["close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=out["close"], window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_low"] = bb.bollinger_lband()

    # Stochastic
    stoch = StochasticOscillator(
        high=out["high"], low=out["low"], close=out["close"], window=14, smooth_window=3
    )
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    # ATR
    out["atr"] = AverageTrueRange(
        high=out["high"], low=out["low"], close=out["close"], window=14
    ).average_true_range()

    # OBV
    out["obv"] = OnBalanceVolumeIndicator(
        close=out["close"], volume=out["volume"]
    ).on_balance_volume()

    return out


def get_latest_signals(df: pd.DataFrame) -> dict:
    """Simple rule-based signal summary for the chat / overview."""
    if df is None or len(df) < 50:
        return {"status": "insufficient data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    signals = {}

    # RSI
    rsi = last.get("rsi")
    if pd.notna(rsi):
        if rsi > 70:
            signals["rsi"] = f"Overbought ({rsi:.1f})"
        elif rsi < 30:
            signals["rsi"] = f"Oversold ({rsi:.1f})"
        else:
            signals["rsi"] = f"Neutral ({rsi:.1f})"

    # MACD
    if pd.notna(last.get("macd")) and pd.notna(last.get("macd_signal")):
        if last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
            signals["macd"] = "Bullish crossover"
        elif last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
            signals["macd"] = "Bearish crossover"
        elif last["macd"] > last["macd_signal"]:
            signals["macd"] = "Bullish"
        else:
            signals["macd"] = "Bearish"

    # Trend via EMAs
    if pd.notna(last.get("ema_12")) and pd.notna(last.get("ema_26")):
        if last["ema_12"] > last["ema_26"]:
            signals["trend"] = "Uptrend (EMA12 > EMA26)"
        else:
            signals["trend"] = "Downtrend (EMA12 < EMA26)"

    # Bollinger
    if pd.notna(last.get("bb_high")) and pd.notna(last.get("close")):
        if last["close"] > last["bb_high"]:
            signals["bb"] = "Price above upper band"
        elif last["close"] < last["bb_low"]:
            signals["bb"] = "Price below lower band"
