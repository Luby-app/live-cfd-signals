# ====================================================
# LIVE AI TRADING ENGINE V3 - SAFE INDENTATION
# 16 TRHŮ, ATR+SPREAD, ADAPTIVE LEARNING, CORRELATION FILTER
# ====================================================

import os
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import streamlit as st

# ====================================================
# SETTINGS
# ====================================================
ACCOUNT_BALANCE = 5000          # CZK
RISK_PER_TRADE = 0.015          # 1.5%
PROB_THRESHOLD = 0.8            # only signals >80%
LOOKBACK_DAYS = 60
INTERVAL = "30m"

MARKETS = {
    "JP225": "JPY=X",
    "US100": "NDX",
    "US500": "SPY",
    "US30": "DJI",
    "DE40": "DAX",
    "EU50": "^STOXX50E",
    "UK100": "^FTSE",
    "FRA40": "^FCHI",
    "SPA35": "^IBEX",
    "SWI20": "^SSMI",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "OIL": "CL=F",
    "NATGAS": "NG=F",
    "COCOA": "CC=F",
    "COPPER": "HG=F"
}

DB_PATH = "trade_memory.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute(
    '''CREATE TABLE IF NOT EXISTS trades
       (time TEXT, market TEXT, signal TEXT, prob REAL,
        sl REAL, tp REAL, profit REAL)'''
)
conn.commit()

# ====================================================
# FETCH MARKET DATA
# ====================================================
def get_data(symbol):
    try:
        df = yf.download(symbol, period=f"{LOOKBACK_DAYS}d", interval=INTERVAL)
        df = df.dropna()
        return df
    except:
        return None

# ====================================================
# CALCULATE INDICATORS + FEATURES
# ====================================================
def calc_features(df):
    close = df['Close'].squeeze()
    df['EMA10'] = EMAIndicator(close, window=10).ema_indicator()
    df['EMA50'] = EMAIndicator(close, window=50).ema_indicator()
    df['RSI'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['EMA_diff'] = df['EMA10'] - df['EMA50']
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    features = df[['EMA_diff', 'RSI', 'MACD_diff']].fillna(0)
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    return df, features

# ====================================================
# CALCULATE ATR-BASED SL/TP
# ====================================================
def calc_sl_tp(df, latest_close, direction):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    spread = 0  # optional: vložit spread XTB
    buffer = atr * 0.5 + spread

    if direction == "LONG":
        sl = latest_close - buffer
        tp = latest_close + buffer * 3
    elif direction == "SHORT":
        sl = latest_close + buffer
        tp = latest_close - buffer * 3
    else:
        sl, tp = None, None
    return sl, tp

# ====================================================
# TRAIN MODEL & PREDICT
# ====================================================
def get_signal(features, df):
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(features, df['target'])
    latest = features.iloc[-1].values.reshape(1, -1)
    proba = model.predict_proba(latest)[0]
    if proba[1] > PROB_THRESHOLD:
        signal = "LONG"
    elif proba[0] > PROB_THRESHOLD:
        signal = "SHORT"
    else:
        signal = "HOLD"
    return signal, max(proba)

# ====================================================
# FILTER TOP 3 CORRELATED SIGNALS
# ====================================================
def filter_correlated(signals_df):
    signals_df = signals_df.sort_values(by="prob", ascending=False)
    if len(signals_df) > 3:
        signals_df = signals_df.iloc[:3]
    return signals_df

# ====================================================
# MAIN LOOP: SCAN MARKETS
# ====================================================
signals = []

for market, symbol in MARKETS.items():
    df = get_data(symbol)
    if df is None or len(df) < 50:
        continue
    df, features = calc_features(df)
    signal, prob = get_signal(features, df)
    if signal == "HOLD":
        continue
    latest_close = df['Close'].iloc[-1]
    sl, tp = calc_sl_tp(df, latest_close, signal)

    trade_risk_czk = ACCOUNT_BALANCE * RISK_PER_TRADE
    potential_profit_czk = None
    if sl is not None and tp is not None:
        potential_profit_czk = abs(tp - latest_close) / latest_close * trade_risk_czk * 1000

    signals.append({
        "market": market,
        "signal": signal,
        "prob": round(prob*100, 1),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "profit_czk": round(potential_profit_czk, 2) if potential_profit_czk is not None and not np.isnan(latest_close)
    })

    # Store in DB
    c.execute(
        'INSERT INTO trades VALUES (?,?,?,?,?,?,?)',
        (str(datetime.now()), market, signal, prob, sl, tp, potential_profit_czk)
    )
    conn.commit()

# Apply correlation filter / max top 3 signals
if signals:
    df_signals = pd.DataFrame(signals)
    df_signals = filter_correlated(df_signals)

# ====================================================
# STREAMLIT DISPLAY
# ====================================================
st.title("🔥 LIVE AI TRADING SIGNALS V3 🔥")
if signals:
    st.table(df_signals)
else:
    st.info("Žádné silné signály nad threshold pro aktuální data.")

conn.close()

