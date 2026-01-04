import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# INSTRUMENTY
# -----------------------------
instruments = {
    "JP225": {"symbol": "^N225", "contract_value": 33500, "leverage": 20},
    "EU50": {"symbol": "^STOXX50E", "contract_value": 14200, "leverage": 20},
    "COCOA": {"symbol": "CC=F", "contract_value": 12100, "leverage": 20},
    "SILVER": {"symbol": "SI=F", "contract_value": 74800, "leverage": 20}
}

lookback_days = 30
interval = "1h"
SL_ATR_mult = 2
TP_ATR_mult = 4
threshold_prob = 0.6
MIN_PROBABILITY = 0.8
atr_min_thresholds = {"JP225": 10,"EU50": 5,"COCOA": 50,"SILVER": 0.5}
TOP_SIGNAL_COUNT = 5

# -----------------------------
# FUNKCE
# -----------------------------
def prepare_data(symbol):
    data = yf.download(symbol, period=f"{lookback_days}d", interval=interval)
    if data.empty or len(data) < 20:
        return None
    close = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()

    data['EMA10'] = EMAIndicator(close, window=10).ema_indicator()
    data['EMA50'] = EMAIndicator(close, window=50).ema_indicator()
    macd = MACD(close)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_diff'] = data['MACD'] - data['MACD_signal']
    data['EMA_diff'] = data['EMA10'] - data['EMA50']

    data['RSI'] = RSIIndicator(close, window=14).rsi()
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    data['STOCH'] = stoch.stoch()
    data['STOCH_signal'] = stoch.stoch_signal()

    data['ATR'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    bb = BollingerBands(close, window=20, window_dev=2)
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()

    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    return data

def simulate_trade(entry_price, atr_value, direction, future_price=None):
    if direction=="LONG":
        sl = entry_price - SL_ATR_mult*atr_value
        tp = entry_price + TP_ATR_mult*atr_value
    else:
        sl = entry_price + SL_ATR_mult*atr_value
        tp = entry_price - TP_ATR_mult*atr_value
    return sl, tp

# -----------------------------
# GENEROVAT SIGNÁLY
# -----------------------------
def generate_live_signals():
    final_summary = []
    for name, params in instruments.items():
        data = prepare_data(params['symbol'])
        if data is None: continue
        i = len(data) - 1

        hist = data.iloc[:i][['EMA_diff','RSI','MACD_diff','Close']].dropna()
        hist['target'] = np.where(hist['Close'].shift(-1) > hist['Close'], 1, -1)
        if len(hist) < 20: continue
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(hist[['EMA_diff','RSI','MACD_diff']], hist['target'])

        latest_features = data[['EMA_diff','RSI','MACD_diff']].iloc[i].to_numpy().reshape(1,-1)
        proba = model.predict_proba(latest_features)[0]

        if proba[1] > threshold_prob:
            signal = "LONG"
            prob = proba[1]
        elif proba[0] > threshold_prob:
            signal = "SHORT"
            prob = proba[0]
        else:
            continue

        # Filtr indikátorů
        price = float(data['Close'].iloc[i])
        ema10 = float(data['EMA10'].iloc[i])
        ema50 = float(data['EMA50'].iloc[i])
        atr = float(data['ATR'].iloc[i])
        rsi = float(data['RSI'].iloc[i])
        stoch = float(data['STOCH'].iloc[i])
        stoch_sig = float(data['STOCH_signal'].iloc[i])
        bb_high = float(data['BB_high'].iloc[i])
        bb_low = float(data['BB_low'].iloc[i])

        if signal=="LONG" and (ema10<ema50 or rsi<50 or stoch<stoch_sig or price>bb_high or atr<atr_min_thresholds[name] or prob<MIN_PROBABILITY):
            continue
        if signal=="SHORT" and (ema10>ema50 or rsi>50 or stoch>stoch_sig or price<bb_low or atr<atr_min_thresholds[name] or prob<MIN_PROBABILITY):
            continue

        sl, tp = simulate_trade(price, atr, signal)
        profit_czk = (tp-price if signal=="LONG" else price-tp) * params['contract_value'] * params['leverage'] / 0.01

        final_summary.append({
            "Index": name,
            "Signal": signal,
            "Price": round(price,2),
            "SL": round(sl,2),
            "TP": round(tp,2),
            "Probability": round(prob*100,1),
            "Potential Profit (CZK)": round(profit_czk,0)
        })

    return final_summary

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Live CFD Signals", layout="wide")
st.title("📊 Live CFD Signály")

signals = generate_live_signals()
if signals:
    df = pd.DataFrame(signals).sort_values(by="Probability", ascending=False).head(TOP_SIGNAL_COUNT)
    def color_row(row):
        return ['background-color: #b3ffb3' if row.Signal=="LONG" else 'background-color: #ffb3b3']*len(row)
    st.dataframe(df.style.apply(color_row, axis=1))
else:
    st.info("Žádné silné signály nad threshold pro aktuální data.")

st.caption("Aktualizace při každém otevření stránky.")

