import os
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# ============================
# SLOŽKY
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================
# NASTAVENÍ TRHŮ (16 instrumentů)
# ============================
# ticker z Yahoo Finance, investice CZK na obchod (odpovídající paku)
instruments = [
    "^N225", "EUR=X", "CC=F", "SI=F",    # JP225, EU50, COCOA, SILVER
    "GC=F", "CL=F", "NG=F", "ZC=F",      # GOLD, CRUDE OIL, NAT GAS, CORN
    "ZS=F", "ZW=F", "SB=F", "HG=F",      # SOYBEAN, WHEAT, SUGAR, COPPER
    "PL=F", "PA=F", "RB=F", "HG=F"       # PLATINUM, PALLADIUM, RBOB GASOLINE, COPPER duplicate
]

trade_risks = [
    1678, 710, 1215, 7479,     # JP225, EU50, COCOA, SILVER
    2000, 1500, 1200, 1000,    # GOLD, CRUDE OIL, NAT GAS, CORN
    1100, 950, 800, 1300,      # SOYBEAN, WHEAT, SUGAR, COPPER
    2500, 2700, 1800, 1300     # PLATINUM, PALLADIUM, RBOB, COPPER duplicate
]

interval = "30m"
lookback_days = 30
prob_threshold = 80  # filtr silných signálů

# ============================
# STREAMLIT UI
# ============================
st.title("Live CFD AI Trading Signals")
st.write(f"Aktuální silné signály (pravděpodobnost ≥ {prob_threshold}%)")

# ============================
# FUNKCE PRO VÝPOČET SIGNÁLU
# ============================
def get_signal(symbol, trade_risk_czk):
    try:
        # stažení dat
        data = yf.download(symbol, period=f"{lookback_days}d", interval=interval)
        if data.empty:
            return {"instrument": symbol, "error": "no data"}
        close = data['Close'].squeeze()

        # indikátory
        data['EMA10'] = EMAIndicator(close, window=10).ema_indicator()
        data['EMA50'] = EMAIndicator(close, window=50).ema_indicator()
        data['RSI'] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['EMA_diff'] = data['EMA10'] - data['EMA50']
        data['MACD_diff'] = data['MACD'] - data['MACD_signal']

        features = data[['EMA_diff', 'RSI', 'MACD_diff']].fillna(0)
        data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

        # model
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(features, data['target'])

        latest = features.iloc[-1].values.reshape(1, -1)
        proba = model.predict_proba(latest)[0]
        signal = "LONG" if proba[1] > 0.6 else "SHORT" if proba[0] > 0.6 else "HOLD"
        latest_close = float(data['Close'].iloc[-1])
        probability_percent = max(proba) * 100

        # filtr podle pravděpodobnosti
        if probability_percent < prob_threshold:
            return None  # slabý signál, ignorujeme

        # stop loss / take profit
        if signal == "LONG":
            sl = latest_close * 0.995  # -0.5%
            tp = latest_close * 1.015  # +1.5%
        elif signal == "SHORT":
            sl = latest_close * 1.005  # +0.5%
            tp = latest_close * 0.985  # -1.5%
        else:
            sl = None
            tp = None

        # bezpečný výpočet profit_CZK
        if tp is not None and sl is not None:
            profit_CZK = abs(tp - latest_close) / latest_close * trade_risk_czk * 1000
        else:
            profit_CZK = None

        return {
            "instrument": symbol,
            "price": latest_close,
            "signal": signal,
            "SL": sl,
            "TP": tp,
            "probability": probability_percent,
            "profit_CZK": profit_CZK
        }

    except Exception as e:
        return {"instrument": symbol, "error": str(e)}

# ============================
# GENEROVÁNÍ SIGNÁLŮ
# ============================
results = []
for i, instrument in enumerate(instruments):
    result = get_signal(instrument, trade_risks[i])
    if result is not None:  # filtr silných signálů
        results.append(result)

# ============================
# STREAMLIT ZOBRAZENÍ
# ============================
if results:
    df_results = pd.DataFrame(results)
    st.table(df_results)

    # ============================
    # ULOŽENÍ HISTORIE
    # ============================
    csv_path = os.path.join(DATA_DIR, "signals_history.csv")
    if os.path.exists(csv_path):
        df_results.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_results.to_csv(csv_path, index=False)

    st.write("📈 Signály zpracovány a uloženy!")
else:
    st.write("Žádné silné signály pro dnešní data.")
