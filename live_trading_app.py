import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier

# ============================
# INSTRUMENTY
# ============================
# ticker: název
instrument_names = {
    "^N225": "JP225",
    "EUR=X": "EU50",
    "CC=F": "COCOA",
    "SI=F": "SILVER",
    "GC=F": "GOLD",
    "CL=F": "CRUDE OIL",
    "NG=F": "NAT GAS",
    "ZC=F": "CORN",
    "ZS=F": "SOYBEAN",
    "ZW=F": "WHEAT",
    "SB=F": "SUGAR",
    "HG=F": "COPPER",
    "PL=F": "PLATINUM",
    "PA=F": "PALLADIUM",
    "RB=F": "RBOB GASOLINE",
    # poslední HG=F duplicate → COPPER již zahrnuto
}

instruments = list(instrument_names.keys())

# odpovídající investice pro XTB CFD (při objemu 0.01)
trade_risks = [
    1678, 710, 1215, 7479,
    2000, 1500, 1200, 1000,
    1100, 950, 800, 1300,
    2500, 2700, 1800
]

lookback_days = 30
interval = "30m"
prob_threshold = 80  # filtr silných signálů

# ============================
# FUNKCE PRO VÝPOČET SIGNÁLU
# ============================
def get_signal(symbol, trade_risk_czk):
    """
    Vrací dict se signálem pro jeden instrument:
    - název instrumentu
    - aktuální cena
    - LONG / SHORT / HOLD
    - SL / TP
    - pravděpodobnost
    - odhad profit v CZK
    """
    try:
        # Stažení historických dat
        data = yf.download(symbol, period=f"{lookback_days}d", interval=interval)
        if data.empty:
            return {"instrument": instrument_names.get(symbol, symbol), "error": "no data"}

        close = data['Close'].squeeze()

        # Indikátory
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

        # Model
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(features, data['target'])

        latest = features.iloc[-1].values.reshape(1, -1)
        proba = model.predict_proba(latest)[0]
        signal = "LONG" if proba[1] > 0.6 else "SHORT" if proba[0] > 0.6 else "HOLD"
        latest_close = float(data['Close'].iloc[-1])
        probability_percent = max(proba) * 100

        # Filtr slabých signálů
        if probability_percent < prob_threshold or signal == "HOLD":
            return None

        # SL / TP
        if signal == "LONG":
            sl = latest_close * 0.995  # -0.5%
            tp = latest_close * 1.015  # +1.5%
        elif signal == "SHORT":
            sl = latest_close * 1.005  # +0.5%
            tp = latest_close * 0.985  # -1.5%
        else:
            sl = tp = None

        # Potenciální profit v CZK
        profit_CZK = abs(tp - latest_close) / latest_close * trade_risk_czk * 1000 if tp and sl else None

        return {
            "instrument": instrument_names.get(symbol, symbol),
            "price": latest_close,
            "signal": signal,
            "SL": sl,
            "TP": tp,
            "probability": probability_percent,
            "profit_CZK": profit_CZK
        }

    except Exception as e:
        return {"instrument": instrument_names.get(symbol, symbol), "error": str(e)}

# ============================
# FUNKCE PRO VŠECHNY INSTRUMENTY
# ============================
def get_all_signals():
    """
    Vrací seznam silných signálů pro všechny instrumenty
    """
    results = []
    for i, symbol in enumerate(instruments):
        result = get_signal(symbol, trade_risks[i])
        if result is not None:
            results.append(result)
    return results
