# BOT TRADING V90.2 BYBIT REAL – PRODUCCIÓN (SIN PROXY)
# ======================================================
# ⚠️ KEYS INCLUIDAS TAL CUAL (SEGÚN PEDIDO)
# Diseñado para FUTUROS PERPETUOS BTCUSDT en Bybit
# ======================================================

import os
import time
import io
import hmac
import hashlib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from scipy.stats import linregress
from datetime import datetime, timezone

plt.rcParams['figure.figsize'] = (12, 6)

# ======================================================
# CONFIGURACIÓN GRÁFICOS
# ======================================================

GRAFICO_VELAS_LIMIT = 120  # cantidad de velas para graficar
MOSTRAR_EMA20 = True
MOSTRAR_ATR = False


# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1"  # 1 minuto
RISK_PER_TRADE = 0.0025   # 0.25%
MAX_TRADES_DAY = 3
LEVERAGE = 1
SLEEP_SECONDS = 60

# ======================================================
# CREDENCIALES (SIN MODIFICAR)
# ======================================================

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ BYBIT_API_KEY o BYBIT_API_SECRET no configuradas")

# ======================================================
# BYBIT ENDPOINT
# ======================================================

BASE_URL = "https://api.bybit.com"

# ======================================================
# TELEGRAM (SIN PROXY)
# ======================================================

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": texto},
            timeout=10
        )
    except Exception:
        pass


def telegram_grafico(fig):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(
            url,
            files={'photo': buf},
            data={'chat_id': TELEGRAM_CHAT_ID},
            timeout=15
        )
        buf.close()
    except Exception:
        pass

# ======================================================
# FIRMA BYBIT
# ======================================================

def sign(params):
    query = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(
        BYBIT_API_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()

# ======================================================
# OBTENER VELAS BYBIT (SIN PROXY)
# ======================================================

def obtener_velas(limit=300):
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": limit
    }
    r = requests.get(
        url,
        params=params,
        timeout=20
    )

    if not r.text:
        raise Exception("Respuesta vacía de Bybit")

    data_json = r.json()

    if "result" not in data_json:
        raise Exception(f"Respuesta inválida Bybit: {data_json}")

    data = data_json['result']['list'][::-1]

    df = pd.DataFrame(data, columns=[
        'time','open','high','low','close','volume','turnover'
    ])

    df[['open','high','low','close','volume']] = df[[
        'open','high','low','close','volume'
    ]].astype(float)

    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)

    df.set_index('time', inplace=True)
    return df

# ======================================================
# INDICADORES
# ======================================================

def calcular_indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)

    df['atr'] = tr.rolling(14).mean()
    return df.dropna()

# ======================================================
# SOPORTE / RESISTENCIA
# ======================================================

def detectar_soportes_resistencias(df):
    soporte = df['close'].rolling(50).min().iloc[-1]
    resistencia = df['close'].rolling(50).max().iloc[-1]
    return soporte, resistencia

# ======================================================
# TENDENCIA
# ======================================================

def detectar_tendencia(df, ventana=80):
    y = df['close'].values[-ventana:]
    x = np.arange(len(y))
    slope, intercept, r, _, _ = linregress(x, y)

    if slope > 0.02:
        direccion = '📈 ALCISTA'
    elif slope < -0.02:
        direccion = '📉 BAJISTA'
    else:
        direccion = '➡️ LATERAL'

    return slope, intercept, direccion

# ======================================================
# MOTOR V90
# ======================================================

def motor_v90(df):
    soporte, resistencia = detectar_soportes_resistencias(df)
    slope, intercept, tendencia = detectar_tendencia(df)
    precio = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    razones = []

    if tendencia == '📈 ALCISTA' and abs(precio - soporte) < atr:
        razones.append('Confluencia: soporte + tendencia alcista')
        return 'Buy', soporte, resistencia, razones

    if tendencia == '📉 BAJISTA' and abs(precio - resistencia) < atr:
        razones.append('Confluencia: resistencia + tendencia bajista')
        return 'Sell', soporte, resistencia, razones

    razones.append('Sin confluencia válida')
    return None, soporte, resistencia, razones

# ======================================================
# LOG
# ======================================================

def log_colab(df, tendencia, slope, soporte, resistencia, decision, razones):
    ahora = datetime.now(timezone.utc)
    precio = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    print("="*100)
    print("🧠 Groq Analyst:", "ACTIVO" if client_groq else "DESACTIVADO")
    print(f"🕒 {ahora} | 💰 BTC: {precio:.2f}")
    print(f"📐 Tendencia: {tendencia} | Slope: {slope:.5f}")
    print(f"🧱 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}")
    print(f"📊 ATR: {atr:.2f}")
    print(f"🎯 Decisión: {decision if decision else 'NO TRADE'}")
    print(f"🧠 Razones: {', '.join(razones)}")
    print("="*100)

# ======================================================
# GROQ
# ======================================================

def analizar_con_groq(resumen):
    if not client_groq:
        return None
    prompt = f"""
Eres un trader cuantitativo profesional.
Analiza este resumen de trading y da recomendaciones claras:
{resumen}
Devuelve:
- Diagnóstico
- Qué mejorar
- Qué evitar
"""
    try:
        r = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"Error Groq: {e}"

# ======================================================
# LOOP PRINCIPAL
# ======================================================

def run_bot():
    telegram_mensaje("🤖 BOT V90.2 BYBIT REAL INICIADO (SIN PROXY)")
    trades_hoy = 0

    while True:
        try:
            df = obtener_velas()
            df = calcular_indicadores(df)

            slope, intercept, tendencia = detectar_tendencia(df)
            decision, soporte, resistencia, razones = motor_v90(df)

            log_colab(df, tendencia, slope, soporte, resistencia, decision, razones)

            if decision and trades_hoy < MAX_TRADES_DAY:
                precio = df['close'].iloc[-1]

                mensaje = (
                    f"📌 ENTRADA PAPER {decision}
"
                    f"💰 Precio: {precio:.2f}
"
                    f"🧠 {', '.join(razones)}"
                )

                telegram_mensaje(mensaje)
                trades_hoy += 1

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"🚨 ERROR: {e}")
            telegram_mensaje(f"🚨 ERROR BOT: {e}")
            time.sleep(60)

# ======================================================
# START
# ======================================================

if __name__ == '__main__':
    run_bot()
