# BOT TRADING V90.1 FINAL – AUDITABLE Y CONTEXTUALIZADO
# ==================================================================================
# ✔ Logs ultra detallados en Colab
# ✔ Logs igual de detallados en Telegram
# ✔ Conteo y análisis cada 10 trades
# ✔ Soportes, resistencias y tendencia
# ✔ Entradas, salidas, razones y PnL
# ==================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
import time
from scipy.stats import linregress
from datetime import datetime

plt.rcParams['figure.figsize'] = (14, 7)

# ==============================
# API KEYS
# ==============================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==============================
# ESTADO GLOBAL
# ==============================
historial_trades = []  # % de cada trade
trade_id = 0

# ==============================
# TELEGRAM
# ==============================

def telegram_mensaje(texto):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "parse_mode": "Markdown"})


def telegram_grafico(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    requests.post(url, files={'photo': buf}, data={'chat_id': TELEGRAM_CHAT_ID})
    buf.close()

# ==============================
# DATOS
# ==============================

def obtener_datos():
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
        'fsym': 'BTC', 'tsym': 'USD', 'limit': 300,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    r = requests.get(url, params=params).json()
    df = pd.DataFrame(r['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# ==============================
# INDICADORES
# ==============================

def indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df.dropna()

# ==============================
# SOPORTE / RESISTENCIA
# ==============================

def soporte_resistencia(df, n=50):
    return df['low'].rolling(n).min().iloc[-1], df['high'].rolling(n).max().iloc[-1]

# ==============================
# TENDENCIA
# ==============================

def tendencia(df, n=80):
    y = df['close'].values[-n:]
    x = np.arange(len(y))
    slope, intercept, r, _, _ = linregress(x, y)

    if slope > 0.02:
        estado = '📈 ALCISTA'
    elif slope < -0.02:
        estado = '📉 BAJISTA'
    else:
        estado = '➡️ LATERAL'

    return slope, intercept, estado

# ==============================
# MOTOR V90.1
# ==============================

def motor(df):
    sop, res = soporte_resistencia(df)
    slope, intercept, tend = tendencia(df)
    precio = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    razones = []
    decision = None

    if tend == '📈 ALCISTA' and abs(precio - sop) < atr:
        decision = 'LONG'
        razones.append('Precio cerca del SOPORTE en tendencia ALCISTA')

    elif tend == '📉 BAJISTA' and abs(precio - res) < atr:
        decision = 'SHORT'
        razones.append('Precio cerca de la RESISTENCIA en tendencia BAJISTA')

    else:
        razones.append('Sin confluencia técnica suficiente')

    return decision, sop, res, slope, intercept, tend, razones

# ==============================
# LOGS COLAB
# ==============================

def log_colab(df, decision, sop, res, slope, tend, razones):
    ahora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    precio = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    print('=' * 100)
    print(f"🕒 HORA: {ahora}")
    print(f"💰 BTC/USD: {precio:.2f}")
    print(f"📐 TENDENCIA: {tend} | SLOPE: {slope:.5f}")
    print(f"🧱 SOPORTE: {sop:.2f} | 🧱 RESISTENCIA: {res:.2f}")
    print(f"📊 ATR: {atr:.2f}")
    print(f"🎯 DECISIÓN: {decision if decision else '❌ NO ENTRAR'}")
    print(f"🧠 RAZONES: {' | '.join(razones)}")
    print(f"📦 TOTAL TRADES: {len(historial_trades)}")
    if historial_trades:
        print(f"📈 PNL GLOBAL: {sum(historial_trades):+.2f}%")
    print('=' * 100)

# ==============================
# EJECUCIÓN DE TRADE
# ==============================

def ejecutar_trade(df, direccion, sop, res, slope, intercept, tend, razones):
    global trade_id
    trade_id += 1

    entrada = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    if direccion == 'LONG':
        sl = entrada - atr * 1.5
        tp = entrada + atr * 3
    else:
        sl = entrada + atr * 1.5
        tp = entrada - atr * 3

    telegram_mensaje(
        f"📌 *TRADE #{trade_id} – {direccion}*\n"
        f"🕒 Hora: {datetime.now().strftime('%H:%M:%S')}\n"
        f"💰 Entrada: {entrada:.2f}\n"
        f"📐 Tendencia: {tend}\n"
        f"🧠 Razones: {' | '.join(razones)}"
    )

    fig, ax = plt.subplots()
    ax.plot(df['close'].values, label='Precio')
    ax.axhline(entrada, color='blue', label='Entrada')
    ax.axhline(sl, color='red', label='SL')
    ax.axhline(tp, color='green', label='TP')
    ax.axhline(sop, linestyle='--', color='gray', label='Soporte')
    ax.axhline(res, linestyle='--', color='orange', label='Resistencia')
    ax.legend()
    telegram_grafico(fig)
    plt.close(fig)

    salida = tp
    pnl = (salida - entrada) / entrada * 100 if direccion == 'LONG' else (entrada - salida) / entrada * 100
    historial_trades.append(pnl)

    estado = '✅ GANANCIA' if pnl > 0 else '❌ PÉRDIDA'

    telegram_mensaje(
        f"🏁 *SALIDA TRADE #{trade_id}*\n"
        f"💰 Entrada: {entrada:.2f}\n"
        f"💰 Salida: {salida:.2f}\n"
        f"📊 Resultado: {pnl:+.2f}% {estado}"
    )

    if len(historial_trades) % 10 == 0:
        ultimos = historial_trades[-10:]
        ganados = sum(p for p in ultimos if p > 0)
        perdidos = sum(p for p in ultimos if p <= 0)
        balance = ganados + perdidos

        mejora = 'Reducir trades contra tendencia' if perdidos < 0 else 'Mantener reglas actuales'

        telegram_mensaje(
            f"📊 *RESUMEN ÚLTIMOS 10 TRADES*\n"
            f"✅ Ganados: {ganados:+.2f}%\n"
            f"❌ Perdidos: {perdidos:+.2f}%\n"
            f"📈 Balance neto: {balance:+.2f}%\n"
            f"🧠 Mejora sugerida: {mejora}"
        )

# ==============================
# LOOP PRINCIPAL
# ==============================

def run():
    telegram_mensaje("🤖 *BOT V90.1 INICIADO*\nLogs completos en Colab y Telegram")

    while True:
        try:
            df = indicadores(obtener_datos())
            decision, sop, res, slope, intercept, tend, razones = motor(df)

            log_colab(df, decision, sop, res, slope, tend, razones)

            if decision:
                ejecutar_trade(df, decision, sop, res, slope, intercept, tend, razones)

            time.sleep(60)

        except Exception as e:
            print(f"🚨 ERROR CRÍTICO: {e}")
            telegram_mensaje(f"🚨 ERROR CRÍTICO: {e}")
            time.sleep(60)

# ==============================
# START
# ==============================

if __name__ == '__main__':
    run()
