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
# PAPER TRADING (SIMULACIÓN)
# ======================================================

PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PNL_GLOBAL = 0.0
PAPER_TRADES = []
PAPER_POSICION_ACTIVA = None
PAPER_PRECIO_ENTRADA = None
PAPER_DECISION_ACTIVA = None
PAPER_TIME_ENTRADA = None
PAPER_SIZE_USD = 0.0
PAPER_SIZE_BTC = 0.0
PAPER_SL = None
PAPER_TP = None
PAPER_ULTIMO_RESULTADO = None
PAPER_ULTIMO_PNL = 0.0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
PAPER_MAX_DRAWDOWN = 0.0
PAPER_BALANCE_MAX = PAPER_BALANCE_INICIAL

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

    try:
        data_json = r.json()
    except Exception:
        raise Exception(f"Bybit devolvió respuesta no-JSON: {r.text}")

    # ======================================================
    # VALIDACIONES FUERTES (ANTI ERROR 'list')
    # ======================================================

    if not isinstance(data_json, dict):
        raise Exception(f"Bybit devolvió JSON no dict: {type(data_json)} | {data_json}")

    if "retCode" in data_json and data_json["retCode"] != 0:
        raise Exception(
            f"Bybit Error retCode={data_json.get('retCode')} "
            f"retMsg={data_json.get('retMsg')} "
            f"result={data_json.get('result')}"
        )

    if "result" not in data_json:
        raise Exception(f"Respuesta inválida Bybit (sin result): {data_json}")

    if not isinstance(data_json["result"], dict):
        raise Exception(
            f"Bybit devolvió result como {type(data_json['result'])} en vez de dict: {data_json['result']}"
        )

    if "list" not in data_json["result"]:
        raise Exception(f"Bybit result sin 'list': {data_json['result']}")

    if not isinstance(data_json["result"]["list"], list):
        raise Exception(
            f"Bybit devolvió result['list'] como {type(data_json['result']['list'])} en vez de list: {data_json['result']['list']}"
        )

    data = data_json["result"]["list"][::-1]

    if len(data) == 0:
        raise Exception(f"Bybit devolvió lista vacía de velas: {data_json}")

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
# GRÁFICO VELAS JAPONESAS + SOPORTE/RESISTENCIA + TENDENCIA
# ======================================================

def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones):
    """
    Genera gráfico de velas japonesas con:
    - Soporte (línea horizontal)
    - Resistencia (línea horizontal)
    - Línea de tendencia inclinada (según slope)
    - EMA20 (opcional)
    - Marcador exacto en la vela de entrada
    """

    try:
        df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)

        if df_plot.empty:
            return None

        # ======================================================
        # DATOS
        # ======================================================

        times = df_plot.index
        opens = df_plot['open'].values
        highs = df_plot['high'].values
        lows = df_plot['low'].values
        closes = df_plot['close'].values

        x = np.arange(len(df_plot))

        # ======================================================
        # CREAR FIGURA
        # ======================================================

        fig, ax = plt.subplots(figsize=(14, 7))

        # ======================================================
        # VELAS JAPONESAS (MATPLOTLIB PURO)
        # ======================================================

        for i in range(len(df_plot)):
            color = 'green' if closes[i] >= opens[i] else 'red'

            # Mecha
            ax.vlines(x[i], lows[i], highs[i], color=color, linewidth=1)

            # Cuerpo
            cuerpo_y = min(opens[i], closes[i])
            cuerpo_h = abs(closes[i] - opens[i])

            if cuerpo_h == 0:
                cuerpo_h = 0.0001

            rect = plt.Rectangle(
                (x[i] - 0.3, cuerpo_y),
                0.6,
                cuerpo_h,
                color=color,
                alpha=0.9
            )
            ax.add_patch(rect)

        # ======================================================
        # SOPORTE / RESISTENCIA (LÍNEAS HORIZONTALES)
        # ======================================================

        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")

        # ======================================================
        # EMA20
        # ======================================================

        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')

        # ======================================================
        # LÍNEA DE TENDENCIA INCLINADA (REGRESIÓN LINEAL)
        # ======================================================

        # Creamos tendencia solo sobre el rango graficado
        tendencia_y = intercept + slope * np.arange(len(df))

        # Ajustar para que coincida con el rango mostrado
        # intercept original es para el dataset completo, así que recalculamos sobre df_plot
        y_plot = df_plot['close'].values
        x_plot = np.arange(len(y_plot))
        slope_plot, intercept_plot, r_plot, _, _ = linregress(x_plot, y_plot)

        tendencia_linea = intercept_plot + slope_plot * x_plot

        ax.plot(x_plot, tendencia_linea, color='white', linewidth=2, linestyle='-', label=f"Tendencia slope {slope_plot:.4f}")

        # ======================================================
        # MARCAR VELA DE ENTRADA (ÚLTIMA VELA CERRADA)
        # ======================================================

        entrada_x = len(df_plot) - 1
        entrada_precio = closes[-1]
        entrada_time = times[-1]

        if decision == 'Buy':
            ax.scatter(entrada_x, entrada_precio, s=200, marker='^', color='lime', edgecolors='black', linewidths=1.5, label='Entrada BUY')
            ax.axvline(entrada_x, color='lime', linestyle=':', linewidth=2)
        elif decision == 'Sell':
            ax.scatter(entrada_x, entrada_precio, s=200, marker='v', color='red', edgecolors='black', linewidths=1.5, label='Entrada SELL')
            ax.axvline(entrada_x, color='red', linestyle=':', linewidth=2)

        # ======================================================
        # TEXTO DE ENTRADA
        # ======================================================

        texto_entrada = (
            f"{decision.upper()}\n"
            f"Precio: {entrada_precio:.2f}\n"
            f"Hora: {entrada_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Razones: {', '.join(razones)}"
        )

        ax.text(
            0.02,
            0.98,
            texto_entrada,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.6)
        )

        # ======================================================
        # FORMATO
        # ======================================================

        ax.set_title(f"BTCUSDT - Velas Japonesas ({INTERVAL}m) - Entrada {decision}")
        ax.set_xlabel("Velas")
        ax.set_ylabel("Precio")

        ax.grid(True, alpha=0.2)

        # Etiquetas de tiempo (cada 10 velas)
        step = max(1, int(len(df_plot) / 10))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([t.strftime('%H:%M') for t in times[::step]], rotation=45)

        ax.legend(loc='lower left')

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"🚨 ERROR GRAFICO: {e}")
        return None

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
# GRÁFICO DE ENTRADA (VELAS + SOPORTE/RESISTENCIA + TENDENCIA)
# ======================================================

def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones):
    try:
        df_plot = df.copy().tail(120)

        fig, ax = plt.subplots(figsize=(14, 7))

        # Velas japonesas manuales
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            o = row['open']
            h = row['high']
            l = row['low']
            c = row['close']

            # mecha
            ax.plot([i, i], [l, h], color='black', linewidth=1)

            # cuerpo
            if c >= o:
                ax.plot([i, i], [o, c], color='green', linewidth=6)
            else:
                ax.plot([i, i], [o, c], color='red', linewidth=6)

        # soporte y resistencia horizontales
        ax.axhline(soporte, color='blue', linestyle='--', linewidth=2, label='Soporte')
        ax.axhline(resistencia, color='purple', linestyle='--', linewidth=2, label='Resistencia')

        # tendencia inclinada basada en slope/intercept
        y_trend = intercept + slope * np.arange(len(df_plot))
        ax.plot(np.arange(len(df_plot)), y_trend, color='orange', linewidth=2, label='Tendencia')

        # marcar vela de entrada (última)
        entrada_index = len(df_plot) - 1
        precio_entrada = df_plot['close'].iloc[-1]

        ax.axvline(entrada_index, color='gold', linestyle='-', linewidth=2, label='Entrada')
        ax.scatter([entrada_index], [precio_entrada], color='gold', s=150, marker='o')

        # Texto de entrada
        texto_entrada = (
            f"{decision.upper()}
"
            f"Precio: {precio_entrada:.2f}
"
            f"Balance: {PAPER_BALANCE:.2f} USD
"
            f"PnL Global: {PAPER_PNL_GLOBAL:.4f} USD
"
            f"Razones: {', '.join(razones)}"
        )

        ax.text(
            0.02,
            0.98,
            texto_entrada,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
        )

        ax.set_title(f"{SYMBOL} - Entrada PAPER {decision}")
        ax.set_xlabel("Velas")
        ax.set_ylabel("Precio")
        ax.grid(True)
        ax.legend()

        return fig

    except Exception as e:
        print(f"🚨 ERROR GRÁFICO: {e}")
        return None

# ======================================================
# LOOP PRINCIPAL
# ======================================================


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
                atr_actual = df['atr'].iloc[-1]
                tiempo_actual = df.index[-1]

                # ======================================================
                # ABRIR POSICIÓN PAPER
                # ======================================================

                apertura = paper_abrir_posicion(
                    decision=decision,
                    precio=precio,
                    atr=atr_actual,
                    soporte=soporte,
                    resistencia=resistencia,
                    razones=razones,
                    tiempo=tiempo_actual
                )

                pnl_flotante = paper_calcular_pnl(precio)

                mensaje = (
                    f"📌 ENTRADA PAPER {decision}
"
                    f"💰 Precio: {precio:.2f}
"
                    f"📍 SL: {PAPER_SL:.2f} | TP: {PAPER_TP:.2f}
"
                    f"📦 Size USD: {PAPER_SIZE_USD:.2f} | Size BTC: {PAPER_SIZE_BTC:.6f}
"
                    f"💵 Balance: {PAPER_BALANCE:.2f} USD
"
                    f"📈 PnL flotante: {pnl_flotante:.4f} USD
"
                    f"📊 PnL Global: {PAPER_PNL_GLOBAL:.4f} USD
"
                    f"🧠 {', '.join(razones)}"
                )

                telegram_mensaje(mensaje)

                # ======================================================
                # GENERAR Y ENVIAR GRÁFICO A TELEGRAM AL ENTRAR
                # ======================================================

                fig = generar_grafico_entrada(
                    df=df,
                    decision=decision,
                    soporte=soporte,
                    resistencia=resistencia,
                    slope=slope,
                    intercept=intercept,
                    razones=razones
                )

                if fig:
                    telegram_grafico(fig)
                    plt.close(fig)

                trades_hoy += 1

            # ======================================================
            # REVISAR STOP LOSS / TAKE PROFIT PAPER
            # ======================================================

            if PAPER_POSICION_ACTIVA is not None:
                precio_actual = df['close'].iloc[-1]
                cierre = paper_revisar_sl_tp(precio_actual)

                if cierre:
                    mensaje_cierre = (
                        f"📌 CIERRE PAPER {cierre['decision']} ({cierre['motivo']})
"
                        f"📍 Entrada: {cierre['entrada']:.2f}
"
                        f"📍 Salida: {cierre['salida']:.2f}
"
                        f"💰 PnL Trade: {cierre['pnl']:.4f} USD
"
                        f"💵 Balance: {cierre['balance']:.2f} USD
"
                        f"📊 PnL Global: {PAPER_PNL_GLOBAL:.4f} USD
"
                        f"🏆 Wins: {PAPER_WIN} | ❌ Loss: {PAPER_LOSS}
"
                        f"📉 Max Drawdown: {PAPER_MAX_DRAWDOWN:.4f} USD"
                    )

                    telegram_mensaje(mensaje_cierre)

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
