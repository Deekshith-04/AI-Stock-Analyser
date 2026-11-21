import pandas as pd
import ta

from services.market_data_service import get_stock_history


def get_technical_indicators(ticker: str):
    """
    Calculate technical indicators from live/historical Yahoo Finance data.

    Indicators:
    - RSI (14)
    - MACD
    - 50-day moving average
    - 200-day moving average
    - Support
    - Resistance
    """

    data = get_stock_history(
        ticker=ticker,
        period="1y",
        interval="1d",
    )

    if data is None or data.empty:
        raise ValueError(f"No historical data available for {ticker}")

    data = data.copy()

    # ---------------------------------------------------------
    # Make sure required columns are numeric
    # ---------------------------------------------------------

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    data = data.dropna(subset=["Close"])

    if len(data) < 50:
        raise ValueError(
            f"Not enough historical data to calculate technical indicators for {ticker}"
        )

    # ---------------------------------------------------------
    # RSI - 14 period
    # ---------------------------------------------------------

    rsi_indicator = ta.momentum.RSIIndicator(
        close=data["Close"],
        window=14,
    )

    data["RSI_14"] = rsi_indicator.rsi()

    # ---------------------------------------------------------
    # MACD
    # ---------------------------------------------------------

    macd_indicator = ta.trend.MACD(
        close=data["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )

    data["MACD"] = macd_indicator.macd()
    data["MACD_SIGNAL"] = macd_indicator.macd_signal()
    data["MACD_HIST"] = macd_indicator.macd_diff()

    # ---------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------

    data["SMA_50"] = data["Close"].rolling(
        window=50
    ).mean()

    data["SMA_200"] = data["Close"].rolling(
        window=200
    ).mean()

    # ---------------------------------------------------------
    # Support / Resistance
    #
    # We use recent 20 trading sessions:
    # Support    = lowest low
    # Resistance = highest high
    # ---------------------------------------------------------

    data["Support_20D"] = data["Low"].rolling(
        window=20
    ).min()

    data["Resistance_20D"] = data["High"].rolling(
        window=20
    ).max()

    # ---------------------------------------------------------
    # Latest row
    # ---------------------------------------------------------

    latest = data.iloc[-1]

    close_price = float(latest["Close"])

    rsi_value = latest["RSI_14"]

    macd_value = latest["MACD"]
    macd_signal = latest["MACD_SIGNAL"]
    macd_hist = latest["MACD_HIST"]

    sma50_value = latest["SMA_50"]
    sma200_value = latest["SMA_200"]

    support_value = latest["Support_20D"]
    resistance_value = latest["Resistance_20D"]

    # ---------------------------------------------------------
    # RSI interpretation
    # ---------------------------------------------------------

    if pd.isna(rsi_value):
        rsi_status = "Unavailable"
    elif rsi_value >= 70:
        rsi_status = "Overbought"
    elif rsi_value <= 30:
        rsi_status = "Oversold"
    elif rsi_value >= 50:
        rsi_status = "Bullish"
    else:
        rsi_status = "Bearish"

    # ---------------------------------------------------------
    # MACD interpretation
    # ---------------------------------------------------------

    if pd.isna(macd_value) or pd.isna(macd_signal):
        macd_status = "Unavailable"
    elif macd_hist > 0:
        macd_status = "Bullish"
    elif macd_hist < 0:
        macd_status = "Bearish"
    else:
        macd_status = "Neutral"

    # ---------------------------------------------------------
    # Helper for safe float conversion
    # ---------------------------------------------------------

    def safe_float(value):
        if pd.isna(value):
            return None
        return round(float(value), 2)

    # ---------------------------------------------------------
    # Final response
    # ---------------------------------------------------------

    return {
        "ticker": ticker,
        "date": latest["Date"].strftime("%Y-%m-%d")
        if "Date" in data.columns
        else None,

        "price": round(close_price, 2),

        "rsi": {
            "value": safe_float(rsi_value),
            "status": rsi_status,
        },

        "macd": {
            "value": safe_float(macd_value),
            "signal": safe_float(macd_signal),
            "histogram": safe_float(macd_hist),
            "status": macd_status,
        },

        "moving_averages": {
            "sma50": safe_float(sma50_value),
            "sma200": safe_float(sma200_value),
        },

        "support_resistance": {
            "support": safe_float(support_value),
            "resistance": safe_float(resistance_value),
        },
    }