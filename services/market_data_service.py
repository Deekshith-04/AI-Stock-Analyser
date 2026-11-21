import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd

# ---------------------------------------------------------
# Short-lived market data cache
# ---------------------------------------------------------

MARKET_CACHE = {}
MARKET_CACHE_TTL = 120  # seconds
HISTORY_CACHE = {}
HISTORY_CACHE_TTL = 300  # seconds
ALL_STOCKS_CACHE = {'timestamp': 0.0, 'data': None}
MARKET_CAP_CACHE = {}


def get_cached_market_data(ticker: str):
    cached = MARKET_CACHE.get(ticker)

    if cached is None:
        return None

    timestamp, data = cached

    if time.time() - timestamp < MARKET_CACHE_TTL:
        return data

    MARKET_CACHE.pop(ticker, None)
    return None


def set_cached_market_data(ticker: str, data):
    MARKET_CACHE[ticker] = (time.time(), data)


# ---------------------------------------------------------
# Indian stock sector fallback
# ---------------------------------------------------------

SECTOR_FALLBACK = {
    "RELIANCE": "Energy",
    "TCS": "Information Technology",
    "INFY": "Information Technology",
    "HDFCBANK": "Financial Services",
    "ICICIBANK": "Financial Services",
    "SBIN": "Financial Services",
    "ITC": "Consumer Defensive",
    "LT": "Industrials",
    "BHARTIARTL": "Communication Services",
    "AXISBANK": "Financial Services",
    "HINDUNILVR": "Consumer Defensive",
    "MARUTI": "Consumer Cyclical",
    "WIPRO": "Information Technology",
    "TATASTEEL": "Basic Materials",
    "BAJFINANCE": "Financial Services",
    "TECHM": "Information Technology",
    "ASIANPAINT": "Basic Materials",
}


# ---------------------------------------------------------
# Live sector universe
# ---------------------------------------------------------

SECTOR_UNIVERSE = {
    "Energy": [
        "RELIANCE.NS",
        "ONGC.NS",
    ],

    "Information Technology": [
        "TCS.NS",
        "INFY.NS",
        "WIPRO.NS",
        "TECHM.NS",
        "HCLTECH.NS",
    ],

    "Financial Services": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "AXISBANK.NS",
        "KOTAKBANK.NS",
        "BAJFINANCE.NS",
        "BAJAJFINSV.NS",
        "JIOFIN.NS",
        "SBILIFE.NS",
        "SHRIRAMFIN.NS",
        "HDFCLIFE.NS",
    ],

    "Consumer Defensive": [
        "ITC.NS",
        "HINDUNILVR.NS",
        "NESTLEIND.NS",
        "TATACONSUM.NS",
    ],

    "Consumer Cyclical": [
        "MARUTI.NS",
        "TITAN.NS",
        "TRENT.NS",
    ],

    "Basic Materials": [
        "TATASTEEL.NS",
        "HINDALCO.NS",
        "JSWSTEEL.NS",
        "ASIANPAINT.NS",
    ],

    "Industrials": [
        "LT.NS",
        "BEL.NS",
        "POWERGRID.NS",
        "NTPC.NS",
    ],

    "Healthcare": [
        "SUNPHARMA.NS",
        "DRREDDY.NS",
        "CIPLA.NS",
        "APOLLOHOSP.NS",
        "MAXHEALTH.NS",
    ],

    "Communication Services": [
        "BHARTIARTL.NS",
    ],

    "Automotive": [
        "BAJAJ-AUTO.NS",
        "EICHERMOT.NS",
        "M&M.NS",
        "TMPV.NS",
    ],

    "Cement": [
        "ULTRACEMCO.NS",
        "GRASIM.NS",
    ],

    "Metals": [
        "COALINDIA.NS",
    ],

    "Unknown": [
        "ADANIENT.NS",
        "ADANIPORTS.NS",
        "INDIGO.NS",
        "ETERNAL.NS",
    ],
}


# ---------------------------------------------------------
# Stock historical data
# ---------------------------------------------------------

def get_stock_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
):
    cache_key = (ticker.upper(), period, interval)
    cached = HISTORY_CACHE.get(cache_key)
    if cached and (time.time() - cached[0] < HISTORY_CACHE_TTL):
        return cached[1].copy()

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No market data found for ticker: {ticker}"
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        c
        for c in required_columns
        if c not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing market-data columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    ).dt.tz_localize(None)

    HISTORY_CACHE[cache_key] = (time.time(), data.copy())
    return data
# ---------------------------------------------------------
# NIFTY 50 historical data
# ---------------------------------------------------------

def get_nifty50_history(
    period: str = "1y",
    interval: str = "1d",
):
    cache_key = ("^NSEI", period, interval)
    cached = HISTORY_CACHE.get(cache_key)
    if cached and (time.time() - cached[0] < HISTORY_CACHE_TTL):
        return cached[1].copy()

    data = yf.download(
        "^NSEI",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No NIFTY 50 market data found."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing NIFTY columns: {missing_columns}"
        )

    data = data[required_columns].copy()
    data["Date"] = pd.to_datetime(
        data["Date"]
    ).dt.tz_localize(None)

    return data


# ---------------------------------------------------------
# Latest stock quote
# ---------------------------------------------------------

def get_latest_stock_quote(ticker: str):
    """
    Get the latest available quote and basic
    day-level market information.
    """
    history = get_stock_history(
        ticker=ticker,
        period="5d",
        interval="1d",
    )

    if len(history) == 0:
        raise ValueError(
            f"No recent data found for {ticker}"
        )

    # Remove rows where Close is missing.
    history = history.dropna(subset=["Close"])

    if len(history) == 0:
        raise ValueError(
            f"No valid closing price found for {ticker}"
        )

    latest = history.iloc[-1]

    if len(history) >= 2:
        previous_close = history.iloc[-2]["Close"]
    else:
        previous_close = latest["Close"]

    close = float(latest["Close"])
    previous_close = float(previous_close)

    change = close - previous_close

    change_percent = (
        (change / previous_close) * 100
        if previous_close != 0
        else 0.0
    )

    def clean_number(value):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if (
            pd.isna(value)
            or value == float("inf")
            or value == float("-inf")
        ):
            return None

        return value

    volume_value = latest["Volume"]

    if pd.isna(volume_value):
        volume = 0
    else:
        volume = int(volume_value)

    return {
        "ticker": ticker.upper(),
        "date": str(
            pd.to_datetime(latest["Date"]).date()
        ),
        "open": clean_number(latest["Open"]),
        "high": clean_number(latest["High"]),
        "low": clean_number(latest["Low"]),
        "close": clean_number(close),
        "volume": volume,
        "previous_close": clean_number(previous_close),
        "change": clean_number(change),
        "change_percent": clean_number(change_percent),
    }
# ---------------------------------------------------------
# Live stock market data
# ---------------------------------------------------------

def get_stock_market_data(ticker: str):
    """
    Return stock data with market/company information
    for the existing React Market UI.
    """
    ticker = ticker.upper()

    # Check short-lived cache first.
    cached_data = get_cached_market_data(ticker)

    if cached_data is not None:
        return cached_data

    # Latest market quote.
    quote = get_latest_stock_quote(ticker)

    # Yahoo Finance company information.
    stock = yf.Ticker(ticker)

    try:
        info = stock.info
    except Exception:
        info = {}

    # Basic company information.
    symbol = ticker.replace(".NS", "")

    company_name = (
        info.get("longName")
        or info.get("shortName")
        or symbol
    )

    # Sector.
    sector = (
        info.get("sector")
        or SECTOR_FALLBACK.get(symbol)
        or "Unknown"
    )

    # Market capitalization.
    market_cap_value = info.get("marketCap")

    try:
        market_cap_value = float(market_cap_value)

        if (
            pd.isna(market_cap_value)
            or market_cap_value == float("inf")
            or market_cap_value == float("-inf")
        ):
            market_cap_value = 0

    except (TypeError, ValueError, OverflowError):
        market_cap_value = 0

    if market_cap_value:
        market_cap = (
            f"₹{market_cap_value / 1_000_000_000_000:.2f} Lakh Cr"
        )
    else:
        market_cap = "N/A"

    market_cap_num = market_cap_value

    # Safely convert Yahoo Finance numeric values.
    def safe_info_number(value):
        try:
            value = float(value)

            if (
                pd.isna(value)
                or value == float("inf")
                or value == float("-inf")
            ):
                return None

            return value

        except (TypeError, ValueError, OverflowError):
            return None

    # 52-week range.
    fifty_two_week_high = safe_info_number(
        info.get("fiftyTwoWeekHigh")
    )

    fifty_two_week_low = safe_info_number(
        info.get("fiftyTwoWeekLow")
    )

    # P/E ratio.
    pe_ratio = safe_info_number(
        info.get("trailingPE")
    )

    # Trend.
    if quote["change_percent"] is None:
        trend = "Neutral"
    elif quote["change_percent"] > 0:
        trend = "Bullish"
    elif quote["change_percent"] < 0:
        trend = "Bearish"
    else:
        trend = "Neutral"

    result = {
        "symbol": symbol,
        "companyName": company_name,
        "sector": sector,
        "price": quote["close"],
        "change": quote["change"],
        "changePercent": quote["change_percent"],
        "volume": quote["volume"],
        "volumeFormatted": f'{quote["volume"]:,}',
        "marketCap": market_cap,
        "marketCapNum": market_cap_num,
        "trend": trend,
        "dayHigh": quote["high"],
        "dayLow": quote["low"],
        "previousClose": quote["previous_close"],
        "fiftyTwoWeekHigh": fifty_two_week_high,
        "fiftyTwoWeekLow": fifty_two_week_low,
        "peRatio": pe_ratio,
    }

    # Store result for 60 seconds.
    set_cached_market_data(ticker, result)

    return result
# ---------------------------------------------------------
# Live market indices
# ---------------------------------------------------------

def get_market_indices():
    """
    Get the latest available values for:
    NIFTY 50, SENSEX and BANK NIFTY.
    """
    index_map = [
        {
            "id": "nifty50",
            "name": "NIFTY 50",
            "symbol": "^NSEI",
        },
        {
            "id": "sensex",
            "name": "SENSEX",
            "symbol": "^BSESN",
        },
        {
            "id": "banknifty",
            "name": "BANK NIFTY",
            "symbol": "^NSEBANK",
        },
    ]

    results = []

    def safe_float(value):
        try:
            value = float(value)

            if (
                pd.isna(value)
                or value == float("inf")
                or value == float("-inf")
            ):
                return None

            return value

        except (TypeError, ValueError, OverflowError):
            return None

    for index in index_map:
        try:
            data = yf.download(
                index["symbol"],
                period="5d",
                interval="1d",
                auto_adjust=False,
                progress=False,
            )

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()

            if "Close" not in data.columns:
                continue

            data = data.dropna(subset=["Close"])

            if data.empty:
                continue

            latest = data.iloc[-1]
            current_value = safe_float(latest["Close"])

            if current_value is None:
                continue

            if len(data) >= 2:
                previous_value = safe_float(
                    data.iloc[-2]["Close"]
                )
            else:
                previous_value = current_value

            if previous_value is None:
                previous_value = current_value

            change = current_value - previous_value

            change_percent = (
                (change / previous_value) * 100
                if previous_value != 0
                else 0.0
            )

            day_high = (
                safe_float(latest["High"])
                if "High" in data.columns
                else None
            )

            day_low = (
                safe_float(latest["Low"])
                if "Low" in data.columns
                else None
            )

            results.append({
                "id": index["id"],
                "name": index["name"],
                "symbol": index["symbol"],
                "value": current_value,
                "change": safe_float(change),
                "changePercent": safe_float(change_percent),
                "positive": change > 0,
                "dayHigh": day_high,
                "dayLow": day_low,
                "date": str(
                    pd.to_datetime(latest["Date"]).date()
                ),
            })

        except Exception as e:
            print(
                f"Failed to fetch {index['symbol']}: {e}"
            )
            continue

    return results
# ---------------------------------------------------------
# Market summary
# ---------------------------------------------------------

MARKET_SUMMARY_TICKERS = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDIGO.NS",
    "INFY.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "MAXHEALTH.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TMPV.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
    "SBIN.NS",
]
def get_market_summary():
    advancing = 0
    declining = 0
    unchanged = 0

    try:
        # Fetch tracked stocks in one Yahoo Finance request.
        data = yf.download(
            MARKET_SUMMARY_TICKERS,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        if data.empty:
            raise ValueError(
                "No market summary data available."
            )

        for ticker in MARKET_SUMMARY_TICKERS:
            try:
                # Handle multi-ticker Yahoo Finance response.
                if ticker not in data.columns.get_level_values(0):
                    continue

                ticker_data = data[ticker].dropna(how="all")

                if (
                    ticker_data.empty
                    or "Close" not in ticker_data.columns
                ):
                    continue

                closes = ticker_data["Close"].dropna()

                if len(closes) == 0:
                    continue

                current_close = float(closes.iloc[-1])

                if len(closes) >= 2:
                    previous_close = float(closes.iloc[-2])
                else:
                    previous_close = current_close

                if previous_close == 0:
                    continue

                change_percent = (
                    (current_close - previous_close)
                    / previous_close
                ) * 100

                if change_percent > 0:
                    advancing += 1
                elif change_percent < 0:
                    declining += 1
                else:
                    unchanged += 1

            except Exception:
                continue

    except Exception:
        # Fall back to individual stock requests.
        for ticker in MARKET_SUMMARY_TICKERS:
            try:
                quote = get_latest_stock_quote(ticker)
                change_percent = quote["change_percent"]

                if change_percent is None:
                    continue
                elif change_percent > 0:
                    advancing += 1
                elif change_percent < 0:
                    declining += 1
                else:
                    unchanged += 1

            except Exception:
                continue

    total = advancing + declining + unchanged

    if total == 0:
        market_sentiment = "Neutral"
        sentiment_score = 50
    else:
        sentiment_score = round(
            (
                advancing + (unchanged * 0.5)
            ) / total * 100
        )

        if sentiment_score >= 60:
            market_sentiment = "Bullish"
        elif sentiment_score <= 40:
            market_sentiment = "Bearish"
        else:
            market_sentiment = "Neutral"

    # Indian market status based on weekday and regular hours.
    # This simple rule does not account for exchange holidays.
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    current_minutes = now.hour * 60 + now.minute

    market_open_time = 9 * 60 + 15
    market_close_time = 15 * 60 + 30
    is_weekday = now.weekday() < 5

    is_market_open = (
        is_weekday
        and market_open_time <= current_minutes < market_close_time
    )

    market_status = "OPEN" if is_market_open else "CLOSED"

    return {
        "status": market_status,
        "session": "Latest Available Market Data",
        "lastUpdated": None,
        "advancing": advancing,
        "declining": declining,
        "unchanged": unchanged,
        "marketSentiment": market_sentiment,
        "sentimentScore": sentiment_score,
        "fiiNet": None,
        "diiNet": None,
        "trackedStocks": total,
    }
# ---------------------------------------------------------
# Historical stock prices
# ---------------------------------------------------------

def get_stock_history_for_chart(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
):
    data = get_stock_history(
        ticker=ticker,
        period=period,
        interval=interval,
    )

    return [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "price": float(row["Close"]),
        }
        for _, row in data.iterrows()
        if pd.notna(row["Close"])
    ]

def get_stock_sector(ticker: str) -> str:
    """Return a stock's sector using the existing fallback mapping."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    return SECTOR_FALLBACK.get(symbol, "Unknown")


COMPANY_NAME_FALLBACK = {
    "ADANIENT": "Adani Enterprises Ltd.",
    "ADANIPORTS": "Adani Ports & SEZ Ltd.",
    "APOLLOHOSP": "Apollo Hospitals Enterprise Ltd.",
    "ASIANPAINT": "Asian Paints Ltd.",
    "AXISBANK": "Axis Bank Ltd.",
    "BAJAJ-AUTO": "Bajaj Auto Ltd.",
    "BAJFINANCE": "Bajaj Finance Ltd.",
    "BAJAJFINSV": "Bajaj Finserv Ltd.",
    "BEL": "Bharat Electronics Ltd.",
    "BHARTIARTL": "Bharti Airtel Ltd.",
    "CIPLA": "Cipla Ltd.",
    "COALINDIA": "Coal India Ltd.",
    "DRREDDY": "Dr. Reddy's Laboratories Ltd.",
    "EICHERMOT": "Eicher Motors Ltd.",
    "ETERNAL": "Eternal Ltd.",
    "GRASIM": "Grasim Industries Ltd.",
    "HCLTECH": "HCL Technologies Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
    "HDFCLIFE": "HDFC Life Insurance Co. Ltd.",
    "HINDALCO": "Hindalco Industries Ltd.",
    "HINDUNILVR": "Hindustan Unilever Ltd.",
    "ICICIBANK": "ICICI Bank Ltd.",
    "INDIGO": "InterGlobe Aviation Ltd.",
    "INFY": "Infosys Ltd.",
    "ITC": "ITC Ltd.",
    "JIOFIN": "Jio Financial Services Ltd.",
    "JSWSTEEL": "JSW Steel Ltd.",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd.",
    "LT": "Larsen & Toubro Ltd.",
    "M&M": "Mahindra & Mahindra Ltd.",
    "MARUTI": "Maruti Suzuki India Ltd.",
    "MAXHEALTH": "Max Healthcare Institute Ltd.",
    "NESTLEIND": "Nestle India Ltd.",
    "NTPC": "NTPC Ltd.",
    "ONGC": "Oil & Natural Gas Corp. Ltd.",
    "POWERGRID": "Power Grid Corp. of India Ltd.",
    "RELIANCE": "Reliance Industries Ltd.",
    "SBILIFE": "SBI Life Insurance Co. Ltd.",
    "SBIN": "State Bank of India",
    "SHRIRAMFIN": "Shriram Finance Ltd.",
    "SUNPHARMA": "Sun Pharmaceutical Industries Ltd.",
    "TATACONSUM": "Tata Consumer Products Ltd.",
    "TMPV": "Tata Motors Passenger Vehicles Ltd.",
    "TATASTEEL": "Tata Steel Ltd.",
    "TCS": "Tata Consultancy Services Ltd.",
    "TECHM": "Tech Mahindra Ltd.",
    "TITAN": "Titan Company Ltd.",
    "TRENT": "Trent Ltd.",
    "ULTRACEMCO": "UltraTech Cement Ltd.",
    "WIPRO": "Wipro Ltd.",
}


def get_all_market_stocks():
    if (
        ALL_STOCKS_CACHE["data"] is not None
        and time.time() - ALL_STOCKS_CACHE["timestamp"] < MARKET_CACHE_TTL
    ):
        return ALL_STOCKS_CACHE["data"]

    ticker_to_sector = {}
    for sec, tickers in SECTOR_UNIVERSE.items():
        for t in tickers:
            ticker_to_sector[t.upper()] = sec

    data = yf.download(
        MARKET_SUMMARY_TICKERS,
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    results = []
    if data is not None and not data.empty:
        top_cols = set(data.columns.get_level_values(0))
        for ticker in MARKET_SUMMARY_TICKERS:
            try:
                if ticker not in top_cols:
                    continue
                t_df = data[ticker].dropna(subset=["Close"])
                if t_df.empty:
                    continue
                latest = t_df.iloc[-1]
                prev = t_df.iloc[-2] if len(t_df) >= 2 else latest
                close = float(latest["Close"])
                prev_close = float(prev["Close"])
                change = close - prev_close
                change_pct = (change / prev_close * 100.0) if prev_close != 0 else 0.0
                vol = int(latest["Volume"]) if pd.notna(latest["Volume"]) else 0
                day_high = float(latest["High"]) if pd.notna(latest["High"]) else close
                day_low = float(latest["Low"]) if pd.notna(latest["Low"]) else close
                symbol = ticker.replace(".NS", "")
                sector = (
                    SECTOR_FALLBACK.get(symbol)
                    or ticker_to_sector.get(ticker.upper())
                    or "Equities"
                )
                if sector == "Unknown":
                    sector = "Equities"
                trend = "Bullish" if change_pct > 0 else ("Bearish" if change_pct < 0 else "Neutral")
                mcap_val = MARKET_CAP_CACHE.get(ticker.upper())
                if mcap_val is None:
                    try:
                        fast_mcap = getattr(yf.Ticker(ticker).fast_info, "market_cap", None)
                        if fast_mcap and not pd.isna(fast_mcap) and fast_mcap > 0:
                            mcap_val = float(fast_mcap)
                            MARKET_CAP_CACHE[ticker.upper()] = mcap_val
                    except Exception:
                        mcap_val = None

                if mcap_val and mcap_val > 0:
                    mcap_str = f"₹{mcap_val / 1_000_000_000_000:.2f} Lakh Cr"
                    mcap_num = mcap_val
                else:
                    mcap_str = "N/A"
                    mcap_num = 0

                item = {
                    "symbol": symbol,
                    "companyName": COMPANY_NAME_FALLBACK.get(symbol, symbol),
                    "sector": sector,
                    "price": round(close, 2),
                    "change": round(change, 2),
                    "changePercent": round(change_pct, 2),
                    "volume": vol,
                    "volumeFormatted": f"{vol:,}",
                    "marketCap": mcap_str,
                    "marketCapNum": mcap_num,
                    "trend": trend,
                    "dayHigh": round(day_high, 2),
                    "dayLow": round(day_low, 2),
                    "previousClose": round(prev_close, 2),
                    "fiftyTwoWeekHigh": None,
                    "fiftyTwoWeekLow": None,
                    "peRatio": None,
                }
                set_cached_market_data(ticker.upper(), item)
                results.append(item)
            except Exception:
                continue

    if results:
        ALL_STOCKS_CACHE["timestamp"] = time.time()
        ALL_STOCKS_CACHE["data"] = results

    return results
