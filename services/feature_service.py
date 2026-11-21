import pandas as pd
import numpy as np
import json

from pathlib import Path

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

RF_FEATURE_NAMES_PATH = MODEL_DIR / "rf_feature_names.json"
RF_MEDIANS_PATH = MODEL_DIR / "rf_training_medians.json"


# ============================================================
# LOAD RF PREPROCESSING ARTIFACTS
# ============================================================

with open(RF_FEATURE_NAMES_PATH, "r") as f:
    RF_FEATURE_NAMES = json.load(f)

with open(RF_MEDIANS_PATH, "r") as f:
    RF_TRAINING_MEDIANS = json.load(f)


# ============================================================
# LSTM FEATURE ORDER
# ============================================================

LSTM_FEATURE_NAMES = [
    "Close",
    "Volume",
    "SMA_20",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Width",
    "ATR_14",
    "Daily_Return_Calc",
    "Volatility_20D_Calc",
    "Volume_Change",
    "NIFTY50_Return",
    "NIFTY50_Volatility_20D",
    "NIFTY50_MA_Distance",
    "Stock_Return",
    "Sector_Avg_Return",
    "Relative_Sector_Return",
    "Sector_Volatility_20D",
]


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_technical_indicators(group):

    group = group.copy()

    # Moving averages
    group["SMA_20"] = SMAIndicator(
        close=group["Close"],
        window=20
    ).sma_indicator()

    group["SMA_50"] = SMAIndicator(
        close=group["Close"],
        window=50
    ).sma_indicator()

    group["EMA_20"] = EMAIndicator(
        close=group["Close"],
        window=20
    ).ema_indicator()

    # RSI
    group["RSI_14"] = RSIIndicator(
        close=group["Close"],
        window=14
    ).rsi()

    # MACD
    macd = MACD(
        close=group["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    group["MACD"] = macd.macd()
    group["MACD_Signal"] = macd.macd_signal()
    group["MACD_Histogram"] = macd.macd_diff()

    # Bollinger Bands
    bollinger = BollingerBands(
        close=group["Close"],
        window=20,
        window_dev=2
    )

    group["BB_High"] = bollinger.bollinger_hband()
    group["BB_Low"] = bollinger.bollinger_lband()
    group["BB_Width"] = bollinger.bollinger_wband()

    # ATR
    atr = AverageTrueRange(
        high=group["High"],
        low=group["Low"],
        close=group["Close"],
        window=14
    )

    group["ATR_14"] = atr.average_true_range()

    # Daily return
    group["Daily_Return_Calc"] = (
        group["Close"].pct_change() * 100
    )

    # 20-day volatility
    group["Volatility_20D_Calc"] = (
        group["Daily_Return_Calc"]
        .rolling(20)
        .std()
    )

    # Volume change
    group["Volume_Change"] = (
        group["Volume"].pct_change() * 100
    )

    return group


# ============================================================
# NIFTY 50 MARKET FEATURES
# ============================================================

def add_nifty_features(nifty_df):

    nifty_df = nifty_df.copy()

    nifty_df["Date"] = pd.to_datetime(
        nifty_df["Date"]
    ).dt.tz_localize(None)

    nifty_df = nifty_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # Market daily return
    nifty_df["NIFTY50_Return"] = (
        nifty_df["Close"].pct_change() * 100
    )

    # 20-day market volatility
    nifty_df["NIFTY50_Volatility_20D"] = (
        nifty_df["NIFTY50_Return"]
        .rolling(20)
        .std()
    )

    # 20-day moving average
    nifty_df["NIFTY50_SMA_20"] = (
        nifty_df["Close"]
        .rolling(20)
        .mean()
    )

    # Distance from 20-day moving average
    nifty_df["NIFTY50_MA_Distance"] = (
        (
            nifty_df["Close"]
            - nifty_df["NIFTY50_SMA_20"]
        )
        / nifty_df["NIFTY50_SMA_20"]
    ) * 100

    return nifty_df


# ============================================================
# SECTOR-RELATIVE FEATURES
# ============================================================

def add_sector_features(stock_df):

    stock_df = stock_df.copy()

    # Stock daily return
    stock_df["Stock_Return"] = (
        stock_df
        .groupby("Ticker")["Close"]
        .pct_change() * 100
    )

    # Sector average return for each date
    sector_daily_return = (
        stock_df
        .groupby(["Date", "Sector"])["Stock_Return"]
        .mean()
        .rename("Sector_Avg_Return")
        .reset_index()
    )

    # Merge sector return
    stock_df = stock_df.merge(
        sector_daily_return,
        on=["Date", "Sector"],
        how="left"
    )

    # Stock performance relative to sector
    stock_df["Relative_Sector_Return"] = (
        stock_df["Stock_Return"]
        - stock_df["Sector_Avg_Return"]
    )

    # Sort sector data
    sector_daily_return = (
        sector_daily_return
        .sort_values(["Sector", "Date"])
    )

    # Sector rolling volatility
    sector_daily_return["Sector_Volatility_20D"] = (
        sector_daily_return
        .groupby("Sector")["Sector_Avg_Return"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Merge sector volatility
    stock_df = stock_df.merge(
        sector_daily_return[
            [
                "Date",
                "Sector",
                "Sector_Volatility_20D"
            ]
        ],
        on=["Date", "Sector"],
        how="left"
    )

    return stock_df


# ============================================================
# FUNDAMENTAL GROWTH FEATURES
# ============================================================

def add_fundamental_growth_features(fundamental_df):

    fundamental_df = fundamental_df.copy()

    fundamental_df = (
        fundamental_df
        .sort_values(["Ticker", "Period"])
        .reset_index(drop=True)
    )

    # Previous-quarter values
    fundamental_df["Previous_Net_Profit"] = (
        fundamental_df
        .groupby("Ticker")["Net_Profit"]
        .shift(1)
    )

    fundamental_df["Previous_EPS"] = (
        fundamental_df
        .groupby("Ticker")["EPS"]
        .shift(1)
    )

    fundamental_df["Previous_Sales"] = (
        fundamental_df
        .groupby("Ticker")["Sales"]
        .shift(1)
    )

    # Net profit growth
    fundamental_df["Net_Profit_Growth"] = np.where(
        fundamental_df["Previous_Net_Profit"].abs() > 1e-9,
        (
            (
                fundamental_df["Net_Profit"]
                - fundamental_df["Previous_Net_Profit"]
            )
            / fundamental_df["Previous_Net_Profit"].abs()
        ) * 100,
        np.nan
    )

    # EPS growth
    fundamental_df["EPS_Growth"] = np.where(
        fundamental_df["Previous_EPS"].abs() > 1e-9,
        (
            (
                fundamental_df["EPS"]
                - fundamental_df["Previous_EPS"]
            )
            / fundamental_df["Previous_EPS"].abs()
        ) * 100,
        np.nan
    )

    # Sales growth
    fundamental_df["Sales_Growth"] = np.where(
        fundamental_df["Previous_Sales"].abs() > 1e-9,
        (
            (
                fundamental_df["Sales"]
                - fundamental_df["Previous_Sales"]
            )
            / fundamental_df["Previous_Sales"].abs()
        ) * 100,
        np.nan
    )

    return fundamental_df


# ============================================================
# FUNDAMENTAL MERGE
# ============================================================

def merge_fundamentals(stock_df, fundamental_df):

    stock_df = stock_df.copy()
    fundamental_df = fundamental_df.copy()

    # Clean dates
    stock_df["Date"] = pd.to_datetime(
        stock_df["Date"]
    ).dt.tz_localize(None)

    fundamental_df["Available_From"] = pd.to_datetime(
        fundamental_df["Available_From"]
    ).dt.tz_localize(None)

    # EPS already exists in stock data
    if "EPS" in fundamental_df.columns:
        fundamental_df = fundamental_df.drop(
            columns=["EPS"]
        )

    # Sort merge keys
    stock_df = (
        stock_df
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    fundamental_df = (
        fundamental_df
        .sort_values(["Available_From", "Ticker"])
        .reset_index(drop=True)
    )

    # Use most recent available fundamental information
    stock_df = pd.merge_asof(
        stock_df,
        fundamental_df,
        left_on="Date",
        right_on="Available_From",
        by="Ticker",
        direction="backward"
    )

    return stock_df


# ============================================================
# RF FEATURE PREPARATION
# ============================================================

def prepare_rf_features(df):

    df = df.copy()

    # Features for which the original model
    # used missing-value indicator columns
    indicator_source_features = [
        "NIFTY50_Return",
        "NIFTY50_Volatility_20D",
        "NIFTY50_MA_Distance",
        "Stock_Return",
        "Sector_Avg_Return",
        "Relative_Sector_Return",
        "Sector_Volatility_20D",
        "Net_Profit_Growth",
        "EPS_Growth",
        "Sales_Growth",
        "OPM_Percent",
        "Other_Income",
        "Profit_Before_Tax",
        "Tax_Percent",
        "Net_Profit",
    ]

    # Create missing indicators BEFORE imputation
    for feature in indicator_source_features:

        if feature not in df.columns:
            df[feature] = np.nan

        df[f"{feature}_Missing"] = (
            df[feature]
            .isna()
            .astype(int)
        )

    # Apply exact training medians
    for feature, median in RF_TRAINING_MEDIANS.items():

        if feature in df.columns:
            df[feature] = (
                df[feature]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(median)
            )

    # Remaining NaN values use zero fallback
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    # Verify all required RF features exist
    missing_features = [
        feature
        for feature in RF_FEATURE_NAMES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing RF features: {missing_features}"
        )

    # Exact feature order used during training
    return df[RF_FEATURE_NAMES].copy()


# ============================================================
# LSTM FEATURE PREPARATION
# ============================================================

def prepare_lstm_sequence(df):

    df = df.copy()

    missing_features = [
        feature
        for feature in LSTM_FEATURE_NAMES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing LSTM features: {missing_features}"
        )

    features = df[LSTM_FEATURE_NAMES].copy()

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features = features.fillna(0)

    return features