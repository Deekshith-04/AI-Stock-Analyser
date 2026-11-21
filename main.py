from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.market_data_service import (
    get_latest_stock_quote,
    get_stock_market_data,
    get_market_indices,
    get_market_summary,
    get_all_market_stocks,
    MARKET_SUMMARY_TICKERS,
)

app = FastAPI(
    title="AI Stock Analysis & Recommendation System",
    description="Backend API for stock market analysis and ML predictions",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://ai-stock-analyser.pages.dev",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "AI Stock Analysis Backend is running",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ---------------------------------------------------------
# Model information
# TensorFlow loads only when this endpoint is requested.
# ---------------------------------------------------------

@app.get("/model-info")
def model_info():
    try:
        from services.model_service import get_model_info

        return get_model_info()

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model service unavailable: {e}",
        )


# ---------------------------------------------------------
# Stock prediction
# Prediction models load only when this endpoint is called.
# ---------------------------------------------------------

@app.get("/predict/{ticker}")
def predict(ticker: str):
    try:
        from services.prediction_service import predict_stock

        result = predict_stock(
            ticker=ticker,
            period="6mo",
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# Latest stock quote
# ---------------------------------------------------------

@app.get("/quote/{ticker}")
def quote(ticker: str):
    try:
        return get_latest_stock_quote(
            ticker=ticker,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# Market data for frontend
# ---------------------------------------------------------

@app.get("/market/{ticker}")
def market(ticker: str):
    try:
        return get_stock_market_data(
            ticker=ticker,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# Market indices
# ---------------------------------------------------------

@app.get("/market-indices")
def market_indices():
    try:
        return get_market_indices()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# Market summary
# ---------------------------------------------------------

@app.get("/market-summary")
def market_summary():
    try:
        return get_market_summary()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# Historical stock data for charts
# ---------------------------------------------------------

@app.get("/history/{ticker}")
def history(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
):
    try:
        from services.market_data_service import (
            get_stock_history_for_chart
        )

        return get_stock_history_for_chart(
            ticker=ticker,
            period=period,
            interval=interval,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------

@app.get("/technical/{ticker}")
def technical(ticker: str):
    try:
        from services.technical_service import (
            get_technical_indicators
        )

        return get_technical_indicators(
            ticker=ticker,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# All tracked market stocks
# ---------------------------------------------------------

@app.get("/market-stocks")
def market_stocks():
    try:
        return get_all_market_stocks()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
