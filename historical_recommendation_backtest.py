
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------
# Configuration: standalone evaluation only
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "final_stock_model_package"
    / "final_stock_recommendations.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_results"
HORIZON_DAYS = 10


def normalize_ticker(value):
    ticker = str(value).strip().upper()

    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker

    if "." not in ticker:
        return ticker + ".NS"

    return ticker


def download_history(ticker, start_date, end_date):
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        return pd.DataFrame()

    # Handle Yahoo Finance MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    if "Date" not in data.columns or "Close" not in data.columns:
        return pd.DataFrame()

    data["Date"] = (
        pd.to_datetime(data["Date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    data["Close"] = pd.to_numeric(
        data["Close"], errors="coerce"
    )

    data = (
        data.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    return data


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Recommendation CSV not found: {INPUT_FILE}"
        )

    recommendations = pd.read_csv(INPUT_FILE)

    required = {"Date", "Ticker", "Recommendation"}
    missing = required - set(recommendations.columns)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    recommendations["Date"] = (
        pd.to_datetime(recommendations["Date"], errors="coerce")
        .dt.normalize()
    )

    recommendations["Ticker"] = (
        recommendations["Ticker"].map(normalize_ticker)
    )

    recommendations["Recommendation"] = (
        recommendations["Recommendation"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    recommendations = recommendations.dropna(subset=["Date"])

    recommendations = recommendations[
        recommendations["Recommendation"].isin(
            ["BUY", "HOLD", "SELL"]
        )
    ].copy()

    if recommendations.empty:
        raise ValueError("No valid recommendation rows found.")

    # Include enough calendar days after the last signal to
    # retrieve its next 10 trading sessions.
    first_date = recommendations["Date"].min()
    last_date = recommendations["Date"].max()

    start_date = (first_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = (last_date + pd.Timedelta(days=35)).strftime("%Y-%m-%d")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_rows = []
    failed_tickers = []

    tickers = sorted(recommendations["Ticker"].unique())

    print(f"Recommendations: {len(recommendations):,}")
    print(f"Unique stocks: {len(tickers)}")
    print(f"Evaluation horizon: {HORIZON_DAYS} trading sessions")
    print()

    for number, ticker in enumerate(tickers, start=1):
        print(f"[{number}/{len(tickers)}] Downloading {ticker}...")

        try:
            history = download_history(
                ticker, start_date, end_date
            )

            if history.empty:
                failed_tickers.append(ticker)
                continue

            history_dates = history["Date"].tolist()
            closes = history["Close"].to_numpy(dtype=float)

            date_positions = {
                date: index
                for index, date in enumerate(history_dates)
            }

            stock_signals = recommendations[
                recommendations["Ticker"] == ticker
            ]

            for _, signal in stock_signals.iterrows():
                signal_date = signal["Date"]
                position = date_positions.get(signal_date)

                # Require an exact trading-date match.
                if position is None:
                    continue

                future_position = position + HORIZON_DAYS

                # There must be 10 later trading sessions.
                if future_position >= len(closes):
                    continue

                initial_close = closes[position]
                future_close = closes[future_position]

                if (
                    not np.isfinite(initial_close)
                    or not np.isfinite(future_close)
                    or initial_close <= 0
                ):
                    continue

                forward_return = (
                    (future_close / initial_close) - 1
                ) * 100

                recommendation = signal["Recommendation"]

                if recommendation == "BUY":
                    direction_correct = forward_return > 0
                elif recommendation == "SELL":
                    direction_correct = forward_return < 0
                else:
                    direction_correct = None

                result_rows.append({
                    "Date": signal_date.date().isoformat(),
                    "Ticker": ticker,
                    "Recommendation": recommendation,
                    "Entry_Close_Adjusted": round(
                        float(initial_close), 4
                    ),
                    "Close_After_10_Sessions": round(
                        float(future_close), 4
                    ),
                    "Forward_Return_Percent": round(
                        float(forward_return), 4
                    ),
                    "Direction_Correct": direction_correct,
                    "Outcome_Date": (
                        history_dates[future_position]
                        .date()
                        .isoformat()
                    ),
                })

        except Exception as exc:
            failed_tickers.append(ticker)
            print(f"  Download/evaluation failed: {exc}")

        time.sleep(0.3)

    if not result_rows:
        raise RuntimeError(
            "No evaluable rows were produced. "
            "Check ticker formats and downloaded price history."
        )

    results = pd.DataFrame(result_rows)

    detailed_path = OUTPUT_DIR / "backtest_10day_details.csv"
    results.to_csv(detailed_path, index=False)

    # Directional hit rate applies only to BUY and SELL.
    directional = results[
        results["Recommendation"].isin(["BUY", "SELL"])
    ]

    summary_rows = []

    for recommendation in ["BUY", "HOLD", "SELL"]:
        group = results[
            results["Recommendation"] == recommendation
        ]

        if group.empty:
            continue

        row = {
            "Recommendation": recommendation,
            "Evaluated_Count": len(group),
            "Mean_10Day_Return_Percent": round(
                group["Forward_Return_Percent"].mean(), 4
            ),
            "Median_10Day_Return_Percent": round(
                group["Forward_Return_Percent"].median(), 4
            ),
            "Positive_Return_Percent": round(
                (group["Forward_Return_Percent"] > 0).mean() * 100,
                2,
            ),
            "Directional_Hit_Rate_Percent": None,
        }

        if recommendation in ["BUY", "SELL"]:
            row["Directional_Hit_Rate_Percent"] = round(
                group["Direction_Correct"].astype(bool).mean() * 100,
                2,
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "backtest_10day_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n========== BACKTEST SUMMARY ==========")
    print(summary.to_string(index=False))
    print("\nEvaluated recommendations:", len(results))
    print("Stocks evaluated:", results["Ticker"].nunique())
    print("Stocks with download failures:", len(set(failed_tickers)))
    print("Detailed results:", detailed_path)
    print("Summary results:", summary_path)
    print("\nThis evaluates saved historical recommendations,")
    print("not the current model's walk-forward accuracy.")
    print("Returns exclude transaction costs and taxes.")


if __name__ == "__main__":
    main()
