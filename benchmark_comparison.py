
from pathlib import Path

import pandas as pd
import yfinance as yf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BACKEND_DIR / "evaluation_results"

DETAILS_FILE = RESULTS_DIR / "backtest_10day_details.csv"
OUTPUT_FILE = RESULTS_DIR / "benchmark_comparison.csv"

BENCHMARK_TICKER = "^NSEI"  # NIFTY 50


def download_benchmark(start_date, end_date):
    data = yf.download(
        BENCHMARK_TICKER,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError("Could not download NIFTY 50 history.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data["Date"] = (
        pd.to_datetime(data["Date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

    return (
        data.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )


def main():
    if not DETAILS_FILE.exists():
        raise FileNotFoundError(
            f"Backtest results not found: {DETAILS_FILE}"
        )

    signals = pd.read_csv(DETAILS_FILE)

    required = {
        "Date",
        "Ticker",
        "Recommendation",
        "Forward_Return_Percent",
        "Outcome_Date",
    }

    missing = required - set(signals.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    signals["Date"] = (
        pd.to_datetime(signals["Date"], errors="coerce")
        .dt.normalize()
    )

    signals["Outcome_Date"] = (
        pd.to_datetime(signals["Outcome_Date"], errors="coerce")
        .dt.normalize()
    )

    signals = signals.dropna(
        subset=["Date", "Outcome_Date", "Forward_Return_Percent"]
    ).copy()

    start_date = (
        signals["Date"].min() - pd.Timedelta(days=10)
    ).strftime("%Y-%m-%d")

    end_date = (
        signals["Outcome_Date"].max() + pd.Timedelta(days=5)
    ).strftime("%Y-%m-%d")

    benchmark = download_benchmark(start_date, end_date)

    benchmark_dates = benchmark["Date"].tolist()
    benchmark_closes = benchmark["Close"].tolist()

    date_to_position = {
        date: position
        for position, date in enumerate(benchmark_dates)
    }

    benchmark_returns = []

    for _, signal in signals.iterrows():
        start_position = date_to_position.get(signal["Date"])
        end_position = date_to_position.get(signal["Outcome_Date"])

        if start_position is None or end_position is None:
            continue

        if end_position <= start_position:
            continue

        start_close = benchmark_closes[start_position]
        end_close = benchmark_closes[end_position]

        if start_close <= 0:
            continue

        nifty_return = (end_close / start_close - 1) * 100
        stock_return = float(signal["Forward_Return_Percent"])

        benchmark_returns.append({
            "Date": signal["Date"].date().isoformat(),
            "Ticker": signal["Ticker"],
            "Recommendation": signal["Recommendation"],
            "Stock_10Day_Return_Percent": stock_return,
            "NIFTY50_Matched_Period_Return_Percent": nifty_return,
            "Stock_Minus_NIFTY_Percentage_Points": (
                stock_return - nifty_return
            ),
            "Outcome_Date": signal["Outcome_Date"].date().isoformat(),
        })

    if not benchmark_returns:
        raise RuntimeError(
            "No dates matched between stock results and NIFTY 50."
        )

    comparison = pd.DataFrame(benchmark_returns)
    comparison.to_csv(OUTPUT_FILE, index=False)

    print("\n========== BENCHMARK COMPARISON ==========")
    print("Matched observations:", len(comparison))
    print(
        "Mean stock return:",
        round(comparison["Stock_10Day_Return_Percent"].mean(), 4),
        "%",
    )
    print(
        "Mean NIFTY 50 return:",
        round(
            comparison["NIFTY50_Matched_Period_Return_Percent"].mean(),
            4,
        ),
        "%",
    )
    print(
        "Mean stock-minus-NIFTY difference:",
        round(
            comparison[
                "Stock_Minus_NIFTY_Percentage_Points"
            ].mean(),
            4,
        ),
        "percentage points",
    )

    print("\nBy recommendation:")
    print(
        comparison.groupby("Recommendation")[
            [
                "Stock_10Day_Return_Percent",
                "NIFTY50_Matched_Period_Return_Percent",
                "Stock_Minus_NIFTY_Percentage_Points",
            ]
        ].mean().round(4).to_string()
    )

    print("\nSaved:", OUTPUT_FILE)
    print(
        "\nNote: observations overlap in time. "
        "This is a descriptive benchmark comparison, "
        "not proof of predictive skill or investment profitability."
    )


if __name__ == "__main__":
    main()
