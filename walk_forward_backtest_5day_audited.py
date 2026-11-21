"""
Walk-forward historical evaluation for AI Stock Analyser.

Uses the existing Random Forest and LSTM models.
Does not modify the existing application or model files.

Evaluation horizon: 5 trading sessions
Independent evaluation labels:
    BUY  : return > +1.5%
    SELL : return < -1.5%
    HOLD : otherwise
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from services.market_data_service import (
    SECTOR_UNIVERSE,
    get_stock_history,
    get_nifty50_history,
)

from services.feature_service import (
    add_technical_indicators,
    add_nifty_features,
    add_sector_features,
    prepare_rf_features,
    prepare_lstm_sequence,
)

from services.model_service import (
    rf_model,
    lstm_model,
    lstm_scaler,
    lstm_label_encoder,
    RF_WEIGHT,
    LSTM_WEIGHT,
    LSTM_SEQUENCE_LENGTH,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "evaluation_results_5day"
PERIOD = "5y"
HORIZON = 5
THRESHOLD = 1.5
MIN_HISTORY = LSTM_SEQUENCE_LENGTH

CLASSES = ["BUY", "HOLD", "SELL"]


def normalize_dates(df):
    """Normalize dates to timezone-naive calendar dates."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize()
    return df


def outcome_label(return_pct):
    if return_pct > THRESHOLD:
        return "BUY"
    if return_pct < -THRESHOLD:
        return "SELL"
    return "HOLD"


def download_historical_data():
    """Download the configured stock universe and NIFTY 50."""
    sector_lookup = {
        ticker: sector
        for sector, tickers in SECTOR_UNIVERSE.items()
        for ticker in tickers
    }

    tickers = list(sector_lookup)
    frames = []
    failures = []

    print(f"Downloading {len(tickers)} stocks ({PERIOD})...")

    for i, ticker in enumerate(tickers, start=1):
        try:
            df = get_stock_history(
                ticker=ticker,
                period=PERIOD,
                interval="1d",
            )

            if df is None or df.empty:
                raise ValueError("No historical data returned")

            df = normalize_dates(df)
            df = df.dropna(subset=["Close"])
            df = df.sort_values("Date").drop_duplicates("Date")
            df["Ticker"] = ticker
            df["Sector"] = sector_lookup[ticker]
            frames.append(df)

            print(f"[{i}/{len(tickers)}] Downloaded {ticker}: {len(df)} rows")

        except Exception as exc:
            failures.append({"Ticker": ticker, "Error": str(exc)})
            print(f"[{i}/{len(tickers)}] FAILED {ticker}: {exc}")

    if not frames:
        raise RuntimeError("No stock data was downloaded.")

    stocks = pd.concat(frames, ignore_index=True)
    stocks = stocks.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print("Downloading NIFTY 50...")
    nifty = get_nifty50_history(period=PERIOD, interval="1d")

    if nifty is None or nifty.empty:
        raise RuntimeError("NIFTY 50 history could not be downloaded.")

    nifty = normalize_dates(nifty)
    nifty = nifty.sort_values("Date").drop_duplicates("Date")
    nifty = add_nifty_features(nifty)

    return stocks, nifty, failures


def build_historical_features(stocks, nifty):
    """
    Build causal rolling features.

    Indicators use prices through the prediction date only.
    Sector features use peer returns from the same date.
    """
    print("Calculating technical indicators...")

    technical_frames = []

    for ticker, group in stocks.groupby("Ticker", sort=False):
        group = group.sort_values("Date").copy()
        group = add_technical_indicators(group)
        technical_frames.append(group)

    data = pd.concat(technical_frames, ignore_index=True)

    nifty_columns = [
        "Date",
        "NIFTY50_Return",
        "NIFTY50_Volatility_20D",
        "NIFTY50_MA_Distance",
    ]

    data = data.merge(
        nifty[nifty_columns],
        on="Date",
        how="left",
        validate="many_to_one",
    )

    data = add_sector_features(data)

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return data


def evaluate_stock(ticker, group):
    """Generate historical predictions and evaluate future outcomes."""
    group = group.sort_values("Date").reset_index(drop=True)

    if len(group) < MIN_HISTORY + HORIZON:
        print(f"Skipping {ticker}: insufficient history")
        return []

    # Use the same preprocessing functions as the live application.
    rf_features = prepare_rf_features(group)
    lstm_features = prepare_lstm_sequence(group)

    # Only dates with a complete future evaluation horizon are scored.
    last_prediction_index = len(group) - HORIZON - 1

    if last_prediction_index < MIN_HISTORY - 1:
        return []

    prediction_indices = np.arange(
        MIN_HISTORY - 1,
        last_prediction_index + 1,
    )

    # RF predictions can be batched.
    rf_input = rf_features.iloc[prediction_indices]
    rf_probabilities = rf_model.predict_proba(rf_input)
    rf_classes = [str(c) for c in rf_model.classes_]

    # Build each LSTM sequence using data ending on its prediction date.
    sequences = np.stack([
        lstm_features.iloc[
            i - MIN_HISTORY + 1 : i + 1
        ].to_numpy(dtype=np.float32)
        for i in prediction_indices
    ])

    n_samples, sequence_length, n_features = sequences.shape

    # Apply the existing trained scaler without refitting it.
    scaled_sequences = lstm_scaler.transform(
        sequences.reshape(-1, n_features)
    ).reshape(n_samples, sequence_length, n_features)

    lstm_probabilities = lstm_model.predict(
        scaled_sequences,
        batch_size=64,
        verbose=0,
    )

    lstm_classes = [
        str(c) for c in lstm_label_encoder.classes_
    ]

    results = []

    for j, i in enumerate(prediction_indices):
        # The outcome uses the close exactly 5 trading sessions after the signal.
        start_close = float(group.iloc[i]["Close"])
        future_close = float(group.iloc[i + HORIZON]["Close"])

        if (
            not np.isfinite(start_close)
            or not np.isfinite(future_close)
            or start_close <= 0
        ):
            continue

        future_return = (
            (future_close / start_close) - 1
        ) * 100

        actual_label = outcome_label(future_return)

        # Map both models' probabilities into the same class order.
        rf_map = dict(zip(rf_classes, rf_probabilities[j]))
        lstm_map = dict(zip(lstm_classes, lstm_probabilities[j]))

        ensemble = {
            cls: (
                RF_WEIGHT * float(rf_map.get(cls, 0.0))
                + LSTM_WEIGHT * float(lstm_map.get(cls, 0.0))
            )
            for cls in CLASSES
        }

        prediction = max(ensemble, key=ensemble.get)

        # Directional hit is defined only for BUY and SELL signals.
        if prediction == "BUY":
            directional_hit = future_return > 0
        elif prediction == "SELL":
            directional_hit = future_return < 0
        else:
            directional_hit = np.nan

        results.append({
            "Ticker": ticker,
            "Prediction_Date": group.iloc[i]["Date"],
            "Outcome_Date": group.iloc[i + HORIZON]["Date"],
            "Close_On_Prediction_Date": start_close,
            "Close_After_5_Sessions": future_close,
            "Predicted_Recommendation": prediction,
            "Actual_5Day_Return_Percent": future_return,
            "Actual_Evaluation_Label": actual_label,
            "Label_Match": prediction == actual_label,
            "Directional_Hit": directional_hit,
            "BUY_Probability_Percent": ensemble["BUY"] * 100,
            "HOLD_Probability_Percent": ensemble["HOLD"] * 100,
            "SELL_Probability_Percent": ensemble["SELL"] * 100,
        })

    return results


def create_reports(results, failures):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    details = pd.DataFrame(results)

    if details.empty:
        raise RuntimeError(
            "No predictions were evaluated. Check downloads and features."
        )

    details.to_csv(
        OUTPUT_DIR / "walk_forward_details.csv",
        index=False,
    )

    # Summary grouped by the model's predicted recommendation.
    summary_rows = []

    for recommendation, group in details.groupby(
        "Predicted_Recommendation"
    ):
        directional = group["Directional_Hit"].dropna()

        summary_rows.append({
            "Predicted_Recommendation": recommendation,
            "Evaluated_Count": len(group),
            "Label_Match_Percent": group["Label_Match"].mean() * 100,
            "Mean_5Day_Return_Percent":
                group["Actual_5Day_Return_Percent"].mean(),
            "Median_5Day_Return_Percent":
                group["Actual_5Day_Return_Percent"].median(),
            "Positive_Return_Percent":
                (group["Actual_5Day_Return_Percent"] > 0).mean() * 100,
            "Directional_Hit_Rate_Percent":
                directional.mean() * 100 if len(directional) else np.nan,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "walk_forward_summary.csv",
        index=False,
    )

    # Per-stock breakdown.
    stock_rows = []

    for ticker, group in details.groupby("Ticker"):
        stock_rows.append({
            "Ticker": ticker,
            "Evaluated_Count": len(group),
            "Label_Match_Percent": group["Label_Match"].mean() * 100,
            "Mean_5Day_Return_Percent":
                group["Actual_5Day_Return_Percent"].mean(),
            "BUY_Count":
                (group["Predicted_Recommendation"] == "BUY").sum(),
            "HOLD_Count":
                (group["Predicted_Recommendation"] == "HOLD").sum(),
            "SELL_Count":
                (group["Predicted_Recommendation"] == "SELL").sum(),
        })

    by_stock = pd.DataFrame(stock_rows).sort_values(
        "Label_Match_Percent",
        ascending=False,
    )

    by_stock.to_csv(
        OUTPUT_DIR / "walk_forward_by_stock.csv",
        index=False,
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "walk_forward_download_failures.csv",
            index=False,
        )

    print("\n" + "=" * 65)
    print("WALK-FORWARD BACKTEST COMPLETE")
    print("=" * 65)
    print(f"Evaluated predictions: {len(details):,}")
    print(f"Stocks evaluated: {details['Ticker'].nunique()}")
    print(
        "Overall evaluation-label match: "
        f"{details['Label_Match'].mean() * 100:.2f}%"
    )
    print(
        "Mean 5-session stock return across observations: "
        f"{details['Actual_5Day_Return_Percent'].mean():.3f}%"
    )

    print("\nPerformance by predicted recommendation:")
    print(summary.round(3).to_string(index=False))

    print(f"\nReports saved to:\n{OUTPUT_DIR}")
    print("\nIMPORTANT:")
    print("- Evaluation labels are independent 5-session labels.")
    print("- They are not verified copies of the original training labels.")
    print("- Returns exclude transaction costs, taxes, and slippage.")
    print("- Overlapping prediction periods are not independent trades.")


def main():
    print("AI STOCK ANALYSER â€” WALK-FORWARD BACKTEST")
    print(f"History: {PERIOD}")
    print(f"Outcome horizon: {HORIZON} trading sessions")
    print(f"Independent label threshold: +/-{THRESHOLD}%")
    print(f"RF weight: {RF_WEIGHT}; LSTM weight: {LSTM_WEIGHT}")

    stocks, nifty, failures = download_historical_data()

    print(f"\nStocks downloaded: {stocks['Ticker'].nunique()}")
    print(f"Date range: {stocks['Date'].min().date()} to "
          f"{stocks['Date'].max().date()}")

    data = build_historical_features(stocks, nifty)

    all_results = []
    tickers = list(data["Ticker"].unique())

    for number, (ticker, group) in enumerate(
        data.groupby("Ticker", sort=False),
        start=1,
    ):
        print(f"\nEvaluating [{number}/{len(tickers)}] {ticker}...")

        try:
            all_results.extend(evaluate_stock(ticker, group))
        except Exception as exc:
            print(f"FAILED {ticker}: {exc}")
            failures.append({"Ticker": ticker, "Error": str(exc)})

    create_reports(all_results, failures)


if __name__ == "__main__":
    main()
