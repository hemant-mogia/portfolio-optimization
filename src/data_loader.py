import os
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime
import time


def load_config(config_path="./data/config.yaml"):
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def download_with_retry(ticker, start_date, end_date, retries=3, delay=5):
    """Download data with retries in case of failure."""
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if data.empty or "Close" not in data.columns:
                raise ValueError(f"No valid data returned for {ticker}")
            return data
        except Exception as e:
            print(f"[{ticker}] Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                print(f"[{ticker}] Failed after {retries} attempts.")
                return None


def download_prices(tickers, start_date, end_date, save_dir="./data/raw"):
    """Download and save stock price data for a list of tickers."""
    os.makedirs(save_dir, exist_ok=True)

    if not end_date:
        end_date = datetime.today().strftime("%Y-%m-%d")

    price_series_list = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        data = download_with_retry(ticker, start_date, end_date)
        if data is None:
            print(f"Warning: No data for {ticker}. Skipping.")
            continue

        if "Close" not in data.columns:
            print(f"Warning: 'Close' missing for {ticker}. Skipping.")
            continue

        adj_close = data["Close"].dropna()
        if adj_close.empty:
            print(f"Warning: Adjusted Close empty for {ticker}. Skipping.")
            continue

        # Save individual ticker CSVs
        file_path = os.path.join(save_dir, f"{ticker}.csv")
        data.to_csv(file_path)
        print(f"Saved {ticker} data to {file_path}")

        adj_close.name = ticker  # name the series for concat
        price_series_list.append(adj_close)

    if price_series_list:
        combined_df = pd.concat(price_series_list, axis=1)
        combined_file_path = os.path.join(save_dir, "combined_prices.csv")
        combined_df.to_csv(combined_file_path)
        print(f"Saved combined prices to {combined_file_path}")
        print(f"Downloaded data for {len(price_series_list)} tickers with {combined_df.shape[0]} dates.")
    else:
        print("No valid data downloaded for any ticker.")


def main():
    """Main function to drive the data download process."""
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    tickers = config.get("tickers", [])
    data_source = config.get("data_source", {})
    start_date = data_source.get("start_date", "2015-01-01")
    end_date = data_source.get("end_date", None)

    if not tickers:
        print("No tickers found in config file.")
        return

    download_prices(tickers, start_date, end_date)


if __name__ == "__main__":
    main()
