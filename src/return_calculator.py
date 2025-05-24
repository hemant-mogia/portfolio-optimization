import os
import pandas as pd

def load_raw_prices(raw_dir="./data/raw"):
    # Load combined prices CSV or aggregate individual CSVs
    combined_path = os.path.join(raw_dir, "combined_prices.csv")
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined prices file not found: {combined_path}")
    df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    return df

def clean_prices(df):
    # Forward fill missing data then backfill
    return df.ffill().bfill()

def calculate_returns(df):
    # Daily simple returns
    returns = df.pct_change().dropna()
    return returns

def save_processed_data(prices, returns, processed_dir="./data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    prices.to_csv(os.path.join(processed_dir, "cleaned_prices.csv"))
    returns.to_csv(os.path.join(processed_dir, "daily_returns.csv"))

def main():
    prices_raw = load_raw_prices()
    prices_clean = clean_prices(prices_raw)
    returns = calculate_returns(prices_clean)
    save_processed_data(prices_clean, returns)

if __name__ == "__main__":
    main()
