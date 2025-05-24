import os
import pandas as pd
import numpy as np

def load_returns(processed_dir="./data/processed"):
    path = os.path.join(processed_dir, "daily_returns.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)

def load_weights(output_dir="./outputs"):
    path = os.path.join(output_dir, "optimized_weights.csv")
    weights_df = pd.read_csv(path, index_col=0)
    return weights_df["weight"]

def calculate_portfolio_returns(returns, weights):
    # Align returns columns and weights index
    weights = weights.reindex(returns.columns).fillna(0)
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

def compute_cumulative_returns(daily_returns):
    return (1 + daily_returns).cumprod() - 1

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

def annualized_performance(daily_returns, trading_days=252):
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()
    annual_return = (1 + mean_daily) ** trading_days - 1
    annual_vol = std_daily * np.sqrt(trading_days)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan
    return annual_return, annual_vol, sharpe_ratio

def backtest():
    returns = load_returns()
    weights = load_weights()
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    cum_returns = compute_cumulative_returns(portfolio_returns)
    ann_ret, ann_vol, sharpe = annualized_performance(portfolio_returns)
    mdd = max_drawdown(cum_returns)

    print(f"Annualized Return: {ann_ret:.2%}")
    print(f"Annualized Volatility: {ann_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")

    # Save results
    results = {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
    }
    output_dir = "./outputs/reports"
    os.makedirs(output_dir, exist_ok=True)
    pd.Series(results).to_csv(os.path.join(output_dir, "backtest_summary.csv"))

    # Save cumulative returns for plotting later
    cum_returns.to_csv(os.path.join(output_dir, "cumulative_returns.csv"))

if __name__ == "__main__":
    backtest()
