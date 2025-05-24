import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
import cvxpy as cp


def plot_cumulative_returns(returns_df, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    cum_returns = (1 + returns_df).cumprod() - 1

    plt.figure(figsize=(12, 7))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col)

    # Add text with final cumulative returns
    final_returns = cum_returns.iloc[-1]
    text_str = "Final Cum Returns:\n"
    for asset, ret in final_returns.items():
        text_str += f"{asset}: {ret:.2%}\n"
    
    # Identify best and worst performers
    best_asset = final_returns.idxmax()
    worst_asset = final_returns.idxmin()

    text_str += f"\nBest Performer: {best_asset} ({final_returns[best_asset]:.2%})"
    text_str += f"\nWorst Performer: {worst_asset} ({final_returns[worst_asset]:.2%})"

    # Recommendation example
    text_str += "\n\nRecommendation:\nConsider increasing allocation to best performer while monitoring risk."

    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    plt.gcf().text(0.75, 0.3, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0, 0.7, 1])  # leave space on right for text box
    plt.savefig(os.path.join(output_dir, "cumulative_returns.png"))
    plt.close()
    print("Saved cumulative_returns.png")


def plot_return_distribution(returns_df, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))
    returns_df.plot(kind='hist', bins=50, alpha=0.6)
    plt.title("Return Distribution")
    plt.xlabel("Daily Returns")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "return_distribution.png"))
    plt.close()
    print("Saved return_distribution.png")

def plot_rolling_volatility(returns_df, window=20, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    rolling_vol = returns_df.rolling(window=window).std() * (252 ** 0.5)  # annualized volatility

    plt.figure(figsize=(12, 7))
    for col in rolling_vol.columns:
        plt.plot(rolling_vol.index, rolling_vol[col], label=col)

    # Metrics
    mean_vol = rolling_vol.mean()
    max_vol = rolling_vol.max()

    text_str = "Rolling Volatility Stats (Annualized):\n"
    for asset in rolling_vol.columns:
        text_str += f"{asset} - Mean: {mean_vol[asset]:.2%}, Max: {max_vol[asset]:.2%}\n"

    text_str += "\nRecommendation:\nReview high volatility assets to manage risk exposure."

    plt.title(f"Rolling {window}-Day Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)

    plt.gcf().text(0.75, 0.4, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.savefig(os.path.join(output_dir, f"rolling_volatility_{window}d.png"))
    plt.close()
    print(f"Saved rolling_volatility_{window}d.png")


def plot_correlation_heatmap(returns_df, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    corr = returns_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    print("Saved correlation_heatmap.png")


def plot_drawdowns(returns_df, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    cum_returns = (1 + returns_df).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max

    plt.figure(figsize=(12, 7))
    for col in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[col], label=col)

    # Max drawdown
    max_dd = drawdowns.min()
    text_str = "Max Drawdowns:\n"
    for asset in drawdowns.columns:
        text_str += f"{asset}: {max_dd[asset]:.2%}\n"
    text_str += "\nRecommendation:\nMax drawdowns indicate risk - consider stop-loss or diversification."

    plt.title("Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)

    plt.gcf().text(0.75, 0.3, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.savefig(os.path.join(output_dir, "drawdowns.png"))
    plt.close()
    print("Saved drawdowns.png")


def plot_portfolio_weights(weights_df, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))
    weights_df.plot.area(alpha=0.7)
    plt.title("Portfolio Weights Over Time")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "portfolio_weights_over_time.png"))
    plt.close()
    print("Saved portfolio_weights_over_time.png")

def plot_portfolio_performance(portfolio_returns, benchmark_returns=None, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    cum_portfolio = (1 + portfolio_returns).cumprod() - 1

    plt.figure(figsize=(12, 7))
    plt.plot(cum_portfolio.index, cum_portfolio, label="Portfolio")

    # Metrics
    total_days = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days
    years = total_days / 365.25
    cagr = (1 + portfolio_returns).prod() ** (1/years) - 1
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

    text_str = f"CAGR: {cagr:.2%}\nSharpe Ratio: {sharpe:.2f}\n"

    if benchmark_returns is not None:
        cum_benchmark = (1 + benchmark_returns).cumprod() - 1
        plt.plot(cum_benchmark.index, cum_benchmark, label="Benchmark", linestyle="--")

        # Benchmark metrics
        cagr_b = (1 + benchmark_returns).prod() ** (1/years) - 1
        sharpe_b = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        text_str += f"Benchmark CAGR: {cagr_b:.2%}\nBenchmark Sharpe: {sharpe_b:.2f}\n"

        # Recommendation based on performance
        if cagr > cagr_b:
            text_str += "\nPortfolio outperformed benchmark. Maintain strategy."
        else:
            text_str += "\nPortfolio underperformed benchmark. Review asset allocation."

    plt.title("Portfolio Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    plt.gcf().text(0.75, 0.3, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.savefig(os.path.join(output_dir, "portfolio_performance.png"))
    plt.close()
    print("Saved portfolio_performance.png")


def plot_asset_allocation(weights_df, date=None, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    if date is None:
        date = weights_df.index[-1]
    else:
        date = pd.to_datetime(date)

    if date not in weights_df.index:
        raise ValueError(f"Date {date} not found in weights data.")

    allocation = weights_df.loc[date]

    plt.figure(figsize=(8, 8))
    allocation.plot.pie(autopct="%1.1f%%", startangle=90, counterclock=False)
    plt.title(f"Asset Allocation on {date.strftime('%Y-%m-%d')}")
    plt.ylabel("")
    plt.tight_layout()
    pie_path = os.path.join(output_dir, f"asset_allocation_pie_{date.strftime('%Y%m%d')}.png")
    plt.savefig(pie_path)
    plt.close()
    print(f"Saved {pie_path}")

    plt.figure(figsize=(10, 6))
    allocation.plot.bar(color=sns.color_palette("tab10"))
    plt.title(f"Asset Allocation on {date.strftime('%Y-%m-%d')}")
    plt.ylabel("Weight")
    plt.xlabel("Asset")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f"asset_allocation_bar_{date.strftime('%Y%m%d')}.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved {bar_path}")

def optimize_portfolio(expected_returns, cov_matrix, target_return):
    """
    Minimize portfolio variance for a target return with long-only weights.
    """
    n = len(expected_returns)
    w = cp.Variable(n)
    ret = expected_returns @ w
    risk = cp.quad_form(w, cov_matrix)
    constraints = [cp.sum(w) == 1, ret == target_return, w >= 0]
    problem = cp.Problem(cp.Minimize(risk), constraints)
    problem.solve(solver=cp.SCS)

    if w.value is None:
        return None, None, None

    weights = w.value
    portfolio_return = expected_returns @ weights
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    return weights, portfolio_return, portfolio_volatility

def compute_efficient_frontier(expected_returns, cov_matrix, num_points=50):
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    target_returns = np.linspace(min_return, max_return, num_points)

    frontier = []
    for r in target_returns:
        weights, ret, vol = optimize_portfolio(expected_returns, cov_matrix, r)
        if weights is not None:
            frontier.append({'weights': weights, 'return': ret, 'volatility': vol})
    return frontier

def plot_efficient_frontier(frontier_results, output_dir="./outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)

    returns = [res['return'] for res in frontier_results]
    volatilities = [res['volatility'] for res in frontier_results]

    plt.figure(figsize=(10, 7))
    plt.plot(volatilities, returns, 'b--', marker='o')

    # Highlight max Sharpe ratio portfolio
    sharpe_ratios = np.array(returns) / np.array(volatilities)
    max_sharpe_idx = sharpe_ratios.argmax()
    plt.scatter(volatilities[max_sharpe_idx], returns[max_sharpe_idx], color='red', s=100, label='Max Sharpe Ratio')

    text_str = (f"Max Sharpe Portfolio:\n"
                f"Return: {returns[max_sharpe_idx]:.2%}\n"
                f"Volatility: {volatilities[max_sharpe_idx]:.2%}\n"
                f"Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.2f}\n\n"
                "Recommendation:\nSelect portfolio near max Sharpe ratio for optimal risk-return balance.")

    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.grid(True)
    plt.legend()
    plt.gcf().text(0.65, 0.2, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0, 0.65, 1])
    plt.savefig(os.path.join(output_dir, "efficient_frontier.png"))
    plt.close()
    print("Saved efficient_frontier.png")


def print_recommendations(frontier_results):
    # Pick portfolio with max Sharpe ratio assuming risk-free rate ~0.02
    risk_free_rate = 0.02
    sharpe_ratios = [(res['return'] - risk_free_rate) / res['volatility'] for res in frontier_results]
    max_idx = np.argmax(sharpe_ratios)
    best = frontier_results[max_idx]

    print("\n--- Portfolio Optimization Recommendations ---")
    print(f"Max Sharpe Ratio Portfolio at return = {best['return']:.4f}, volatility = {best['volatility']:.4f}")
    print("Weights:")
    for i, w in enumerate(best['weights']):
        print(f"  Asset {i}: {w:.4f}")
    print("Recommendation: Consider this portfolio for optimal risk-adjusted returns.")

if __name__ == "__main__":
    # Load tickers from config.yaml (example)
    config_path = "./data/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        tickers = config.get("tickers", [])
    else:
        tickers = ["Asset_A", "Asset_B", "Asset_C", "Asset_D", "Asset_E"]

    n_assets = len(tickers)
    dates = pd.date_range("2023-01-01", periods=100, freq='B')

    # Generate synthetic returns data
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(100, n_assets) / 100, index=dates, columns=tickers)

    # Calculate expected returns (mean daily returns annualized)
    mean_daily_returns = returns.mean()
    expected_returns = mean_daily_returns.values * 252

    # Calculate covariance matrix (annualized)
    cov_matrix = returns.cov().values * 252

    # Example portfolio weights over time (random, normalized)
    weights = pd.DataFrame(np.random.rand(100, n_assets), index=dates, columns=tickers)
    weights = weights.div(weights.sum(axis=1), axis=0)

    # Portfolio returns time series
    portfolio_returns = (returns * weights).sum(axis=1)

    # Generate all plots
    plot_cumulative_returns(returns)
    plot_return_distribution(returns)
    plot_rolling_volatility(returns)
    plot_correlation_heatmap(returns)
    plot_drawdowns(returns)
    plot_portfolio_weights(weights)
    plot_portfolio_performance(portfolio_returns)
    plot_asset_allocation(weights)  # last date

    # Efficient frontier calculation and plotting
    frontier_results = compute_efficient_frontier(expected_returns, cov_matrix)
    plot_efficient_frontier(frontier_results)

    # Print actionable recommendations
    print_recommendations(frontier_results)
