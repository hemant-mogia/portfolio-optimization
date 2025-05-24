import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def load_returns(processed_dir="./data/processed"):
    path = os.path.join(processed_dir, "daily_returns.csv")
    print(f"Loading returns data from {path}...")
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Returns data loaded: {returns.shape[0]} rows, {returns.shape[1]} assets.")
    return returns

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio(returns, risk_free_rate=0.0):
    print("Starting portfolio optimization to maximize Sharpe ratio...")
    mean_returns = returns.mean() * 252  # Annualize returns
    cov_matrix = returns.cov() * 252     # Annualize covariance

    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]

    result = minimize(negative_sharpe, initial_guess, args=args, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    if not result.success:
        raise Exception(f"Optimization failed: {result.message}")

    weights = result.x
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    print(f"Optimization successful.")
    print(f"Expected annual return: {ret:.4f}")
    print(f"Annual volatility: {vol:.4f}")
    print(f"Sharpe ratio: {sharpe:.4f}")

    return weights

def save_weights(weights, tickers, output_dir="./outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(weights, index=tickers, columns=["weight"])
    path = os.path.join(output_dir, "optimized_weights.csv")
    df.to_csv(path)
    print(f"Optimized weights saved to {path}.")

#--------------------------------------EFFICIENT FRONTIER----------------------

def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

def portfolio_return(weights, mean_returns):
    return weights.T @ mean_returns

def efficient_frontier(mean_returns, cov_matrix, return_targets, short_selling=False):
    print("Calculating efficient frontier...")
    num_assets = len(mean_returns)
    results = []

    bounds = None if short_selling else [(0, 1) for _ in range(num_assets)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    for i, target_return in enumerate(return_targets, 1):
        cons = constraints + [{'type': 'eq', 'fun': lambda w, target=target_return: portfolio_return(w, mean_returns) - target}]
        init_guess = np.array(num_assets * [1. / num_assets])

        res = minimize(portfolio_variance, init_guess,
                       args=(cov_matrix,),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=cons)
        if res.success:
            volatility = np.sqrt(res.fun)
            weights = res.x
            results.append({
                'return': target_return,
                'volatility': volatility,
                'weights': weights
            })
            print(f"  {i}/{len(return_targets)}: Return={target_return:.4f}, Volatility={volatility:.4f}")
        else:
            print(f"  Optimization failed for target return {target_return:.4f}")

    print("Efficient frontier calculation completed.")
    return results

#--------------------------------------EFFICIENT FRONTIER----------------------

if __name__ == "__main__":
    returns = load_returns()
    tickers = returns.columns.tolist()

    # Optimize portfolio by maximizing Sharpe ratio
    weights = optimize_portfolio(returns)
    save_weights(weights, tickers)

    # Prepare inputs for efficient frontier
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return_targets = np.linspace(mean_returns.min(), mean_returns.max(), 20)

    # Calculate efficient frontier
    frontier = efficient_frontier(mean_returns.values, cov_matrix.values, return_targets)

    # (Optional) You can save frontier weights or plot it here...
