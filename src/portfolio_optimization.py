import numpy as np
from scipy.optimize import minimize

def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

def portfolio_return(weights, mean_returns):
    return weights.T @ mean_returns

def efficient_frontier(mean_returns, cov_matrix, return_targets, short_selling=False):
    """
    Calculate Efficient Frontier for a range of target returns.
    
    Args:
        mean_returns: pd.Series of annualized returns
        cov_matrix: pd.DataFrame covariance matrix
        return_targets: list or np.array of target returns
        short_selling: allow negative weights if True
    
    Returns:
        List of dicts with keys: 'return', 'volatility', 'weights'
    """
    num_assets = len(mean_returns)
    results = []

    bounds = None if short_selling else [(0, 1) for _ in range(num_assets)]
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # sum weights = 1
    ]

    for target_return in return_targets:
        # Add target return constraint
        cons = constraints + [{'type': 'eq', 'fun': lambda w, target=target_return: portfolio_return(w, mean_returns) - target}]
        
        # Initial guess
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
        else:
            print(f"Optimization failed for target return {target_return}")

    return results
