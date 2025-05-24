import pandas as pd
import numpy as np

def load_returns(filepath):
    """
    Load returns data from CSV or any source.
    Assumes a DataFrame with datetime index and columns as tickers.
    """
    returns = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return returns

def calculate_mean_covariance(returns_df, trading_days=252):
    """
    Calculate annualized mean returns and covariance matrix.
    """
    mean_returns = returns_df.mean() * trading_days
    cov_matrix = returns_df.cov() * trading_days
    return mean_returns, cov_matrix
