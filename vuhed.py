import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load the data
file_names = ['DISHTV.NS.csv', 'HATHWAY.NS.csv', 'NAZARA.NS.csv','NETWORK18.NS.csv','PVRINOX.NS.csv','SAREGAMA.NS.csv','SUNTV.NS.csv','TIPSINDLTD.NS.csv','TV18BRDCST.NS.csv','ZEEL.NS.csv']
data_frames = {file_name: pd.read_csv(file_name, parse_dates=True, index_col='Date') for file_name in file_names}

# Calculate daily returns
returns = pd.DataFrame()
for name, df in data_frames.items():
    returns[name] = df['Close'].pct_change().dropna()
returns.dropna(inplace=True)

# Calculate the covariance matrix of the returns
cov_matrix = returns.cov()

# Number of assets
num_assets = len(returns.columns)

# Objective function (portfolio variance)
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

# Constraints: sum of weights = 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for weights: each weight between 0 and 1 (no short selling)
bounds = tuple((0, 1) for asset in range(num_assets))

# Initial guess (equal distribution)
init_guess = num_assets * [1. / num_assets,]

# Optimize
result = minimize(portfolio_variance, init_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
min_var_weights = result.x

print("Minimum Variance Portfolio Weights:", min_var_weights)
