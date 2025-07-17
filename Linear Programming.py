import numpy as np
import pandas as pd
from scipy.optimize import linprog

def optimize_portfolio(prices_df, budget):
    # Assume buy on first day and sell on last day
    buy_prices = prices_df.iloc[:, 1]
    sell_prices = prices_df.iloc[:, -1]
    profit_per_share = sell_prices - buy_prices
    
    c = -profit_per_share.values
    A_ub = [buy_prices.values]
    b_ub = [budget]
    bounds = [(0, None)] * len(profit_per_share)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return result, profit_per_share


prices = pd.read_csv("stock_prices.csv")

budget = 100000

result, profit = optimize_portfolio(prices, budget)

# Print Optimal Result
prices['OptimalShares'] = result.x
prices['ExpectedProfit'] = profit * result.x
print(prices[['Stock', 'OptimalShares', 'ExpectedProfit']])
print("\nTotal expected profit:", prices['ExpectedProfit'].sum())
    

