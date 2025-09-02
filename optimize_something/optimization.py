""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Saw Thanda Oo  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: soo7			  		  		    	 		 		   		 		  
GT ID: 904056931 		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	 	 		  		  		    	 		 		   		 		  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from util import get_data, plot_data  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def author():
    """Return the GT username of the student"""
    return "soo7"

def study_group():
    """Return list of GT usernames of study group members"""
    return ["904056931"]

def get_portfolio_value(prices, allocs, start_val=1.0):
    """Compute daily portfolio value given prices and allocations"""
    # Normalize prices to start at 1.0
    normed_prices = prices / prices.iloc[0]
    # Apply allocations
    allocated = normed_prices * allocs
    # Compute portfolio value
    port_val = allocated.sum(axis=1) * start_val
    return port_val

def get_portfolio_stats(port_val, daily_rf=0.0, samples_per_year=252.0):
    """Compute portfolio statistics"""
    # Calculate daily returns
    daily_returns = port_val.pct_change().iloc[1:]
    
    # Portfolio statistics
    cr = port_val.iloc[-1] / port_val.iloc[0] - 1  # Cumulative return
    adr = daily_returns.mean()  # Average daily return
    sddr = daily_returns.std()  # Standard deviation of daily returns (volatility)
    
    # Sharpe ratio using sample standard deviation
    sr = np.sqrt(samples_per_year) * (daily_returns - daily_rf).mean() / sddr
    
    return cr, adr, sddr, sr

def negative_sharpe_ratio(allocs, prices, daily_rf=0.0, samples_per_year=252.0):
    """Objective function: negative Sharpe ratio to minimize"""
    # Ensure allocations sum to 1.0
    allocs = allocs / np.sum(allocs)
    
    # Get portfolio value
    port_val = get_portfolio_value(prices, allocs)
    
    # Get portfolio stats
    cr, adr, sddr, sr = get_portfolio_stats(port_val, daily_rf, samples_per_year)
    
    # Return negative Sharpe ratio (minimizing negative = maximizing positive)
    return -sr

def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), 
                      syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    """Find optimal allocations that maximize Sharpe ratio"""
    
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    
    # Initial guess: uniform allocation
    n_assets = len(syms)
    initial_guess = np.array([1.0/n_assets] * n_assets)
    
    # Bounds: each allocation between 0.0 and 1.0
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    
    # Constraint: allocations must sum to 1.0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    
    # Optimize: minimize negative Sharpe ratio
    result = minimize(negative_sharpe_ratio, initial_guess, 
                     args=(prices, 0.0, 252.0),
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    # Get optimal allocations
    optimal_allocs = result.x
    
    # Normalize to ensure they sum to 1.0
    optimal_allocs = optimal_allocs / np.sum(optimal_allocs)
    
    # Get portfolio statistics with optimal allocations
    port_val = get_portfolio_value(prices, optimal_allocs)
    cr, adr, sddr, sr = get_portfolio_stats(port_val)
    
    # Generate plot if requested
    if gen_plot:
        # Create normalized comparison plot
        normed_port = port_val / port_val.iloc[0]
        normed_spy = prices_SPY / prices_SPY.iloc[0]
        
        df_temp = pd.concat([normed_port, normed_spy], 
                           keys=['Portfolio', 'SPY'], axis=1)
        
        plt.figure(figsize=(10, 6))
        plot_data(df_temp, title='Daily Portfolio Value and SPY', 
                 ylabel='Normalized Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Figure1.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return optimal_allocs, cr, adr, sddr, sr

def test_code():
    """Test the optimization function"""
    # Test with example parameters
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 1, 1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    
    print("Testing portfolio optimization...")
    allocs, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False)
    
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Optimal Allocations: {allocs}")
    print(f"Sharpe Ratio: {sr:.4f}")
    print(f"Volatility (stdev of daily returns): {sddr:.6f}")
    print(f"Average Daily Return: {adr:.6f}")
    print(f"Cumulative Return: {cr:.4f}")
    
    # Generate Figure 1 with required parameters
    print("\nGenerating Figure 1...")
    figure_start = dt.datetime(2008, 6, 1)
    figure_end = dt.datetime(2009, 6, 1)
    figure_symbols = ['IBM', 'X', 'GLD', 'JPM']
    
    figure_allocs, figure_cr, figure_adr, figure_sddr, figure_sr = optimize_portfolio(
        sd=figure_start, ed=figure_end, syms=figure_symbols, gen_plot=True)
    
    print(f"Figure 1 - Optimal Allocations: {figure_allocs}")
    print(f"Figure 1 - Sharpe Ratio: {figure_sr:.4f}")

if __name__ == "__main__":
    test_code()
