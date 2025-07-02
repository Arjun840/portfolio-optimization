#!/usr/bin/env python3
"""
Data Analysis Script for Portfolio Optimization

This script analyzes the fetched historical data and provides
insights about the portfolio assets.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def load_latest_data(data_dir="data"):
    """Load the latest price and returns data."""
    try:
        price_file = os.path.join(data_dir, "price_matrix_latest.pkl")
        returns_file = os.path.join(data_dir, "returns_matrix_latest.pkl")
        
        if os.path.exists(price_file) and os.path.exists(returns_file):
            prices = pd.read_pickle(price_file)
            returns = pd.read_pickle(returns_file)
            return prices, returns
        else:
            # Try CSV files
            price_file = os.path.join(data_dir, "price_matrix_latest.csv")
            returns_file = os.path.join(data_dir, "returns_matrix_latest.csv")
            
            prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
            returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            return prices, returns
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_portfolio_stats(returns_df):
    """Calculate portfolio statistics."""
    # Annualized statistics (assuming 252 trading days per year)
    annual_returns = returns_df.mean() * 252
    annual_volatility = returns_df.std() * np.sqrt(252)
    sharpe_ratios = annual_returns / annual_volatility
    
    # Cumulative returns
    cumulative_returns = (1 + returns_df).cumprod() - 1
    total_returns = cumulative_returns.iloc[-1]
    
    # Maximum drawdown
    running_max = (1 + returns_df).cumprod().expanding().max()
    drawdowns = (1 + returns_df).cumprod() / running_max - 1
    max_drawdowns = drawdowns.min()
    
    # Combine into DataFrame
    stats = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratios,
        'Total Return': total_returns,
        'Max Drawdown': max_drawdowns
    })
    
    return stats

def display_summary(prices_df, returns_df):
    """Display a summary of the data."""
    print("=" * 80)
    print("PORTFOLIO DATA ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Basic info
    print(f"Data Period: {prices_df.index[0].strftime('%Y-%m-%d')} to {prices_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of Assets: {len(prices_df.columns)}")
    print(f"Number of Trading Days: {len(prices_df)}")
    print(f"Years of Data: {len(prices_df) / 252:.1f}")
    
    # Asset categories
    print(f"\nAsset Breakdown:")
    etf_symbols = ['SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'EEM', 'TLT', 'GLD', 'VNQ']
    etfs = [col for col in prices_df.columns if col in etf_symbols]
    stocks = [col for col in prices_df.columns if col not in etf_symbols]
    
    print(f"  Individual Stocks: {len(stocks)}")
    print(f"  ETFs: {len(etfs)}")
    
    print(f"\nData Quality:")
    missing_data = prices_df.isnull().sum()
    print(f"  Assets with complete data: {(missing_data == 0).sum()}")
    if (missing_data > 0).any():
        print(f"  Assets with missing data: {(missing_data > 0).sum()}")
        assets_with_missing = missing_data[missing_data > 0]
        for asset, missing_count in assets_with_missing.items():
            print(f"    {asset}: {missing_count} missing days")

def display_top_performers(stats_df, metric='Annual Return', top_n=10):
    """Display top performing assets by a given metric."""
    print(f"\nTOP {top_n} ASSETS BY {metric.upper()}:")
    print("-" * 60)
    
    top_assets = stats_df.nlargest(top_n, metric)
    
    for i, (asset, row) in enumerate(top_assets.iterrows(), 1):
        print(f"{i:2d}. {asset:6s} | {metric}: {row[metric]:8.2%} | "
              f"Volatility: {row['Annual Volatility']:7.2%} | "
              f"Sharpe: {row['Sharpe Ratio']:6.2f}")

def display_risk_analysis(stats_df):
    """Display risk analysis."""
    print(f"\nRISK ANALYSIS:")
    print("-" * 60)
    
    # Highest volatility
    print("Most Volatile Assets:")
    most_volatile = stats_df.nlargest(5, 'Annual Volatility')
    for asset, row in most_volatile.iterrows():
        print(f"  {asset:6s}: {row['Annual Volatility']:7.2%} volatility")
    
    print("\nLowest Volatility Assets:")
    least_volatile = stats_df.nsmallest(5, 'Annual Volatility')
    for asset, row in least_volatile.iterrows():
        print(f"  {asset:6s}: {row['Annual Volatility']:7.2%} volatility")
    
    print("\nWorst Drawdowns:")
    worst_drawdowns = stats_df.nsmallest(5, 'Max Drawdown')
    for asset, row in worst_drawdowns.iterrows():
        print(f"  {asset:6s}: {row['Max Drawdown']:7.2%} max drawdown")

def display_correlation_insights(returns_df):
    """Display correlation insights."""
    print(f"\nCORRELATION INSIGHTS:")
    print("-" * 60)
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Find highly correlated pairs (excluding self-correlation)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.7:  # High positive correlation
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    print("Highly Correlated Asset Pairs (>0.7):")
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for asset1, asset2, corr in high_corr_pairs[:10]:
        print(f"  {asset1:6s} - {asset2:6s}: {corr:6.3f}")
    
    # Find diversifying assets (low correlation with SPY if available)
    if 'SPY' in corr_matrix.columns:
        spy_corrs = corr_matrix['SPY'].drop('SPY').sort_values()
        print(f"\nLowest Correlation with SPY (Diversifiers):")
        for asset, corr in spy_corrs.head(5).items():
            print(f"  {asset:6s}: {corr:6.3f}")

def main():
    """Main analysis function."""
    # Load data
    print("Loading portfolio data...")
    prices, returns = load_latest_data()
    
    if prices is None or returns is None:
        print("No data found! Please run fetch_data.py first.")
        return
    
    # Calculate statistics
    print("Calculating portfolio statistics...")
    stats = calculate_portfolio_stats(returns)
    
    # Display analysis
    display_summary(prices, returns)
    display_top_performers(stats, 'Annual Return')
    display_top_performers(stats, 'Sharpe Ratio')
    display_risk_analysis(stats)
    display_correlation_insights(returns)
    
    print(f"\nFull statistics saved to 'portfolio_stats.csv'")
    stats.to_csv('portfolio_stats.csv')
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 