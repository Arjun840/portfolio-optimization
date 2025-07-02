#!/usr/bin/env python3
"""
Example Usage of Portfolio Data Fetcher

This script demonstrates different ways to use the data fetcher with
various asset universes and configurations.
"""

import sys
import os

# Add the scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetch_data import PortfolioDataFetcher
from config import UNIVERSE_CONFIGS, DEFAULT_CONFIG
import pandas as pd

def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("=== Example 1: Basic Usage ===")
    
    # Create fetcher with default settings
    fetcher = PortfolioDataFetcher(data_dir="data/basic_example")
    
    # Run the full pipeline with 5 years of data
    fetcher.run_full_pipeline(years_back=5)

def example_dow30_portfolio():
    """Example 2: Fetch data for Dow 30 stocks only."""
    print("\n=== Example 2: Dow 30 Portfolio ===")
    
    # Create a custom fetcher for Dow 30
    fetcher = PortfolioDataFetcher(data_dir="data/dow30")
    
    # Override the asset universe with Dow 30
    fetcher.all_assets = UNIVERSE_CONFIGS['dow30']
    
    # Run with 7 years of data
    fetcher.run_full_pipeline(years_back=7)

def example_tech_focus():
    """Example 3: Technology-focused portfolio."""
    print("\n=== Example 3: Technology Focus ===")
    
    fetcher = PortfolioDataFetcher(data_dir="data/tech_focus")
    fetcher.all_assets = UNIVERSE_CONFIGS['tech_focus']
    
    # Shorter timeframe for tech stocks
    fetcher.run_full_pipeline(years_back=3)

def example_etfs_only():
    """Example 4: ETFs only for broad market exposure."""
    print("\n=== Example 4: ETFs Only ===")
    
    fetcher = PortfolioDataFetcher(data_dir="data/etfs_only")
    fetcher.all_assets = UNIVERSE_CONFIGS['etfs_only']
    
    # Longer timeframe for ETFs
    fetcher.run_full_pipeline(years_back=15)

def example_custom_analysis():
    """Example 5: Custom analysis with fetched data."""
    print("\n=== Example 5: Custom Analysis ===")
    
    # Fetch a small set of data for demonstration
    fetcher = PortfolioDataFetcher(data_dir="data/analysis_example")
    
    # Use a subset of blue-chip stocks
    blue_chips = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'JNJ': 'Johnson & Johnson',
        'PG': 'Procter & Gamble Co.',
        'KO': 'Coca-Cola Company'
    }
    
    fetcher.all_assets = blue_chips
    
    # Fetch data
    asset_data = fetcher.fetch_all_assets(years_back=5)
    
    if asset_data:
        # Create price matrix
        price_matrix = fetcher.create_price_matrix(asset_data)
        returns_matrix = fetcher.calculate_returns(price_matrix)
        
        # Perform some basic analysis
        print(f"\nPrice Matrix Shape: {price_matrix.shape}")
        print(f"Returns Matrix Shape: {returns_matrix.shape}")
        
        # Calculate some basic statistics
        print(f"\nBasic Statistics (Annualized):")
        print("-" * 40)
        
        annual_returns = returns_matrix.mean() * 252
        annual_volatility = returns_matrix.std() * (252 ** 0.5)
        sharpe_ratios = annual_returns / annual_volatility
        
        stats_df = pd.DataFrame({
            'Annual Return': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratios
        })
        
        print(stats_df.round(4))
        
        # Correlation matrix
        print(f"\nCorrelation Matrix:")
        print("-" * 40)
        correlation_matrix = returns_matrix.corr()
        print(correlation_matrix.round(3))
        
        # Save the analysis
        fetcher.save_data(asset_data, price_matrix, returns_matrix)

def example_incremental_update():
    """Example 6: Incremental data update."""
    print("\n=== Example 6: Incremental Update ===")
    
    # First, fetch historical data
    fetcher = PortfolioDataFetcher(data_dir="data/incremental")
    
    # Small universe for demonstration
    small_universe = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq 100 ETF',
        'TLT': 'Long-Term Treasury ETF',
        'GLD': 'Gold ETF'
    }
    
    fetcher.all_assets = small_universe
    
    # Initial fetch
    print("Fetching initial dataset...")
    fetcher.run_full_pipeline(years_back=2)
    
    # You could later update this with more recent data by:
    # 1. Loading existing data
    # 2. Fetching only recent dates
    # 3. Appending new data
    print("Initial dataset complete. You can later update this incrementally.")

def main():
    """Run all examples."""
    print("Portfolio Data Fetcher - Usage Examples")
    print("=" * 50)
    
    # Example 1: Basic usage
    example_basic_usage()
    
    # Example 2: Specific asset universe
    example_dow30_portfolio()
    
    # Example 3: Sector focus
    example_tech_focus()
    
    # Example 4: ETFs only
    example_etfs_only()
    
    # Example 5: Custom analysis
    example_custom_analysis()
    
    # Example 6: Incremental updates
    example_incremental_update()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check the 'data/' directory for output files.")

if __name__ == "__main__":
    main() 