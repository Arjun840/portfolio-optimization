#!/usr/bin/env python3
"""
Historical Data Fetcher for Portfolio Optimization

This script fetches historical price data for a diversified set of assets
across different sectors using the yfinance library.

Features:
- Downloads daily OHLC data for 30+ stocks across various sectors
- Covers the last 10 years of historical data
- Includes data quality checks and error handling
- Saves data in multiple formats (CSV, pickle)
- Provides comprehensive logging
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fetch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PortfolioDataFetcher:
    """
    A comprehensive data fetcher for portfolio optimization.
    
    This class handles downloading, processing, and storing historical
    price data for a diversified set of financial assets.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data fetcher with a specified data directory."""
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Diversified portfolio across sectors
        self.stock_universe = {
            # Technology
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'NVDA': 'NVIDIA Corporation',
            'META': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.',
            
            # Healthcare
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer Inc.',
            'UNH': 'UnitedHealth Group',
            'ABBV': 'AbbVie Inc.',
            
            # Financial Services
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp',
            'WFC': 'Wells Fargo & Company',
            'GS': 'Goldman Sachs Group',
            'V': 'Visa Inc.',
            
            # Consumer Discretionary
            'AMZN': 'Amazon.com Inc.',
            'HD': 'Home Depot Inc.',
            'MCD': 'McDonald\'s Corporation',
            'NKE': 'Nike Inc.',
            'SBUX': 'Starbucks Corporation',
            
            # Consumer Staples
            'PG': 'Procter & Gamble Co.',
            'KO': 'Coca-Cola Company',
            'PEP': 'PepsiCo Inc.',
            'WMT': 'Walmart Inc.',
            
            # Energy
            'XOM': 'Exxon Mobil Corporation',
            'CVX': 'Chevron Corporation',
            'COP': 'ConocoPhillips',
            
            # Industrial
            'BA': 'Boeing Company',
            'CAT': 'Caterpillar Inc.',
            'GE': 'General Electric Company',
            
            # Communication Services
            'VZ': 'Verizon Communications',
            'T': 'AT&T Inc.',
            'DIS': 'Walt Disney Company',
            
            # Real Estate & Utilities
            'NEE': 'NextEra Energy Inc.',
            'AMT': 'American Tower Corporation',
            
            # Materials
            'LIN': 'Linde plc',
            'NEM': 'Newmont Corporation'
        }
        
        # ETFs for broader market exposure
        self.etf_universe = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'Nasdaq 100 ETF',
            'IWM': 'Russell 2000 ETF',
            'VTI': 'Total Stock Market ETF',
            'EFA': 'International Developed Markets ETF',
            'EEM': 'Emerging Markets ETF',
            'TLT': 'Long-Term Treasury ETF',
            'GLD': 'Gold ETF',
            'VNQ': 'Real Estate ETF'
        }
        
        # Combine all assets
        self.all_assets = {**self.stock_universe, **self.etf_universe}
        
        logger.info(f"Initialized fetcher with {len(self.all_assets)} assets")
    
    def ensure_data_directory(self) -> None:
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def fetch_single_asset(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single asset.
        
        Args:
            symbol: Stock/ETF symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)
            
            if data.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return None
            
            # Clean column names
            data.columns = [col.title() for col in data.columns]
            
            # Add symbol column
            data['Symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_all_assets(self, years_back: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all assets in the universe.
        
        Args:
            years_back: Number of years of historical data to fetch
            
        Returns:
            Dictionary mapping symbols to their historical data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching data from {start_str} to {end_str}")
        
        asset_data = {}
        failed_assets = []
        
        for symbol, name in self.all_assets.items():
            logger.info(f"Fetching data for {symbol} ({name})")
            data = self.fetch_single_asset(symbol, start_str, end_str)
            
            if data is not None:
                asset_data[symbol] = data
            else:
                failed_assets.append(symbol)
        
        logger.info(f"Successfully fetched data for {len(asset_data)} assets")
        if failed_assets:
            logger.warning(f"Failed to fetch data for: {', '.join(failed_assets)}")
        
        return asset_data
    
    def create_price_matrix(self, asset_data: Dict[str, pd.DataFrame], 
                          price_type: str = 'Close') -> pd.DataFrame:
        """
        Create a matrix of prices with dates as index and symbols as columns.
        
        Args:
            asset_data: Dictionary of asset data
            price_type: Type of price to extract ('Close', 'Open', 'High', 'Low')
            
        Returns:
            DataFrame with aligned price data
        """
        price_data = {}
        
        for symbol, data in asset_data.items():
            if price_type in data.columns:
                price_data[symbol] = data[price_type]
        
        # Combine into single DataFrame
        price_matrix = pd.DataFrame(price_data)
        
        # Forward fill missing values (up to 5 days)
        price_matrix = price_matrix.fillna(method='ffill', limit=5)
        
        # Drop any remaining rows with too many missing values
        threshold = len(price_matrix.columns) * 0.8  # Keep rows with at least 80% data
        price_matrix = price_matrix.dropna(thresh=threshold)
        
        logger.info(f"Created price matrix with shape: {price_matrix.shape}")
        return price_matrix
    
    def calculate_returns(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Args:
            price_matrix: DataFrame with price data
            
        Returns:
            DataFrame with daily returns
        """
        returns = price_matrix.pct_change().dropna()
        logger.info(f"Calculated returns matrix with shape: {returns.shape}")
        return returns
    
    def data_quality_report(self, asset_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate a data quality report.
        
        Args:
            asset_data: Dictionary of asset data
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_assets': len(asset_data),
            'date_ranges': {},
            'data_points': {},
            'missing_data': {}
        }
        
        for symbol, data in asset_data.items():
            report['date_ranges'][symbol] = {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d'),
                'days': len(data)
            }
            
            report['data_points'][symbol] = len(data)
            report['missing_data'][symbol] = data.isnull().sum().to_dict()
        
        return report
    
    def save_data(self, asset_data: Dict[str, pd.DataFrame], 
                  price_matrix: pd.DataFrame, returns_matrix: pd.DataFrame) -> None:
        """
        Save all data to various formats.
        
        Args:
            asset_data: Raw asset data
            price_matrix: Processed price matrix
            returns_matrix: Calculated returns matrix
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual asset data
        individual_dir = os.path.join(self.data_dir, 'individual_assets')
        os.makedirs(individual_dir, exist_ok=True)
        
        for symbol, data in asset_data.items():
            data.to_csv(os.path.join(individual_dir, f'{symbol}.csv'))
        
        # Save combined matrices
        price_matrix.to_csv(os.path.join(self.data_dir, f'price_matrix_{timestamp}.csv'))
        returns_matrix.to_csv(os.path.join(self.data_dir, f'returns_matrix_{timestamp}.csv'))
        
        # Save as pickle for faster loading
        price_matrix.to_pickle(os.path.join(self.data_dir, f'price_matrix_{timestamp}.pkl'))
        returns_matrix.to_pickle(os.path.join(self.data_dir, f'returns_matrix_{timestamp}.pkl'))
        
        # Save latest versions (overwrite)
        price_matrix.to_csv(os.path.join(self.data_dir, 'price_matrix_latest.csv'))
        returns_matrix.to_csv(os.path.join(self.data_dir, 'returns_matrix_latest.csv'))
        price_matrix.to_pickle(os.path.join(self.data_dir, 'price_matrix_latest.pkl'))
        returns_matrix.to_pickle(os.path.join(self.data_dir, 'returns_matrix_latest.pkl'))
        
        logger.info(f"Data saved to {self.data_dir} directory")
    
    def run_full_pipeline(self, years_back: int = 10) -> None:
        """
        Run the complete data fetching and processing pipeline.
        
        Args:
            years_back: Number of years of historical data to fetch
        """
        logger.info("=== Starting Portfolio Data Fetching Pipeline ===")
        
        # Fetch raw data
        asset_data = self.fetch_all_assets(years_back)
        
        if not asset_data:
            logger.error("No data was successfully fetched. Exiting.")
            return
        
        # Create price matrix
        price_matrix = self.create_price_matrix(asset_data)
        
        # Calculate returns
        returns_matrix = self.calculate_returns(price_matrix)
        
        # Generate quality report
        quality_report = self.data_quality_report(asset_data)
        
        # Save data
        self.save_data(asset_data, price_matrix, returns_matrix)
        
        # Print summary
        self.print_summary(quality_report, price_matrix, returns_matrix)
        
        logger.info("=== Data Fetching Pipeline Completed ===")
    
    def print_summary(self, quality_report: Dict, 
                     price_matrix: pd.DataFrame, returns_matrix: pd.DataFrame) -> None:
        """Print a summary of the fetched data."""
        print("\n" + "="*60)
        print("PORTFOLIO DATA SUMMARY")
        print("="*60)
        
        print(f"Total Assets: {quality_report['total_assets']}")
        print(f"Price Matrix Shape: {price_matrix.shape}")
        print(f"Returns Matrix Shape: {returns_matrix.shape}")
        print(f"Date Range: {price_matrix.index.min().strftime('%Y-%m-%d')} to {price_matrix.index.max().strftime('%Y-%m-%d')}")
        
        print(f"\nAssets with most data points:")
        data_points = quality_report['data_points']
        top_assets = sorted(data_points.items(), key=lambda x: x[1], reverse=True)[:10]
        for symbol, points in top_assets:
            asset_name = self.all_assets.get(symbol, 'Unknown')
            print(f"  {symbol}: {points} days ({asset_name})")
        
        print(f"\nData saved to: {os.path.abspath(self.data_dir)}")
        print("="*60)


def main():
    """Main function to run the data fetcher."""
    fetcher = PortfolioDataFetcher()
    
    # Run with 10 years of data
    fetcher.run_full_pipeline(years_back=10)


if __name__ == "__main__":
    main() 