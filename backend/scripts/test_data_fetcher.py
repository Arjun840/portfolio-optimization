#!/usr/bin/env python3
"""
Test Suite for Portfolio Data Fetcher

This module contains comprehensive tests for the data fetching system,
including unit tests, integration tests, and data validation.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
import warnings

# Add the scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetch_data import PortfolioDataFetcher
from config import UNIVERSE_CONFIGS, DEFAULT_CONFIG

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestPortfolioDataFetcher(unittest.TestCase):
    """Test cases for the PortfolioDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create a fetcher with a small test universe
        self.test_assets = {
            'AAPL': 'Apple Inc.',
            'SPY': 'S&P 500 ETF',
            'GLD': 'Gold ETF'
        }
        
        self.fetcher = PortfolioDataFetcher(data_dir=self.test_dir)
        self.fetcher.all_assets = self.test_assets
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def test_fetcher_initialization(self):
        """Test that the fetcher initializes correctly."""
        self.assertEqual(self.fetcher.data_dir, self.test_dir)
        self.assertEqual(self.fetcher.all_assets, self.test_assets)
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_fetch_single_asset_valid_symbol(self):
        """Test fetching data for a single valid asset."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 1 month of data
        
        data = self.fetcher.fetch_single_asset(
            'AAPL', 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('Close', data.columns)
        self.assertIn('Symbol', data.columns)
        self.assertEqual(data['Symbol'].iloc[0], 'AAPL')
    
    def test_fetch_single_asset_invalid_symbol(self):
        """Test fetching data for an invalid asset symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = self.fetcher.fetch_single_asset(
            'INVALID_SYMBOL_XYZ', 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        self.assertIsNone(data)
    
    def test_create_price_matrix(self):
        """Test creation of price matrix from asset data."""
        # Create mock asset data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        mock_data = {}
        for symbol in self.test_assets.keys():
            mock_data[symbol] = pd.DataFrame({
                'Close': np.random.randn(10).cumsum() + 100,
                'Open': np.random.randn(10).cumsum() + 100,
                'High': np.random.randn(10).cumsum() + 100,
                'Low': np.random.randn(10).cumsum() + 100,
                'Volume': np.random.randint(1000, 10000, 10),
                'Symbol': symbol
            }, index=dates)
        
        price_matrix = self.fetcher.create_price_matrix(mock_data)
        
        self.assertIsInstance(price_matrix, pd.DataFrame)
        self.assertEqual(len(price_matrix.columns), len(self.test_assets))
        self.assertEqual(len(price_matrix), 10)
        
        # Check that all symbols are present
        for symbol in self.test_assets.keys():
            self.assertIn(symbol, price_matrix.columns)
    
    def test_calculate_returns(self):
        """Test calculation of returns from price data."""
        # Create mock price data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'AAPL': [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
            'SPY': [200, 201, 203, 202, 204, 205, 204, 206, 207, 208]
        }, index=dates)
        
        returns = self.fetcher.calculate_returns(prices)
        
        self.assertIsInstance(returns, pd.DataFrame)
        self.assertEqual(len(returns), 9)  # One less due to pct_change
        self.assertEqual(len(returns.columns), 2)
        
        # Check that returns are calculated correctly
        expected_aapl_return_1 = (101 - 100) / 100
        self.assertAlmostEqual(returns['AAPL'].iloc[0], expected_aapl_return_1, places=6)
    
    def test_data_quality_report(self):
        """Test generation of data quality report."""
        # Create mock asset data with some missing values
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        mock_data = {}
        for i, symbol in enumerate(self.test_assets.keys()):
            data = pd.DataFrame({
                'Close': np.random.randn(10).cumsum() + 100,
                'Volume': np.random.randint(1000, 10000, 10),
                'Symbol': symbol
            }, index=dates)
            
            # Add some missing data
            if i == 0:  # First asset has missing data
                data.loc[dates[2], 'Close'] = np.nan
                
            mock_data[symbol] = data
        
        report = self.fetcher.data_quality_report(mock_data)
        
        self.assertIn('total_assets', report)
        self.assertIn('date_ranges', report)
        self.assertIn('data_points', report)
        self.assertIn('missing_data', report)
        
        self.assertEqual(report['total_assets'], len(self.test_assets))
        
        # Check that missing data is detected
        first_symbol = list(self.test_assets.keys())[0]
        self.assertGreater(report['missing_data'][first_symbol]['Close'], 0)


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration settings."""
    
    def test_universe_configs_exist(self):
        """Test that all configured universes exist and are valid."""
        required_configs = ['sp500_top', 'dow30', 'tech_focus', 'etfs_only', 'diversified']
        
        for config_name in required_configs:
            self.assertIn(config_name, UNIVERSE_CONFIGS)
            universe = UNIVERSE_CONFIGS[config_name]
            self.assertIsInstance(universe, dict)
            self.assertGreater(len(universe), 0)
            
            # Check that all entries have symbol and name
            for symbol, name in universe.items():
                self.assertIsInstance(symbol, str)
                self.assertIsInstance(name, str)
                self.assertGreater(len(symbol), 0)
                self.assertGreater(len(name), 0)
    
    def test_default_config(self):
        """Test that default configuration is valid."""
        required_keys = ['years_back', 'price_type', 'data_directory']
        
        for key in required_keys:
            self.assertIn(key, DEFAULT_CONFIG)
        
        self.assertIsInstance(DEFAULT_CONFIG['years_back'], int)
        self.assertGreater(DEFAULT_CONFIG['years_back'], 0)
        self.assertIn(DEFAULT_CONFIG['price_type'], ['Close', 'Open', 'High', 'Low'])


class TestDataIntegrity(unittest.TestCase):
    """Test cases for data integrity and validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        # Load actual data if it exists
        cls.data_dir = "data"
        cls.prices = None
        cls.returns = None
        
        try:
            if os.path.exists(os.path.join(cls.data_dir, "price_matrix_latest.pkl")):
                cls.prices = pd.read_pickle(os.path.join(cls.data_dir, "price_matrix_latest.pkl"))
                cls.returns = pd.read_pickle(os.path.join(cls.data_dir, "returns_matrix_latest.pkl"))
        except Exception:
            pass
    
    def test_price_data_integrity(self):
        """Test integrity of actual price data."""
        if self.prices is None:
            self.skipTest("No price data available for testing")
        
        # Basic integrity checks
        self.assertGreater(len(self.prices), 100)  # At least 100 days of data
        self.assertGreater(len(self.prices.columns), 5)  # At least 5 assets
        
        # Check for reasonable price ranges
        for column in self.prices.columns:
            prices = self.prices[column].dropna()
            self.assertGreater(prices.min(), 0)  # Prices should be positive
            self.assertLess(prices.max() / prices.min(), 1000)  # No extreme price ratios
    
    def test_returns_data_integrity(self):
        """Test integrity of returns data."""
        if self.returns is None:
            self.skipTest("No returns data available for testing")
        
        # Check returns are reasonable (between -50% and +50% daily)
        for column in self.returns.columns:
            returns = self.returns[column].dropna()
            self.assertGreater(returns.min(), -0.5)  # No more than 50% daily loss
            self.assertLess(returns.max(), 0.5)     # No more than 50% daily gain
            
            # Check mean daily return is reasonable (between -10% and +10% annually)
            annual_return = returns.mean() * 252
            self.assertGreater(annual_return, -0.1)
            self.assertLess(annual_return, 0.1)
    
    def test_data_consistency(self):
        """Test consistency between price and returns data."""
        if self.prices is None or self.returns is None:
            self.skipTest("No data available for testing")
        
        # Returns should be one day shorter than prices
        self.assertEqual(len(self.returns), len(self.prices) - 1)
        
        # Column names should match
        self.assertEqual(set(self.returns.columns), set(self.prices.columns))
        
        # Date alignment
        self.assertEqual(self.returns.index[0], self.prices.index[1])
        self.assertEqual(self.returns.index[-1], self.prices.index[-1])


class TestPerformance(unittest.TestCase):
    """Test cases for performance and efficiency."""
    
    def test_small_dataset_performance(self):
        """Test performance with a small dataset."""
        import time
        
        # Create a temporary fetcher
        test_dir = tempfile.mkdtemp()
        try:
            fetcher = PortfolioDataFetcher(data_dir=test_dir)
            fetcher.all_assets = {'SPY': 'S&P 500 ETF', 'AAPL': 'Apple Inc.'}
            
            start_time = time.time()
            asset_data = fetcher.fetch_all_assets(years_back=1)  # 1 year only
            fetch_time = time.time() - start_time
            
            # Should complete within reasonable time (30 seconds for 2 assets, 1 year)
            self.assertLess(fetch_time, 30)
            
            if asset_data:
                start_time = time.time()
                price_matrix = fetcher.create_price_matrix(asset_data)
                returns_matrix = fetcher.calculate_returns(price_matrix)
                processing_time = time.time() - start_time
                
                # Processing should be very fast (under 1 second)
                self.assertLess(processing_time, 1)
                
        finally:
            shutil.rmtree(test_dir)


def run_quick_tests():
    """Run a subset of quick tests for development."""
    print("Running Quick Tests...")
    print("=" * 50)
    
    # Test 1: Configuration validation
    print("✓ Testing configuration...")
    try:
        assert 'sp500_top' in UNIVERSE_CONFIGS
        assert len(UNIVERSE_CONFIGS['dow30']) == 30
        print("  ✅ Configuration tests passed")
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
    
    # Test 2: Data loading test
    print("✓ Testing data loading...")
    try:
        if os.path.exists("data/price_matrix_latest.pkl"):
            prices = pd.read_pickle("data/price_matrix_latest.pkl")
            returns = pd.read_pickle("data/returns_matrix_latest.pkl")
            assert len(prices) > 100
            assert len(returns) == len(prices) - 1
            print(f"  ✅ Data loading passed ({len(prices)} days, {len(prices.columns)} assets)")
        else:
            print("  ⚠️  No data files found - run fetch_data.py first")
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
    
    # Test 3: Basic functionality
    print("✓ Testing basic functionality...")
    try:
        test_dir = tempfile.mkdtemp()
        fetcher = PortfolioDataFetcher(data_dir=test_dir)
        fetcher.all_assets = {'SPY': 'S&P 500 ETF'}
        
        # Test single asset fetch (just structure, not actual network call)
        assert hasattr(fetcher, 'fetch_single_asset')
        assert hasattr(fetcher, 'create_price_matrix')
        assert hasattr(fetcher, 'calculate_returns')
        
        shutil.rmtree(test_dir)
        print("  ✅ Basic functionality tests passed")
    except Exception as e:
        print(f"  ❌ Basic functionality failed: {e}")
    
    print("=" * 50)
    print("Quick tests completed!")


def run_full_test_suite():
    """Run the complete test suite."""
    print("Running Full Test Suite...")
    print("=" * 50)
    
    # Create test loader
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioDataFetcher))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Portfolio Data Fetcher')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick tests only (no network calls)')
    parser.add_argument('--full', action='store_true',
                       help='Run full test suite')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_tests()
    elif args.full:
        run_full_test_suite()
    else:
        # Default: run quick tests
        print("No test type specified. Running quick tests.")
        print("Use --quick for quick tests or --full for complete suite.")
        print()
        run_quick_tests() 