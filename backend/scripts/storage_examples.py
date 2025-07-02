#!/usr/bin/env python3
"""
Data Storage Examples for Portfolio Optimization

This script demonstrates how to use the flexible data storage system
with different backends and configurations.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_storage import DataStorageManager, LocalFileStorage, SQLiteStorage
from fetch_data import PortfolioDataFetcher

def example_1_local_file_storage():
    """Example 1: Using local file storage with different formats."""
    print("=== Example 1: Local File Storage ===")
    
    # Test different file formats
    formats = ['pickle', 'csv']
    if 'pyarrow' in sys.modules:
        formats.append('parquet')
    
    for file_format in formats:
        print(f"\n--- Testing {file_format.upper()} format ---")
        
        # Create storage manager
        storage = DataStorageManager(
            storage_type="local_files",
            data_dir=f"data/example_{file_format}",
            file_format=file_format
        )
        
        # Load existing data if available
        try:
            prices, returns = storage.load_portfolio_data()
            if prices is not None:
                print(f"✅ Loaded existing data: {prices.shape}")
                
                # Test filtering
                recent_prices = storage.storage.load_price_data(
                    start_date='2024-01-01',
                    assets=['AAPL', 'MSFT', 'SPY']
                )
                if recent_prices is not None:
                    print(f"✅ Filtered data: {recent_prices.shape}")
                
                # Get storage info
                info = storage.get_storage_info()
                print(f"✅ Storage info: {info['assets_count']} assets, format: {info.get('file_format', 'N/A')}")
            else:
                print("⚠️  No existing data found")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def example_2_sqlite_storage():
    """Example 2: Using SQLite database storage."""
    print("\n=== Example 2: SQLite Database Storage ===")
    
    try:
        # Create SQLite storage
        storage = DataStorageManager(
            storage_type="sqlite",
            db_path="data/portfolio_example.db"
        )
        
        # Load existing data from file storage to migrate
        file_storage = DataStorageManager(storage_type="local_files", data_dir="data")
        prices, returns = file_storage.load_portfolio_data()
        
        if prices is not None and returns is not None:
            print(f"Found existing data to migrate: {prices.shape}")
            
            # Save to SQLite
            success = storage.save_portfolio_data(prices, returns, {
                'source': 'migrated_from_files',
                'migration_date': datetime.now().isoformat()
            })
            
            if success:
                print("✅ Successfully saved data to SQLite")
                
                # Test querying
                recent_data = storage.storage.load_price_data(
                    start_date='2024-01-01',
                    assets=['AAPL', 'MSFT']
                )
                
                if recent_data is not None:
                    print(f"✅ Queried recent data: {recent_data.shape}")
                
                # Get available assets
                assets = storage.storage.get_available_assets()
                print(f"✅ Available assets: {len(assets)} assets")
                
                # Get storage info
                info = storage.get_storage_info()
                print(f"✅ Database info: {info}")
                
            else:
                print("❌ Failed to save data to SQLite")
        else:
            print("⚠️  No source data available for SQLite example")
            
    except Exception as e:
        print(f"❌ SQLite example failed: {e}")

def example_3_performance_comparison():
    """Example 3: Compare performance of different storage formats."""
    print("\n=== Example 3: Performance Comparison ===")
    
    try:
        from data_storage import benchmark_storage_formats
        
        print("Running storage format benchmarks...")
        results = benchmark_storage_formats()
        
        print("\nBenchmark Results:")
        print("-" * 60)
        print("Format    | Save Time | Load Time | File Size | Integrity")
        print("-" * 60)
        
        for format_name, metrics in results.items():
            print(f"{format_name:8s} | {metrics['save_time']:8.3f}s | "
                  f"{metrics['load_time']:8.3f}s | {metrics['file_size_mb']:8.2f}MB | "
                  f"{'✅' if metrics['data_integrity'] else '❌'}")
        
        # Recommend best format
        if results:
            fastest_save = min(results.items(), key=lambda x: x[1]['save_time'])
            fastest_load = min(results.items(), key=lambda x: x[1]['load_time'])
            smallest_size = min(results.items(), key=lambda x: x[1]['file_size_mb'])
            
            print(f"\nRecommendations:")
            print(f"📊 Fastest save: {fastest_save[0]} ({fastest_save[1]['save_time']:.3f}s)")
            print(f"⚡ Fastest load: {fastest_load[0]} ({fastest_load[1]['load_time']:.3f}s)")
            print(f"💾 Smallest file: {smallest_size[0]} ({smallest_size[1]['file_size_mb']:.2f}MB)")
            
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")

def example_4_data_migration():
    """Example 4: Migrate data between storage types."""
    print("\n=== Example 4: Data Migration ===")
    
    try:
        # Start with file storage
        source_storage = DataStorageManager(
            storage_type="local_files",
            data_dir="data",
            file_format="pickle"
        )
        
        # Load data
        prices, returns = source_storage.load_portfolio_data()
        
        if prices is not None:
            print(f"Source data loaded: {prices.shape}")
            
            # Migrate to SQLite
            print("Migrating to SQLite...")
            success = source_storage.migrate_data(
                target_storage_type="sqlite",
                db_path="data/migrated_portfolio.db"
            )
            
            if success:
                print("✅ Migration to SQLite successful")
                
                # Verify migration
                target_storage = DataStorageManager(
                    storage_type="sqlite",
                    db_path="data/migrated_portfolio.db"
                )
                
                migrated_prices, migrated_returns = target_storage.load_portfolio_data()
                
                if migrated_prices is not None:
                    # Check data integrity
                    shapes_match = (migrated_prices.shape == prices.shape and 
                                  migrated_returns.shape == returns.shape)
                    
                    if shapes_match:
                        print("✅ Data migration integrity verified")
                    else:
                        print("⚠️  Data shapes don't match after migration")
                        print(f"Original: {prices.shape}, {returns.shape}")
                        print(f"Migrated: {migrated_prices.shape}, {migrated_returns.shape}")
                else:
                    print("❌ Could not load migrated data")
            else:
                print("❌ Migration failed")
        else:
            print("⚠️  No source data available for migration")
            
    except Exception as e:
        print(f"❌ Migration example failed: {e}")

def example_5_custom_storage_workflow():
    """Example 5: Custom workflow with storage management."""
    print("\n=== Example 5: Custom Storage Workflow ===")
    
    try:
        # Create a custom workflow that combines data fetching with storage
        print("Setting up custom portfolio workflow...")
        
        # Initialize storage
        storage = DataStorageManager(
            storage_type="local_files",
            data_dir="data/custom_workflow",
            file_format="pickle"
        )
        
        # Check if we have recent data
        info = storage.get_storage_info()
        
        if info['price_data_available']:
            # Load existing data
            prices, returns = storage.load_portfolio_data()
            print(f"✅ Loaded existing data: {prices.shape}")
            
            # Check data freshness
            latest_date = prices.index[-1]
            days_old = (datetime.now() - latest_date).days
            
            print(f"Data is {days_old} days old")
            
            if days_old > 7:  # If data is more than a week old
                print("Data is stale, fetching fresh data...")
                
                # Fetch fresh data (small sample for demo)
                fetcher = PortfolioDataFetcher(data_dir="temp_fetch")
                fetcher.all_assets = {
                    'AAPL': 'Apple Inc.',
                    'MSFT': 'Microsoft Corporation',
                    'SPY': 'S&P 500 ETF'
                }
                
                # Fetch recent data
                fresh_data = fetcher.fetch_all_assets(years_back=0.25)  # 3 months
                
                if fresh_data:
                    fresh_prices = fetcher.create_price_matrix(fresh_data)
                    fresh_returns = fetcher.calculate_returns(fresh_prices)
                    
                    # Save fresh data
                    storage.save_portfolio_data(fresh_prices, fresh_returns, {
                        'update_type': 'fresh_fetch',
                        'assets_updated': list(fresh_prices.columns)
                    })
                    
                    print("✅ Fresh data fetched and saved")
                else:
                    print("⚠️  Could not fetch fresh data")
            else:
                print("✅ Data is current, no update needed")
        else:
            print("⚠️  No existing data found, use fetch_data.py to get initial dataset")
        
        # Demonstrate data analysis with storage
        prices, returns = storage.load_portfolio_data()
        if prices is not None:
            # Quick analysis
            latest_prices = prices.iloc[-1]
            print(f"\nLatest Prices (sample):")
            for asset in list(latest_prices.index)[:3]:
                print(f"  {asset}: ${latest_prices[asset]:.2f}")
            
            # Storage efficiency
            print(f"\nStorage Details:")
            print(f"  Storage type: {info.get('storage_type', 'Unknown')}")
            print(f"  File format: {info.get('file_format', 'N/A')}")
            print(f"  Data directory: {info.get('data_directory', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Custom workflow failed: {e}")

def example_6_portfolio_analysis_with_storage():
    """Example 6: Portfolio analysis using storage system."""
    print("\n=== Example 6: Portfolio Analysis with Storage ===")
    
    try:
        # Load data using storage system
        storage = DataStorageManager(storage_type="local_files", data_dir="data")
        prices, returns = storage.load_portfolio_data()
        
        if prices is None or returns is None:
            print("⚠️  No data available. Run fetch_data.py first.")
            return
        
        print(f"Analyzing portfolio data: {prices.shape}")
        
        # Performance analysis for different time periods
        periods = {
            'YTD': '2024-01-01',
            'Last Year': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            'Last 6 Months': (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        }
        
        for period_name, start_date in periods.items():
            try:
                period_prices, period_returns = storage.load_portfolio_data(
                    start_date=start_date,
                    assets=['AAPL', 'MSFT', 'SPY', 'QQQ', 'GLD']  # Sample assets
                )
                
                if period_prices is not None and len(period_prices) > 0:
                    # Calculate performance
                    total_returns = (period_prices.iloc[-1] / period_prices.iloc[0] - 1)
                    
                    print(f"\n{period_name} Performance:")
                    for asset in total_returns.index:
                        print(f"  {asset}: {total_returns[asset]:8.2%}")
                else:
                    print(f"\n{period_name}: No data available")
                    
            except Exception as pe:
                print(f"\n{period_name}: Error - {pe}")
        
        # Storage efficiency report
        info = storage.get_storage_info()
        print(f"\nStorage Efficiency:")
        print(f"  Total assets: {info.get('assets_count', 0)}")
        print(f"  Date range: {info.get('date_range', 'Unknown')}")
        print(f"  Storage type: {info.get('storage_type', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def main():
    """Run all storage examples."""
    print("🗄️  PORTFOLIO DATA STORAGE EXAMPLES")
    print("=" * 60)
    
    # Example 1: Local file storage
    example_1_local_file_storage()
    
    # Example 2: SQLite storage
    example_2_sqlite_storage()
    
    # Example 3: Performance comparison
    example_3_performance_comparison()
    
    # Example 4: Data migration
    example_4_data_migration()
    
    # Example 5: Custom workflow
    example_5_custom_storage_workflow()
    
    # Example 6: Portfolio analysis
    example_6_portfolio_analysis_with_storage()
    
    print("\n" + "=" * 60)
    print("✅ All storage examples completed!")
    print("\nKey Takeaways:")
    print("📁 Use pickle format for best performance in development")
    print("🗃️  Use SQLite for structured queries and multi-user access")
    print("📊 Use CSV for data sharing and debugging")
    print("⚡ Consider Parquet for large datasets and analytics")

if __name__ == "__main__":
    main() 