# Portfolio Data Storage Guide

A complete guide to storing and managing historical financial data for portfolio optimization projects.

## 🗄️ Storage Options Overview

Your portfolio optimization project now supports multiple storage backends optimized for different use cases:

### **Local Files** (Best for Development)
- **Formats**: CSV, Pickle, Parquet
- **Best for**: Development, prototyping, small datasets
- **Performance**: ⚡ Pickle (fastest), 💾 Parquet (smallest), 📊 CSV (readable)

### **SQLite Database** (Best for Production)
- **Best for**: Multi-user access, structured queries, reliability
- **Performance**: Good for medium datasets (< 1M rows)
- **Features**: ACID compliance, concurrent reads, backup-friendly

### **PostgreSQL** (Best for Scale)
- **Best for**: Large datasets, high concurrency, enterprise use
- **Performance**: Excellent for large datasets (> 1M rows)
- **Features**: Advanced queries, replication, extensions

## 🚀 Quick Start

### Basic Usage
```python
from scripts.storage_config import create_storage_manager

# Use development configuration (fast local files)
storage = create_storage_manager("development")

# Save your portfolio data
storage.save_portfolio_data(prices, returns)

# Load data back
prices, returns = storage.load_portfolio_data()
```

### Load Existing Data
```python
# Load all your existing portfolio data
storage = create_storage_manager("development")
prices, returns = storage.load_portfolio_data()

print(f"Loaded: {prices.shape[0]} days, {prices.shape[1]} assets")
```

## 📊 Environment Configurations

### Development Environment
```python
# Fast local storage with pickle format
storage = create_storage_manager("development")
# Uses: data/dev/ directory, pickle format, auto-backup enabled
```

### Testing Environment  
```python
# Isolated test data
storage = create_storage_manager("testing")
# Uses: data/test/ directory, no backups, isolated from main data
```

### Production Environment
```python
# Reliable SQLite database
storage = create_storage_manager("production")
# Uses: SQLite database, auto-backup, ACID compliance
```

### Analytics Environment
```python
# Optimized for large datasets
storage = create_storage_manager("analytics")
# Uses: Parquet format, compression, optimized for analytics
```

### Data Sharing
```python
# Export data in CSV format
storage = create_storage_manager("sharing")
# Uses: CSV format, data/export/ directory, human-readable
```

## 💡 Practical Examples

### Example 1: Development Workflow
```python
from scripts.storage_config import create_storage_manager
from scripts.fetch_data import PortfolioDataFetcher

# 1. Set up development storage
storage = create_storage_manager("development")

# 2. Fetch fresh data
fetcher = PortfolioDataFetcher()
asset_data = fetcher.fetch_all_assets(years_back=5)

# 3. Process and save
prices = fetcher.create_price_matrix(asset_data)
returns = fetcher.calculate_returns(prices)

metadata = {
    'fetch_date': datetime.now().isoformat(),
    'data_source': 'yfinance',
    'years_back': 5
}

storage.save_portfolio_data(prices, returns, metadata)
print("✅ Data saved to development storage")
```

### Example 2: Data Analysis with Filtering
```python
# Load specific assets and date ranges
storage = create_storage_manager("development")

# Get tech stocks for last year
tech_prices, tech_returns = storage.load_portfolio_data(
    start_date='2024-01-01',
    assets=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
)

# Quick analysis
if tech_prices is not None:
    annual_returns = tech_returns.mean() * 252
    print("Tech Stock Annual Returns:")
    for asset, ret in annual_returns.items():
        print(f"  {asset}: {ret:.2%}")
```

### Example 3: Production Deployment
```python
# Set up production storage with SQLite
storage = create_storage_manager("production")

# Load development data and migrate
dev_storage = create_storage_manager("development")
prices, returns = dev_storage.load_portfolio_data()

if prices is not None:
    # Save to production database
    success = storage.save_portfolio_data(prices, returns, {
        'deployment_date': datetime.now().isoformat(),
        'environment': 'production',
        'source': 'migrated_from_development'
    })
    
    if success:
        print("✅ Data deployed to production")
        
        # Verify deployment
        prod_info = storage.get_storage_info()
        print(f"Production DB: {prod_info['assets_count']} assets")
```

### Example 4: Data Export for Sharing
```python
# Export data to CSV for sharing
storage = create_storage_manager("sharing")

# Load latest data
main_storage = create_storage_manager("development")
prices, returns = main_storage.load_portfolio_data()

if prices is not None:
    # Save in CSV format for sharing
    storage.save_portfolio_data(prices, returns, {
        'export_purpose': 'data_sharing',
        'format': 'csv',
        'exported_by': 'portfolio_optimizer'
    })
    
    print("✅ Data exported to CSV in data/export/")
    print("Ready for sharing with external tools")
```

### Example 5: Performance Benchmarking
```python
import time
from scripts.data_storage import benchmark_storage_formats

# Compare storage formats
print("Benchmarking storage formats...")
results = benchmark_storage_formats()

print("Format Performance Comparison:")
for format_name, metrics in results.items():
    print(f"{format_name}:")
    print(f"  Save: {metrics['save_time']:.3f}s")
    print(f"  Load: {metrics['load_time']:.3f}s") 
    print(f"  Size: {metrics['file_size_mb']:.2f}MB")
    print()
```

## 🔧 Advanced Usage

### Custom Configuration
```python
from scripts.storage_config import create_custom_config, StorageType, FileFormat

# Create custom configuration
custom_config = create_custom_config(
    storage_type="local_files",
    data_dir="data/custom",
    file_format=FileFormat.PARQUET,
    description="Custom high-performance storage"
)

# Use custom configuration
storage = DataStorageManager(
    storage_type=custom_config.storage_type.value,
    data_dir=custom_config.data_dir,
    file_format=custom_config.file_format.value
)
```

### Data Migration Between Backends
```python
# Migrate from local files to SQLite
source_storage = create_storage_manager("development")
target_storage = create_storage_manager("production")

# Perform migration
success = source_storage.migrate_data(
    target_storage_type="sqlite",
    db_path="data/production/portfolio.db"
)

if success:
    print("✅ Successfully migrated to SQLite")
else:
    print("❌ Migration failed")
```

### Environment Auto-Detection
```python
from scripts.storage_config import detect_environment, create_storage_manager

# Automatically detect environment
env = detect_environment()
print(f"Detected environment: {env}")

# Use appropriate storage
storage = create_storage_manager(env)
```

## 📈 Performance Guidelines

### **For Development (< 1M data points)**
```python
# Use pickle format for speed
storage = create_storage_manager("development")
# Expected: Save <0.1s, Load <0.05s
```

### **For Analytics (> 1M data points)**
```python
# Use Parquet for compression and speed
storage = create_storage_manager("analytics")
# Expected: 50% smaller files, fast columnar access
```

### **For Production (reliability needed)**
```python
# Use SQLite for ACID compliance
storage = create_storage_manager("production")  
# Expected: Concurrent access, backup-friendly
```

### **For Sharing (interoperability)**
```python
# Use CSV for maximum compatibility
storage = create_storage_manager("sharing")
# Expected: Human-readable, works with Excel/R/Python
```

## 🛠 Configuration Options

### Storage Types
- **`local_files`**: File-based storage (CSV/Pickle/Parquet)
- **`sqlite`**: SQLite database (single file, serverless)
- **`postgresql`**: PostgreSQL database (requires server)

### File Formats (Local Files Only)
- **`pickle`**: Python binary format (fastest, smallest)
- **`csv`**: Comma-separated values (human-readable)
- **`parquet`**: Columnar format (compressed, analytics-optimized)

### Environment Variables
```bash
# Set environment for auto-detection
export PORTFOLIO_ENV=production

# Use in your code
storage = create_storage_manager()  # Auto-detects environment
```

## 🔍 Monitoring and Maintenance

### Check Storage Status
```python
storage = create_storage_manager("development")
info = storage.get_storage_info()

print(f"Storage Type: {info['storage_type']}")
print(f"Assets: {info['assets_count']}")
print(f"Date Range: {info['date_range']}")
print(f"Data Available: {info['price_data_available']}")
```

### Data Validation
```python
prices, returns = storage.load_portfolio_data()

if prices is not None:
    # Check data quality
    missing_data = prices.isnull().sum().sum()
    date_gaps = len(pd.date_range(prices.index[0], prices.index[-1], freq='D')) - len(prices)
    
    print(f"Missing values: {missing_data}")
    print(f"Date gaps: {date_gaps}")
    print(f"Data integrity: {'✅' if missing_data == 0 else '⚠️'}")
```

## 🎯 Best Practices

### 1. **Choose the Right Storage for Your Use Case**
- **Development**: Use pickle format for speed
- **Production**: Use SQLite for reliability  
- **Analytics**: Use Parquet for large datasets
- **Sharing**: Use CSV for compatibility

### 2. **Data Organization**
```python
# Good: Environment-specific directories
storage_dev = create_storage_manager("development")     # data/dev/
storage_prod = create_storage_manager("production")     # data/production/
storage_test = create_storage_manager("testing")        # data/test/
```

### 3. **Error Handling**
```python
try:
    storage = create_storage_manager("production")
    prices, returns = storage.load_portfolio_data()
    
    if prices is None:
        print("No data available, fetching fresh data...")
        # Fallback to data fetching
    else:
        print(f"Loaded {prices.shape[0]} days of data")
        
except Exception as e:
    print(f"Storage error: {e}")
    # Implement fallback strategy
```

### 4. **Performance Monitoring**
```python
import time

start_time = time.time()
storage.save_portfolio_data(prices, returns)
save_time = time.time() - start_time

print(f"Save performance: {save_time:.3f}s for {prices.shape} data")
# Monitor and optimize if save_time > expected thresholds
```

## 🚨 Troubleshooting

### Common Issues

**1. "No data available"**
```python
# Check if files exist
storage = create_storage_manager("development")
info = storage.get_storage_info()
print(f"Data available: {info['price_data_available']}")

# If False, run data fetching first
if not info['price_data_available']:
    print("Run: python3 scripts/fetch_data.py")
```

**2. "Storage format not supported"**
```python
# Install missing dependencies
# For Parquet: pip install pyarrow
# For PostgreSQL: pip install psycopg2-binary
```

**3. "Database locked" (SQLite)**
```python
# Ensure no other processes are using the database
# Check file permissions
# Consider using WAL mode for concurrent access
```

---

## 📚 Summary

Your portfolio optimization project now has a flexible, high-performance storage system that can scale from development to production:

✅ **Multiple Storage Backends**: Files, SQLite, PostgreSQL  
✅ **Optimized Formats**: Pickle (speed), Parquet (compression), CSV (sharing)  
✅ **Environment Configurations**: Dev, Test, Production, Analytics, Sharing  
✅ **Easy Migration**: Move data between storage types seamlessly  
✅ **Performance Monitoring**: Built-in benchmarking and optimization  
✅ **Production Ready**: ACID compliance, backups, concurrent access  

**Next Steps**: Choose your storage configuration and start building your portfolio optimization algorithms with confidence in your data infrastructure! 