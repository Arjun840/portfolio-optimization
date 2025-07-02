# Testing Guide for Portfolio Data Fetcher

This guide shows you different ways to test your portfolio data fetching system, from quick manual tests to comprehensive automated testing.

## 🚀 Quick Start - Manual Testing

### 1. Basic Functionality Test
```bash
cd backend
source venv/bin/activate

# Test the main data fetcher
python3 scripts/fetch_data.py

# Test the analysis
python3 scripts/analyze_data.py
```

### 2. Quick Automated Tests
```bash
# Run quick tests (no network calls, very fast)
python3 scripts/test_data_fetcher.py --quick

# Run full test suite (includes network calls, slower)
python3 scripts/test_data_fetcher.py --full
```

## 🧪 Testing Levels

### Level 1: Smoke Tests (30 seconds)
Quick checks to ensure basic functionality works:

```bash
# 1. Configuration check
python3 -c "from scripts.config import UNIVERSE_CONFIGS; print('✅ Config loaded')"

# 2. Import test
python3 -c "from scripts.fetch_data import PortfolioDataFetcher; print('✅ Imports work')"

# 3. Data loading test (if data exists)
python3 -c "import pandas as pd; print('✅ Data loaded:', pd.read_pickle('data/price_matrix_latest.pkl').shape if os.path.exists('data/price_matrix_latest.pkl') else 'No data')"
```

### Level 2: Unit Tests (2-5 minutes)
Test individual components:

```bash
# Run specific test classes
python3 -c "import scripts.test_data_fetcher as t; t.run_quick_tests()"

# Test configuration only
python3 -m unittest scripts.test_data_fetcher.TestConfiguration -v

# Test data integrity (if data exists)
python3 -m unittest scripts.test_data_fetcher.TestDataIntegrity -v
```

### Level 3: Integration Tests (5-15 minutes)
Test the full pipeline with real network calls:

```bash
# Full test suite
python3 scripts/test_data_fetcher.py --full

# Or run specific integration test
python3 -m unittest scripts.test_data_fetcher.TestPerformance -v
```

### Level 4: End-to-End Tests (Manual)
Test real-world scenarios:

```bash
# Test with small dataset
python3 scripts/example_usage.py

# Test different configurations
python3 -c "
from scripts.fetch_data import PortfolioDataFetcher
from scripts.config import UNIVERSE_CONFIGS
fetcher = PortfolioDataFetcher('data/test')
fetcher.all_assets = {'AAPL': 'Apple', 'SPY': 'S&P 500'}
fetcher.run_full_pipeline(years_back=1)
"
```

## 🔧 Custom Testing

### Test Your Own Asset Universe
```python
# Create a test script: test_my_assets.py
from scripts.fetch_data import PortfolioDataFetcher

# Your custom assets
my_assets = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Google',
    'BTC-USD': 'Bitcoin',  # Test crypto
    'GLD': 'Gold ETF'
}

fetcher = PortfolioDataFetcher('data/my_test')
fetcher.all_assets = my_assets

# Test with short timeframe first
try:
    fetcher.run_full_pipeline(years_back=1)
    print("✅ Test passed!")
except Exception as e:
    print(f"❌ Test failed: {e}")
```

### Performance Testing
```python
import time
from scripts.fetch_data import PortfolioDataFetcher

# Time the fetch process
start = time.time()
fetcher = PortfolioDataFetcher('data/perf_test')
fetcher.all_assets = {'SPY': 'S&P 500', 'QQQ': 'Nasdaq'}
fetcher.run_full_pipeline(years_back=2)
duration = time.time() - start

print(f"Fetch completed in {duration:.1f} seconds")
# Should be under 30 seconds for 2 assets, 2 years
```

## 🐛 Debugging Tests

### Check Data Quality
```python
from scripts.analyze_data import load_latest_data

prices, returns = load_latest_data()
if prices is not None:
    print(f"Price data shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Missing data: {prices.isnull().sum().sum()}")
    print(f"Assets: {list(prices.columns)}")
else:
    print("No data found")
```

### Test Network Connectivity
```python
import yfinance as yf

# Test if yfinance works
try:
    spy = yf.Ticker("SPY")
    data = spy.history(period="5d")
    print(f"✅ Network test passed: {len(data)} days fetched")
except Exception as e:
    print(f"❌ Network test failed: {e}")
```

### Validate Specific Assets
```python
from scripts.fetch_data import PortfolioDataFetcher

fetcher = PortfolioDataFetcher()

# Test problematic symbols
test_symbols = ['AAPL', 'BRK-B', 'INVALID_SYMBOL']
for symbol in test_symbols:
    data = fetcher.fetch_single_asset(symbol, '2024-01-01', '2024-01-31')
    if data is not None:
        print(f"✅ {symbol}: {len(data)} days")
    else:
        print(f"❌ {symbol}: Failed")
```

## 📊 Test Data Validation

### Return Reasonableness Check
```python
import pandas as pd
import numpy as np

# Load returns data
returns = pd.read_pickle('data/returns_matrix_latest.pkl')

# Check for extreme returns
for col in returns.columns:
    daily_returns = returns[col].dropna()
    extreme_gains = (daily_returns > 0.2).sum()  # >20% daily gain
    extreme_losses = (daily_returns < -0.2).sum()  # >20% daily loss
    
    if extreme_gains > 10 or extreme_losses > 10:
        print(f"⚠️  {col}: {extreme_gains} extreme gains, {extreme_losses} extreme losses")
    else:
        print(f"✅ {col}: Returns look reasonable")
```

### Correlation Sanity Check
```python
# Check for perfect correlations (might indicate data issues)
corr_matrix = returns.corr()
perfect_corr = []

for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        corr_val = corr_matrix.iloc[i, j]
        if corr_val > 0.99:  # Nearly perfect correlation
            asset1 = corr_matrix.index[i]
            asset2 = corr_matrix.index[j]
            perfect_corr.append((asset1, asset2, corr_val))

if perfect_corr:
    print("⚠️  High correlations found:")
    for a1, a2, corr in perfect_corr:
        print(f"   {a1} - {a2}: {corr:.3f}")
else:
    print("✅ No suspiciously high correlations")
```

## 🔄 Continuous Testing

### Set up a Daily Test
Create `daily_test.py`:
```python
#!/usr/bin/env python3
"""Daily automated test"""
import os
from datetime import datetime
from scripts.test_data_fetcher import run_quick_tests

# Log file
log_file = f"test_log_{datetime.now().strftime('%Y%m%d')}.txt"

print(f"Running daily tests... (log: {log_file})")
run_quick_tests()

# Test data freshness
from scripts.analyze_data import load_latest_data
prices, returns = load_latest_data()
if prices is not None:
    latest_date = prices.index[-1]
    days_old = (datetime.now() - latest_date).days
    if days_old > 7:
        print(f"⚠️  Data is {days_old} days old - consider updating")
    else:
        print(f"✅ Data is current ({days_old} days old)")
```

### Run Tests in CI/CD
```bash
# Add to your deployment script
echo "Running portfolio data tests..."
cd backend
source venv/bin/activate
python3 scripts/test_data_fetcher.py --quick
if [ $? -eq 0 ]; then
    echo "✅ Tests passed"
else
    echo "❌ Tests failed"
    exit 1
fi
```

## 🎯 Test Coverage

Our test suite covers:

- ✅ **Configuration validation**: Asset universes, default settings
- ✅ **Data fetching**: Single assets, error handling, invalid symbols  
- ✅ **Data processing**: Price matrices, returns calculation
- ✅ **Data quality**: Missing data detection, integrity checks
- ✅ **Performance**: Timing benchmarks, memory usage
- ✅ **Integration**: Full pipeline end-to-end
- ✅ **Real data validation**: Price ranges, return distributions

## 🆘 Troubleshooting Test Failures

### Common Issues and Solutions

**1. Import Errors**
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"
python3 -c "import scripts.fetch_data; print('✅ Import fixed')"
```

**2. Network/API Issues**
```bash
# Test yfinance directly
python3 -c "import yfinance as yf; print(yf.Ticker('SPY').history(period='1d'))"
```

**3. Data File Issues**
```bash
# Check file permissions and sizes
ls -la data/
# Regenerate data if needed
python3 scripts/fetch_data.py
```

**4. Memory Issues with Large Datasets**
```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

## 🏆 Best Practices

1. **Always run quick tests first** before data fetching
2. **Test with small datasets** before full fetches  
3. **Check data quality** after each fetch
4. **Monitor performance** for regression detection
5. **Keep test data separate** from production data
6. **Document any test failures** and their solutions

---

**Remember**: Good testing saves time in the long run by catching issues early! 