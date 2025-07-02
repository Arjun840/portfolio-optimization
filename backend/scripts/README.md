# Portfolio Data Fetcher

A comprehensive system for fetching, processing, and analyzing historical price data for portfolio optimization using yfinance.

## Overview

This system fetches historical price data for a diversified set of assets across different sectors and asset classes. It includes stocks from major indices (S&P 500, Dow 30) and popular ETFs for broad market exposure.

## Features

- ✅ **Diversified Asset Universe**: 45+ assets across multiple sectors
- ✅ **10 Years of Historical Data**: Daily OHLC data from 2015-2025
- ✅ **Multiple Data Formats**: CSV and Pickle for flexibility
- ✅ **Data Quality Checks**: Missing data detection and handling
- ✅ **Portfolio Analysis**: Statistical analysis and risk metrics
- ✅ **Configurable Universes**: Easy to customize asset selection
- ✅ **Comprehensive Logging**: Detailed execution logs

## Current Dataset

Our current dataset includes:

### **Individual Stocks (36 assets)**
- **Technology**: AAPL, MSFT, GOOGL, NVDA, META, TSLA
- **Healthcare**: JNJ, PFE, UNH, ABBV
- **Financial**: JPM, BAC, WFC, GS, V
- **Consumer**: AMZN, HD, MCD, NKE, SBUX, PG, KO, PEP, WMT
- **Energy**: XOM, CVX, COP
- **Industrial**: BA, CAT, GE
- **Communication**: VZ, T, DIS
- **Utilities/REIT**: NEE, AMT
- **Materials**: LIN, NEM

### **ETFs (9 assets)**
- **Broad Market**: SPY, QQQ, IWM, VTI
- **International**: EFA, EEM
- **Alternative Assets**: TLT, GLD, VNQ

### **Key Statistics from Latest Analysis**
- **Data Period**: July 2015 - June 2025 (10 years)
- **Trading Days**: 2,512 days
- **Complete Data**: 100% (all 45 assets)
- **Top Performer**: NVDA (70.5% annual return, 1.41 Sharpe ratio)
- **Lowest Volatility**: Gold (GLD) at 14.5% annual volatility
- **Best Diversifier**: TLT (Long-term Treasury) with -0.18 correlation to SPY

## Quick Start

### 1. Basic Data Fetching
```bash
cd backend
source venv/bin/activate
python3 scripts/fetch_data.py
```

### 2. Analyze Fetched Data
```bash
python3 scripts/analyze_data.py
```

### 3. Run Examples
```bash
python3 scripts/example_usage.py
```

## File Structure

```
backend/scripts/
├── fetch_data.py      # Main data fetcher class
├── config.py          # Asset universe configurations
├── analyze_data.py    # Portfolio analysis tools
├── example_usage.py   # Usage examples
└── README.md          # This file

backend/data/
├── price_matrix_latest.csv     # Latest price data (CSV)
├── returns_matrix_latest.csv   # Latest returns data (CSV)
├── price_matrix_latest.pkl     # Latest price data (Pickle)
├── returns_matrix_latest.pkl   # Latest returns data (Pickle)
├── price_matrix_YYYYMMDD_HHMMSS.*  # Timestamped versions
├── returns_matrix_YYYYMMDD_HHMMSS.* # Timestamped versions
└── individual_assets/          # Individual CSV files per asset
    ├── AAPL.csv
    ├── MSFT.csv
    └── ...
```

## Configuration Options

The system supports multiple asset universe configurations in `config.py`:

- `sp500_top`: Top S&P 500 holdings by market cap
- `dow30`: All 30 Dow Jones Industrial Average stocks
- `tech_focus`: Technology-focused portfolio (20 stocks)
- `etfs_only`: Popular ETFs across asset classes
- `diversified`: Combined stocks and ETFs (default)

## Usage Examples

### Example 1: Fetch Dow 30 Data
```python
from fetch_data import PortfolioDataFetcher
from config import UNIVERSE_CONFIGS

fetcher = PortfolioDataFetcher(data_dir="data/dow30")
fetcher.all_assets = UNIVERSE_CONFIGS['dow30']
fetcher.run_full_pipeline(years_back=5)
```

### Example 2: Custom Asset Universe
```python
custom_assets = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'SPY': 'S&P 500 ETF',
    'TLT': 'Long-term Treasury ETF'
}

fetcher = PortfolioDataFetcher()
fetcher.all_assets = custom_assets
fetcher.run_full_pipeline(years_back=3)
```

### Example 3: Load and Analyze Data
```python
import pandas as pd

# Load latest data
prices = pd.read_pickle('data/price_matrix_latest.pkl')
returns = pd.read_pickle('data/returns_matrix_latest.pkl')

# Calculate annual statistics
annual_returns = returns.mean() * 252
annual_volatility = returns.std() * (252 ** 0.5)
sharpe_ratios = annual_returns / annual_volatility
```

## Data Quality Features

- **Missing Data Handling**: Forward-fill up to 5 days, drop assets with >20% missing data
- **Data Validation**: Automatic checks for data completeness and date ranges
- **Error Handling**: Robust error handling with detailed logging
- **Quality Reports**: Comprehensive data quality reporting

## Performance Optimizations

- **Pickle Format**: Fast binary serialization for large datasets
- **Incremental Updates**: Support for updating existing datasets
- **Efficient Memory Usage**: Optimized pandas operations
- **Parallel Processing**: Ready for parallel asset fetching (future enhancement)

## Key Insights from Current Dataset

### **Top Performers (2015-2025)**
1. **NVIDIA (NVDA)**: 70.5% annual return, 1.41 Sharpe ratio
2. **Tesla (TSLA)**: 45.9% annual return, 0.78 Sharpe ratio
3. **Microsoft (MSFT)**: 29.4% annual return, 1.08 Sharpe ratio

### **Risk Characteristics**
- **Most Volatile**: Tesla (59.2%), NVIDIA (50.1%), Boeing (41.3%)
- **Least Volatile**: Gold (14.5%), Treasuries (15.2%), International (17.6%)
- **Largest Drawdowns**: GE (-81%), Boeing (-78%), Meta (-77%)

### **Diversification Opportunities**
- **Best Diversifiers**: TLT (-0.18 correlation), GLD (0.05), NEM (0.20)
- **Highly Correlated**: SPY-VTI (0.996), SPY-QQQ (0.934), JPM-BAC (0.892)

## Next Steps

This dataset is ready for:
1. **Modern Portfolio Theory** optimization
2. **Risk parity** portfolio construction
3. **Factor analysis** and style attribution
4. **Backtesting** trading strategies
5. **Monte Carlo** simulations
6. **Machine learning** model training

## Troubleshooting

### Common Issues

1. **Symbol Not Found**: Check if the ticker symbol is correct (e.g., NKE not NIKE)
2. **Missing Data**: Some assets may have limited history or trading halts
3. **API Limits**: yfinance may throttle requests; add delays if needed
4. **Network Issues**: Ensure stable internet connection for data fetching

### Support

For issues or questions:
1. Check the logs in `data_fetch.log`
2. Review the data quality report from the analysis
3. Verify asset symbols in `config.py`

## Dependencies

- `yfinance`: Yahoo Finance data fetching
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `datetime`: Date/time handling

All dependencies are listed in `requirements.txt`. 