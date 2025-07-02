"""
Configuration file for Portfolio Data Fetcher

This file contains asset universes, parameters, and settings for the data fetcher.
You can easily modify the assets here without changing the main code.
"""

# S&P 500 Top Holdings by Market Cap (diversified across sectors)
SP500_TOP_HOLDINGS = {
    # Technology (Large Cap)
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation', 
    'GOOGL': 'Alphabet Inc. Class A',
    'GOOG': 'Alphabet Inc. Class C',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corporation',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    
    # Healthcare
    'UNH': 'UnitedHealth Group Inc.',
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc.',
    'ABBV': 'AbbVie Inc.',
    'TMO': 'Thermo Fisher Scientific',
    'ABT': 'Abbott Laboratories',
    'DHR': 'Danaher Corporation',
    
    # Financial Services
    'BRK-B': 'Berkshire Hathaway Inc. Class B',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.',
    'BAC': 'Bank of America Corp',
    'WFC': 'Wells Fargo & Company',
    'GS': 'Goldman Sachs Group Inc.',
    
    # Consumer Discretionary
    'HD': 'Home Depot Inc.',
    'PG': 'Procter & Gamble Co.',
    'NFLX': 'Netflix Inc.',
    'DIS': 'Walt Disney Company',
    'MCD': 'McDonald\'s Corporation',
    'NIKE': 'Nike Inc.',
    'SBUX': 'Starbucks Corporation',
    
    # Consumer Staples
    'KO': 'Coca-Cola Company',
    'PEP': 'PepsiCo Inc.',
    'WMT': 'Walmart Inc.',
    'COST': 'Costco Wholesale Corp',
    
    # Energy
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    
    # Industrial
    'BA': 'Boeing Company',
    'CAT': 'Caterpillar Inc.',
    'GE': 'General Electric Company',
    'LMT': 'Lockheed Martin Corp',
    
    # Communication Services
    'VZ': 'Verizon Communications Inc.',
    'T': 'AT&T Inc.',
    'CRM': 'Salesforce Inc.',
    
    # Utilities & Real Estate
    'NEE': 'NextEra Energy Inc.',
    'AMT': 'American Tower Corp'
}

# Dow Jones Industrial Average (30 stocks)
DOW_30 = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'UNH': 'UnitedHealth Group Inc.',
    'GS': 'Goldman Sachs Group Inc.',
    'HD': 'Home Depot Inc.',
    'CAT': 'Caterpillar Inc.',
    'AMGN': 'Amgen Inc.',
    'MCD': "McDonald's Corporation",
    'CRM': 'Salesforce Inc.',
    'V': 'Visa Inc.',
    'BA': 'Boeing Company',
    'HON': 'Honeywell International Inc.',
    'AXP': 'American Express Company',
    'JPM': 'JPMorgan Chase & Co.',
    'IBM': 'International Business Machines Corp',
    'PG': 'Procter & Gamble Co.',
    'JNJ': 'Johnson & Johnson',
    'CVX': 'Chevron Corporation',
    'DIS': 'Walt Disney Company',
    'MRK': 'Merck & Co. Inc.',
    'WMT': 'Walmart Inc.',
    'NKE': 'Nike Inc.',
    'KO': 'Coca-Cola Company',
    'TRV': 'Travelers Companies Inc.',
    'MMM': '3M Company',
    'DOW': 'Dow Inc.',
    'CSCO': 'Cisco Systems Inc.',
    'VZ': 'Verizon Communications Inc.',
    'INTC': 'Intel Corporation',
    'WBA': 'Walgreens Boots Alliance Inc.'
}

# Technology Focus Portfolio
TECH_PORTFOLIO = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    'NFLX': 'Netflix Inc.',
    'CRM': 'Salesforce Inc.',
    'ORCL': 'Oracle Corporation',
    'ADBE': 'Adobe Inc.',
    'INTC': 'Intel Corporation',
    'CSCO': 'Cisco Systems Inc.',
    'IBM': 'International Business Machines',
    'QCOM': 'Qualcomm Inc.',
    'AVGO': 'Broadcom Inc.',
    'TXN': 'Texas Instruments Inc.',
    'AMAT': 'Applied Materials Inc.',
    'MU': 'Micron Technology Inc.',
    'AMD': 'Advanced Micro Devices Inc.'
}

# Popular ETFs for diversification
POPULAR_ETFS = {
    # Broad Market
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ Trust ETF',
    'IWM': 'iShares Russell 2000 ETF',
    'VTI': 'Vanguard Total Stock Market ETF',
    'VEA': 'Vanguard FTSE Developed Markets ETF',
    'VWO': 'Vanguard FTSE Emerging Markets ETF',
    
    # Sector ETFs
    'XLK': 'Technology Select Sector SPDR Fund',
    'XLF': 'Financial Select Sector SPDR Fund',
    'XLV': 'Health Care Select Sector SPDR Fund',
    'XLE': 'Energy Select Sector SPDR Fund',
    'XLI': 'Industrial Select Sector SPDR Fund',
    'XLP': 'Consumer Staples Select Sector SPDR Fund',
    'XLY': 'Consumer Discretionary Select Sector SPDR Fund',
    'XLU': 'Utilities Select Sector SPDR Fund',
    'XLRE': 'Real Estate Select Sector SPDR Fund',
    'XLB': 'Materials Select Sector SPDR Fund',
    'XLC': 'Communication Services Select Sector SPDR Fund',
    
    # Bond ETFs
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF',
    'LQD': 'iShares iBoxx Investment Grade Corporate Bond ETF',
    'HYG': 'iShares iBoxx High Yield Corporate Bond ETF',
    
    # Commodity ETFs
    'GLD': 'SPDR Gold Trust',
    'SLV': 'iShares Silver Trust',
    'USO': 'United States Oil Fund',
    'UNG': 'United States Natural Gas Fund',
    
    # International
    'EFA': 'iShares MSCI EAFE ETF',
    'EEM': 'iShares MSCI Emerging Markets ETF',
    'FXI': 'iShares China Large-Cap ETF',
    'EWJ': 'iShares MSCI Japan ETF'
}

# Default configuration settings
DEFAULT_CONFIG = {
    'years_back': 10,
    'price_type': 'Close',
    'data_directory': 'data',
    'log_level': 'INFO',
    'save_individual_files': True,
    'save_pickle': True,
    'forward_fill_limit': 5,
    'missing_data_threshold': 0.8
}

# Asset universe configurations
UNIVERSE_CONFIGS = {
    'sp500_top': SP500_TOP_HOLDINGS,
    'dow30': DOW_30,
    'tech_focus': TECH_PORTFOLIO,
    'etfs_only': POPULAR_ETFS,
    'diversified': {**SP500_TOP_HOLDINGS, **POPULAR_ETFS}
}

# Risk-free rate proxies (for Sharpe ratio calculations)
RISK_FREE_PROXIES = {
    'TNX': '10-Year Treasury Note Yield',
    'IRX': '3-Month Treasury Bill Yield',
    'FVX': '5-Year Treasury Note Yield'
} 