# Portfolio Feature Engineering Guide

## Overview
This guide explains the comprehensive feature engineering system built for ML-driven portfolio optimization. We've created **118 features** across **45 assets** that capture various aspects of asset behavior for clustering, prediction, and optimization models.

## Feature Categories

### 1. Basic Risk-Return Features (18 features)
**Purpose**: Fundamental asset characteristics for risk-return profiling

**Key Features**:
- `mean_annual_return` - Average annual return
- `annual_volatility` - Annualized volatility (risk measure)
- `sharpe_ratio` - Risk-adjusted return metric
- `sortino_ratio` - Downside risk-adjusted return
- `max_drawdown` - Maximum peak-to-trough decline
- `var_95`, `cvar_95` - Value at Risk measures

**Use Cases**:
- Asset ranking and selection
- Risk budget allocation
- Performance attribution

### 2. Distribution Features (6 features)
**Purpose**: Capture return distribution characteristics

**Key Features**:
- `skewness` - Asymmetry of return distribution
- `kurtosis` - Tail thickness (fat tail risk)
- `jarque_bera_stat` - Normality test statistic

**Use Cases**:
- Tail risk modeling
- Options pricing adjustments
- Stress testing scenarios

### 3. Momentum Features (17 features)
**Purpose**: Trend and momentum indicators for timing models

**Key Features**:
- `momentum_21d`, `momentum_63d`, `momentum_252d` - Multi-timeframe momentum
- `vol_adj_momentum_*` - Volatility-adjusted momentum
- `rsi_14` - Relative Strength Index
- `bollinger_position` - Position within Bollinger Bands
- `trend_strength` - Linear trend coefficient

**Use Cases**:
- Tactical asset allocation
- Entry/exit timing
- Regime detection

### 4. Rolling Statistics Features (28 features)
**Purpose**: Multi-timeframe risk and return characteristics

**Key Features**:
- `rolling_mean_*`, `rolling_std_*` - Rolling averages and volatility
- `rolling_sharpe_*` - Rolling risk-adjusted returns
- `vol_stability_*` - Volatility clustering measures

**Timeframes**: 21, 63, 126, 252 days

**Use Cases**:
- Dynamic risk management
- Regime change detection
- Adaptive portfolio rebalancing

### 5. Correlation Features (10 features)
**Purpose**: Diversification and market relationship measures

**Key Features**:
- `market_correlation` - Correlation with market indices
- `sector_correlation` - Correlation within sectors
- `safe_haven_score` - Negative correlation with risky assets
- `mean_correlation` - Average correlation with all assets

**Use Cases**:
- Diversification optimization
- Risk factor analysis
- Safe haven identification

### 6. Sector Dummy Variables (26 features)
**Purpose**: Categorical encoding for sector and asset type

**Categories**:
- **Sectors**: Technology, Healthcare, Financial, Consumer, Energy, Industrial, etc.
- **Asset Types**: Stock, ETF
- **Special Flags**: `is_tech`, `is_defensive`, `is_cyclical`

**Use Cases**:
- Sector rotation strategies
- Industry-neutral portfolios
- Constraint-based optimization

### 7. Advanced Features (7 features)
**Purpose**: Sophisticated ML-ready characteristics

**Key Features**:
- `information_ratio_spy` - Excess return vs benchmark
- `beta_spy` - Market sensitivity
- `tail_ratio` - Upside vs downside extremes
- `liquidity_proxy` - Trading liquidity estimate

**Use Cases**:
- Alpha generation models
- Market neutral strategies
- Risk factor decomposition

## Clustering Results

### Discovered Asset Clusters

#### Cluster 0: Blue Chip Stocks (19 assets)
**Assets**: AAPL, JNJ, PFE, UNH, ABBV, AMZN, HD, MCD, SBUX, PG, KO, PEP, WMT, XOM, CVX, COP, VZ, NEE, LIN

**Characteristics**:
- Return: 14.5% ± 6.0%
- Volatility: 24.0% ± 5.5%
- Sharpe: 0.61 ± 0.21

**Portfolio Role**: Quality holdings, dividend income, stability

#### Cluster 1: High-Growth Tech (5 assets)
**Assets**: MSFT, GOOGL, NVDA, META, TSLA

**Characteristics**:
- Return: 39.0% ± 18.5%
- Volatility: 39.6% ± 13.8%
- Sharpe: 0.98 ± 0.28

**Portfolio Role**: Growth driver, high risk/reward

#### Cluster 2: Traditional Value (12 assets)
**Assets**: JPM, BAC, WFC, GS, V, BA, CAT, GE, T, DIS, AMT, NEM

**Characteristics**:
- Return: 15.4% ± 5.1%
- Volatility: 29.6% ± 5.0%
- Sharpe: 0.53 ± 0.20

**Portfolio Role**: Value exposure, cyclical sectors

#### Cluster 3: Defensive Assets (6 assets)
**Assets**: IWM, EFA, EEM, TLT, GLD, VNQ

**Characteristics**:
- Return: 8.0% ± 3.8%
- Volatility: 18.0% ± 3.2%
- Sharpe: 0.44 ± 0.24

**Portfolio Role**: Diversification, downside protection

#### Cluster 4: Market ETFs (3 assets)
**Assets**: SPY, QQQ, VTI

**Characteristics**:
- Return: 16.0% ± 3.2%
- Volatility: 19.0% ± 2.6%
- Sharpe: 0.84 ± 0.05

**Portfolio Role**: Core holdings, market exposure

## How to Use Features for ML Models

### 1. Asset Selection Models
```python
# Use risk-return and correlation features
features_for_selection = [
    'sharpe_ratio', 'max_drawdown', 'mean_correlation',
    'market_correlation', 'annual_volatility'
]
```

### 2. Return Prediction Models
```python
# Use momentum and rolling statistics
features_for_prediction = [
    'momentum_21d', 'momentum_63d', 'rsi_14',
    'rolling_mean_21d', 'vol_adj_momentum_21d',
    'trend_strength'
]
```

### 3. Risk Models
```python
# Use volatility and tail risk features
features_for_risk = [
    'annual_volatility', 'var_95', 'cvar_95',
    'tail_ratio', 'downside_deviation',
    'volatility_of_volatility'
]
```

### 4. Clustering & Portfolio Construction
```python
# Use comprehensive feature set
features_for_clustering = [
    # Risk-return
    'sharpe_ratio', 'annual_volatility', 'mean_annual_return',
    # Momentum
    'momentum_21d', 'momentum_63d',
    # Correlation
    'market_correlation', 'sector_correlation',
    # Sector dummies
    'sector_Technology', 'sector_Healthcare', 'is_etf'
]
```

## Portfolio Allocation Framework

Based on clustering results, here's a strategic allocation framework:

### Core Holdings (25-30%)
- **Assets**: SPY, QQQ, VTI (Market ETFs)
- **Rationale**: High Sharpe ratio, moderate volatility
- **Features**: High `market_correlation`, low `vol_stability`

### Defensive Holdings (20-25%)
- **Assets**: TLT, GLD, EFA, EEM (Defensive cluster)
- **Rationale**: Low volatility, diversification benefits
- **Features**: Low `annual_volatility`, negative `safe_haven_score`

### Growth Holdings (15-20%)
- **Assets**: MSFT, GOOGL, NVDA (High-growth tech)
- **Rationale**: Excellent risk-adjusted performance
- **Features**: High `sharpe_ratio`, positive `momentum_*`

### Value Holdings (10-15%)
- **Assets**: JPM, V, BA (Traditional value)
- **Rationale**: Sector diversification, cyclical exposure
- **Features**: Moderate `sharpe_ratio`, sector diversity

### Quality Holdings (10-15%)
- **Assets**: AAPL, JNJ, PG (Blue chip stocks)
- **Rationale**: Stability, dividend income
- **Features**: Low `max_drawdown`, high `sector_correlation`

## Next Steps for ML Implementation

### 1. Portfolio Optimization
- **Mean-Variance**: Use `mean_annual_return` and covariance matrix
- **Risk Parity**: Use `annual_volatility` and correlation features
- **Black-Litterman**: Incorporate momentum and sector features for views

### 2. Machine Learning Models
- **Return Prediction**: XGBoost/Random Forest using momentum features
- **Risk Prediction**: LSTM using rolling statistics
- **Regime Detection**: Clustering using correlation and volatility features

### 3. Dynamic Strategies
- **Rebalancing**: Use `vol_regime` and `trend_strength` for timing
- **Risk Management**: Monitor `var_95` and `tail_ratio` for risk control
- **Alpha Generation**: Combine multiple feature categories in ensemble models

## Files Generated

### Data Files
- `data/asset_features.pkl` - Feature matrix (optimized for ML)
- `data/asset_features.csv` - Human-readable features
- `data/clustered_assets.csv` - Assets with cluster assignments

### Metadata Files
- `data/feature_groups.json` - Feature categorization
- `data/cluster_analysis.json` - Detailed cluster statistics
- `data/feature_summary.txt` - Statistical summary

### Visualizations
- `data/asset_clusters_pca.png` - PCA cluster visualization

## Usage Examples

### Load Features for ML
```python
import pandas as pd

# Load feature matrix
features = pd.read_pickle('data/asset_features.pkl')

# Load specific feature groups
with open('data/feature_groups.json', 'r') as f:
    feature_groups = json.load(f)

# Select features for your model
momentum_features = features[feature_groups['momentum']]
risk_features = features[feature_groups['basic_risk_return']]
```

### Create Custom Feature Combinations
```python
# Portfolio optimization features
optimization_features = features[
    feature_groups['basic_risk_return'] + 
    feature_groups['correlation'] + 
    feature_groups['sector_dummies']
]

# Prediction model features
prediction_features = features[
    feature_groups['momentum'] + 
    feature_groups['rolling_stats'][:10]  # Top 10 rolling features
]
```

Your portfolio optimization system now has a comprehensive foundation for sophisticated ML models! 🚀 