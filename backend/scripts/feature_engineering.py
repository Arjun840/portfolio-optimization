#!/usr/bin/env python3
"""
Feature Engineering for Portfolio Optimization ML Models

This script creates comprehensive features for each asset that capture:
- Risk-return characteristics
- Momentum and technical indicators
- Rolling statistics
- Sector and asset type information
- Correlation-based features

Features are prepared for clustering, classification, and optimization models.
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class PortfolioFeatureEngineer:
    """Comprehensive feature engineering for portfolio optimization."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize feature engineer with data directory."""
        self.data_dir = data_dir
        self.prices = None
        self.returns = None
        self.log_returns = None
        self.features = None
        
        # Asset categorization
        self.asset_sectors = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Technology',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            
            # Financial
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 
            'GS': 'Financial', 'V': 'Financial',
            
            # Consumer
            'AMZN': 'Consumer', 'HD': 'Consumer', 'MCD': 'Consumer', 'SBUX': 'Consumer',
            'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            
            # Industrial
            'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial',
            
            # Utilities & Communication
            'VZ': 'Communication', 'T': 'Communication', 'DIS': 'Communication',
            'NEE': 'Utilities', 'AMT': 'REIT',
            
            # Materials
            'LIN': 'Materials', 'NEM': 'Materials',
            
            # ETFs
            'SPY': 'ETF_Broad', 'QQQ': 'ETF_Tech', 'IWM': 'ETF_Small',
            'VTI': 'ETF_Broad', 'EFA': 'ETF_International', 'EEM': 'ETF_Emerging',
            'TLT': 'ETF_Bonds', 'GLD': 'ETF_Commodity', 'VNQ': 'ETF_REIT'
        }
        
        self.asset_types = {
            # Individual stocks
            **{asset: 'Stock' for asset in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA',
                                           'JNJ', 'PFE', 'UNH', 'ABBV', 'JPM', 'BAC', 'WFC',
                                           'GS', 'V', 'AMZN', 'HD', 'MCD', 'SBUX', 'PG', 'KO',
                                           'PEP', 'WMT', 'XOM', 'CVX', 'COP', 'BA', 'CAT', 'GE',
                                           'VZ', 'T', 'DIS', 'NEE', 'AMT', 'LIN', 'NEM']},
            # ETFs
            **{asset: 'ETF' for asset in ['SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'EEM', 'TLT', 'GLD', 'VNQ']}
        }
    
    def load_data(self) -> bool:
        """Load cleaned price and returns data."""
        try:
            logger.info("Loading cleaned data...")
            
            self.prices = pd.read_pickle(os.path.join(self.data_dir, "cleaned_prices.pkl"))
            self.returns = pd.read_pickle(os.path.join(self.data_dir, "cleaned_returns.pkl"))
            self.log_returns = pd.read_pickle(os.path.join(self.data_dir, "log_returns.pkl"))
            
            logger.info(f"Loaded data: {self.prices.shape[0]} days, {self.prices.shape[1]} assets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def calculate_basic_features(self) -> pd.DataFrame:
        """Calculate basic risk-return features for each asset."""
        logger.info("Calculating basic risk-return features...")
        
        features = {}
        
        for asset in self.returns.columns:
            asset_returns = self.returns[asset].dropna()
            asset_prices = self.prices[asset].dropna()
            
            # Basic return statistics
            features[asset] = {
                # Return metrics
                'mean_daily_return': asset_returns.mean(),
                'mean_annual_return': asset_returns.mean() * 252,
                'median_daily_return': asset_returns.median(),
                
                # Volatility metrics
                'daily_volatility': asset_returns.std(),
                'annual_volatility': asset_returns.std() * np.sqrt(252),
                'volatility_of_volatility': asset_returns.rolling(30).std().std(),
                
                # Risk-adjusted returns
                'sharpe_ratio': (asset_returns.mean() * 252) / (asset_returns.std() * np.sqrt(252)),
                'sortino_ratio': self._calculate_sortino_ratio(asset_returns),
                'calmar_ratio': self._calculate_calmar_ratio(asset_returns),
                
                # Distribution characteristics
                'skewness': asset_returns.skew(),
                'kurtosis': asset_returns.kurtosis(),
                'jarque_bera_stat': self._jarque_bera_test(asset_returns),
                
                # Extreme risk measures
                'var_95': asset_returns.quantile(0.05),
                'var_99': asset_returns.quantile(0.01),
                'cvar_95': asset_returns[asset_returns <= asset_returns.quantile(0.05)].mean(),
                'max_drawdown': self._calculate_max_drawdown(asset_returns),
                
                # Price-based features
                'price_range_ratio': asset_prices.max() / asset_prices.min(),
                'current_vs_mean_price': asset_prices.iloc[-1] / asset_prices.mean(),
                'price_stability': 1 / (asset_prices.std() / asset_prices.mean()),  # Inverse CV
            }
        
        return pd.DataFrame(features).T
    
    def calculate_momentum_features(self, periods: List[int] = [5, 10, 21, 63, 126, 252]) -> pd.DataFrame:
        """Calculate momentum and trend features."""
        logger.info("Calculating momentum features...")
        
        momentum_features = {}
        
        for asset in self.returns.columns:
            asset_returns = self.returns[asset].dropna()
            asset_prices = self.prices[asset].dropna()
            
            features = {}
            
            # Momentum over different periods
            for period in periods:
                if len(asset_returns) >= period:
                    # Simple momentum (cumulative return)
                    momentum = (1 + asset_returns.tail(period)).prod() - 1
                    features[f'momentum_{period}d'] = momentum
                    
                    # Volatility-adjusted momentum
                    vol = asset_returns.tail(period).std()
                    features[f'vol_adj_momentum_{period}d'] = momentum / vol if vol > 0 else 0
            
            # Recent performance metrics (key for clustering)
            if len(asset_returns) >= 21:
                features['recent_1month_return'] = (1 + asset_returns.tail(21)).prod() - 1
                features['recent_1month_volatility'] = asset_returns.tail(21).std() * np.sqrt(252)
                features['recent_1month_sharpe'] = (
                    features['recent_1month_return'] * 12 / features['recent_1month_volatility'] 
                    if features['recent_1month_volatility'] > 0 else 0
                )
            
            if len(asset_returns) >= 63:
                # 3-month return (quarter performance)
                features['recent_3month_return'] = (1 + asset_returns.tail(63)).prod() - 1
                features['recent_3month_volatility'] = asset_returns.tail(63).std() * np.sqrt(252)
                features['recent_3month_sharpe'] = (
                    features['recent_3month_return'] * 4 / features['recent_3month_volatility']
                    if features['recent_3month_volatility'] > 0 else 0
                )
                
                # Recent vs historical performance comparison
                historical_return = (1 + asset_returns.head(-63)).mean() * 252 if len(asset_returns) > 126 else 0
                features['recent_vs_historical_return'] = (
                    features['recent_3month_return'] * 4 - historical_return
                    if historical_return != 0 else 0
                )
            
            if len(asset_returns) >= 126:
                # 6-month return (half-year performance) 
                features['recent_6month_return'] = (1 + asset_returns.tail(126)).prod() - 1
                features['recent_6month_volatility'] = asset_returns.tail(126).std() * np.sqrt(252)
            
            # Technical indicators
            if len(asset_prices) >= 252:
                # Moving averages
                ma_short = asset_prices.rolling(20).mean()
                ma_long = asset_prices.rolling(50).mean()
                
                features['ma_ratio_20_50'] = (ma_short.iloc[-1] / ma_long.iloc[-1] - 1) if ma_long.iloc[-1] > 0 else 0
                features['price_vs_ma20'] = (asset_prices.iloc[-1] / ma_short.iloc[-1] - 1) if ma_short.iloc[-1] > 0 else 0
                features['price_vs_ma50'] = (asset_prices.iloc[-1] / ma_long.iloc[-1] - 1) if ma_long.iloc[-1] > 0 else 0
                
                # RSI (simplified)
                features['rsi_14'] = self._calculate_rsi(asset_prices, 14)
                
                # Bollinger Band position
                features['bollinger_position'] = self._calculate_bollinger_position(asset_prices)
            
            # Trend strength
            features['trend_strength'] = self._calculate_trend_strength(asset_returns)
            
            # Recent volatility vs historical
            if len(asset_returns) >= 63:
                recent_vol = asset_returns.tail(21).std()
                historical_vol = asset_returns.tail(252).std()
                features['vol_regime'] = recent_vol / historical_vol if historical_vol > 0 else 1
            
            # Momentum acceleration (change in momentum)
            if len(asset_returns) >= 126:
                recent_momentum = features.get('momentum_63d', 0)
                older_momentum = (1 + asset_returns.tail(126).head(63)).prod() - 1
                features['momentum_acceleration'] = recent_momentum - older_momentum
            
            # Recent performance ranking metrics
            if len(asset_returns) >= 252:
                # Percentile of recent returns in historical context
                all_63d_returns = []
                for i in range(63, len(asset_returns), 21):  # Every 21 days
                    period_return = (1 + asset_returns.iloc[i-63:i]).prod() - 1
                    all_63d_returns.append(period_return)
                
                if all_63d_returns and len(all_63d_returns) > 1:
                    current_3m_return = features.get('recent_3month_return', 0)
                    features['recent_performance_percentile'] = (
                        np.sum(np.array(all_63d_returns) <= current_3m_return) / len(all_63d_returns)
                    )
            
            momentum_features[asset] = features
        
        return pd.DataFrame(momentum_features).T.fillna(0)
    
    def calculate_rolling_features(self, windows: List[int] = [21, 63, 126, 252]) -> pd.DataFrame:
        """Calculate rolling statistics features."""
        logger.info("Calculating rolling statistics features...")
        
        rolling_features = {}
        
        for asset in self.returns.columns:
            asset_returns = self.returns[asset].dropna()
            
            features = {}
            
            for window in windows:
                if len(asset_returns) >= window:
                    rolling_returns = asset_returns.rolling(window)
                    
                    # Rolling statistics
                    features[f'rolling_mean_{window}d'] = rolling_returns.mean().iloc[-1]
                    features[f'rolling_std_{window}d'] = rolling_returns.std().iloc[-1]
                    features[f'rolling_sharpe_{window}d'] = (
                        rolling_returns.mean().iloc[-1] / rolling_returns.std().iloc[-1] 
                        if rolling_returns.std().iloc[-1] > 0 else 0
                    )
                    
                    # Rolling extremes
                    features[f'rolling_min_{window}d'] = rolling_returns.min().iloc[-1]
                    features[f'rolling_max_{window}d'] = rolling_returns.max().iloc[-1]
                    
                    # Rolling percentiles
                    features[f'rolling_q25_{window}d'] = rolling_returns.quantile(0.25).iloc[-1]
                    features[f'rolling_q75_{window}d'] = rolling_returns.quantile(0.75).iloc[-1]
                    
                    # Stability measures
                    rolling_vol = rolling_returns.std()
                    features[f'vol_stability_{window}d'] = 1 / rolling_vol.std() if rolling_vol.std() > 0 else 1
            
            rolling_features[asset] = features
        
        return pd.DataFrame(rolling_features).T.fillna(0)
    
    def calculate_correlation_features(self) -> pd.DataFrame:
        """Calculate correlation-based features."""
        logger.info("Calculating correlation features...")
        
        correlation_matrix = self.returns.corr()
        correlation_features = {}
        
        # Market proxies
        market_proxies = ['SPY', 'QQQ', 'VTI']
        available_market_proxies = [proxy for proxy in market_proxies if proxy in self.returns.columns]
        
        for asset in self.returns.columns:
            features = {}
            
            # Market correlation
            if available_market_proxies:
                market_correlations = [correlation_matrix.loc[asset, proxy] 
                                     for proxy in available_market_proxies 
                                     if proxy != asset]
                if market_correlations:
                    features['market_correlation'] = np.mean(market_correlations)
                    features['max_market_correlation'] = np.max(market_correlations)
            
            # Sector correlation (average correlation with same sector assets)
            asset_sector = self.asset_sectors.get(asset, 'Unknown')
            same_sector_assets = [a for a, sector in self.asset_sectors.items() 
                                if sector == asset_sector and a != asset and a in self.returns.columns]
            
            if same_sector_assets:
                sector_correlations = [correlation_matrix.loc[asset, other] for other in same_sector_assets]
                features['sector_correlation'] = np.mean(sector_correlations)
                features['max_sector_correlation'] = np.max(sector_correlations)
            else:
                features['sector_correlation'] = 0
                features['max_sector_correlation'] = 0
            
            # Overall correlation characteristics
            asset_correlations = correlation_matrix.loc[asset].drop(asset)
            features['mean_correlation'] = asset_correlations.mean()
            features['median_correlation'] = asset_correlations.median()
            features['max_correlation'] = asset_correlations.max()
            features['min_correlation'] = asset_correlations.min()
            features['correlation_dispersion'] = asset_correlations.std()
            
            # Safe haven characteristics (negative correlation with risky assets)
            risky_assets = ['TSLA', 'NVDA', 'META', 'BA']  # High volatility assets
            available_risky_assets = [a for a in risky_assets if a in self.returns.columns and a != asset]
            
            if available_risky_assets:
                risky_correlations = [correlation_matrix.loc[asset, risky] for risky in available_risky_assets]
                features['safe_haven_score'] = -np.mean(risky_correlations)  # Negative correlation = safe haven
            else:
                features['safe_haven_score'] = 0
            
            correlation_features[asset] = features
        
        return pd.DataFrame(correlation_features).T.fillna(0)
    
    def create_sector_dummies(self) -> pd.DataFrame:
        """Create sector and asset type dummy variables."""
        logger.info("Creating sector dummy variables...")
        
        # Get all unique sectors
        all_sectors = set(self.asset_sectors.values())
        all_asset_types = set(self.asset_types.values())
        
        dummy_features = {}
        
        for asset in self.returns.columns:
            features = {}
            
            # Sector dummies
            asset_sector = self.asset_sectors.get(asset, 'Unknown')
            for sector in all_sectors:
                features[f'sector_{sector}'] = 1 if asset_sector == sector else 0
            
            # Asset type dummies
            asset_type = self.asset_types.get(asset, 'Unknown')
            for atype in all_asset_types:
                features[f'type_{atype}'] = 1 if asset_type == atype else 0
            
            # Special categories
            features['is_etf'] = 1 if asset_type == 'ETF' else 0
            features['is_tech'] = 1 if asset_sector == 'Technology' else 0
            features['is_defensive'] = 1 if asset_sector in ['Utilities', 'Consumer', 'Healthcare'] else 0
            features['is_cyclical'] = 1 if asset_sector in ['Technology', 'Financial', 'Industrial'] else 0
            features['is_commodity'] = 1 if 'Commodity' in asset_sector or 'Materials' in asset_sector else 0
            
            dummy_features[asset] = features
        
        return pd.DataFrame(dummy_features).T
    
    def calculate_advanced_features(self) -> pd.DataFrame:
        """Calculate advanced features for ML models."""
        logger.info("Calculating advanced features...")
        
        advanced_features = {}
        
        for asset in self.returns.columns:
            asset_returns = self.returns[asset].dropna()
            asset_prices = self.prices[asset].dropna()
            
            features = {}
            
            # Regime indicators
            features['bull_market_days'] = self._count_bull_market_days(asset_returns)
            features['bear_market_days'] = self._count_bear_market_days(asset_returns)
            features['sideways_market_days'] = len(asset_returns) - features['bull_market_days'] - features['bear_market_days']
            
            # Consistency measures
            features['positive_days_ratio'] = (asset_returns > 0).sum() / len(asset_returns)
            features['large_move_days'] = (np.abs(asset_returns) > 2 * asset_returns.std()).sum() / len(asset_returns)
            
            # Information ratios vs benchmarks
            if 'SPY' in self.returns.columns and asset != 'SPY':
                excess_returns = asset_returns - self.returns['SPY']
                features['information_ratio_spy'] = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
                features['tracking_error_spy'] = excess_returns.std()
                features['beta_spy'] = self._calculate_beta(asset_returns, self.returns['SPY'])
            
            # Liquidity proxies (using volatility and price patterns)
            features['liquidity_proxy'] = 1 / asset_returns.std()  # Lower volatility = higher liquidity proxy
            features['price_impact_proxy'] = np.abs(asset_returns).mean()  # Price impact proxy
            
            # Tail risk measures
            features['tail_ratio'] = asset_returns.quantile(0.95) / np.abs(asset_returns.quantile(0.05))
            features['upside_potential'] = asset_returns[asset_returns > asset_returns.mean()].mean()
            features['downside_deviation'] = np.sqrt(((asset_returns[asset_returns < 0]) ** 2).mean())
            
            advanced_features[asset] = features
        
        return pd.DataFrame(advanced_features).T.fillna(0)
    
    def create_feature_matrix(self) -> pd.DataFrame:
        """Create comprehensive feature matrix for ML models."""
        logger.info("Creating comprehensive feature matrix...")
        
        # Calculate all feature categories
        basic_features = self.calculate_basic_features()
        momentum_features = self.calculate_momentum_features()
        rolling_features = self.calculate_rolling_features()
        correlation_features = self.calculate_correlation_features()
        sector_dummies = self.create_sector_dummies()
        advanced_features = self.calculate_advanced_features()
        
        # Combine all features
        feature_matrices = [
            basic_features,
            momentum_features, 
            rolling_features,
            correlation_features,
            sector_dummies,
            advanced_features
        ]
        
        # Concatenate along columns
        self.features = pd.concat(feature_matrices, axis=1)
        
        # Handle any remaining NaN values
        self.features = self.features.fillna(0)
        
        # Add feature metadata
        self.features.index.name = 'Asset'
        
        logger.info(f"Created feature matrix: {self.features.shape[0]} assets × {self.features.shape[1]} features")
        
        return self.features
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis and selection."""
        if self.features is None:
            raise ValueError("Feature matrix not created yet. Run create_feature_matrix() first.")
        
        feature_columns = list(self.features.columns)
        
        groups = {
            'basic_risk_return': [col for col in feature_columns if any(keyword in col for keyword in 
                                ['mean_', 'annual_', 'volatility', 'sharpe', 'sortino', 'calmar'])],
            
            'distribution': [col for col in feature_columns if any(keyword in col for keyword in 
                           ['skewness', 'kurtosis', 'jarque', 'var_', 'cvar'])],
            
            'momentum': [col for col in feature_columns if any(keyword in col for keyword in 
                        ['momentum_', 'trend_', 'rsi', 'bollinger', 'ma_'])],
            
            'rolling_stats': [col for col in feature_columns if 'rolling_' in col],
            
            'correlation': [col for col in feature_columns if any(keyword in col for keyword in 
                          ['correlation', 'safe_haven'])],
            
            'sector_dummies': [col for col in feature_columns if col.startswith(('sector_', 'type_', 'is_'))],
            
            'advanced': [col for col in feature_columns if any(keyword in col for keyword in 
                        ['regime', 'bull_', 'bear_', 'information_', 'beta_', 'tail_', 'liquidity'])]
        }
        
        return groups
    
    def save_features(self, output_dir: str = "data") -> bool:
        """Save feature matrix and metadata."""
        try:
            if self.features is None:
                logger.error("No features to save. Run create_feature_matrix() first.")
                return False
            
            # Save feature matrix
            feature_file = os.path.join(output_dir, "asset_features.pkl")
            self.features.to_pickle(feature_file)
            
            feature_csv = os.path.join(output_dir, "asset_features.csv")
            self.features.to_csv(feature_csv)
            
            # Save feature groups metadata
            feature_groups = self.get_feature_groups()
            groups_file = os.path.join(output_dir, "feature_groups.json")
            
            import json
            with open(groups_file, 'w') as f:
                json.dump(feature_groups, f, indent=2)
            
            # Save feature summary
            summary_file = os.path.join(output_dir, "feature_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("PORTFOLIO ASSET FEATURES SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Feature Matrix Shape: {self.features.shape}\n")
                f.write(f"Assets: {self.features.shape[0]}\n")
                f.write(f"Features: {self.features.shape[1]}\n\n")
                
                f.write("FEATURE GROUPS:\n")
                f.write("-" * 20 + "\n")
                for group, features in feature_groups.items():
                    f.write(f"{group}: {len(features)} features\n")
                
                f.write("\nFEATURE STATISTICS:\n")
                f.write("-" * 20 + "\n")
                f.write(str(self.features.describe()))
            
            logger.info(f"Features saved to {output_dir}")
            logger.info(f"Files created:")
            logger.info(f"  - {feature_file} (pickle format)")
            logger.info(f"  - {feature_csv} (CSV format)")
            logger.info(f"  - {groups_file} (feature groups)")
            logger.info(f"  - {summary_file} (summary report)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            return False
    
    # Helper methods for calculations
    def _calculate_sortino_ratio(self, returns: pd.Series, target_return: float = 0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - target_return
        downside_deviation = np.sqrt((excess_returns[excess_returns < 0] ** 2).mean())
        return excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        annual_return = returns.mean() * 252
        max_dd = self._calculate_max_drawdown(returns)
        return annual_return / abs(max_dd) if max_dd != 0 else 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        return drawdown.min()
    
    def _jarque_bera_test(self, returns: pd.Series) -> float:
        """Simplified Jarque-Bera test statistic."""
        n = len(returns)
        skew = returns.skew()
        kurt = returns.kurtosis()
        jb_stat = (n / 6) * (skew**2 + (kurt**2) / 4)
        return jb_stat
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate position within Bollinger Bands."""
        if len(prices) < period:
            return 0.5  # Middle position
        
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if current_upper == current_lower:
            return 0.5
        
        position = (current_price - current_lower) / (current_upper - current_lower)
        return np.clip(position, 0, 1)
    
    def _calculate_trend_strength(self, returns: pd.Series, period: int = 21) -> float:
        """Calculate trend strength using linear regression slope."""
        if len(returns) < period:
            return 0
        
        recent_returns = returns.tail(period)
        x = np.arange(len(recent_returns))
        
        # Linear regression slope
        slope = np.polyfit(x, recent_returns.values, 1)[0]
        return slope * 252  # Annualized trend
    
    def _count_bull_market_days(self, returns: pd.Series, threshold: float = 0.001) -> int:
        """Count days in bull market (rising trend)."""
        return (returns > threshold).sum()
    
    def _count_bear_market_days(self, returns: pd.Series, threshold: float = -0.001) -> int:
        """Count days in bear market (falling trend)."""
        return (returns < threshold).sum()
    
    def _calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta vs market."""
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0
        
        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data.iloc[:, 1].var()
        
        return covariance / market_variance if market_variance > 0 else 1.0


def main():
    """Main function to run feature engineering."""
    print("Starting Portfolio Feature Engineering...")
    print("=" * 60)
    
    # Initialize feature engineer
    engineer = PortfolioFeatureEngineer()
    
    # Load data
    if not engineer.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Create features
    print("\n1. Creating comprehensive feature matrix...")
    features = engineer.create_feature_matrix()
    
    # Display feature summary
    print(f"\n2. Feature Matrix Created:")
    print(f"   • Shape: {features.shape}")
    print(f"   • Assets: {features.shape[0]}")
    print(f"   • Features per asset: {features.shape[1]}")
    
    # Show feature groups
    feature_groups = engineer.get_feature_groups()
    print(f"\n3. Feature Groups:")
    for group, group_features in feature_groups.items():
        print(f"   • {group}: {len(group_features)} features")
    
    # Save features
    print(f"\n4. Saving features...")
    engineer.save_features()
    
    # Display sample features
    print(f"\n5. Sample Features (Top 5 Assets):")
    print("   Basic Risk-Return Features:")
    basic_cols = ['mean_annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
    available_basic_cols = [col for col in basic_cols if col in features.columns]
    if available_basic_cols:
        print(features[available_basic_cols].head().round(4))
    
    print("\n" + "=" * 60)
    print("Feature Engineering Complete!")
    print("Files created:")
    print("• data/asset_features.pkl - Feature matrix (for ML models)")
    print("• data/asset_features.csv - Feature matrix (human readable)")
    print("• data/feature_groups.json - Feature group definitions")
    print("• data/feature_summary.txt - Detailed summary")
    print("\nReady for ML model training! 🚀")


if __name__ == "__main__":
    main() 