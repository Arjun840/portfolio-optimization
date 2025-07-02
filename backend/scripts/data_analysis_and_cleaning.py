#!/usr/bin/env python3
"""
Data Analysis and Cleaning Script for Portfolio Optimization

This script performs comprehensive data analysis and cleaning including:
- Data quality assessment and missing value handling
- Daily and log returns calculation
- Statistical analysis (mean, volatility, Sharpe ratios)
- Correlation analysis
- Data visualization (price histories, distributions, correlations)
- Data normalization and preprocessing for ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class PortfolioDataAnalyzer:
    """Comprehensive data analysis and cleaning for portfolio optimization."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the analyzer with data directory."""
        self.data_dir = data_dir
        self.prices = None
        self.returns = None
        self.log_returns = None
        self.cleaned_prices = None
        self.cleaned_returns = None
        self.analysis_results = {}
        
    def load_data(self) -> bool:
        """Load price and returns data from storage."""
        try:
            logger.info("Loading data...")
            
            # Try pickle files first (faster)
            price_file = os.path.join(self.data_dir, "price_matrix_latest.pkl")
            returns_file = os.path.join(self.data_dir, "returns_matrix_latest.pkl")
            
            if os.path.exists(price_file) and os.path.exists(returns_file):
                self.prices = pd.read_pickle(price_file)
                self.returns = pd.read_pickle(returns_file)
            else:
                # Fallback to CSV
                price_file = os.path.join(self.data_dir, "price_matrix_latest.csv")
                returns_file = os.path.join(self.data_dir, "returns_matrix_latest.csv")
                
                self.prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
                self.returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            
            logger.info(f"Loaded price data: {self.prices.shape}")
            logger.info(f"Loaded returns data: {self.returns.shape}")
            logger.info(f"Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def assess_data_quality(self) -> Dict:
        """Comprehensive data quality assessment."""
        logger.info("Assessing data quality...")
        
        quality_report = {
            'price_data': self._assess_single_dataset(self.prices, 'prices'),
            'returns_data': self._assess_single_dataset(self.returns, 'returns'),
            'data_alignment': self._check_data_alignment(),
            'outliers': self._detect_outliers(),
            'trading_days': self._check_trading_calendar()
        }
        
        self.analysis_results['quality_report'] = quality_report
        return quality_report
    
    def _assess_single_dataset(self, data: pd.DataFrame, data_type: str) -> Dict:
        """Assess quality of a single dataset."""
        assessment = {
            'shape': data.shape,
            'missing_values': {
                'total': data.isnull().sum().sum(),
                'by_asset': data.isnull().sum().to_dict(),
                'by_date': data.isnull().sum(axis=1).sum()
            },
            'date_range': {
                'start': data.index[0],
                'end': data.index[-1],
                'total_days': len(data),
                'business_days': len(pd.bdate_range(data.index[0], data.index[-1]))
            },
            'data_types': data.dtypes.to_dict(),
            'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        # Additional checks for price data
        if data_type == 'prices':
            assessment['price_ranges'] = {
                asset: {
                    'min': float(data[asset].min()),
                    'max': float(data[asset].max()),
                    'mean': float(data[asset].mean()),
                    'ratio': float(data[asset].max() / data[asset].min())
                }
                for asset in data.columns
            }
        
        # Additional checks for returns data
        elif data_type == 'returns':
            assessment['return_stats'] = {
                asset: {
                    'min': float(data[asset].min()),
                    'max': float(data[asset].max()),
                    'mean': float(data[asset].mean()),
                    'std': float(data[asset].std()),
                    'extreme_days': len(data[data[asset].abs() > 0.2])  # >20% daily moves
                }
                for asset in data.columns
            }
        
        return assessment
    
    def _check_data_alignment(self) -> Dict:
        """Check alignment between price and returns data."""
        return {
            'date_alignment': self.prices.index[1:].equals(self.returns.index),
            'column_alignment': set(self.prices.columns) == set(self.returns.columns),
            'shape_consistency': len(self.returns) == len(self.prices) - 1
        }
    
    def _detect_outliers(self) -> Dict:
        """Detect outliers in returns data."""
        outliers = {}
        
        for asset in self.returns.columns:
            returns_series = self.returns[asset].dropna()
            
            # Statistical outliers (>3 standard deviations)
            mean_return = returns_series.mean()
            std_return = returns_series.std()
            statistical_outliers = returns_series[
                np.abs(returns_series - mean_return) > 3 * std_return
            ]
            
            # Extreme movements (>15% daily)
            extreme_moves = returns_series[np.abs(returns_series) > 0.15]
            
            outliers[asset] = {
                'statistical_outliers': len(statistical_outliers),
                'extreme_moves': len(extreme_moves),
                'outlier_dates': statistical_outliers.index.tolist()[:5],  # Limit to first 5
                'extreme_dates': extreme_moves.index.tolist()[:5]  # Limit to first 5
            }
        
        return outliers
    
    def _check_trading_calendar(self) -> Dict:
        """Check for missing trading days."""
        expected_business_days = pd.bdate_range(
            self.prices.index[0], 
            self.prices.index[-1]
        )
        
        missing_days = expected_business_days.difference(self.prices.index)
        
        return {
            'expected_days': len(expected_business_days),
            'actual_days': len(self.prices),
            'missing_days': len(missing_days),
            'missing_dates': missing_days.tolist()[:10] if len(missing_days) > 0 else []  # Limit output
        }
    
    def clean_data(self, method: str = 'forward_fill', 
                   max_missing_threshold: float = 0.05) -> bool:
        """Clean the data by handling missing values and outliers."""
        logger.info(f"Cleaning data using method: {method}")
        
        try:
            # Start with copies
            self.cleaned_prices = self.prices.copy()
            self.cleaned_returns = self.returns.copy()
            
            # Handle missing values in prices
            if method == 'forward_fill':
                self.cleaned_prices = self.cleaned_prices.fillna(method='ffill', limit=5)
                self.cleaned_prices = self.cleaned_prices.fillna(method='bfill', limit=2)
            elif method == 'interpolate':
                self.cleaned_prices = self.cleaned_prices.interpolate(method='linear', limit=3)
            elif method == 'drop':
                # Drop assets with too many missing values
                missing_pct = self.cleaned_prices.isnull().sum() / len(self.cleaned_prices)
                assets_to_keep = missing_pct[missing_pct <= max_missing_threshold].index
                self.cleaned_prices = self.cleaned_prices[assets_to_keep]
                
                # Drop dates with too many missing values
                missing_pct_dates = self.cleaned_prices.isnull().sum(axis=1) / len(self.cleaned_prices.columns)
                dates_to_keep = missing_pct_dates[missing_pct_dates <= max_missing_threshold].index
                self.cleaned_prices = self.cleaned_prices.loc[dates_to_keep]
            
            # Recalculate returns from cleaned prices
            self.cleaned_returns = self.cleaned_prices.pct_change().dropna()
            
            # Calculate log returns for stability
            self.log_returns = np.log(self.cleaned_prices / self.cleaned_prices.shift(1)).dropna()
            
            # Handle extreme outliers in returns (cap at 5 standard deviations)
            for asset in self.cleaned_returns.columns:
                returns_series = self.cleaned_returns[asset]
                mean_ret = returns_series.mean()
                std_ret = returns_series.std()
                
                # Cap extreme values
                upper_bound = mean_ret + 5 * std_ret
                lower_bound = mean_ret - 5 * std_ret
                
                self.cleaned_returns[asset] = np.clip(
                    self.cleaned_returns[asset], 
                    lower_bound, 
                    upper_bound
                )
            
            logger.info(f"Cleaned price data shape: {self.cleaned_prices.shape}")
            logger.info(f"Cleaned returns data shape: {self.cleaned_returns.shape}")
            logger.info(f"Log returns data shape: {self.log_returns.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            return False
    
    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistical measures."""
        logger.info("Calculating portfolio statistics...")
        
        if self.cleaned_returns is None:
            logger.warning("No cleaned returns data available. Using raw returns.")
            returns_data = self.returns
            log_returns_data = np.log(self.prices / self.prices.shift(1)).dropna()
        else:
            returns_data = self.cleaned_returns
            log_returns_data = self.log_returns
        
        stats = {}
        
        for asset in returns_data.columns:
            asset_returns = returns_data[asset].dropna()
            asset_log_returns = log_returns_data[asset].dropna()
            
            # Basic statistics
            daily_return = asset_returns.mean()
            daily_volatility = asset_returns.std()
            
            # Annualized metrics (assuming 252 trading days)
            annual_return = daily_return * 252
            annual_volatility = daily_volatility * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Risk metrics
            var_95 = asset_returns.quantile(0.05)  # 5% VaR
            var_99 = asset_returns.quantile(0.01)  # 1% VaR
            
            # Skewness and Kurtosis
            skewness = asset_returns.skew()
            kurtosis = asset_returns.kurtosis()
            
            # Maximum drawdown
            prices_series = self.cleaned_prices[asset] if self.cleaned_prices is not None else self.prices[asset]
            cumulative = (1 + asset_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative / rolling_max) - 1
            max_drawdown = drawdown.min()
            
            stats[asset] = {
                'daily_return': daily_return,
                'daily_volatility': daily_volatility,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'max_drawdown': max_drawdown,
                'total_observations': len(asset_returns)
            }
        
        self.analysis_results['statistics'] = stats
        return stats
    
    def calculate_correlation_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate correlation matrices for returns and log returns."""
        logger.info("Calculating correlation matrices...")
        
        returns_data = self.cleaned_returns if self.cleaned_returns is not None else self.returns
        log_returns_data = self.log_returns if self.log_returns is not None else np.log(self.prices / self.prices.shift(1)).dropna()
        
        returns_corr = returns_data.corr()
        log_returns_corr = log_returns_data.corr()
        
        self.analysis_results['correlations'] = {
            'returns': returns_corr,
            'log_returns': log_returns_corr
        }
        
        return returns_corr, log_returns_corr
    
    def create_visualizations(self, output_dir: str = "analysis_plots") -> bool:
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Price histories
            self._plot_price_histories(output_dir)
            
            # 2. Returns distributions
            self._plot_returns_distributions(output_dir)
            
            # 3. Correlation heatmaps
            self._plot_correlation_heatmaps(output_dir)
            
            # 4. Risk-return scatter
            self._plot_risk_return_scatter(output_dir)
            
            # 5. Rolling statistics
            self._plot_rolling_statistics(output_dir)
            
            # 6. Drawdown analysis
            self._plot_drawdown_analysis(output_dir)
            
            logger.info(f"Visualizations saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            return False
    
    def _plot_price_histories(self, output_dir: str):
        """Plot price histories for selected assets."""
        prices_data = self.cleaned_prices if self.cleaned_prices is not None else self.prices
        
        # Select a few representative assets
        selected_assets = ['AAPL', 'SPY', 'TSLA', 'GLD', 'TLT']
        available_assets = [asset for asset in selected_assets if asset in prices_data.columns]
        
        if not available_assets:
            available_assets = prices_data.columns[:5].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, asset in enumerate(available_assets[:5]):
            # Normalize to start at 100
            normalized_prices = 100 * prices_data[asset] / prices_data[asset].iloc[0]
            
            axes[i].plot(normalized_prices.index, normalized_prices, linewidth=1.5)
            axes[i].set_title(f'{asset} - Normalized Price History', fontweight='bold')
            axes[i].set_ylabel('Normalized Price (Base=100)')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Plot all assets together (normalized)
        for asset in available_assets:
            normalized_prices = 100 * prices_data[asset] / prices_data[asset].iloc[0]
            axes[5].plot(normalized_prices.index, normalized_prices, 
                        linewidth=1, label=asset, alpha=0.8)
        
        axes[5].set_title('All Selected Assets - Normalized Comparison', fontweight='bold')
        axes[5].set_ylabel('Normalized Price (Base=100)')
        axes[5].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'price_histories.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_returns_distributions(self, output_dir: str):
        """Plot returns distributions."""
        returns_data = self.cleaned_returns if self.cleaned_returns is not None else self.returns
        
        selected_assets = ['AAPL', 'SPY', 'TSLA', 'GLD', 'TLT']
        available_assets = [asset for asset in selected_assets if asset in returns_data.columns]
        
        if not available_assets:
            available_assets = returns_data.columns[:5].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, asset in enumerate(available_assets[:5]):
            asset_returns = returns_data[asset].dropna()
            
            # Histogram with normal distribution overlay
            axes[i].hist(asset_returns, bins=50, density=True, alpha=0.7, 
                        color='skyblue', edgecolor='black')
            
            # Normal distribution overlay
            mu, sigma = asset_returns.mean(), asset_returns.std()
            x = np.linspace(asset_returns.min(), asset_returns.max(), 100)
            normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            axes[i].plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
            
            axes[i].set_title(f'{asset} - Daily Returns Distribution', fontweight='bold')
            axes[i].set_xlabel('Daily Returns')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Q-Q plot for normality test - simplified version to avoid scipy dependency
        if available_assets:
            asset = available_assets[0]
            asset_returns = returns_data[asset].dropna()
            sorted_returns = np.sort(asset_returns)
            n = len(sorted_returns)
            theoretical_quantiles = np.linspace(0.01, 0.99, n)
            
            axes[5].scatter(theoretical_quantiles, sorted_returns, alpha=0.6)
            axes[5].set_title(f'{asset} - Quantile Plot', fontweight='bold')
            axes[5].set_xlabel('Theoretical Quantiles')
            axes[5].set_ylabel('Sample Quantiles')
            axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'returns_distributions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmaps(self, output_dir: str):
        """Plot correlation heatmaps."""
        returns_corr, log_returns_corr = self.calculate_correlation_matrix()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Returns correlation
        mask1 = np.triu(np.ones_like(returns_corr))
        sns.heatmap(returns_corr, mask=mask1, annot=True, cmap='RdYlBu_r', 
                    center=0, fmt='.2f', ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('Daily Returns Correlation Matrix', fontweight='bold', pad=20)
        
        # Log returns correlation
        mask2 = np.triu(np.ones_like(log_returns_corr))
        sns.heatmap(log_returns_corr, mask=mask2, annot=True, cmap='RdYlBu_r',
                    center=0, fmt='.2f', ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('Log Returns Correlation Matrix', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmaps.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_return_scatter(self, output_dir: str):
        """Plot risk-return scatter plot."""
        if 'statistics' not in self.analysis_results:
            self.calculate_statistics()
        
        stats = self.analysis_results['statistics']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Risk-Return scatter
        volatilities = [stats[asset]['annual_volatility'] for asset in stats.keys()]
        returns = [stats[asset]['annual_return'] for asset in stats.keys()]
        sharpe_ratios = [stats[asset]['sharpe_ratio'] for asset in stats.keys()]
        
        scatter = ax1.scatter(volatilities, returns, c=sharpe_ratios, 
                             cmap='RdYlGn', s=100, alpha=0.7)
        
        for i, asset in enumerate(stats.keys()):
            ax1.annotate(asset, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Annual Volatility')
        ax1.set_ylabel('Annual Return')
        ax1.set_title('Risk-Return Profile (Color = Sharpe Ratio)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Sharpe Ratio')
        
        # Sharpe ratio bar chart
        assets = list(stats.keys())
        sharpe_values = [stats[asset]['sharpe_ratio'] for asset in assets]
        
        colors = ['green' if sr > 0 else 'red' for sr in sharpe_values]
        bars = ax2.bar(range(len(assets)), sharpe_values, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(assets)))
        ax2.set_xticklabels(assets, rotation=45, ha='right')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratios by Asset', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_return_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rolling_statistics(self, output_dir: str):
        """Plot rolling statistics."""
        returns_data = self.cleaned_returns if self.cleaned_returns is not None else self.returns
        
        # Select representative assets
        selected_assets = ['AAPL', 'SPY', 'TSLA']
        available_assets = [asset for asset in selected_assets if asset in returns_data.columns]
        
        if not available_assets:
            available_assets = returns_data.columns[:3].tolist()
        
        fig, axes = plt.subplots(len(available_assets), 2, figsize=(16, 4 * len(available_assets)))
        
        if len(available_assets) == 1:
            axes = axes.reshape(1, -1)
        
        for i, asset in enumerate(available_assets):
            asset_returns = returns_data[asset].dropna()
            
            # Rolling mean (30-day)
            rolling_mean = asset_returns.rolling(30).mean() * 252  # Annualized
            axes[i, 0].plot(rolling_mean.index, rolling_mean, linewidth=1.5)
            axes[i, 0].set_title(f'{asset} - 30-Day Rolling Annual Return', fontweight='bold')
            axes[i, 0].set_ylabel('Annualized Return')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].tick_params(axis='x', rotation=45)
            
            # Rolling volatility (30-day)
            rolling_vol = asset_returns.rolling(30).std() * np.sqrt(252)  # Annualized
            axes[i, 1].plot(rolling_vol.index, rolling_vol, linewidth=1.5, color='orange')
            axes[i, 1].set_title(f'{asset} - 30-Day Rolling Annual Volatility', fontweight='bold')
            axes[i, 1].set_ylabel('Annualized Volatility')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rolling_statistics.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown_analysis(self, output_dir: str):
        """Plot drawdown analysis."""
        returns_data = self.cleaned_returns if self.cleaned_returns is not None else self.returns
        
        selected_assets = ['AAPL', 'SPY', 'TSLA']
        available_assets = [asset for asset in selected_assets if asset in returns_data.columns]
        
        if not available_assets:
            available_assets = returns_data.columns[:3].tolist()
        
        fig, axes = plt.subplots(len(available_assets), 1, figsize=(14, 4 * len(available_assets)))
        
        if len(available_assets) == 1:
            axes = [axes]
        
        for i, asset in enumerate(available_assets):
            asset_returns = returns_data[asset].dropna()
            
            # Calculate cumulative returns and drawdown
            cumulative = (1 + asset_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative / rolling_max) - 1
            
            # Plot cumulative returns
            ax1 = axes[i]
            ax1.plot(cumulative.index, cumulative, label='Cumulative Return', linewidth=1.5)
            ax1.plot(rolling_max.index, rolling_max, label='Peak', 
                    linewidth=1, linestyle='--', alpha=0.7)
            ax1.set_ylabel('Cumulative Return')
            ax1.set_title(f'{asset} - Cumulative Returns and Drawdowns', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot drawdowns on secondary axis
            ax2 = ax1.twinx()
            ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
            ax2.set_ylabel('Drawdown', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax1.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drawdown_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str = "data_analysis_report.txt") -> bool:
        """Generate a comprehensive text report."""
        try:
            with open(output_file, 'w') as f:
                f.write("PORTFOLIO DATA ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Data Quality Section
                if 'quality_report' in self.analysis_results:
                    f.write("DATA QUALITY ASSESSMENT\n")
                    f.write("-" * 30 + "\n")
                    quality = self.analysis_results['quality_report']
                    
                    f.write(f"Price Data Shape: {quality['price_data']['shape']}\n")
                    f.write(f"Returns Data Shape: {quality['returns_data']['shape']}\n")
                    f.write(f"Missing Values (Prices): {quality['price_data']['missing_values']['total']}\n")
                    f.write(f"Missing Values (Returns): {quality['returns_data']['missing_values']['total']}\n")
                    f.write(f"Date Range: {quality['price_data']['date_range']['start']} to {quality['price_data']['date_range']['end']}\n")
                    f.write(f"Trading Days: {quality['trading_days']['actual_days']} / {quality['trading_days']['expected_days']} expected\n\n")
                
                # Statistics Section
                if 'statistics' in self.analysis_results:
                    f.write("ASSET STATISTICS\n")
                    f.write("-" * 20 + "\n")
                    stats = self.analysis_results['statistics']
                    
                    f.write(f"{'Asset':<8} {'Ann.Ret':<8} {'Ann.Vol':<8} {'Sharpe':<8} {'MaxDD':<8}\n")
                    f.write("-" * 50 + "\n")
                    
                    for asset, stat in stats.items():
                        f.write(f"{asset:<8} {stat['annual_return']:<8.3f} {stat['annual_volatility']:<8.3f} "
                               f"{stat['sharpe_ratio']:<8.3f} {stat['max_drawdown']:<8.3f}\n")
                    
                    f.write("\n")
                
                # Top performers
                if 'statistics' in self.analysis_results:
                    f.write("TOP PERFORMERS\n")
                    f.write("-" * 15 + "\n")
                    stats = self.analysis_results['statistics']
                    
                    # Best Sharpe ratios
                    best_sharpe = sorted(stats.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)[:5]
                    f.write("Best Sharpe Ratios:\n")
                    for asset, stat in best_sharpe:
                        f.write(f"  {asset}: {stat['sharpe_ratio']:.3f}\n")
                    
                    # Highest returns
                    best_returns = sorted(stats.items(), key=lambda x: x[1]['annual_return'], reverse=True)[:5]
                    f.write("\nHighest Annual Returns:\n")
                    for asset, stat in best_returns:
                        f.write(f"  {asset}: {stat['annual_return']:.3f}\n")
                    
                    # Lowest volatility
                    lowest_vol = sorted(stats.items(), key=lambda x: x[1]['annual_volatility'])[:5]
                    f.write("\nLowest Volatility:\n")
                    for asset, stat in lowest_vol:
                        f.write(f"  {asset}: {stat['annual_volatility']:.3f}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("Report generation completed successfully.\n")
            
            logger.info(f"Report saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return False
    
    def save_cleaned_data(self, output_dir: str = "data") -> bool:
        """Save cleaned data to files."""
        try:
            if self.cleaned_prices is None or self.cleaned_returns is None:
                logger.warning("No cleaned data to save")
                return False
            
            # Save cleaned data
            self.cleaned_prices.to_csv(os.path.join(output_dir, "cleaned_prices.csv"))
            self.cleaned_returns.to_csv(os.path.join(output_dir, "cleaned_returns.csv"))
            self.log_returns.to_csv(os.path.join(output_dir, "log_returns.csv"))
            
            # Save as pickle for faster loading
            self.cleaned_prices.to_pickle(os.path.join(output_dir, "cleaned_prices.pkl"))
            self.cleaned_returns.to_pickle(os.path.join(output_dir, "cleaned_returns.pkl"))
            self.log_returns.to_pickle(os.path.join(output_dir, "log_returns.pkl"))
            
            logger.info(f"Cleaned data saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cleaned data: {e}")
            return False


def main():
    """Main function to run the complete analysis."""
    print("Starting Portfolio Data Analysis and Cleaning...")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PortfolioDataAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Assess data quality
    print("\n1. Assessing data quality...")
    quality_report = analyzer.assess_data_quality()
    
    # Clean data
    print("\n2. Cleaning data...")
    analyzer.clean_data(method='forward_fill')
    
    # Calculate statistics
    print("\n3. Calculating statistics...")
    stats = analyzer.calculate_statistics()
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    analyzer.create_visualizations()
    
    # Generate report
    print("\n5. Generating report...")
    analyzer.generate_report()
    
    # Save cleaned data
    print("\n6. Saving cleaned data...")
    analyzer.save_cleaned_data()
    
    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("Check the following outputs:")
    print("- analysis_plots/ - Visualization files")
    print("- data_analysis_report.txt - Comprehensive report")
    print("- data/cleaned_*.csv - Cleaned datasets")


if __name__ == "__main__":
    main() 