import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from fastapi import HTTPException

# Import will be handled locally

logger = logging.getLogger(__name__)

def load_stock_data() -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Load and prepare stock data with cluster information"""
    try:
        # Load portfolio data using CSV files for compatibility
        prices = pd.read_csv('data/cleaned_prices.csv', index_col=0, parse_dates=True)
        returns = pd.read_csv('data/cleaned_returns.csv', index_col=0, parse_dates=True)
        
        # Calculate statistics
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratios = annual_returns / annual_volatility
        
        data = {
            'prices': prices,
            'returns': returns,
            'annual_returns': annual_returns,
            'annual_volatility': annual_volatility,
            'sharpe_ratios': sharpe_ratios,
            'correlations': returns.corr()
        }
        
        # Load asset features for additional info
        features_path = Path("data/asset_features.csv")
        if features_path.exists():
            features_df = pd.read_csv(features_path)
        else:
            # Create basic features from calculated data
            features_df = pd.DataFrame({
                'Asset': returns.columns,
                'Sector': 'Unknown',
                'Sharpe': sharpe_ratios.values,
                'Volatility': annual_volatility.values,
                'Expected_Return': annual_returns.values
            })
        
        # Load cluster info
        cluster_path = Path("data/cluster_analysis.json")
        if cluster_path.exists():
            with open(cluster_path, 'r') as f:
                cluster_data = json.load(f)
        else:
            # Create default cluster mapping
            cluster_data = {"asset_clusters": {asset: 0 for asset in returns.columns}}
        
        # Add asset_features and clusters to data dict for compatibility
        data['asset_features'] = features_df
        data['clusters'] = cluster_data.get("asset_clusters", {})
        
        return data, features_df, cluster_data
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load stock data")

def calculate_portfolio_metrics(weights: Dict[str, float], returns_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive portfolio metrics"""
    try:
        # Convert weights to pandas Series aligned with returns
        weight_series = pd.Series(weights).reindex(returns_data.columns, fill_value=0)
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data * weight_series).sum(axis=1)
        
        # Calculate metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Effective number of assets (inverse of Herfindahl index)
        effective_assets = 1 / (weight_series ** 2).sum() if (weight_series ** 2).sum() > 0 else 1
        
        # Concentration risk (max weight)
        concentration_risk = weight_series.max()
        
        # Calculate VaR (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            "expected_return": float(annual_return),
            "volatility": float(annual_vol),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "var_95": float(var_95),
            "effective_assets": float(effective_assets),
            "concentration_risk": float(concentration_risk)
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        # Return default metrics if calculation fails
        return {
            "expected_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "effective_assets": 1.0,
            "concentration_risk": 1.0
        }

def get_cluster_names() -> Dict[int, str]:
    """Get cluster ID to name mapping"""
    return {
        0: "Materials-Dominated",
        1: "Low-Momentum/Low-Risk", 
        2: "ETF/Diversified",
        3: "High-Momentum/High-Risk"
    }

def get_cluster_info() -> Dict[int, Dict[str, Any]]:
    """Get detailed cluster information"""
    return {
        0: {
            "name": "Materials-Dominated",
            "description": "Tactical allocation cluster focused on materials and commodities",
            "asset_count": 3,
            "avg_sharpe": 0.7,
            "avg_volatility": 0.25,
            "role": "Tactical Allocation",
            "target_allocation": "5-15%"
        },
        1: {
            "name": "Low-Momentum/Low-Risk",
            "description": "Stability-focused assets with lower volatility",
            "asset_count": 21,
            "avg_sharpe": 0.6,
            "avg_volatility": 0.20,
            "role": "Stability Core",
            "target_allocation": "20-30%"
        },
        2: {
            "name": "ETF/Diversified",
            "description": "Broad market ETFs and diversified holdings",
            "asset_count": 6,
            "avg_sharpe": 0.8,
            "avg_volatility": 0.18,
            "role": "Core Holdings",
            "target_allocation": "25-35%"
        },
        3: {
            "name": "High-Momentum/High-Risk",
            "description": "Growth-oriented assets with higher potential returns",
            "asset_count": 15,
            "avg_sharpe": 1.2,
            "avg_volatility": 0.35,
            "role": "Growth Engine",
            "target_allocation": "15-25%"
        }
    }

def generate_sample_historical_data(returns_data: pd.DataFrame, portfolio_weights: Dict[str, float], 
                                   benchmark: str = "SPY", days: int = 252) -> Dict[str, Any]:
    """Generate sample historical performance data"""
    try:
        # Get recent data
        recent_returns = returns_data.tail(days)
        dates = recent_returns.index.strftime('%Y-%m-%d').tolist()
        
        # Calculate portfolio returns
        weight_series = pd.Series(portfolio_weights).reindex(recent_returns.columns, fill_value=0)
        portfolio_returns = (recent_returns * weight_series).sum(axis=1).tolist()
        
        # Get benchmark returns
        if benchmark in recent_returns.columns:
            benchmark_returns = recent_returns[benchmark].tolist()
        else:
            # Use average market return if benchmark not available
            benchmark_returns = recent_returns.mean(axis=1).tolist()
        
        # Calculate cumulative returns
        cumulative_portfolio = np.cumprod(1 + np.array(portfolio_returns)).tolist()
        cumulative_benchmark = np.cumprod(1 + np.array(benchmark_returns)).tolist()
        
        # Calculate rolling metrics (30-day windows)
        rolling_sharpe = []
        rolling_volatility = []
        
        for i in range(30, len(portfolio_returns)):
            window_returns = portfolio_returns[i-30:i]
            rolling_vol = np.std(window_returns) * np.sqrt(252)
            rolling_sharpe_val = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252) if np.std(window_returns) > 0 else 0
            
            rolling_volatility.append(rolling_vol)
            rolling_sharpe.append(rolling_sharpe_val)
        
        # Pad the beginning with None values
        rolling_sharpe = [None] * 30 + rolling_sharpe
        rolling_volatility = [None] * 30 + rolling_volatility
        
        return {
            "dates": dates,
            "portfolio_returns": [float(r) for r in portfolio_returns],
            "benchmark_returns": [float(r) for r in benchmark_returns],
            "cumulative_portfolio": [float(r) for r in cumulative_portfolio],
            "cumulative_benchmark": [float(r) for r in cumulative_benchmark],
            "rolling_sharpe": [float(r) if r is not None else None for r in rolling_sharpe],
            "rolling_volatility": [float(r) if r is not None else None for r in rolling_volatility]
        }
        
    except Exception as e:
        logger.error(f"Error generating historical data: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate historical data") 