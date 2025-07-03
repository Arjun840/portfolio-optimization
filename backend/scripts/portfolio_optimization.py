#!/usr/bin/env python3
"""
Portfolio Optimization Algorithm Implementation

This script implements advanced portfolio optimization strategies:
1. Mean-Variance Optimization with Efficient Frontier
2. Sharpe Ratio Maximization 
3. Cluster-based Portfolio Construction
4. Risk Parity and Minimum Variance approaches
5. Advanced Constraints (max weight per asset/cluster, sector limits)
6. Testing framework for historical vs ML-predicted returns

Uses existing return predictions and clustering analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import json
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Advanced portfolio optimization with multiple strategies and constraints."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the portfolio optimizer."""
        self.data_dir = data_dir
        self.returns = None
        self.prices = None
        self.cluster_data = None
        self.asset_features = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Advanced constraint parameters
        self.constraints_config = {
            'max_weight_per_asset': 0.20,        # Max 20% per asset
            'max_weight_per_cluster': 0.40,      # Max 40% per cluster
            'min_assets': 5,                     # Minimum 5 assets with weight > 1%
            'sector_limits': {                   # Sector allocation limits
                'Technology': 0.30,
                'Financial': 0.25,
                'Healthcare': 0.20,
                'Consumer': 0.25,
                'ETF_Broad': 0.30,
                'ETF_Tech': 0.15
            }
        }
        
        # Optimization results storage
        self.efficient_frontier = None
        self.optimal_portfolios = {}
        self.constraint_tests = {}
        
    def load_data(self) -> bool:
        """Load all necessary data for optimization."""
        try:
            print("📊 Loading portfolio optimization data...")
            
            # Load returns data
            self.returns = pd.read_pickle(f'{self.data_dir}/cleaned_returns.pkl')
            self.prices = pd.read_pickle(f'{self.data_dir}/cleaned_prices.pkl')
            
            # Load cluster analysis
            self.cluster_data = pd.read_csv(f'{self.data_dir}/asset_selection_framework.csv')
            
            # Load asset features
            self.asset_features = pd.read_pickle(f'{self.data_dir}/asset_features.pkl')
            
            print(f"✅ Loaded data:")
            print(f"   • Returns: {self.returns.shape}")
            print(f"   • Assets with features: {len(self.cluster_data)}")
            print(f"   • Clusters: {sorted(self.cluster_data['Cluster'].unique())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_expected_returns(self, method: str = "historical") -> pd.Series:
        """Calculate expected returns using different methods."""
        print(f"🔮 Calculating expected returns using {method} method...")
        
        if method == "historical":
            # Simple historical mean returns (annualized)
            expected_returns = self.returns.mean() * 252
            
        elif method == "enhanced":
            # Use asset features for enhanced return estimates
            expected_returns = {}
            
            for _, asset_data in self.cluster_data.iterrows():
                asset = asset_data['Asset']
                if asset in self.returns.columns:
                    # Combine historical returns with selection score and recent performance
                    hist_return = self.returns[asset].mean() * 252
                    selection_score = asset_data['Selection_Score']
                    recent_return = asset_data['Recent_3M_Return'] * 4  # Annualized
                    
                    # Weighted combination
                    enhanced_return = (0.5 * hist_return + 
                                     0.3 * recent_return + 
                                     0.2 * selection_score * 0.15)  # Scale selection score
                    
                    expected_returns[asset] = enhanced_return
            
            expected_returns = pd.Series(expected_returns)
            
        elif method == "risk_adjusted":
            # Risk-adjusted expected returns based on Sharpe ratios
            expected_returns = {}
            
            for _, asset_data in self.cluster_data.iterrows():
                asset = asset_data['Asset']
                if asset in self.returns.columns:
                    sharpe_ratio = asset_data['Sharpe_Ratio']
                    volatility = asset_data['Annual_Volatility']
                    
                    # Expected return = risk_free_rate + sharpe * volatility
                    risk_adjusted_return = self.risk_free_rate + sharpe_ratio * volatility
                    expected_returns[asset] = risk_adjusted_return
            
            expected_returns = pd.Series(expected_returns)
            
        elif method == "conservative":
            # Conservative estimates (80% of historical)
            historical_returns = self.returns.mean() * 252
            expected_returns = historical_returns * 0.8
            
        elif method == "ml_enhanced":
            # ML-enhanced returns using existing predictions and features
            expected_returns = {}
            
            for _, asset_data in self.cluster_data.iterrows():
                asset = asset_data['Asset']
                if asset in self.returns.columns:
                    # Combine multiple signals
                    hist_return = self.returns[asset].mean() * 252
                    selection_score = asset_data['Selection_Score']
                    recent_return = asset_data['Recent_3M_Return'] * 4
                    sharpe_signal = asset_data['Sharpe_Ratio'] * 0.1
                    
                    # ML-focused combination
                    enhanced_return = (
                        0.3 * hist_return +           # Historical base
                        0.4 * recent_return +         # Recent momentum
                        0.2 * selection_score * 0.2 + # ML selection score
                        0.1 * sharpe_signal           # Risk-adjusted signal
                    )
                    
                    expected_returns[asset] = enhanced_return
            
            expected_returns = pd.Series(expected_returns)
        
        # Ensure we only have assets that exist in returns data
        self.expected_returns = expected_returns[expected_returns.index.isin(self.returns.columns)]
        
        print(f"   • Expected returns calculated for {len(self.expected_returns)} assets")
        print(f"   • Mean expected return: {self.expected_returns.mean():.1%}")
        print(f"   • Range: {self.expected_returns.min():.1%} to {self.expected_returns.max():.1%}")
        
        return self.expected_returns
    
    def calculate_covariance_matrix(self, method: str = "historical") -> pd.DataFrame:
        """Calculate covariance matrix using different methods."""
        print(f"📊 Calculating covariance matrix using {method} method...")
        
        if method == "historical":
            # Simple historical covariance (annualized)
            self.covariance_matrix = self.returns.cov() * 252
            
        elif method == "shrinkage":
            # Ledoit-Wolf shrinkage estimator
            from sklearn.covariance import LedoitWolf
            
            lw = LedoitWolf()
            cov_shrunk = lw.fit(self.returns.dropna()).covariance_
            self.covariance_matrix = pd.DataFrame(
                cov_shrunk * 252, 
                index=self.returns.columns, 
                columns=self.returns.columns
            )
            
        elif method == "exponential":
            # Exponentially weighted covariance
            ewm_cov = self.returns.ewm(span=60).cov().iloc[-len(self.returns.columns):]
            self.covariance_matrix = ewm_cov * 252
        
        # Filter to assets with expected returns
        assets = self.expected_returns.index
        self.covariance_matrix = self.covariance_matrix.loc[assets, assets]
        
        print(f"   • Covariance matrix: {self.covariance_matrix.shape}")
        print(f"   • Average volatility: {np.sqrt(np.diag(self.covariance_matrix)).mean():.1%}")
        
        return self.covariance_matrix
    
    def create_constraints(self, constraint_type: str = "enhanced") -> List[Dict]:
        """Create optimization constraints based on configuration."""
        n_assets = len(self.expected_returns)
        constraints = []
        
        # Basic constraint: weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        })
        
        if constraint_type in ["enhanced", "strict"]:
            # Maximum weight per asset
            max_asset_weight = self.constraints_config['max_weight_per_asset']
            for i in range(n_assets):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i: max_asset_weight - x[idx]
                })
            
            # Minimum number of assets (at least min_assets with weight > 1%)
            min_assets = self.constraints_config['min_assets']
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: np.sum(x > 0.01) - min_assets
            })
        
        if constraint_type == "strict":
            # Maximum weight per cluster
            max_cluster_weight = self.constraints_config['max_weight_per_cluster']
            
            # Create cluster mapping
            asset_to_cluster = dict(zip(self.cluster_data['Asset'], self.cluster_data['Cluster']))
            clusters = sorted(self.cluster_data['Cluster'].unique())
            
            for cluster_id in clusters:
                cluster_assets = [asset for asset in self.expected_returns.index 
                                if asset_to_cluster.get(asset) == cluster_id]
                if cluster_assets:
                    cluster_indices = [list(self.expected_returns.index).index(asset) 
                                     for asset in cluster_assets]
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, indices=cluster_indices: max_cluster_weight - np.sum([x[i] for i in indices])
                    })
            
            # Sector limits
            asset_to_sector = dict(zip(self.cluster_data['Asset'], self.cluster_data['Sector']))
            
            for sector, max_weight in self.constraints_config['sector_limits'].items():
                sector_assets = [asset for asset in self.expected_returns.index 
                               if asset_to_sector.get(asset) == sector]
                if sector_assets:
                    sector_indices = [list(self.expected_returns.index).index(asset) 
                                    for asset in sector_assets]
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, indices=sector_indices: max_weight - np.sum([x[i] for i in indices])
                    })
        
        print(f"   📋 Created {len(constraints)} constraints ({constraint_type} mode)")
        return constraints
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio performance metrics."""
        returns = np.dot(weights, self.expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / volatility if volatility > 0 else 0
        return returns, volatility, sharpe
    
    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Objective function for Sharpe ratio maximization."""
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        return self.portfolio_performance(weights)[1]
    
    def maximize_sharpe_ratio(self, constraint_type: str = "enhanced") -> Dict:
        """Find the portfolio that maximizes the Sharpe ratio."""
        print(f"🎯 Optimizing for maximum Sharpe ratio with {constraint_type} constraints...")
        
        n_assets = len(self.expected_returns)
        
        # Create constraints
        constraints = self.create_constraints(constraint_type)
        
        # Bounds
        if constraint_type == "basic":
            bounds = tuple((0, 1) for _ in range(n_assets))
        else:
            max_weight = self.constraints_config['max_weight_per_asset']
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess
        if constraint_type == "basic":
            x0 = np.array([1/n_assets] * n_assets)
        else:
            initial_weight = min(1/n_assets, self.constraints_config['max_weight_per_asset'])
            x0 = np.array([initial_weight] * n_assets)
            x0 = x0 / np.sum(x0)  # Normalize
        
        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            returns, volatility, sharpe = self.portfolio_performance(optimal_weights)
            
            # Analyze constraints
            constraint_analysis = self.analyze_constraints(optimal_weights)
            
            portfolio = {
                'weights': pd.Series(optimal_weights, index=self.expected_returns.index),
                'expected_return': returns,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'optimization_method': f'max_sharpe_{constraint_type}',
                'constraint_analysis': constraint_analysis
            }
            
            print(f"   ✅ Optimal Sharpe ratio: {sharpe:.3f}")
            print(f"   📈 Expected return: {returns:.1%}")
            print(f"   📊 Volatility: {volatility:.1%}")
            print(f"   🎯 Max weight: {optimal_weights.max():.1%}")
            print(f"   📊 Active assets: {np.sum(optimal_weights > 0.01)}")
            
            return portfolio
        else:
            print(f"   ❌ Optimization failed: {result.message}")
            return None
    
    def minimize_variance(self, target_return: float = None, constraint_type: str = "enhanced") -> Dict:
        """Find the minimum variance portfolio."""
        target_str = f" with target return {target_return:.1%}" if target_return else ""
        print(f"🛡️  Optimizing for minimum variance{target_str} with {constraint_type} constraints...")
        
        n_assets = len(self.expected_returns)
        
        # Constraints
        constraints = self.create_constraints(constraint_type)
        
        if target_return is not None:
            # Add return constraint
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, self.expected_returns) - target_return
            })
        
        # Bounds
        if constraint_type == "basic":
            bounds = tuple((0, 1) for _ in range(n_assets))
        else:
            max_weight = self.constraints_config['max_weight_per_asset']
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess
        if constraint_type == "basic":
            x0 = np.array([1/n_assets] * n_assets)
        else:
            initial_weight = min(1/n_assets, self.constraints_config['max_weight_per_asset'])
            x0 = np.array([initial_weight] * n_assets)
            x0 = x0 / np.sum(x0)
        
        # Optimize
        result = minimize(
            self.portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            returns, volatility, sharpe = self.portfolio_performance(optimal_weights)
            
            constraint_analysis = self.analyze_constraints(optimal_weights)
            
            portfolio = {
                'weights': pd.Series(optimal_weights, index=self.expected_returns.index),
                'expected_return': returns,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'optimization_method': f'min_variance_{constraint_type}',
                'constraint_analysis': constraint_analysis
            }
            
            print(f"   ✅ Minimum volatility: {volatility:.1%}")
            print(f"   📈 Expected return: {returns:.1%}")
            print(f"   🎯 Sharpe ratio: {sharpe:.3f}")
            print(f"   📊 Max weight: {optimal_weights.max():.1%}")
            print(f"   📊 Active assets: {np.sum(optimal_weights > 0.01)}")
            
            return portfolio
        else:
            print(f"   ❌ Optimization failed: {result.message}")
            return None
    
    def analyze_constraints(self, weights: np.ndarray) -> Dict:
        """Analyze how well the portfolio satisfies constraints."""
        analysis = {}
        
        # Asset concentration
        max_weight = weights.max()
        analysis['max_asset_weight'] = max_weight
        analysis['max_weight_constraint_satisfied'] = max_weight <= self.constraints_config['max_weight_per_asset'] + 1e-6
        
        # Number of active assets
        active_assets = np.sum(weights > 0.01)
        analysis['active_assets'] = active_assets
        analysis['min_assets_constraint_satisfied'] = active_assets >= self.constraints_config['min_assets']
        
        # Cluster concentration
        asset_to_cluster = dict(zip(self.cluster_data['Asset'], self.cluster_data['Cluster']))
        cluster_weights = {}
        
        for i, asset in enumerate(self.expected_returns.index):
            cluster_id = asset_to_cluster.get(asset, -1)
            if cluster_id not in cluster_weights:
                cluster_weights[cluster_id] = 0
            cluster_weights[cluster_id] += weights[i]
        
        analysis['cluster_weights'] = cluster_weights
        analysis['max_cluster_weight'] = max(cluster_weights.values()) if cluster_weights else 0
        analysis['max_cluster_constraint_satisfied'] = all(
            weight <= self.constraints_config['max_weight_per_cluster'] + 1e-6 
            for weight in cluster_weights.values()
        )
        
        # Sector concentration
        asset_to_sector = dict(zip(self.cluster_data['Asset'], self.cluster_data['Sector']))
        sector_weights = {}
        
        for i, asset in enumerate(self.expected_returns.index):
            sector = asset_to_sector.get(asset, 'Unknown')
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weights[i]
        
        analysis['sector_weights'] = sector_weights
        
        return analysis
    
    def test_constraint_behavior(self) -> Dict:
        """Test optimizer behavior with different return methods and constraints."""
        print("\n🧪 TESTING CONSTRAINT BEHAVIOR")
        print("=" * 60)
        
        test_results = {}
        return_methods = ["historical", "ml_enhanced", "conservative"]
        constraint_types = ["basic", "enhanced", "strict"]
        
        for return_method in return_methods:
            print(f"\n📊 Testing with {return_method} returns...")
            
            # Calculate returns for this method
            self.calculate_expected_returns(return_method)
            self.calculate_covariance_matrix("historical")
            
            test_results[return_method] = {}
            
            for constraint_type in constraint_types:
                print(f"   Testing {constraint_type} constraints...")
                
                # Test max Sharpe optimization
                result = self.maximize_sharpe_ratio(constraint_type)
                
                if result is not None:
                    test_results[return_method][constraint_type] = {
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe': result['sharpe_ratio'],
                        'max_weight': result['weights'].max(),
                        'active_assets': np.sum(result['weights'] > 0.01),
                        'constraint_analysis': result['constraint_analysis'],
                        'top_3_assets': result['weights'].sort_values(ascending=False).head(3).to_dict()
                    }
                else:
                    test_results[return_method][constraint_type] = {'optimization_failed': True}
        
        self.constraint_tests = test_results
        return test_results
    
    def create_constraint_comparison_visual(self) -> None:
        """Create visualization comparing different constraint levels."""
        print("📊 Creating constraint comparison visualization...")
        
        # Test with ML-enhanced returns
        self.calculate_expected_returns("ml_enhanced")
        self.calculate_covariance_matrix("historical")
        
        results = {}
        constraint_types = ["basic", "enhanced", "strict"]
        
        for constraint_type in constraint_types:
            result = self.maximize_sharpe_ratio(constraint_type)
            if result is not None:
                results[constraint_type] = result
        
        if len(results) < 2:
            print("   ⚠️  Need at least 2 successful optimizations for comparison")
            return
        
        # Create comparison visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Risk-Return
        for constraint_type, result in results.items():
            ax1.scatter(result['volatility'], result['expected_return'], 
                       s=150, label=f"{constraint_type.title()} Constraints")
        
        ax1.set_xlabel('Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Risk-Return by Constraint Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Asset concentration
        constraint_names = list(results.keys())
        max_weights = [results[ct]['constraint_analysis']['max_asset_weight'] for ct in constraint_names]
        
        bars = ax2.bar(constraint_names, max_weights, color=['red', 'orange', 'green'])
        ax2.set_ylabel('Maximum Asset Weight')
        ax2.set_title('Asset Concentration by Constraint Level')
        ax2.axhline(y=self.constraints_config['max_weight_per_asset'], 
                   color='red', linestyle='--', label='Constraint Limit')
        ax2.legend()
        
        # Add value labels
        for bar, value in zip(bars, max_weights):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{value:.1%}', ha='center', va='bottom')
        
        # Plot 3: Number of active assets
        active_assets = [results[ct]['constraint_analysis']['active_assets'] for ct in constraint_names]
        
        bars = ax3.bar(constraint_names, active_assets, color=['red', 'orange', 'green'])
        ax3.set_ylabel('Number of Active Assets (>1%)')
        ax3.set_title('Diversification by Constraint Level')
        ax3.axhline(y=self.constraints_config['min_assets'], 
                   color='blue', linestyle='--', label='Minimum Required')
        ax3.legend()
        
        # Add value labels
        for bar, value in zip(bars, active_assets):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value}', ha='center', va='bottom')
        
        # Plot 4: Sharpe ratio comparison
        sharpe_ratios = [results[ct]['sharpe_ratio'] for ct in constraint_names]
        
        bars = ax4.bar(constraint_names, sharpe_ratios, color=['red', 'orange', 'green'])
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Risk-Adjusted Performance by Constraint Level')
        
        # Add value labels
        for bar, value in zip(bars, sharpe_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/../analysis_plots/constraint_comparison_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """Generate the efficient frontier."""
        print(f"📈 Generating efficient frontier with {n_portfolios} portfolios...")
        
        # Find min and max possible returns
        min_vol_portfolio = self.minimize_variance()
        max_sharpe_portfolio = self.maximize_sharpe_ratio()
        
        if min_vol_portfolio is None or max_sharpe_portfolio is None:
            print("   ❌ Failed to find boundary portfolios")
            return None
        
        min_return = min_vol_portfolio['expected_return']
        max_return = max_sharpe_portfolio['expected_return']
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        frontier_portfolios = []
        
        for target_return in target_returns:
            portfolio = self.minimize_variance(target_return)
            if portfolio is not None:
                frontier_portfolios.append({
                    'return': portfolio['expected_return'],
                    'volatility': portfolio['volatility'],
                    'sharpe': portfolio['sharpe_ratio'],
                    'weights': portfolio['weights']
                })
        
        self.efficient_frontier = pd.DataFrame(frontier_portfolios)
        
        print(f"   ✅ Generated {len(self.efficient_frontier)} efficient portfolios")
        
        return self.efficient_frontier
    
    def cluster_based_optimization(self, cluster_strategy: str = "equal_weight") -> Dict:
        """Optimize portfolio using cluster-based approach."""
        print(f"🎯 Cluster-based optimization with {cluster_strategy} strategy...")
        
        clusters = self.cluster_data['Cluster'].unique()
        cluster_portfolios = {}
        
        # Step 1: Optimize within each cluster
        for cluster_id in clusters:
            cluster_assets = self.cluster_data[
                self.cluster_data['Cluster'] == cluster_id
            ]['Asset'].tolist()
            
            # Filter to assets with data
            cluster_assets = [a for a in cluster_assets if a in self.expected_returns.index]
            
            if len(cluster_assets) == 0:
                continue
                
            print(f"   Optimizing Cluster {cluster_id} ({len(cluster_assets)} assets)...")
            
            # Create sub-problem for this cluster
            cluster_returns = self.expected_returns[cluster_assets]
            cluster_cov = self.covariance_matrix.loc[cluster_assets, cluster_assets]
            
            # Optimize within cluster (max Sharpe)
            n_assets = len(cluster_assets)
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(n_assets))
            x0 = np.array([1/n_assets] * n_assets)
            
            def cluster_negative_sharpe(weights):
                ret = np.dot(weights, cluster_returns)
                vol = np.sqrt(np.dot(weights.T, np.dot(cluster_cov, weights)))
                return -(ret - self.risk_free_rate) / vol if vol > 0 else -1e6
            
            result = minimize(
                cluster_negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                cluster_weights = pd.Series(result.x, index=cluster_assets)
                ret = np.dot(result.x, cluster_returns)
                vol = np.sqrt(np.dot(result.x.T, np.dot(cluster_cov, result.x)))
                sharpe = (ret - self.risk_free_rate) / vol
                
                cluster_portfolios[cluster_id] = {
                    'weights': cluster_weights,
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'assets': cluster_assets
                }
                
                print(f"      Cluster {cluster_id} Sharpe: {sharpe:.3f}")
        
        # Step 2: Allocate across clusters
        if cluster_strategy == "equal_weight":
            cluster_allocation = {cid: 1/len(cluster_portfolios) for cid in cluster_portfolios.keys()}
            
        elif cluster_strategy == "risk_parity":
            # Allocate inversely proportional to cluster volatility
            total_inv_vol = sum(1/p['volatility'] for p in cluster_portfolios.values())
            cluster_allocation = {
                cid: (1/p['volatility']) / total_inv_vol 
                for cid, p in cluster_portfolios.items()
            }
            
        elif cluster_strategy == "sharpe_weighted":
            # Allocate proportional to cluster Sharpe ratios
            total_sharpe = sum(max(p['sharpe'], 0) for p in cluster_portfolios.values())
            cluster_allocation = {
                cid: max(p['sharpe'], 0) / total_sharpe 
                for cid, p in cluster_portfolios.items()
            }
        
        # Step 3: Combine into final portfolio
        final_weights = pd.Series(0.0, index=self.expected_returns.index)
        
        for cluster_id, cluster_weight in cluster_allocation.items():
            cluster_portfolio = cluster_portfolios[cluster_id]
            for asset, asset_weight in cluster_portfolio['weights'].items():
                final_weights[asset] = cluster_weight * asset_weight
        
        # Calculate final portfolio metrics
        returns, volatility, sharpe = self.portfolio_performance(final_weights.values)
        
        portfolio = {
            'weights': final_weights,
            'expected_return': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'cluster_allocation': cluster_allocation,
            'cluster_portfolios': cluster_portfolios,
            'optimization_method': f'cluster_{cluster_strategy}'
        }
        
        print(f"   ✅ Final portfolio Sharpe: {sharpe:.3f}")
        print(f"   📊 Cluster allocation: {cluster_allocation}")
        
        return portfolio
    
    def create_portfolio_comparison(self) -> pd.DataFrame:
        """Create a comparison of different optimization strategies."""
        print("\n🔍 PORTFOLIO OPTIMIZATION COMPARISON")
        print("=" * 60)
        
        strategies = [
            ("Maximum Sharpe", lambda: self.maximize_sharpe_ratio()),
            ("Minimum Variance", lambda: self.minimize_variance()),
            ("Max Sharpe Enhanced", lambda: self.maximize_sharpe_ratio("enhanced")),
            ("Max Sharpe Strict", lambda: self.maximize_sharpe_ratio("strict")),
            ("Cluster Equal Weight", lambda: self.cluster_based_optimization("equal_weight")),
            ("Cluster Sharpe Weighted", lambda: self.cluster_based_optimization("sharpe_weighted"))
        ]
        
        results = []
        
        for strategy_name, strategy_func in strategies:
            print(f"\n🎯 Running {strategy_name} optimization...")
            
            portfolio = strategy_func()
            
            if portfolio is not None:
                # Calculate additional metrics
                weights = portfolio['weights']
                
                # Concentration metrics
                effective_assets = 1 / np.sum(weights**2)  # Effective number of assets
                max_weight = weights.max()
                
                # Cluster diversification
                cluster_weights = self.cluster_data.set_index('Asset')['Cluster'].reindex(weights.index)
                cluster_allocation = weights.groupby(cluster_weights).sum()
                cluster_concentration = 1 / np.sum(cluster_allocation**2)
                
                result = {
                    'Strategy': strategy_name,
                    'Expected_Return': portfolio['expected_return'],
                    'Volatility': portfolio['volatility'],
                    'Sharpe_Ratio': portfolio['sharpe_ratio'],
                    'Max_Weight': max_weight,
                    'Effective_Assets': effective_assets,
                    'Cluster_Diversification': cluster_concentration,
                    'Top_5_Holdings': self._get_top_holdings(weights, 5)
                }
                
                results.append(result)
                self.optimal_portfolios[strategy_name] = portfolio
                
                print(f"   ✅ {strategy_name}: Sharpe {portfolio['sharpe_ratio']:.3f}, "
                      f"Return {portfolio['expected_return']:.1%}, Vol {portfolio['volatility']:.1%}")
        
        comparison_df = pd.DataFrame(results)
        return comparison_df
    
    def _get_top_holdings(self, weights: pd.Series, n: int = 5) -> str:
        """Get top N holdings as a formatted string."""
        top_weights = weights.sort_values(ascending=False).head(n)
        holdings = [f"{asset} ({weight:.1%})" for asset, weight in top_weights.items() if weight > 0.01]
        return ", ".join(holdings)
    
    def visualize_efficient_frontier(self) -> None:
        """Create visualization of the efficient frontier and optimal portfolios."""
        if self.efficient_frontier is None:
            print("   ⚠️  Generate efficient frontier first")
            return
        
        print("📊 Creating efficient frontier visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Efficient Frontier
        ax1.plot(self.efficient_frontier['volatility'], 
                self.efficient_frontier['return'], 
                'b-', linewidth=2, label='Efficient Frontier')
        
        # Plot optimal portfolios
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        markers = ['o', 's', '^', 'D', 'v', '*']
        
        for i, (name, portfolio) in enumerate(self.optimal_portfolios.items()):
            if i < len(colors):
                ax1.scatter(portfolio['volatility'], portfolio['expected_return'],
                          color=colors[i], marker=markers[i], s=100, 
                          label=name, zorder=5)
        
        ax1.set_xlabel('Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Efficient Frontier and Optimal Portfolios')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sharpe Ratio vs Volatility
        ax2.plot(self.efficient_frontier['volatility'], 
                self.efficient_frontier['sharpe'], 
                'g-', linewidth=2, label='Sharpe Ratio')
        
        # Mark optimal portfolios
        for i, (name, portfolio) in enumerate(self.optimal_portfolios.items()):
            if i < len(colors):
                ax2.scatter(portfolio['volatility'], portfolio['sharpe_ratio'],
                          color=colors[i], marker=markers[i], s=100, 
                          label=name, zorder=5)
        
        ax2.set_xlabel('Volatility')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio vs Volatility')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/../analysis_plots/enhanced_efficient_frontier.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_optimization_results(self) -> None:
        """Save all optimization results."""
        print("\n💾 Saving optimization results...")
        
        # Save portfolio comparison
        if len(self.optimal_portfolios) > 0:
            comparison_df = pd.DataFrame([
                {
                    'Strategy': name,
                    'Expected_Return': p['expected_return'],
                    'Volatility': p['volatility'],
                    'Sharpe_Ratio': p['sharpe_ratio'],
                    'Optimization_Method': p['optimization_method']
                }
                for name, p in self.optimal_portfolios.items()
            ])
            
            comparison_df.to_csv(f'{self.data_dir}/enhanced_optimization_results.csv', index=False)
        
        # Save detailed portfolios
        detailed_results = {}
        for name, portfolio in self.optimal_portfolios.items():
            # Convert to serializable format
            detailed_results[name] = {
                'expected_return': float(portfolio['expected_return']),
                'volatility': float(portfolio['volatility']),
                'sharpe_ratio': float(portfolio['sharpe_ratio']),
                'optimization_method': portfolio['optimization_method'],
                'weights': {asset: float(weight) for asset, weight in portfolio['weights'].items() if weight > 0.001}
            }
            
            # Add constraint analysis if available
            if 'constraint_analysis' in portfolio:
                constraint_analysis = portfolio['constraint_analysis']
                detailed_results[name]['constraint_analysis'] = {
                    'max_asset_weight': float(constraint_analysis['max_asset_weight']),
                    'active_assets': int(constraint_analysis['active_assets']),
                    'max_cluster_weight': float(constraint_analysis.get('max_cluster_weight', 0)),
                    'constraints_satisfied': {
                        'max_weight': bool(constraint_analysis['max_weight_constraint_satisfied']),
                        'min_assets': bool(constraint_analysis['min_assets_constraint_satisfied']),
                        'cluster_limits': bool(constraint_analysis.get('max_cluster_constraint_satisfied', True))
                    }
                }
            
            # Add cluster info for cluster-based strategies
            if 'cluster_allocation' in portfolio:
                detailed_results[name]['cluster_allocation'] = {
                    str(k): float(v) for k, v in portfolio['cluster_allocation'].items()
                }
        
        with open(f'{self.data_dir}/enhanced_detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save constraint tests if available
        if hasattr(self, 'constraint_tests') and self.constraint_tests:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            with open(f'{self.data_dir}/constraint_behavior_tests.json', 'w') as f:
                json.dump(convert_numpy(self.constraint_tests), f, indent=2)
        
        # Save efficient frontier if available
        if self.efficient_frontier is not None:
            self.efficient_frontier.to_csv(f'{self.data_dir}/enhanced_efficient_frontier.csv', index=False)
        
        # Save constraint configuration
        with open(f'{self.data_dir}/constraint_configuration.json', 'w') as f:
            json.dump(self.constraints_config, f, indent=2)
        
        print(f"   ✅ Enhanced results saved:")
        print(f"      • {self.data_dir}/enhanced_optimization_results.csv")
        print(f"      • {self.data_dir}/enhanced_detailed_results.json")
        print(f"      • {self.data_dir}/constraint_configuration.json")
        if hasattr(self, 'constraint_tests') and self.constraint_tests:
            print(f"      • {self.data_dir}/constraint_behavior_tests.json")
        if self.efficient_frontier is not None:
            print(f"      • {self.data_dir}/enhanced_efficient_frontier.csv")
    
    def run_complete_optimization(self, 
                                expected_returns_method: str = "enhanced",
                                covariance_method: str = "historical",
                                test_constraints: bool = True) -> None:
        """Run the complete enhanced portfolio optimization workflow."""
        print("🚀 ENHANCED PORTFOLIO OPTIMIZATION FRAMEWORK")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        # Test constraint behavior first if requested
        if test_constraints:
            self.test_constraint_behavior()
            self.create_constraint_comparison_visual()
        
        # Calculate expected returns and covariance
        self.calculate_expected_returns(expected_returns_method)
        self.calculate_covariance_matrix(covariance_method)
        
        # Generate efficient frontier
        self.generate_efficient_frontier()
        
        # Run all optimization strategies
        comparison_df = self.create_portfolio_comparison()
        
        print(f"\n📊 ENHANCED OPTIMIZATION RESULTS SUMMARY")
        print("=" * 60)
        print(comparison_df[['Strategy', 'Expected_Return', 'Volatility', 'Sharpe_Ratio', 'Max_Weight']].to_string(index=False))
        
        # Create visualizations
        self.visualize_efficient_frontier()
        
        # Save results
        self.save_optimization_results()
        
        print(f"\n" + "=" * 80)
        print("✅ ENHANCED PORTFOLIO OPTIMIZATION COMPLETE!")
        
        if len(self.optimal_portfolios) > 0:
            best_sharpe = max(self.optimal_portfolios.values(), key=lambda p: p['sharpe_ratio'])
            min_vol = min(self.optimal_portfolios.values(), key=lambda p: p['volatility'])
            print(f"🎯 Best Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f} ({best_sharpe['optimization_method']})")
            print(f"🔒 Lowest Volatility: {min_vol['volatility']:.1%} ({min_vol['optimization_method']})")


def main():
    """Run enhanced portfolio optimization example."""
    optimizer = PortfolioOptimizer(data_dir="data")
    
    # Run with enhanced returns, historical covariance, and constraint testing
    optimizer.run_complete_optimization(
        expected_returns_method="ml_enhanced",
        covariance_method="historical",
        test_constraints=True
    )


if __name__ == "__main__":
    main() 