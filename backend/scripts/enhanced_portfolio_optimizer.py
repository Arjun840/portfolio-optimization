#!/usr/bin/env python3
"""
Enhanced Portfolio Optimization with Advanced Constraints

This script implements portfolio optimization with sophisticated constraints:
1. Maximum weight per asset (prevent over-concentration)
2. Maximum weight per cluster (ensure diversification)
3. Sector allocation limits
4. Minimum number of assets
5. Testing framework for historical vs ML-predicted returns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

warnings.filterwarnings('ignore')

class EnhancedPortfolioOptimizer:
    """Enhanced portfolio optimizer with advanced constraints."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the enhanced optimizer."""
        self.data_dir = data_dir
        self.returns = None
        self.prices = None
        self.cluster_data = None
        self.asset_features = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.risk_free_rate = 0.02
        
        # Enhanced constraint parameters
        self.constraints_config = {
            'max_weight_per_asset': 0.20,        # Max 20% per asset
            'max_weight_per_cluster': 0.40,      # Max 40% per cluster
            'min_assets': 5,                     # Minimum 5 assets
            'sector_limits': {                   # Sector allocation limits
                'Technology': 0.30,
                'Financial': 0.25,
                'Healthcare': 0.20,
                'Consumer': 0.25,
                'ETF_Broad': 0.30,
                'ETF_Tech': 0.15
            }
        }
        
        # Results storage
        self.optimization_results = {}
        self.constraint_tests = {}
        
    def load_data(self) -> bool:
        """Load all necessary data."""
        try:
            print("📊 Loading enhanced optimization data...")
            
            self.returns = pd.read_pickle(f'{self.data_dir}/cleaned_returns.pkl')
            self.prices = pd.read_pickle(f'{self.data_dir}/cleaned_prices.pkl')
            self.cluster_data = pd.read_csv(f'{self.data_dir}/asset_selection_framework.csv')
            self.asset_features = pd.read_pickle(f'{self.data_dir}/asset_features.pkl')
            
            print(f"✅ Data loaded: {self.returns.shape[0]} observations, {len(self.cluster_data)} assets")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_expected_returns(self, method: str = "historical") -> pd.Series:
        """Calculate expected returns with multiple methods for testing."""
        print(f"🔮 Calculating expected returns using {method} method...")
        
        if method == "historical":
            # Simple historical mean (annualized)
            expected_returns = self.returns.mean() * 252
            
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
                    sharpe_signal = asset_data['Sharpe_Ratio'] * 0.1  # Scale Sharpe
                    
                    # Weighted combination with ML focus
                    enhanced_return = (
                        0.3 * hist_return +           # Historical base
                        0.4 * recent_return +         # Recent momentum
                        0.2 * selection_score * 0.2 + # ML selection score
                        0.1 * sharpe_signal           # Risk-adjusted signal
                    )
                    
                    expected_returns[asset] = enhanced_return
            
            expected_returns = pd.Series(expected_returns)
            
        elif method == "conservative":
            # Conservative estimates (historical with downward adjustment)
            historical_returns = self.returns.mean() * 252
            expected_returns = historical_returns * 0.8  # 80% of historical
            
        elif method == "optimistic":
            # Optimistic estimates (using top performers)
            expected_returns = {}
            
            for _, asset_data in self.cluster_data.iterrows():
                asset = asset_data['Asset']
                if asset in self.returns.columns:
                    hist_return = self.returns[asset].mean() * 252
                    recent_boost = max(asset_data['Recent_3M_Return'] * 4, 0) * 0.5
                    expected_returns[asset] = hist_return + recent_boost
            
            expected_returns = pd.Series(expected_returns)
        
        # Filter to available assets
        self.expected_returns = expected_returns[expected_returns.index.isin(self.returns.columns)]
        
        print(f"   • Expected returns for {len(self.expected_returns)} assets")
        print(f"   • Mean: {self.expected_returns.mean():.1%}, Range: {self.expected_returns.min():.1%} to {self.expected_returns.max():.1%}")
        
        return self.expected_returns
    
    def calculate_covariance_matrix(self, method: str = "historical") -> pd.DataFrame:
        """Calculate covariance matrix."""
        print(f"📊 Calculating covariance matrix using {method} method...")
        
        if method == "historical":
            self.covariance_matrix = self.returns.cov() * 252
        elif method == "shrinkage":
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            cov_shrunk = lw.fit(self.returns.dropna()).covariance_
            self.covariance_matrix = pd.DataFrame(
                cov_shrunk * 252,
                index=self.returns.columns,
                columns=self.returns.columns
            )
        
        # Filter to expected returns assets
        assets = self.expected_returns.index
        self.covariance_matrix = self.covariance_matrix.loc[assets, assets]
        
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
    
    def optimize_portfolio(self, 
                         objective: str = "max_sharpe", 
                         constraint_type: str = "enhanced",
                         target_return: float = None) -> Dict:
        """Enhanced portfolio optimization with constraints."""
        print(f"🎯 Optimizing portfolio: {objective} with {constraint_type} constraints")
        
        n_assets = len(self.expected_returns)
        
        # Create constraints
        constraints = self.create_constraints(constraint_type)
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, self.expected_returns) - target_return
            })
        
        # Bounds: 0 <= weight <= max_weight_per_asset
        max_weight = self.constraints_config['max_weight_per_asset']
        if constraint_type == "basic":
            bounds = tuple((0, 1) for _ in range(n_assets))
        else:
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Objective functions
        if objective == "max_sharpe":
            def objective_func(weights):
                return -self.portfolio_performance(weights)[2]  # Negative Sharpe
        elif objective == "min_variance":
            def objective_func(weights):
                return self.portfolio_performance(weights)[1]  # Volatility
        elif objective == "max_return":
            def objective_func(weights):
                return -self.portfolio_performance(weights)[0]  # Negative return
        
        # Initial guess: equal weights (but respect constraints)
        if constraint_type == "basic":
            x0 = np.array([1/n_assets] * n_assets)
        else:
            # Start with constrained equal weights
            initial_weight = min(1/n_assets, max_weight)
            x0 = np.array([initial_weight] * n_assets)
            x0 = x0 / np.sum(x0)  # Normalize
        
        # Optimize
        try:
            result = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = result.x
                returns, volatility, sharpe = self.portfolio_performance(optimal_weights)
                
                # Analyze constraint satisfaction
                constraint_analysis = self.analyze_constraints(optimal_weights)
                
                portfolio = {
                    'weights': pd.Series(optimal_weights, index=self.expected_returns.index),
                    'expected_return': returns,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'objective': objective,
                    'constraint_type': constraint_type,
                    'constraint_analysis': constraint_analysis,
                    'optimization_success': True
                }
                
                print(f"   ✅ Optimization successful:")
                print(f"   📈 Return: {returns:.1%}, Vol: {volatility:.1%}, Sharpe: {sharpe:.3f}")
                print(f"   🎯 Max weight: {optimal_weights.max():.1%}, Active assets: {np.sum(optimal_weights > 0.01)}")
                
                return portfolio
            else:
                print(f"   ❌ Optimization failed: {result.message}")
                return {'optimization_success': False, 'message': result.message}
                
        except Exception as e:
            print(f"   ❌ Optimization error: {e}")
            return {'optimization_success': False, 'error': str(e)}
    
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
    
    def test_optimizer_behavior(self) -> Dict:
        """Test optimizer with different return methods and constraints."""
        print("\n🧪 TESTING OPTIMIZER BEHAVIOR")
        print("=" * 60)
        
        test_results = {}
        
        # Test scenarios
        return_methods = ["historical", "ml_enhanced", "conservative", "optimistic"]
        constraint_types = ["basic", "enhanced", "strict"]
        objectives = ["max_sharpe", "min_variance"]
        
        for return_method in return_methods:
            print(f"\n📊 Testing with {return_method} returns...")
            
            # Calculate returns for this method
            self.calculate_expected_returns(return_method)
            self.calculate_covariance_matrix("historical")
            
            test_results[return_method] = {}
            
            for constraint_type in constraint_types:
                test_results[return_method][constraint_type] = {}
                
                for objective in objectives:
                    result = self.optimize_portfolio(objective, constraint_type)
                    
                    if result.get('optimization_success', False):
                        # Simplified result for comparison
                        test_results[return_method][constraint_type][objective] = {
                            'return': result['expected_return'],
                            'volatility': result['volatility'],
                            'sharpe': result['sharpe_ratio'],
                            'max_weight': result['weights'].max(),
                            'active_assets': np.sum(result['weights'] > 0.01),
                            'top_3_assets': result['weights'].sort_values(ascending=False).head(3).to_dict()
                        }
                    else:
                        test_results[return_method][constraint_type][objective] = {
                            'optimization_failed': True
                        }
        
        self.constraint_tests = test_results
        return test_results
    
    def create_constraint_comparison(self) -> None:
        """Create comparison of different constraint levels."""
        print("\n📊 CONSTRAINT COMPARISON ANALYSIS")
        print("=" * 60)
        
        # Use ML enhanced returns for comparison
        self.calculate_expected_returns("ml_enhanced")
        self.calculate_covariance_matrix("historical")
        
        comparison_results = []
        
        constraint_scenarios = [
            ("No Constraints", "basic"),
            ("Enhanced Constraints", "enhanced"), 
            ("Strict Constraints", "strict")
        ]
        
        for scenario_name, constraint_type in constraint_scenarios:
            print(f"\n🎯 Testing {scenario_name}...")
            
            # Max Sharpe optimization
            result = self.optimize_portfolio("max_sharpe", constraint_type)
            
            if result.get('optimization_success', False):
                constraint_analysis = result['constraint_analysis']
                
                comparison_results.append({
                    'Scenario': scenario_name,
                    'Expected_Return': result['expected_return'],
                    'Volatility': result['volatility'],
                    'Sharpe_Ratio': result['sharpe_ratio'],
                    'Max_Asset_Weight': constraint_analysis['max_asset_weight'],
                    'Active_Assets': constraint_analysis['active_assets'],
                    'Max_Cluster_Weight': constraint_analysis.get('max_cluster_weight', 0),
                    'Top_Holdings': self._get_top_holdings(result['weights'], 3)
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        print(f"\n📋 Constraint Comparison Results:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def _get_top_holdings(self, weights: pd.Series, n: int = 3) -> str:
        """Get top N holdings as formatted string."""
        top_weights = weights.sort_values(ascending=False).head(n)
        return ", ".join([f"{asset} ({weight:.1%})" for asset, weight in top_weights.items() if weight > 0.01])
    
    def visualize_constraint_impact(self) -> None:
        """Visualize the impact of constraints on portfolio allocation."""
        print("📊 Creating constraint impact visualization...")
        
        # Compare basic vs enhanced vs strict constraints
        self.calculate_expected_returns("ml_enhanced")
        self.calculate_covariance_matrix("historical")
        
        results = {}
        constraint_types = ["basic", "enhanced", "strict"]
        
        for constraint_type in constraint_types:
            result = self.optimize_portfolio("max_sharpe", constraint_type)
            if result.get('optimization_success', False):
                results[constraint_type] = result
        
        if len(results) < 2:
            print("   ⚠️  Need at least 2 successful optimizations for comparison")
            return
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Risk-Return comparison
        for constraint_type, result in results.items():
            ax1.scatter(result['volatility'], result['expected_return'], 
                       s=100, label=f"{constraint_type.title()} Constraints")
        
        ax1.set_xlabel('Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Risk-Return by Constraint Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Asset concentration
        constraint_names = list(results.keys())
        max_weights = [results[ct]['constraint_analysis']['max_asset_weight'] for ct in constraint_names]
        
        ax2.bar(constraint_names, max_weights, color=['red', 'orange', 'green'])
        ax2.set_ylabel('Maximum Asset Weight')
        ax2.set_title('Asset Concentration by Constraint Level')
        ax2.axhline(y=self.constraints_config['max_weight_per_asset'], 
                   color='red', linestyle='--', label='Constraint Limit')
        ax2.legend()
        
        # Plot 3: Number of active assets
        active_assets = [results[ct]['constraint_analysis']['active_assets'] for ct in constraint_names]
        
        ax3.bar(constraint_names, active_assets, color=['red', 'orange', 'green'])
        ax3.set_ylabel('Number of Active Assets (>1%)')
        ax3.set_title('Diversification by Constraint Level')
        ax3.axhline(y=self.constraints_config['min_assets'], 
                   color='blue', linestyle='--', label='Minimum Required')
        ax3.legend()
        
        # Plot 4: Sharpe ratio comparison
        sharpe_ratios = [results[ct]['sharpe_ratio'] for ct in constraint_names]
        
        bars = ax4.bar(constraint_names, sharpe_ratios, color=['red', 'orange', 'green'])
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Risk-Adjusted Performance by Constraint Level')
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/../analysis_plots/constraint_impact_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_enhanced_results(self) -> None:
        """Save all enhanced optimization results."""
        print("\n💾 Saving enhanced optimization results...")
        
        # Save constraint test results
        if hasattr(self, 'constraint_tests') and self.constraint_tests:
            with open(f'{self.data_dir}/constraint_behavior_tests.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                json.dump(convert_numpy(self.constraint_tests), f, indent=2)
        
        # Save constraint configuration
        with open(f'{self.data_dir}/constraint_configuration.json', 'w') as f:
            json.dump(self.constraints_config, f, indent=2)
        
        print(f"   ✅ Enhanced results saved:")
        if hasattr(self, 'constraint_tests') and self.constraint_tests:
            print(f"      • {self.data_dir}/constraint_behavior_tests.json")
        print(f"      • {self.data_dir}/constraint_configuration.json")
    
    def run_enhanced_optimization_suite(self) -> None:
        """Run complete enhanced optimization with all tests."""
        print("🚀 ENHANCED PORTFOLIO OPTIMIZATION SUITE")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        # Test 1: Behavior testing with different return methods
        self.test_optimizer_behavior()
        
        # Test 2: Constraint comparison
        constraint_comparison = self.create_constraint_comparison()
        
        # Test 3: Visualize constraint impact
        self.visualize_constraint_impact()
        
        # Save all results
        self.save_enhanced_results()
        
        print(f"\n" + "=" * 80)
        print("✅ ENHANCED OPTIMIZATION SUITE COMPLETE!")
        print("📊 Check analysis_plots/ for constraint impact visualizations")
        print("📁 Check data/ for detailed test results and configurations")


def main():
    """Run enhanced portfolio optimization example."""
    optimizer = EnhancedPortfolioOptimizer(data_dir="data")
    optimizer.run_enhanced_optimization_suite()


if __name__ == "__main__":
    main() 