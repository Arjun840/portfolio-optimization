#!/usr/bin/env python3
"""
Portfolio Construction Example Using Cluster Analysis

This script demonstrates how to use cluster analysis results to construct
diversified portfolios that avoid concentration in similar assets.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class ClusterBasedPortfolioConstructor:
    """Construct portfolios using cluster analysis for diversification."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the portfolio constructor."""
        self.data_dir = data_dir
        self.cluster_labels = None
        self.asset_framework = None
        self.cluster_analysis = None
        
    def load_cluster_data(self) -> bool:
        """Load cluster analysis results."""
        try:
            print("📊 Loading cluster analysis results...")
            
            # Load cluster labels
            self.cluster_labels = pd.read_csv(f'{self.data_dir}/cluster_labels_for_portfolio.csv')
            
            # Load asset selection framework
            self.asset_framework = pd.read_csv(f'{self.data_dir}/asset_selection_framework.csv')
            
            # Load comprehensive analysis
            with open(f'{self.data_dir}/comprehensive_cluster_analysis.json', 'r') as f:
                self.cluster_analysis = json.load(f)
            
            print(f"✅ Loaded cluster data for {len(self.cluster_labels)} assets")
            print(f"   • Clusters: {sorted(self.cluster_labels['Cluster'].unique())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading cluster data: {e}")
            return False
    
    def get_cluster_characteristics(self) -> Dict:
        """Get cluster characteristics for portfolio allocation."""
        cluster_chars = {}
        
        for cluster_id in sorted(self.cluster_labels['Cluster'].unique()):
            cluster_name = f'Cluster_{cluster_id}'
            
            if cluster_name in self.cluster_analysis['cluster_composition_analysis']:
                analysis = self.cluster_analysis['cluster_composition_analysis'][cluster_name]
                
                cluster_chars[cluster_id] = {
                    'theme': analysis['theme'],
                    'size': analysis['size'],
                    'performance': analysis['performance_summary'],
                    'allocation': self.cluster_analysis['portfolio_construction_guidelines']['cluster_allocation_framework'][cluster_name]
                }
        
        return cluster_chars
    
    def construct_conservative_portfolio(self, target_amount: float = 100000) -> Dict:
        """Construct a conservative portfolio emphasizing stability."""
        print(f"\n💼 CONSTRUCTING CONSERVATIVE PORTFOLIO")
        print("=" * 60)
        
        # Conservative allocation weights (emphasizing defensive assets)
        cluster_weights = {
            0: 0.10,  # Materials/Commodities - small tactical allocation
            1: 0.45,  # Defensive/Low-Risk - large defensive allocation  
            2: 0.35,  # ETFs - solid core holdings
            3: 0.10   # High-Momentum - small growth allocation
        }
        
        portfolio = self._construct_portfolio_by_weights(cluster_weights, target_amount)
        portfolio['strategy'] = 'Conservative'
        portfolio['risk_profile'] = 'Low'
        portfolio['target_return'] = '8-12% annually'
        
        return portfolio
    
    def construct_balanced_portfolio(self, target_amount: float = 100000) -> Dict:
        """Construct a balanced portfolio with moderate risk."""
        print(f"\n💼 CONSTRUCTING BALANCED PORTFOLIO")
        print("=" * 60)
        
        # Balanced allocation weights
        cluster_weights = {
            0: 0.10,  # Materials/Commodities
            1: 0.30,  # Defensive/Low-Risk
            2: 0.35,  # ETFs - core holdings
            3: 0.25   # High-Momentum - moderate growth
        }
        
        portfolio = self._construct_portfolio_by_weights(cluster_weights, target_amount)
        portfolio['strategy'] = 'Balanced'
        portfolio['risk_profile'] = 'Moderate'
        portfolio['target_return'] = '12-16% annually'
        
        return portfolio
    
    def construct_growth_portfolio(self, target_amount: float = 100000) -> Dict:
        """Construct a growth-oriented portfolio."""
        print(f"\n💼 CONSTRUCTING GROWTH PORTFOLIO")
        print("=" * 60)
        
        # Growth allocation weights (emphasizing high-momentum assets)
        cluster_weights = {
            0: 0.15,  # Materials/Commodities - tactical allocation
            1: 0.20,  # Defensive/Low-Risk - minimal defensive
            2: 0.25,  # ETFs - reduced core holdings
            3: 0.40   # High-Momentum - large growth allocation
        }
        
        portfolio = self._construct_portfolio_by_weights(cluster_weights, target_amount)
        portfolio['strategy'] = 'Growth'
        portfolio['risk_profile'] = 'High'
        portfolio['target_return'] = '16-25% annually'
        
        return portfolio
    
    def _construct_portfolio_by_weights(self, cluster_weights: Dict, target_amount: float) -> Dict:
        """Construct portfolio given cluster weights."""
        portfolio = {
            'total_amount': target_amount,
            'allocations': [],
            'cluster_breakdown': {},
            'sector_breakdown': {},
            'performance_metrics': {}
        }
        
        total_sharpe = 0
        total_volatility = 0
        total_return = 0
        
        for cluster_id, weight in cluster_weights.items():
            cluster_amount = target_amount * weight
            
            # Get top assets from this cluster
            cluster_assets = self.asset_framework[
                self.asset_framework['Cluster'] == cluster_id
            ].head(5)  # Top 5 assets per cluster
            
            # Distribute cluster amount among top assets
            assets_in_cluster = len(cluster_assets)
            amount_per_asset = cluster_amount / assets_in_cluster
            
            cluster_info = {
                'cluster_id': cluster_id,
                'cluster_weight': weight,
                'cluster_amount': cluster_amount,
                'assets': []
            }
            
            for _, asset_row in cluster_assets.iterrows():
                allocation = {
                    'symbol': asset_row['Asset'],
                    'sector': asset_row['Sector'],
                    'cluster': cluster_id,
                    'amount': amount_per_asset,
                    'weight': amount_per_asset / target_amount,
                    'sharpe_ratio': asset_row['Sharpe_Ratio'],
                    'volatility': asset_row['Annual_Volatility'],
                    'recent_return': asset_row['Recent_3M_Return'],
                    'selection_score': asset_row['Selection_Score'],
                    'recommendation': asset_row['Recommendation']
                }
                
                portfolio['allocations'].append(allocation)
                cluster_info['assets'].append(allocation)
                
                # Accumulate weighted metrics
                asset_weight = amount_per_asset / target_amount
                total_sharpe += asset_row['Sharpe_Ratio'] * asset_weight
                total_volatility += asset_row['Annual_Volatility'] * asset_weight
                
                # Update sector breakdown
                sector = asset_row['Sector']
                if sector not in portfolio['sector_breakdown']:
                    portfolio['sector_breakdown'][sector] = {'amount': 0, 'weight': 0}
                portfolio['sector_breakdown'][sector]['amount'] += amount_per_asset
                portfolio['sector_breakdown'][sector]['weight'] += asset_weight
            
            portfolio['cluster_breakdown'][cluster_id] = cluster_info
        
        # Calculate portfolio metrics
        portfolio['performance_metrics'] = {
            'weighted_sharpe': total_sharpe,
            'weighted_volatility': total_volatility,
            'expected_return': total_sharpe * total_volatility,  # Simplified calculation
            'num_assets': len(portfolio['allocations']),
            'num_clusters': len(cluster_weights),
            'num_sectors': len(portfolio['sector_breakdown'])
        }
        
        return portfolio
    
    def display_portfolio(self, portfolio: Dict) -> None:
        """Display portfolio details."""
        print(f"\n📊 {portfolio['strategy'].upper()} PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"💰 Total Amount: ${portfolio['total_amount']:,.2f}")
        print(f"📈 Strategy: {portfolio['strategy']}")
        print(f"⚠️  Risk Profile: {portfolio['risk_profile']}")
        print(f"🎯 Target Return: {portfolio['target_return']}")
        
        # Performance metrics
        metrics = portfolio['performance_metrics']
        print(f"\n📊 Portfolio Metrics:")
        print(f"   • Weighted Sharpe Ratio: {metrics['weighted_sharpe']:.3f}")
        print(f"   • Weighted Volatility: {metrics['weighted_volatility']:.1%}")
        print(f"   • Expected Return: {metrics['expected_return']:.1%}")
        print(f"   • Number of Assets: {metrics['num_assets']}")
        print(f"   • Number of Clusters: {metrics['num_clusters']}")
        print(f"   • Number of Sectors: {metrics['num_sectors']}")
        
        # Cluster breakdown
        print(f"\n🎯 Cluster Allocation:")
        for cluster_id, cluster_info in portfolio['cluster_breakdown'].items():
            print(f"   • Cluster {cluster_id}: ${cluster_info['cluster_amount']:,.2f} "
                  f"({cluster_info['cluster_weight']:.1%}) - {len(cluster_info['assets'])} assets")
        
        # Sector breakdown
        print(f"\n🏢 Sector Allocation:")
        sorted_sectors = sorted(portfolio['sector_breakdown'].items(), 
                              key=lambda x: x[1]['weight'], reverse=True)
        for sector, data in sorted_sectors:
            print(f"   • {sector}: ${data['amount']:,.2f} ({data['weight']:.1%})")
        
        # Top holdings
        print(f"\n🏆 Top Holdings:")
        sorted_holdings = sorted(portfolio['allocations'], 
                               key=lambda x: x['weight'], reverse=True)[:10]
        for holding in sorted_holdings:
            print(f"   • {holding['symbol']} ({holding['sector']}): "
                  f"${holding['amount']:,.2f} ({holding['weight']:.1%}) - "
                  f"Sharpe: {holding['sharpe_ratio']:.3f}")
    
    def save_portfolio(self, portfolio: Dict, filename: str) -> None:
        """Save portfolio to JSON file."""
        with open(f'{self.data_dir}/{filename}', 'w') as f:
            json.dump(portfolio, f, indent=2, default=str)
        print(f"✅ Portfolio saved to {self.data_dir}/{filename}")
    
    def create_portfolio_comparison(self, portfolios: List[Dict]) -> pd.DataFrame:
        """Create comparison table of different portfolios."""
        comparison_data = []
        
        for portfolio in portfolios:
            metrics = portfolio['performance_metrics']
            
            row = {
                'Strategy': portfolio['strategy'],
                'Risk_Profile': portfolio['risk_profile'],
                'Target_Return': portfolio['target_return'],
                'Weighted_Sharpe': f"{metrics['weighted_sharpe']:.3f}",
                'Weighted_Volatility': f"{metrics['weighted_volatility']:.1%}",
                'Expected_Return': f"{metrics['expected_return']:.1%}",
                'Num_Assets': metrics['num_assets'],
                'Num_Clusters': metrics['num_clusters'],
                'Num_Sectors': metrics['num_sectors']
            }
            
            # Add cluster allocations
            for cluster_id in range(4):  # Assuming 4 clusters
                cluster_weight = 0
                if cluster_id in portfolio['cluster_breakdown']:
                    cluster_weight = portfolio['cluster_breakdown'][cluster_id]['cluster_weight']
                row[f'Cluster_{cluster_id}_Weight'] = f"{cluster_weight:.1%}"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def run_portfolio_construction_examples(self) -> None:
        """Run complete portfolio construction examples."""
        print("💼 CLUSTER-BASED PORTFOLIO CONSTRUCTION")
        print("=" * 80)
        
        # Load cluster data
        if not self.load_cluster_data():
            return
        
        # Display cluster characteristics
        cluster_chars = self.get_cluster_characteristics()
        print(f"\n🎯 CLUSTER CHARACTERISTICS:")
        for cluster_id, chars in cluster_chars.items():
            print(f"   Cluster {cluster_id} ({chars['theme']}):")
            print(f"      • Size: {chars['size']} assets")
            print(f"      • Role: {chars['allocation']['role']}")
            print(f"      • Target Weight: {chars['allocation']['target_weight']}")
        
        # Construct different portfolio strategies
        conservative = self.construct_conservative_portfolio()
        balanced = self.construct_balanced_portfolio()
        growth = self.construct_growth_portfolio()
        
        # Display portfolios
        self.display_portfolio(conservative)
        self.display_portfolio(balanced)
        self.display_portfolio(growth)
        
        # Create comparison
        portfolios = [conservative, balanced, growth]
        comparison_df = self.create_portfolio_comparison(portfolios)
        
        print(f"\n📊 PORTFOLIO COMPARISON:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        # Save portfolios and comparison
        self.save_portfolio(conservative, 'conservative_portfolio.json')
        self.save_portfolio(balanced, 'balanced_portfolio.json')
        self.save_portfolio(growth, 'growth_portfolio.json')
        comparison_df.to_csv(f'{self.data_dir}/portfolio_comparison.csv', index=False)
        
        print(f"\n" + "=" * 80)
        print("✅ PORTFOLIO CONSTRUCTION EXAMPLES COMPLETE!")
        print("🎯 Three diversified portfolio strategies created using cluster analysis!")
        print("📁 Files saved for portfolio optimization implementation.")


def main():
    """Main function to run portfolio construction examples."""
    constructor = ClusterBasedPortfolioConstructor()
    constructor.run_portfolio_construction_examples()


if __name__ == "__main__":
    main() 