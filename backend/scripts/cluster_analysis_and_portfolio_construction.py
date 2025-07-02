#!/usr/bin/env python3
"""
Comprehensive Cluster Analysis and Portfolio Construction

This script analyzes the clusters formed by K-means clustering to understand:
1. What clusters represent (sectors, growth vs value, etc.)
2. Cluster composition and characteristics 
3. Sector distribution within clusters
4. Growth vs value classification
5. Portfolio construction guidelines using clusters
6. Diversification strategies to avoid similar assets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class ClusterAnalyzer:
    """Comprehensive cluster analysis for portfolio construction."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the analyzer."""
        self.data_dir = data_dir
        self.features = None
        self.clustered_data = None
        self.cluster_analysis = None
        
        # Asset categorization for analysis
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
        
        # Growth vs Value classification based on characteristics
        self.growth_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMZN']
        self.value_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'V', 'XOM', 'CVX', 'COP']
        self.defensive_stocks = ['JNJ', 'PFE', 'UNH', 'PG', 'KO', 'PEP', 'WMT', 'NEE']
        
    def load_data(self) -> bool:
        """Load clustered data and features."""
        try:
            print("📊 Loading clustered data and features...")
            
            # Load clustered data - asset names are in the index
            self.clustered_data = pd.read_csv(f'{self.data_dir}/clustered_assets.csv', index_col=0)
            
            # Create asset column from index for easier processing
            self.clustered_data['Asset'] = self.clustered_data.index
            
            # Load features
            self.features = pd.read_pickle(f'{self.data_dir}/asset_features.pkl')
            
            # Load cluster analysis if available
            try:
                with open(f'{self.data_dir}/advanced_kmeans_analysis.json', 'r') as f:
                    self.cluster_analysis = json.load(f)
            except:
                self.cluster_analysis = None
            
            print(f"✅ Loaded data for {len(self.clustered_data)} assets")
            print(f"   • Clusters found: {len(self.clustered_data['Cluster'].unique())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_cluster_composition(self) -> Dict:
        """Analyze what each cluster represents."""
        print("\n🔍 ANALYZING CLUSTER COMPOSITION")
        print("=" * 60)
        
        composition_analysis = {}
        
        for cluster_id in sorted(self.clustered_data['Cluster'].unique()):
            cluster_assets = self.clustered_data[self.clustered_data['Cluster'] == cluster_id]['Asset'].tolist()
            
            print(f"\n📈 CLUSTER {cluster_id} ANALYSIS:")
            print("-" * 40)
            print(f"Assets ({len(cluster_assets)}): {', '.join(cluster_assets[:10])}{'...' if len(cluster_assets) > 10 else ''}")
            
            # Sector distribution
            sector_counts = {}
            for asset in cluster_assets:
                sector = self.asset_sectors.get(asset, 'Unknown')
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            print(f"\n🏢 Sector Distribution:")
            for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(cluster_assets)) * 100
                print(f"   • {sector}: {count} assets ({percentage:.1f}%)")
            
            # Growth vs Value analysis
            growth_count = sum(1 for asset in cluster_assets if asset in self.growth_stocks)
            value_count = sum(1 for asset in cluster_assets if asset in self.value_stocks)
            defensive_count = sum(1 for asset in cluster_assets if asset in self.defensive_stocks)
            
            print(f"\n📊 Style Distribution:")
            print(f"   • Growth stocks: {growth_count} ({growth_count/len(cluster_assets)*100:.1f}%)")
            print(f"   • Value stocks: {value_count} ({value_count/len(cluster_assets)*100:.1f}%)")
            print(f"   • Defensive stocks: {defensive_count} ({defensive_count/len(cluster_assets)*100:.1f}%)")
            
            # Performance characteristics
            cluster_features = self.features.loc[cluster_assets]
            
            key_metrics = {
                'mean_annual_return': 'Annual Return',
                'annual_volatility': 'Annual Volatility', 
                'sharpe_ratio': 'Sharpe Ratio',
                'recent_3month_return': 'Recent 3M Return',
                'market_correlation': 'Market Correlation'
            }
            
            print(f"\n📈 Performance Characteristics:")
            for metric, label in key_metrics.items():
                if metric in cluster_features.columns:
                    mean_val = cluster_features[metric].mean()
                    if 'return' in metric or 'volatility' in metric:
                        print(f"   • {label}: {mean_val:.1%}")
                    else:
                        print(f"   • {label}: {mean_val:.3f}")
            
            # Determine cluster theme
            cluster_theme = self._determine_cluster_theme(cluster_assets, sector_counts, 
                                                        growth_count, value_count, defensive_count)
            print(f"\n🎯 Cluster Theme: {cluster_theme}")
            
            # Store analysis
            composition_analysis[f'Cluster_{cluster_id}'] = {
                'assets': cluster_assets,
                'size': len(cluster_assets),
                'sectors': sector_counts,
                'style_distribution': {
                    'growth': growth_count,
                    'value': value_count,
                    'defensive': defensive_count
                },
                'theme': cluster_theme,
                'performance_summary': {
                    metric: float(cluster_features[metric].mean()) 
                    for metric in key_metrics.keys() 
                    if metric in cluster_features.columns
                }
            }
        
        return composition_analysis
    
    def _determine_cluster_theme(self, assets: List[str], sectors: Dict, 
                               growth_count: int, value_count: int, defensive_count: int) -> str:
        """Determine the main theme/characteristic of a cluster."""
        total_assets = len(assets)
        
        # Check if dominated by specific sector
        dominant_sector = max(sectors.items(), key=lambda x: x[1])
        if dominant_sector[1] / total_assets > 0.4:  # >40% in one sector
            return f"{dominant_sector[0]}-Dominated"
        
        # Check if dominated by style
        if growth_count / total_assets > 0.5:
            return "Growth-Oriented"
        elif value_count / total_assets > 0.4:
            return "Value-Oriented"
        elif defensive_count / total_assets > 0.5:
            return "Defensive/Dividend"
        
        # Check if ETF-heavy
        etf_count = sum(1 for asset in assets if asset in ['SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'EEM', 'TLT', 'GLD', 'VNQ'])
        if etf_count / total_assets > 0.3:
            return "ETF/Diversified"
        
        # Check performance characteristics
        if 'recent_3month_return' in self.features.columns:
            cluster_features = self.features.loc[assets]
            recent_performance = cluster_features['recent_3month_return'].mean()
            volatility = cluster_features['annual_volatility'].mean()
            
            if recent_performance > 0.1 and volatility > 0.25:
                return "High-Momentum/High-Risk"
            elif recent_performance < 0 and volatility < 0.25:
                return "Low-Momentum/Low-Risk"
        
        return "Mixed/Diversified"
    
    def analyze_sector_clusters(self) -> None:
        """Analyze how sectors are distributed across clusters."""
        print(f"\n🏢 SECTOR DISTRIBUTION ACROSS CLUSTERS")
        print("=" * 60)
        
        # Create sector-cluster matrix
        sector_cluster_matrix = {}
        
        for asset, sector in self.asset_sectors.items():
            if asset in self.clustered_data['Asset'].values:
                cluster = self.clustered_data[self.clustered_data['Asset'] == asset]['Cluster'].iloc[0]
                
                if sector not in sector_cluster_matrix:
                    sector_cluster_matrix[sector] = {}
                
                if cluster not in sector_cluster_matrix[sector]:
                    sector_cluster_matrix[sector][cluster] = 0
                
                sector_cluster_matrix[sector][cluster] += 1
        
        # Display analysis
        for sector, clusters in sector_cluster_matrix.items():
            total_assets = sum(clusters.values())
            print(f"\n📊 {sector} ({total_assets} assets):")
            
            for cluster_id in sorted(clusters.keys()):
                count = clusters[cluster_id]
                percentage = (count / total_assets) * 100
                print(f"   • Cluster {cluster_id}: {count} assets ({percentage:.1f}%)")
        
        # Identify sector concentration insights
        print(f"\n🎯 SECTOR CONCENTRATION INSIGHTS:")
        
        concentrated_sectors = []
        diversified_sectors = []
        
        for sector, clusters in sector_cluster_matrix.items():
            total_assets = sum(clusters.values())
            max_cluster_count = max(clusters.values())
            concentration = max_cluster_count / total_assets
            
            if concentration > 0.7:  # >70% in one cluster
                dominant_cluster = max(clusters.items(), key=lambda x: x[1])[0]
                concentrated_sectors.append((sector, dominant_cluster, concentration))
            elif len(clusters) >= 2:
                diversified_sectors.append(sector)
        
        if concentrated_sectors:
            print(f"   • Concentrated sectors (>70% in one cluster):")
            for sector, cluster, conc in concentrated_sectors:
                print(f"     - {sector}: {conc:.1%} in Cluster {cluster}")
        
        if diversified_sectors:
            print(f"   • Well-diversified sectors across clusters:")
            for sector in diversified_sectors:
                print(f"     - {sector}")
    
    def create_cluster_visualization(self) -> None:
        """Create comprehensive cluster visualization."""
        print(f"\n📊 Creating cluster visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cluster sizes
        cluster_sizes = self.clustered_data['Cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'][:len(cluster_sizes)])
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Assets')
        axes[0, 0].set_title('Assets per Cluster')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Sector distribution heatmap
        sector_data = []
        for asset in self.clustered_data['Asset']:
            sector = self.asset_sectors.get(asset, 'Unknown')
            cluster = self.clustered_data[self.clustered_data['Asset'] == asset]['Cluster'].iloc[0]
            sector_data.append({'Asset': asset, 'Sector': sector, 'Cluster': cluster})
        
        sector_df = pd.DataFrame(sector_data)
        sector_cluster_counts = sector_df.groupby(['Sector', 'Cluster']).size().unstack(fill_value=0)
        
        sns.heatmap(sector_cluster_counts, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
        axes[0, 1].set_title('Sector Distribution Across Clusters')
        axes[0, 1].set_xlabel('Cluster')
        
        # Plot 3: Performance by cluster
        performance_data = []
        for cluster_id in self.clustered_data['Cluster'].unique():
            cluster_assets = self.clustered_data[self.clustered_data['Cluster'] == cluster_id]['Asset'].tolist()
            cluster_features = self.features.loc[cluster_assets]
            
            if 'mean_annual_return' in cluster_features.columns and 'annual_volatility' in cluster_features.columns:
                performance_data.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Return': cluster_features['mean_annual_return'].mean(),
                    'Volatility': cluster_features['annual_volatility'].mean()
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            scatter = axes[1, 0].scatter(perf_df['Volatility'], perf_df['Return'], 
                                       c=range(len(perf_df)), cmap='viridis', s=100)
            
            for i, row in perf_df.iterrows():
                axes[1, 0].annotate(row['Cluster'], (row['Volatility'], row['Return']), 
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[1, 0].set_xlabel('Annual Volatility')
            axes[1, 0].set_ylabel('Annual Return')
            axes[1, 0].set_title('Risk-Return by Cluster')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Recent performance by cluster
        if 'recent_3month_return' in self.features.columns:
            recent_performance = []
            for cluster_id in self.clustered_data['Cluster'].unique():
                cluster_assets = self.clustered_data[self.clustered_data['Cluster'] == cluster_id]['Asset'].tolist()
                cluster_features = self.features.loc[cluster_assets]
                recent_performance.append(cluster_features['recent_3month_return'].mean())
            
            axes[1, 1].bar(range(len(recent_performance)), recent_performance, 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'orange'][:len(recent_performance)])
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Recent 3-Month Return')
            axes[1, 1].set_title('Recent Performance by Cluster')
            axes[1, 1].set_xticks(range(len(recent_performance)))
            axes[1, 1].set_xticklabels([f'Cluster {i}' for i in range(len(recent_performance))])
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add horizontal line at 0
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/cluster_composition_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Cluster visualization saved to {self.data_dir}/cluster_composition_analysis.png")
    
    def generate_portfolio_construction_guidelines(self, composition_analysis: Dict) -> Dict:
        """Generate portfolio construction guidelines using cluster analysis."""
        print(f"\n💼 PORTFOLIO CONSTRUCTION GUIDELINES")
        print("=" * 60)
        
        guidelines = {
            'diversification_strategy': {},
            'cluster_allocation_framework': {},
            'asset_selection_rules': {},
            'risk_management_guidelines': {}
        }
        
        # Analyze cluster characteristics for allocation
        cluster_allocations = {}
        
        for cluster_name, analysis in composition_analysis.items():
            theme = analysis['theme']
            performance = analysis['performance_summary']
            size = analysis['size']
            
            # Determine allocation strategy
            if 'Growth' in theme or 'High-Momentum' in theme:
                allocation = {
                    'target_weight': '15-25%',
                    'role': 'Growth Engine',
                    'rationale': 'High-performing assets for portfolio growth',
                    'risk_level': 'High',
                    'selection_criteria': 'Pick top 3-5 assets by Sharpe ratio'
                }
            elif 'Defensive' in theme or 'Low-Risk' in theme:
                allocation = {
                    'target_weight': '20-30%',
                    'role': 'Stability Core',
                    'rationale': 'Defensive assets for downside protection',
                    'risk_level': 'Low',
                    'selection_criteria': 'Pick assets with lowest volatility and max drawdown'
                }
            elif 'ETF' in theme or 'Diversified' in theme:
                allocation = {
                    'target_weight': '25-35%',
                    'role': 'Core Holdings',
                    'rationale': 'Broad market exposure and diversification',
                    'risk_level': 'Moderate',
                    'selection_criteria': 'Pick broad market ETFs (SPY, VTI) and sector ETFs'
                }
            elif 'Value' in theme:
                allocation = {
                    'target_weight': '10-20%',
                    'role': 'Value Play',
                    'rationale': 'Undervalued assets with recovery potential',
                    'risk_level': 'Moderate',
                    'selection_criteria': 'Pick assets with low P/E ratios and strong fundamentals'
                }
            else:
                allocation = {
                    'target_weight': '5-15%',
                    'role': 'Tactical Allocation',
                    'rationale': 'Specialized exposure and diversification',
                    'risk_level': 'Variable',
                    'selection_criteria': 'Pick 1-2 representative assets'
                }
            
            cluster_allocations[cluster_name] = allocation
            
            print(f"\n🎯 {cluster_name} ({theme}):")
            print(f"   • Target Weight: {allocation['target_weight']}")
            print(f"   • Role: {allocation['role']}")
            print(f"   • Selection: {allocation['selection_criteria']}")
        
        guidelines['cluster_allocation_framework'] = cluster_allocations
        
        # Diversification rules
        print(f"\n🔄 DIVERSIFICATION RULES:")
        
        diversification_rules = [
            "Never allocate >40% to any single cluster",
            "Ensure representation from at least 3 different clusters",
            "Limit sector concentration to <30% of total portfolio",
            "Balance growth and defensive allocations (60/40 to 40/60 range)",
            "Include at least one ETF for broad market exposure",
            "Monitor cluster performance and rebalance quarterly"
        ]
        
        for i, rule in enumerate(diversification_rules, 1):
            print(f"   {i}. {rule}")
        
        guidelines['diversification_strategy']['rules'] = diversification_rules
        
        # Asset selection guidelines
        print(f"\n🎯 ASSET SELECTION GUIDELINES:")
        
        selection_guidelines = {
            'within_cluster_selection': [
                "Rank assets by Sharpe ratio within each cluster",
                "Consider correlation - avoid highly correlated assets (>0.8)",
                "Prefer liquid assets (large cap stocks and major ETFs)",
                "Check recent momentum - recent 3M performance indicator"
            ],
            'cross_cluster_balance': [
                "Ensure no cluster dominates portfolio (max 35%)",
                "Balance high-volatility and low-volatility clusters",
                "Include both momentum and defensive characteristics",
                "Consider sector exposure across all selections"
            ]
        }
        
        for category, rules in selection_guidelines.items():
            print(f"   📊 {category.replace('_', ' ').title()}:")
            for rule in rules:
                print(f"      • {rule}")
        
        guidelines['asset_selection_rules'] = selection_guidelines
        
        # Risk management guidelines
        print(f"\n⚠️  RISK MANAGEMENT GUIDELINES:")
        
        risk_guidelines = [
            "Monitor cluster performance shifts monthly",
            "Rebalance if any cluster deviates >5% from target",
            "Set stop-losses for high-momentum cluster assets",
            "Increase defensive allocation during market stress",
            "Use cluster correlation for portfolio VaR calculation"
        ]
        
        for i, guideline in enumerate(risk_guidelines, 1):
            print(f"   {i}. {guideline}")
        
        guidelines['risk_management_guidelines'] = risk_guidelines
        
        return guidelines
    
    def create_asset_selection_framework(self) -> pd.DataFrame:
        """Create a framework for selecting assets from each cluster."""
        print(f"\n🎯 CREATING ASSET SELECTION FRAMEWORK")
        print("=" * 60)
        
        selection_framework = []
        
        for cluster_id in sorted(self.clustered_data['Cluster'].unique()):
            cluster_assets = self.clustered_data[self.clustered_data['Cluster'] == cluster_id]['Asset'].tolist()
            cluster_features = self.features.loc[cluster_assets]
            
            # Calculate selection scores
            for asset in cluster_assets:
                asset_data = cluster_features.loc[asset]
                
                # Selection score based on multiple factors
                score_components = {}
                
                if 'sharpe_ratio' in asset_data:
                    score_components['sharpe_score'] = min(asset_data['sharpe_ratio'] / 2.0, 1.0)  # Cap at 1.0
                
                if 'annual_volatility' in asset_data:
                    # Lower volatility = higher score (inverted)
                    score_components['stability_score'] = max(0, 1 - asset_data['annual_volatility'] / 0.6)
                
                if 'recent_3month_return' in asset_data:
                    # Recent momentum score
                    score_components['momentum_score'] = min(max(asset_data['recent_3month_return'] / 0.2 + 0.5, 0), 1)
                
                if 'market_correlation' in asset_data:
                    # Diversification score (lower correlation = higher score for diversification)
                    score_components['diversification_score'] = max(0, 1 - abs(asset_data['market_correlation']))
                
                # Calculate overall selection score
                overall_score = np.mean(list(score_components.values()))
                
                # Asset characteristics
                sector = self.asset_sectors.get(asset, 'Unknown')
                asset_type = 'Growth' if asset in self.growth_stocks else \
                           'Value' if asset in self.value_stocks else \
                           'Defensive' if asset in self.defensive_stocks else 'Other'
                
                selection_framework.append({
                    'Asset': asset,
                    'Cluster': cluster_id,
                    'Sector': sector,
                    'Type': asset_type,
                    'Selection_Score': overall_score,
                    'Sharpe_Ratio': asset_data.get('sharpe_ratio', 0),
                    'Annual_Volatility': asset_data.get('annual_volatility', 0),
                    'Recent_3M_Return': asset_data.get('recent_3month_return', 0),
                    'Market_Correlation': asset_data.get('market_correlation', 0),
                    'Recommendation': self._get_recommendation(overall_score, cluster_id)
                })
        
        selection_df = pd.DataFrame(selection_framework)
        selection_df = selection_df.sort_values(['Cluster', 'Selection_Score'], ascending=[True, False])
        
        # Display top picks from each cluster
        print(f"\n🏆 TOP PICKS FROM EACH CLUSTER:")
        
        for cluster_id in sorted(selection_df['Cluster'].unique()):
            cluster_data = selection_df[selection_df['Cluster'] == cluster_id]
            top_picks = cluster_data.head(3)
            
            print(f"\n   Cluster {cluster_id} - Top 3 Picks:")
            for _, asset in top_picks.iterrows():
                print(f"      1. {asset['Asset']} ({asset['Sector']}) - Score: {asset['Selection_Score']:.3f}")
                print(f"         Sharpe: {asset['Sharpe_Ratio']:.3f}, Vol: {asset['Annual_Volatility']:.1%}, "
                      f"Recent 3M: {asset['Recent_3M_Return']:.1%}")
        
        return selection_df
    
    def _get_recommendation(self, score: float, cluster_id: int) -> str:
        """Get recommendation based on selection score."""
        if score > 0.7:
            return "Strong Buy"
        elif score > 0.6:
            return "Buy"
        elif score > 0.5:
            return "Hold"
        elif score > 0.4:
            return "Cautious"
        else:
            return "Avoid"
    
    def save_cluster_labels_and_analysis(self, composition_analysis: Dict, 
                                       guidelines: Dict, selection_df: pd.DataFrame) -> None:
        """Save all cluster analysis results."""
        print(f"\n💾 SAVING CLUSTER ANALYSIS RESULTS")
        print("=" * 60)
        
        # Save cluster labels for portfolio construction
        cluster_labels = self.clustered_data[['Asset', 'Cluster']].copy()
        
        # Add sector and type information
        cluster_labels['Sector'] = cluster_labels['Asset'].map(self.asset_sectors)
        cluster_labels['Asset_Type'] = cluster_labels['Asset'].apply(
            lambda x: 'Growth' if x in self.growth_stocks else
                     'Value' if x in self.value_stocks else
                     'Defensive' if x in self.defensive_stocks else 'Other'
        )
        
        # Save cluster labels
        cluster_labels.to_csv(f'{self.data_dir}/cluster_labels_for_portfolio.csv', index=False)
        
        # Save asset selection framework
        selection_df.to_csv(f'{self.data_dir}/asset_selection_framework.csv', index=False)
        
        # Save comprehensive analysis
        comprehensive_analysis = {
            'cluster_composition_analysis': composition_analysis,
            'portfolio_construction_guidelines': guidelines,
            'summary_statistics': {
                'total_assets': len(self.clustered_data),
                'num_clusters': len(self.clustered_data['Cluster'].unique()),
                'sectors_represented': len(set(self.asset_sectors.values())),
                'growth_stocks_count': len([a for a in cluster_labels['Asset'] if a in self.growth_stocks]),
                'value_stocks_count': len([a for a in cluster_labels['Asset'] if a in self.value_stocks]),
                'defensive_stocks_count': len([a for a in cluster_labels['Asset'] if a in self.defensive_stocks])
            }
        }
        
        # Save to JSON
        with open(f'{self.data_dir}/comprehensive_cluster_analysis.json', 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2)
        
        print(f"✅ Results saved:")
        print(f"   • {self.data_dir}/cluster_labels_for_portfolio.csv")
        print(f"   • {self.data_dir}/asset_selection_framework.csv") 
        print(f"   • {self.data_dir}/comprehensive_cluster_analysis.json")
        print(f"   • {self.data_dir}/cluster_composition_analysis.png")
        
        # Display file purposes
        print(f"\n📋 FILE PURPOSES:")
        print(f"   • cluster_labels_for_portfolio.csv: Asset-cluster mapping for portfolio optimization")
        print(f"   • asset_selection_framework.csv: Scored assets for selection within clusters")
        print(f"   • comprehensive_cluster_analysis.json: Complete analysis for strategy development")
        print(f"   • cluster_composition_analysis.png: Visual analysis of cluster characteristics")
    
    def run_complete_analysis(self) -> None:
        """Run complete cluster analysis."""
        print("🔍 COMPREHENSIVE CLUSTER ANALYSIS AND PORTFOLIO CONSTRUCTION")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        # Analyze cluster composition
        composition_analysis = self.analyze_cluster_composition()
        
        # Analyze sector distribution
        self.analyze_sector_clusters()
        
        # Create visualizations
        self.create_cluster_visualization()
        
        # Generate portfolio construction guidelines
        guidelines = self.generate_portfolio_construction_guidelines(composition_analysis)
        
        # Create asset selection framework
        selection_df = self.create_asset_selection_framework()
        
        # Save all results
        self.save_cluster_labels_and_analysis(composition_analysis, guidelines, selection_df)
        
        print(f"\n" + "=" * 80)
        print("✅ COMPREHENSIVE CLUSTER ANALYSIS COMPLETE!")
        print("🎯 Ready for portfolio optimization using cluster-based diversification!")


def main():
    """Main function to run cluster analysis."""
    analyzer = ClusterAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 