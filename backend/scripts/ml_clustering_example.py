#!/usr/bin/env python3
"""
Asset Clustering Example using Engineered Features

This script demonstrates how to use the engineered features for:
1. K-means clustering to group similar assets
2. Feature importance analysis
3. Cluster visualization and interpretation
4. Portfolio construction based on clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import json

warnings.filterwarnings('ignore')

def load_features():
    """Load the engineered features."""
    try:
        features = pd.read_pickle('data/asset_features.pkl')
        with open('data/feature_groups.json', 'r') as f:
            feature_groups = json.load(f)
        return features, feature_groups
    except Exception as e:
        print(f"Error loading features: {e}")
        return None, None

def select_clustering_features(features, feature_groups):
    """Select the most relevant features for clustering with proper scaling considerations."""
    
    # Core risk-return features (well-scaled)
    risk_return_features = [
        'mean_annual_return', 'annual_volatility', 'sharpe_ratio', 
        'max_drawdown', 'sortino_ratio'
    ]
    
    # Recent performance features (key for momentum clustering)
    recent_performance_features = [
        'recent_1month_return', 'recent_3month_return', 'recent_6month_return',
        'recent_3month_sharpe', 'recent_vs_historical_return', 
        'momentum_21d', 'momentum_63d', 'momentum_acceleration'
    ]
    
    # Market relationship features
    correlation_features = [
        'market_correlation', 'sector_correlation', 'mean_correlation',
        'safe_haven_score', 'correlation_dispersion'
    ]
    
    # Technical indicators (already well-scaled)
    technical_features = [
        'rsi_14', 'bollinger_position', 'trend_strength', 'vol_regime'
    ]
    
    # Sector dummies (binary, naturally scaled)
    sector_features = feature_groups['sector_dummies']
    
    # Combine all feature categories
    clustering_features = (
        risk_return_features + 
        recent_performance_features + 
        correlation_features +
        technical_features +
        sector_features
    )
    
    # Filter to available features
    available_features = [f for f in clustering_features if f in features.columns]
    selected_features = features[available_features]
    
    print(f"Selected {len(available_features)} features for clustering:")
    print(f"  • Risk-Return: {len([f for f in risk_return_features if f in available_features])}")
    print(f"  • Recent Performance: {len([f for f in recent_performance_features if f in available_features])}")
    print(f"  • Correlations: {len([f for f in correlation_features if f in available_features])}")
    print(f"  • Technical: {len([f for f in technical_features if f in available_features])}")
    print(f"  • Sector Dummies: {len([f for f in sector_features if f in available_features])}")
    
    return selected_features

def perform_clustering(features_subset, n_clusters=6):
    """Perform K-means clustering with proper feature scaling analysis."""
    
    # Analyze feature scales before standardization
    print(f"\nFeature Scale Analysis (before standardization):")
    print(f"Feature ranges:")
    for col in features_subset.columns[:10]:  # Show first 10 features
        min_val, max_val = features_subset[col].min(), features_subset[col].max()
        print(f"  {col}: [{min_val:.4f}, {max_val:.4f}] (range: {max_val-min_val:.4f})")
    
    # Standardize features (critical for clustering)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_subset)
    
    # Verify standardization
    scaled_df = pd.DataFrame(scaled_features, columns=features_subset.columns)
    print(f"\nAfter standardization (should be ~0 mean, ~1 std):")
    print(f"  Mean range: [{scaled_df.mean().min():.4f}, {scaled_df.mean().max():.4f}]")
    print(f"  Std range: [{scaled_df.std().min():.4f}, {scaled_df.std().max():.4f}]")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    
    # Analyze feature importance in clustering
    feature_importance = np.abs(kmeans.cluster_centers_).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': features_subset.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features for Clustering:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    return cluster_labels, silhouette_avg, scaler, kmeans, feature_importance_df

def visualize_clusters(features_subset, cluster_labels, asset_names):
    """Visualize clusters using PCA."""
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_subset)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': pca_features[:, 0],
        'PC2': pca_features[:, 1],
        'Cluster': cluster_labels,
        'Asset': asset_names
    })
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    for i in range(len(np.unique(cluster_labels))):
        cluster_data = plot_df[plot_df['Cluster'] == i]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                   c=colors[i], label=f'Cluster {i}', alpha=0.7, s=100)
        
        # Add asset labels
        for idx, row in cluster_data.iterrows():
            plt.annotate(row['Asset'], (row['PC1'], row['PC2']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Asset Clusters (PCA Visualization)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/asset_clusters_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return pca.explained_variance_ratio_

def analyze_clusters(features, cluster_labels, asset_names):
    """Analyze cluster characteristics."""
    # Add cluster labels to features
    clustered_features = features.copy()
    clustered_features['Cluster'] = cluster_labels
    clustered_features['Asset'] = asset_names
    
    # Calculate cluster statistics
    cluster_stats = {}
    
    key_metrics = ['mean_annual_return', 'annual_volatility', 'sharpe_ratio', 
                   'max_drawdown', 'market_correlation']
    
    for cluster_id in range(len(np.unique(cluster_labels))):
        cluster_data = clustered_features[clustered_features['Cluster'] == cluster_id]
        
        stats = {}
        for metric in key_metrics:
            if metric in cluster_data.columns:
                stats[metric] = {
                    'mean': cluster_data[metric].mean(),
                    'std': cluster_data[metric].std(),
                    'min': cluster_data[metric].min(),
                    'max': cluster_data[metric].max()
                }
        
        stats['assets'] = cluster_data['Asset'].tolist()
        stats['count'] = len(cluster_data)
        
        cluster_stats[f'Cluster_{cluster_id}'] = stats
    
    return cluster_stats, clustered_features

def create_cluster_summary(cluster_stats):
    """Create a readable cluster summary."""
    print("\n" + "="*80)
    print("ASSET CLUSTERING ANALYSIS")
    print("="*80)
    
    for cluster_name, stats in cluster_stats.items():
        print(f"\n{cluster_name.upper()}:")
        print("-" * 40)
        print(f"Assets ({stats['count']}): {', '.join(stats['assets'])}")
        
        if 'mean_annual_return' in stats:
            print(f"Avg Annual Return: {stats['mean_annual_return']['mean']:.1%} ± {stats['mean_annual_return']['std']:.1%}")
        if 'annual_volatility' in stats:
            print(f"Avg Volatility: {stats['annual_volatility']['mean']:.1%} ± {stats['annual_volatility']['std']:.1%}")
        if 'sharpe_ratio' in stats:
            print(f"Avg Sharpe Ratio: {stats['sharpe_ratio']['mean']:.3f} ± {stats['sharpe_ratio']['std']:.3f}")
        if 'market_correlation' in stats:
            print(f"Market Correlation: {stats['market_correlation']['mean']:.3f} ± {stats['market_correlation']['std']:.3f}")

def analyze_recent_performance_by_cluster(clustered_features):
    """Analyze recent performance patterns by cluster."""
    print(f"\n" + "="*70)
    print("RECENT PERFORMANCE ANALYSIS BY CLUSTER")
    print("="*70)
    
    recent_features = ['recent_1month_return', 'recent_3month_return', 'recent_6month_return']
    available_recent = [f for f in recent_features if f in clustered_features.columns]
    
    if not available_recent:
        print("No recent performance features available.")
        return
    
    for cluster_id in sorted(clustered_features['Cluster'].unique()):
        cluster_data = clustered_features[clustered_features['Cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} Recent Performance:")
        print(f"Assets: {', '.join(cluster_data['Asset'].tolist()[:5])}{'...' if len(cluster_data) > 5 else ''}")
        
        for feature in available_recent:
            if feature in cluster_data.columns:
                mean_perf = cluster_data[feature].mean()
                std_perf = cluster_data[feature].std()
                best_asset = cluster_data.loc[cluster_data[feature].idxmax(), 'Asset']
                best_perf = cluster_data[feature].max()
                print(f"  {feature.replace('_', ' ').title()}: {mean_perf:.1%} ± {std_perf:.1%} (best: {best_asset} at {best_perf:.1%})")

def suggest_portfolio_allocation(cluster_stats):
    """Suggest portfolio allocation based on clusters."""
    print(f"\n" + "="*60)
    print("PORTFOLIO ALLOCATION SUGGESTIONS")
    print("="*60)
    
    # Identify cluster characteristics
    suggestions = {}
    
    for cluster_name, stats in cluster_stats.items():
        if 'sharpe_ratio' in stats and 'annual_volatility' in stats:
            avg_sharpe = stats['sharpe_ratio']['mean']
            avg_vol = stats['annual_volatility']['mean']
            
            if avg_sharpe > 0.8 and avg_vol < 0.25:
                suggestions[cluster_name] = {
                    'allocation': '25-30%',
                    'role': 'Core Holdings (High quality, moderate risk)',
                    'rationale': 'Strong risk-adjusted returns with manageable volatility'
                }
            elif avg_sharpe > 1.0:
                suggestions[cluster_name] = {
                    'allocation': '15-20%',
                    'role': 'Growth Holdings (High Sharpe ratio)',
                    'rationale': 'Excellent risk-adjusted performance'
                }
            elif avg_vol < 0.2:
                suggestions[cluster_name] = {
                    'allocation': '20-25%',
                    'role': 'Defensive Holdings (Low volatility)',
                    'rationale': 'Portfolio stability and downside protection'
                }
            elif avg_vol > 0.4:
                suggestions[cluster_name] = {
                    'allocation': '5-10%',
                    'role': 'Satellite Holdings (High risk/reward)',
                    'rationale': 'Portfolio enhancement with limited allocation'
                }
            else:
                suggestions[cluster_name] = {
                    'allocation': '10-15%',
                    'role': 'Diversifier',
                    'rationale': 'Balanced risk-return profile'
                }
    
    for cluster_name, suggestion in suggestions.items():
        print(f"\n{cluster_name}:")
        print(f"  Role: {suggestion['role']}")
        print(f"  Suggested Allocation: {suggestion['allocation']}")
        print(f"  Rationale: {suggestion['rationale']}")

def main():
    """Main clustering analysis."""
    print("Asset Clustering Analysis using Engineered Features")
    print("=" * 60)
    
    # Load features
    features, feature_groups = load_features()
    if features is None:
        print("Failed to load features. Run feature_engineering.py first.")
        return
    
    print(f"Loaded {features.shape[0]} assets with {features.shape[1]} features")
    
    # Select clustering features
    features_subset = select_clustering_features(features, feature_groups)
    print(f"Selected {features_subset.shape[1]} features for clustering")
    
    # Determine optimal number of clusters
    print("\nDetermining optimal number of clusters...")
    silhouette_scores = []
    k_range = range(3, 8)
    
    for k in k_range:
        _, score, _, _, _ = perform_clustering(features_subset, k)
        silhouette_scores.append(score)
        print(f"k={k}: Silhouette Score = {score:.3f}")
    
    # Use best k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Perform final clustering
    cluster_labels, silhouette_avg, scaler, kmeans, feature_importance = perform_clustering(features_subset, optimal_k)
    
    print(f"Final clustering silhouette score: {silhouette_avg:.3f}")
    
    # Visualize clusters
    print("\nCreating cluster visualizations...")
    explained_var = visualize_clusters(features_subset, cluster_labels, features.index)
    print(f"PCA explains {sum(explained_var):.1%} of variance")
    
    # Analyze clusters
    print("\nAnalyzing cluster characteristics...")
    cluster_stats, clustered_features = analyze_clusters(features, cluster_labels, features.index)
    
    # Analyze recent performance by cluster
    analyze_recent_performance_by_cluster(clustered_features)
    
    # Create summary
    create_cluster_summary(cluster_stats)
    
    # Portfolio suggestions
    suggest_portfolio_allocation(cluster_stats)
    
    # Save results
    print(f"\nSaving clustering results...")
    clustered_features.to_csv('data/clustered_assets.csv')
    
    with open('data/cluster_analysis.json', 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        serializable_stats = {}
        for cluster, stats in cluster_stats.items():
            serializable_stats[cluster] = {}
            for key, value in stats.items():
                if key == 'assets':
                    serializable_stats[cluster][key] = value
                elif key == 'count':
                    serializable_stats[cluster][key] = int(value)
                elif isinstance(value, dict):
                    serializable_stats[cluster][key] = {k: float(v) for k, v in value.items()}
                else:
                    serializable_stats[cluster][key] = value
        
        json.dump(serializable_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE!")
    print("Files created:")
    print("• data/clustered_assets.csv - Assets with cluster assignments")
    print("• data/cluster_analysis.json - Detailed cluster statistics")
    print("• data/asset_clusters_pca.png - Cluster visualization")
    print("\nUse these clusters for:")
    print("• Portfolio diversification across clusters")
    print("• Risk management by cluster characteristics")
    print("• Strategic asset allocation decisions")
    print("="*60)

if __name__ == "__main__":
    main() 