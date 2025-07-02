#!/usr/bin/env python3
"""
Advanced K-Means Clustering for Portfolio Optimization

This script implements comprehensive K-Means clustering with:
1. Elbow method for optimal k selection
2. Multiple evaluation metrics (silhouette, inertia, Calinski-Harabasz)
3. Advanced dimensionality reduction (PCA, t-SNE)
4. Cluster interpretability analysis
5. Portfolio diversification strategies based on clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from kneed import KneeLocator
import warnings
import json
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

class AdvancedKMeansAnalyzer:
    """Advanced K-Means clustering analyzer for portfolio optimization."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the analyzer."""
        self.data_dir = data_dir
        self.features = None
        self.scaled_features = None
        self.scaler = None
        self.feature_groups = None
        self.clustering_results = {}
        
    def load_data(self) -> bool:
        """Load feature data."""
        try:
            print("📊 Loading feature data...")
            self.features = pd.read_pickle(f'{self.data_dir}/asset_features.pkl')
            
            with open(f'{self.data_dir}/feature_groups.json', 'r') as f:
                self.feature_groups = json.load(f)
            
            print(f"✅ Loaded {self.features.shape[0]} assets with {self.features.shape[1]} features")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def prepare_clustering_features(self) -> pd.DataFrame:
        """Prepare and select features for clustering."""
        print("\n🔧 Preparing features for clustering...")
        
        # Select diverse feature categories for clustering
        clustering_features = []
        
        # Core risk-return metrics
        core_features = [
            'mean_annual_return', 'annual_volatility', 'sharpe_ratio',
            'max_drawdown', 'sortino_ratio'
        ]
        clustering_features.extend(core_features)
        
        # Recent performance (momentum clustering)
        momentum_features = [
            'recent_1month_return', 'recent_3month_return', 'recent_6month_return',
            'momentum_21d', 'momentum_63d', 'momentum_acceleration'
        ]
        clustering_features.extend(momentum_features)
        
        # Market relationships
        correlation_features = [
            'market_correlation', 'sector_correlation', 'safe_haven_score',
            'mean_correlation', 'correlation_dispersion'
        ]
        clustering_features.extend(correlation_features)
        
        # Technical indicators
        technical_features = [
            'rsi_14', 'bollinger_position', 'trend_strength', 'vol_regime'
        ]
        clustering_features.extend(technical_features)
        
        # Rolling statistics (select key ones)
        rolling_features = [
            'rolling_mean_63d', 'rolling_std_63d', 'rolling_sharpe_63d',
            'vol_stability_63d'
        ]
        clustering_features.extend(rolling_features)
        
        # Sector representation (reduce dimensionality)
        sector_features = ['is_etf', 'is_tech', 'is_defensive', 'is_cyclical', 'is_commodity']
        clustering_features.extend(sector_features)
        
        # Filter available features
        available_features = [f for f in clustering_features if f in self.features.columns]
        selected_features = self.features[available_features]
        
        print(f"✅ Selected {len(available_features)} features for clustering")
        print(f"   • Core risk-return: {len([f for f in core_features if f in available_features])}")
        print(f"   • Momentum: {len([f for f in momentum_features if f in available_features])}")
        print(f"   • Correlations: {len([f for f in correlation_features if f in available_features])}")
        print(f"   • Technical: {len([f for f in technical_features if f in available_features])}")
        print(f"   • Rolling stats: {len([f for f in rolling_features if f in available_features])}")
        print(f"   • Sector: {len([f for f in sector_features if f in available_features])}")
        
        return selected_features
    
    def standardize_features(self, features: pd.DataFrame) -> np.ndarray:
        """Standardize features for clustering."""
        print("\n⚖️  Standardizing features...")
        
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(features)
        
        # Verification
        scaled_df = pd.DataFrame(self.scaled_features, columns=features.columns)
        print(f"✅ Standardization complete:")
        print(f"   • Mean range: [{scaled_df.mean().min():.4f}, {scaled_df.mean().max():.4f}]")
        print(f"   • Std range: [{scaled_df.std().min():.4f}, {scaled_df.std().max():.4f}]")
        
        return self.scaled_features
    
    def elbow_method_analysis(self, k_range: range = range(2, 15)) -> Dict:
        """Perform elbow method analysis with multiple metrics."""
        print(f"\n📈 Performing elbow method analysis (k={k_range.start} to {k_range.stop-1})...")
        
        inertias = []
        silhouette_scores = []
        calinski_harabasz_scores = []
        k_values = list(k_range)
        
        for k in k_values:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(self.scaled_features, labels)
            calinski = calinski_harabasz_score(self.scaled_features, labels)
            
            inertias.append(inertia)
            silhouette_scores.append(silhouette)
            calinski_harabasz_scores.append(calinski)
            
            print(f"   k={k}: Inertia={inertia:.0f}, Silhouette={silhouette:.3f}, Calinski-Harabasz={calinski:.1f}")
        
        # Find optimal k using different methods
        results = {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': calinski_harabasz_scores
        }
        
        # Elbow method for inertia
        try:
            knee_locator = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
            optimal_k_elbow = knee_locator.elbow
        except:
            optimal_k_elbow = None
        
        # Best silhouette score
        optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
        
        # Best Calinski-Harabasz score
        optimal_k_calinski = k_values[np.argmax(calinski_harabasz_scores)]
        
        results.update({
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'optimal_k_calinski': optimal_k_calinski
        })
        
        print(f"\n🎯 Optimal k suggestions:")
        print(f"   • Elbow method: {optimal_k_elbow}")
        print(f"   • Best silhouette: {optimal_k_silhouette} (score: {max(silhouette_scores):.3f})")
        print(f"   • Best Calinski-Harabasz: {optimal_k_calinski} (score: {max(calinski_harabasz_scores):.1f})")
        
        return results
    
    def plot_elbow_analysis(self, results: Dict) -> None:
        """Plot elbow method results."""
        print("\n📊 Creating elbow analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Inertia (Elbow method)
        axes[0, 0].plot(results['k_values'], results['inertias'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia (Within-cluster sum of squares)')
        axes[0, 0].set_title('Elbow Method for Optimal k')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Highlight optimal k if found
        if results['optimal_k_elbow']:
            axes[0, 0].axvline(x=results['optimal_k_elbow'], color='red', linestyle='--', 
                              label=f'Optimal k = {results["optimal_k_elbow"]}')
            axes[0, 0].legend()
        
        # Plot 2: Silhouette scores
        axes[0, 1].plot(results['k_values'], results['silhouette_scores'], 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score vs Number of Clusters')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Highlight best silhouette
        best_k = results['optimal_k_silhouette']
        axes[0, 1].axvline(x=best_k, color='red', linestyle='--', 
                          label=f'Best k = {best_k}')
        axes[0, 1].legend()
        
        # Plot 3: Calinski-Harabasz scores
        axes[1, 0].plot(results['k_values'], results['calinski_harabasz_scores'], 'mo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Score vs Number of Clusters')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Highlight best Calinski-Harabasz
        best_k_ch = results['optimal_k_calinski']
        axes[1, 0].axvline(x=best_k_ch, color='red', linestyle='--', 
                          label=f'Best k = {best_k_ch}')
        axes[1, 0].legend()
        
        # Plot 4: Combined normalized scores
        # Normalize scores to 0-1 range for comparison
        norm_silhouette = (np.array(results['silhouette_scores']) - min(results['silhouette_scores'])) / \
                         (max(results['silhouette_scores']) - min(results['silhouette_scores']))
        norm_calinski = (np.array(results['calinski_harabasz_scores']) - min(results['calinski_harabasz_scores'])) / \
                       (max(results['calinski_harabasz_scores']) - min(results['calinski_harabasz_scores']))
        norm_inertia = 1 - (np.array(results['inertias']) - min(results['inertias'])) / \
                      (max(results['inertias']) - min(results['inertias']))  # Invert since lower is better
        
        axes[1, 1].plot(results['k_values'], norm_silhouette, 'g-', label='Silhouette (normalized)', linewidth=2)
        axes[1, 1].plot(results['k_values'], norm_calinski, 'm-', label='Calinski-Harabasz (normalized)', linewidth=2)
        axes[1, 1].plot(results['k_values'], norm_inertia, 'b-', label='Inertia (inverted, normalized)', linewidth=2)
        
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Normalized Score')
        axes[1, 1].set_title('Combined Clustering Metrics (Normalized)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/elbow_method_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Elbow analysis plots saved to {self.data_dir}/elbow_method_analysis.png")
    
    def perform_optimal_clustering(self, optimal_k: int, features_df: pd.DataFrame) -> Dict:
        """Perform clustering with optimal k."""
        print(f"\n🎯 Performing optimal K-means clustering with k={optimal_k}...")
        
        # Fit final model
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.scaled_features)
        
        # Calculate final metrics
        silhouette = silhouette_score(self.scaled_features, labels)
        calinski = calinski_harabasz_score(self.scaled_features, labels)
        inertia = kmeans.inertia_
        
        print(f"✅ Final clustering metrics:")
        print(f"   • Silhouette score: {silhouette:.3f}")
        print(f"   • Calinski-Harabasz score: {calinski:.1f}")
        print(f"   • Inertia: {inertia:.0f}")
        
        # Feature importance analysis
        feature_importance = np.abs(kmeans.cluster_centers_).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': features_df.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 most important features:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
        
        results = {
            'kmeans_model': kmeans,
            'labels': labels,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'inertia': inertia,
            'feature_importance': feature_importance_df,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        return results
    
    def dimensionality_reduction_analysis(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Perform dimensionality reduction analysis."""
        print(f"\n📐 Performing dimensionality reduction analysis...")
        
        # PCA Analysis
        pca = PCA()
        pca_features = pca.fit_transform(self.scaled_features)
        
        # Explained variance analysis
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
        n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
        
        print(f"✅ PCA Analysis:")
        print(f"   • Components for 90% variance: {n_components_90}")
        print(f"   • Components for 95% variance: {n_components_95}")
        print(f"   • First 2 components explain: {cumsum_variance[1]:.1%} of variance")
        
        # 2D PCA for visualization
        pca_2d = PCA(n_components=2)
        pca_2d_features = pca_2d.fit_transform(self.scaled_features)
        
        # t-SNE for non-linear dimensionality reduction
        print("   • Computing t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_df)-1))
        tsne_features = tsne.fit_transform(self.scaled_features)
        
        results = {
            'pca_full': pca,
            'pca_features': pca_features,
            'pca_2d_features': pca_2d_features,
            'tsne_features': tsne_features,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumsum_variance': cumsum_variance,
            'n_components_90': n_components_90,
            'n_components_95': n_components_95
        }
        
        return results
    
    def plot_dimensionality_reduction(self, dim_results: Dict, labels: np.ndarray, 
                                    features_df: pd.DataFrame) -> None:
        """Plot dimensionality reduction results."""
        print("\n📊 Creating dimensionality reduction plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: PCA Explained Variance
        axes[0, 0].plot(range(1, 21), dim_results['explained_variance_ratio'][:20], 'bo-')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA: Explained Variance by Component')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Highlight 90% and 95% variance lines
        axes[0, 0].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% variance')
        axes[0, 0].axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95% variance')
        axes[0, 0].legend()
        
        # Plot 2: Cumulative Explained Variance
        axes[0, 1].plot(range(1, 21), dim_results['cumsum_variance'][:20], 'go-')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].set_title('PCA: Cumulative Explained Variance')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
        axes[0, 1].axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
        axes[0, 1].legend()
        
        # Plot 3: PCA 2D Clustering
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(labels))))
        for i, color in enumerate(colors):
            cluster_mask = labels == i
            axes[1, 0].scatter(dim_results['pca_2d_features'][cluster_mask, 0], 
                             dim_results['pca_2d_features'][cluster_mask, 1],
                             c=[color], label=f'Cluster {i}', alpha=0.7, s=60)
        
        axes[1, 0].set_xlabel(f'PC1 ({dim_results["explained_variance_ratio"][0]:.1%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({dim_results["explained_variance_ratio"][1]:.1%} variance)')
        axes[1, 0].set_title('PCA: 2D Cluster Visualization')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: t-SNE 2D Clustering
        for i, color in enumerate(colors):
            cluster_mask = labels == i
            axes[1, 1].scatter(dim_results['tsne_features'][cluster_mask, 0], 
                             dim_results['tsne_features'][cluster_mask, 1],
                             c=[color], label=f'Cluster {i}', alpha=0.7, s=60)
        
        axes[1, 1].set_xlabel('t-SNE Component 1')
        axes[1, 1].set_ylabel('t-SNE Component 2')
        axes[1, 1].set_title('t-SNE: 2D Cluster Visualization')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/dimensionality_reduction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dimensionality reduction plots saved to {self.data_dir}/dimensionality_reduction_analysis.png")
    
    def cluster_interpretation_analysis(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Perform detailed cluster interpretation analysis."""
        print(f"\n🔍 Performing cluster interpretation analysis...")
        
        # Add cluster labels to features
        analysis_df = features_df.copy()
        analysis_df['Cluster'] = labels
        analysis_df['Asset'] = features_df.index
        
        cluster_profiles = {}
        
        # Key metrics for interpretation
        key_metrics = [
            'mean_annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown',
            'recent_3month_return', 'market_correlation', 'safe_haven_score'
        ]
        
        available_metrics = [m for m in key_metrics if m in analysis_df.columns]
        
        for cluster_id in range(len(np.unique(labels))):
            cluster_data = analysis_df[analysis_df['Cluster'] == cluster_id]
            
            profile = {
                'assets': cluster_data['Asset'].tolist(),
                'size': len(cluster_data),
                'metrics': {}
            }
            
            # Calculate statistics for each metric
            for metric in available_metrics:
                profile['metrics'][metric] = {
                    'mean': cluster_data[metric].mean(),
                    'std': cluster_data[metric].std(),
                    'min': cluster_data[metric].min(),
                    'max': cluster_data[metric].max(),
                    'median': cluster_data[metric].median()
                }
            
            # Identify cluster characteristics
            profile['characteristics'] = self._identify_cluster_characteristics(cluster_data, available_metrics)
            
            cluster_profiles[f'Cluster_{cluster_id}'] = profile
        
        print(f"✅ Cluster interpretation complete for {len(cluster_profiles)} clusters")
        
        return cluster_profiles
    
    def _identify_cluster_characteristics(self, cluster_data: pd.DataFrame, metrics: List[str]) -> Dict[str, str]:
        """Identify key characteristics of a cluster."""
        characteristics = {}
        
        if 'annual_volatility' in metrics:
            vol_mean = cluster_data['annual_volatility'].mean()
            if vol_mean < 0.2:
                characteristics['risk_level'] = 'Low Risk'
            elif vol_mean < 0.3:
                characteristics['risk_level'] = 'Moderate Risk'
            else:
                characteristics['risk_level'] = 'High Risk'
        
        if 'sharpe_ratio' in metrics:
            sharpe_mean = cluster_data['sharpe_ratio'].mean()
            if sharpe_mean > 0.8:
                characteristics['performance'] = 'High Performance'
            elif sharpe_mean > 0.5:
                characteristics['performance'] = 'Moderate Performance'
            else:
                characteristics['performance'] = 'Low Performance'
        
        if 'recent_3month_return' in metrics:
            recent_return = cluster_data['recent_3month_return'].mean()
            if recent_return > 0.1:
                characteristics['momentum'] = 'Strong Momentum'
            elif recent_return > 0.05:
                characteristics['momentum'] = 'Moderate Momentum'
            elif recent_return > 0:
                characteristics['momentum'] = 'Weak Momentum'
            else:
                characteristics['momentum'] = 'Negative Momentum'
        
        if 'market_correlation' in metrics:
            corr_mean = cluster_data['market_correlation'].mean()
            if corr_mean > 0.7:
                characteristics['market_exposure'] = 'High Market Beta'
            elif corr_mean > 0.4:
                characteristics['market_exposure'] = 'Moderate Market Beta'
            else:
                characteristics['market_exposure'] = 'Low Market Beta'
        
        # Sector analysis
        sector_cols = [col for col in cluster_data.columns if col.startswith('sector_')]
        if sector_cols:
            dominant_sector = None
            max_sector_count = 0
            for col in sector_cols:
                sector_count = cluster_data[col].sum()
                if sector_count > max_sector_count:
                    max_sector_count = sector_count
                    dominant_sector = col.replace('sector_', '')
            
            if max_sector_count > len(cluster_data) * 0.3:  # More than 30% of cluster
                characteristics['dominant_sector'] = dominant_sector
        
        return characteristics
    
    def portfolio_diversification_strategy(self, cluster_profiles: Dict) -> Dict[str, str]:
        """Generate portfolio diversification strategy based on clusters."""
        print(f"\n💼 Generating portfolio diversification strategy...")
        
        strategy = {}
        
        for cluster_name, profile in cluster_profiles.items():
            characteristics = profile['characteristics']
            metrics = profile['metrics']
            
            # Determine allocation based on cluster characteristics
            risk_level = characteristics.get('risk_level', 'Unknown')
            performance = characteristics.get('performance', 'Unknown')
            momentum = characteristics.get('momentum', 'Unknown')
            
            if risk_level == 'Low Risk' and performance in ['High Performance', 'Moderate Performance']:
                allocation = 'Core Holdings (25-30%)'
                rationale = 'Stable foundation with good risk-adjusted returns'
            
            elif performance == 'High Performance' and momentum == 'Strong Momentum':
                allocation = 'Growth Holdings (15-20%)'
                rationale = 'High-performing assets with strong recent momentum'
            
            elif risk_level == 'Low Risk':
                allocation = 'Defensive Holdings (20-25%)'
                rationale = 'Portfolio stability and downside protection'
            
            elif momentum == 'Strong Momentum':
                allocation = 'Momentum Holdings (10-15%)'
                rationale = 'Capitalize on recent performance trends'
            
            elif risk_level == 'High Risk' and performance == 'High Performance':
                allocation = 'Satellite Holdings (5-10%)'
                rationale = 'High risk/reward with limited allocation'
            
            else:
                allocation = 'Diversifier (5-15%)'
                rationale = 'Additional diversification benefits'
            
            strategy[cluster_name] = {
                'allocation': allocation,
                'rationale': rationale,
                'assets': profile['assets'][:5],  # Show first 5 assets
                'total_assets': len(profile['assets'])
            }
        
        return strategy
    
    def save_results(self, elbow_results: Dict, clustering_results: Dict, 
                   cluster_profiles: Dict, diversification_strategy: Dict) -> None:
        """Save all analysis results."""
        print(f"\n💾 Saving analysis results...")
        
        # Prepare serializable results
        results = {
            'elbow_analysis': {
                'k_values': [int(k) for k in elbow_results['k_values']],
                'optimal_k_elbow': int(elbow_results['optimal_k_elbow']) if elbow_results['optimal_k_elbow'] else None,
                'optimal_k_silhouette': int(elbow_results['optimal_k_silhouette']),
                'optimal_k_calinski': int(elbow_results['optimal_k_calinski']),
                'best_silhouette_score': float(max(elbow_results['silhouette_scores'])),
                'best_calinski_score': float(max(elbow_results['calinski_harabasz_scores']))
            },
            'final_clustering': {
                'optimal_k': int(len(np.unique(clustering_results['labels']))),
                'silhouette_score': float(clustering_results['silhouette_score']),
                'calinski_harabasz_score': float(clustering_results['calinski_harabasz_score']),
                'inertia': float(clustering_results['inertia'])
            },
            'cluster_profiles': self._serialize_cluster_profiles(cluster_profiles),
            'diversification_strategy': diversification_strategy
        }
        
        # Save to JSON
        with open(f'{self.data_dir}/advanced_kmeans_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save feature importance
        clustering_results['feature_importance'].to_csv(
            f'{self.data_dir}/clustering_feature_importance.csv', index=False
        )
        
        print(f"✅ Results saved:")
        print(f"   • {self.data_dir}/advanced_kmeans_analysis.json")
        print(f"   • {self.data_dir}/clustering_feature_importance.csv")
        print(f"   • {self.data_dir}/elbow_method_analysis.png")
        print(f"   • {self.data_dir}/dimensionality_reduction_analysis.png")
    
    def _serialize_cluster_profiles(self, cluster_profiles: Dict) -> Dict:
        """Convert cluster profiles to JSON-serializable format."""
        serialized = {}
        
        for cluster_name, profile in cluster_profiles.items():
            serialized[cluster_name] = {
                'assets': profile['assets'],
                'size': int(profile['size']),
                'characteristics': profile['characteristics'],
                'metrics': {}
            }
            
            # Convert numpy types to Python types
            for metric, stats in profile['metrics'].items():
                serialized[cluster_name]['metrics'][metric] = {
                    k: float(v) for k, v in stats.items()
                }
        
        return serialized
    
    def run_complete_analysis(self) -> None:
        """Run the complete advanced K-means analysis."""
        print("🚀 ADVANCED K-MEANS CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return
        
        # Prepare features
        features_df = self.prepare_clustering_features()
        scaled_features = self.standardize_features(features_df)
        
        # Elbow method analysis
        elbow_results = self.elbow_method_analysis()
        self.plot_elbow_analysis(elbow_results)
        
        # Determine optimal k (using silhouette score as primary metric)
        optimal_k = elbow_results['optimal_k_silhouette']
        print(f"\n🎯 Using k={optimal_k} based on silhouette score optimization")
        
        # Perform optimal clustering
        clustering_results = self.perform_optimal_clustering(optimal_k, features_df)
        
        # Dimensionality reduction analysis
        dim_results = self.dimensionality_reduction_analysis(features_df, clustering_results['labels'])
        self.plot_dimensionality_reduction(dim_results, clustering_results['labels'], features_df)
        
        # Cluster interpretation
        cluster_profiles = self.cluster_interpretation_analysis(features_df, clustering_results['labels'])
        
        # Portfolio diversification strategy
        diversification_strategy = self.portfolio_diversification_strategy(cluster_profiles)
        
        # Display results
        self.display_final_results(cluster_profiles, diversification_strategy)
        
        # Save results
        self.save_results(elbow_results, clustering_results, cluster_profiles, diversification_strategy)
        
        print(f"\n" + "=" * 60)
        print("✅ ADVANCED K-MEANS ANALYSIS COMPLETE!")
    
    def display_final_results(self, cluster_profiles: Dict, diversification_strategy: Dict) -> None:
        """Display final analysis results."""
        print(f"\n" + "=" * 70)
        print("ADVANCED CLUSTERING RESULTS & PORTFOLIO STRATEGY")
        print("=" * 70)
        
        for cluster_name, profile in cluster_profiles.items():
            print(f"\n{cluster_name.upper()}:")
            print("-" * 50)
            print(f"Assets ({profile['size']}): {', '.join(profile['assets'][:7])}{'...' if len(profile['assets']) > 7 else ''}")
            
            # Show characteristics
            characteristics = profile['characteristics']
            print(f"Characteristics: {', '.join([f'{k}: {v}' for k, v in characteristics.items()])}")
            
            # Show key metrics
            if 'mean_annual_return' in profile['metrics']:
                ret = profile['metrics']['mean_annual_return']['mean']
                vol = profile['metrics']['annual_volatility']['mean']
                sharpe = profile['metrics']['sharpe_ratio']['mean']
                print(f"Performance: {ret:.1%} return, {vol:.1%} volatility, {sharpe:.2f} Sharpe")
            
            if 'recent_3month_return' in profile['metrics']:
                recent = profile['metrics']['recent_3month_return']['mean']
                print(f"Recent 3M: {recent:.1%}")
            
            # Show strategy
            strategy = diversification_strategy.get(cluster_name, {})
            if strategy:
                print(f"Portfolio Role: {strategy['allocation']}")
                print(f"Rationale: {strategy['rationale']}")


def main():
    """Main function to run advanced K-means analysis."""
    analyzer = AdvancedKMeansAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 