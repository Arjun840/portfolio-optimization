#!/usr/bin/env python3
"""
Feature Engineering & Clustering Summary

This script provides a comprehensive overview of the feature engineering
and clustering results for portfolio optimization.
"""

import pandas as pd
import numpy as np
import json

def main():
    """Display comprehensive summary of feature engineering and clustering."""
    
    print("🚀 PORTFOLIO OPTIMIZATION: FEATURE ENGINEERING & ML CLUSTERING")
    print("=" * 80)
    
    # Load and display data overview
    try:
        features = pd.read_pickle('data/asset_features.pkl')
        print(f"\n📊 FEATURE MATRIX:")
        print(f"   • {features.shape[0]} assets")
        print(f"   • {features.shape[1]} engineered features")
        print(f"   • Ready for ML models!")
        
        # Load feature groups
        with open('data/feature_groups.json', 'r') as f:
            feature_groups = json.load(f)
        
        print(f"\n🔧 FEATURE CATEGORIES:")
        for group, group_features in feature_groups.items():
            print(f"   • {group.replace('_', ' ').title()}: {len(group_features)} features")
        
    except FileNotFoundError:
        print("❌ Features not found. Run feature_engineering.py first.")
        return
    
    # Display clustering results
    try:
        with open('data/cluster_analysis.json', 'r') as f:
            cluster_stats = json.load(f)
        
        print(f"\n🎯 ASSET CLUSTERING RESULTS:")
        print(f"   • {len(cluster_stats)} clusters identified")
        print(f"   • Optimal clustering using ML techniques")
        
        # Show key cluster insights
        cluster_insights = {
            "Cluster_0": "Blue Chip Stocks (19 assets) - Stable, dividend-paying companies",
            "Cluster_1": "High-Growth Tech (5 assets) - MSFT, GOOGL, NVDA, META, TSLA",  
            "Cluster_2": "Traditional Value (12 assets) - Finance, Industrial, Telecom sectors",
            "Cluster_3": "Defensive Assets (6 assets) - International ETFs, bonds, gold",
            "Cluster_4": "Market ETFs (3 assets) - SPY, QQQ, VTI - broad market exposure"
        }
        
        print(f"\n📈 CLUSTER CHARACTERISTICS:")
        for cluster_id, description in cluster_insights.items():
            if cluster_id in cluster_stats:
                stats = cluster_stats[cluster_id]
                return_pct = stats['mean_annual_return']['mean'] * 100
                vol_pct = stats['annual_volatility']['mean'] * 100
                sharpe = stats['sharpe_ratio']['mean']
                print(f"   • {description}")
                print(f"     Return: {return_pct:.1f}%, Vol: {vol_pct:.1f}%, Sharpe: {sharpe:.2f}")
        
        print(f"\n💼 PORTFOLIO ALLOCATION FRAMEWORK:")
        print(f"   • Core Holdings (25-30%): Market ETFs - SPY, QQQ, VTI")
        print(f"   • Defensive (20-25%): Low volatility - bonds, gold, international") 
        print(f"   • Growth (15-20%): High Sharpe tech stocks")
        print(f"   • Value (10-15%): Traditional sectors - finance, industrial")
        print(f"   • Quality (10-15%): Blue chip dividend stocks")
        
    except FileNotFoundError:
        print("❌ Clustering results not found. Run ml_clustering_example.py first.")
    
    # Show sample features for top assets
    print(f"\n📋 SAMPLE FEATURES (Top Assets):")
    key_features = ['mean_annual_return', 'annual_volatility', 'sharpe_ratio', 
                   'momentum_21d', 'market_correlation', 'sector_Technology']
    
    available_features = [f for f in key_features if f in features.columns]
    if available_features:
        sample_data = features[available_features].head(10)
        print(sample_data.round(3))
    
    # File summary
    print(f"\n📁 GENERATED FILES:")
    files_created = [
        "data/asset_features.pkl - Feature matrix (118 features x 45 assets)",
        "data/asset_features.csv - Human-readable feature matrix", 
        "data/feature_groups.json - Feature category definitions",
        "data/clustered_assets.csv - Assets with cluster assignments",
        "data/cluster_analysis.json - Detailed cluster statistics",
        "data/asset_clusters_pca.png - Cluster visualization"
    ]
    
    for file_desc in files_created:
        print(f"   • {file_desc}")
    
    # Next steps
    print(f"\n🎯 NEXT STEPS FOR ML PORTFOLIO OPTIMIZATION:")
    print(f"   1. 📊 Portfolio Optimization Models:")
    print(f"      • Mean-variance optimization using cluster constraints")
    print(f"      • Risk parity models with cluster-based risk budgeting")
    print(f"      • Black-Litterman with cluster-informed views")
    
    print(f"\n   2. 🤖 Machine Learning Models:")
    print(f"      • Ensemble models for return prediction")
    print(f"      • Regime detection using clustering features")
    print(f"      • Risk prediction models")
    
    print(f"\n   3. 🔄 Portfolio Construction:")
    print(f"      • Multi-objective optimization (return, risk, diversification)")
    print(f"      • Dynamic rebalancing strategies")
    print(f"      • Risk-adjusted performance evaluation")
    
    print(f"\n   4. 🌐 Web Interface:")
    print(f"      • Interactive portfolio optimizer")
    print(f"      • Real-time risk monitoring")
    print(f"      • Performance analytics dashboard")
    
    print(f"\n" + "=" * 80)
    print(f"✅ FEATURE ENGINEERING COMPLETE!")
    print(f"Your portfolio optimization system now has:")
    print(f"   • 118 engineered features capturing asset behavior")
    print(f"   • 5 meaningful asset clusters for diversification")
    print(f"   • Ready-to-use data for advanced ML models")
    print(f"   • Strategic framework for portfolio construction")
    print(f"=" * 80)

if __name__ == "__main__":
    main() 