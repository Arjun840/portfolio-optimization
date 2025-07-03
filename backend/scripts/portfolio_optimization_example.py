#!/usr/bin/env python3
"""
Portfolio Optimization Usage Examples

This script demonstrates how to use the portfolio optimization system
for different investment scenarios and risk profiles.
"""

from portfolio_optimization import PortfolioOptimizer
import pandas as pd
import numpy as np

def aggressive_growth_portfolio():
    """Example: Aggressive growth portfolio optimization."""
    print("🚀 AGGRESSIVE GROWTH PORTFOLIO")
    print("=" * 50)
    
    optimizer = PortfolioOptimizer(data_dir="data")
    
    # Load data
    if not optimizer.load_data():
        return
    
    # Calculate expected returns using enhanced method
    optimizer.calculate_expected_returns("enhanced")
    optimizer.calculate_covariance_matrix("historical")
    
    # Maximum Sharpe ratio portfolio
    portfolio = optimizer.maximize_sharpe_ratio()
    
    if portfolio:
        print(f"📈 Expected Return: {portfolio['expected_return']:.1%}")
        print(f"📊 Volatility: {portfolio['volatility']:.1%}")
        print(f"🎯 Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        print(f"\n💼 Top Holdings:")
        
        top_holdings = portfolio['weights'].sort_values(ascending=False).head(5)
        for asset, weight in top_holdings.items():
            print(f"   • {asset}: {weight:.1%}")
    
    return portfolio

def conservative_portfolio():
    """Example: Conservative minimum variance portfolio."""
    print("\n🛡️  CONSERVATIVE PORTFOLIO")
    print("=" * 50)
    
    optimizer = PortfolioOptimizer(data_dir="data")
    
    # Load data
    if not optimizer.load_data():
        return
    
    # Calculate expected returns and covariance
    optimizer.calculate_expected_returns("risk_adjusted")
    optimizer.calculate_covariance_matrix("historical")
    
    # Minimum variance portfolio
    portfolio = optimizer.minimize_variance()
    
    if portfolio:
        print(f"📈 Expected Return: {portfolio['expected_return']:.1%}")
        print(f"📊 Volatility: {portfolio['volatility']:.1%}")
        print(f"🎯 Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        print(f"\n💼 Top Holdings:")
        
        top_holdings = portfolio['weights'].sort_values(ascending=False).head(5)
        for asset, weight in top_holdings.items():
            print(f"   • {asset}: {weight:.1%}")
    
    return portfolio

def diversified_cluster_portfolio():
    """Example: Diversified cluster-based portfolio."""
    print("\n🎯 DIVERSIFIED CLUSTER PORTFOLIO")
    print("=" * 50)
    
    optimizer = PortfolioOptimizer(data_dir="data")
    
    # Load data
    if not optimizer.load_data():
        return
    
    # Calculate expected returns and covariance
    optimizer.calculate_expected_returns("enhanced")
    optimizer.calculate_covariance_matrix("historical")
    
    # Cluster-based portfolio with Sharpe weighting
    portfolio = optimizer.cluster_based_optimization("sharpe_weighted")
    
    if portfolio:
        print(f"📈 Expected Return: {portfolio['expected_return']:.1%}")
        print(f"📊 Volatility: {portfolio['volatility']:.1%}")
        print(f"🎯 Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        
        print(f"\n🎪 Cluster Allocation:")
        for cluster_id, allocation in portfolio['cluster_allocation'].items():
            print(f"   • Cluster {cluster_id}: {allocation:.1%}")
        
        print(f"\n💼 Top Holdings:")
        top_holdings = portfolio['weights'].sort_values(ascending=False).head(5)
        for asset, weight in top_holdings.items():
            print(f"   • {asset}: {weight:.1%}")
    
    return portfolio

def custom_target_return_portfolio(target_return: float = 0.20):
    """Example: Portfolio optimized for a specific target return."""
    print(f"\n🎯 TARGET RETURN PORTFOLIO ({target_return:.1%})")
    print("=" * 50)
    
    optimizer = PortfolioOptimizer(data_dir="data")
    
    # Load data
    if not optimizer.load_data():
        return
    
    # Calculate expected returns and covariance
    optimizer.calculate_expected_returns("enhanced")
    optimizer.calculate_covariance_matrix("historical")
    
    # Minimum variance portfolio with target return
    portfolio = optimizer.minimize_variance(target_return)
    
    if portfolio:
        print(f"📈 Expected Return: {portfolio['expected_return']:.1%}")
        print(f"📊 Volatility: {portfolio['volatility']:.1%}")
        print(f"🎯 Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        print(f"\n💼 Top Holdings:")
        
        top_holdings = portfolio['weights'].sort_values(ascending=False).head(5)
        for asset, weight in top_holdings.items():
            print(f"   • {asset}: {weight:.1%}")
    else:
        print("❌ Could not achieve target return with current constraints")
    
    return portfolio

def portfolio_comparison():
    """Compare different portfolio strategies side by side."""
    print("\n📊 PORTFOLIO STRATEGY COMPARISON")
    print("=" * 60)
    
    optimizer = PortfolioOptimizer(data_dir="data")
    
    # Load data
    if not optimizer.load_data():
        return
    
    # Calculate expected returns and covariance
    optimizer.calculate_expected_returns("enhanced")
    optimizer.calculate_covariance_matrix("historical")
    
    # Run comparison
    comparison_df = optimizer.create_portfolio_comparison()
    
    # Display results
    print("\nStrategy Performance Summary:")
    print("-" * 60)
    
    for _, row in comparison_df.iterrows():
        print(f"\n{row['Strategy']}:")
        print(f"   📈 Return: {row['Expected_Return']:.1%}")
        print(f"   📊 Volatility: {row['Volatility']:.1%}")
        print(f"   🎯 Sharpe: {row['Sharpe_Ratio']:.3f}")
        print(f"   🎪 Diversification: {row['Effective_Assets']:.1f} effective assets")
    
    return comparison_df

def risk_budget_example():
    """Example: Risk budgeting across different asset classes."""
    print("\n⚖️  RISK BUDGETING EXAMPLE")
    print("=" * 50)
    
    optimizer = PortfolioOptimizer(data_dir="data")
    
    # Load data
    if not optimizer.load_data():
        return
    
    # Calculate expected returns and covariance
    optimizer.calculate_expected_returns("enhanced")
    optimizer.calculate_covariance_matrix("historical")
    
    # Risk parity cluster approach
    portfolio = optimizer.cluster_based_optimization("risk_parity")
    
    if portfolio:
        print(f"📈 Expected Return: {portfolio['expected_return']:.1%}")
        print(f"📊 Volatility: {portfolio['volatility']:.1%}")
        print(f"🎯 Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        
        print(f"\n⚖️  Risk Budget by Cluster:")
        for cluster_id, allocation in portfolio['cluster_allocation'].items():
            print(f"   • Cluster {cluster_id}: {allocation:.1%} allocation")
        
        # Calculate risk contribution
        weights = portfolio['weights'].values
        cov_matrix = optimizer.covariance_matrix
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        print(f"\n📊 Risk Contributions:")
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_var
        
        risk_contrib_series = pd.Series(risk_contrib, index=portfolio['weights'].index)
        top_risk_contrib = risk_contrib_series.sort_values(ascending=False).head(5)
        
        for asset, contrib in top_risk_contrib.items():
            print(f"   • {asset}: {contrib:.1%} of portfolio risk")
    
    return portfolio

def main():
    """Run all portfolio optimization examples."""
    print("🎯 PORTFOLIO OPTIMIZATION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Aggressive Growth
    aggressive_portfolio = aggressive_growth_portfolio()
    
    # Example 2: Conservative
    conservative_portfolio_result = conservative_portfolio()
    
    # Example 3: Diversified Cluster-based
    diversified_portfolio = diversified_cluster_portfolio()
    
    # Example 4: Target Return
    target_portfolio = custom_target_return_portfolio(0.25)  # 25% target return
    
    # Example 5: Strategy Comparison
    comparison_results = portfolio_comparison()
    
    # Example 6: Risk Budgeting
    risk_budget_portfolio = risk_budget_example()
    
    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("📁 Check the 'data' folder for detailed results and visualizations")
    print("📈 Use these examples as templates for your investment strategies")

if __name__ == "__main__":
    main() 