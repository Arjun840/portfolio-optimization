#!/usr/bin/env python3
"""
Portfolio Optimization Web Dashboard

A simple Flask web interface to view portfolio data, analysis results,
and visualizations in your browser.
"""

from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.data_analysis_and_cleaning import PortfolioDataAnalyzer

app = Flask(__name__)

# Initialize the data analyzer
analyzer = PortfolioDataAnalyzer()

def load_portfolio_data():
    """Load portfolio data and calculate basic statistics."""
    try:
        # Load cleaned data
        prices = pd.read_pickle('data/cleaned_prices.pkl')
        returns = pd.read_pickle('data/cleaned_returns.pkl')
        
        # Calculate statistics
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratios = annual_returns / annual_volatility
        
        return {
            'prices': prices,
            'returns': returns,
            'annual_returns': annual_returns,
            'annual_volatility': annual_volatility,
            'sharpe_ratios': sharpe_ratios,
            'correlations': returns.corr()
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string for web display."""
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url

@app.route('/')
def dashboard():
    """Main dashboard page."""
    data = load_portfolio_data()
    
    if data is None:
        return render_template('error.html', 
                             error="Could not load portfolio data. Please run data analysis first.")
    
    # Calculate summary statistics
    summary_stats = {
        'total_assets': len(data['prices'].columns),
        'date_range': f"{data['prices'].index[0].strftime('%Y-%m-%d')} to {data['prices'].index[-1].strftime('%Y-%m-%d')}",
        'total_days': len(data['prices']),
        'best_performer': data['annual_returns'].idxmax(),
        'best_return': data['annual_returns'].max(),
        'best_sharpe': data['sharpe_ratios'].idxmax(),
        'best_sharpe_value': data['sharpe_ratios'].max(),
        'avg_correlation': data['correlations'].values[np.triu_indices_from(data['correlations'].values, k=1)].mean()
    }
    
    return render_template('dashboard.html', stats=summary_stats)

@app.route('/api/top_performers')
def top_performers():
    """API endpoint for top performing assets."""
    data = load_portfolio_data()
    if data is None:
        return jsonify({'error': 'Data not available'})
    
    # Top 10 by Sharpe ratio
    top_sharpe = data['sharpe_ratios'].nlargest(10)
    
    performers = []
    for asset, sharpe in top_sharpe.items():
        performers.append({
            'asset': asset,
            'sharpe_ratio': round(sharpe, 3),
            'annual_return': round(data['annual_returns'][asset], 3),
            'annual_volatility': round(data['annual_volatility'][asset], 3)
        })
    
    return jsonify(performers)

@app.route('/api/correlation_matrix')
def correlation_matrix():
    """API endpoint for correlation matrix data."""
    data = load_portfolio_data()
    if data is None:
        return jsonify({'error': 'Data not available'})
    
    # Convert correlation matrix to JSON-friendly format
    corr_data = []
    assets = list(data['correlations'].columns)
    
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            corr_data.append({
                'asset1': asset1,
                'asset2': asset2,
                'correlation': round(data['correlations'].iloc[i, j], 3)
            })
    
    return jsonify({
        'assets': assets,
        'correlations': corr_data
    })

@app.route('/plots/risk_return')
def risk_return_plot():
    """Generate risk-return scatter plot."""
    data = load_portfolio_data()
    if data is None:
        return "Data not available", 400
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    volatilities = data['annual_volatility'].values
    returns = data['annual_returns'].values
    sharpe_ratios = data['sharpe_ratios'].values
    
    scatter = ax.scatter(volatilities, returns, c=sharpe_ratios, 
                        cmap='RdYlGn', s=100, alpha=0.7)
    
    # Add asset labels
    for i, asset in enumerate(data['annual_returns'].index):
        ax.annotate(asset, (volatilities[i], returns[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Annual Volatility')
    ax.set_ylabel('Annual Return')
    ax.set_title('Risk-Return Profile (Color = Sharpe Ratio)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio')
    
    plot_url = create_plot_base64(fig)
    
    return f'<img src="data:image/png;base64,{plot_url}" style="max-width:100%; height:auto;">'

@app.route('/plots/price_history')
def price_history_plot():
    """Generate price history plot for top performers."""
    data = load_portfolio_data()
    if data is None:
        return "Data not available", 400
    
    # Select top 5 performers by Sharpe ratio
    top_assets = data['sharpe_ratios'].nlargest(5).index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for asset in top_assets:
        # Normalize prices to start at 100
        normalized_prices = 100 * data['prices'][asset] / data['prices'][asset].iloc[0]
        ax.plot(normalized_prices.index, normalized_prices, 
               linewidth=2, label=asset, alpha=0.8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price (Base=100)')
    ax.set_title('Price History - Top 5 Performers (Normalized)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    plot_url = create_plot_base64(fig)
    
    return f'<img src="data:image/png;base64,{plot_url}" style="max-width:100%; height:auto;">'

@app.route('/plots/correlation_heatmap')
def correlation_heatmap():
    """Generate correlation heatmap."""
    data = load_portfolio_data()
    if data is None:
        return "Data not available", 400
    
    # Select subset of assets for readability
    important_assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'TLT', 'GLD']
    available_assets = [asset for asset in important_assets if asset in data['correlations'].columns]
    
    if len(available_assets) < 5:
        available_assets = data['correlations'].columns[:10].tolist()
    
    corr_subset = data['correlations'].loc[available_assets, available_assets]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_subset, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.2f', ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
    
    plot_url = create_plot_base64(fig)
    
    return f'<img src="data:image/png;base64,{plot_url}" style="max-width:100%; height:auto;">'

@app.route('/data/summary')
def data_summary():
    """Display data summary page."""
    try:
        # Read the analysis report if it exists
        report_path = 'data_analysis_report.txt'
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_content = f.read()
        else:
            report_content = "Analysis report not found. Please run the data analysis script first."
        
        return render_template('report.html', report=report_content)
    except Exception as e:
        return f"Error loading report: {e}"

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_available': os.path.exists('data/cleaned_prices.pkl')
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 PORTFOLIO OPTIMIZATION WEB DASHBOARD")
    print("=" * 60)
    print("Starting Flask web server...")
    
    # Try port 5000, if busy use 5001
    import socket
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) != 0
    
    port = 5000 if is_port_available(5000) else 5001
    
    print(f"📊 Dashboard will be available at: http://localhost:{port}")
    print("📈 API endpoints:")
    print("   • /api/top_performers - Top performing assets")
    print("   • /api/correlation_matrix - Correlation data")
    print("   • /plots/risk_return - Risk-return plot")
    print("   • /plots/price_history - Price history chart")
    print("   • /data/summary - Analysis report")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=port) 