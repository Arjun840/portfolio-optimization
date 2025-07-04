from fastapi import HTTPException, Depends, status
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

from .models import *
from .auth import get_current_user, register_user, authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from .data_utils import load_stock_data, calculate_portfolio_metrics, get_cluster_names, get_cluster_info, generate_sample_historical_data
from scripts.portfolio_optimization import PortfolioOptimizer

logger = logging.getLogger(__name__)

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

async def signup(user_data: UserSignup) -> Token:
    """Register a new user"""
    user = register_user(user_data.email, user_data.password, user_data.full_name)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

async def login(user_data: UserLogin) -> Token:
    """Authenticate user and return access token"""
    user = authenticate_user(user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

# ============================================================================
# DATA ENDPOINTS
# ============================================================================

async def get_stocks(current_user: dict = Depends(get_current_user)) -> List[StockInfo]:
    """Retrieve list of available stocks with cluster and performance information"""
    try:
        data, features_df, cluster_data = load_stock_data()
        
        stocks = []
        asset_clusters = cluster_data.get("asset_clusters", {})
        cluster_names = get_cluster_names()
        
        for _, row in features_df.iterrows():
            asset = row['Asset']
            cluster_id = asset_clusters.get(asset, 0)
            
            stocks.append(StockInfo(
                symbol=asset,
                name=asset,  # Could be enhanced with company names
                sector=row.get('Sector', 'Unknown'),
                cluster=int(cluster_id),
                cluster_name=cluster_names.get(int(cluster_id), f"Cluster {cluster_id}"),
                sharpe_ratio=float(row.get('Sharpe', 0.0)),
                volatility=float(row.get('Volatility', 0.0)),
                expected_return=float(row.get('Expected_Return', 0.0)) if 'Expected_Return' in row else 0.0
            ))
        
        return stocks
        
    except Exception as e:
        logger.error(f"Error retrieving stocks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stock data")

async def get_clusters(current_user: dict = Depends(get_current_user)) -> List[ClusterInfo]:
    """Retrieve cluster information and characteristics"""
    try:
        cluster_info = get_cluster_info()
        
        clusters = []
        for cluster_id, info in cluster_info.items():
            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                cluster_name=info["name"],
                description=info["description"],
                asset_count=info["asset_count"],
                avg_sharpe=info["avg_sharpe"],
                avg_volatility=info["avg_volatility"],
                role=info["role"],
                target_allocation=info["target_allocation"]
            ))
        
        return clusters
        
    except Exception as e:
        logger.error(f"Error retrieving clusters: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cluster data")

# ============================================================================
# PORTFOLIO OPTIMIZATION ENDPOINTS
# ============================================================================

async def optimize_portfolio(
    request: OptimizationRequest,
    portfolio_value: float = 100000,
    current_user: dict = Depends(get_current_user)
) -> OptimizedPortfolio:
    """
    Optimize portfolio based on user preferences and constraints
    """
    try:
        # Load data
        data, features_df, cluster_data = load_stock_data()
        
        # Initialize optimizer and load data
        optimizer = PortfolioOptimizer(data_dir="data")
        if not optimizer.load_data():
            raise HTTPException(status_code=500, detail="Failed to load optimization data")
        
        # Calculate expected returns and covariance matrix
        optimizer.calculate_expected_returns(method=request.return_method)
        optimizer.calculate_covariance_matrix(method="historical")
        
        # Set optimization parameters based on request
        if request.target_return:
            result = optimizer.minimize_variance(
                target_return=request.target_return,
                constraint_type=request.constraint_level
            )
        else:
            if request.optimization_objective == "max_sharpe":
                result = optimizer.maximize_sharpe_ratio(constraint_type=request.constraint_level)
            else:
                result = optimizer.minimize_variance(constraint_type=request.constraint_level)
        
        # Process optimization result
        weights = result['weights']
        
        # Filter out zero weights
        significant_weights = {k: v for k, v in weights.items() if v > 0.001}
        
        # Calculate metrics
        metrics_data = calculate_portfolio_metrics(significant_weights, data['returns'])
        
        metrics = PortfolioMetrics(**metrics_data)
        
        # Create holdings
        holdings = []
        cluster_allocation = {}
        sector_allocation = {}
        
        cluster_names = get_cluster_names()
        
        for symbol, weight in significant_weights.items():
            # Get asset info
            asset_info = features_df[features_df['Asset'] == symbol]
            if len(asset_info) > 0:
                asset_row = asset_info.iloc[0]
                sector = asset_row.get('Sector', 'Unknown')
                cluster_id = cluster_data.get("asset_clusters", {}).get(symbol, 0)
                
                allocation_amount = weight * portfolio_value
                
                holdings.append(PortfolioHolding(
                    symbol=symbol,
                    weight=float(weight),
                    allocation_amount=float(allocation_amount),
                    expected_return=float(asset_row.get('Expected_Return', 0.0)) if 'Expected_Return' in asset_row else 0.0,
                    volatility=float(asset_row.get('Volatility', 0.0)),
                    sharpe_ratio=float(asset_row.get('Sharpe', 0.0)),
                    sector=sector,
                    cluster=int(cluster_id)
                ))
                
                # Aggregate cluster allocation
                cluster_allocation[int(cluster_id)] = cluster_allocation.get(int(cluster_id), 0) + weight
                
                # Aggregate sector allocation
                sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
        
        # Sort holdings by weight
        holdings.sort(key=lambda x: x.weight, reverse=True)
        
        # Create portfolio response
        portfolio = OptimizedPortfolio(
            portfolio_id=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy=f"{request.risk_tolerance.title()} Risk - {request.optimization_objective.replace('_', ' ').title()}",
            total_amount=portfolio_value,
            holdings=holdings,
            metrics=metrics,
            cluster_allocation=cluster_allocation,
            sector_allocation=sector_allocation,
            optimization_details={
                "constraint_level": request.constraint_level,
                "return_method": request.return_method,
                "objective": request.optimization_objective,
                "max_position_size": request.max_position_size,
                "min_assets": request.min_assets,
                "optimization_status": result.get('success', True),
                "optimization_message": result.get('message', 'Optimization completed successfully')
            },
            created_at=datetime.now()
        )
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")

async def get_historical_performance(
    portfolio_id: str,
    benchmark: str = "SPY",
    current_user: dict = Depends(get_current_user)
) -> HistoricalPerformance:
    """
    Get historical performance data for a portfolio compared to benchmark
    """
    try:
        # This is a simplified implementation
        # In production, you'd store portfolio weights and calculate actual historical performance
        
        data, _, _ = load_stock_data()
        returns_data = data['returns']
        
        # For demo purposes, create sample performance data
        # In production, you would retrieve actual portfolio weights from database
        sample_weights = {"SPY": 0.4, "AAPL": 0.3, "GOOGL": 0.2, "MSFT": 0.1}
        
        historical_data = generate_sample_historical_data(
            returns_data, sample_weights, benchmark, days=252
        )
        
        return HistoricalPerformance(**historical_data)
        
    except Exception as e:
        logger.error(f"Error retrieving historical performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve historical performance")

async def get_efficient_frontier(
    constraint_level: str = "enhanced",
    return_method: str = "ml_enhanced",
    current_user: dict = Depends(get_current_user)
) -> EfficientFrontierData:
    """
    Generate efficient frontier data for portfolio visualization
    """
    try:
        # Load data
        data, _, _ = load_stock_data()
        
        # Initialize optimizer and load data
        optimizer = PortfolioOptimizer(data_dir="data")
        if not optimizer.load_data():
            raise HTTPException(status_code=500, detail="Failed to load optimization data")
        
        # Calculate expected returns and covariance matrix
        optimizer.calculate_expected_returns(method=return_method)
        optimizer.calculate_covariance_matrix(method="historical")
        
        # Generate efficient frontier
        frontier_df = optimizer.generate_efficient_frontier(n_portfolios=20)
        
        # Convert DataFrame to the expected format
        frontier_data = {
            'returns': frontier_df['Return'].tolist(),
            'volatilities': frontier_df['Volatility'].tolist(),
            'sharpe_ratios': frontier_df['Sharpe'].tolist(),
            'max_sharpe': {
                'return': float(frontier_df.loc[frontier_df['Sharpe'].idxmax(), 'Return']),
                'volatility': float(frontier_df.loc[frontier_df['Sharpe'].idxmax(), 'Volatility']),
                'sharpe': float(frontier_df['Sharpe'].max())
            },
            'min_variance': {
                'return': float(frontier_df.loc[frontier_df['Volatility'].idxmin(), 'Return']),
                'volatility': float(frontier_df['Volatility'].min()),
                'sharpe': float(frontier_df.loc[frontier_df['Volatility'].idxmin(), 'Sharpe'])
            }
        }
        
        return EfficientFrontierData(
            returns=[float(r) for r in frontier_data['returns']],
            volatilities=[float(v) for v in frontier_data['volatilities']],
            sharpe_ratios=[float(s) for s in frontier_data['sharpe_ratios']],
            max_sharpe_portfolio={
                "return": float(frontier_data['max_sharpe']['return']),
                "volatility": float(frontier_data['max_sharpe']['volatility']),
                "sharpe_ratio": float(frontier_data['max_sharpe']['sharpe'])
            },
            min_variance_portfolio={
                "return": float(frontier_data['min_variance']['return']),
                "volatility": float(frontier_data['min_variance']['volatility']),
                "sharpe_ratio": float(frontier_data['min_variance']['sharpe'])
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating efficient frontier: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate efficient frontier") 