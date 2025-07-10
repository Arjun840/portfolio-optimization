from fastapi import HTTPException, Depends, status
from datetime import datetime, timedelta
import logging
import uuid
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .models import *
from .auth import get_current_user, register_user, authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_user_saved_portfolios, save_user_portfolio, delete_user_portfolio, update_user_portfolio
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
                sharpe_ratio=float(row.get('sharpe_ratio', 0.0)),
                volatility=float(row.get('annual_volatility', 0.0)),
                expected_return=float(row.get('mean_annual_return', 0.0))
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
                    expected_return=float(asset_row.get('mean_annual_return', 0.0)),
                    volatility=float(asset_row.get('annual_volatility', 0.0)),
                    sharpe_ratio=float(asset_row.get('sharpe_ratio', 0.0)),
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

async def optimize_custom_portfolio(
    request: CustomPortfolioRequest,
    portfolio_value: float = 100000,
    current_user: dict = Depends(get_current_user)
) -> OptimizedPortfolio:
    """
    Optimize a custom portfolio created by the user
    """
    try:
        # Validate initial holdings weights sum
        total_weight = sum(holding.weight for holding in request.holdings)
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail=f"Portfolio weights must sum to 1.0, got {total_weight:.3f}")
        
        # Load data
        data, features_df, cluster_data = load_stock_data()
        
        # Initialize optimizer and load data
        optimizer = PortfolioOptimizer(data_dir="data")
        if not optimizer.load_data():
            raise HTTPException(status_code=500, detail="Failed to load optimization data")
        
        # Calculate expected returns and covariance matrix
        optimizer.calculate_expected_returns(method=request.return_method)
        optimizer.calculate_covariance_matrix(method="historical")
        
        # Convert custom holdings to optimizer format
        initial_weights = {holding.symbol: holding.weight for holding in request.holdings}
        
        # Validate all symbols exist in the data
        available_assets = set(data['returns'].columns)
        invalid_symbols = [symbol for symbol in initial_weights.keys() if symbol not in available_assets]
        if invalid_symbols:
            raise HTTPException(status_code=400, detail=f"Invalid symbols: {invalid_symbols}")
        
        # Set up optimization based on type
        if request.optimization_type == "improve":
            # Try to improve the portfolio by allowing new assets and optimizing for better risk-return
            if request.target_return:
                result = optimizer.minimize_variance(
                    target_return=request.target_return,
                    constraint_type=request.constraint_level
                )
            else:
                result = optimizer.maximize_sharpe_ratio(
                    constraint_type=request.constraint_level
                )
                
        elif request.optimization_type == "rebalance":
            # For rebalancing, optimize ONLY the existing assets (no new assets allowed)
            # Use a simplified optimization approach for better stability
            existing_assets = list(initial_weights.keys())
            
            # Validate existing assets
            valid_assets = [asset for asset in existing_assets if asset in optimizer.expected_returns.index]
            if len(valid_assets) == 0:
                raise HTTPException(status_code=400, detail="No valid existing assets found for rebalancing")
            
            # Use basic constraint level for rebalancing to avoid over-constraining
            constraint_level = "basic" if request.constraint_level == "strict" else request.constraint_level
            
            # Run full optimization then filter to existing assets
            if request.target_return:
                result = optimizer.minimize_variance(
                    target_return=request.target_return,
                    constraint_type=constraint_level
                )
            else:
                result = optimizer.maximize_sharpe_ratio(
                    constraint_type=constraint_level
                )
            
            # Force the result to only include existing assets
            if result is not None:
                all_weights = result['weights'].to_dict()
                # Keep only existing assets
                existing_weights = {asset: all_weights.get(asset, 0) for asset in valid_assets}
                # Renormalize to sum to 1
                total_weight = sum(existing_weights.values())
                if total_weight > 0:
                    existing_weights = {k: v/total_weight for k, v in existing_weights.items()}
                
                # Update the result
                result['weights'] = pd.Series(existing_weights)
                
        else:  # risk_adjust
            # Adjust portfolio to achieve a target risk level
            # Calculate current portfolio metrics
            current_weights_array = np.array([initial_weights.get(asset, 0) for asset in optimizer.expected_returns.index])
            current_return = np.dot(current_weights_array, optimizer.expected_returns)
            current_volatility = np.sqrt(np.dot(current_weights_array.T, np.dot(optimizer.covariance_matrix, current_weights_array)))
            
            # Determine target based on request
            if request.target_return:
                # If target_return is provided, interpret it as target volatility
                target_volatility = request.target_return
                # Calculate a reasonable return target for this volatility
                target_return = current_return * (target_volatility / current_volatility) if current_volatility > 0 else current_return
            else:
                # Default: reduce risk by 10% while maintaining similar return
                target_volatility = current_volatility * 0.9
                target_return = current_return * 0.95  # Slightly lower return for lower risk
            
            # Use minimize_variance to achieve the target
            result = optimizer.minimize_variance(
                target_return=target_return,
                constraint_type=request.constraint_level
            )
        
        # Process optimization result
        if result is None:
            raise HTTPException(status_code=500, detail=f"Optimization failed for {request.optimization_type} - no result returned")
        
        if 'weights' not in result:
            raise HTTPException(status_code=500, detail=f"Optimization result missing weights for {request.optimization_type}")
            
        optimized_weights_series = result['weights']
        optimized_weights = optimized_weights_series.to_dict()
        
        # Apply custom portfolio logic based on optimization type
        if request.optimization_type == "rebalance":
            # For rebalancing, ensure we only have the original assets
            # (this should already be handled by the pre-filtering above)
            original_assets = set(initial_weights.keys())
            optimized_weights = {k: v for k, v in optimized_weights.items() if k in original_assets}
            
        elif request.optimization_type == "improve" and not request.allow_new_assets:
            # If improving but not allowing new assets, filter to original
            original_assets = set(initial_weights.keys())
            optimized_weights = {k: v for k, v in optimized_weights.items() if k in original_assets}
            # Renormalize weights
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
        
        # If preserve_core_holdings is True, ensure core holdings remain
        if request.preserve_core_holdings and request.core_holdings:
            for symbol in request.core_holdings:
                if symbol in initial_weights:
                    current_weight = optimized_weights.get(symbol, 0)
                    min_weight = initial_weights[symbol] * 0.5  # Preserve at least 50%
                    if current_weight < min_weight:
                        optimized_weights[symbol] = min_weight
        
        # Filter out very small weights
        significant_weights = {k: v for k, v in optimized_weights.items() if v > 0.001}
        
        # Normalize weights to sum to 1
        total_weight = sum(significant_weights.values())
        if total_weight > 0:
            significant_weights = {k: v/total_weight for k, v in significant_weights.items()}
        
        # Calculate metrics for both original and optimized portfolios
        original_metrics = calculate_portfolio_metrics(initial_weights, data['returns'])
        optimized_metrics = calculate_portfolio_metrics(significant_weights, data['returns'])
        
        metrics = PortfolioMetrics(**optimized_metrics)
        
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
                    expected_return=float(asset_row.get('mean_annual_return', 0.0)),
                    volatility=float(asset_row.get('annual_volatility', 0.0)),
                    sharpe_ratio=float(asset_row.get('sharpe_ratio', 0.0)),
                    sector=sector,
                    cluster=int(cluster_id)
                ))
                
                # Aggregate cluster allocation
                cluster_allocation[int(cluster_id)] = cluster_allocation.get(int(cluster_id), 0) + weight
                
                # Aggregate sector allocation
                sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
        
        # Sort holdings by weight
        holdings.sort(key=lambda x: x.weight, reverse=True)
        
        # Calculate improvement metrics with more detailed analysis
        improvement_metrics = {
            "original_sharpe": original_metrics["sharpe_ratio"],
            "optimized_sharpe": optimized_metrics["sharpe_ratio"],
            "sharpe_improvement_pct": ((optimized_metrics["sharpe_ratio"] - original_metrics["sharpe_ratio"]) / abs(original_metrics["sharpe_ratio"]) * 100) if original_metrics["sharpe_ratio"] != 0 else 0,
            "original_return": original_metrics["expected_return"],
            "optimized_return": optimized_metrics["expected_return"],
            "return_change_pct": ((optimized_metrics["expected_return"] - original_metrics["expected_return"]) / abs(original_metrics["expected_return"]) * 100) if original_metrics["expected_return"] != 0 else 0,
            "original_volatility": original_metrics["volatility"],
            "optimized_volatility": optimized_metrics["volatility"],
            "volatility_change_pct": ((optimized_metrics["volatility"] - original_metrics["volatility"]) / original_metrics["volatility"] * 100) if original_metrics["volatility"] != 0 else 0,
            "optimization_type": request.optimization_type,
            "original_assets": len(initial_weights),
            "optimized_assets": len(significant_weights),
            "assets_added": len(set(significant_weights.keys()) - set(initial_weights.keys())),
            "assets_removed": len(set(initial_weights.keys()) - set(significant_weights.keys()))
        }
        
        # Add optimization-specific metrics
        if request.optimization_type == "rebalance":
            improvement_metrics["rebalance_message"] = f"Rebalanced {len(initial_weights)} existing assets without adding new holdings"
        elif request.optimization_type == "improve":
            if improvement_metrics["assets_added"] > 0:
                improvement_metrics["improve_message"] = f"Added {improvement_metrics['assets_added']} new assets to improve portfolio performance"
            else:
                improvement_metrics["improve_message"] = "Optimized existing holdings without adding new assets"
        elif request.optimization_type == "risk_adjust":
            improvement_metrics["risk_adjust_message"] = f"Adjusted portfolio risk level. Volatility changed by {improvement_metrics['volatility_change_pct']:.1f}%"
        
        # Create portfolio response
        portfolio = OptimizedPortfolio(
            portfolio_id=f"custom_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy=f"Custom Portfolio - {request.optimization_type.replace('_', ' ').title()}",
            total_amount=portfolio_value,
            holdings=holdings,
            metrics=metrics,
            cluster_allocation=cluster_allocation,
            sector_allocation=sector_allocation,
            optimization_details={
                "constraint_level": request.constraint_level,
                "return_method": request.return_method,
                "optimization_type": request.optimization_type,
                "max_position_size": request.max_position_size,
                "allow_new_assets": request.allow_new_assets,
                "preserve_core_holdings": request.preserve_core_holdings,
                "optimization_status": result.get('success', True),
                "optimization_message": result.get('message', f'Custom portfolio optimization ({request.optimization_type}) completed successfully'),
                "improvement_metrics": improvement_metrics
            },
            created_at=datetime.now()
        )
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error optimizing custom portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Custom portfolio optimization failed: {str(e)}")

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
            'returns': frontier_df['return'].tolist(),
            'volatilities': frontier_df['volatility'].tolist(),
            'sharpe_ratios': frontier_df['sharpe'].tolist(),
            'max_sharpe': {
                'return': float(frontier_df.loc[frontier_df['sharpe'].idxmax(), 'return']),
                'volatility': float(frontier_df.loc[frontier_df['sharpe'].idxmax(), 'volatility']),
                'sharpe': float(frontier_df['sharpe'].max())
            },
            'min_variance': {
                'return': float(frontier_df.loc[frontier_df['volatility'].idxmin(), 'return']),
                'volatility': float(frontier_df['volatility'].min()),
                'sharpe': float(frontier_df.loc[frontier_df['volatility'].idxmin(), 'sharpe'])
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

# ============================================================================
# SAVED PORTFOLIO MANAGEMENT ENDPOINTS
# ============================================================================

async def save_portfolio(
    request: SavePortfolioRequest,
    current_user: dict = Depends(get_current_user)
) -> SavedPortfolioDetail:
    """Save a portfolio for the current user"""
    try:
        user_email = current_user["email"]
        saved_id = str(uuid.uuid4())
        
        # Prepare portfolio data for storage
        portfolio_data = {
            "saved_id": saved_id,
            "portfolio_name": request.portfolio_name,
            "description": request.description,
            "portfolio_data": request.portfolio_data.dict(),
            "created_at": datetime.now(),
            "last_updated": datetime.now()
        }
        
        # Save to user's portfolio collection
        success = save_user_portfolio(user_email, saved_id, portfolio_data)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save portfolio")
        
        return SavedPortfolioDetail(**portfolio_data)
        
    except Exception as e:
        logger.error(f"Error saving portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save portfolio: {str(e)}")

async def get_saved_portfolios(
    current_user: dict = Depends(get_current_user)
) -> List[SavedPortfolioSummary]:
    """Get all saved portfolios for the current user"""
    try:
        user_email = current_user["email"]
        saved_portfolios = get_user_saved_portfolios(user_email)
        
        summaries = []
        for saved_id, portfolio_data in saved_portfolios.items():
            portfolio_metrics = portfolio_data["portfolio_data"]["metrics"]
            
            summary = SavedPortfolioSummary(
                saved_id=saved_id,
                portfolio_name=portfolio_data["portfolio_name"],
                description=portfolio_data.get("description"),
                strategy=portfolio_data["portfolio_data"]["strategy"],
                total_amount=float(portfolio_data["portfolio_data"]["total_amount"]),
                expected_return=float(portfolio_metrics["expected_return"]),
                volatility=float(portfolio_metrics["volatility"]),
                sharpe_ratio=float(portfolio_metrics["sharpe_ratio"]),
                created_at=portfolio_data["created_at"],
                last_updated=portfolio_data["last_updated"]
            )
            summaries.append(summary)
        
        # Sort by last updated (most recent first)
        summaries.sort(key=lambda x: x.last_updated, reverse=True)
        
        return summaries
        
    except Exception as e:
        logger.error(f"Error retrieving saved portfolios: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve saved portfolios")

async def get_saved_portfolio(
    saved_id: str,
    current_user: dict = Depends(get_current_user)
) -> SavedPortfolioDetail:
    """Get a specific saved portfolio by ID"""
    try:
        user_email = current_user["email"]
        saved_portfolios = get_user_saved_portfolios(user_email)
        
        if saved_id not in saved_portfolios:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio_data = saved_portfolios[saved_id]
        
        # Convert portfolio_data back to proper models
        optimized_portfolio = OptimizedPortfolio(**portfolio_data["portfolio_data"])
        
        return SavedPortfolioDetail(
            saved_id=saved_id,
            portfolio_name=portfolio_data["portfolio_name"],
            description=portfolio_data.get("description"),
            portfolio_data=optimized_portfolio,
            created_at=portfolio_data["created_at"],
            last_updated=portfolio_data["last_updated"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving saved portfolio {saved_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve saved portfolio")

async def update_saved_portfolio(
    saved_id: str,
    request: UpdatePortfolioRequest,
    current_user: dict = Depends(get_current_user)
) -> SavedPortfolioDetail:
    """Update metadata for a saved portfolio"""
    try:
        user_email = current_user["email"]
        
        updates = {}
        if request.portfolio_name is not None:
            updates["portfolio_name"] = request.portfolio_name
        if request.description is not None:
            updates["description"] = request.description
        
        updates["last_updated"] = datetime.now()
        
        success = update_user_portfolio(user_email, saved_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Return updated portfolio
        return await get_saved_portfolio(saved_id, current_user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating saved portfolio {saved_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update saved portfolio")

async def delete_saved_portfolio(
    saved_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete a saved portfolio"""
    try:
        user_email = current_user["email"]
        
        success = delete_user_portfolio(user_email, saved_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return {"message": "Portfolio deleted successfully", "saved_id": saved_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting saved portfolio {saved_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete saved portfolio")

async def compare_portfolios(
    request: PortfolioComparisonRequest,
    current_user: dict = Depends(get_current_user)
) -> PortfolioComparison:
    """Compare multiple saved portfolios"""
    try:
        user_email = current_user["email"]
        saved_portfolios = get_user_saved_portfolios(user_email)
        
        portfolios = []
        comparison_metrics = {
            "expected_return": [],
            "volatility": [],
            "sharpe_ratio": [],
            "total_amount": []
        }
        
        for saved_id in request.portfolio_ids:
            if saved_id not in saved_portfolios:
                raise HTTPException(status_code=404, detail=f"Portfolio {saved_id} not found")
            
            portfolio_data = saved_portfolios[saved_id]
            metrics = portfolio_data["portfolio_data"]["metrics"]
            
            # Create summary
            summary = SavedPortfolioSummary(
                saved_id=saved_id,
                portfolio_name=portfolio_data["portfolio_name"],
                description=portfolio_data.get("description"),
                strategy=portfolio_data["portfolio_data"]["strategy"],
                total_amount=float(portfolio_data["portfolio_data"]["total_amount"]),
                expected_return=float(metrics["expected_return"]),
                volatility=float(metrics["volatility"]),
                sharpe_ratio=float(metrics["sharpe_ratio"]),
                created_at=portfolio_data["created_at"],
                last_updated=portfolio_data["last_updated"]
            )
            portfolios.append(summary)
            
            # Add to comparison metrics
            comparison_metrics["expected_return"].append(float(metrics["expected_return"]))
            comparison_metrics["volatility"].append(float(metrics["volatility"]))
            comparison_metrics["sharpe_ratio"].append(float(metrics["sharpe_ratio"]))
            comparison_metrics["total_amount"].append(float(portfolio_data["portfolio_data"]["total_amount"]))
        
        # Find best performers
        best_performers = {}
        for metric, values in comparison_metrics.items():
            if metric == "volatility":
                # Lower is better for volatility
                best_idx = values.index(min(values))
            else:
                # Higher is better for other metrics
                best_idx = values.index(max(values))
            best_performers[metric] = request.portfolio_ids[best_idx]
        
        return PortfolioComparison(
            portfolios=portfolios,
            comparison_metrics=comparison_metrics,
            best_performers=best_performers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing portfolios: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare portfolios") 