from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

# ============================================================================
# AUTHENTICATION MODELS
# ============================================================================

class UserSignup(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password")
    full_name: str = Field(..., description="User full name")

class UserLogin(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

# ============================================================================
# STOCK AND CLUSTER MODELS
# ============================================================================

class StockInfo(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    cluster: int
    cluster_name: str
    sharpe_ratio: float
    volatility: float
    expected_return: float
    market_cap: Optional[str] = None

class ClusterInfo(BaseModel):
    cluster_id: int
    cluster_name: str
    description: str
    asset_count: int
    avg_sharpe: float
    avg_volatility: float
    role: str
    target_allocation: str

# ============================================================================
# OPTIMIZATION REQUEST MODELS
# ============================================================================

class OptimizationRequest(BaseModel):
    risk_tolerance: str = Field(..., description="low, moderate, high")
    target_return: Optional[float] = Field(None, description="Target annual return (0.1 = 10%)")
    max_position_size: Optional[float] = Field(0.2, description="Maximum weight per asset")
    min_assets: Optional[int] = Field(5, description="Minimum number of assets")
    constraint_level: str = Field("enhanced", description="basic, enhanced, strict")
    return_method: str = Field("ml_enhanced", description="historical, enhanced, ml_enhanced, conservative")
    optimization_objective: str = Field("max_sharpe", description="max_sharpe, min_variance, target_return")
    include_assets: Optional[List[str]] = Field(None, description="Assets to include")
    exclude_assets: Optional[List[str]] = Field(None, description="Assets to exclude")

# ============================================================================
# PORTFOLIO RESPONSE MODELS
# ============================================================================

class PortfolioHolding(BaseModel):
    symbol: str
    weight: float
    allocation_amount: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sector: str
    cluster: int

class PortfolioMetrics(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    effective_assets: float
    concentration_risk: float

class OptimizedPortfolio(BaseModel):
    portfolio_id: str
    strategy: str
    total_amount: float
    holdings: List[PortfolioHolding]
    metrics: PortfolioMetrics
    cluster_allocation: Dict[int, float]
    sector_allocation: Dict[str, float]
    optimization_details: Dict[str, Any]
    created_at: datetime

class HistoricalPerformance(BaseModel):
    dates: List[str]
    portfolio_returns: List[float]
    benchmark_returns: List[float]
    cumulative_portfolio: List[float]
    cumulative_benchmark: List[float]
    rolling_sharpe: List[float]
    rolling_volatility: List[float]

class EfficientFrontierData(BaseModel):
    returns: List[float]
    volatilities: List[float]
    sharpe_ratios: List[float]
    max_sharpe_portfolio: Dict[str, float]
    min_variance_portfolio: Dict[str, float] 