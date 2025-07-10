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

class CustomHolding(BaseModel):
    symbol: str
    weight: float = Field(..., ge=0, le=1, description="Weight as decimal (0.1 = 10%)")

class CustomPortfolioRequest(BaseModel):
    holdings: List[CustomHolding] = Field(..., description="Initial portfolio holdings")
    optimization_type: str = Field("improve", description="improve (add new assets to optimize), rebalance (only adjust existing assets), risk_adjust (target specific risk level)")
    target_return: Optional[float] = Field(None, description="For improve/rebalance: target annual return (0.1 = 10%). For risk_adjust: target volatility (0.15 = 15%)")
    max_position_size: Optional[float] = Field(0.3, description="Maximum weight per asset")
    constraint_level: str = Field("enhanced", description="basic, enhanced, strict")
    return_method: str = Field("ml_enhanced", description="historical, enhanced, ml_enhanced, conservative")
    allow_new_assets: bool = Field(True, description="Allow adding new assets to portfolio (ignored for rebalance)")
    preserve_core_holdings: bool = Field(False, description="Keep some holdings fixed")
    core_holdings: Optional[List[str]] = Field(None, description="Assets to keep in portfolio")

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

# ============================================================================
# SAVED PORTFOLIO MODELS
# ============================================================================

class SavePortfolioRequest(BaseModel):
    portfolio_name: str = Field(..., description="User-friendly name for the portfolio")
    description: Optional[str] = Field(None, description="Optional description")
    portfolio_data: OptimizedPortfolio = Field(..., description="Portfolio to save")

class SavedPortfolioSummary(BaseModel):
    saved_id: str
    portfolio_name: str
    description: Optional[str]
    strategy: str
    total_amount: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    created_at: datetime
    last_updated: datetime

class SavedPortfolioDetail(BaseModel):
    saved_id: str
    portfolio_name: str
    description: Optional[str]
    portfolio_data: OptimizedPortfolio
    created_at: datetime
    last_updated: datetime

class UpdatePortfolioRequest(BaseModel):
    portfolio_name: Optional[str] = Field(None, description="New name for portfolio")
    description: Optional[str] = Field(None, description="New description")

class PortfolioComparisonRequest(BaseModel):
    portfolio_ids: List[str] = Field(..., description="List of saved portfolio IDs to compare")

class PortfolioComparison(BaseModel):
    portfolios: List[SavedPortfolioSummary]
    comparison_metrics: Dict[str, List[float]]
    best_performers: Dict[str, str]  # metric -> portfolio_id 