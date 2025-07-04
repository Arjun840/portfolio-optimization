from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import API components
from api.models import *
from api.endpoints import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio Optimization API",
    description="Advanced portfolio optimization and asset allocation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/", summary="API Health Check")
async def root():
    """Root endpoint for API health check"""
    return {
        "message": "Portfolio Optimization API",
        "status": "active",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "authentication": ["/auth/signup", "/auth/login"],
            "data": ["/data/stocks", "/data/clusters"],
            "optimization": ["/optimize", "/data/efficient-frontier"],
            "analysis": ["/data/historical/{portfolio_id}"]
        }
    }

# Authentication endpoints
@app.post("/auth/signup", response_model=Token, summary="User Registration")
async def signup_endpoint(user_data: UserSignup):
    return await signup(user_data)

@app.post("/auth/login", response_model=Token, summary="User Login")
async def login_endpoint(user_data: UserLogin):
    return await login(user_data)

# Data endpoints
@app.get("/data/stocks", response_model=List[StockInfo], summary="Get Available Stocks")
async def get_stocks_endpoint(current_user: dict = Depends(get_current_user)):
    return await get_stocks(current_user)

@app.get("/data/clusters", response_model=List[ClusterInfo], summary="Get Cluster Information")
async def get_clusters_endpoint(current_user: dict = Depends(get_current_user)):
    return await get_clusters(current_user)

# Portfolio optimization endpoints
@app.post("/optimize", response_model=OptimizedPortfolio, summary="Optimize Portfolio")
async def optimize_portfolio_endpoint(
    request: OptimizationRequest,
    portfolio_value: float = Query(100000, description="Total portfolio value in USD"),
    current_user: dict = Depends(get_current_user)
):
    return await optimize_portfolio(request, portfolio_value, current_user)

@app.get("/data/historical/{portfolio_id}", response_model=HistoricalPerformance, summary="Get Historical Performance")
async def get_historical_performance_endpoint(
    portfolio_id: str,
    benchmark: str = Query("SPY", description="Benchmark symbol"),
    current_user: dict = Depends(get_current_user)
):
    return await get_historical_performance(portfolio_id, benchmark, current_user)

@app.get("/data/efficient-frontier", response_model=EfficientFrontierData, summary="Get Efficient Frontier Data")
async def get_efficient_frontier_endpoint(
    constraint_level: str = Query("enhanced", description="Constraint level: basic, enhanced, strict"),
    return_method: str = Query("ml_enhanced", description="Return method: historical, enhanced, ml_enhanced, conservative"),
    current_user: dict = Depends(get_current_user)
):
    return await get_efficient_frontier(constraint_level, return_method, current_user)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
