# 🚀 Portfolio Optimization API - Quick Start Guide

## ⚡ Start the Server

```bash
cd backend
source venv/bin/activate
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

## 🔗 Access Points

- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📋 Quick Test Sequence

### 1. Check Server Health
```bash
curl http://localhost:8000/
```

### 2. Register User & Get Token
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123", "full_name": "Test User"}'
```

### 3. Optimize Portfolio (Replace TOKEN with actual JWT)
```bash
TOKEN="your_jwt_token_here"

curl -X POST "http://localhost:8000/optimize?portfolio_value=100000" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "risk_tolerance": "moderate",
    "constraint_level": "enhanced",
    "return_method": "ml_enhanced",
    "optimization_objective": "max_sharpe"
  }'
```

## 🎯 Expected Results

You should get a portfolio with:
- **Expected Return**: ~19.8%
- **Volatility**: ~14.2%
- **Sharpe Ratio**: ~1.39
- **Max Position**: 20% (enhanced constraints)
- **Effective Assets**: 6-8 assets

## 🔧 Optimization Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `risk_tolerance` | `low`, `moderate`, `high` | Risk appetite |
| `constraint_level` | `basic`, `enhanced`, `strict` | Constraint strictness |
| `return_method` | `historical`, `ml_enhanced`, `conservative` | Return estimation |
| `optimization_objective` | `max_sharpe`, `min_variance` | Optimization goal |
| `portfolio_value` | Any positive number | Total investment amount |

## 📚 Full Documentation

Visit http://localhost:8000/docs for complete API documentation with interactive testing interface. 