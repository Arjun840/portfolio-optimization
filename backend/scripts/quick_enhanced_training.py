#!/usr/bin/env python3
"""
Quick Enhanced ML Training Demo

Demonstrates all requested features:
✅ GridSearchCV hyperparameter tuning
✅ TimeSeriesSplit cross-validation (temporal order)
✅ 80/20 temporal train/test split
✅ Multiple evaluation metrics
✅ Enhanced feature engineering
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path

warnings.filterwarnings('ignore')

def load_data():
    """Load Yahoo Finance data."""
    price_file = Path("data/price_matrix_latest.pkl")
    returns_file = Path("data/returns_matrix_latest.pkl")
    
    prices = pd.read_pickle(price_file)
    returns = pd.read_pickle(returns_file)
    
    print(f"📊 Loaded: {prices.shape[0]} days, {prices.shape[1]} assets")
    print(f"📅 Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices, returns

def create_features(asset_returns, asset_prices, horizon_days):
    """Create enhanced features."""
    feature_data = []
    target_data = []
    dates = []
    
    start_idx = 126  # 6 months lookback
    
    for i in range(start_idx, len(asset_returns) - horizon_days):
        date = asset_returns.index[i]
        features = {}
        
        # Multiple timeframe returns and volatility
        for days in [5, 10, 21, 63]:
            if i >= days:
                ret = asset_returns.iloc[i-days:i].sum()
                vol = asset_returns.iloc[i-days:i].std() * np.sqrt(252)
                features[f'return_{days}d'] = ret
                features[f'volatility_{days}d'] = vol
                if vol > 0:
                    features[f'sharpe_{days}d'] = (ret * 252) / vol
                else:
                    features[f'sharpe_{days}d'] = 0.0
        
        # Momentum features
        if i >= 21:
            features['momentum_1m'] = asset_returns.iloc[i-21:i].sum()
        if i >= 63:
            features['momentum_3m'] = asset_returns.iloc[i-63:i].sum()
        
        # Mean reversion
        current_price = asset_prices.iloc[i]
        if i >= 21:
            sma21 = asset_prices.iloc[i-21:i].mean()
            features['price_sma21_ratio'] = current_price / sma21 - 1
        
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            if i >= lag:
                features[f'return_lag_{lag}'] = asset_returns.iloc[i-lag]
        
        # Calendar effects
        features['month'] = date.month
        features['quarter'] = date.quarter
        
        # Target
        future_returns = asset_returns.iloc[i:i+horizon_days]
        if len(future_returns) == horizon_days:
            target = future_returns.sum()
            
            if all(np.isfinite(v) for v in features.values()) and np.isfinite(target):
                feature_data.append(features)
                target_data.append(target)
                dates.append(date)
    
    features_df = pd.DataFrame(feature_data, index=dates)
    targets_series = pd.Series(target_data, index=dates)
    
    return features_df, targets_series

def train_with_hyperparameter_tuning():
    """Demonstrate comprehensive ML training."""
    print("🚀 ENHANCED ML TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 80)
    print("Features:")
    print("✅ GridSearchCV hyperparameter tuning")
    print("✅ TimeSeriesSplit cross-validation (respects temporal order)")
    print("✅ 80/20 temporal train/test split (no forward-looking bias)")
    print("✅ Multiple evaluation metrics")
    print("✅ Enhanced feature engineering")
    
    # Load data
    prices, returns = load_data()
    
    # Model configurations with hyperparameter grids
    models = {
        'ridge': {
            'model': Ridge(random_state=42),
            'param_grid': {'alpha': [0.1, 1.0, 10.0, 100.0]}
        },
        'lasso': {
            'model': Lasso(random_state=42, max_iter=2000),
            'param_grid': {'alpha': [0.01, 0.1, 1.0, 10.0]}
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            }
        }
    }
    
    # Test on sample assets
    test_assets = ['AAPL', 'SPY', 'JPM']
    horizon = 21  # 1 month
    
    results = []
    
    for asset in test_assets:
        if asset not in returns.columns:
            continue
            
        print(f"\n🎯 Training {asset} - 1M horizon")
        print("=" * 40)
        
        # Create features
        asset_returns = returns[asset].dropna()
        asset_prices = prices[asset].dropna()
        features_df, targets = create_features(asset_returns, asset_prices, horizon)
        
        print(f"📊 Features: {len(features_df)} samples × {len(features_df.columns)} features")
        
        # Clean data
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.median())
        targets = targets.fillna(targets.median())
        
        # TEMPORAL SPLIT (80/20) - NO FORWARD LOOKING!
        split_idx = int(len(features_df) * 0.8)
        X_train = features_df.iloc[:split_idx].values
        X_test = features_df.iloc[split_idx:].values
        y_train = targets.iloc[:split_idx].values
        y_test = targets.iloc[split_idx:].values
        
        print(f"📊 Train: {len(X_train)} ({features_df.index[0].date()} to {features_df.index[split_idx-1].date()})")
        print(f"📊 Test:  {len(X_test)} ({features_df.index[split_idx].date()} to {features_df.index[-1].date()})")
        
        # Train each model with hyperparameter tuning
        for model_name, config in models.items():
            print(f"\n🤖 Training {model_name}...")
            
            # TimeSeriesSplit cross-validation (respects temporal order)
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', config['model'])
            ])
            
            # Parameter grid
            param_grid = {f'model__{k}': v for k, v in config['param_grid'].items()}
            
            # GridSearchCV with TimeSeriesSplit
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,  # Temporal cross-validation!
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Fit with hyperparameter tuning
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate
            train_pred = best_model.predict(X_train)
            test_pred = best_model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Hit ratio (directional accuracy)
            hit_ratio = np.mean(np.sign(y_test) == np.sign(test_pred))
            
            print(f"  🔍 Best params: {grid_search.best_params_}")
            print(f"  📈 Best CV score: {grid_search.best_score_:.4f}")
            print(f"  📈 Test R²: {test_r2:.3f}")
            print(f"  📉 Test RMSE: {test_rmse:.4f}")
            print(f"  🎯 Hit Ratio: {hit_ratio:.3f}")
            print(f"  📊 Overfitting: {(train_r2 - test_r2):.3f}")
            
            results.append({
                'Asset': asset,
                'Model': model_name,
                'Test_R2': test_r2,
                'Test_RMSE': test_rmse,
                'Hit_Ratio': hit_ratio,
                'Best_Params': grid_search.best_params_,
                'CV_Score': grid_search.best_score_
            })
    
    # Summary
    print(f"\n📊 TRAINING SUMMARY")
    print("=" * 80)
    
    if results:
        results_df = pd.DataFrame(results)
        
        print("🏆 BEST MODELS BY ASSET:")
        for asset in results_df['Asset'].unique():
            asset_results = results_df[results_df['Asset'] == asset]
            best_model = asset_results.loc[asset_results['Test_R2'].idxmax()]
            print(f"  {asset}: {best_model['Model']} (R² = {best_model['Test_R2']:.3f})")
        
        print(f"\n📈 AVERAGE PERFORMANCE BY MODEL:")
        model_avg = results_df.groupby('Model').agg({
            'Test_R2': ['mean', 'std'],
            'Hit_Ratio': 'mean'
        }).round(3)
        
        for model in model_avg.index:
            r2_mean = model_avg.loc[model, ('Test_R2', 'mean')]
            r2_std = model_avg.loc[model, ('Test_R2', 'std')]
            hit_mean = model_avg.loc[model, ('Hit_Ratio', 'mean')]
            print(f"  {model:15s}: R² = {r2_mean:.3f} ± {r2_std:.3f}, Hit = {hit_mean:.3f}")
    
    print(f"\n🎉 ENHANCED ML TRAINING COMPLETE!")
    print("All requested features successfully implemented!")

if __name__ == "__main__":
    train_with_hyperparameter_tuning()
