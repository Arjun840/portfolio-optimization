#!/usr/bin/env python3
"""
Enhanced ML Training Framework with Hyperparameter Tuning

This script implements comprehensive machine learning model training with:
- GridSearchCV hyperparameter tuning
- TimeSeriesSplit cross-validation (respects temporal order)
- Multiple evaluation metrics
- Proper temporal validation (80/20 split)
- Feature importance analysis
- Model comparison and selection

Models:
- Linear/Ridge/Lasso Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression
- XGBoost (if available)

Evaluation Metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Sharpe Ratio (for returns)
- Information Ratio
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ML Libraries
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, cross_val_score, 
    validation_curve, learning_curve
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - skipping XGBoost models")

warnings.filterwarnings('ignore')

class EnhancedMLTrainer:
    """Enhanced ML trainer with comprehensive hyperparameter tuning and validation."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the enhanced ML trainer."""
        self.data_dir = data_dir
        self.prices = None
        self.returns = None
        self.features = None
        self.models = {}
        self.best_models = {}
        self.cv_results = {}
        self.feature_importance = {}
        
        # Prediction horizons (in trading days)
        self.horizons = {
            '1M': 21,    # 1 month
            '3M': 63,    # 3 months
            '6M': 126,   # 6 months
        }
        
        # Cross-validation setup
        self.n_splits = 5  # 5-fold time series split
        self.test_size = 0.2  # 20% for final testing
        
        # Model configurations with hyperparameter grids
        self.model_configs = self._setup_model_configs()
        
        # Custom scoring functions
        self.scoring_functions = self._setup_scoring_functions()
        
        print(f"🤖 Enhanced ML Trainer initialized")
        print(f"   • Models: {len(self.model_configs)}")
        print(f"   • Horizons: {list(self.horizons.keys())}")
        print(f"   • CV Folds: {self.n_splits}")
        print(f"   • Test Split: {self.test_size:.0%}")
    
    def _setup_model_configs(self) -> Dict:
        """Setup model configurations with hyperparameter grids."""
        configs = {
            'linear_regression': {
                'model': LinearRegression(),
                'param_grid': {},
                'type': 'linear'
            },
            'ridge_regression': {
                'model': Ridge(random_state=42),
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
                },
                'type': 'linear'
            },
            'lasso_regression': {
                'model': Lasso(random_state=42, max_iter=2000),
                'param_grid': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                },
                'type': 'linear'
            },
            'elastic_net': {
                'model': ElasticNet(random_state=42, max_iter=2000),
                'param_grid': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                'type': 'linear'
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'type': 'ensemble'
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                'type': 'ensemble'
            },
            'svr': {
                'model': SVR(),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'linear']
                },
                'type': 'svm'
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'type': 'ensemble'
            }
        
        return configs
    
    def _setup_scoring_functions(self) -> Dict:
        """Setup custom scoring functions."""
        def sharpe_ratio_score(y_true, y_pred):
            """Calculate Sharpe ratio for predictions."""
            if len(y_pred) < 2:
                return 0.0
            
            excess_returns = y_pred - np.mean(y_pred)
            if np.std(excess_returns) == 0:
                return 0.0
            
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        def information_ratio_score(y_true, y_pred):
            """Calculate information ratio."""
            tracking_error = y_true - y_pred
            if np.std(tracking_error) == 0:
                return 0.0
            
            return np.mean(tracking_error) / np.std(tracking_error)
        
        def neg_mse_score(y_true, y_pred):
            """Negative MSE for maximization in GridSearchCV."""
            return -mean_squared_error(y_true, y_pred)
        
        return {
            'neg_mse': make_scorer(neg_mse_score),
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'sharpe_ratio': make_scorer(sharpe_ratio_score),
            'information_ratio': make_scorer(information_ratio_score)
        }
    
    def load_data(self) -> bool:
        """Load price and returns data."""
        try:
            print("\n📊 Loading data...")
            
            # Load latest data
            price_file = Path(self.data_dir) / "price_matrix_latest.pkl"
            returns_file = Path(self.data_dir) / "returns_matrix_latest.pkl"
            
            if price_file.exists() and returns_file.exists():
                self.prices = pd.read_pickle(price_file)
                self.returns = pd.read_pickle(returns_file)
            else:
                # Fallback to CSV
                price_file = Path(self.data_dir) / "price_matrix_latest.csv"
                returns_file = Path(self.data_dir) / "returns_matrix_latest.csv"
                
                self.prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
                self.returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            
            print(f"✅ Data loaded successfully:")
            print(f"   • Price data: {self.prices.shape}")
            print(f"   • Returns data: {self.returns.shape}")
            print(f"   • Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
            print(f"   • Assets: {len(self.prices.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_enhanced_features(self, asset: str, horizon: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Create comprehensive features for prediction."""
        if asset not in self.returns.columns:
            raise ValueError(f"Asset {asset} not found in returns data")
        
        horizon_days = self.horizons[horizon]
        asset_returns = self.returns[asset].dropna()
        asset_prices = self.prices[asset].dropna()
        
        # Market features (using SPY as proxy)
        market_asset = 'SPY' if 'SPY' in self.returns.columns else self.returns.columns[0]
        market_returns = self.returns[market_asset].dropna()
        
        feature_data = []
        target_data = []
        dates = []
        
        # Ensure we have enough data
        min_lookback = 126  # 6 months lookback
        start_idx = max(min_lookback, horizon_days)
        
        for i in range(start_idx, len(asset_returns) - horizon_days):
            end_date = asset_returns.index[i]
            
            # Historical return features (multiple timeframes)
            features = {}
            
            # Recent returns (different periods)
            for period, days in [(5, 5), (10, 10), (21, 21), (63, 63), (126, 126)]:
                if i >= days:
                    period_return = asset_returns.iloc[i-days:i].sum()
                    features[f'return_{days}d'] = period_return
                    
                    # Volatility
                    period_vol = asset_returns.iloc[i-days:i].std() * np.sqrt(252)
                    features[f'volatility_{days}d'] = period_vol
                    
                    # Sharpe ratio
                    if period_vol > 0:
                        features[f'sharpe_{days}d'] = (period_return * 252) / period_vol
                    else:
                        features[f'sharpe_{days}d'] = 0.0
            
            # Momentum features
            if i >= 21:
                features['momentum_1m'] = asset_returns.iloc[i-21:i].sum()
            if i >= 63:
                features['momentum_3m'] = asset_returns.iloc[i-63:i].sum()
            if i >= 126:
                features['momentum_6m'] = asset_returns.iloc[i-126:i].sum()
            
            # Mean reversion features
            if i >= 21:
                sma_21 = asset_prices.iloc[i-21:i].mean()
                features['price_to_sma21'] = asset_prices.iloc[i] / sma_21 - 1
            
            if i >= 63:
                sma_63 = asset_prices.iloc[i-63:i].mean()
                features['price_to_sma63'] = asset_prices.iloc[i] / sma_63 - 1
            
            # Technical indicators
            if i >= 21:
                # RSI approximation
                gains = asset_returns.iloc[i-21:i].clip(lower=0).sum()
                losses = (-asset_returns.iloc[i-21:i].clip(upper=0)).sum()
                if losses > 0:
                    rs = gains / losses
                    features['rsi_21'] = 100 - (100 / (1 + rs))
                else:
                    features['rsi_21'] = 100
            
            # Market-relative features
            if i >= 21:
                asset_return_21 = asset_returns.iloc[i-21:i].sum()
                market_return_21 = market_returns.iloc[i-21:i].sum()
                features['excess_return_21d'] = asset_return_21 - market_return_21
                
                # Beta approximation
                asset_vol = asset_returns.iloc[i-21:i].std()
                market_vol = market_returns.iloc[i-21:i].std()
                if market_vol > 0:
                    corr = asset_returns.iloc[i-21:i].corr(market_returns.iloc[i-21:i])
                    features['beta_21d'] = corr * (asset_vol / market_vol)
                else:
                    features['beta_21d'] = 1.0
            
            # Lagged returns
            for lag in [1, 2, 3, 5, 10]:
                if i >= lag:
                    features[f'return_lag_{lag}'] = asset_returns.iloc[i-lag]
            
            # Volatility clustering
            if i >= 21:
                recent_vol = asset_returns.iloc[i-5:i].std()
                historical_vol = asset_returns.iloc[i-21:i-5].std()
                if historical_vol > 0:
                    features['vol_ratio'] = recent_vol / historical_vol
                else:
                    features['vol_ratio'] = 1.0
            
            # Calendar effects
            features['month'] = end_date.month
            features['quarter'] = end_date.quarter
            features['day_of_week'] = end_date.dayofweek
            features['is_month_end'] = 1 if end_date.day >= 25 else 0
            
            # Market environment features
            if i >= 63:
                market_vol_3m = market_returns.iloc[i-63:i].std() * np.sqrt(252)
                features['market_volatility_3m'] = market_vol_3m
                features['market_return_3m'] = market_returns.iloc[i-63:i].sum()
            
            # Target: Future return over the horizon
            future_returns = asset_returns.iloc[i:i+horizon_days]
            if len(future_returns) == horizon_days:
                target = future_returns.sum()
                
                # Only include if all features are finite
                if all(np.isfinite(v) for v in features.values()) and np.isfinite(target):
                    feature_data.append(features)
                    target_data.append(target)
                    dates.append(end_date)
        
        if len(feature_data) == 0:
            raise ValueError(f"No valid feature data created for {asset}")
        
        # Create DataFrames
        features_df = pd.DataFrame(feature_data, index=dates)
        targets_series = pd.Series(target_data, index=dates, name=f'target_{horizon}')
        
        print(f"   Created {len(features_df)} samples with {len(features_df.columns)} features")
        
        return features_df, targets_series
    
    def perform_hyperparameter_tuning(self, model_name: str, model_config: Dict,
                                    X_train: np.ndarray, y_train: np.ndarray) -> Tuple[object, Dict]:
        """Perform hyperparameter tuning using GridSearchCV with TimeSeriesSplit."""
        print(f"   🔍 Tuning hyperparameters for {model_name}...")
        
        # Setup time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_config['model'])
        ])
        
        # Prepare parameter grid for pipeline
        param_grid = {}
        for param, values in model_config['param_grid'].items():
            param_grid[f'model__{param}'] = values
        
        # Perform grid search
        if len(param_grid) > 0:
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',  # Primary metric
                n_jobs=-1,
                verbose=0,
                return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Extract results
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': {
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                }
            }
            
            print(f"      ✅ Best params: {grid_search.best_params_}")
            print(f"      ✅ Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # No hyperparameters to tune
            best_model = pipeline
            best_model.fit(X_train, y_train)
            tuning_results = {'best_params': {}, 'best_score': None}
        
        return best_model, tuning_results
    
    def evaluate_model_comprehensive(self, model: object, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray, 
                                   model_name: str) -> Dict:
        """Comprehensive model evaluation with multiple metrics."""
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Basic metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
        }
        
        # RMSE
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
        
        # Overfitting measure
        metrics['overfitting'] = metrics['train_r2'] - metrics['test_r2']
        
        # Financial metrics for returns
        if len(test_pred) > 1:
            # Sharpe ratio of predictions
            pred_sharpe = np.mean(test_pred) / np.std(test_pred) * np.sqrt(252) if np.std(test_pred) > 0 else 0
            metrics['pred_sharpe_ratio'] = pred_sharpe
            
            # Information ratio
            tracking_error = y_test - test_pred
            info_ratio = np.mean(tracking_error) / np.std(tracking_error) if np.std(tracking_error) > 0 else 0
            metrics['information_ratio'] = info_ratio
            
            # Hit ratio (percentage of correct direction predictions)
            if len(y_test) > 1:
                actual_direction = np.sign(y_test)
                pred_direction = np.sign(test_pred)
                hit_ratio = np.mean(actual_direction == pred_direction)
                metrics['hit_ratio'] = hit_ratio
        
        # Feature importance (for tree-based models)
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            metrics['feature_importance'] = model.named_steps['model'].feature_importances_
        elif hasattr(model.named_steps['model'], 'coef_'):
            metrics['feature_importance'] = np.abs(model.named_steps['model'].coef_)
        
        return metrics, train_pred, test_pred
    
    def train_models_for_asset(self, asset: str, horizon: str) -> Dict:
        """Train and tune all models for a specific asset and horizon."""
        print(f"\n🎯 Training models for {asset} ({horizon} horizon)")
        print("=" * 60)
        
        try:
            # Create features and targets
            features_df, targets = self.create_enhanced_features(asset, horizon)
            
            if len(features_df) < 100:  # Minimum data requirement
                print(f"⚠️  Insufficient data for {asset} ({len(features_df)} samples)")
                return {}
            
            # Clean data
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(features_df.median())
            
            # Remove any remaining NaN targets
            valid_mask = ~targets.isna()
            features_df = features_df[valid_mask]
            targets = targets[valid_mask]
            
            # Temporal split (80% train, 20% test)
            split_point = int(len(features_df) * (1 - self.test_size))
            X_train = features_df.iloc[:split_point].values
            X_test = features_df.iloc[split_point:].values
            y_train = targets.iloc[:split_point].values
            y_test = targets.iloc[split_point:].values
            
            print(f"   📊 Data split: Train={len(X_train)}, Test={len(X_test)}")
            print(f"   📅 Train period: {features_df.index[0].date()} to {features_df.index[split_point-1].date()}")
            print(f"   📅 Test period: {features_df.index[split_point].date()} to {features_df.index[-1].date()}")
            
            asset_results = {
                'models': {},
                'performance': {},
                'cv_results': {},
                'features': features_df.columns.tolist(),
                'data_info': {
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'train_start': features_df.index[0],
                    'train_end': features_df.index[split_point-1],
                    'test_start': features_df.index[split_point],
                    'test_end': features_df.index[-1],
                    'n_features': len(features_df.columns)
                }
            }
            
            # Train each model
            for model_name, model_config in self.model_configs.items():
                try:
                    print(f"\n   🤖 Training {model_name}...")
                    
                    # Hyperparameter tuning
                    best_model, tuning_results = self.perform_hyperparameter_tuning(
                        model_name, model_config, X_train, y_train
                    )
                    
                    # Comprehensive evaluation
                    metrics, train_pred, test_pred = self.evaluate_model_comprehensive(
                        best_model, X_train, y_train, X_test, y_test, model_name
                    )
                    
                    # Store results
                    asset_results['models'][model_name] = best_model
                    asset_results['performance'][model_name] = metrics
                    asset_results['cv_results'][model_name] = tuning_results
                    
                    # Print key metrics
                    print(f"      📈 Test R²: {metrics['test_r2']:.3f}")
                    print(f"      📉 Test RMSE: {metrics['test_rmse']:.4f}")
                    print(f"      🎯 Hit Ratio: {metrics.get('hit_ratio', 0):.3f}")
                    
                except Exception as e:
                    print(f"      ❌ Error training {model_name}: {e}")
                    continue
            
            # Find best model
            if asset_results['performance']:
                best_model_name = max(
                    asset_results['performance'].items(),
                    key=lambda x: x[1]['test_r2']
                )[0]
                print(f"\n   🏆 Best model: {best_model_name} (R² = {asset_results['performance'][best_model_name]['test_r2']:.3f})")
                asset_results['best_model'] = best_model_name
            
            return asset_results
            
        except Exception as e:
            print(f"❌ Error processing {asset} - {horizon}: {e}")
            return {}
    
    def train_all_models(self, assets: List[str] = None, horizons: List[str] = None) -> None:
        """Train models for all specified assets and horizons."""
        if assets is None:
            # Use a representative sample
            sample_assets = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'JPM', 'JNJ', 'PG', 'SPY', 'QQQ', 'TLT', 'GLD']
            assets = [asset for asset in sample_assets if asset in self.returns.columns]
        
        if horizons is None:
            horizons = ['1M', '3M']  # Start with shorter horizons
        
        print(f"\n🚀 ENHANCED ML TRAINING WITH HYPERPARAMETER TUNING")
        print("=" * 80)
        print(f"📊 Training Configuration:")
        print(f"   • Assets: {len(assets)} ({', '.join(assets)})")
        print(f"   • Horizons: {horizons}")
        print(f"   • Models: {list(self.model_configs.keys())}")
        print(f"   • CV Folds: {self.n_splits}")
        print(f"   • Test Split: {self.test_size:.0%}")
        
        self.results = {}
        training_summary = {}
        
        for asset in assets:
            if asset not in self.returns.columns:
                print(f"⚠️  Asset {asset} not found in returns data")
                continue
            
            self.results[asset] = {}
            training_summary[asset] = {}
            
            for horizon in horizons:
                print(f"\n{'='*20} {asset} - {horizon} {'='*20}")
                
                try:
                    result = self.train_models_for_asset(asset, horizon)
                    self.results[asset][horizon] = result
                    
                    # Summary for this asset-horizon
                    if result and 'performance' in result and result['performance']:
                        summary = {}
                        for model_name, perf in result['performance'].items():
                            summary[model_name] = {
                                'test_r2': perf['test_r2'],
                                'test_rmse': perf['test_rmse'],
                                'hit_ratio': perf.get('hit_ratio', 0)
                            }
                        training_summary[asset][horizon] = summary
                        
                except Exception as e:
                    print(f"❌ Error processing {asset} - {horizon}: {e}")
                    continue
        
        self.training_summary = training_summary
        print(f"\n✅ Enhanced ML training complete!")
        self._print_training_summary()
    
    def _print_training_summary(self) -> None:
        """Print comprehensive training summary."""
        print(f"\n📊 TRAINING SUMMARY")
        print("=" * 80)
        
        # Collect all results for analysis
        all_results = []
        for asset, horizons in self.training_summary.items():
            for horizon, models in horizons.items():
                for model_name, metrics in models.items():
                    all_results.append({
                        'Asset': asset,
                        'Horizon': horizon,
                        'Model': model_name,
                        'Test_R2': metrics['test_r2'],
                        'Test_RMSE': metrics['test_rmse'],
                        'Hit_Ratio': metrics['hit_ratio']
                    })
        
        if not all_results:
            print("No results to summarize.")
            return
        
        df = pd.DataFrame(all_results)
        
        # Best models by metric
        print("\n🏆 TOP PERFORMERS:")
        
        # Best R² scores
        best_r2 = df.nlargest(5, 'Test_R2')[['Asset', 'Horizon', 'Model', 'Test_R2']]
        print("\n   Best R² Scores:")
        for _, row in best_r2.iterrows():
            print(f"      {row['Asset']} ({row['Horizon']}) - {row['Model']}: {row['Test_R2']:.3f}")
        
        # Best Hit Ratios
        best_hit = df.nlargest(5, 'Hit_Ratio')[['Asset', 'Horizon', 'Model', 'Hit_Ratio']]
        print("\n   Best Hit Ratios:")
        for _, row in best_hit.iterrows():
            print(f"      {row['Asset']} ({row['Horizon']}) - {row['Model']}: {row['Hit_Ratio']:.3f}")
        
        # Model performance summary
        print("\n📈 AVERAGE MODEL PERFORMANCE:")
        model_avg = df.groupby('Model').agg({
            'Test_R2': ['mean', 'std'],
            'Test_RMSE': ['mean', 'std'],
            'Hit_Ratio': ['mean', 'std']
        }).round(3)
        
        for model in model_avg.index:
            r2_mean, r2_std = model_avg.loc[model, ('Test_R2', 'mean')], model_avg.loc[model, ('Test_R2', 'std')]
            hit_mean, hit_std = model_avg.loc[model, ('Hit_Ratio', 'mean')], model_avg.loc[model, ('Hit_Ratio', 'std')]
            print(f"   {model:20s}: R²={r2_mean:.3f}±{r2_std:.3f}, Hit={hit_mean:.3f}±{hit_std:.3f}")
    
    def save_results(self) -> None:
        """Save all training results and models."""
        print(f"\n💾 Saving enhanced ML training results...")
        
        if not hasattr(self, 'results'):
            print("No results to save.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create results directory
        results_dir = Path(self.data_dir) / 'enhanced_ml_results'
        results_dir.mkdir(exist_ok=True)
        
        # Save performance summary
        if hasattr(self, 'training_summary'):
            all_results = []
            for asset, horizons in self.training_summary.items():
                for horizon, models in horizons.items():
                    for model_name, metrics in models.items():
                        result_row = {
                            'Asset': asset,
                            'Horizon': horizon,
                            'Model': model_name,
                            'Test_R2': metrics['test_r2'],
                            'Test_RMSE': metrics['test_rmse'],
                            'Hit_Ratio': metrics['hit_ratio']
                        }
                        
                        # Add CV results if available
                        if (asset in self.results and horizon in self.results[asset] and 
                            'cv_results' in self.results[asset][horizon] and
                            model_name in self.results[asset][horizon]['cv_results']):
                            cv_result = self.results[asset][horizon]['cv_results'][model_name]
                            result_row['CV_Score'] = cv_result.get('best_score', np.nan)
                        
                        all_results.append(result_row)
            
            if all_results:
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(results_dir / f'enhanced_ml_results_{timestamp}.csv', index=False)
                results_df.to_csv(results_dir / 'enhanced_ml_results_latest.csv', index=False)
        
        # Save detailed results (serializable parts only)
        serializable_results = {}
        for asset, horizons in self.results.items():
            serializable_results[asset] = {}
            for horizon, data in horizons.items():
                if data:  # Only save non-empty results
                    serializable_results[asset][horizon] = {
                        'performance': data.get('performance', {}),
                        'cv_results': {
                            model_name: {
                                'best_params': cv_data.get('best_params', {}),
                                'best_score': cv_data.get('best_score')
                            } for model_name, cv_data in data.get('cv_results', {}).items()
                        },
                        'features': data.get('features', []),
                        'data_info': data.get('data_info', {}),
                        'best_model': data.get('best_model')
                    }
        
        # Save to JSON
        with open(results_dir / f'enhanced_ml_detailed_{timestamp}.json', 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        with open(results_dir / 'enhanced_ml_detailed_latest.json', 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save models (pickle)
        models_to_save = {}
        for asset, horizons in self.results.items():
            models_to_save[asset] = {}
            for horizon, data in horizons.items():
                if 'models' in data:
                    models_to_save[asset][horizon] = data['models']
        
        if models_to_save:
            with open(results_dir / f'enhanced_ml_models_{timestamp}.pkl', 'wb') as f:
                pickle.dump(models_to_save, f)
            
            with open(results_dir / 'enhanced_ml_models_latest.pkl', 'wb') as f:
                pickle.dump(models_to_save, f)
        
        print(f"✅ Results saved to {results_dir}:")
        print(f"   • Performance CSV: enhanced_ml_results_latest.csv")
        print(f"   • Detailed JSON: enhanced_ml_detailed_latest.json")
        print(f"   • Models pickle: enhanced_ml_models_latest.pkl")
    
    def run_enhanced_training(self, assets: List[str] = None, horizons: List[str] = None) -> None:
        """Run the complete enhanced ML training pipeline."""
        print("🚀 ENHANCED ML TRAINING WITH HYPERPARAMETER TUNING")
        print("=" * 80)
        print("Features:")
        print("✅ GridSearchCV hyperparameter tuning")
        print("✅ TimeSeriesSplit cross-validation (temporal order preserved)")
        print("✅ 80/20 temporal train/test split")
        print("✅ Comprehensive evaluation metrics")
        print("✅ Feature engineering with 30+ features")
        print("✅ Multiple model types with optimized parameters")
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data. Exiting.")
            return
        
        # Train models
        self.train_all_models(assets, horizons)
        
        # Save results
        self.save_results()
        
        print(f"\n🎉 ENHANCED ML TRAINING COMPLETE!")
        print("🔮 Optimized models ready for portfolio optimization!")


def main():
    """Main function to run enhanced ML training."""
    print("🚀 Starting Enhanced ML Training with Yahoo Finance Data")
    print("=" * 80)
    
    # Initialize trainer
    trainer = EnhancedMLTrainer(data_dir="data")
    
    # Run training with sample assets
    sample_assets = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'JPM', 'JNJ', 'SPY', 'QQQ', 'TLT', 'GLD']
    horizons = ['1M', '3M']
    
    trainer.run_enhanced_training(assets=sample_assets, horizons=horizons)


if __name__ == "__main__":
    main() 