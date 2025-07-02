#!/usr/bin/env python3
"""
Supervised Learning for Return Prediction

This script implements multiple ML models to predict asset returns:
- Linear Regression
- Random Forest 
- Gradient Boosting

Prediction horizons: 1 month, 3 months
"""

import pandas as pd
import numpy as np
import json
import warnings
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

class ReturnPredictionFramework:
    """Framework for predicting asset returns using ML models."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the prediction framework."""
        self.data_dir = data_dir
        self.prices = None
        self.returns = None
        self.models = {}
        
        # Prediction horizons (in trading days)
        self.horizons = {
            '1M': 21,    # 1 month
            '3M': 63,    # 3 months  
        }
        
        # Model configurations
        self.model_configs = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
        }
        
    def load_data(self) -> bool:
        """Load price data and returns."""
        try:
            print("📊 Loading data for return prediction...")
            
            # Load prices and returns
            self.prices = pd.read_pickle(f'{self.data_dir}/cleaned_prices.pkl')
            self.returns = pd.read_pickle(f'{self.data_dir}/log_returns.pkl')
            
            print(f"✅ Loaded data:")
            print(f"   • Prices: {self.prices.shape}")
            print(f"   • Returns: {self.returns.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_prediction_features(self, asset: str, horizon: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features for predicting future returns."""
        print(f"🔧 Creating prediction features for {asset} ({horizon} horizon)...")
        
        # Get asset data
        asset_returns = self.returns[asset].dropna()
        asset_prices = self.prices[asset].dropna()
        
        feature_data = []
        target_data = []
        dates = []
        
        horizon_days = self.horizons[horizon]
        lookback_window = 100  # Look back 100 days for features
        
        for i in range(lookback_window, len(asset_returns) - horizon_days):
            end_date = asset_returns.index[i]
            
            # Historical data windows
            recent_returns = asset_returns.iloc[i-21:i]  # Last month
            medium_returns = asset_returns.iloc[i-63:i]  # Last quarter
            
            # Safe calculations
            def safe_calc(func, series, default=0.0):
                try:
                    val = func(series)
                    return val if pd.notna(val) and np.isfinite(val) else default
                except:
                    return default
            
            # Basic features
            features = {
                'recent_1m_return': safe_calc(lambda x: x.sum(), recent_returns),
                'recent_3m_return': safe_calc(lambda x: x.sum(), medium_returns),
                'recent_1m_volatility': safe_calc(lambda x: x.std() * np.sqrt(252), recent_returns),
                'recent_3m_volatility': safe_calc(lambda x: x.std() * np.sqrt(252), medium_returns),
                'momentum_1m': safe_calc(lambda x: x.iloc[-1] if len(x) > 0 else 0, recent_returns),
                'return_lag_1': asset_returns.iloc[i-1] if i > 0 else 0,
                'return_lag_5': asset_returns.iloc[i-5] if i >= 5 else 0,
                'month': end_date.month,
                'quarter': end_date.quarter
            }
            
            # Market environment (using SPY if available)
            if 'SPY' in self.returns.columns:
                spy_returns = self.returns['SPY'].iloc[i-21:i]
                features['market_return_1m'] = safe_calc(lambda x: x.sum(), spy_returns)
                features['market_volatility'] = safe_calc(lambda x: x.std() * np.sqrt(252), spy_returns)
            else:
                features['market_return_1m'] = 0
                features['market_volatility'] = 0
            
            # Target: Future return over the horizon
            target = safe_calc(lambda x: x.sum(), asset_returns.iloc[i:i+horizon_days])
            
            # Only add if all values are finite
            if all(np.isfinite([v for v in features.values()]) + [target]):
                feature_data.append(features)
                target_data.append(target)
                dates.append(end_date)
        
        # Create DataFrame
        feature_df = pd.DataFrame(feature_data, index=dates)
        target_df = pd.Series(target_data, index=dates, name=f'target_{horizon}')
        
        return feature_df, target_df
    
    def train_models_for_asset(self, asset: str, horizon: str) -> Dict:
        """Train all models for a specific asset and horizon."""
        print(f"\n🤖 Training models for {asset} ({horizon} horizon)")
        print("=" * 50)
        
        # Create features and targets
        features_df, targets = self.create_prediction_features(asset, horizon)
        
        if len(features_df) < 30:  # Minimum data requirement
            print(f"⚠️  Insufficient data for {asset} ({len(features_df)} samples)")
            return {}
        
        # Clean data
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Split data (time series split - use first 80% for training)
        split_point = int(len(features_df) * 0.8)
        X_train = features_df.iloc[:split_point]
        X_test = features_df.iloc[split_point:]
        y_train = targets.iloc[:split_point]
        y_test = targets.iloc[split_point:]
        
        if len(X_train) < 10 or len(X_test) < 3:
            print(f"⚠️  Insufficient data after split for {asset}")
            return {}
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        asset_models = {}
        performance_results = {}
        
        # Train models
        for model_name, model in self.model_configs.items():
            try:
                print(f"   Training {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                # Performance metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                performance = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(features_df.columns, model.feature_importances_))
                    performance['feature_importance'] = feature_importance
                
                asset_models[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'performance': performance
                }
                
                performance_results[model_name] = performance
                
                print(f"      Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error training {model_name}: {e}")
        
        print(f"   ✅ Trained {len(performance_results)} models successfully")
        
        return {
            'models': asset_models,
            'performance': performance_results,
            'features': features_df.columns.tolist(),
            'data_samples': len(features_df)
        }
    
    def train_all_models(self, assets: List[str] = None, horizons: List[str] = None) -> None:
        """Train models for all assets and horizons."""
        if assets is None:
            assets = self.returns.columns.tolist()
        
        if horizons is None:
            horizons = list(self.horizons.keys())
        
        print(f"\n🚀 TRAINING RETURN PREDICTION MODELS")
        print("=" * 80)
        print(f"Assets: {len(assets)}")
        print(f"Horizons: {horizons}")
        print(f"Models: {list(self.model_configs.keys())}")
        
        for asset in assets:
            if asset not in self.returns.columns:
                print(f"⚠️  Asset {asset} not found in returns data")
                continue
            
            self.models[asset] = {}
            
            for horizon in horizons:
                try:
                    result = self.train_models_for_asset(asset, horizon)
                    self.models[asset][horizon] = result
                    
                    if result and 'performance' in result and len(result['performance']) > 0:
                        # Store best model performance
                        best_model = max(result['performance'].items(), 
                                       key=lambda x: x[1].get('test_r2', -np.inf))
                        print(f"   🏆 Best model: {best_model[0]} (R² = {best_model[1].get('test_r2', 0):.3f})")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {asset} - {horizon}: {e}")
        
        print(f"\n✅ Model training complete!")
    
    def create_performance_summary(self) -> pd.DataFrame:
        """Create comprehensive performance summary."""
        print("\n📊 Creating performance summary...")
        
        summary_data = []
        
        for asset in self.models:
            for horizon in self.models[asset]:
                if 'performance' in self.models[asset][horizon]:
                    for model_name, perf in self.models[asset][horizon]['performance'].items():
                        summary_data.append({
                            'Asset': asset,
                            'Horizon': horizon,
                            'Model': model_name,
                            'Test_R2': perf.get('test_r2', np.nan),
                            'Test_RMSE': perf.get('test_rmse', np.nan),
                            'Train_R2': perf.get('train_r2', np.nan),
                            'Overfitting': perf.get('train_r2', 0) - perf.get('test_r2', 0)
                        })
        
        summary_df = pd.DataFrame(summary_data)
        
        if len(summary_df) > 0:
            # Calculate average performance by model
            avg_performance = summary_df.groupby('Model').agg({
                'Test_R2': 'mean',
                'Test_RMSE': 'mean',
                'Overfitting': 'mean'
            }).round(4)
            
            print(f"\n📈 Average Model Performance:")
            print(avg_performance.to_string())
            
            # Best models by horizon
            for horizon in summary_df['Horizon'].unique():
                horizon_data = summary_df[summary_df['Horizon'] == horizon]
                if len(horizon_data) > 0:
                    best_overall = horizon_data.loc[horizon_data['Test_R2'].idxmax()]
                    avg_r2 = horizon_data['Test_R2'].mean()
                    
                    print(f"\n{horizon} Horizon:")
                    print(f"   • Best Model: {best_overall['Model']} ({best_overall['Asset']}) - R² = {best_overall['Test_R2']:.3f}")
                    print(f"   • Average R²: {avg_r2:.3f}")
                    print(f"   • Models Trained: {len(horizon_data)}")
        
        return summary_df
    
    def save_results(self) -> None:
        """Save all models and results."""
        print(f"\n💾 Saving prediction models and results...")
        
        # Save performance summary CSV
        performance_df = self.create_performance_summary()
        performance_df.to_csv(f'{self.data_dir}/ml_prediction_results.csv', index=False)
        
        # Save models summary (only serializable data)
        models_summary = {}
        for asset in self.models:
            models_summary[asset] = {}
            for horizon in self.models[asset]:
                if 'performance' in self.models[asset][horizon]:
                    # Create clean performance dict without model objects
                    clean_performance = {}
                    for model_name, perf in self.models[asset][horizon]['performance'].items():
                        clean_performance[model_name] = {
                            'train_r2': float(perf.get('train_r2', 0)),
                            'test_r2': float(perf.get('test_r2', 0)),
                            'train_rmse': float(perf.get('train_rmse', 0)),
                            'test_rmse': float(perf.get('test_rmse', 0))
                        }
                        # Add feature importance if available
                        if 'feature_importance' in perf:
                            clean_performance[model_name]['feature_importance'] = {
                                k: float(v) for k, v in perf['feature_importance'].items()
                            }
                    
                    models_summary[asset][horizon] = {
                        'performance': clean_performance,
                        'features': self.models[asset][horizon].get('features', []),
                        'data_samples': self.models[asset][horizon].get('data_samples', 0)
                    }
        
        with open(f'{self.data_dir}/return_prediction_models_summary.json', 'w') as f:
            json.dump(models_summary, f, indent=2)
        
        print(f"   ✅ Results saved:")
        print(f"      • {self.data_dir}/ml_prediction_results.csv")
        print(f"      • {self.data_dir}/return_prediction_models_summary.json")
    
    def run_complete_prediction_framework(self, 
                                        sample_assets: List[str] = None,
                                        horizons: List[str] = None) -> None:
        """Run the complete return prediction framework."""
        print("🤖 SUPERVISED LEARNING - RETURN PREDICTION FRAMEWORK")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        # Use sample of assets if not specified
        if sample_assets is None:
            # Select diverse sample
            sample_assets = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'JNJ', 'PG', 'SPY', 'QQQ', 'TLT', 'GLD']
            sample_assets = [asset for asset in sample_assets if asset in self.returns.columns]
        
        if horizons is None:
            horizons = ['1M', '3M']
        
        print(f"🎯 Training models for {len(sample_assets)} assets across {len(horizons)} horizons")
        
        # Train models
        self.train_all_models(sample_assets, horizons)
        
        # Create performance analysis and save results
        self.save_results()
        
        print(f"\n" + "=" * 80)
        print("✅ RETURN PREDICTION FRAMEWORK COMPLETE!")
        print("🔮 Models ready for portfolio optimization!")


def main():
    """Main function to run return prediction framework."""
    framework = ReturnPredictionFramework()
    framework.run_complete_prediction_framework()


if __name__ == "__main__":
    main()