#!/usr/bin/env python3
"""
Data Storage Manager for Portfolio Optimization

This module provides a flexible data storage system that supports:
- Local files (CSV, Pickle, Parquet)
- SQLite database for local development
- PostgreSQL for production
- Performance optimization and caching
- Easy migration between storage types
"""

import pandas as pd
import numpy as np
import os
import sqlite3
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
import warnings

# Optional dependencies with graceful fallbacks
try:
    import psycopg2
    from sqlalchemy import create_engine, text
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    print("PostgreSQL support not available. Install psycopg2-binary for PostgreSQL support.")

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("Parquet support not available. Install pyarrow for better performance.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataStorageBase(ABC):
    """Abstract base class for data storage backends."""
    
    @abstractmethod
    def save_price_data(self, prices: pd.DataFrame, metadata: Dict = None) -> bool:
        """Save price data to storage."""
        pass
    
    @abstractmethod
    def load_price_data(self, start_date: str = None, end_date: str = None, 
                       assets: List[str] = None) -> Optional[pd.DataFrame]:
        """Load price data from storage."""
        pass
    
    @abstractmethod
    def save_returns_data(self, returns: pd.DataFrame, metadata: Dict = None) -> bool:
        """Save returns data to storage."""
        pass
    
    @abstractmethod
    def load_returns_data(self, start_date: str = None, end_date: str = None,
                         assets: List[str] = None) -> Optional[pd.DataFrame]:
        """Load returns data from storage."""
        pass
    
    @abstractmethod
    def get_available_assets(self) -> List[str]:
        """Get list of available assets in storage."""
        pass
    
    @abstractmethod
    def get_data_info(self) -> Dict:
        """Get information about stored data."""
        pass


class LocalFileStorage(DataStorageBase):
    """Local file-based storage using CSV, Pickle, or Parquet formats."""
    
    def __init__(self, data_dir: str = "data", file_format: str = "pickle"):
        """
        Initialize local file storage.
        
        Args:
            data_dir: Directory to store data files
            file_format: Storage format ('csv', 'pickle', 'parquet')
        """
        self.data_dir = data_dir
        self.file_format = file_format.lower()
        
        if self.file_format == 'parquet' and not HAS_PARQUET:
            logger.warning("Parquet not available, falling back to pickle")
            self.file_format = 'pickle'
        
        self._ensure_directory()
        logger.info(f"Initialized LocalFileStorage with format: {self.file_format}")
    
    def _ensure_directory(self):
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'metadata'), exist_ok=True)
    
    def _get_file_path(self, data_type: str, timestamp: bool = False) -> str:
        """Get file path for data type."""
        if timestamp:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{data_type}_{ts}"
        else:
            filename = f"{data_type}_latest"
        
        extension = {'csv': '.csv', 'pickle': '.pkl', 'parquet': '.parquet'}[self.file_format]
        return os.path.join(self.data_dir, filename + extension)
    
    def save_price_data(self, prices: pd.DataFrame, metadata: Dict = None) -> bool:
        """Save price data to local files."""
        try:
            # Save timestamped version
            timestamped_path = self._get_file_path('price_matrix', timestamp=True)
            latest_path = self._get_file_path('price_matrix', timestamp=False)
            
            # Save in specified format
            if self.file_format == 'csv':
                prices.to_csv(timestamped_path)
                prices.to_csv(latest_path)
            elif self.file_format == 'pickle':
                prices.to_pickle(timestamped_path)
                prices.to_pickle(latest_path)
            elif self.file_format == 'parquet':
                prices.to_parquet(timestamped_path)
                prices.to_parquet(latest_path)
            
            # Save metadata
            if metadata:
                self._save_metadata('price_matrix', metadata)
            
            logger.info(f"Saved price data: {prices.shape} to {latest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save price data: {e}")
            return False
    
    def load_price_data(self, start_date: str = None, end_date: str = None, 
                       assets: List[str] = None) -> Optional[pd.DataFrame]:
        """Load price data from local files."""
        try:
            file_path = self._get_file_path('price_matrix', timestamp=False)
            
            if not os.path.exists(file_path):
                logger.warning(f"Price data file not found: {file_path}")
                return None
            
            # Load data based on format
            if self.file_format == 'csv':
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif self.file_format == 'pickle':
                data = pd.read_pickle(file_path)
            elif self.file_format == 'parquet':
                data = pd.read_parquet(file_path)
            
            # Apply filters
            data = self._filter_data(data, start_date, end_date, assets)
            
            logger.info(f"Loaded price data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return None
    
    def save_returns_data(self, returns: pd.DataFrame, metadata: Dict = None) -> bool:
        """Save returns data to local files."""
        try:
            timestamped_path = self._get_file_path('returns_matrix', timestamp=True)
            latest_path = self._get_file_path('returns_matrix', timestamp=False)
            
            if self.file_format == 'csv':
                returns.to_csv(timestamped_path)
                returns.to_csv(latest_path)
            elif self.file_format == 'pickle':
                returns.to_pickle(timestamped_path)
                returns.to_pickle(latest_path)
            elif self.file_format == 'parquet':
                returns.to_parquet(timestamped_path)
                returns.to_parquet(latest_path)
            
            if metadata:
                self._save_metadata('returns_matrix', metadata)
            
            logger.info(f"Saved returns data: {returns.shape} to {latest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save returns data: {e}")
            return False
    
    def load_returns_data(self, start_date: str = None, end_date: str = None,
                         assets: List[str] = None) -> Optional[pd.DataFrame]:
        """Load returns data from local files."""
        try:
            file_path = self._get_file_path('returns_matrix', timestamp=False)
            
            if not os.path.exists(file_path):
                logger.warning(f"Returns data file not found: {file_path}")
                return None
            
            if self.file_format == 'csv':
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif self.file_format == 'pickle':
                data = pd.read_pickle(file_path)
            elif self.file_format == 'parquet':
                data = pd.read_parquet(file_path)
            
            data = self._filter_data(data, start_date, end_date, assets)
            
            logger.info(f"Loaded returns data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load returns data: {e}")
            return None
    
    def get_available_assets(self) -> List[str]:
        """Get list of available assets."""
        try:
            price_data = self.load_price_data()
            if price_data is not None:
                return list(price_data.columns)
            return []
        except Exception:
            return []
    
    def get_data_info(self) -> Dict:
        """Get information about stored data."""
        info = {
            'storage_type': 'local_files',
            'file_format': self.file_format,
            'data_directory': self.data_dir,
            'price_data_available': False,
            'returns_data_available': False,
            'assets_count': 0,
            'date_range': None
        }
        
        try:
            # Check price data
            price_file = self._get_file_path('price_matrix', timestamp=False)
            if os.path.exists(price_file):
                info['price_data_available'] = True
                prices = self.load_price_data()
                if prices is not None:
                    info['assets_count'] = len(prices.columns)
                    info['date_range'] = (prices.index[0].strftime('%Y-%m-%d'), 
                                        prices.index[-1].strftime('%Y-%m-%d'))
            
            # Check returns data
            returns_file = self._get_file_path('returns_matrix', timestamp=False)
            info['returns_data_available'] = os.path.exists(returns_file)
            
        except Exception as e:
            logger.error(f"Error getting data info: {e}")
        
        return info
    
    def _filter_data(self, data: pd.DataFrame, start_date: str = None, 
                    end_date: str = None, assets: List[str] = None) -> pd.DataFrame:
        """Apply filters to data."""
        if start_date:
            data = data[data.index >= start_date]
        
        if end_date:
            data = data[data.index <= end_date]
        
        if assets:
            available_assets = [asset for asset in assets if asset in data.columns]
            if available_assets:
                data = data[available_assets]
        
        return data
    
    def _save_metadata(self, data_type: str, metadata: Dict):
        """Save metadata to JSON file."""
        metadata_path = os.path.join(self.data_dir, 'metadata', f'{data_type}_metadata.json')
        metadata['last_updated'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


class SQLiteStorage(DataStorageBase):
    """SQLite database storage for local development."""
    
    def __init__(self, db_path: str = "data/portfolio.db"):
        """Initialize SQLite storage."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_database()
        logger.info(f"Initialized SQLiteStorage: {db_path}")
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Price data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    date TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    price REAL NOT NULL,
                    PRIMARY KEY (date, asset)
                )
            """)
            
            # Returns data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS returns_data (
                    date TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    return_value REAL NOT NULL,
                    PRIMARY KEY (date, asset)
                )
            """)
            
            # Metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_asset ON price_data(asset)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_returns_date ON returns_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_returns_asset ON returns_data(asset)")
            
            conn.commit()
    
    def save_price_data(self, prices: pd.DataFrame, metadata: Dict = None) -> bool:
        """Save price data to SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute("DELETE FROM price_data")
                
                # Convert to long format and save
                price_long = prices.stack().reset_index()
                price_long.columns = ['date', 'asset', 'price']
                price_long['date'] = price_long['date'].dt.strftime('%Y-%m-%d')
                
                price_long.to_sql('price_data', conn, if_exists='append', index=False)
                
                # Save metadata
                if metadata:
                    self._save_metadata_sqlite(conn, 'price_data', metadata)
                
                conn.commit()
            
            logger.info(f"Saved price data to SQLite: {prices.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save price data to SQLite: {e}")
            return False
    
    def load_price_data(self, start_date: str = None, end_date: str = None, 
                       assets: List[str] = None) -> Optional[pd.DataFrame]:
        """Load price data from SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT date, asset, price FROM price_data"
                conditions = []
                
                if start_date:
                    conditions.append(f"date >= '{start_date}'")
                if end_date:
                    conditions.append(f"date <= '{end_date}'")
                if assets:
                    asset_list = "', '".join(assets)
                    conditions.append(f"asset IN ('{asset_list}')")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY date, asset"
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    return None
                
                # Convert to wide format
                data = df.pivot(index='date', columns='asset', values='price')
                data.index = pd.to_datetime(data.index)
                
                logger.info(f"Loaded price data from SQLite: {data.shape}")
                return data
                
        except Exception as e:
            logger.error(f"Failed to load price data from SQLite: {e}")
            return None
    
    def save_returns_data(self, returns: pd.DataFrame, metadata: Dict = None) -> bool:
        """Save returns data to SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM returns_data")
                
                returns_long = returns.stack().reset_index()
                returns_long.columns = ['date', 'asset', 'return_value']
                returns_long['date'] = returns_long['date'].dt.strftime('%Y-%m-%d')
                
                returns_long.to_sql('returns_data', conn, if_exists='append', index=False)
                
                if metadata:
                    self._save_metadata_sqlite(conn, 'returns_data', metadata)
                
                conn.commit()
            
            logger.info(f"Saved returns data to SQLite: {returns.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save returns data to SQLite: {e}")
            return False
    
    def load_returns_data(self, start_date: str = None, end_date: str = None,
                         assets: List[str] = None) -> Optional[pd.DataFrame]:
        """Load returns data from SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT date, asset, return_value FROM returns_data"
                conditions = []
                
                if start_date:
                    conditions.append(f"date >= '{start_date}'")
                if end_date:
                    conditions.append(f"date <= '{end_date}'")
                if assets:
                    asset_list = "', '".join(assets)
                    conditions.append(f"asset IN ('{asset_list}')")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY date, asset"
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    return None
                
                data = df.pivot(index='date', columns='asset', values='return_value')
                data.index = pd.to_datetime(data.index)
                
                logger.info(f"Loaded returns data from SQLite: {data.shape}")
                return data
                
        except Exception as e:
            logger.error(f"Failed to load returns data from SQLite: {e}")
            return None
    
    def get_available_assets(self) -> List[str]:
        """Get list of available assets."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT asset FROM price_data ORDER BY asset")
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []
    
    def get_data_info(self) -> Dict:
        """Get information about stored data."""
        info = {
            'storage_type': 'sqlite',
            'database_path': self.db_path,
            'price_data_available': False,
            'returns_data_available': False,
            'assets_count': 0,
            'date_range': None
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check price data
                cursor = conn.execute("SELECT COUNT(*) FROM price_data")
                price_count = cursor.fetchone()[0]
                info['price_data_available'] = price_count > 0
                
                if price_count > 0:
                    cursor = conn.execute("SELECT COUNT(DISTINCT asset) FROM price_data")
                    info['assets_count'] = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT MIN(date), MAX(date) FROM price_data")
                    date_range = cursor.fetchone()
                    info['date_range'] = date_range
                
                # Check returns data
                cursor = conn.execute("SELECT COUNT(*) FROM returns_data")
                returns_count = cursor.fetchone()[0]
                info['returns_data_available'] = returns_count > 0
                
        except Exception as e:
            logger.error(f"Error getting data info from SQLite: {e}")
        
        return info
    
    def _save_metadata_sqlite(self, conn, data_type: str, metadata: Dict):
        """Save metadata to SQLite."""
        metadata_str = json.dumps(metadata)
        timestamp = datetime.now().isoformat()
        
        conn.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (data_type, metadata_str, timestamp))


class PostgreSQLStorage(DataStorageBase):
    """PostgreSQL storage for production use."""
    
    def __init__(self, connection_string: str):
        """Initialize PostgreSQL storage."""
        if not HAS_POSTGRES:
            raise ImportError("PostgreSQL support requires psycopg2-binary")
        
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        self._initialize_database()
        logger.info("Initialized PostgreSQL storage")
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self.engine.connect() as conn:
            # Price data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS price_data (
                    date DATE NOT NULL,
                    asset VARCHAR(20) NOT NULL,
                    price DECIMAL(12,4) NOT NULL,
                    PRIMARY KEY (date, asset)
                )
            """))
            
            # Returns data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS returns_data (
                    date DATE NOT NULL,
                    asset VARCHAR(20) NOT NULL,
                    return_value DECIMAL(8,6) NOT NULL,
                    PRIMARY KEY (date, asset)
                )
            """))
            
            # Metadata table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key VARCHAR(100) PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            
            # Create indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_price_asset ON price_data(asset)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_returns_date ON returns_data(date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_returns_asset ON returns_data(asset)"))
            
            conn.commit()
    
    def save_price_data(self, prices: pd.DataFrame, metadata: Dict = None) -> bool:
        """Save price data to PostgreSQL."""
        try:
            with self.engine.connect() as conn:
                # Clear existing data
                conn.execute(text("DELETE FROM price_data"))
                
                # Convert to long format
                price_long = prices.stack().reset_index()
                price_long.columns = ['date', 'asset', 'price']
                
                # Save to PostgreSQL
                price_long.to_sql('price_data', conn, if_exists='append', index=False)
                
                # Save metadata
                if metadata:
                    self._save_metadata_postgres(conn, 'price_data', metadata)
                
                conn.commit()
            
            logger.info(f"Saved price data to PostgreSQL: {prices.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save price data to PostgreSQL: {e}")
            return False
    
    def load_price_data(self, start_date: str = None, end_date: str = None, 
                       assets: List[str] = None) -> Optional[pd.DataFrame]:
        """Load price data from PostgreSQL."""
        try:
            query = "SELECT date, asset, price FROM price_data"
            conditions = []
            
            if start_date:
                conditions.append(f"date >= '{start_date}'")
            if end_date:
                conditions.append(f"date <= '{end_date}'")
            if assets:
                asset_list = "', '".join(assets)
                conditions.append(f"asset IN ('{asset_list}')")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY date, asset"
            
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn)
            
            if df.empty:
                return None
            
            data = df.pivot(index='date', columns='asset', values='price')
            data.index = pd.to_datetime(data.index)
            
            logger.info(f"Loaded price data from PostgreSQL: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load price data from PostgreSQL: {e}")
            return None
    
    # Similar implementations for returns_data and other methods...
    # (Following the same pattern as SQLite but with PostgreSQL-specific optimizations)
    
    def save_returns_data(self, returns: pd.DataFrame, metadata: Dict = None) -> bool:
        """Save returns data to PostgreSQL."""
        # Implementation similar to save_price_data
        pass
    
    def load_returns_data(self, start_date: str = None, end_date: str = None,
                         assets: List[str] = None) -> Optional[pd.DataFrame]:
        """Load returns data from PostgreSQL."""
        # Implementation similar to load_price_data
        pass
    
    def get_available_assets(self) -> List[str]:
        """Get list of available assets."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT DISTINCT asset FROM price_data ORDER BY asset"))
                return [row[0] for row in result.fetchall()]
        except Exception:
            return []
    
    def get_data_info(self) -> Dict:
        """Get information about stored data."""
        # Implementation similar to SQLite version
        pass
    
    def _save_metadata_postgres(self, conn, data_type: str, metadata: Dict):
        """Save metadata to PostgreSQL."""
        metadata_str = json.dumps(metadata)
        timestamp = datetime.now()
        
        conn.execute(text("""
            INSERT INTO metadata (key, value, updated_at)
            VALUES (:key, :value, :timestamp)
            ON CONFLICT (key) DO UPDATE SET
            value = EXCLUDED.value,
            updated_at = EXCLUDED.updated_at
        """), {"key": data_type, "value": metadata_str, "timestamp": timestamp})


class DataStorageManager:
    """Main data storage manager that handles multiple storage backends."""
    
    def __init__(self, storage_type: str = "local_files", **kwargs):
        """
        Initialize storage manager.
        
        Args:
            storage_type: Type of storage ('local_files', 'sqlite', 'postgresql')
            **kwargs: Storage-specific configuration
        """
        self.storage_type = storage_type
        self.storage = self._create_storage(storage_type, **kwargs)
        logger.info(f"Initialized DataStorageManager with {storage_type}")
    
    def _create_storage(self, storage_type: str, **kwargs) -> DataStorageBase:
        """Create appropriate storage backend."""
        if storage_type == "local_files":
            return LocalFileStorage(**kwargs)
        elif storage_type == "sqlite":
            return SQLiteStorage(**kwargs)
        elif storage_type == "postgresql":
            return PostgreSQLStorage(**kwargs)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def save_portfolio_data(self, prices: pd.DataFrame, returns: pd.DataFrame, 
                          metadata: Dict = None) -> bool:
        """Save both price and returns data."""
        try:
            # Add metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'data_shape': f"{prices.shape[0]} days, {prices.shape[1]} assets",
                'date_range': f"{prices.index[0]} to {prices.index[-1]}",
                'assets': list(prices.columns),
                'storage_type': self.storage_type,
                'created_at': datetime.now().isoformat()
            })
            
            # Save both datasets
            price_success = self.storage.save_price_data(prices, metadata)
            returns_success = self.storage.save_returns_data(returns, metadata)
            
            return price_success and returns_success
            
        except Exception as e:
            logger.error(f"Failed to save portfolio data: {e}")
            return False
    
    def load_portfolio_data(self, start_date: str = None, end_date: str = None,
                          assets: List[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load both price and returns data."""
        try:
            prices = self.storage.load_price_data(start_date, end_date, assets)
            returns = self.storage.load_returns_data(start_date, end_date, assets)
            return prices, returns
        except Exception as e:
            logger.error(f"Failed to load portfolio data: {e}")
            return None, None
    
    def get_storage_info(self) -> Dict:
        """Get comprehensive storage information."""
        return self.storage.get_data_info()
    
    def migrate_data(self, target_storage_type: str, **kwargs) -> bool:
        """Migrate data to a different storage backend."""
        try:
            # Load all data from current storage
            prices, returns = self.load_portfolio_data()
            
            if prices is None or returns is None:
                logger.error("No data available for migration")
                return False
            
            # Create new storage backend
            new_storage = self._create_storage(target_storage_type, **kwargs)
            
            # Save data to new storage
            metadata = {'migrated_from': self.storage_type, 'migration_date': datetime.now().isoformat()}
            
            price_success = new_storage.save_price_data(prices, metadata)
            returns_success = new_storage.save_returns_data(returns, metadata)
            
            if price_success and returns_success:
                logger.info(f"Successfully migrated data from {self.storage_type} to {target_storage_type}")
                return True
            else:
                logger.error("Failed to save data to new storage")
                return False
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False


def benchmark_storage_formats():
    """Benchmark different storage formats for performance comparison."""
    import time
    import tempfile
    import shutil
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    assets = [f'ASSET_{i:02d}' for i in range(50)]
    
    np.random.seed(42)
    price_data = pd.DataFrame(
        np.random.randn(1000, 50).cumsum(axis=0) + 100,
        index=dates,
        columns=assets
    )
    
    results = {}
    
    for file_format in ['csv', 'pickle', 'parquet']:
        if file_format == 'parquet' and not HAS_PARQUET:
            continue
            
        temp_dir = tempfile.mkdtemp()
        try:
            storage = LocalFileStorage(temp_dir, file_format)
            
            # Benchmark save
            start_time = time.time()
            storage.save_price_data(price_data)
            save_time = time.time() - start_time
            
            # Benchmark load
            start_time = time.time()
            loaded_data = storage.load_price_data()
            load_time = time.time() - start_time
            
            # Get file size
            file_path = storage._get_file_path('price_matrix')
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            
            results[file_format] = {
                'save_time': save_time,
                'load_time': load_time,
                'file_size_mb': file_size,
                'data_integrity': loaded_data.equals(price_data)
            }
            
        finally:
            shutil.rmtree(temp_dir)
    
    return results


if __name__ == "__main__":
    # Example usage and benchmarking
    print("Data Storage System - Example Usage")
    print("=" * 50)
    
    # Run benchmarks
    print("Running storage format benchmarks...")
    benchmark_results = benchmark_storage_formats()
    
    print("\nBenchmark Results:")
    print("-" * 30)
    for format_name, metrics in benchmark_results.items():
        print(f"{format_name.upper()}:")
        print(f"  Save time: {metrics['save_time']:.3f}s")
        print(f"  Load time: {metrics['load_time']:.3f}s")
        print(f"  File size: {metrics['file_size_mb']:.2f} MB")
        print(f"  Data integrity: {metrics['data_integrity']}")
        print()
    
    print("Storage system ready for use!") 