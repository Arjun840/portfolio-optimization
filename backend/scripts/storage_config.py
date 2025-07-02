#!/usr/bin/env python3
"""
Storage Configuration for Portfolio Optimization

This module provides easy configuration for different storage backends
and environments (development, testing, production).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

class StorageType(Enum):
    """Supported storage types."""
    LOCAL_FILES = "local_files"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"

class FileFormat(Enum):
    """Supported file formats for local storage."""
    CSV = "csv"
    PICKLE = "pickle"
    PARQUET = "parquet"

@dataclass
class StorageConfig:
    """Configuration for data storage."""
    
    # Storage type
    storage_type: StorageType = StorageType.LOCAL_FILES
    
    # Local file storage settings
    data_dir: str = "data"
    file_format: FileFormat = FileFormat.PICKLE
    
    # Database settings
    db_path: str = "data/portfolio.db"  # SQLite
    connection_string: Optional[str] = None  # PostgreSQL
    
    # Performance settings
    enable_compression: bool = True
    cache_size: int = 100  # Number of query results to cache
    
    # Backup settings
    auto_backup: bool = True
    backup_dir: str = "data/backups"
    max_backups: int = 10
    
    # Metadata
    description: str = "Portfolio data storage"
    tags: Dict[str, Any] = field(default_factory=dict)

# Predefined configurations for different environments

# Development configuration - fast local storage
DEVELOPMENT_CONFIG = StorageConfig(
    storage_type=StorageType.LOCAL_FILES,
    file_format=FileFormat.PICKLE,
    data_dir="data/dev",
    description="Development environment - local pickle files",
    tags={"environment": "development", "optimized_for": "speed"}
)

# Testing configuration - separate directory, minimal data
TESTING_CONFIG = StorageConfig(
    storage_type=StorageType.LOCAL_FILES,
    file_format=FileFormat.PICKLE,
    data_dir="data/test",
    auto_backup=False,
    description="Testing environment - isolated data",
    tags={"environment": "testing", "isolated": True}
)

# Production configuration - SQLite for reliability
PRODUCTION_CONFIG = StorageConfig(
    storage_type=StorageType.SQLITE,
    db_path="data/production/portfolio.db",
    auto_backup=True,
    backup_dir="data/production/backups",
    description="Production environment - SQLite database",
    tags={"environment": "production", "reliability": "high"}
)

# Analytics configuration - optimized for large datasets
ANALYTICS_CONFIG = StorageConfig(
    storage_type=StorageType.LOCAL_FILES,
    file_format=FileFormat.PARQUET,
    data_dir="data/analytics",
    enable_compression=True,
    description="Analytics environment - Parquet format for large datasets",
    tags={"environment": "analytics", "optimized_for": "large_datasets"}
)

# Sharing configuration - CSV for interoperability
SHARING_CONFIG = StorageConfig(
    storage_type=StorageType.LOCAL_FILES,
    file_format=FileFormat.CSV,
    data_dir="data/export",
    auto_backup=False,
    description="Data sharing - CSV format for compatibility",
    tags={"environment": "sharing", "format": "human_readable"}
)

# Configuration registry
CONFIG_REGISTRY = {
    "development": DEVELOPMENT_CONFIG,
    "dev": DEVELOPMENT_CONFIG,
    "testing": TESTING_CONFIG,
    "test": TESTING_CONFIG,
    "production": PRODUCTION_CONFIG,
    "prod": PRODUCTION_CONFIG,
    "analytics": ANALYTICS_CONFIG,
    "sharing": SHARING_CONFIG,
    "export": SHARING_CONFIG,
}

def get_config(environment: str = "development") -> StorageConfig:
    """
    Get storage configuration for specified environment.
    
    Args:
        environment: Environment name ('development', 'testing', 'production', etc.)
        
    Returns:
        StorageConfig object
        
    Raises:
        ValueError: If environment is not found
    """
    env_key = environment.lower()
    if env_key not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown environment '{environment}'. Available: {available}")
    
    return CONFIG_REGISTRY[env_key]

def create_storage_manager(environment: str = "development", **override_kwargs):
    """
    Create a storage manager with the specified environment configuration.
    
    Args:
        environment: Environment name
        **override_kwargs: Override any configuration parameters
        
    Returns:
        DataStorageManager instance
    """
    from data_storage import DataStorageManager
    
    config = get_config(environment)
    
    # Convert config to kwargs
    kwargs = {
        "storage_type": config.storage_type.value,
    }
    
    # Add storage-specific parameters
    if config.storage_type == StorageType.LOCAL_FILES:
        kwargs.update({
            "data_dir": config.data_dir,
            "file_format": config.file_format.value
        })
    elif config.storage_type == StorageType.SQLITE:
        kwargs["db_path"] = config.db_path
    elif config.storage_type == StorageType.POSTGRESQL:
        if config.connection_string:
            kwargs["connection_string"] = config.connection_string
        else:
            raise ValueError("PostgreSQL connection string not configured")
    
    # Apply overrides
    kwargs.update(override_kwargs)
    
    # Ensure directories exist
    if config.storage_type == StorageType.LOCAL_FILES:
        os.makedirs(config.data_dir, exist_ok=True)
    elif config.storage_type == StorageType.SQLITE:
        os.makedirs(os.path.dirname(config.db_path), exist_ok=True)
    
    if config.auto_backup:
        os.makedirs(config.backup_dir, exist_ok=True)
    
    return DataStorageManager(**kwargs)

def get_recommended_config_for_use_case(use_case: str) -> StorageConfig:
    """
    Get recommended configuration for specific use cases.
    
    Args:
        use_case: Use case description
        
    Returns:
        Recommended StorageConfig
    """
    use_case_lower = use_case.lower()
    
    recommendations = {
        "development": DEVELOPMENT_CONFIG,
        "prototyping": DEVELOPMENT_CONFIG,
        "testing": TESTING_CONFIG,
        "unit_tests": TESTING_CONFIG,
        "production": PRODUCTION_CONFIG,
        "deployment": PRODUCTION_CONFIG,
        "analytics": ANALYTICS_CONFIG,
        "research": ANALYTICS_CONFIG,
        "big_data": ANALYTICS_CONFIG,
        "sharing": SHARING_CONFIG,
        "export": SHARING_CONFIG,
        "interoperability": SHARING_CONFIG,
    }
    
    for keyword, config in recommendations.items():
        if keyword in use_case_lower:
            return config
    
    # Default to development
    return DEVELOPMENT_CONFIG

def create_custom_config(
    storage_type: str = "local_files",
    **kwargs
) -> StorageConfig:
    """
    Create a custom storage configuration.
    
    Args:
        storage_type: Type of storage ('local_files', 'sqlite', 'postgresql')
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom StorageConfig
    """
    return StorageConfig(
        storage_type=StorageType(storage_type),
        **kwargs
    )

# Environment detection
def detect_environment() -> str:
    """
    Automatically detect the current environment based on context.
    
    Returns:
        Detected environment name
    """
    # Check environment variables
    env_var = os.getenv('PORTFOLIO_ENV', '').lower()
    if env_var in CONFIG_REGISTRY:
        return env_var
    
    # Check if we're in a test context
    if 'pytest' in os.getenv('_', '') or 'test' in os.getcwd().lower():
        return 'testing'
    
    # Check for production indicators
    if os.getenv('ENVIRONMENT', '').lower() in ['prod', 'production']:
        return 'production'
    
    # Default to development
    return 'development'

def print_config_summary(config: StorageConfig):
    """Print a summary of the storage configuration."""
    print(f"Storage Configuration Summary")
    print(f"=" * 40)
    print(f"Type: {config.storage_type.value}")
    print(f"Description: {config.description}")
    
    if config.storage_type == StorageType.LOCAL_FILES:
        print(f"Format: {config.file_format.value}")
        print(f"Directory: {config.data_dir}")
    elif config.storage_type == StorageType.SQLITE:
        print(f"Database: {config.db_path}")
    elif config.storage_type == StorageType.POSTGRESQL:
        print(f"Connection: {config.connection_string or 'Not configured'}")
    
    print(f"Auto backup: {config.auto_backup}")
    if config.auto_backup:
        print(f"Backup dir: {config.backup_dir}")
    
    if config.tags:
        print(f"Tags: {config.tags}")

if __name__ == "__main__":
    # Demo different configurations
    print("Portfolio Data Storage Configurations")
    print("=" * 50)
    
    environments = ["development", "testing", "production", "analytics", "sharing"]
    
    for env in environments:
        print(f"\n{env.upper()} CONFIGURATION:")
        print("-" * 30)
        config = get_config(env)
        print_config_summary(config) 