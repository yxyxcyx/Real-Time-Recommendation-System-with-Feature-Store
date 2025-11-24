"""Configuration loader with environment variable support."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf
from loguru import logger


class ConfigLoader:
    """Handles configuration loading with environment overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the config loader.
        
        Args:
            config_path: Path to the configuration file
        """
        if config_path is None:
            config_path = os.getenv(
                "RECSYS_CONFIG", 
                "configs/config.yaml"
            )
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._apply_env_overrides()
        
    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = OmegaConf.create(config_dict)
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_mapping = {
            "RECSYS_ENV": "system.environment",
            "RECSYS_DEBUG": "system.debug",
            "REDIS_URL": "feature_store.online_store.connection_string",
            "KAFKA_BOOTSTRAP_SERVERS": "streaming.kafka.bootstrap_servers",
            "MLFLOW_TRACKING_URI": "monitoring.mlflow.tracking_uri",
            "API_HOST": "serving.api.host",
            "API_PORT": "serving.api.port",
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                OmegaConf.update(self.config, config_path, value)
                logger.info(f"Applied environment override: {env_var} -> {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key)
        except Exception:
            return default
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self.config, resolve=True)
    
    def save(self, path: str):
        """Save current configuration to file.
        
        Args:
            path: Path to save configuration
        """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"Saved configuration to {path}")


# Global configuration instance
_config_instance: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get global configuration instance.
    
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance


def reset_config():
    """Reset global configuration instance."""
    global _config_instance
    _config_instance = None
