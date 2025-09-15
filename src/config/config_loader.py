import os
from pathlib import Path
import sys
from typing import Dict, Any, Optional

import yaml

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.main_config import MainConfig


def load_config(config_path: Optional[str] = None) -> MainConfig:
    """
    Load configuration from YAML file with default fallback.

    Parameters
    ----------
    config_path : Optional[str], default=None
        Path to YAML configuration file

    Returns
    -------
    MainConfig
        Loaded configuration object

    Raises
    ------
    FileNotFoundError
        If specified config file does not exist
    """
    # Creating a default configuration
    default_config = MainConfig()

    if config_path is None:
        return default_config

    # Loading user config from YAML
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    with open(config_path, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f)

    if user_config is None:
        return default_config

    # Recursively update the default configuration with user settings
    updated_config = _deep_update(default_config.model_dump(), user_config)

    # Create a configuration object
    return MainConfig.model_validate(updated_config)

def _deep_update(default_dict: Dict[str, Any], user_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dictionary with user values.

    Parameters
    ----------
    default_dict : Dict[str, Any]
        Dictionary with default values
    user_dict : Dict[str, Any]
        Dictionary with user values

    Returns
    -------
    Dict[str, Any]
        Updated dictionary
    """
    for key, value in user_dict.items():
        if (key in default_dict and
            isinstance(default_dict[key], dict) and
            isinstance(value, dict)):
            _deep_update(default_dict[key], value)
        else:
            default_dict[key] = value
    return default_dict

def save_config(config: MainConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : MainConfig
        Configuration object to save
    config_path : str
        Path for saving configuration file
    """
    config_path = Path(config_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, allow_unicode=True)

