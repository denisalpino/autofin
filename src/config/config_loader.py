import os, sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.main_config import MainConfig


def load_config(config_path: Optional[str] = None) -> MainConfig:
    """
    Загружает конфигурацию из YAML файла, объединяя с настройками по умолчанию

    Parameters:
    -----------
    config_path : str, optional
        Путь к YAML файлу с конфигурацией. Если не указан, используются настройки по умолчанию.

    Returns:
    --------
    MainConfig
        Загруженная конфигурация
    """
    # Создаем конфигурацию по умолчанию
    default_config = MainConfig()

    if config_path is None:
        return default_config

    # Загружаем пользовательский конфиг из YAML
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    with open(config_path, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f)

    if user_config is None:
        return default_config

    # Рекурсивно обновляем конфигурацию по умолчанию пользовательскими настройками
    updated_config = _deep_update(default_config.model_dump(), user_config)

    # Создаем объект конфигурации
    return MainConfig.model_validate(updated_config)

def _deep_update(default_dict: Dict[str, Any], user_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Рекурсивно обновляет словарь по умолчанию пользовательскими значениями

    Parameters:
    -----------
    default_dict : Dict[str, Any]
        Словарь с настройками по умолчанию
    user_dict : Dict[str, Any]
        Словарь с пользовательскими настройками

    Returns:
    --------
    Dict[str, Any]
        Обновленный словарь
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
    Сохраняет конфигурацию в YAML файл

    Parameters:
    -----------
    config : MainConfig
        Конфигурация для сохранения
    config_path : str
        Путь для сохранения файла
    """
    config_path = Path(config_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, allow_unicode=True)

