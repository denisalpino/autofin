import os, sys
import re
from pathlib import Path
from typing import List

from pandas import (
    DataFrame,
    read_parquet,
    read_csv,
    read_json,
    read_excel
)

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.schemas.data import DataLoaderConfig
from src.config.config_loader import load_config


class DataLoader:
    """Универсальный загрузчик данных из файлов различных форматов"""

    def __init__(self, config: DataLoaderConfig | str | None = None):
        if isinstance(config, str):
            # Загружаем конфигурацию
            self.config = load_config(config)
        elif isinstance(config, DataLoaderConfig):
            # Присваиваем конфигурацию
            self.config = config
        elif isinstance(config, None):
            # Устанавливаем базовую конфигурацию
            self.config = DataLoaderConfig()
        else:
            raise TypeError("`config` must be string of path to YAML file or DataLoaderConfig instance or None")

        self.loader_map = {
            ".csv": self._load_csv,
            ".parquet": self._load_parquet,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".json": self._load_json
        }

    def load(self, source: str) -> List[DataFrame]:
        """
        Загружает данные из файла или директории

        Parameters:
        -----------
        source : str
            Путь к файлу или директории с файлами

        Returns:
        --------
        List[DataFrame]
            Список DataFrame с данными
        """
        source_path = Path(source)

        if source_path.is_file():
            return [self._load_single_file(source_path)]
        elif source_path.is_dir():
            return self._load_from_directory(source_path)
        else:
            raise ValueError(f"Источник {source} не является файлом или директорией")

    def _load_from_directory(self, directory: Path) -> List[DataFrame]:
        """Загружает все поддерживаемые файлы из директории"""
        dfs = []

        for file_path in directory.iterdir():
            if file_path.is_file() and self._is_supported_format(file_path):
                try:
                    df = self._load_single_file(file_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"Ошибка при загрузке файла {file_path}: {e}")

        return dfs

    def _load_single_file(self, file_path: Path) -> DataFrame:
        """Загружает данные из одного файла"""
        # Определяем формат файла
        file_format = self._detect_file_format(file_path)

        # Загружаем данные
        loader_func = self.loader_map[file_format]
        df = loader_func(file_path)

        # Переименовываем колонки согласно схеме
        df = self._rename_columns(df)

        # Проверяем обязательные колонки
        self._validate_required_columns(df)

        # Добавляем тикер
        df = self._add_ticker_column(df, file_path)

        return df

    def _rename_columns(self, df: DataFrame) -> DataFrame:
        """Переименовывает колонки согласно схеме"""
        # Создаем обратный маппинг: стандартное_имя -> текущее_имя
        rename_dict = {}
        mapping_dict = self.config.mapping.model_dump()

        for standard_name, current_name in mapping_dict.items():
            if current_name and current_name in df.columns:
                rename_dict[current_name] = standard_name

        return df.rename(columns=rename_dict)

    def _validate_required_columns(self, df: DataFrame):
        """Проверяет наличие обязательных колонок"""
        required_columns = {
            self.config.mapping.timestamps,
            self.config.mapping.open,
            self.config.mapping.high,
            self.config.mapping.low,
            self.config.mapping.close
        }

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

    def _add_ticker_column(self, df: DataFrame, file_path: Path) -> DataFrame:
        """Добавляет колонку с тикером на основе имени файла"""
        match = re.search(self.config.ticker_pattern, file_path.stem)
        if match:
            ticker = match.group(1)
            df = df.copy()
            df["ticker"] = ticker
            df["ticker"] = df["ticker"].astype("category")
        else:
            raise ValueError(f"Не удалось извлечь тикер из имени файла: {file_path.name}")

        return df

    def _is_supported_format(self, file_path: Path) -> bool:
        """Проверяет, поддерживается ли формат файла"""
        ext = file_path.suffix.lower()
        return ext in self.loader_map

    def _detect_file_format(self, file_path: Path) -> str:
        """Определяет формат файла по расширению"""
        if self.config.file_format:
            return f".{self.config.file_format.value}"

        ext = file_path.suffix.lower()
        if ext in self.loader_map:
            return ext

        raise ValueError(f"Неизвестный формат файла: {ext}")

    def _load_csv(self, file_path: Path) -> DataFrame:
        """Загружает CSV файл"""
        return read_csv(file_path, **self.config.read_options)

    def _load_parquet(self, file_path: Path) -> DataFrame:
        """Загружает Parquet файл"""
        return read_parquet(file_path, **self.config.read_options)

    def _load_excel(self, file_path: Path) -> DataFrame:
        """Загружает Excel файл"""
        return read_excel(file_path, **self.config.read_options)

    def _load_json(self, file_path: Path) -> DataFrame:
        """Загружает JSON файл"""
        return read_json(file_path, **self.config.read_options)
