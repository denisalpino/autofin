import os, sys
import re
from pathlib import Path
from types import NoneType
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
    """Universal data loader for multiple file formats."""

    def __init__(self, config: DataLoaderConfig | str | None = None):
        """
        Parameters
        ----------
        config : Union[DataLoaderConfig, str, None], default=None
            Configuration object, path to config file, or None for default config

        Raises
        ------
        TypeError
            If config is not a string path, DataLoaderConfig instance, or None
        """
        if isinstance(config, str):
            # Loading configuration
            self.config = load_config(config)
        elif isinstance(config, DataLoaderConfig):
            # Assign configuration
            self.config = config
        elif isinstance(config, NoneType):
            # Setting up the basic configuration
            self.config = DataLoaderConfig()
        else:
            raise TypeError("Config must be path string, DataLoaderConfig instance, or None")

        self.loader_map = {
            ".csv": self._load_csv,
            ".parquet": self._load_parquet,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".json": self._load_json
        }

    def load(self, source: str) -> List[DataFrame]:
        """
        Load data from file or directory.

        Parameters
        ----------
        source : str
            Path to file or directory containing data files

        Returns
        -------
        List[DataFrame]
            List of loaded DataFrames

        Raises
        ------
        ValueError
            If source is not a valid file or directory
        """
        source_path = Path(source)
        project_root = Path(sys.path[0]).parent
        full_path = project_root / source_path

        if full_path.is_file():
            return [self._load_single_file(full_path)]
        elif full_path.is_dir():
            return self._load_from_directory(full_path)
        else:
            raise ValueError(f"Source {full_path} is not a valid file or directory")

    def _load_from_directory(self, directory: Path) -> List[DataFrame]:
        """
        Load all supported files from directory.

        Parameters
        ----------
        directory : Path
            Directory path to load files from

        Returns
        -------
        List[DataFrame]
            List of DataFrames from all supported files in directory
        """
        dfs = []

        for file_path in directory.iterdir():
            if file_path.is_file() and self._is_supported_format(file_path):
                try:
                    df = self._load_single_file(file_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return dfs

    def _load_single_file(self, file_path: Path) -> DataFrame:
        """
        Load and process single file.

        Parameters
        ----------
        file_path : Path
            Path to the file to load

        Returns
        -------
        DataFrame
            Processed DataFrame with standardized columns and ticker

        Raises
        ------
        ValueError
            If required columns are missing or ticker cannot be extracted
        """
        # Determining the file format
        file_format = self._detect_file_format(file_path)

        # Loading data
        loader_func = self.loader_map[file_format]
        df = loader_func(file_path)

        # Checking the required fields
        self._validate_required_columns(df)

        # Rename the columns according to the schema
        df = self._rename_columns(df)

        # Add ticker
        df = self._add_ticker_column(df, file_path)

        return df

    def _rename_columns(self, df: DataFrame) -> DataFrame:
        """Rename columns to standardized names using configuration mapping."""
        # Create a reverse mapping: standard_name -> current_name
        rename_dict = {}
        mapping_dict = self.config.mapping.model_dump()

        for standard_name, current_name in mapping_dict.items():
            if current_name and current_name in df.columns:
                rename_dict[current_name] = standard_name

        return df.rename(columns=rename_dict)

    def _validate_required_columns(self, df: DataFrame):
        """Validate presence of required columns"""
        required_columns = self.config.required_columns

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _add_ticker_column(self, df: DataFrame, file_path: Path) -> DataFrame:
        """Add ticker column extracted from filename"""
        match = re.search(self.config.ticker_pattern, file_path.stem)
        if match:
            ticker = match.group(1)
            df = df.copy()
            df["ticker"] = ticker
            df["ticker"] = df["ticker"].astype("category")
        else:
            raise ValueError(f"Could not extract ticker from {file_path.name}")

        return df

    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported"""
        ext = file_path.suffix.lower()
        return ext in self.loader_map

    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from extension or configuration"""
        if self.config.file_format:
            return f".{self.config.file_format.value}"

        ext = file_path.suffix.lower()
        if ext in self.loader_map:
            return ext

        raise ValueError(f"Unsupported file format: {ext}")

    def _load_csv(self, file_path: Path) -> DataFrame:
        """Load CSV file using configured options"""
        return read_csv(file_path, **self.config.read_options)

    def _load_parquet(self, file_path: Path) -> DataFrame:
        """Load Parquet file using configured options"""
        return read_parquet(file_path, **self.config.read_options)

    def _load_excel(self, file_path: Path) -> DataFrame:
        """Load Excel file using configured options"""
        return read_excel(file_path, **self.config.read_options)

    def _load_json(self, file_path: Path) -> DataFrame:
        """Load JSON file using configured options"""
        return read_json(file_path, **self.config.read_options)
