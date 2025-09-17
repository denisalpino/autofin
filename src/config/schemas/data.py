# TODO: Add note to the docs
# All files must be in the same format, have the same structure, and read options

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict

from numpy.typing import NDArray
from pandas import DataFrame
from pydantic import BaseModel, Field


class FileFormat(str, Enum):
    """Supported file formats for data loading."""
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"
    JSON = "json"

class ColumnSource(str, Enum):
    TIMESTAMPS = "timestamps"
    OPEN       = "open"
    HIGH       = "high"
    LOW        = "low"
    CLOSE      = "close"
    ADJCLOSE   = "adjclose"
    VOLUME     = "volume"

class DatasetSchema(BaseModel):
    """
    Schema for dataset column mapping.

    Attributes
    ----------
    timestamps : str, default="timestamps"
        Name of the timestamp column
    open : str, default="open"
        Name of the open price column
    high : str, default="high"
        Name of the high price column
    low : str, default="low"
        Name of the low price column
    close : str, default="close"
        Name of the close price column
    adjclose : Optional[str], default=None
        Name of the adjusted close price column
    volume : Optional[str], default=None
        Name of the volume column
    """
    timestamps: str           = Field(default="timestamps")
    open:       str           = Field(default="open")
    high:       str           = Field(default="high")
    low:        str           = Field(default="low")
    close:      str           = Field(default="close")
    adjclose:   Optional[str] = Field(default=None)
    volume:     Optional[str] = Field(default=None)


class DataLoaderConfig(BaseModel):
    """
    Configuration for data loader.

    Attributes
    ----------
    mapping : DatasetSchema, default=DatasetSchema()
        Column mapping schema
    file_format : Optional[FileFormat], default=None
        Force specific file format (autodetected if None)
    read_options : Dict, default={}
        Additional options for pandas read functions
    ticker_pattern : str, default=r"^([^_]+)"
        Regex pattern for extracting ticker from filename
    """
    mapping:         DatasetSchema        = Field(default_factory=DatasetSchema)
    file_format:     Optional[FileFormat] = Field(None, description="File format (autodetected if None)")
    read_options:    Dict                 = Field(default_factory=dict, description="Options for pandas read functions")
    ticker_pattern:  str                  = Field(r"^([^_]+)", description="Regex pattern to extract ticker from filename")

    # Настройка для использования значений перечислений
    class Config:
        use_enum_values = True

    @property
    def required_columns(self) -> set:
        """Set of required column names."""
        return {
            self.mapping.timestamps,
            self.mapping.open,
            self.mapping.high,
            self.mapping.low,
            self.mapping.close
        }


@dataclass
class Dataset:
    """
    Container for processed dataset.

    Attributes
    ----------
    raw_features : DataFrame
        Raw loaded features
    train : Optional[NDArray], default=None
        Training data
    val : Optional[NDArray], default=None
        Validation data
    test : Optional[NDArray], default=None
        Test data
    cv_splits : Optional[tuple], default=None
        Cross-validation splits
    """
    """Класс для хранения обработанных данных"""
    raw_features: DataFrame
    train: Optional[NDArray] = None
    val: Optional[NDArray] = None
    test: Optional[NDArray] = None
    cv_splits: Optional[tuple] = None
