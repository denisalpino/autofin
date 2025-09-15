from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from pandas import DataFrame
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class FileFormat(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"
    JSON = "json"

class DatasetSchema(BaseModel):
    """Схема для описания и маппинга структуры набора данных"""
    timestamps: str           = Field(default="timestamps")
    open:       str           = Field(default="open")
    high:       str           = Field(default="high")
    low:        str           = Field(default="low")
    close:      str           = Field(default="close")
    adjclose:   Optional[str] = Field(default=None)
    volume:     Optional[str] = Field(default=None)


class DataLoaderConfig(BaseModel):
    mapping:         DatasetSchema        = Field(default_factory=DatasetSchema)
    file_format:     Optional[FileFormat] = Field(None, description="Формат файла (если не указан, будет определен автоматически)")
    read_options:    Dict                 = Field(default_factory=dict, description="Параметры для чтения файлов")
    ticker_pattern:  str                  = Field(r"^([^_]+)", description="regex pattern как извлекать тикер из имени файла")

    # Настройка для использования значений перечислений
    class Config:
        use_enum_values = True

    @property
    def required_columns(self) -> set:
        """Возвращает множество обязательных колонок"""
        return {
            self.mapping.timestamp_col,
            self.mapping.open_col,
            self.mapping.high_col,
            self.mapping.low_col,
            self.mapping.close_col
        }

# Все файлы должны быть одного формата, иметь одинаковую структуру и опции чтения

@dataclass
class Dataset:
    """Класс для хранения обработанных данных"""
    raw_features: DataFrame
    train: Optional[NDArray] = None
    val: Optional[NDArray] = None
    test: Optional[NDArray] = None
    cv_splits: Optional[tuple] = None
