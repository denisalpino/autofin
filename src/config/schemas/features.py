import os
import sys
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.schemas.indicators import IndicatorConfig
from src.config.constants import BASE_INDICATORS


class ColumnSource(str, Enum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    ADJCLOSE = "adjclose"
    VOLUME = "volume"

class TimeFeature(str, Enum):
    MINUTE    = "minute"
    HOUR      = "hour"
    DAY       = "day"
    DAYOFWEEK = "day_of_week"
    MONTH     = "month"

class ReturnsMethod(str, Enum):
    RAW     = "raw"
    PERCENT = "percent"
    LOG     = "log"

class DimRedMethod(str, Enum):
    UMAP = "umap"
    PCA  = "pca"

class ReturnsConfig(BaseModel):
    column:  str = "close"
    method:  ReturnsMethod = ReturnsMethod.RAW
    period:  int = 1

class LaggingConfig(BaseModel):
    column:  str = "returns"
    period:  int = 5


class FeatureConfig(BaseModel):
    indicators:    List[IndicatorConfig]  = Field(default_factory=lambda: list(BASE_INDICATORS))
    time_features: List[TimeFeature]      = Field(default_factory=lambda: [TimeFeature.DAYOFWEEK])
    lags:          List[LaggingConfig]    = Field(default_factory=lambda: [LaggingConfig()])
    returns:       ReturnsConfig          = Field(default_factory=ReturnsConfig)
    dimred:        Optional[DimRedMethod] = Field(default=None)
