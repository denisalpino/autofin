import os, sys
from typing import Optional

from pydantic import BaseModel, Field

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.schemas.data import DataLoaderConfig
#from src.config.schemas.features import FeatureConfig
#from src.config.schemas.model import ModelConfig
#from src.config.schemas.training import TrainingConfig
#from src.config.schemas.scaling import ScalingConfig


class MainConfig(BaseModel):
    """Главный конфигурационный класс для проекта"""
    data_loader: Optional[DataLoaderConfig] = Field(default_factory=DataLoaderConfig)
    #features:    Optional[FeatureConfig]    = Field(default_factory=FeatureConfig)
    #model:       Optional[ModelConfig]      = Field(default_factory=ModelConfig)
    #training:    Optional[TrainingConfig]   = Field(default_factory=TrainingConfig)
    #scaling:     Optional[ScalingConfig]    = Field(default_factory=ScalingConfig)

    class Config:
        use_enum_values = True
        extra = "ignore"  # Игнорировать лишние поля в YAML
