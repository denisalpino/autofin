from pydantic import BaseModel
from typing import Optional, Set, Literal, Any, Type
from sklearn.base import TransformerMixin

from src.config.constants import NON_SCALABLE_COLS


class ScalingConfig(BaseModel):
    method: Literal["minmax", "standard", "robust", "custom"] = "minmax"
    custom_scaler: Optional[TransformerMixin] = None  # Для пользовательских скейлеров
    non_scalable_cols: Set[str] = NON_SCALABLE_COLS
    include_target: bool = True

    class Config:
        arbitrary_types_allowed = True

scaling = {
    "method": "minmax",
    "non_scalable_cols": {"ticker"},
    "include_target": True
}