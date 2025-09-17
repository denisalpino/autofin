from enum import Enum
from typing import Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class TimeUnit(str, Enum):
    """Supported time units for interval configuration."""
    MINUTE = "m"
    HOUR = "h"
    DAY = "d"
    MONTH = "M"
    WEEK = "w"


class CVWindowType(str, Enum):
    """Supported types of cross-validation window."""
    EXPANDING = "expanding"
    ROLLING = "rolling"


class SplitMethod(str, Enum):
    """Supported split methods."""
    CV = "cv"
    SIMPLE = "simple"


class CVConfig(BaseModel):
    """
    Configuration for time-based cross-validation with groups.

    Attributes
    ----------
    val_folds : int
        Number of validation folds in sequence
    val_interval : str
        Size of one validation window, e.g., '1d', '15m', '1M'
    test_interval : Optional[str]
        Size of test period at the end of the series, e.g., '5d', '12h'
    train_interval : Optional[str]
        For window='rolling': size of the training window
    window : CVWindowType
        Type of training window: expanding or rolling
    min_train_samples : int
        Minimum observations in train for a valid split
    """
    val_folds:         int           = Field(default=1, ge=1, description="Number of validation folds in sequence")
    val_interval:      str           = Field(default="7d", description="Size of one validation window")
    test_interval:     Optional[str] = Field(default=None, description="Size of test period at the end")
    train_interval:    Optional[str] = Field(default=None, description="Training window size only for rolling windows")
    window:            CVWindowType  = Field(default=CVWindowType.EXPANDING, description="Type of training window")
    min_train_samples: int           = Field(default=1, ge=1, description="Minimum observations in train for a valid split")

    @field_validator("val_interval", "test_interval", "train_interval", mode="before")
    @classmethod
    def validate_time_interval(cls, value: Optional[str]) -> Optional[str]:
        """Validate time interval string format and values."""
        if value is None:
            return None

        # Parse interval
        try:
            n = int(value[:-1].lstrip("0"))
            unit = value[-1]
        except (ValueError, IndexError):
            raise ValueError(f"Invalid interval format: {value}")

        # Validate based on unit
        try:
            time_unit = TimeUnit(unit)

            if time_unit == TimeUnit.MINUTE and not (1 <= n < 60):
                raise ValueError(f'Minutes must be between 1 and 59, got {n}')
            elif time_unit == TimeUnit.HOUR and not (1 <= n < 24):
                raise ValueError(f'Hours must be between 1 and 23, got {n}')
            elif time_unit == TimeUnit.DAY and not (1 <= n < 31):
                raise ValueError(f'Days must be between 1 and 30, got {n}')
            elif time_unit == TimeUnit.MONTH and not (1 <= n < 12):
                raise ValueError(f'Months must be between 1 and 11, got {n}')
        except ValueError as e:
            raise ValueError(e)

        return str(n) + unit

    @model_validator(mode="after")
    def validate_window_constraints(self) -> "CVConfig":
        """Validate constraints between window type and training interval."""
        if self.window == CVWindowType.ROLLING and self.train_interval is None:
            raise ValueError("Training interval must be provided for rolling windows")
        return self


class SimpleSplitConfig(BaseModel):
    """
    Configuration for simple deterministic split by observation order.

    Attributes
    ----------
    ratios : Tuple[int, int, int]
        Ratios for train, validation, test splits (sum must be 100)
    """
    ratios: Tuple[int, int, int] = Field(
        (80, 10, 10),
        description="Ratios for train, validation, test splits (sum must be 100)"
    )

    @field_validator("ratios", mode="before")
    @classmethod
    def validate_ratios(cls, ratios: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Validate that ratios are valid and sum to 100."""
        if len(ratios) != 3:
            raise ValueError("Ratios must contain exactly three values")

        if any(ratio < 0 for ratio in ratios):
            raise ValueError("All ratios must be non-negative")

        if any(100 <= ratio for ratio in ratios):
            raise ValueError("All ratios must be lower than 100")

        if sum(ratios) != 100:
            raise ValueError(f"Ratios must sum to 100, got {sum(ratios)}")

        return ratios


class SplitConfig(BaseModel):
    """
    Unified configuration for data splitting strategies.

    Attributes
    ----------
    method : SplitMethod
        Splitting method to use (cross-validation or simple split)
    cv_config : Optional[CVConfig]
        Configuration for cross-validation (required if method is 'cv')
    simple_config : Optional[SimpleSplitConfig]
        Configuration for simple split (required if method is 'simple')
    """
    method:        SplitMethod                 = Field(default=SplitMethod.CV, description="Splitting method to use")
    cv_config:     Optional[CVConfig]          = Field(default=None, description="Configuration for cross-validation")
    simple_config: Optional[SimpleSplitConfig] = Field(default=None, description="Configuration for simple split")

    @model_validator(mode="after")
    def validate_configuration(self) -> "SplitConfig":
        """Ensure the correct configuration is provided based on the selected method."""
        if self.method == SplitMethod.CV and self.cv_config is None:
            self.cv_config = self.cv_config or CVConfig()
            self.simple_config = None

        if self.method == SplitMethod.SIMPLE and self.simple_config is None:
            self.simple_config = self.simple_config or SimpleSplitConfig()
            self.cv_config = None

        return self


class SplitIndices(BaseModel):
    """
    Container for train/validation/test split indices.

    Attributes
    ----------
    train_indices : List[int]
        Indices for training samples
    validation_indices : Optional[List[int]]
        Indices for validation samples
    test_indices : Optional[List[int]]
        Indices for test samples
    group : Optional[str]
        Group identifier for this split
    """
    train_indices: List[int]
    validation_indices: Optional[List[int]] = None
    test_indices: Optional[List[int]] = None
    group: Optional[str] = None

    def __iter__(self) -> Iterator[List[int]]:
        """Iterator implementation for unpacking split indices."""
        yield self.train_indices
        if self.validation_indices is not None:
            yield self.validation_indices
        if self.test_indices is not None:
            yield self.test_indices


class SplitResult(BaseModel):
    """
    Container for all splits for a group.

    Attributes
    ----------
    group : str
        Group identifier
    train_test_split : Optional[SplitIndices]
        Full train-test split (if test_interval is specified)
    validation_splits : List[SplitIndices]
        List of validation splits
    """
    group: str
    train_test_split: Optional[SplitIndices] = None
    validation_splits: List[SplitIndices] = Field(default_factory=list)
