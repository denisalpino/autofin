import os
import sys

from pydantic import ValidationError
import pytest

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.schemas.splitting import (
    CVConfig,
    CVWindowType,
    SimpleSplitConfig,
    SplitConfig,
    SplitMethod,
    TimeUnit,
)


class TestSplitConfig:
    """Test cases for SplitConfig model validation and default behavior."""

    @pytest.mark.parametrize(
        "input_config,expected_output",
        [
            (
                {"method": "simple", "simple_config": SimpleSplitConfig()},
                SplitConfig(
                    method=SplitMethod.SIMPLE,
                    cv_config=None,
                    simple_config=SimpleSplitConfig(ratios=(80, 10, 10))
                )
            ),
            (
                {"method": "cv", "cv_config": CVConfig(val_interval="1d")},
                SplitConfig(
                    method=SplitMethod.CV,
                    cv_config=CVConfig(val_interval="1d"),
                    simple_config=None
                )
            ),
            (
                {"method": "cv", "simple_config": SimpleSplitConfig()},
                SplitConfig(
                    method=SplitMethod.CV,
                    cv_config=CVConfig(),
                    simple_config=None
                )
            ),
            (
                {"method": "simple", "cv_config": CVConfig()},
                SplitConfig(
                    method=SplitMethod.SIMPLE,
                    cv_config=None,
                    simple_config=SimpleSplitConfig()
                )
            )
        ]
    )
    def test_valid_configurations(self, input_config, expected_output):
        """Test that valid configurations produce expected results."""
        assert SplitConfig(**input_config) == expected_output

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValidationError."""
        with pytest.raises(ValidationError):
            SplitConfig(method="invalid_method")


class TestCVConfig:
    """Test cases for CVConfig model validation and time interval parsing."""

    def test_valid_complex_configuration(self):
        """Test configuration with all parameters set."""
        config = CVConfig(
            val_folds=4,
            val_interval="11M",
            test_interval="23h",
            train_interval="1d",
            window="rolling"
        )
        assert config.val_interval == "11M"
        assert config.test_interval == "23h"
        assert config.train_interval == "1d"
        assert config.window == CVWindowType.ROLLING

    @pytest.mark.parametrize(
        "invalid_config",
        [
            {"window": "rolling"},  # Missing train_interval
            {"val_folds": 0},  # Value below minimum
            {"val_interval": "1X"},  # Invalid unit
            {"val_interval": "60m"},  # Minutes out of range
            {"val_interval": "0h"},  # Zero value
            {"val_interval": "m"},  # Missing numeric part
            {"val_interval": "16"},  # Missing unit
        ]
    )
    def test_invalid_configurations_raise_errors(self, invalid_config):
        """Test various invalid configurations raise ValidationError."""
        with pytest.raises(ValidationError):
            CVConfig(**invalid_config)

    def test_leading_zero_in_interval(self):
        """Test that intervals with leading zeros are properly normalized."""
        config = CVConfig(val_interval="01m")
        assert config.val_interval == "1m"


class TestSimpleSplitConfig:
    """Test cases for SimpleSplitConfig ratio validation."""

    @pytest.mark.parametrize(
        "invalid_ratios",
        [
            (5, 5),  # Wrong number of elements
            (-1, 5, 5),  # Negative value
            (100, 5, 5),  # Value >= 100
            (100, 0, 0),  # Sum != 100 with large value
        ]
    )
    def test_invalid_ratios_raise_errors(self, invalid_ratios):
        """Test that invalid ratio combinations raise ValidationError."""
        with pytest.raises(ValidationError):
            SimpleSplitConfig(ratios=invalid_ratios)


class TestTimeUnit:
    """Test cases for TimeUnit enum functionality."""

    def test_time_unit_values(self):
        """Test that TimeUnit enum has correct string values."""
        assert TimeUnit.MINUTE.value == "m"
        assert TimeUnit.HOUR.value == "h"
        assert TimeUnit.DAY.value == "d"
        assert TimeUnit.MONTH.value == "M"
        assert TimeUnit.WEEK.value == "w"
