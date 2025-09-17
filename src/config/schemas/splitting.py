from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class CVConfig(BaseModel):
    """
    Конфиг кросс‑валидации по времени c группами.
    Поля соответствуют сигнатуре GroupTimeSeriesSplit.
    """
    val_folds: int = Field(1, ge=1, description="Число валидационных фолдов подряд")
    val_interval: str = Field("7d", description="Размер одного валидационного окна, напр. '1d', '15m', '1M' (предполагаемый интервал переобучения модели)")
    test_interval: Optional[str] = Field(None, description="Размер тестового периода в конце ряда, напр. '5d', '12h', '1M'")
    train_interval: Optional[str] = Field(None, description="Для window='rolling': размер обучающего окна; по умолчанию = interval")
    window: Literal["expanding", "rolling"] = Field("expanding", description="Тип окна для train: нарастающее или скользящее")
    min_train_samples: int = Field(1, ge=1, description="Минимум наблюдений в train для валидного сплита")

    @field_validator("val_interval", "train_interval", "test_interval", mode="before")
    @classmethod
    def _validate_interval(self, s: Optional[str]) -> str:
        """Parse and validatetime interval string into pandas offset object."""
        if s is None: return s

        # Parse interval
        n, unit = s[:-1], s[-1]

        try:
            n = int(n)
        except:
            raise ValueError(f"Unit prefix must be integer, while {n} obtained.")

        if unit == 'm':
            if n < 1 or n >= 60:
                raise ValueError(f'Unit "m" supports only integers from 1 to 60, while {n} obtained.')
            return f"{n}{unit}"
        if unit == 'h':
            if n < 1 or n >= 24:
                raise ValueError(f'Unit "h" supports only integers from 1 to 24, while {n} obtained.')
            return f"{n}{unit}"
        if unit == 'd':
            if n < 1 or n >= 31:
                raise ValueError(f'Unit "d" supports only integers from 1 to 31, while {n} obtained.')
            return f"{n}{unit}"
        if unit == 'M':
            if n < 1 or n >= 12:
                raise ValueError(f'Unit "M" supports only integers from 1 to 12, while {n} obtained.')
            return f"{n}{unit}"
        raise ValueError(f"Unsupported unit {unit} obtained. Use 'm', 'h', 'd', or 'M'")

    @model_validator(mode="after")
    def _check_window_constraints(self):
        if self.window == "rolling" and self.train_interval is None:
            raise ValueError("For window='rolling' must be provided train_interval.")
        return self


class SimpleSplitConfig(BaseModel):
    """Простой детерминированный сплит по порядку наблюдений."""
    split: Tuple[int, int, int] = Field((80, 10, 10), description="Доли в процентах (train, val, test); сумма = 100",)

    @field_validator("split", mode="before")
    @classmethod
    def _validate_split(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(v) != 3:
            raise ValueError("split должен иметь вид (train, val, test) из трёх целых.")
        if any(not (0 <= x <= 100) for x in v):
            raise ValueError("Каждый элемент split должен быть в диапазоне [0, 100].")
        if sum(v) != 100:
            raise ValueError(f"Сумма split должна быть равна 100, получено {sum(v)}.")
        return v


class SplitConfig(BaseModel):
    """
    Унифицированный конфиг: выбирает режим и его параметры.
    - method='cv'  -> использовать кросс-валидацию (по умолчанию)
    - method='simple' -> использовать простой сплит (проценты)
    """
    method: Literal["cv", "simple"] = Field(dafault="cv")
    cv:     Optional[CVConfig] = None
    simple: Optional[SimpleSplitConfig] = None

    @model_validator(mode="after")
    def _fill_defaults(self):
        if self.method == "cv":
            # Заполняем дефолты, если cv не указан
            self.cv = self.cv or CVConfig()
            self.simple = None
        elif self.method == "simple":
            self.simple = self.simple or SimpleSplitConfig()
            self.cv = None
        return self
