# autofin

[![Status](https://img.shields.io/badge/status-active%20development-orange)](https://github.com/denisalpino/autofin)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/denisalpino/autofin)
[![License](https://img.shields.io/badge/license-All%20rights%20reserved-red)](https://github.com/denisalpino/autofin)

Микро-библиотека для автоматизации предобработки финансовых данных, разработки ML/DL моделей, их анализа и поддержки. **Проект развивается из активных экспериментов** — каждый компонент проверен на реальных финансовых данных.

## 📑 Оглавление

- [🎯 Что делает библиотеку особенной?](#-что-делает-библиотеку-особенной)
- [🚀 Быстрый старт (2 минуты)](#-быстрый-старт-2-минуты)
  - [Установка](#установка)
  - [Минимальный рабочий пример](#минимальный-рабочий-пример)
- [🔬 Основано на реальных экспериментах](#-основано-на-реальных-экспериментах)
  - [📉 Цены нестационарны — прогнозируем доходности](#-цены-нестационарны--прогнозируем-доходности)
  - [🎯 Винсоризация убирает шум без потери информации](#-винсоризация-убирает-шум-без-потери-информации)
  - [📊 Feature Selection критически важен](#-feature-selection-критически-важен)
  - [⚡ Распределение доходностей — ключ к пониманию рынка](#-распределение-доходностей--ключ-к-пониманию-рынка)
- [🛠️ Как использовать: два подхода](#️-как-использовать-два-подхода)
  - [1. Декларативный режим (LowCode)](#1-декларативный-режим-lowcode--для-быстрых-экспериментов-в-работе)
  - [2. Императивный режим](#2-императивный-режим--для-полного-контроля-в-работе)
- [📁 Структура проекта](#-структура-проекта-только-важное)
- [🎯 Ключевые компоненты](#-ключевые-компоненты)
  - [Data Loaders](#data-loaders-srcdataloaders)
  - [Feature Engineering](#feature-engineering-srcfeatures)
  - [Time Series Splitting](#time-series-splitting-srcdatasplitters)
  - [Visualization](#visualization-srcutilsvisualization)
- [🚀 Рекомендуемые конфигурации](#-рекомендуемые-конфигурации)
- [🧪 Тестирование и качество кода](#-тестирование-и-качество-кода)
- [📈 Дорожная карта развития](#-дорожная-карта-развития)
  - [🔥 Ближайшие задачи](#-ближайшие-задачи)
  - [🚀 В разработке](#-в-разработке)
  - [💡 Идеи для будущего](#-идеи-для-будущего)
- [🤝 Как внести вклад](#-как-внести-вклад)
- [📄 Лицензия](#-лицензия)
- [📞 Контакты и поддержка](#-контакты-и-поддержка)

---

## 🎯 Что делает библиотеку особенной?

| Особенность | Преимущество |
|-------------|--------------|
| **📊 Data-Centric архитектура** | Компоненты созданы на основе глубокого EDA финансовых временных рядов |
| **⚡ Двойной режим работы** | Быстрое прототипирование (LowCode) + полный контроль (императивный) |
| **🛡️ Защита от утечек** | Продвинутая кросс-валидация с группами и временными барьерами |
| **📈 Готовые финансовые фичи** | 20+ технических индикаторов, временные фичи, лаги "из коробки" |

## 🚀 Быстрый старт (2 минуты)

### Установка
```bash
git clone https://github.com/denisalpino/autofin.git
cd autofin
pip install -r requirements.txt
```

### Минимальный рабочий пример
```python
from src.data.loaders.data_loader import DataLoader
from src.features.feature_builder import FeatureBuilder
from src.config.config_loader import load_config

# Загрузка конфигурации (все настройки в одном месте)
config = load_config("configs/default.yaml")

# Загрузка данных (поддержка CSV/Parquet/Excel/JSON)
loader = DataLoader(config.data_loader)
dfs = loader.load("data/raw")  # Автоматически определяет формат
df = dfs[0]

# Построение признаков (20+ индикаторов готовы к использованию)
feature_builder = FeatureBuilder(config.features)
features_df = feature_builder.build_features(df)

print(f"Создано {len(features_df.columns)} признаков")
print(features_df[['timestamp', 'close', 'RSI_14', 'MACD_12_26_9']].tail())
```

## 🔬 Основано на реальных экспериментах

Библиотека создавалась в процессе активного исследования данных. Вот ключевые инсайты из нашего EDA:

### 📉 Цены нестационарны — прогнозируем доходности
Анализ показал, что прямые предсказания цен неэффективны. Решение — переход к прогнозу доходностей. Ниже находится *визуализация ценового ряда и соответствующих доходностей*

![Цены и доходности](docs/img/prices_ytd.png)

### 🎯 Винсоризация убирает шум без потери информации
99.9% винсоризация значительно уменьшает влияние экстремальных значений, сохраняя структуру временного ряда.

<div align="center">
<table>
  <tr>
    <td align="center"><strong>До винсоризации</strong></td>
    <td align="center"><strong>После винсоризации</strong></td>
  </tr>
  <tr>
    <td><img src="docs/img/returns_ytd.png" alt="Доходности до обработки" width="95%"></td>
    <td><img src="docs/img/returns_win_ytd.png" alt="Доходности после обработки" width="95%"></td>
  </tr>
  <tr>
    <td align="center"><em>Высокая волатильность, экстремальные выбросы</em></td>
    <td align="center"><em>Сглаженное распределение, сохранена структура</em></td>
  </tr>
</table>
</div>

### 📊 Feature Selection критически важен
Матрица корреляций выявила сильную мультиколлинеарность индикаторов с базовыми OHLC. Автоматический отбор фичей улучшает качество моделей.

**После отбора признаков:**
![Матрица корреляций (оптимизированная)](docs/img/corr_fit.png)

### ⚡ Распределение доходностей — ключ к пониманию рынка
Анализ распределения помог выбрать оптимальные методы предобработки и валидации.

![Распределение доходностей](docs/img/returns_dist.png)

## 🛠️ Как использовать: два подхода

### 1. Декларативный режим (LowCode) — для быстрых экспериментов (В работе)

Создайте YAML-конфиг и запустите пайплайн:

```yaml
# configs/quick_experiment.yaml
data_loader:
  mapping:
    timestamp: timestamp
    open: open
    high: high
    low: low
    close: close
    volume: volume

features:
  time_features: [HOUR_SIN, HOUR_COS, DAY_OF_WEEK_SIN]
  returns:
    column: close
    method: PERCENT
    period: 1
  indicators:
    - name: RSI
      window: 14
    - name: MACD
      short_window: 12
      long_window: 26
    - name: ATR
      window: 14
  lags:
    - columns: [returns]
      lags: [1, 2, 3, 5, 10]

splitting:
  mode: group_time_series
  params:
    k_folds: 5
    train_interval: 4M
    val_interval: 7d
    test_interval: 14d
    padding: 45d
    window: rolling
```

```python
from src.config.config_loader import load_config
from src.pipeline import Pipeline

config = load_config("configs/quick_experiment.yaml")
pipeline = Pipeline(config=config)

# Весь пайплайн выполняется автоматически
results = pipeline.run("data/raw")
```

### 2. Императивный режим — для полного контроля (В работе)

Используйте отдельные компоненты там, где это нужно:

```python
import pandas as pd
from src.data.loaders.data_loader import DataLoader
from src.features.feature_builder import FeatureBuilder
from src.data.splitters.cross_validation import GroupTimeSeriesSplit

# Загрузка данных
loader = DataLoader()
dfs = loader.load("data/raw/*.csv")
df = pd.concat(dfs)

# Фича инжиниринг
feature_builder = FeatureBuilder()
features = feature_builder.build_features(df)

# Продвинутая кросс-валидация
cv = GroupTimeSeriesSplit(
    k_folds=5,
    val_interval="7d",
    train_interval="4M",
    test_interval="14d",
    padding="45d",
    window="rolling"
)

split_result = cv.split(
    groups=df['ticker'],
    timestamps=pd.to_datetime(df['timestamp'])
)

# Визуализация сплитов
fig = cv.plot_split(
    y=df['returns'],
    groups=df['ticker'],
    timestamps=pd.to_datetime(df['timestamp'])
    theme="dark",
    title="Cross-Validation Scheme"
)
fig.show()
```

![Схема кросс-валидации](docs/img/cross_val.png)

*Визуализация временных сплитов с группами — защита от утечек*

## 📁 Структура проекта (только важное)

```
autofin/
├── src/                   # Ядро библиотеки
│   ├── config/            # Pydantic-схемы для валидации
│   ├── data/              # Загрузка, обработка, разбиение
│   ├── features/          # 20+ технических индикаторов, лаговые и временные признаки
│   └── utils/             # Визуализация и утилиты
├── notebooks/             # 🧪 Живые эксперименты!
│   ├── 01_data_engineering/
│   └── 02_model_engineering/
├── docs/img/              # 📊 Все графики из EDA
└── tests/                 # ✅ Юнит-тесты
```

## 🎯 Ключевые компоненты

### Data Loaders (`src/data/loaders/`)
- **Умная загрузка**: автоматическое определение формата (CSV/Parquet/Excel/JSON)
- **Валидация данных**: проверка обязательных колонок (OHLC)
- **Маппинг колонок**: приведение к стандартным именам

```python
loader = DataLoader({
    'mapping': {'timestamp': 'date', 'close': 'price_close'},
    'read_options': {'sep': ';'}
})
```

### Feature Engineering (`src/features/`)
- **Временные фичи**: циклическое кодирование (час, день недели и т.д.)
- **Технические индикаторы**: RSI, MACD, ATR, Bollinger Bands (+15 других)
- **Статистики**: лаги, скользящие средние, доходности
- **Готовые конфиги**: оптимальные наборы фичей

### Time Series Splitting (`src/data/splitters/`)
- **Финансовая адаптация**: настройка временных интервалов, а не долей или количества наблюдений
- **GroupTimeSeriesSplit**: защита от утечек между тикерами
- **Временные барьеры**: padding между train/val/test
- **Скользящие окна**: expanding/rolling для реалистичной валидации

### Visualization (`src/utils/visualization/`)
- **Графики цен/доходностей**: интерактивные Plotly-диаграммы
- **Матрицы корреляций**: тепловые карты с аннотациями
- **Диагностика моделей**: feature importance, кривые обучения

## 🚀 Рекомендуемые конфигурации

```yaml
features:
  time_features: [HOUR_SIN, HOUR_COS, MINUTE_SIN, MINUTE_COS]
  returns:
    method: LOG
    period: 1
  indicators:
    - name: RSI
      window: 14
    - name: ATR
      window: 10
  lags:
    - columns: [returns, RSI_14]
      lags: [1, 2, 3, 5, 10]
```

## 🧪 Тестирование и качество кода

```bash
# Запуск всех тестов
pytest -vv

# Тесты с покрытием
pytest --cov=src --cov-report=html
```

## 📈 Дорожная карта развития

### 🔥 Ближайшие задачи
- [ ] **AutoML для финансов**: автоматический подбор фичей и гиперпараметров
- [ ] **Ensemble модели**: комбинирование предсказаний нескольких моделей
- [ ] **Реальное бэктестирование**: интеграция с vectorbt

### 🚀 В разработке
- [ ] **LSTM**: Разработать пайплайн обучения LSTM-подобных моделей
- [ ] **Продакшн-готовность**: Docker, API, документация

### 💡 Идеи для будущего
- [ ] **LoRA**: Провести ряд эксперементов с трансфером LoRA на архитектуру LSTM
- [ ] **JEPA**: Интегрировать обучаемые эмбеддинги с CHARM архитектурой для многомерных временных рядов с описанием каналов данных
- [ ] **BI-аналитика**: подключить real-time и post-train дашборды на основе имеющихся визуализаций
- [ ] **Социальные сигналы**: интеграция с новостями и соцсетями

## 🤝 Как внести вклад

Библиотека растет из экспериментов — ваш опыт важен!

1. **Найдите проблему** или идею для улучшения
2. **Обсудите в issues** перед coding
3. **Сделайте форк** и создайте feature branch
4. **Добавьте тесты** для новой функциональности
5. **Обновите документацию** и примеры
6. **Откройте Pull Request** с описанием изменений

## 📄 Лицензия

На текущий момент проект распространяется под "All rights reserved". Мы активно рассматриваем переход на открытую лицензию (скорее всего MIT или Apache 2.0).

## 📞 Контакты и поддержка

- **Author**: [Denis Alpino](https://github.com/denisalpino)
- **Repo**: [https://github.com/denisalpino/autofin](https://github.com/denisalpino/autofin)
- **Issues**: [Report a bug/idea](https://github.com/denisalpino/autofin/issues)
- **Discussions**: [Feature discussion](https://github.com/denisalpino/autofin/discussions)

---

> **💡 Проект живёт и развивается вместе с вашими экспериментами!** Каждая новая идея, баг-репорт или PR делает библиотеку лучше.

*Последнее обновление: Сентябрь 2025 г.*
