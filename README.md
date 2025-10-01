autofin
---

[![Status](https://img.shields.io/badge/status-active%20development-orange)](https://github.com/denisalpino/autofin)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/denisalpino/autofin)
[![License](https://img.shields.io/badge/license-All%20rights%20reserved-red)](https://github.com/denisalpino/autofin)

Микро-библиотека для автоматизации предобработки финансовых данных, разработки ML/DL моделей, их анализа и поддержки. **Проект развивается из активных экспериментов** — каждый компонент проверен на реальных финансовых данных.

## 📑 Оглавление

- [autofin](#autofin)
- [📑 Оглавление](#-оглавление)
- [🎯 Что делает библиотеку особенной?](#-что-делает-библиотеку-особенной)
- [🔬 Эксперименты](#-эксперименты)
  - [📉 Стационарность](#-стационарность)
  - [🎯 Шум и аномалии (винсоризация)](#-шум-и-аномалии-винсоризация)
  - [📊 Отбор признаков](#-отбор-признаков)
  - [⚡ Распределение доходностей](#-распределение-доходностей)
  - [⚙️ Обучение моделей](#️-обучение-моделей)
  - [⚙️ Оптимизация торговой стратегии](#️-оптимизация-торговой-стратегии)
  - [🚀 Бэктестирование](#-бэктестирование)
    - [⚙️ Условия тестирования](#️-условия-тестирования)
    - [📊 Результаты торговых стратегий](#-результаты-торговых-стратегий)
- [🚀 Быстрый старт](#-быстрый-старт)
  - [Установка](#установка)
  - [Минимальный рабочий пример](#минимальный-рабочий-пример)
- [🛠️ Как использовать: два подхода](#️-как-использовать-два-подхода)
  - [Декларативный режим (LowCode) — для быстрых экспериментов (В работе)](#декларативный-режим-lowcode--для-быстрых-экспериментов-в-работе)
  - [Императивный режим — для полного контроля (В работе)](#императивный-режим--для-полного-контроля-в-работе)
- [📁 Структура проекта (только важное)](#-структура-проекта-только-важное)
- [🎯 Ключевые компоненты](#-ключевые-компоненты)
  - [Data Loaders](#data-loaders)
  - [Feature Engineering](#feature-engineering)
  - [Time Series Splitting](#time-series-splitting)
  - [Visualization](#visualization)
- [🚀 Рекомендуемые конфигурации](#-рекомендуемые-конфигурации)
- [🧪 Тестирование и качество кода](#-тестирование-и-качество-кода)
- [📈 Дорожная карта развития](#-дорожная-карта-развития)
  - [🔥 Ближайшие задачи](#-ближайшие-задачи)
  - [🚀 В разработке](#-в-разработке)
  - [💡 Идеи для будущего](#-идеи-для-будущего)
- [🤝 Как внести вклад](#-как-внести-вклад)
- [📄 Лицензия](#-лицензия)
- [📞 Контакты](#-контакты)

---

## 🎯 Что делает библиотеку особенной?

| Особенность | Преимущество |
|-------------|--------------|
| **📊 Data-Centric архитектура** | Компоненты созданы на основе глубокого EDA финансовых временных рядов |
| **⚡ Двойной режим работы** | Быстрое прототипирование (LowCode) + полный контроль (императивный) |
| **🛡️ Защита от утечек** | Продвинутая кросс-валидация с группами и временными барьерами |
| **📈 Готовые финансовые фичи** | 20+ технических индикаторов, временные фичи, лаги "из коробки" |

## 🔬 Эксперименты

Библиотека создавалась в процессе активного исследования данных. Вот ключевые инсайты из нашего EDA:

### 📉 Стационарность
Анализ показал, что прямые предсказания цен неэффективны. Решение — переход к прогнозу доходностей. Ниже находится *визуализация ценового ряда и соответствующих доходностей*

![Цены и доходности](docs/img/prices_ytd.png)

### 🎯 Шум и аномалии (винсоризация)
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

### 📊 Отбор признаков
Матрица корреляций выявила сильную мультиколлинеарность индикаторов с базовыми OHLC. Автоматический отбор фичей улучшает качество моделей.

**После отбора признаков:**
![Матрица корреляций (оптимизированная)](docs/img/corr_fit.png)

### ⚡ Распределение доходностей
Анализ распределения помог выбрать оптимальные методы предобработки и валидации.

![Распределение доходностей](docs/img/returns_dist.png)

### ⚙️ Обучение моделей

logloss + auc

### ⚙️ Оптимизация торговой стратегии

...

### 🚀 Бэктестирование

<div align="center">

#### ⚙️ Условия тестирования

<table>
  <tr>
    <td align="center" width="33%">
      <strong>📊 Настройки тестирования</strong><br>
      <table>
        <tr><td>Комиссия за сделку</td><td>⚠️ 0.02%</td></tr>
        <tr><td>Проскальзывание</td><td>⚠️ 0.01%</td></tr>
        <tr><td>Период</td><td>⚠️ 84 дня</td></tr>
        <tr><td>Депозит</td><td>⚠️ $1,000</td></tr>
      </table>
    </td>
    <td align="center" width="33%">
      <strong>🎯 Параметры стратегии</strong><br>
      <table>
        <tr><td>Регулирование магнитуды входов</td><td>❌</td></tr>
        <tr><td>Переворот позиции</td><td>✅</td></tr>
        <tr><td>Базовый порог</td><td>✅</td></tr>
        <tr><td>Дельта на вход в лонг</td><td>✅</td></tr>
        <tr><td>Эпсилон на выход из лонга</td><td>✅</td></tr>
        <tr><td>Дельта на вход в шорт</td><td>✅</td></tr>
        <tr><td>Эпсилон на выход из шорта</td><td>✅</td></tr>
      </table>
    </td>
    <td align="center" width="33%">
      <strong>🛡️ Управление рисками</strong><br>
      <table>
        <tr><td>Стоп-лосс</td><td>✅</td></tr>
        <tr><td>Тейк-профит</td><td>❌</td></tr>
        <tr><td>Трейлинг-стоп</td><td>✅</td></tr>
      </table>
    </td>
  </tr>
</table>

<br>

#### 📊 Результаты торговых стратегий

<table>
  <tr>
    <td align="center" width="50%" style="vertical-align: top;">
      <strong>BNB/USDT</strong>
    </td>
    <td align="center" width="50%" style="vertical-align: top;">
      <strong>LINK/USDT</strong>
    </td>
  </tr>
  <tr>
    <td align="center" style="vertical-align: top;">
      <img src="docs/img/BNB_backtesting.png" alt="BNB Backtesting Results" width="95%">
    </td>
    <td align="center" style="vertical-align: top;">
      <img src="docs/img/LINK_backtesting.png" alt="LINK Backtesting Results" width="95%">
    </td>
  </tr>
  <tr>
    <td align="center" style="vertical-align: top;">
      <details>
        <summary><em>📈 Показать метрики BNB</em></summary>
        <pre>

End Value                        $1,640 ✅
Total Return                    +64.05% ✅
B&H Return                         +49.41%
Total Fees Paid                       $249
Max Drawdown                         8.65%
Max Drawdown Duration     15 days 04:00:00
Total Trades                           507
Win Rate                         62.52% ✅
Best Trade                          +2.89%
Worst Trade                         -1.55%
Avg Winning Trade                +0.45% ⚠️
Avg Losing Trade                 -0.48% ⚠️
Profit Factor                      1.54 ✅
Expectancy                        $1.26 ✅
Sharpe Ratio                       6.55 ✅
        </pre>
      </details>
    </td>
    <td align="center" style="vertical-align: top;">
      <details>
        <summary><em>📈 Показать метрики LINK</em></summary>
        <pre>

End Value                        $2,539 ✅
Total Return                   +153.87% ✅
B&H Return                        +101.85%
Total Fees Paid                       $585
Max Drawdown                         6.05%
Max Drawdown Duration               7 days
Total Trades                           855
Win Rate                         52.05% ⚠️
Best Trade                          +8.86%
Worst Trade                         -1.54%
Avg Winning Trade                +0.59% ✅
Avg Losing Trade                 -0.41% ✅
Profit Factor                      1.54 ✅
Expectancy                         $1.8 ✅
Sharpe Ratio                       8.98 ✅
        </pre>
      </details>
    </td>
  </tr>

  <tr>
    <td align="center" width="50%" style="vertical-align: top;">
      <strong>ETH/USDT</strong>
    </td>
    <td align="center" width="50%" style="vertical-align: top;">
      <strong>BTC/USDT</strong>
    </td>
  </tr>
  <tr>
    <td align="center" style="vertical-align: top;">
      <img src="docs/img/ETH_backtesting.png" alt="ETH Backtesting Results" width="95%">
    </td>
    <td align="center" style="vertical-align: top;">
      <img src="docs/img/BTC_backtesting.png" alt="BTC Backtesting Results" width="95%">
    </td>
  </tr>
  <tr>
    <td align="center" style="vertical-align: top;">
      <details>
        <summary><em>📈 Показать метрики ETH</em></summary>
        <pre>

End Value                          $905 ❌
Total Return                      -9.5% ❌
B&H Return                         +40.38%
Total Fees Paid                    $146.34
Max Drawdown                        14.56%
Max Drawdown Duration     65 days 22:00:00
Total Trades                           378
Win Rate                         55.03% ⚠️
Best Trade                          +3.13%
Worst Trade                         -1.55%
Avg Winning Trade                +0.32% ❌
Avg Losing Trade                 -0.45% ❌
Profit Factor                      0.87 ❌
Expectancy                       -$0.25 ❌
Sharpe Ratio                      -1.64 ❌
        </pre>
      </details>
    </td>
    <td align="center" style="vertical-align: top;">
      <details>
        <summary><em>📈 Показать метрики BTC</em></summary>
        <pre>

End Value                        $1,012 ⚠️
Total Return                      +1.2% ⚠️
B&H Return                         +50.87%
Total Fees Paid                        $95
Max Drawdown                        13.74%
Max Drawdown Duration     62 days 05:15:00
Total Trades                           218
Win Rate                         52.75% ⚠️
Best Trade                         +12.03%
Worst Trade                         -1.55%
Avg Winning Trade                +0.47% ❌
Avg Losing Trade                 -0.50% ❌
Profit Factor                      1.02 ❌
Expectancy                        $0.05 ❌
Sharpe Ratio                       0.33 ❌
        </pre>
      </details>
    </td>
  </tr>

  <tr>
    <td align="center" colspan="2" style="vertical-align: top;">
      <strong>OP/USDT</strong>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2" style="vertical-align: top;">
      <img src="docs/img/OP_backtesting.png" alt="OP Backtesting Results" width="45%">
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2" style="vertical-align: top;">
      <details>
        <summary><em>📈 Показать метрики OP</em></summary>
        <pre>

End Value                        $2,459 ⚠️
Total Return                     145.9% ⚠️
B&H Return                          179.6%
Total Fees Paid                      265.3
Max Drawdown                         11.4%
Max Drawdown Duration     21 days 10:30:00
Total Trades                           400
Win Rate                          53.5% ⚠️
Best Trade                          10.68%
Worst Trade                         -1.46%
Avg Winning Trade                 0.96% ✅
Avg Losing Trade                 -0.61% ✅
Profit Factor                      1.79 ✅
Expectancy                        $3.65 ✅
Sharpe Ratio                        9.1 ✅
        </pre>
      </details>
    </td>
  </tr>
</table>
</div>

## 🚀 Быстрый старт

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

## 🛠️ Как использовать: два подхода

### Декларативный режим (LowCode) — для быстрых экспериментов (В работе)

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

### Императивный режим — для полного контроля (В работе)

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

### Data Loaders
- **Умная загрузка**: автоматическое определение формата (CSV/Parquet/Excel/JSON)
- **Валидация данных**: проверка обязательных колонок (OHLC)
- **Маппинг колонок**: приведение к стандартным именам

```python
loader = DataLoader({
    'mapping': {'timestamp': 'date', 'close': 'price_close'},
    'read_options': {'sep': ';'}
})
```

### Feature Engineering
- **Временные фичи**: циклическое кодирование (час, день недели и т.д.)
- **Технические индикаторы**: RSI, MACD, ATR, Bollinger Bands (+15 других)
- **Статистики**: лаги, скользящие средние, доходности
- **Готовые конфиги**: оптимальные наборы фичей

### Time Series Splitting
- **Финансовая адаптация**: настройка временных интервалов, а не долей или количества наблюдений
- **GroupTimeSeriesSplit**: защита от утечек между тикерами
- **Временные барьеры**: padding между train/val/test
- **Скользящие окна**: expanding/rolling для реалистичной валидации

### Visualization
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

## 📞 Контакты

- **Author**: [Denis Alpino](https://github.com/denisalpino)
- **Repo**: [https://github.com/denisalpino/autofin](https://github.com/denisalpino/autofin)
- **Issues**: [Report a bug/idea](https://github.com/denisalpino/autofin/issues)
- **Discussions**: [Feature discussion](https://github.com/denisalpino/autofin/discussions)

---

> **💡 Проект живёт и развивается вместе с вашими экспериментами!** Каждая новая идея, баг-репорт или PR делает библиотеку лучше.

*Последнее обновление: Сентябрь 2025 г.*
