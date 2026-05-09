# CP2: прогноз ETH/USD Close на 1 минуту вперёд

Студент: Холодилов Семён Максимович\
Группа: БИВ238

Проект решает задачу регрессии временного ряда: по минутным OHLCV-свечам предсказать `Close(t+1)`. Основная метрика - MAE, потому что она измеряется в долларах и прямо показывает среднюю абсолютную ошибку прогноза. Дополнительно считаются RMSE и R2.

## Что именно закрывает CP2

-   Проверка пропусков, дублей, timestamps, OHLCV-логики и временных разрывов.
-   Проверка экстремальных минутных log-return как выбросов.
-   Feature engineering без data leakage: лаги, rolling-признаки, EMA, RSI, временные признаки.
-   Хронологический train/validation/test split без shuffle.
-   Baseline без feature engineering и финансовый baseline `Close(t+1) = Close(t)`.
-   5+ моделей: LinearRegression, Ridge, ElasticNet, RandomForest, ExtraTrees, Ridge+PCA95.
-   Weighted blending ансамбль `Naive + Ridge`.
-   Перебор гиперпараметров на validation split.
-   Ruff, pytest, requirements/pyproject, Dockerfile и docker-compose.
-   Самостоятельный парсер Binance API: `scripts/download_binance_klines.py`.

## Структура

``` text
.
├── data/sample/ethusd_1m_sample.csv
├── docs/DEPLOYMENT.md
├── notebooks/cp2_eth_full_pipeline.ipynb
├── report/
│   ├── report.md
│   ├── report.pdf
│   ├── experiment_results.csv
│   ├── hyperparameter_search_details.csv
│   ├── dataset_description.json
│   ├── cleaning_report.json
│   ├── outlier_report.json
│   └── figures/
├── scripts/
│   ├── download_binance_klines.py
│   ├── make_api_example.py
│   └── train.py
├── src/eth_price/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Данные текущего CP2-прогона

-   120000 строк, 10 колонок;
-   период: `2025-07-20 03:05:00` - `2025-10-11 11:04:00`;
-   пропуски: 0;
-   дубликаты timestamp: 0;
-   некорректные OHLCV-строки: 0;
-   разрывы временного ряда не по 1 минуте: 0;
-   после feature engineering: 119938 строк, 118 признаков;
-   найдено 186 экстремальных минутных log-return по порогу 6 sigma.

Экстремальные доходности не удаляются автоматически, потому что для криптовалют они могут быть реальными рыночными движениями. Удаляются только физически невозможные записи.

## Быстрый запуск

Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python scripts/train.py
```

Linux/macOS:

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/train.py
```

`python scripts/train.py` пересчитывает метрики, графики и модельный артефакт `models/eth_next_close_model.joblib`.

## Самостоятельный парсинг данных

Пример контрольного запуска Binance API:

``` bash
python scripts/download_binance_klines.py \
  --symbol ETHUSDT \
  --interval 1m \
  --start 2025-01-01T00:00:00Z \
  --end 2025-01-01T02:00:00Z \
  --output data/raw/ethusdt_1m_binance_api_check.csv
```

В локальном CP2-прогоне эта команда сохранила 121 строку.

## Проверки качества кода

``` bash
ruff check .
ruff format --check .
pytest -q
```

При сборке этого CP2-архива проверки прошли: ruff без ошибок, pytest - 4 passed.

## Docker

``` bash
docker compose build
docker compose up --build
```

После запуска:

-   FastAPI: `http://localhost:8000/docs`
-   Streamlit: `http://localhost:8501`

Для `POST /predict` сначала должен быть создан файл модели через `python scripts/train.py`.

## Итоговые результаты CP2-прогона на 120000 строках

| Model | Val MAE | Test MAE | Test RMSE | Comment |
|----|---:|---:|---:|----|
| Naive close(t) | 1.8767 | **2.3458** | 4.3833 | лучший test MAE |
| Weighted blend: naive + Ridge | 1.8766 | 2.3469 | **4.3796** | лучший test RMSE, финальный ансамбль |
| Baseline LinearRegression without FE | **1.8759** | 2.3501 | 4.4084 | baseline без FE |
| Ridge | 1.8877 | 2.4123 | 4.8587 | лучшая ML-модель внутри blend |
| ElasticNet | 1.8888 | 2.3910 | 4.7675 | регуляризованная линейная модель |
| LinearRegression | 1.8893 | 2.3922 | 4.7594 | линейная модель на FE |
| Ridge+PCA95 | 2.0069 | 2.5646 | 5.5623 | PCA ухудшил качество |
| RandomForest | 2.6785 | 2.7207 | 4.6479 | деревья хуже линейных моделей |
| ExtraTrees | 2.7043 | 2.8806 | 5.5024 | деревья хуже линейных моделей |

Вывод: на горизонте 1 минута naive baseline `Close(t+1) = Close(t)` очень силён. Weighted blend немного улучшает RMSE, но не улучшает MAE. Поэтому нельзя утверждать, что сложные ML-модели существенно превзошли baseline.
