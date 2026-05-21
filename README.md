# ETHUSD Price Prediction — CP3

Проект решает учебную ML-зачу: прогнозирование цены закрытия следующей минутной свечи ETHUSD по последним минутным OHLCV-свечам Binance.

CP3 добавляет к результатам предыдущих этапов:

- FastAPI-сервис для инференса модели;
- Streamlit-интерфейс для демонстрации;
- Docker/Docker Compose для локального деплоя;
- PDF/Markdown-отчёт;
- тесты API и примеры запросов.

> Важно: это учебный ML-проект, а не финансовая рекомендация. На контрольном пересчёте для CP3 простой persistence baseline оказался лучше Ridge по RMSE на test-части, поэтому API возвращает оба прогноза и отдельно показывает рекомендованный вариант по метрикам.

---

## Структура проекта

```text
.
├── app/
│   ├── main.py                  # FastAPI API: /health, /model-info, /predict
│   └── streamlit_app.py          # Streamlit demo UI
├── data/
│   └── raw/
│       └── ETHUSD_1m_Binance.zip # полный датасет, не хранится в git при большом размере
├── examples/
│   ├── sample_candles.csv        # маленький CSV для Streamlit-демо
│   └── sample_payload.json       # пример JSON-запроса к /predict
├── models/
│   ├── final_model.joblib        # сохранённая модель
│   └── model_metadata.json       # метрики, признаки, описание сплита
├── reports/
│   ├── cp3_report.md             # отчёт CP3 в Markdown
│   ├── cp3_report.pdf            # отчёт CP3 в PDF
│   └── video_link.md             # ссылка на видео работы деплоя
├── scripts/
│   ├── train_export_model.py     # переобучение и экспорт модели
│   └── curl_predict.sh           # пример curl-запроса
├── src/
│   └── eth_cp3/                  # код подготовки признаков и инференса
├── tests/
│   └── test_api.py               # smoke-тесты API
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Данные

Основной датасет: `ETHUSD_1m_Binance.zip`, внутри которого лежит CSV с минутными свечами ETHUSD.

Для обучения/пересчёта модели архив должен лежать здесь:

```text
data/raw/ETHUSD_1m_Binance.zip
```

Полный CSV не нужно загружать через Streamlit. Интерфейс нужен для демонстрации инференса, поэтому для него используется маленький файл:

```text
examples/sample_candles.csv
```

Это сделано специально: браузерная загрузка больших файлов в Streamlit ограничена и не нужна для проверки деплоя.

---

## Быстрый запуск на Windows PowerShell

Команды ниже выполняются из корня репозитория.

### 1. Создать виртуальное окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если PowerShell запрещает активацию окружения:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### 2. Проверить тесты и линтер

```powershell
$env:PYTHONPATH="src;."
python -m pytest -q
python -m ruff check .
```

Ожидаемо тесты API должны проходить:

```text
2 passed
```

### 3. Запустить FastAPI

```powershell
$env:PYTHONPATH="src;."
python -m uvicorn app.main:app --reload
```

После запуска открыть:

```text
http://localhost:8000/docs
```

Проверка health endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Ожидаемый ответ:

```text
status
------
ok
```

Информация о модели:

```powershell
Invoke-RestMethod http://localhost:8000/model-info
```

### 4. Проверить `/predict`

В новом PowerShell-терминале, пока FastAPI продолжает работать:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src;."
Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -InFile "examples\sample_payload.json"
```

В ответе должны быть поля:

```text
model_prediction
persistence_baseline_prediction
recommended_prediction
last_close
model_used
```

### 5. Запустить Streamlit

FastAPI должен продолжать работать в первом терминале.

Во втором терминале:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src;."
python -m streamlit run app\streamlit_app.py
```

Открыть:

```text
http://localhost:8501
```

Для проверки загружать файл:

```text
examples/sample_candles.csv
```

Не загружать полный `ETHUSD_1m_Binance.csv`: он слишком большой для браузерного демо и для CP3 не нужен.

---

## Запуск через Docker

Из корня проекта:

```powershell
docker compose up --build
```

После сборки открыть:

```text
FastAPI:   http://localhost:8000/docs
Streamlit: http://localhost:8501
```

Остановить контейнеры:

```powershell
docker compose down
```

---

## Переобучение модели

Чтобы заново пересчитать модель и метрики, положить архив с данными сюда:

```text
data/raw/ETHUSD_1m_Binance.zip
```

Затем выполнить:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src;."
python scripts\train_export_model.py
```

Скрипт обновляет:

```text
models/final_model.joblib
models/model_metadata.json
```

Метаданные можно посмотреть так:

```powershell
Get-Content models\model_metadata.json
```

---

## Текущий честный статус модели

Финальная модель для деплоя: `Ridge(alpha=1.0, solver=lsqr)`.

На test-части текущего CP3-пересчёта:

| Подход | MAE | RMSE | MAPE, % | Direction accuracy, % |
|---|---:|---:|---:|---:|
| Ridge | 3.0068 | 6.3111 | 0.0707 | 49.85 |
| Persistence baseline | 2.9404 | 5.9451 | 0.0690 | 0.47 |

Вывод: по RMSE baseline оказался лучше Ridge, поэтому в API отдельно возвращается `recommended_prediction`, выбранный по метрикам из `models/model_metadata.json`.

---


## Типичные проблемы

### `PYTHONPATH=src` не работает в PowerShell

Это Linux/macOS-синтаксис. В PowerShell нужно так:

```powershell
$env:PYTHONPATH="src;."
```

После этого запускать нужную команду, например:

```powershell
python -m uvicorn app.main:app --reload
```

### `ModuleNotFoundError: No module named 'app'`

Скорее всего, CP3-файлы распакованы во вложенную папку `cp3_submit`, а не в корень репозитория.

Проверить:

```powershell
Get-ChildItem
Get-ChildItem .\cp3_submit
```

Если внутри `cp3_submit` лежат `app`, `src`, `models`, `reports`, нужно скопировать их в корень:

```powershell
Copy-Item .\cp3_submit\* . -Recurse -Force
```

### `Fatal error in launcher` у pip

Такое бывает, если `.venv` был скопирован из другой папки. Нужно удалить и создать окружение заново:

```powershell
deactivate
Remove-Item -Recurse -Force .\.venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если `.venv` не удаляется:

```powershell
cmd /c rmdir /s /q .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Streamlit не принимает большой CSV

Для демонстрации надо использовать:

```text
examples/sample_candles.csv
```

Полный датасет используется только для обучения через `scripts/train_export_model.py`, а не для загрузки в веб-интерфейс.
