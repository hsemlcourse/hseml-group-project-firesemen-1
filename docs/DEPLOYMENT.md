# Деплой проекта

Этот файл относится в основном к CP3, но инфраструктура уже добавлена в CP2.

## FastAPI

```bash
python scripts/train.py
PYTHONPATH=src uvicorn eth_price.api:app --reload --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @docs/assets/predict_request_example.json
```

Swagger UI: `http://localhost:8000/docs`.

## Streamlit

```bash
python scripts/train.py
PYTHONPATH=src streamlit run src/eth_price/streamlit_app.py
```

Интерфейс: `http://localhost:8501`.

## Docker Compose

```bash
docker compose up --build
```

После запуска:

- API: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

Для полноценной проверки `/predict` нужен модельный файл `models/eth_next_close_model.joblib`, который создаётся командой `python scripts/train.py`.
