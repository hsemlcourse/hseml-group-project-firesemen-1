.PHONY: install lint test train api streamlit docker-up report

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	ruff check src scripts tests
	ruff format --check src scripts tests

test:
	pytest -q

train:
	PYTHONPATH=src python scripts/train.py --data-path data/sample/ethusd_1m_sample.csv --max-rows 120000

api:
	PYTHONPATH=src uvicorn eth_price.api:app --reload --host 0.0.0.0 --port 8000

streamlit:
	PYTHONPATH=src streamlit run src/eth_price/streamlit_app.py

docker-up:
	docker compose up --build

report:
	pandoc report/report.md -o report/report.pdf --pdf-engine=xelatex --resource-path=.:report
