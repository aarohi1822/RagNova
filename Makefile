install:
	python -m pip install -e ".[dev]"

run-api:
	uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

run-ui:
	streamlit run frontend/streamlit_app.py --server.port 8501

test:
	pytest -q

lint:
	ruff check backend frontend tests scripts

