install:
	python -m pip install -r requirements.txt

install-eval:
	python -m pip install -r requirements-eval.txt

run:
	streamlit run frontend/streamlit_app.py --server.port 8501

test:
	pytest -q

lint:
	ruff check frontend tests eval
