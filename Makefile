.PHONY: run test lint format demo-data

VENV := .venv/bin

run:
	$(VENV)/streamlit run app/main.py

test:
	$(VENV)/pytest tests

lint:
	$(VENV)/ruff check .
	$(VENV)/black --check .

format:
	$(VENV)/ruff check --fix .
	$(VENV)/black .

demo-data:
	$(VENV)/python scripts/make_demo_data.py
