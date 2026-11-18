.PHONY: test run ui
test:
	python -m pytest -q
run:
	uvicorn app.main:app --port 8000
ui:
	streamlit run ui/streamlit_app.py
