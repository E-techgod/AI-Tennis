PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit

.PHONY: venv install run test check

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV_PIP) install -r requirements.txt

run:
	$(STREAMLIT) run rag_app.py

test:
	$(VENV_PYTHON) -m unittest test_rag_agent.py

check:
	$(VENV_PYTHON) -m py_compile rag_agent.py rag_app.py test_rag_agent.py
