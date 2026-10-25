PYTHON ?= ../../.venv/bin/python

.PHONY: format lint

format:
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m black --check .
	$(PYTHON) -m flake8 .
