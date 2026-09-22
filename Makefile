ifeq ($(OS),Windows_NT)
SHELL := C:/Program Files/Git/bin/bash.exe
.SHELLFLAGS := -c
endif

PYTHON ?= python3

.PHONY: setup build lint test-dev test-review

setup:
	$(PYTHON) -m pip install --requirement requirements.lock

build:
	$(PYTHON) -m compileall -q src tests/dev
	PYTHONPATH=src $(PYTHON) -m orbit_predict.cli --check-inputs

lint:
	$(PYTHON) -m compileall -q src tests/dev

test-dev:
	mkdir -p reports
	PYTHONPATH=src $(PYTHON) -m pytest tests/dev --junitxml=reports/dev.xml

test-review:
	mkdir -p reports tests/review
	PYTHONPATH=src $(PYTHON) -m pytest tests/review --junitxml=reports/review.xml

