PYTHON = python3
PIP = pip3
MAIN = src/main.py
TEST_FILE = src/test_pipeline.py
MODEL_DIR = models
DATA_FILE = loan_data.csv

.PHONY: help install setup data train test validate clean lint format security ci all

help:
	@echo "=== ML Project Makefile Commands ==="
	@echo ""
	@echo "Installation:"
	@echo "  make install     - Install dependencies from requirements.txt"
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo ""
	@echo "Model Pipeline:"
	@echo "  make data        - Prepare and preprocess data"
	@echo "  make train       - Train the ML model"
	@echo "  make validate    - Validate model performance"
	@echo ""
	@echo "CI/CD & Code Quality:"
	@echo "  make test        - Run unit tests"
	@echo "  make lint        - Check code quality with flake8"
	@echo "  make format      - Format code with black"
	@echo "  make security    - Check for security issues with bandit"
	@echo "  make ci          - Run full CI pipeline (lint + test + security)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean generated files"
	@echo "  make all         - Run complete pipeline (install + ci + train)"
	@echo ""

# I. Installation Section
install:
	pip install -r requirements.txt

setup:
	python -m venv venv
	pip install --upgrade pip
	pip install -r requirements.txt

# II. Model Execution Steps
data:
	python -c "from src.model_pipeline import prepare_data; import pandas as pd; df = pd.read_csv('$(DATA_FILE)'); prepare_data(df)"

train:
	python $(MAIN) --train

validate:
	python $(MAIN) --validate
	
# III. CI/CD Steps
test:
	python $(TEST_FILE)

lint:
	flake8 src/ --max-line-length=120 --count --statistics

format:
	black src/ --line-length=120

security:
	bandit -r src/ -f html -o security_report.html

ci: lint test security

# Complete pipeline
all: install ci train validate

# Utilities
clean:
	rm -rf $(MODEL_DIR)/*.pkl
	rm -rf $(MODEL_DIR)/*.json
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf .pytest_cache
	rm -rf *.html

# Auto-trigger on file changes
watch:
	while true; do inotifywait -e modify,create,delete --exclude '(__pycache__|\.pyc|\.pkl|\.json)' -r .; make ci; done