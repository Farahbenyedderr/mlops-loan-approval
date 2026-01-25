PYTHON = python
MAIN = src/main.py
MODEL_DIR = models


# ============================
# MLflow Makefile Commands
# ============================

# (Optional) Define the directory where mlruns will be stored

# Configuration
PORT := 5000
MLFLOW_URI := sqlite:///MLflow/mlflow_data/mlflow.db
MLFLOW_DIR := MLflow/mlflow_data
MODEL_NAME := LoanDefaultModel_RF
IMAGE_NAME = mlops-loan-approval-api



.PHONY: help install format format-check sort-imports isort-check security-check api run prepare evaluate train clean lint test ci mlflow-ui expose api docker-build docker-run

# Default target - show help when running 'make' without arguments
.DEFAULT_GOAL := help

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri $(MLFLOW_URI) --port $(PORT) --host 0.0.0.0

clean-mlflow:
	@echo "Cleaning MLflow directory..."
	rm -rf $(MLFLOW_DIR)/mlflow.db
	rm -rf $(MLFLOW_DIR)/artifacts/*
	rm -rf mlruns
	@echo "Done."

help:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  📋 LOAN MODEL - MAKEFILE COMMANDS"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "🚀 QUICK START:"
	@echo "  make install          - Install all dependencies"
	@echo "  make run              - Run the main application"
	@echo "  make api              - Start API server (Swagger at :8000/docs)"
	@echo ""
	@echo "🤖 ML PIPELINE:"
	@echo "  make prepare          - Prepare and clean data"
	@echo "  make train            - Train the machine learning model"
	@echo "  make evaluate         - Evaluate model performance"
	@echo ""
	@echo "🔧 CODE QUALITY:"
	@echo "  make format           - Format code with black"
	@echo "  make format-check     - Check code formatting"
	@echo "  make sort-imports     - Sort imports with isort"
	@echo "  make isort-check      - Check import sorting"
	@echo "  make lint             - Run flake8 linter"
	@echo "  make security-check   - Run security scan with bandit"
	@echo ""
	@echo "🧪 TESTING:"
	@echo "  make test             - Run pytest tests"
	@echo ""
	@echo "🔄 CI/CD:"
	@echo "  make ci               - Run full CI pipeline (format, lint, security, ML)"
	@echo ""
	@echo "🧹 CLEANUP:"
	@echo "  make clean            - Remove generated model files"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "💡 TIP: Run 'make ci' before committing to ensure code quality"
	@echo "════════════════════════════════════════════════════════════════"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install flake8 pytest

format:				# applique black
	black src tests

format-check:
	black --check src tests

sort-imports:               # applique le tri
	isort src tests

isort-check:
	isort --check-only src tests

security-check:            # scan only, fail on any issue
	bandit -r src -ll

api:
	@echo "Starting API server..."
	@echo "Swagger UI available at: http://localhost:8000/docs"
	@echo "ReDoc available at: http://localhost:8000/redoc"
	uvicorn app:app --host 127.0.0.1 --port 8000 --reload

run:
	$(PYTHON) $(MAIN)

prepare: 
	$(PYTHON) $(MAIN) --prepare_data_main

evaluate:
	$(PYTHON) $(MAIN) --evaluate_model_main	

#####
PARAMS ?= 
train: 
	$(PYTHON) $(MAIN) --train_model_main --params '$(PARAMS)'
#####

clean:
	rm -f $(MODEL_DIR)/*.pkl

lint:
	flake8 src/ --max-line-length=120

test:
	pytest -q tests/

ci: format sort-imports lint security-check prepare train evaluate 

expose:
	nohup $(MAKE) mlflow-ui > mlflow.log 2>&1 &
	$(MAKE) api

docker-build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

docker-run:
	@echo "Running container with FastAPI + MLflow..."
	docker run $(IMAGE_NAME)



elastic-kibana-up:
	@echo "Starting Elasticsearch and Kibana..."
	docker-compose up -d elasticsearch kibana
	@echo "Services started!"
	@echo "Elasticsearch: http://localhost:9200"
	@echo "Kibana: http://localhost:5601"

elastic-kibana-down:
	@echo "Stopping Elasticsearch and Kibana..."
	docker-compose stop elasticsearch kibana