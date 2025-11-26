PYTHON = python3
PIP = pip3
MAIN = src/main.py
MODEL_DIR = models
DATA_FILE = data/loan_data.csv

.PHONY: help install setup prepare train validate test clean lint format security ci all

help:
	@echo "=== MLOps Project Commands ==="
	@echo ""
	@echo "Installation :"
	@echo "  make install     - Installer les dépendances"
	@echo "  make setup       - Créer un environnement virtuel (optionnel)"
	@echo ""
	@echo "Pipeline ML :"
	@echo "  make prepare     - Préparer et nettoyer les données"
	@echo "  make train       - Entraîner le modèle"
	@echo "  make validate    - Valider un modèle existant"
	@echo ""
	@echo "Qualité & CI/CD :"
	@echo "  make test        - Lancer les tests unitaires"
	@echo "  make lint        - Vérifier la qualité du code"
	@echo "  make format      - Formater le code (black)"
	@echo "  make security    - Analyse de sécurité bandit"
	@echo "  make ci          - Pipeline CI complet"
	@echo ""
	@echo "Utilitaires :"
	@echo "  make clean       - Nettoyer les fichiers générés"
	@echo "  make all         - Pipeline complet (install + ci + train)"
	@echo ""

# ============================
# INSTALLATION
# ============================

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup:
	$(PYTHON) -m venv venv
	./venv/bin/pip install -r requirements.txt

# ============================
# PIPELINE ML
# ============================

prepare:
	$(PYTHON) -c "import pandas as pd; from src.model import prepare_data; df=pd.read_csv('$(DATA_FILE)'); df2=prepare_data(df); print('Preparation OK')"

train:
	$(PYTHON) $(MAIN) --train

validate:
	$(PYTHON) $(MAIN) --validate

# ============================
# CI/CD
# ============================

test:
	pytest -q tests/

lint:
	flake8 src/ --max-line-length=120 --statistics

format:
	black src/ --line-length=120

security:
	bandit -r src/ -f html -o security_report.html

ci: lint test security

# ============================
# UTILITAIRES
# ============================

clean:
	rm -rf $(MODEL_DIR)/*.pkl
	rm -rf __pycache__ src/__pycache__ .pytest_cache *.html

all: install ci train validate
