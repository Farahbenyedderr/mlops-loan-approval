

# Configuration
DATA_FILE = data/loan_data.csv
MODEL_PATH = models/loan_risk_model.pkl
PROCESSED_NAME = prepared_data.csv

# Couleurs
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
BLUE = \033[0;34m
NC = \033[0m # No Color

.PHONY: all prepare train evaluate full clean install help setup check

# Cible par défaut
all: help

# Installation des dépendances
install:
	@echo "$(GREEN) Installation des dépendances...$(NC)"
	pip install -r requirements.txt

# Setup de la structure
setup:
	@echo "$(BLUE) Création de la structure de dossiers...$(NC)"
	mkdir -p data results models logs
	@echo "$(GREEN) Structure créée: data/, results/, models/, logs/$(NC)"
	@echo "$(YELLOW) Placez votre fichier 'loan_data.csv' dans le dossier 'data/'$(NC)"

# Vérification des données source
check-source:
	@echo "$(BLUE) Vérification du fichier source...$(NC)"
	@if [ -f "$(DATA_FILE)" ]; then \
		echo "$(GREEN) Fichier source trouvé: $(DATA_FILE)$(NC)"; \
	else \
		echo "$(YELLOW)  Fichier source non trouvé: $(DATA_FILE)$(NC)"; \
		echo "$(YELLOW) Placez votre fichier 'loan_data.csv' dans le dossier 'data/'$(NC)"; \
		exit 1; \
	fi

# Vérification des données préparées
check-prepared:
	@echo "$(BLUE) Vérification des données préparées...$(NC)"
	@if ls results/*.csv >/dev/null 2>&1; then \
		echo "$(GREEN) Fichier préparé trouvé$(NC)"; \
	else \
		echo "$(YELLOW) Aucun fichier préparé dans results/$(NC)"; \
		echo " Exécutez: make prepare"; \
		exit 1; \
	fi

# Préparation seulement
prepare: setup check-source
	@echo "$(GREEN) Préparation des données...$(NC)"
	python src/main.py --mode prepare --output_name $(PROCESSED_NAME)

# Entraînement seulement (utilise les données préparées)
train: check-prepared
	@echo "$(GREEN) Entraînement du modèle...$(NC)"
	python src/main.py --mode train --model_path $(MODEL_PATH) --optimize

# Entraînement rapide (sans optimisation)
train-fast: check-prepared
	@echo "$(GREEN) Entraînement rapide...$(NC)"
	python src/main.py --mode train --model_path $(MODEL_PATH)

# Évaluation seulement
evaluate: check-prepared
	@echo "$(GREEN) Évaluation du modèle...$(NC)"
	python src/main.py --mode evaluate --model_path $(MODEL_PATH)

# Pipeline complète
full: setup check-source
	@echo "$(GREEN) Pipeline complète...$(NC)"
	python src/main.py --mode full --model_path $(MODEL_PATH) --optimize

# Prédiction sur nouvelles données
predict: check-prepared
	@echo "$(GREEN) Prédiction sur nouvelles données...$(NC)"
	@if [ -z "$(PREDICT_DATA)" ]; then \
		echo "$(YELLOW)  Spécifiez le fichier de données avec PREDICT_DATA=chemin/vers/fichier.csv"; \
		echo "Exemple: make predict PREDICT_DATA=new_data.csv"; \
		exit 1; \
	fi
	python main.py --mode predict --input_data $(PREDICT_DATA) --model_path $(MODEL_PATH)

# Nettoyage complet
clean:
	@echo "$(RED) Nettoyage complet...$(NC)"
	rm -rf results/*
	rm -rf models/*
	rm -rf logs/*
	rm -f *.pkl
	rm -f *.csv
	rm -rf __pycache__
	@echo "$(GREEN) Nettoyage terminé$(NC)"

# Nettoyage léger (garde les modèles)
clean-light:
	@echo "$(YELLOW) Nettoyage léger...$(NC)"
	rm -rf results/*
	rm -rf logs/*
	rm -f *.csv
	@echo "$(GREEN) Nettoyage léger terminé$(NC)"

# Liste les fichiers préparés
list-prepared:
	@echo "$(BLUE) Fichiers dans results/:$(NC)"
	@ls -la results/ 2>/dev/null || echo "$(YELLOW)Le dossier results/ est vide$(NC)"

# Liste les modèles
list-models:
	@echo "$(BLUE) Fichiers dans models/:$(NC)"
	@ls -la models/ 2>/dev/null || echo "$(YELLOW)Le dossier models/ est vide$(NC)"

# Informations sur le projet
info:
	@echo "$(BLUE) INFORMATIONS DU PROJET$(NC)"
	@echo "Structure:"
	@echo "  data/     - $(YELLOW)Données brutes (loan_data.csv à placer ici)$(NC)"
	@echo "  results/  - $(GREEN)Données préparées (généré automatiquement)$(NC)"
	@echo "  models/   - $(GREEN)Modèles entraînés (généré automatiquement)$(NC)"
	@echo "  logs/     - $(GREEN)Fichiers de log (généré automatiquement)$(NC)"
	@echo ""
	@echo "Fichiers préparés:"
	@make list-prepared
	@echo ""
	@echo "Modèles sauvegardés:"
	@make list-models
	@echo ""
	@echo "Fichier source:"
	@if [ -f "$(DATA_FILE)" ]; then \
		echo "$(GREEN) Présent: $(DATA_FILE)$(NC)"; \
	else \
		echo "$(YELLOW) Absent: $(DATA_FILE)$(NC)"; \
	fi


# Aide
help:
	@echo "$(GREEN) LOAN RISK PREDICTION MODEL - MAKE COMMANDS$(NC)"
	@echo ""
	@echo " Commandes principales:"
	@echo "  $(GREEN)make prepare$(NC)      - Préparer les données (vide results/ et crée nouveau fichier)"
	@echo "  $(GREEN)make train$(NC)        - Entraînement (utilise le fichier dans results/)"
	@echo "  $(GREEN)make evaluate$(NC)     - Évaluation"
	@echo "  $(GREEN)make full$(NC)         - Pipeline complète (prepare + train)"
	@echo "  $(GREEN)make predict$(NC)      - Prédiction (ex: make predict PREDICT_DATA=new_data.csv)"
	@echo ""
	@echo " Vérifications:"
	@echo "  $(GREEN)make check-source$(NC) - Vérifier le fichier source"
	@echo "  $(GREEN)make check-prepared$(NC)- Vérifier les données préparées"
	@echo "  $(GREEN)make info$(NC)         - Informations complètes du projet"
	@echo ""
	@echo " Gestion des fichiers:"
	@echo "  $(GREEN)make list-prepared$(NC)- Lister les fichiers dans results/"
	@echo "  $(GREEN)make list-models$(NC)  - Lister les modèles dans models/"
	@echo ""
	@echo " Nettoyage:"
	@echo "  $(GREEN)make clean$(NC)        - Nettoyage complet"
	@echo "  $(GREEN)make clean-light$(NC)  - Nettoyage léger (garde les modèles)"
	@echo ""
	@echo " Workflow recommandé:"
	@echo "  1. $(GREEN)make setup$(NC)                     # Créer la structure"
	@echo "  2. $(YELLOW)Placer loan_data.csv dans data/$(NC)   # Votre fichier source"
	@echo "  3. $(GREEN)make prepare$(NC)                   # Préparer les données"
	@echo "  4. $(GREEN)make train$(NC)                     # Entraîner le modèle"
	@echo "  5. $(GREEN)make evaluate$(NC)                  # Évaluer les performances"