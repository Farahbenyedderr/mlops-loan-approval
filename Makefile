# ============================
#         MAKEFILE
# ============================

PYTHON=python3
PIP=pip
MAIN=main.py
TEST=test.py
REQ=requirements.txt

# ----------------------------
# Install dependencies
# ----------------------------
install:
	$(PIP) install -r $(REQ)

# ----------------------------
# Run the training pipeline
# ----------------------------
train:
	$(PYTHON) $(MAIN) --train

# ----------------------------
# Validate a saved model
# ----------------------------
validate:
	$(PYTHON) $(MAIN) --validate

# ----------------------------
# Run the entire test suite
# ----------------------------
test:
	$(PYTHON) $(TEST)

# ----------------------------
# Format code automatically
# ----------------------------
format:
	black .
	isort .

# ----------------------------
# Lint code (optional)
# ----------------------------
lint:
	flake8 .

# ----------------------------
# Clean temporary files
# ----------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f *.pkl
	rm -f test_model.pkl
	rm -rf models/*.pkl

# ----------------------------
# Full rebuild (clean + reinstall + train)
# ----------------------------
rebuild: clean install train
