PYTHON = python
MAIN = src/main.py
MODEL_DIR = models

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install flake8 pytest

train:
	$(PYTHON) $(MAIN)

run: train

clean:
	rm -f $(MODEL_DIR)/*.pkl

lint:
	flake8 src/ --max-line-length=120

test:
	pytest -q tests/

ci: lint test train
