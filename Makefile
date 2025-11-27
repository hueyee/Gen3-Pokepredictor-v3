.PHONY: install data scrape sequence train test clean help

# Default target
help:
	@echo "Pokemon Prediction Project"
	@echo ""
	@echo "Usage:"
	@echo "  make install    - Install dependencies"
	@echo "  make scrape     - Scrape replay data from Pokemon Showdown"
	@echo "  make sequence   - Process data and extract Pokemon sequences"
	@echo "  make train      - Train the prediction models"
	@echo "  make test       - Test and evaluate the models"
	@echo "  make data       - Run full data pipeline (scrape + sequence)"
	@echo "  make all        - Run full pipeline (data + train + test)"
	@echo "  make clean      - Remove generated files"

# Install dependencies
install:
	pip install -r requirements.txt

# Scrape data from Pokemon Showdown
scrape:
	python -m src.data.scraper

# Process data and extract sequences
sequence:
	python -m src.data.sequencer

# Full data pipeline
data: scrape sequence

# Train models
train:
	python -m src.models.train

# Test models
test:
	python -m src.models.test

# Run full pipeline
all: data train test

# Clean generated files
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/data/__pycache__
	rm -rf src/models/__pycache__
	rm -rf src/features/__pycache__
	rm -f *.png
	rm -f *.csv
