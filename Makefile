PYTHON := python3
PROJECT := wafer-defect-lab

.PHONY: help install data preprocess train infer eval clean

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make data         Download or check datasets"
	@echo "  make preprocess   Prepare dataset for anomaly detection"
	@echo "  make train        Train anomaly baseline"
	@echo "  make infer        Run inference and generate heatmaps"
	@echo "  make eval         Evaluate results"
	@echo "  make clean        Remove temporary outputs"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/prepare_data.py
