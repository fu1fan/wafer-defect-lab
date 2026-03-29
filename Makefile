PYTHON := python3
CONDA_RUN := conda run -n torch
PROJECT := wafer-defect-lab

.PHONY: help install data preprocess train eval cam clean smoke-test

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make data         Download and build interim datasets"
	@echo "  make preprocess   Build processed (224x224) datasets"
	@echo "  make train        Train ResNet baseline (binary)"
	@echo "  make eval         Evaluate best checkpoint"
	@echo "  make cam          Generate GradCAM heatmaps"
	@echo "  make smoke-test   Quick 1-epoch pipeline test"
	@echo "  make clean        Remove temporary outputs"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(CONDA_RUN) $(PYTHON) scripts/prepare_data.py

preprocess:
	$(CONDA_RUN) $(PYTHON) scripts/process_data.py

train:
	$(CONDA_RUN) $(PYTHON) scripts/train_classifier.py

eval:
	$(CONDA_RUN) $(PYTHON) scripts/eval_classifier.py \
		--checkpoint outputs/wm811k_resnet_baseline/best.pt

cam:
	$(CONDA_RUN) $(PYTHON) scripts/visualize_cam.py \
		--checkpoint outputs/wm811k_resnet_baseline/best.pt

smoke-test:
	$(CONDA_RUN) $(PYTHON) scripts/train_classifier.py --smoke-test --output-dir outputs/smoke_test

clean:
	rm -rf outputs/smoke_test
