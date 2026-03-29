PYTHON := python3
CONDA_RUN := conda run -n torch
PROJECT := wafer-defect-lab
IMAGE ?= ghcr.io/$(shell git config --get remote.origin.url | sed -E 's#(git@|https://)github.com[:/]##; s#\.git$$##' | tr '[:upper:]' '[:lower:]')
OUTPUT_ROOT ?= $(or $(WAFERLAB_OUTPUT_ROOT),outputs)

.PHONY: help install install-gpu data preprocess train eval cam clean smoke-test docker-build docker-train-local remote-deploy remote-train remote-fetch-weights

help:
	@echo "Available commands:"
	@echo "  make install      Install base Python dependencies"
	@echo "  make install-gpu  Install base deps + PyTorch 2.9.1 CUDA 12.8 wheels"
	@echo "  make data         Download and build interim datasets"
	@echo "  make preprocess   Build processed (224x224) datasets"
	@echo "  make train        Train ResNet baseline locally (CUDA first, CPU fallback)"
	@echo "  make eval         Evaluate best checkpoint"
	@echo "  make cam          Generate GradCAM heatmaps"
	@echo "  make smoke-test   Quick 1-epoch pipeline test"
	@echo "  make docker-build Build the training image"
	@echo "  make docker-train-local  Run training through Docker on this machine"
	@echo "  make remote-deploy Deploy code/environment and optionally prepare data on a remote shell host"
	@echo "  make remote-train Run remote training from a selected config and auto-download reports"
	@echo "  make remote-fetch-weights Download checkpoint files for the latest or selected remote run"
	@echo "  make clean        Remove temporary outputs"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-gpu:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-cu128.txt

data:
	$(CONDA_RUN) $(PYTHON) scripts/prepare_data.py

preprocess:
	$(CONDA_RUN) $(PYTHON) scripts/process_data.py

train:
	$(CONDA_RUN) bash scripts/run_train.sh

eval:
	$(CONDA_RUN) $(PYTHON) scripts/eval_classifier.py \
		--checkpoint $(OUTPUT_ROOT)/wm811k_resnet_baseline/best.pt

cam:
	$(CONDA_RUN) $(PYTHON) scripts/visualize_cam.py \
		--checkpoint $(OUTPUT_ROOT)/wm811k_resnet_baseline/best.pt

smoke-test:
	$(CONDA_RUN) bash scripts/run_train.sh --smoke-test --output-dir $(OUTPUT_ROOT)/smoke_test

docker-build:
	docker build -t $(PROJECT):local .

docker-train-local:
	IMAGE=$(PROJECT):local bash scripts/docker_train_local.sh

remote-deploy:
	$(PYTHON) scripts-remote/deploy.py \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(REMOTE_PROJECT_ROOT),--project-root $(REMOTE_PROJECT_ROOT)) \
		$(if $(REMOTE_DATA_ROOT),--data-root $(REMOTE_DATA_ROOT)) \
		$(if $(REMOTE_OUTPUT_ROOT),--output-root $(REMOTE_OUTPUT_ROOT)) \
		$(if $(REMOTE_PYTHON_BIN),--python-bin $(REMOTE_PYTHON_BIN)) \
		$(if $(REMOTE_BOOTSTRAP_CMD),--bootstrap-cmd $(REMOTE_BOOTSTRAP_CMD)) \
		$(if $(LOCAL_REPORT_ROOT),--local-report-root $(LOCAL_REPORT_ROOT)) \
		$(if $(PREPARE_DATA),--prepare-data) \
		$(if $(DATASET),--dataset $(DATASET)) \
		$(foreach subset,$(PROCESS_SUBSETS),--process-subset $(subset)) \
		$(if $(FORCE_DATA),--force-data) \
		$(if $(SKIP_SYNC),--skip-sync) \
		$(if $(SKIP_BOOTSTRAP),--skip-bootstrap)

remote-train:
	$(PYTHON) scripts-remote/train.py \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(REMOTE_PROJECT_ROOT),--project-root $(REMOTE_PROJECT_ROOT)) \
		$(if $(REMOTE_DATA_ROOT),--data-root $(REMOTE_DATA_ROOT)) \
		$(if $(REMOTE_OUTPUT_ROOT),--output-root $(REMOTE_OUTPUT_ROOT)) \
		$(if $(REMOTE_PYTHON_BIN),--python-bin $(REMOTE_PYTHON_BIN)) \
		$(if $(LOCAL_REPORT_ROOT),--local-report-root $(LOCAL_REPORT_ROOT)) \
		$(if $(CONFIG),--config $(CONFIG)) \
		$(if $(RUN_ID),--run-id $(RUN_ID)) \
		$(if $(NO_FETCH_REPORTS),--no-fetch-reports) \
		$(ARGS)

remote-fetch-weights:
	$(PYTHON) scripts-remote/fetch_weights.py \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(RUN_ID),--run-id $(RUN_ID)) \
		$(if $(PATTERN),--pattern $(PATTERN)) \
		$(if $(LOCAL_REPORT_ROOT),--local-report-root $(LOCAL_REPORT_ROOT))

clean:
	rm -rf outputs/smoke_test
