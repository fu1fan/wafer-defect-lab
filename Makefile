PYTHON := python3
CONDA_RUN := conda run -n torch
PROJECT := wafer-defect-lab
IMAGE ?= ghcr.io/$(shell git config --get remote.origin.url | sed -E 's#(git@|https://)github.com[:/]##; s#\.git$$##' | tr '[:upper:]' '[:lower:]')
OUTPUT_ROOT ?= $(or $(WAFERLAB_OUTPUT_ROOT),outputs)

.PHONY: help install data preprocess train eval cam clean smoke-test docker-build docker-train-local remote-train fetch-outputs

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make data         Download and build interim datasets"
	@echo "  make preprocess   Build processed (224x224) datasets"
	@echo "  make train        Train ResNet baseline locally (CUDA first, CPU fallback)"
	@echo "  make eval         Evaluate best checkpoint"
	@echo "  make cam          Generate GradCAM heatmaps"
	@echo "  make smoke-test   Quick 1-epoch pipeline test"
	@echo "  make docker-build Build the training image"
	@echo "  make docker-train-local  Run training through Docker on this machine"
	@echo "  make remote-train SSH to a remote GPU host and start training"
	@echo "  make fetch-outputs Pull remote outputs back to local outputs/"
	@echo "  make clean        Remove temporary outputs"

install:
	$(PYTHON) -m pip install -r requirements.txt

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

remote-train:
	@test -n "$(HOST)" || (echo "Usage: make remote-train HOST=user@server [IMAGE=ghcr.io/org/repo:tag] [ARGS='--smoke-test']" && exit 1)
	bash scripts/remote_train.sh "$(HOST)" "$(or $(IMAGE),$(PROJECT):local)" $(ARGS)

fetch-outputs:
	@test -n "$(HOST)" || (echo "Usage: make fetch-outputs HOST=user@server [SUBDIR=wm811k_resnet_baseline]" && exit 1)
	bash scripts/fetch_outputs.sh "$(HOST)" "$(SUBDIR)"

clean:
	rm -rf outputs/smoke_test
