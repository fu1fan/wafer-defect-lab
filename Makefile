PYTHON := python3
CONDA_RUN := conda run -n torch
PROJECT := wafer-defect-lab
OUTPUT_ROOT ?= $(or $(WAFERLAB_OUTPUT_ROOT),outputs)

.PHONY: help install data preprocess train eval cam clean smoke-test remote-deploy remote-run remote-prepare-data remote-process-data remote-train remote-fetch-all-output

help:
	@echo "Available commands:"
	@echo "  make install      Install base project dependencies (torch handled separately)"
	@echo "  make data         Download and build interim datasets"
	@echo "  make preprocess   Build processed (224x224) datasets"
	@echo "  make train        Train ResNet baseline locally (CUDA first, CPU fallback)"
	@echo "  make eval         Evaluate best checkpoint"
	@echo "  make cam          Generate GradCAM heatmaps"
	@echo "  make smoke-test   Quick 1-epoch pipeline test"
	@echo "  make remote-deploy Deploy code/environment and optionally prepare data on a remote shell host"
	@echo "  make remote-run Run any scripts/ entrypoint remotely after syncing code, then mirror small outputs back"
	@echo "  make remote-prepare-data Run scripts/prepare_data.py remotely through remote_run"
	@echo "  make remote-process-data Run scripts/process_data.py remotely through remote_run"
	@echo "  make remote-train Run remote training with the generic remote-run workflow"
	@echo "  make remote-fetch-all-output Download full remote outputs, including large files"
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

remote-deploy:
	$(PYTHON) scripts-remote/deploy.py \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(REMOTE_PROJECT_ROOT),--project-root $(REMOTE_PROJECT_ROOT)) \
		$(if $(REMOTE_DATA_ROOT),--data-root $(REMOTE_DATA_ROOT)) \
		$(if $(REMOTE_OUTPUT_ROOT),--output-root $(REMOTE_OUTPUT_ROOT)) \
		$(if $(REMOTE_DEPLOYMENT_MODE),--deployment-mode $(REMOTE_DEPLOYMENT_MODE)) \
		$(if $(REMOTE_PYTHON_BIN),--python-bin $(REMOTE_PYTHON_BIN)) \
		$(if $(REMOTE_BOOTSTRAP_CMD),--bootstrap-cmd $(REMOTE_BOOTSTRAP_CMD)) \
		$(if $(LOCAL_REPORT_ROOT),--local-report-root $(LOCAL_REPORT_ROOT)) \
		$(if $(SKIP_TORCH_INSTALL),--skip-torch-install) \
		$(if $(PREPARE_DATA),--prepare-data) \
		$(if $(DATASET),--dataset $(DATASET)) \
		$(foreach subset,$(PROCESS_SUBSETS),--process-subset $(subset)) \
		$(if $(FORCE_DATA),--force-data) \
		$(if $(SKIP_SYNC),--skip-sync) \
		$(if $(SKIP_BOOTSTRAP),--skip-bootstrap)

remote-run:
	$(PYTHON) scripts-remote/remote_run.py $(SCRIPT) \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(REMOTE_PROJECT_ROOT),--project-root $(REMOTE_PROJECT_ROOT)) \
		$(if $(REMOTE_DATA_ROOT),--data-root $(REMOTE_DATA_ROOT)) \
		$(if $(REMOTE_OUTPUT_ROOT),--output-root $(REMOTE_OUTPUT_ROOT)) \
		$(if $(REMOTE_PYTHON_BIN),--python-bin $(REMOTE_PYTHON_BIN)) \
		$(if $(RUN_ID),--run-id $(RUN_ID)) \
		$(if $(LOCAL_OUTPUT_ROOT),--local-output-root $(LOCAL_OUTPUT_ROOT)) \
		$(if $(SYNC_MAX_SIZE),--sync-max-size $(SYNC_MAX_SIZE)) \
		$(if $(NO_SYNC_OUTPUTS),--no-sync-outputs) \
		$(ARGS)

remote-prepare-data:
	$(PYTHON) scripts-remote/remote_run.py scripts/prepare_data.py \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(REMOTE_PROJECT_ROOT),--project-root $(REMOTE_PROJECT_ROOT)) \
		$(if $(REMOTE_DATA_ROOT),--data-root $(REMOTE_DATA_ROOT)) \
		$(if $(REMOTE_OUTPUT_ROOT),--output-root $(REMOTE_OUTPUT_ROOT)) \
		$(if $(REMOTE_PYTHON_BIN),--python-bin $(REMOTE_PYTHON_BIN)) \
		$(if $(LOCAL_OUTPUT_ROOT),--local-output-root $(LOCAL_OUTPUT_ROOT)) \
		$(if $(SYNC_MAX_SIZE),--sync-max-size $(SYNC_MAX_SIZE)) \
		$(if $(NO_SYNC_OUTPUTS),--no-sync-outputs) \
		$(if $(DATASET),--dataset $(DATASET)) \
		$(if $(FORCE_DATA),--force) \
		-- $(ARGS)

remote-process-data:
	$(PYTHON) scripts-remote/remote_run.py scripts/process_data.py \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(REMOTE_PROJECT_ROOT),--project-root $(REMOTE_PROJECT_ROOT)) \
		$(if $(REMOTE_DATA_ROOT),--data-root $(REMOTE_DATA_ROOT)) \
		$(if $(REMOTE_OUTPUT_ROOT),--output-root $(REMOTE_OUTPUT_ROOT)) \
		$(if $(REMOTE_PYTHON_BIN),--python-bin $(REMOTE_PYTHON_BIN)) \
		$(if $(LOCAL_OUTPUT_ROOT),--local-output-root $(LOCAL_OUTPUT_ROOT)) \
		$(if $(SYNC_MAX_SIZE),--sync-max-size $(SYNC_MAX_SIZE)) \
		$(if $(NO_SYNC_OUTPUTS),--no-sync-outputs) \
		$(foreach subset,$(PROCESS_SUBSETS),--subset $(subset)) \
		$(if $(FORCE_DATA),--force) \
		-- $(ARGS)

remote-train:
	$(PYTHON) scripts-remote/remote_run.py scripts/train_classifier.py \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(REMOTE_PROJECT_ROOT),--project-root $(REMOTE_PROJECT_ROOT)) \
		$(if $(REMOTE_DATA_ROOT),--data-root $(REMOTE_DATA_ROOT)) \
		$(if $(REMOTE_OUTPUT_ROOT),--output-root $(REMOTE_OUTPUT_ROOT)) \
		$(if $(REMOTE_PYTHON_BIN),--python-bin $(REMOTE_PYTHON_BIN)) \
		$(if $(RUN_ID),--run-id $(RUN_ID)) \
		$(if $(LOCAL_OUTPUT_ROOT),--local-output-root $(LOCAL_OUTPUT_ROOT)) \
		$(if $(SYNC_MAX_SIZE),--sync-max-size $(SYNC_MAX_SIZE)) \
		$(if $(NO_SYNC_OUTPUTS),--no-sync-outputs) \
		$(if $(CONFIG),--config $(CONFIG)) \
		-- $(ARGS)

remote-fetch-all-output:
	$(PYTHON) scripts-remote/fetch_all_output.py \
		$(if $(HOST),--host $(HOST)) \
		$(if $(PORT),--port $(PORT)) \
		$(if $(RUN_ID),--run-id $(RUN_ID)) \
		$(if $(REMOTE_SUBDIR),--remote-subdir $(REMOTE_SUBDIR)) \
		$(if $(ALL),--all) \
		$(if $(LOCAL_OUTPUT_ROOT),--local-output-root $(LOCAL_OUTPUT_ROOT))

clean:
	rm -rf outputs/smoke_test
