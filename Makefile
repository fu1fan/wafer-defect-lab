PYTHON := python3
CONDA_RUN := conda run -n torch
PROJECT := wafer-defect-lab
OUTPUT_ROOT ?= $(or $(WAFERLAB_OUTPUT_ROOT),outputs)

.PHONY: help install data preprocess train eval cam clean smoke-test \
        remote-deploy remote-run remote-prepare-data remote-process-data \
        remote-train remote-fetch-all-output

# ── Shared remote flag builder ───────────────────────────────────────
# Reusable macro that expands common remote SSH/path flags.
_REMOTE_FLAGS = \
	$(if $(HOST),--host $(HOST)) \
	$(if $(PORT),--port $(PORT)) \
	$(if $(REMOTE_PROJECT_ROOT),--project-root $(REMOTE_PROJECT_ROOT)) \
	$(if $(REMOTE_DATA_ROOT),--data-root $(REMOTE_DATA_ROOT)) \
	$(if $(REMOTE_OUTPUT_ROOT),--output-root $(REMOTE_OUTPUT_ROOT)) \
	$(if $(REMOTE_PYTHON_BIN),--python-bin $(REMOTE_PYTHON_BIN))

_REMOTE_RUN_FLAGS = $(_REMOTE_FLAGS) \
	$(if $(RUN_ID),--run-id $(RUN_ID)) \
	$(if $(LOCAL_OUTPUT_ROOT),--local-output-root $(LOCAL_OUTPUT_ROOT)) \
	$(if $(SYNC_MAX_SIZE),--sync-max-size $(SYNC_MAX_SIZE)) \
	$(if $(NO_SYNC_OUTPUTS),--no-sync-outputs)

# ── Local workflows ──────────────────────────────────────────────────

help:
	@echo "Local:"
	@echo "  make install       Install project as editable package"
	@echo "  make data          Download + build interim datasets"
	@echo "  make preprocess    Build processed 224×224 datasets"
	@echo "  make train         Train ResNet baseline (CUDA first, CPU fallback)"
	@echo "  make eval          Evaluate best checkpoint"
	@echo "  make cam           Generate GradCAM heatmaps"
	@echo "  make smoke-test    Quick 1-epoch pipeline test"
	@echo ""
	@echo "Remote:"
	@echo "  make remote-deploy          Deploy env & optionally prepare data"
	@echo "  make remote-run SCRIPT=...  Run any script remotely"
	@echo "  make remote-prepare-data    Download data remotely"
	@echo "  make remote-process-data    Process data remotely"
	@echo "  make remote-train           Train remotely"
	@echo "  make remote-fetch-all-output  Download remote outputs"
	@echo ""
	@echo "  make clean         Remove temporary outputs"

install:
	pip install -e .

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

# ── Remote workflows ─────────────────────────────────────────────────

remote-deploy:
	$(PYTHON) scripts-remote/deploy.py $(_REMOTE_FLAGS) \
		$(if $(REMOTE_DEPLOYMENT_MODE),--deployment-mode $(REMOTE_DEPLOYMENT_MODE)) \
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
	$(PYTHON) scripts-remote/remote_run.py $(SCRIPT) $(_REMOTE_RUN_FLAGS) $(ARGS)

remote-prepare-data:
	$(PYTHON) scripts-remote/remote_run.py scripts/prepare_data.py $(_REMOTE_RUN_FLAGS) \
		$(if $(DATASET),--dataset $(DATASET)) \
		$(if $(FORCE_DATA),--force) \
		-- $(ARGS)

remote-process-data:
	$(PYTHON) scripts-remote/remote_run.py scripts/process_data.py $(_REMOTE_RUN_FLAGS) \
		$(foreach subset,$(PROCESS_SUBSETS),--subset $(subset)) \
		$(if $(FORCE_DATA),--force) \
		-- $(ARGS)

remote-train:
	$(PYTHON) scripts-remote/remote_run.py scripts/train_classifier.py $(_REMOTE_RUN_FLAGS) \
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
