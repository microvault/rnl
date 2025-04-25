IMAGE_NAME   = rnl-docker-cuda
TAG          = latest
IMAGE        = $(IMAGE_NAME):$(TAG)
VERSION      = 1.1

MODE              ?= learn
MAX_TIMESTEP_GLOBAL ?= 20000
SEED              ?= 1
HIDDEN_SIZE       ?= 60,40
ACTIVATION        ?= ReLU
BATCH_SIZE        ?= 8
NUM_ENVS          ?= 16
DEVICE            ?= cpu
LEARN_STEP        ?= 512
CHECKPOINT        ?= 10
CHECKPOINT_PATH   ?= ppo_policy_network_medium
LR                ?= 0.0003
GAE_LAMBDA        ?= 0.95
ENT_COEF          ?= 0.05
VF_COEF           ?= 0.5
MAX_GRAD_NORM     ?= 0.5
UPDATE_EPOCHS     ?= 10
NAME              ?= rnl-test-v1
SCALAR            ?= 1
CONTROL           ?= False
USE_WANDB         ?= True
USE_AGENTS        ?= False
PRETRAINED        ?= None
VERBOSE           ?= True
MAP_NAME          ?= map
ENV_TYPE 		  ?= turn
OBSTACLE_PERCENTAGE ?= 20.0
MAP_SIZE          ?= 3.5
POLICY_TYPE       ?= PPO
TYPE_MODEL        ?= attention

TRAIN_ARGS = \
	$(MODE) \
	--agent $(USE_AGENTS) \
	--max_timestep_global $(MAX_TIMESTEP_GLOBAL) \
	--seed $(SEED) \
	--hidden_size $(HIDDEN_SIZE) \
	--activation $(ACTIVATION) \
	--batch_size $(BATCH_SIZE) \
	--num_envs $(NUM_ENVS) \
	--device $(DEVICE) \
	--checkpoint $(CHECKPOINT) \
	--checkpoint_path $(CHECKPOINT_PATH) \
	--lr $(LR) \
	--learn_step $(LEARN_STEP) \
	--gae_lambda $(GAE_LAMBDA) \
	--ent_coef $(ENT_COEF) \
	--vf_coef $(VF_COEF) \
	--max_grad_norm $(MAX_GRAD_NORM) \
	--update_epochs $(UPDATE_EPOCHS) \
	--name $(NAME) \
	--controller False \
	--pretrained $(PRETRAINED) \
	--verbose $(VERBOSE) \
	--debug True \
	--scalar $(SCALAR) \
	--use_wandb $(USE_WANDB) \
	--env_type $(ENV_TYPE) \
 	--obstacle_percentage $(OBSTACLE_PERCENTAGE) \
  	--map_size $(MAP_SIZE) \
   	--policy_type $(POLICY_TYPE) \
    --type_model $(TYPE_MODEL)

DOCKER_RUN_COMMON = -e WANDB_API_KEY=$(WANDB_API_KEY) \
	-e GEMINI_API_KEY=$(GEMINI_API_KEY) \
	--gpus all \
	--network host \
	--privileged \
	--memory=16g \
	-it \
	-v $(PWD):/workdir \
	$(IMAGE_NAME)

define PRINT_CONFIG
	@echo "MODE=$(MODE)"
	@echo "MAX_TIMESTEP_GLOBAL=$(MAX_TIMESTEP_GLOBAL)"
	@echo "SEED=$(SEED)"
	@echo "HIDDEN_SIZE=$(HIDDEN_SIZE)"
	@echo "ACTIVATION=$(ACTIVATION)"
	@echo "BATCH_SIZE=$(BATCH_SIZE)"
	@echo "NUM_ENVS=$(NUM_ENVS)"
	@echo "DEVICE=$(DEVICE)"
	@echo "CHECKPOINT=$(CHECKPOINT)"
	@echo "LR=$(LR)"
	@echo "GAE_LAMBDA=$(GAE_LAMBDA)"
	@echo "ENT_COEF=$(ENT_COEF)"
	@echo "VF_COEF=$(VF_COEF)"
	@echo "MAX_GRAD_NORM=$(MAX_GRAD_NORM)"
	@echo "UPDATE_EPOCHS=$(UPDATE_EPOCHS)"
	@echo "NAME=$(NAME)"
	@echo "WANDB_API_KEY=$(WANDB_API_KEY)"
	@echo "GEMINI_API_KEY=$(GEMINI_API_KEY)"
	@echo "USE_WANDB=$(USE_WANDB)"
	@echo "SCALAR=$(SCALAR)"
	@echo "CONTROL=$(CONTROL)"
	@echo "USE_AGENTS=$(USE_AGENTS)"
	@echo "PRETRAINED=$(PRETRAINED)"
	@echo "VERBOSE=$(VERBOSE)"
	@echo "ENV_TYPE=$(ENV_TYPE)"
	@echo "OBSTACLE_PERCENTAGE=$(OBSTACLE_PERCENTAGE)"
	@echo "MAP_SIZE=$(MAP_SIZE)"
	@echo "POLICY_TYPE=$(POLICY_TYPE)"
	@echo "TYPE_MODEL=$(TYPE_MODEL)"
	@echo
endef

.PHONY: sim
sim:
	@uv run python -m main sim \
	    --controller $(CONTROL) \
    	--scalar $(SCALAR) \
    	--env_type $(ENV_TYPE) \
		--debug True

.PHONY: learn
learn:
	@uv run python -m main $(TRAIN_ARGS)

.PHONY: probe
probe:
	@uv run python -m main run \
		--num_envs $(NUM_ENVS) \
		--device $(DEVICE) \
		--max_timestep_global $(MAX_TIMESTEP_GLOBAL) \
		--controller False \
		--seed $(SEED) \
		--debug True \
		--scalar $(SCALAR)

.PHONY: probe-spawn
probe-spawn:
	@uv run python rnl/training/probes.py

.PHONY: test_without_coverage
test_without_coverage:
	@uv run pytest -s -x -vv -p no:warnings

.PHONY: gazebo
gazebo:
	@uv run python rnl/engine/generate.py --folder ./data/$(MAP_NAME) --name $(MAP_NAME)

.PHONY: test
test:
	@uv run pytest -s -x --cov=rnl -vv -p no:warnings

.PHONY: plot_test
plot_test:
	@uv run python tests/test_reward.py

.PHONY: post_test
post_test:
	@uv run coverage html

.PHONY: publish
publish:
	@uv publish --build -u __token__ -p $(RNL_PYPI_TOKEN)

.PHONY: install_with_dev
install_with_dev:
	@uv sync

.PHONY: install
install:
	@uv sync --no-group dev

.PHONY: build
build:
	@docker build -f docker/Dockerfile -t $(IMAGE_NAME) .

.PHONY: visualize
visualize:
	@uv run rnl/training/visualize.py --save output_map.rrd

.PHONY: train
train:
	@echo
	$(PRINT_CONFIG)
	@docker run -d $(DOCKER_RUN_COMMON) $(TRAIN_ARGS)

.PHONY: train-without
train-without:
	@echo
	$(PRINT_CONFIG)
	@docker run $(DOCKER_RUN_COMMON) $(TRAIN_ARGS)

.PHONY: clean
clean:
	@sudo docker image prune -f

.PHONY: push
push:
	@docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
	@docker push $(IMAGE_NAME):latest

UV     = uv
BLACK  = $(UV) run black
ISORT  = $(UV) run isort
RUFF   = $(UV) run ruff
SRC_DIR= ./

.PHONY: lint
lint:
	$(ISORT) $(SRC_DIR)
	$(BLACK) $(SRC_DIR)
	$(RUFF) check $(SRC_DIR)
