IMAGE_NAME = rnl-docker-cuda
TAG = latest
IMAGE = $(IMAGE_NAME):$(TAG)
VERSION = 1.1

MODE ?= learn
MAX_TIMESTEP_GLOBAL ?= 3_000_00
SEED ?= 1
HIDDEN_SIZE ?= 40,30
ACTIVATION ?= ReLU
BATCH_SIZE ?= 1024
NUM_ENVS ?= 1
DEVICE ?= mps
CHECKPOINT ?= 13_02_2025
LR ?= 0.0003
GAE_LAMBDA ?= 0.95
ACTION_STD_INIT ?= 0.6
CLIP_COEF ?= 0.2
ENT_COEF ?= 0.05
VF_COEF ?= 0.5
MAX_GRAD_NORM ?= 0.5
UPDATE_EPOCHS ?= 10
NAME ?= rnl-v1
TYPE_REWARD ?= time

.PHONY: sim
sim:
	@poetry run python -m main sim


.PHONY: learn
learn:
	@poetry run python -m main $(MODE) \
    	--max_timestep_global $(MAX_TIMESTEP_GLOBAL) \
    	--seed $(SEED) \
    	--hidden_size $(HIDDEN_SIZE) \
    	--activation $(ACTIVATION) \
    	--batch_size $(BATCH_SIZE) \
    	--num_envs $(NUM_ENVS) \
    	--device $(DEVICE) \
    	--checkpoint $(CHECKPOINT) \
    	--lr $(LR) \
    	--gae_lambda $(GAE_LAMBDA) \
    	--action_std_init $(ACTION_STD_INIT) \
    	--clip_coef $(CLIP_COEF) \
    	--ent_coef $(ENT_COEF) \
    	--vf_coef $(VF_COEF) \
    	--max_grad_norm $(MAX_GRAD_NORM) \
    	--update_epochs $(UPDATE_EPOCHS) \
    	--name $(NAME) \
    	--type_reward $(TYPE_REWARD)

.PHONY: probe
probe:
	@poetry run python -m main run \
    	--num_envs $(NUM_ENVS) \
    	--device $(DEVICE) \
    	--type_reward $(TYPE_REWARD) \
     	--type_reward $(TYPE_REWARD)


.PHONY: test_without_coverage
test_without_coverage:
	@poetry run pytest -s -x -vv -p no:warnings


.PHONY: test
test:
	@poetry run pytest -s -x --cov=rnl -vv -p no:warnings


.PHONY: plot_test
plot_test:
	@poetry run python tests/test_reward.py


.PHONY: post_test
post_test:
	@poetry run coverage html


.PHONY: publish
publish:
	@poetry publish --build -u __token__ -p $(RNL_PYPI_TOKEN)


.PHONY: install_with_dev
install_with_dev:
		@poetry install


.PHONY: install
install:
	@poetry install --without dev


.PHONY: build
build:
	@docker build -f docker/Dockerfile -t rnl-docker-cuda .


.PHONY: train
train:
	@echo
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
	@echo "ACTION_STD_INIT=$(ACTION_STD_INIT)"
	@echo "CLIP_COEF=$(CLIP_COEF)"
	@echo "ENT_COEF=$(ENT_COEF)"
	@echo "VF_COEF=$(VF_COEF)"
	@echo "MAX_GRAD_NORM=$(MAX_GRAD_NORM)"
	@echo "UPDATE_EPOCHS=$(UPDATE_EPOCHS)"
	@echo "NAME=$(NAME)"
	@echo "WANDB_API_KEY=$(WANDB_API_KEY)"
	@echo "TYPE_REWARD=$(TYPE_REWARD)"
	@echo
	@docker run -d \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	--gpus all \
	--network host \
	--privileged \
	--memory=16g \
	-it \
	-v $(PWD):/workdir \
	rnl-docker-cuda \
	--MODE $(MODE) \
	--max_timestep_global $(MAX_TIMESTEP_GLOBAL) \
	--seed $(SEED) \
	--buffer_size $(BUFFER_SIZE) \
	--hidden_size $(HIDDEN_SIZE) \
	--activation $(ACTIVATION) \
	--batch_size $(BATCH_SIZE) \
	--num_envs $(NUM_ENVS) \
	--device $(DEVICE) \
	--checkpoint $(CHECKPOINT) \
	--lr $(LR) \
	--gae_lambda $(GAE_LAMBDA) \
	--action_std_init $(ACTION_STD_INIT) \
	--clip_coef $(CLIP_COEF) \
	--ent_coef $(ENT_COEF) \
	--vf_coef $(VF_COEF) \
	--max_grad_norm $(MAX_GRAD_NORM) \
	--update_epochs $(UPDATE_EPOCHS) \
	--name $(NAME) \
	--type_reward $(TYPE_REWARD)

.PHONY: clean
clean:
	@sudo docker image prune -f

.PHONY: push
push:
	@docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
	@docker push $(IMAGE_NAME):latest

POETRY=poetry
BLACK=$(POETRY) run black
ISORT=$(POETRY) run isort
RUFF=$(POETRY) run ruff

SRC_DIR=./

.PHONY: lint
lint:
	$(ISORT) $(SRC_DIR)
	$(BLACK) $(SRC_DIR)
	$(RUFF) check $(SRC_DIR)
