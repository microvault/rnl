IMAGE_NAME = rnl-docker-cuda
TAG = latest
IMAGE = $(IMAGE_NAME):$(TAG)
VERSION = 1.1

MODE ?= learn
MAX_TIMESTEP_GLOBAL ?= 20000
SEED ?= 1
HIDDEN_SIZE ?= 20,10
ACTIVATION ?= ReLU
BATCH_SIZE ?= 8
NUM_ENVS ?= 16
DEVICE ?= cpu
LEARN_STEP ?= 512
CHECKPOINT ?= 10000
CHECKPOINT_PATH ?= 07_03_2025
LR ?= 0.0003
GAE_LAMBDA ?= 0.95
ACTION_STD_INIT ?= 0.6
CLIP_COEF ?= 0.2
ENT_COEF ?= 0.05
VF_COEF ?= 0.5
MAX_GRAD_NORM ?= 0.5
UPDATE_EPOCHS ?= 10
NAME ?= rnl-v1
SCALAR ?= 1
CONTROL ?= False

.PHONY: sim
sim:
	@uv run python -m main sim \
	    --controller $(CONTROL) \
     	--debug True \
      	--scalar $(SCALAR)

.PHONY: learn
learn:
	@uv run python -m main $(MODE) \
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
    	--action_std_init $(ACTION_STD_INIT) \
    	--clip_coef $(CLIP_COEF) \
    	--ent_coef $(ENT_COEF) \
    	--vf_coef $(VF_COEF) \
    	--max_grad_norm $(MAX_GRAD_NORM) \
    	--update_epochs $(UPDATE_EPOCHS) \
    	--name $(NAME) \
     	--controller False \
      	--debug True \
       	--scalar $(SCALAR)

.PHONY: probe
probe:
	@uv run python -m main run \
    	--num_envs $(NUM_ENVS) \
    	--device $(DEVICE) \
        --max_timestep_global $(MAX_TIMESTEP_GLOBAL) \
        --num_envs $(NUM_ENVS) \
    	--controller False \
     	--debug True \
      	--scalar $(SCALAR)


.PHONY: test_without_coverage
test_without_coverage:
	@uv run pytest -s -x -vv -p no:warnings


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
		@uv install


.PHONY: install
install:
	@uv install --without dev


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
   	--action_std_init $(ACTION_STD_INIT) \
   	--clip_coef $(CLIP_COEF) \
   	--ent_coef $(ENT_COEF) \
   	--vf_coef $(VF_COEF) \
   	--max_grad_norm $(MAX_GRAD_NORM) \
   	--update_epochs $(UPDATE_EPOCHS) \
   	--name $(NAME) \
   	--type_reward $(TYPE_REWARD) \
   	--controller False \
   	--debug False \
   	--scalar $(SCALAR)

.PHONY: clean
clean:
	@sudo docker image prune -f

.PHONY: push
push:
	@docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
	@docker push $(IMAGE_NAME):latest

UV=uv
BLACK=$(UV) run black
ISORT=$(UV) run isort
RUFF=$(UV) run ruff

SRC_DIR=./

.PHONY: lint
lint:
	$(ISORT) $(SRC_DIR)
	$(BLACK) $(SRC_DIR)
	$(RUFF) check $(SRC_DIR)
