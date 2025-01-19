IMAGE_NAME = ninim/rnl-docker-cuda
TAG = latest
IMAGE = $(IMAGE_NAME):$(TAG)
VERSION = 1.1

.PHONY: sim
sim:
	@poetry run python -m main sim

.PHONY: learn
learn:
	@poetry run python -m main learn

.PHONY: probe
probe:
	@poetry run python -m main run

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

.PHONY: build-cuda
build-cuda:
	@sudo docker build \
		--platform=linux/amd64 \
		-t ninim/rnl-docker-cuda:$(VERSION) .

.PHONY: build-nocuda
build-nocuda:
	@sudo docker build \
		--platform=linux/arm64 \
		-t rnl-docker-nocuda .

.PHONY: train-model
train-model:
	@docker run -e WANDB_API_KEY=$(WANDB_API_KEY) --gpus all --network host --privileged --memory=16g --cpus=8 -it -v $(PWD)/rnl:/workdir/rnl ninim/rnl-docker-cuda

.PHONY: start-cuda
start-cuda:
	@sudo docker run --platform=linux/arm64 -it --net=host -v $(PWD)/rnl:/workdir rnl-docker-nocuda

.PHONY: clean
clean:
	@sudo docker image prune -f

.PHONY: run
run:
	@poetry run python rnl/benchmarks/train_with_turtlebot.py

.PHONY: run-gpu
run-gpu:
	@accelerate launch --config_file rnl/benchmarks/accelerate.yaml rnl/benchmarks/train_with_acc.py


.PHONY: push
push:
	@docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
	@docker push $(IMAGE_NAME):latest

.PHONY: server
server:
	@sh server.sh

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
