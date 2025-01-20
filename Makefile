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

.PHONY: build
build:
	@docker build -t rnl-docker-cuda .

.PHONY: train
train:
	@docker run -d -e WANDB_API_KEY=$(WANDB_API_KEY) --gpus all --network host --privileged --memory=16g -it -v $(PWD):/workdir rnl-docker-cuda

.PHONY: clean
clean:
	@sudo docker image prune -f

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
