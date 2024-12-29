IMAGE_NAME = ninim/rnl-docker-cuda
TAG = latest
IMAGE = $(IMAGE_NAME):$(TAG)
VERSION = 1.1

.PHONY: run
run:
	@poetry run python -m train run

.PHONY: train
train:
	@poetry run python -m train train

.PHONY: test_without_coverage
test_without_coverage:
	@poetry run pytest -s -x -vv -p no:warnings

.PHONY: test
test:
	@poetry run pytest -s -x --cov=rnl -vv -p no:warnings

.PHONY: post_test
post_test:
	@poetry run coverage html

.PHONY: publish
publish:
	@poetry publish --build -u __token__ -p $RNL_PYPI_TOKEN


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
		--build-arg BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04 \
		-t ninim/rnl-docker-cuda:$(VERSION) .

.PHONY: build-nocuda
build-nocuda:
	@sudo docker build \
		--platform=linux/arm64 \
		--build-arg BASE_IMAGE=ubuntu:22.04 \
		-t rnl-docker-nocuda .

.PHONY: start
start:
	@sudo docker run -e WANDB_API_KEY=$(WANDB_API_KEY) --platform=linux/arm64 -it --net=host -v $(PWD)/rnl:/workdir/rnl rnl-docker-nocuda

.PHONY: start-cuda
start-cuda:
	@sudo docker run --platform=linux/amd64 -it --net=host -v $(PWD):/workdir rnl-docker-cuda

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
