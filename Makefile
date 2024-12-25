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
		-t rnl-docker-cuda .

.PHONY: build-macos
build-macos:
	@sudo docker build \
		--platform=linux/arm64 \
		--build-arg BASE_IMAGE=ubuntu:22.04 \
		-t rnl-docker-nocuda .

.PHONY: start-cuda
start-cuda:
	@sudo docker run -it --net=host -v rnl:/workdir/rnl rnl-docker-cuda

.PHONY: start
start:
	@sudo docker run --platform=linux/arm64 -it --net=host -v rnl:/workdir/rnl -v $(PWD)/train.py:/workdir/train.py rnl-docker-nocuda

clean:
	@sudo docker image prune -f

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
