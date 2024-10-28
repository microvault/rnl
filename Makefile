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

.PHONY: build
build:
	@sudo docker build -t rnl-docker .

.PHONY: start
start:
	@echo "Starting training ..."
	@sudo docker run -it --net=host -v rnl:/workdir/rnl  rnl-docker

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
