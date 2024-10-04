.PHONY: run
run:
	@poetry run python -m train_model_base run

.PHONY: train
train:
	@poetry run python -m train_model_base train

.PHONY: test_without_coverage
test_without_coverage:
	@poetry run pytest -s -x -vv

.PHONY: format
format:
	@black .py*
	@isort .py*
	@ruff .py*

.PHONY: test
test:
	@poetry run pytest -s -x --cov=rnl -vv

.PHONY: post_test
post_test:
	@poetry run coverage html

.PHONY: publish
publish:
	@poetry publish --build -u __token__ -p $RNL_PYPI_TOKEN

.PHONY: install
install:
		@poetry install

.PHONY: build
build:
	@sudo docker build -t rnl-docker .

.PHONY: start
start:
	@echo "Starting training ..."
	@sudo docker run -it --net=host --memory=40g rnl-docker bash -c "make train"
