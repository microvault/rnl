.PHONY: train
train:
	@poetry run python -m microvault.training.train

.PHONY: tune
tune:
	@poetry run python -m microvault.training.finetune

.PHONY: eval
eval:
	@poetry run python -m microvault.training.eval

.PHONY: gen
gen:
	@poetry run python -m microvault.environment.generate

.PHONY: format
format:
	@black .py*
	@isort .py*
	@ruff .py*

.PHONY: test
test:
	@poetry run pytest -s -x --cov=microvault -vv

.PHONY: post_test
post_test:
	@poetry run coverage html

.PHONY: publish
publish:
	@poetry publish --build -u __token__ -p $MICROVAULT_PYPI_TOKEN

.PHONY: install
install:
		@poetry install

.PHONY: build
build:
	@sudo docker build -t microvault-docker .

.PHONY: start
start:
	@echo "Starting training ..."
	@sudo docker run -it --net=host --memory=40g microvault-docker bash -c "make train"
