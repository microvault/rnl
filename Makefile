.PHONY: train
train:
	@poetry run python -m microvault.training.run

.PHONY: tune
tune:
	@poetry run python -m microvault.training.finetune

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
	@poetry run task test
