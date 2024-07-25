# === Start Training  === #
.PHONY: train
train:
	@echo "Starting Training ..."
	@poetry run python -m microvault.training.train

# === Generate World === #
.PHONY: gen
gen:
	@echo "Generate World ..."
	@poetry run python -m microvault.environment.generate

.PHONY: format
format:
	@echo "Formatting with Black, iSort, and Ruff ..."
	@black .py*
	@isort .py*
	@ruff .py*
