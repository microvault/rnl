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
	@[ -n "$(file)" ] || (echo "Error: 'file' variable is not set"; exit 1)
	@black $(file)
	@isort $(file)
	@ruff $(file)
