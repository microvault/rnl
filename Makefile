# === Start Traininig === #
.PHONY: sim
sim:
	@echo "Starting Training ..."
	@python microvault/environment/continuous.py

# === Generate World === #
.PHONY: gen
gen:
	@echo "Generate World ..."
	@python microvault/environment/generate.py

# === Pull Github === #
.PHONY: up
up:
	@echo "Git Pull ..."
	@git pull
