# === Start Traininig === #
.PHONY: sim
sim:
	@echo "Starting Training ..."
	@python microvault/environment/continuous.py

# === Generate World === #
.PHONY: gen
gen:
	@echo "Starting Training ..."
	@python microvault/environment/generate.py
