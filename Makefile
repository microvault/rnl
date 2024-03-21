# === Clean docker === #
.PHONY: clean
clean:
	@echo "Closing all running docker containers ..."
	@sudo docker system prune -f

# === Stop Docker === #
.PHONY: kill
kill:
	@echo "Stopping Docker ..."
	@sudo docker stop $(shell sudo docker ps -a -q)
