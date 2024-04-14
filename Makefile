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


# Configurações
PYTHON := python
BLACK := black
ISORT := isort
RUFF := ruff

# Diretórios
SRC_DIR := ./microvault

# Encontre todos os arquivos .py no diretório src
PYTHON_FILES := $(shell find $(SRC_DIR) -type f -name "*.py")

# Alvos
.PHONY: all format lint

all: format lint

# Formatar todos os arquivos .py usando black e isort
format:
	@echo "Running Black and isort..."
	$(BLACK) $(PYTHON_FILES)
	$(ISORT) $(PYTHON_FILES)

# Executar o linter ruff em todos os arquivos .py
lint:
	@echo "Running ruff..."
	$(RUFF) $(PYTHON_FILES)
