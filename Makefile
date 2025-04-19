.DEFAULT_GOAL := help

UV_SYSTEM_PYTHON := 1

.PHONY: install
install:  ## Install a virtual environment
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@uv venv
	@uv pip install -r requirements.txt

#.PHONY: fmt
#fmt:  install ## Run autoformatting and linting
#	@uv pip install pre-commit
#	@uv run pre-commit install
#	@uv run pre-commit run --all-files

#.PHONY: clean
#clean:  ## Clean up caches and build artifacts
#	@git clean -X -d -f

.PHONY: test
test: install ## Run tests
	@uv pip install pytest
	@uv run  pytest tests

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
