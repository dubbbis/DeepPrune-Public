# DeepPrune Docker Makefile

.PHONY: help build start stop logs restart clean

help: ## Show available commands
	@echo 'DeepPrune - Available Commands:'
	@echo ''
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  make %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Set up local Python environment (non-Docker)
	pyenv local 3.11.3
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

build: ## Build the Docker image
	docker-compose build

start: ## Start the application
	docker-compose up -d
	@echo ""
	@echo "DeepPrune is starting..."
	@echo "Access the app at: http://localhost:8501"

stop: ## Stop the application
	docker-compose down

logs: ## View application logs
	docker-compose logs -f

restart: ## Restart the application
	docker-compose restart

clean: ## Stop and remove all containers and volumes
	docker-compose down -v
	docker system prune -f

rebuild: ## Rebuild from scratch
	docker-compose down -v
	docker-compose build --no-cache
	docker-compose up -d
	@echo ""
	@echo "DeepPrune has been rebuilt and started"
	@echo "Access the app at: http://localhost:8501"
