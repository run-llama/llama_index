# Makefile for LlamaIndex deployment
.PHONY: help setup install run docker-build docker-up docker-down docker-logs clean test

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "LlamaIndex Deployment Commands"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initial setup (copy .env, create directories)
	@echo "Setting up deployment environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please edit it with your API keys"; \
	else \
		echo ".env file already exists"; \
	fi
	@mkdir -p data storage
	@echo "Setup complete!"

install: ## Install Python dependencies
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
	@echo "Dependencies installed!"

install-uv: ## Install dependencies using uv (faster)
	@echo "Installing dependencies with uv..."
	@uv pip install -r requirements.txt
	@echo "Dependencies installed!"

run: ## Run the FastAPI server locally
	@echo "Starting FastAPI server..."
	@python api_server.py

run-dev: ## Run with auto-reload
	@echo "Starting development server with auto-reload..."
	@uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Build Docker image
	@echo "Building Docker image..."
	@docker build -t llamaindex-app:latest .
	@echo "Docker image built!"

docker-up: ## Start all services with Docker Compose
	@echo "Starting Docker services..."
	@docker-compose up -d
	@echo "Services started! Check status with 'make docker-ps'"

docker-up-build: ## Build and start Docker services
	@echo "Building and starting Docker services..."
	@docker-compose up -d --build
	@echo "Services started!"

docker-down: ## Stop all Docker services
	@echo "Stopping Docker services..."
	@docker-compose down
	@echo "Services stopped!"

docker-down-volumes: ## Stop Docker services and remove volumes
	@echo "Stopping services and removing volumes..."
	@docker-compose down -v
	@echo "Services stopped and volumes removed!"

docker-logs: ## Show Docker logs (all services)
	@docker-compose logs -f

docker-logs-app: ## Show Docker logs (app only)
	@docker-compose logs -f app

docker-ps: ## Show running Docker containers
	@docker-compose ps

docker-restart: ## Restart Docker services
	@echo "Restarting Docker services..."
	@docker-compose restart
	@echo "Services restarted!"

docker-restart-app: ## Restart app service only
	@echo "Restarting app service..."
	@docker-compose restart app
	@echo "App restarted!"

health: ## Check application health
	@echo "Checking application health..."
	@curl -f http://localhost:8000/health || echo "Service not responding"

test-query: ## Test query endpoint
	@echo "Testing query endpoint..."
	@curl -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"query": "What is this about?"}'

test-upload: ## Test file upload (requires TEST_FILE variable)
	@if [ -z "$(TEST_FILE)" ]; then \
		echo "Usage: make test-upload TEST_FILE=path/to/file.txt"; \
	else \
		curl -X POST http://localhost:8000/upload -F "file=@$(TEST_FILE)"; \
	fi

ingest: ## Trigger document ingestion
	@echo "Triggering document ingestion..."
	@curl -X POST http://localhost:8000/ingest \
		-H "Content-Type: application/json" \
		-d '{"rebuild": false}'

ingest-rebuild: ## Rebuild entire index
	@echo "Rebuilding index..."
	@curl -X POST http://localhost:8000/ingest \
		-H "Content-Type: application/json" \
		-d '{"rebuild": true}'

clean: ## Clean up temporary files and caches
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@echo "Cleanup complete!"

clean-storage: ## Remove index storage (keeps data files)
	@echo "Removing index storage..."
	@rm -rf storage/*
	@echo "Storage cleaned!"

clean-all: clean clean-storage ## Clean everything including storage
	@echo "Full cleanup complete!"

shell-app: ## Open shell in app container
	@docker-compose exec app sh

shell-chroma: ## Open shell in chroma container
	@docker-compose exec chroma sh

backup-storage: ## Backup index storage
	@echo "Backing up storage..."
	@tar -czf storage-backup-$$(date +%Y%m%d-%H%M%S).tar.gz storage/
	@echo "Backup created!"

backup-data: ## Backup data directory
	@echo "Backing up data..."
	@tar -czf data-backup-$$(date +%Y%m%d-%H%M%S).tar.gz data/
	@echo "Backup created!"

deploy-aws: ## Deploy to AWS (requires AWS CLI configured)
	@echo "Deploying to AWS..."
	@echo "This is a placeholder - customize for your AWS setup"
	@echo "Example: docker push to ECR, update ECS service, etc."

deploy-gcp: ## Deploy to GCP Cloud Run
	@echo "Deploying to GCP Cloud Run..."
	@gcloud builds submit --tag gcr.io/$$(gcloud config get-value project)/llamaindex-app
	@gcloud run deploy llamaindex-app \
		--image gcr.io/$$(gcloud config get-value project)/llamaindex-app \
		--platform managed \
		--region us-central1 \
		--allow-unauthenticated

docs: ## Open API documentation in browser
	@echo "Opening API docs..."
	@python -m webbrowser http://localhost:8000/docs

monitor: ## Show real-time container stats
	@docker stats

lint: ## Run linting (requires ruff)
	@echo "Running linter..."
	@ruff check api_server.py

format: ## Format code (requires black)
	@echo "Formatting code..."
	@black api_server.py

dev: setup install ## Full development setup
	@echo "Development environment ready!"
	@echo "1. Edit .env with your API keys"
	@echo "2. Add documents to ./data/"
	@echo "3. Run 'make run-dev' to start the server"

prod-check: ## Check production readiness
	@echo "Production readiness checklist:"
	@echo "================================"
	@if [ -f .env ]; then \
		echo "✓ .env file exists"; \
		if grep -q "OPENAI_API_KEY=sk-" .env; then \
			echo "✓ OPENAI_API_KEY is set"; \
		else \
			echo "✗ OPENAI_API_KEY not configured"; \
		fi; \
	else \
		echo "✗ .env file missing"; \
	fi
	@if [ -d data ] && [ "$$(ls -A data 2>/dev/null)" ]; then \
		echo "✓ Data directory has files"; \
	else \
		echo "✗ Data directory is empty"; \
	fi
	@echo ""
	@echo "Next steps:"
	@echo "1. Ensure all environment variables are set"
	@echo "2. Review security settings"
	@echo "3. Set up SSL/TLS certificates"
	@echo "4. Configure monitoring and logging"
	@echo "5. Test with 'make health'"
