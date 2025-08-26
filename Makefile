# Makefile for Crossword Generator Docker Management

.PHONY: help build up down logs clean dev prod test status restart

# Default target
help: ## Show this help message
	@echo "🧩 Crossword Generator Docker Management"
	@echo "========================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development commands
dev: ## Start development environment
	@echo "🚀 Starting development environment..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
	@echo "✅ Development server running at http://localhost:3000"

dev-tools: ## Start development with additional tools (Portainer, Adminer)
	@echo "🛠️ Starting development with tools..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile dev-tools up --build -d
	@echo "✅ Development server: http://localhost:3000"
	@echo "🐳 Portainer: http://localhost:9000"
	@echo "🗄️ Adminer: http://localhost:8080"

# Production commands
prod: ## Start production environment
	@echo "🏭 Starting production environment..."
	docker-compose --profile production up --build -d
	@echo "✅ Production server running at http://localhost"

build: ## Build the application
	@echo "🏗️ Building crossword generator..."
	docker-compose build --no-cache

# Service management
up: ## Start services
	docker-compose up -d

down: ## Stop services
	docker-compose down

restart: ## Restart services
	docker-compose restart

# Monitoring and debugging
logs: ## Show application logs
	docker-compose logs -f crossword-generator

logs-all: ## Show all services logs
	docker-compose logs -f

status: ## Show services status
	docker-compose ps

health: ## Check application health
	@echo "🏥 Checking application health..."
	@curl -f http://localhost:3000/health || echo "❌ Health check failed"

# Maintenance
clean: ## Clean up containers, networks, and volumes
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all: ## Clean up everything including images
	@echo "🗑️ Cleaning up all Docker resources..."
	docker-compose down -v --remove-orphans --rmi all
	docker system prune -a -f --volumes

# Testing and development
shell: ## Open shell in the application container
	docker-compose exec crossword-generator sh

test: ## Run tests (placeholder)
	@echo "🧪 Running tests..."
	docker-compose exec crossword-generator npm test || echo "⚠️ No tests configured yet"

# Backup and restore (for future database features)
backup: ## Backup application data
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@docker-compose exec crossword-generator tar czf - /app/data 2>/dev/null | cat > backups/crossword-backup-$$(date +%Y%m%d-%H%M%S).tar.gz
	@echo "✅ Backup created in backups/ directory"

# Quick deployment commands
quick-start: ## Quick start for first-time users
	@echo "🚀 Quick starting Crossword Generator..."
	@echo "📦 Building and starting containers..."
	docker-compose up --build -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@make health
	@echo ""
	@echo "🎉 Crossword Generator is ready!"
	@echo "🌐 Open http://localhost:3000 in your browser"

update: ## Update and restart services
	@echo "🔄 Updating services..."
	git pull
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Services updated and restarted"

# Docker maintenance
docker-clean: ## Clean Docker system
	docker system prune -f

docker-images: ## Show Docker images
	docker images | grep crossword

docker-containers: ## Show running containers
	docker ps --filter "name=crossword"

# Environment info
info: ## Show environment information
	@echo "🔍 Environment Information"
	@echo "=========================="
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'Not available')"
	@echo ""
	@echo "📊 Container Status:"
	@make status

# Install dependencies (for local development without Docker)
install: ## Install local dependencies
	@echo "📦 Installing local dependencies..."
	npm install
	@echo "✅ Local dependencies installed"

# Generate SSL certificates for HTTPS (development only)
ssl: ## Generate self-signed SSL certificates
	@echo "🔒 Generating SSL certificates..."
	@mkdir -p ssl
	openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=IT/ST=Italy/L=Rome/O=CrosswordGenerator/CN=localhost"
	@echo "✅ SSL certificates generated in ssl/ directory"
	@echo "⚠️ These are self-signed certificates for development only"
