#!/bin/bash

# LlamaIndex Deployment Quick Start Script
# This script helps you get started quickly with local or Docker deployment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

# Main script
print_header "LlamaIndex Deployment Quick Start"

echo ""
echo "Choose deployment method:"
echo "1) Local deployment (Python)"
echo "2) Docker deployment"
echo "3) Docker Compose (Full stack)"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        print_header "Local Python Deployment"

        # Check Python
        if ! check_command python3; then
            print_error "Python 3 is required but not installed"
            exit 1
        fi

        # Check pip
        if ! check_command pip; then
            print_error "pip is required but not installed"
            exit 1
        fi

        # Setup .env
        if [ ! -f .env ]; then
            print_warning ".env file not found"
            cp .env.example .env
            print_success "Created .env from .env.example"
            print_warning "Please edit .env and add your OPENAI_API_KEY"
            read -p "Press enter after you've added your API key..."
        else
            print_success ".env file exists"
        fi

        # Check for API key
        if grep -q "OPENAI_API_KEY=sk-" .env; then
            print_success "OPENAI_API_KEY is configured"
        else
            print_error "OPENAI_API_KEY not found in .env"
            echo "Please edit .env and add: OPENAI_API_KEY=sk-your-key-here"
            exit 1
        fi

        # Create directories
        mkdir -p data storage
        print_success "Created data and storage directories"

        # Check for data
        if [ -z "$(ls -A data 2>/dev/null)" ]; then
            print_warning "data directory is empty"
            echo "Creating sample document..."
            echo "This is a sample document for testing LlamaIndex deployment." > data/sample.txt
            echo "LlamaIndex is a data framework for LLM applications." >> data/sample.txt
            echo "It provides tools for data ingestion, indexing, and querying." >> data/sample.txt
            print_success "Created sample.txt in data directory"
        else
            print_success "Data directory contains files"
        fi

        # Install dependencies
        print_warning "Installing Python dependencies (this may take a few minutes)..."

        # Check for uv
        if check_command uv; then
            print_success "Using uv for faster installation"
            uv pip install -r requirements.txt
        else
            pip install -r requirements.txt
        fi

        print_success "Dependencies installed"

        # Start server
        print_header "Starting Server"
        print_success "Server will be available at: http://localhost:8000"
        print_success "API docs at: http://localhost:8000/docs"
        echo ""
        print_warning "Press Ctrl+C to stop the server"
        echo ""
        sleep 2

        python api_server.py
        ;;

    2)
        print_header "Docker Deployment"

        # Check Docker
        if ! check_command docker; then
            print_error "Docker is required but not installed"
            print_warning "Install Docker from: https://docs.docker.com/get-docker/"
            exit 1
        fi

        # Setup .env
        if [ ! -f .env ]; then
            cp .env.example .env
            print_success "Created .env file"
            print_warning "Please edit .env and add your OPENAI_API_KEY"
            read -p "Press enter after you've added your API key..."
        fi

        # Create directories
        mkdir -p data storage

        # Sample data
        if [ -z "$(ls -A data 2>/dev/null)" ]; then
            echo "This is a sample document for testing LlamaIndex deployment." > data/sample.txt
            print_success "Created sample.txt in data directory"
        fi

        # Build image
        print_warning "Building Docker image..."
        docker build -t llamaindex-app:latest .
        print_success "Docker image built"

        # Run container
        print_header "Starting Container"
        docker run -d \
            --name llamaindex-app \
            -p 8000:8000 \
            --env-file .env \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/storage:/app/storage \
            llamaindex-app:latest

        print_success "Container started"
        print_success "Server available at: http://localhost:8000"
        print_success "API docs at: http://localhost:8000/docs"
        echo ""
        echo "Useful commands:"
        echo "  docker logs -f llamaindex-app     # View logs"
        echo "  docker stop llamaindex-app        # Stop container"
        echo "  docker start llamaindex-app       # Start container"
        echo "  docker rm llamaindex-app          # Remove container"
        ;;

    3)
        print_header "Docker Compose Deployment"

        # Check Docker Compose
        if ! check_command docker-compose && ! docker compose version &> /dev/null; then
            print_error "Docker Compose is required but not installed"
            exit 1
        fi

        # Determine compose command
        if docker compose version &> /dev/null; then
            COMPOSE_CMD="docker compose"
        else
            COMPOSE_CMD="docker-compose"
        fi

        # Setup .env
        if [ ! -f .env ]; then
            cp .env.example .env
            print_success "Created .env file"
            print_warning "Please edit .env and add your OPENAI_API_KEY"
            read -p "Press enter after you've added your API key..."
        fi

        # Create directories
        mkdir -p data storage

        # Sample data
        if [ -z "$(ls -A data 2>/dev/null)" ]; then
            echo "This is a sample document for testing LlamaIndex deployment." > data/sample.txt
            print_success "Created sample.txt"
        fi

        # Start services
        print_header "Starting Services"
        print_warning "Starting all services (app, chroma, postgres, redis)..."
        $COMPOSE_CMD up -d

        print_success "All services started!"
        echo ""
        print_success "Services:"
        echo "  - API Server:    http://localhost:8000"
        echo "  - API Docs:      http://localhost:8000/docs"
        echo "  - Chroma DB:     http://localhost:8001"
        echo "  - PostgreSQL:    localhost:5432"
        echo "  - Redis:         localhost:6379"
        echo ""
        echo "Useful commands:"
        echo "  $COMPOSE_CMD logs -f          # View all logs"
        echo "  $COMPOSE_CMD logs -f app      # View app logs"
        echo "  $COMPOSE_CMD ps               # Show running services"
        echo "  $COMPOSE_CMD stop             # Stop all services"
        echo "  $COMPOSE_CMD down             # Stop and remove containers"
        echo "  $COMPOSE_CMD down -v          # Stop and remove volumes"
        echo ""

        # Wait for health check
        print_warning "Waiting for services to be ready..."
        sleep 5

        # Check health
        if curl -sf http://localhost:8000/health > /dev/null; then
            print_success "Application is healthy!"
        else
            print_warning "Application may still be starting up..."
            echo "Check logs with: $COMPOSE_CMD logs -f app"
        fi
        ;;

    4)
        print_success "Exiting"
        exit 0
        ;;

    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_header "Setup Complete!"
print_success "Your LlamaIndex application is running"
echo ""
echo "Next steps:"
echo "  1. Visit http://localhost:8000/docs for API documentation"
echo "  2. Add your documents to the ./data/ directory"
echo "  3. Trigger indexing: curl -X POST http://localhost:8000/ingest"
echo "  4. Query: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\":\"your question\"}'"
echo ""
print_success "Happy coding!"
