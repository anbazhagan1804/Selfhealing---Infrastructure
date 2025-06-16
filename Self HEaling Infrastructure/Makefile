# SelfHealing Sentinel - Makefile

# Variables
PROJECT_NAME := selfhealing-sentinel
DOCKER_REGISTRY := localhost:5000
VERSION := 0.1.0

# Docker image names
API_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)/sentinel-api:$(VERSION)
ANOMALY_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)/anomaly-detector:$(VERSION)
REMEDIATION_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)/remediation-orchestrator:$(VERSION)
DEMO_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)/demo-app:$(VERSION)
MAIN_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)/main:$(VERSION)

# Kubernetes namespace
NAMESPACE := sentinel

# Python settings
PYTHON := python
PIP := pip

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "SelfHealing Sentinel - Self-Healing Infrastructure Project"
	@echo ""
	@echo "Usage:"
	@echo "  make setup         - Create project directory structure"
	@echo "  make install       - Install Python dependencies"
	@echo "  make lint          - Run linting checks"
	@echo "  make test          - Run tests"
	@echo "  make docker-build  - Build all Docker images"
	@echo "  make docker-push   - Push all Docker images to registry"
	@echo "  make run-local     - Run the system locally using Docker Compose"
	@echo "  make stop-local    - Stop the local Docker Compose deployment"
	@echo "  make deploy-k8s    - Deploy to Kubernetes"
	@echo "  make clean         - Clean up generated files"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Environment Variables:"
	@echo "  DOCKER_REGISTRY    - Docker registry URL (default: localhost:5000)"
	@echo "  VERSION            - Version tag for images (default: 0.1.0)"
	@echo "  NAMESPACE          - Kubernetes namespace (default: sentinel)"

# Setup target
.PHONY: setup
setup:
	@echo "Creating project directory structure..."
	powershell -ExecutionPolicy Bypass -File setup.ps1

# Install dependencies
.PHONY: install
install:
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt

# Linting
.PHONY: lint
lint:
	@echo "Running linting checks..."
	flake8 .
	isort --check-only --profile black .
	black --check .
	mypy .

# Testing
.PHONY: test
test:
	@echo "Running tests..."
	pytest --cov=. --cov-report=term-missing

# Docker build
.PHONY: docker-build
docker-build:
	@echo "Building Docker images..."
	docker build --target sentinel-api -t $(API_IMAGE) .
	docker build --target anomaly-detector -t $(ANOMALY_IMAGE) .
	docker build --target remediation-orchestrator -t $(REMEDIATION_IMAGE) .
	docker build --target demo-app -t $(DEMO_IMAGE) .
	docker build --target main -t $(MAIN_IMAGE) .

# Docker push
.PHONY: docker-push
docker-push:
	@echo "Pushing Docker images to registry..."
	docker push $(API_IMAGE)
	docker push $(ANOMALY_IMAGE)
	docker push $(REMEDIATION_IMAGE)
	docker push $(DEMO_IMAGE)
	docker push $(MAIN_IMAGE)

# Run locally with Docker Compose
.PHONY: run-local
run-local:
	@echo "Starting local deployment with Docker Compose..."
	docker-compose up -d

# Stop local deployment
.PHONY: stop-local
stop-local:
	@echo "Stopping local deployment..."
	docker-compose down

# Deploy to Kubernetes
.PHONY: deploy-k8s
deploy-k8s:
	@echo "Deploying to Kubernetes in namespace $(NAMESPACE)..."
	kubectl apply -f kubernetes-monitoring-deployment.yaml
	kubectl apply -f kubernetes-sentinel-deployment.yaml
	kubectl apply -f keda-scaledobject.yaml

# Create Kubernetes ConfigMaps from config files
.PHONY: create-configmaps
create-configmaps:
	@echo "Creating Kubernetes ConfigMaps from config files..."
	kubectl create configmap sentinel-settings --from-file=config/settings.yaml -n $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap sentinel-thresholds --from-file=config/thresholds.yaml -n $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap sentinel-remediation --from-file=config/remediation.yaml -n $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap prometheus-config --from-file=prometheus.yml -n $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap alertmanager-config --from-file=alertmanager.yml -n $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -

# Clean up
.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/