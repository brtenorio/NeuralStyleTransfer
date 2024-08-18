# Makefile for the Neural Style Transfer project

# Variables
PROJECT_NAME = Neural Style Transfer
PYTHON = python3
PIP = pip
POETRY = poetry

# Targets
.PHONY: all setup install run clean test

all: setup install test

setup:
	@echo "Setting up virtual environment..."
	$(POETRY) env use $(PYTHON)

install:
	@echo "Installing dependencies..."
	$(POETRY) install

run-app:
	@echo "Running the application..."
	$(POETRY) run streamlit run neural_style_transfer/app.py --server.port 8080

docker-build:
	@echo "Building Docker image..."
	docker-compose build

docker-run-app:
	@echo "Running application in Docker..."
	docker-compose up app

docker-down:
	@echo "Running application in Docker..."
	docker-compose down

clean:
	@echo "Cleaning up..."
	$(POETRY) env remove $(PYTHON)

clear-cache:
	@echo "Cleaning up..."
	$(POETRY) cache clear --all .

