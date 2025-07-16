.PHONY: help install install-dev test lint format clean build publish

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=hf_org_stats --cov-report=html --cov-report=term --cov-report=xml

test-cov-fail: ## Run tests with coverage and fail if below threshold
	pytest tests/ --cov=hf_org_stats --cov-fail-under=80

coverage-report: ## Generate and open coverage report
	pytest tests/ --cov=hf_org_stats --cov-report=html
	open htmlcov/index.html

lint: ## Run linting checks
	flake8 .
	black --check .
	mypy .

format: ## Format code with black
	black .

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	python -m build

publish: ## Publish to PyPI (requires twine)
	twine upload dist/*

check-security: ## Run security checks
	bandit -r .
	safety check

run-example: ## Run example with arcee-ai organization
	python hf_org_stats.py --organization arcee-ai --verbose

setup-pre-commit: ## Set up pre-commit hooks
	pip install pre-commit
	pre-commit install

update-deps: ## Update dependencies
	pip install --upgrade pip
	pip install --upgrade -e ".[dev]"
