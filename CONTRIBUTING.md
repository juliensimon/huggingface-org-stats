# Contributing to Hugging Face Organization Statistics

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating a new issue, please:

1. Check if the issue has already been reported
2. Use the issue template if available
3. Provide detailed information including:
   - Python version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages (if any)

### Suggesting Enhancements

We welcome feature requests! When suggesting enhancements:

1. Describe the feature clearly
2. Explain the use case and benefits
3. Consider implementation complexity
4. Check if similar functionality already exists

### Code Contributions

#### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/huggingface-org-stats.git
   cd huggingface-org-stats
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

#### Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards below

3. Run tests and linting:
   ```bash
   # Format code
   black .

   # Lint code
   flake8 .

   # Type checking
   mypy .

   # Run tests
   pytest
   ```

4. Commit your changes with a descriptive message:
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. Push to your fork and create a pull request

#### Coding Standards

- **Code Style**: Follow [PEP 8](https://pep8.org/) with Black formatting
- **Type Hints**: Use type hints for all function parameters and return values
- **Documentation**: Add docstrings for all public functions and classes
- **Tests**: Write tests for new functionality
- **Commit Messages**: Use [Conventional Commits](https://conventionalcommits.org/) format

#### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(cli): add --verbose flag for detailed logging
fix(api): handle rate limiting errors gracefully
docs(readme): update installation instructions
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hf_org_stats

# Run with coverage and fail if below threshold
pytest --cov=hf_org_stats --cov-fail-under=80

# Run specific test file
pytest tests/test_hf_org_stats.py

# Run with verbose output
pytest -v

# Generate HTML coverage report
pytest --cov=hf_org_stats --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and error cases
- Mock external API calls
- Use fixtures for common setup

## 📝 Documentation

When adding new features or changing existing ones:

1. Update the README.md if needed
2. Add docstrings to new functions/classes
3. Update command-line help text
4. Consider adding examples

## 🔧 Development Tools

### Pre-commit Hooks

This project uses pre-commit hooks to automatically run code quality checks. Set them up with:

```bash
# Install pre-commit hooks
make setup-pre-commit

# Or manually:
pip install pre-commit
pre-commit install
```

The hooks will automatically run on every commit and include:
- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Bandit**: Security checks
- **detect-secrets**: Secret detection
- **Various file checks**: YAML, JSON, TOML validation

### IDE Configuration

Recommended VS Code settings (`.vscode/settings.json`):

```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## 🚀 Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release tag
4. Build and publish to PyPI (if applicable)

## 📞 Getting Help

If you need help with contributing:

1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Join our community discussions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Thank you for contributing! 🎉
