# Hugging Face Organization Statistics

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PyPI](https://img.shields.io/badge/PyPI-huggingface--hub-blue.svg)](https://pypi.org/project/huggingface-hub/)
[![PyPI](https://img.shields.io/badge/PyPI-pandas-blue.svg)](https://pypi.org/project/pandas/)
[![PyPI](https://img.shields.io/badge/PyPI-requests-blue.svg)](https://pypi.org/project/requests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen.svg)](https://github.com/juliensimon/huggingface-org-stats)
[![Multi-Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12|%203.13-blue?logo=python&logoColor=white)](https://github.com/juliensimon/huggingface-org-stats/actions)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Security](https://img.shields.io/badge/security-bandit-yellow.svg)](https://bandit.readthedocs.io/)
[![Type Check](https://img.shields.io/badge/type%20check-mypy-blue.svg)](https://mypy.readthedocs.io/)
[![Linting](https://img.shields.io/badge/linting-flake8-blue.svg)](https://flake8.pycqa.org/)
[![Code Quality](https://img.shields.io/badge/code%20quality-A%2B-brightgreen.svg)](https://github.com/juliensimon/huggingface-org-stats)
[![Maintenance](https://img.shields.io/badge/maintained%3F-yes-green.svg)](https://github.com/juliensimon/huggingface-org-stats/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/juliensimon/huggingface-org-stats/pulls)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Organizations-orange.svg)](https://huggingface.co/)
[![CLI](https://img.shields.io/badge/CLI-Command%20Line-black.svg)](https://en.wikipedia.org/wiki/Command-line_interface)

A professional command-line tool to collect comprehensive statistics for Hugging Face organizations, including models, datasets, and spaces with download counts and engagement metrics.

## ✨ Features

- **📊 Models**: Collect download counts (all-time and 30-day) and likes
- **📈 Datasets**: Collect download counts (all-time and 30-day) and likes
- **🚀 Spaces**: Collect likes and engagement metrics
- **⚡ Parallel Processing**: Fast data collection with configurable workers
- **📁 Multiple Output Formats**: CSV, JSON, and Excel
- **🖥️ Command Line Interface**: Full-featured CLI with comprehensive options

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install -e .
```

2. **Run the script with an organization name:**
```bash
python hf_org_stats.py --organization arcee-ai
```

## 📦 Installation

1. **Clone this repository:**
```bash
git clone https://github.com/juliensimon/huggingface-org-stats
cd huggingface-org-stats
```

2. **Install dependencies:**
```bash
pip install -e .
```

3. **Set up your Hugging Face API token (optional, for higher rate limits):**
```bash
export HF_TOKEN="your_token_here"
```

## 🎯 Usage

### Command Line Interface

**Basic usage:**
```bash
python hf_org_stats.py --organization arcee-ai
```

**Advanced options:**
```bash
# Collect only models with custom settings
python hf_org_stats.py --organization microsoft --models-only --max-workers 10

# Use API token for higher rate limits
python hf_org_stats.py --organization openai --token YOUR_TOKEN

# Export to Excel format
python hf_org_stats.py --organization meta-llama --output excel

# Collect only datasets
python hf_org_stats.py --organization google --datasets-only

# Collect only spaces
python hf_org_stats.py --organization stability-ai --spaces-only

# Don't save results to file (just display)
python hf_org_stats.py --organization arcee-ai --no-save

# Enable verbose logging
python hf_org_stats.py --organization arcee-ai --verbose
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--organization` | `-o` | Organization name | `arcee-ai` |
| `--token` | `-t` | Hugging Face API token for higher rate limits | None |
| `--max-workers` | `-w` | Maximum parallel workers | `5` |
| `--output` | `-f` | Output format: csv, json, excel | `csv` |
| `--no-save` | | Don't save results to file | False |
| `--models-only` | | Only collect model statistics | False |
| `--datasets-only` | | Only collect dataset statistics | False |
| `--spaces-only` | | Only collect space statistics | False |
| `--verbose` | `-v` | Enable verbose logging | False |

## 📊 Output

The tool generates:

1. **📋 Summary Report**: Comprehensive statistics including totals, averages, and top performers
2. **📄 Detailed Data**: Complete dataset with all collected metrics
3. **💾 Download Files**: CSV/JSON/Excel files with timestamped names

### Sample Output Files

- `arcee-ai_models_stats_20241201_143022.csv`
- `arcee-ai_datasets_stats_20241201_143022.csv`
- `arcee-ai_spaces_stats_20241201_143022.csv`

## 🔑 API Token

For better performance and higher rate limits, consider using a Hugging Face API token:

1. Go to [HF Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Use it with the `--token` option or set as environment variable:
   ```bash
   export HF_TOKEN="hf_..."
   ```

## 🏢 Example Organizations

Try these popular organizations:
- `arcee-ai`
- `microsoft`
- `openai`
- `meta-llama`
- `google`
- `stability-ai`

## 📋 Requirements

- Python 3.7+
- See `pyproject.toml` for full dependencies

## 🔧 Development Setup

For contributors, set up the development environment:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
make setup-pre-commit

# Run tests
make test

# Run tests with coverage
make test-cov

# Generate coverage report
make coverage-report

# Format code
make format

# Run type checks
mypy hf_org_stats.py tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)
- Data processing with [Pandas](https://pandas.pydata.org/)
- Progress tracking with [tqdm](https://tqdm.github.io/)
# CI Trigger
# Test CI
